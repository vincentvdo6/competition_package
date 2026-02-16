#!/usr/bin/env python
"""Greedy ensemble selection for vanilla GRU models.

Operates on raw .pt checkpoints — no MODEL_REGISTRY, no zips, no normalizers.
Two phases: (1) cache batch-inference val predictions, (2) greedy forward selection.

Usage:
  # Cache predictions for all checkpoints
  python scripts/greedy_vanilla_ensemble.py cache \
      --checkpoints logs/vanilla_all/*.pt \
      --data datasets/valid.parquet \
      --cache-dir cache/vanilla_preds

  # Greedy forward selection
  python scripts/greedy_vanilla_ensemble.py greedy \
      --cache-dir cache/vanilla_preds --max-models 15

  # Show prediction correlation matrix
  python scripts/greedy_vanilla_ensemble.py diversity \
      --cache-dir cache/vanilla_preds

  # Score a specific set of models
  python scripts/greedy_vanilla_ensemble.py score \
      --cache-dir cache/vanilla_preds \
      --models gru_parity_v1_seed43 vanilla_varA_seed42
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import weighted_pearson_correlation


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_vanilla_model(ckpt_path):
    """Load vanilla RNN (GRU or LSTM) from checkpoint, return (rnn, fc, best_score, best_epoch)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    model_cfg = config.get("model", {})

    h = model_cfg.get("hidden_size", 64)
    nl = model_cfg.get("num_layers", 3)
    inp = model_cfg.get("input_size", 32)
    rnn_type = model_cfg.get("rnn_type", "gru")

    state_dict = ckpt["model_state_dict"]

    if rnn_type == "lstm":
        rnn = nn.LSTM(input_size=inp, hidden_size=h, num_layers=nl,
                      batch_first=True, dropout=0.0, bidirectional=False)
        rnn_sd = {k.replace("lstm.", ""): v for k, v in state_dict.items()
                  if k.startswith("lstm.")}
        rnn.load_state_dict(rnn_sd)
    else:
        rnn = nn.GRU(input_size=inp, hidden_size=h, num_layers=nl,
                     batch_first=True, dropout=0.0, bidirectional=False)
        rnn_sd = {k.replace("gru.", ""): v for k, v in state_dict.items()
                  if k.startswith("gru.")}
        rnn.load_state_dict(rnn_sd)

    fc = nn.Linear(h, 2)
    fc_sd = {k.replace("output_proj.", ""): v for k, v in state_dict.items()
             if k.startswith("output_proj.")}
    fc.load_state_dict(fc_sd)

    return rnn, fc, ckpt.get("best_score", 0), ckpt.get("best_epoch", 0)


# ---------------------------------------------------------------------------
# Batch inference (per-sequence, matching online inference behavior)
# ---------------------------------------------------------------------------
def batch_inference(rnn, fc, df):
    """Run per-sequence forward pass on validation data.

    Works with both GRU and LSTM (passing None auto-initializes hidden state).
    Returns (preds, targets) arrays filtered to need_prediction=True rows.
    Each sequence gets fresh hidden state, matching online inference.
    """
    rnn.eval()
    fc.eval()

    values = df.values
    seq_col = values[:, 0].astype(int)
    need_pred_col = values[:, 2].astype(bool)
    features_all = values[:, 3:35].astype(np.float32)
    targets_all = values[:, 35:].astype(np.float64)

    all_preds = []
    all_targets = []

    unique_seqs = np.unique(seq_col)
    for seq_ix in unique_seqs:
        mask = seq_col == seq_ix
        seq_features = features_all[mask]
        seq_need_pred = need_pred_col[mask]
        seq_targets = targets_all[mask]

        x = torch.from_numpy(seq_features).unsqueeze(0)  # (1, seq_len, 32)
        with torch.no_grad():
            out, _ = rnn(x, None)
            preds = fc(out).squeeze(0).numpy()  # (seq_len, 2)

        preds = np.clip(preds, -6.0, 6.0)
        all_preds.append(preds[seq_need_pred])
        all_targets.append(seq_targets[seq_need_pred])

    return np.concatenate(all_preds), np.concatenate(all_targets)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_predictions(preds, targets):
    """Score predictions using per-target weighted Pearson, averaged."""
    t0_score = weighted_pearson_correlation(targets[:, 0], preds[:, 0])
    t1_score = weighted_pearson_correlation(targets[:, 1], preds[:, 1])
    return (t0_score + t1_score) / 2, t0_score, t1_score


def score_ensemble(pred_list, targets):
    """Score uniform-weight ensemble from list of prediction arrays."""
    ensemble_preds = np.mean(pred_list, axis=0)
    return score_predictions(ensemble_preds, targets)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_cache(args):
    """Cache per-model val predictions as .npz files."""
    os.makedirs(args.cache_dir, exist_ok=True)
    df = pd.read_parquet(args.data)
    print(f"Loaded {len(df)} rows from {args.data}")

    for ckpt_path in args.checkpoints:
        basename = Path(ckpt_path).stem
        # Skip epoch checkpoints
        if "_epoch" in basename:
            print(f"  Skipping epoch checkpoint: {basename}")
            continue

        cache_path = os.path.join(args.cache_dir, f"{basename}.npz")
        if os.path.exists(cache_path) and not args.force:
            print(f"  {basename}: cached (skip)")
            continue

        # Skip models with incompatible input_size (e.g. parity_v2 has 42 features)
        ckpt_tmp = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        inp_size = ckpt_tmp.get("config", {}).get("model", {}).get("input_size", 32)
        if inp_size != 32:
            print(f"  {basename}: SKIP (input_size={inp_size}, need 32)")
            del ckpt_tmp
            continue
        del ckpt_tmp

        print(f"  {basename}: inferring...", end=" ", flush=True)
        t0 = time.time()
        rnn, fc, best_score, best_epoch = load_vanilla_model(ckpt_path)
        preds, targets = batch_inference(rnn, fc, df)
        avg, s_t0, s_t1 = score_predictions(preds, targets)
        elapsed = time.time() - t0

        np.savez_compressed(cache_path, preds=preds, targets=targets)
        print(f"val={avg:.4f} (t0={s_t0:.4f}, t1={s_t1:.4f}) "
              f"ckpt_score={best_score:.4f} [{elapsed:.1f}s]")

    # Save targets once for reuse
    targets_path = os.path.join(args.cache_dir, "_targets.npz")
    if not os.path.exists(targets_path):
        # Load from first cached file
        for f in sorted(os.listdir(args.cache_dir)):
            if f.startswith("_") or not f.endswith(".npz"):
                continue
            data = np.load(os.path.join(args.cache_dir, f))
            np.savez_compressed(targets_path, targets=data["targets"])
            print(f"\nTargets saved ({data['targets'].shape[0]} samples)")
            break

    print("\nCache complete!")


def _load_cache(cache_dir):
    """Load all cached predictions and targets."""
    # Load targets
    targets_path = os.path.join(cache_dir, "_targets.npz")
    if os.path.exists(targets_path):
        targets = np.load(targets_path)["targets"]
    else:
        # Fallback: load from first model
        for f in sorted(os.listdir(cache_dir)):
            if f.startswith("_") or not f.endswith(".npz"):
                continue
            data = np.load(os.path.join(cache_dir, f))
            targets = data["targets"]
            break
        else:
            raise FileNotFoundError("No cached predictions found")

    # Load all model predictions
    models = {}
    for f in sorted(os.listdir(cache_dir)):
        if f.startswith("_") or not f.endswith(".npz"):
            continue
        name = f.replace(".npz", "")
        data = np.load(os.path.join(cache_dir, f))
        models[name] = data["preds"]

    return models, targets


def cmd_greedy(args):
    """Greedy forward selection: build ensemble one model at a time."""
    models, targets = _load_cache(args.cache_dir)
    print(f"Loaded {len(models)} models, {targets.shape[0]} samples\n")

    # Score all singles
    singles = []
    for name, preds in models.items():
        avg, t0, t1 = score_predictions(preds, targets)
        singles.append((name, avg, t0, t1))
    singles.sort(key=lambda x: -x[1])

    print("=== Single Model Ranking ===")
    print(f"{'Rank':<5} {'Model':<40} {'Avg':>8} {'t0':>8} {'t1':>8}")
    print("-" * 72)
    for i, (name, avg, t0, t1) in enumerate(singles[:20], 1):
        print(f"{i:<5} {name:<40} {avg:>8.4f} {t0:>8.4f} {t1:>8.4f}")

    # Greedy forward selection
    print(f"\n=== Greedy Forward Selection (max {args.max_models} models) ===")
    selected = []
    remaining = set(models.keys())
    best_score = -1

    for step in range(args.max_models):
        best_name = None
        best_step_score = -1

        for name in remaining:
            candidate_preds = [models[n] for n in selected] + [models[name]]
            avg, _, _ = score_ensemble(candidate_preds, targets)
            if avg > best_step_score:
                best_step_score = avg
                best_name = name

        if best_name is None:
            break

        selected.append(best_name)
        remaining.discard(best_name)

        # Score current ensemble
        ens_preds = [models[n] for n in selected]
        avg, t0, t1 = score_ensemble(ens_preds, targets)
        delta = avg - best_score if best_score > 0 else 0
        best_score = avg

        print(f"  Step {step+1:>2}: +{best_name:<38} "
              f"ens={avg:.4f} (t0={t0:.4f} t1={t1:.4f}) "
              f"delta={delta:+.5f}")

    print(f"\n=== Final Ensemble ({len(selected)} models) ===")
    print(f"Val score: {best_score:.4f}")
    print(f"\nSelected models (in order):")
    for i, name in enumerate(selected, 1):
        print(f"  {i}. {name}")

    # Count recipe diversity
    recipes = {}
    for name in selected:
        if "parity_v1" in name:
            recipe = "base"
        elif "varA" in name:
            recipe = "varA"
        elif "varB" in name:
            recipe = "varB"
        elif "varC" in name:
            recipe = "varC"
        else:
            recipe = "other"
        recipes[recipe] = recipes.get(recipe, 0) + 1

    print(f"\nRecipe breakdown:")
    for recipe, count in sorted(recipes.items()):
        pct = count / len(selected) * 100
        print(f"  {recipe}: {count} ({pct:.0f}%)")


def cmd_diversity(args):
    """Show prediction correlation matrix."""
    models, targets = _load_cache(args.cache_dir)
    names = sorted(models.keys())
    n = len(names)
    print(f"Loaded {n} models\n")

    # Flatten predictions for correlation
    flat_preds = {name: models[name].flatten() for name in names}

    # Compute correlation matrix
    corr_matrix = np.zeros((n, n))
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            corr_matrix[i, j] = np.corrcoef(flat_preds[name_i], flat_preds[name_j])[0, 1]

    # Print summary stats
    upper_tri = corr_matrix[np.triu_indices(n, k=1)]
    print(f"Pairwise prediction correlations:")
    print(f"  Mean: {upper_tri.mean():.4f}")
    print(f"  Min:  {upper_tri.min():.4f}")
    print(f"  Max:  {upper_tri.max():.4f}")
    print(f"  Std:  {upper_tri.std():.4f}")

    # Find most diverse pairs
    print(f"\nMost diverse pairs (lowest correlation):")
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((names[i], names[j], corr_matrix[i, j]))
    pairs.sort(key=lambda x: x[2])
    for name_i, name_j, corr in pairs[:10]:
        print(f"  {corr:.4f}: {name_i} vs {name_j}")

    # Group by recipe
    recipe_corrs = {}
    for i in range(n):
        for j in range(i+1, n):
            ri = "base" if "parity_v1" in names[i] else names[i].split("_seed")[0].split("_")[-1]
            rj = "base" if "parity_v1" in names[j] else names[j].split("_seed")[0].split("_")[-1]
            key = f"{min(ri,rj)}-{max(ri,rj)}"
            recipe_corrs.setdefault(key, []).append(corr_matrix[i, j])

    print(f"\nMean correlation by recipe pair:")
    for key in sorted(recipe_corrs.keys()):
        vals = recipe_corrs[key]
        print(f"  {key}: {np.mean(vals):.4f} (n={len(vals)})")


def cmd_score(args):
    """Score a specific set of models."""
    models, targets = _load_cache(args.cache_dir)

    # Validate model names
    for name in args.models:
        if name not in models:
            print(f"ERROR: Model '{name}' not found in cache")
            print(f"Available: {', '.join(sorted(models.keys())[:10])}...")
            sys.exit(1)

    pred_list = [models[name] for name in args.models]
    avg, t0, t1 = score_ensemble(pred_list, targets)
    print(f"Ensemble of {len(args.models)} models:")
    print(f"  Val score: {avg:.4f} (t0={t0:.4f}, t1={t1:.4f})")
    print(f"  Models: {', '.join(args.models)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Greedy ensemble selection for vanilla GRU models")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # cache
    p_cache = subparsers.add_parser("cache", help="Cache val predictions")
    p_cache.add_argument("--checkpoints", nargs="+", required=True,
                         help="Paths to .pt checkpoint files")
    p_cache.add_argument("--data", default="datasets/valid.parquet",
                         help="Path to validation parquet")
    p_cache.add_argument("--cache-dir", default="cache/vanilla_preds",
                         help="Directory for cached .npz files")
    p_cache.add_argument("--force", action="store_true",
                         help="Overwrite existing cache files")

    # greedy
    p_greedy = subparsers.add_parser("greedy", help="Greedy forward selection")
    p_greedy.add_argument("--cache-dir", default="cache/vanilla_preds")
    p_greedy.add_argument("--max-models", type=int, default=15)

    # diversity
    p_div = subparsers.add_parser("diversity", help="Show correlation matrix")
    p_div.add_argument("--cache-dir", default="cache/vanilla_preds")

    # score
    p_score = subparsers.add_parser("score", help="Score specific models")
    p_score.add_argument("--cache-dir", default="cache/vanilla_preds")
    p_score.add_argument("--models", nargs="+", required=True,
                         help="Model names (without .npz extension)")

    args = parser.parse_args()
    if args.command == "cache":
        cmd_cache(args)
    elif args.command == "greedy":
        cmd_greedy(args)
    elif args.command == "diversity":
        cmd_diversity(args)
    elif args.command == "score":
        cmd_score(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
