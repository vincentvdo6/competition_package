#!/usr/bin/env python
"""Local ensemble validation with prediction caching and greedy search.

Two-phase workflow:
  1. `infer`  — extract checkpoints from zips, run online inference on valid.parquet,
                cache per-model predictions as .npz files (slow, ~3-7 min/model)
  2. `score`  — load cached predictions, score any ensemble combo instantly,
                run greedy search, bootstrap confidence intervals

Usage examples:
  # Run inference for all GRU models and cache predictions
  python scripts/validate_ensemble_local.py infer --models gru_tw2_s42 gru_tw2_s43

  # Run inference for ALL known models (takes ~2 hours on CPU)
  python scripts/validate_ensemble_local.py infer --all

  # Score a specific ensemble
  python scripts/validate_ensemble_local.py score --models gru_tw2_s42 gru_tw2_s43 gru_p1_s47

  # Score with custom weights (70/30 split)
  python scripts/validate_ensemble_local.py score --models m1 m2 m3 m4 m5 a1 a2 --weights 0.14 0.14 0.14 0.14 0.14 0.15 0.15

  # Greedy search: find best N-model ensemble from cached models
  python scripts/validate_ensemble_local.py greedy --pool all --max-models 8

  # Score a named preset
  python scripts/validate_ensemble_local.py preset --name champion_7030

  # List all models and their cache status
  python scripts/validate_ensemble_local.py list
"""

from __future__ import annotations

import argparse
import itertools
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import weighted_pearson_correlation

CACHE_DIR = ROOT / "cache" / "predictions"
DOWNLOADS = Path(r"C:\Users\Vincent\Downloads")

# ---------------------------------------------------------------------------
# Zip file locations
# ---------------------------------------------------------------------------
ZIPS = {
    "gru5_attn3": DOWNLOADS / "gru5_attn3_uniform8.zip",
    "seed_expansion": DOWNLOADS / "gru_seed_expansion.zip",
    "slim_pearson": DOWNLOADS / "slim_checkpoints_pearson.zip",
    "attn_nb07": DOWNLOADS / "attn_seeds_45_46_47.zip",
    "attn_nb07_s2": DOWNLOADS / "attn_seeds_48_49_50.zip",
    "attn_nb07_s3": DOWNLOADS / "attn_seeds_51_52.zip",
}

# ---------------------------------------------------------------------------
# Model registry
# Each entry: {zip, ckpt_file, norm_file, config, type, val_score}
# type is "gru" or "gru_attention"
# val_score is from notebook 06 or training logs (None if unknown)
# ---------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {}

def _reg(name, zip_key, ckpt, norm, config, mtype="gru", val=None):
    MODEL_REGISTRY[name] = {
        "zip": zip_key, "ckpt": ckpt, "norm": norm,
        "config": config, "type": mtype, "val": val,
    }

# --- From gru5_attn3_uniform8.zip (old combined-loss, seeds 42-46 GRU, 42-44 attn) ---
_reg("gru_tw2_s42", "gru5_attn3", "model_0.pt", "normalizer.npz",   "gru_derived_tightwd_v2")
_reg("gru_tw2_s43", "gru5_attn3", "model_1.pt", "normalizer_1.npz", "gru_derived_tightwd_v2")
_reg("gru_tw2_s44", "gru5_attn3", "model_2.pt", "normalizer_2.npz", "gru_derived_tightwd_v2")
_reg("gru_tw2_s45", "gru5_attn3", "model_3.pt", "normalizer_3.npz", "gru_derived_tightwd_v2")
_reg("gru_tw2_s46", "gru5_attn3", "model_4.pt", "normalizer_4.npz", "gru_derived_tightwd_v2")
_reg("attn_comb_s42", "gru5_attn3", "model_5.pt", "normalizer_5.npz", "gru_attention_clean_v1", "gru_attention")
_reg("attn_comb_s43", "gru5_attn3", "model_6.pt", "normalizer_6.npz", "gru_attention_clean_v1", "gru_attention")
_reg("attn_comb_s44", "gru5_attn3", "model_7.pt", "normalizer_7.npz", "gru_attention_clean_v1", "gru_attention")

# --- From gru_seed_expansion.zip (notebook 06, seeds 47-53 tightwd, 45-50 pearson) ---
for seed in range(47, 54):
    val_scores = {47: 0.2620, 48: 0.2649, 49: 0.2577, 50: 0.2654, 51: 0.2637, 52: 0.2537, 53: 0.2636}
    _reg(f"gru_tw2_s{seed}", "seed_expansion",
         f"gru_derived_tightwd_v2_seed{seed}.pt",
         f"normalizer_gru_derived_tightwd_v2_seed{seed}.npz",
         "gru_derived_tightwd_v2", val=val_scores.get(seed))

for seed in range(45, 51):
    val_scores = {45: 0.2648, 46: 0.2634, 47: 0.2668, 48: 0.2589, 49: 0.2603, 50: 0.2640}
    _reg(f"gru_p1_s{seed}", "seed_expansion",
         f"gru_pearson_v1_seed{seed}.pt",
         f"normalizer_gru_pearson_v1_seed{seed}.npz",
         "gru_pearson_v1", val=val_scores.get(seed))

# --- From slim_checkpoints_pearson.zip (pearson training, seeds 42-44 GRU, 45-46 attn clean, 42-43 attn pearson) ---
for seed in range(42, 45):
    _reg(f"gru_p1_s{seed}", "slim_pearson",
         f"gru_pearson_v1_seed{seed}.pt",
         f"normalizer_gru_pearson_v1_seed{seed}.npz",
         "gru_pearson_v1")

_reg("attn_comb_s45", "slim_pearson", "gru_attention_clean_v1_seed45.pt",
     "normalizer_gru_attention_clean_v1_seed45.npz", "gru_attention_clean_v1", "gru_attention")
_reg("attn_comb_s46", "slim_pearson", "gru_attention_clean_v1_seed46.pt",
     "normalizer_gru_attention_clean_v1_seed46.npz", "gru_attention_clean_v1", "gru_attention")

_reg("attn_pear_s42", "slim_pearson", "gru_attention_pearson_v1_seed42.pt",
     "normalizer_gru_attention_pearson_v1_seed42.npz", "gru_attention_pearson_v1", "gru_attention")
_reg("attn_pear_s43", "slim_pearson", "gru_attention_pearson_v1_seed43.pt",
     "normalizer_gru_attention_pearson_v1_seed43.npz", "gru_attention_pearson_v1", "gru_attention")

# --- From attn_seeds_45_46_47.zip (notebook 07 session 1, combined loss) ---
_reg("attn_nb07_s45", "attn_nb07", "attn_clean_seed45.pt",
     "normalizer_attn_clean_seed45.npz", "gru_attention_clean_v1", "gru_attention", val=0.2599)
_reg("attn_nb07_s46", "attn_nb07", "attn_clean_seed46.pt",
     "normalizer_attn_clean_seed46.npz", "gru_attention_clean_v1", "gru_attention", val=0.2659)
_reg("attn_nb07_s47", "attn_nb07", "attn_clean_seed47.pt",
     "normalizer_attn_clean_seed47.npz", "gru_attention_clean_v1", "gru_attention", val=0.2598)

# --- From attn_seeds_48_49_50.zip (notebook 07 session 2, combined loss) ---
_reg("attn_nb07_s48", "attn_nb07_s2", "attn_clean_seed48.pt",
     "normalizer_attn_clean_seed48.npz", "gru_attention_clean_v1", "gru_attention", val=0.2706)
_reg("attn_nb07_s49", "attn_nb07_s2", "attn_clean_seed49.pt",
     "normalizer_attn_clean_seed49.npz", "gru_attention_clean_v1", "gru_attention", val=0.2560)
_reg("attn_nb07_s50", "attn_nb07_s2", "attn_clean_seed50.pt",
     "normalizer_attn_clean_seed50.npz", "gru_attention_clean_v1", "gru_attention", val=0.2752)

# --- From attn_seeds_51_52.zip (notebook 07 session 3, combined loss) ---
_reg("attn_nb07_s51", "attn_nb07_s3", "attn_clean_seed51.pt",
     "normalizer_attn_clean_seed51.npz", "gru_attention_clean_v1", "gru_attention", val=0.2600)
_reg("attn_nb07_s52", "attn_nb07_s3", "attn_clean_seed52.pt",
     "normalizer_attn_clean_seed52.npz", "gru_attention_clean_v1", "gru_attention", val=0.2641)

# ---------------------------------------------------------------------------
# Named ensembles (presets)
# ---------------------------------------------------------------------------
PRESETS = {
    "champion_7030": {
        "desc": "Current LB champion: 5 GRU tightwd (42-46) + 2 attn combined (42-43), 70/30",
        "models": ["gru_tw2_s42", "gru_tw2_s43", "gru_tw2_s44", "gru_tw2_s45", "gru_tw2_s46",
                    "attn_comb_s42", "attn_comb_s43"],
        "weights": [0.14, 0.14, 0.14, 0.14, 0.14, 0.15, 0.15],
    },
    "champion_clone_v2": {
        "desc": "Best 5 GRU by val + 2 old combined attn, 70/30 (submitted, pending)",
        "models": ["gru_p1_s47", "gru_tw2_s50", "gru_tw2_s48", "gru_p1_s45", "gru_p1_s50",
                    "attn_comb_s42", "attn_comb_s43"],
        "weights": [0.14, 0.14, 0.14, 0.14, 0.14, 0.15, 0.15],
    },
    "5gru_uniform": {
        "desc": "5 GRU tightwd (42-46) uniform — 0.2614 LB baseline",
        "models": ["gru_tw2_s42", "gru_tw2_s43", "gru_tw2_s44", "gru_tw2_s45", "gru_tw2_s46"],
    },
    "fast8_gru": {
        "desc": "8 GRU-only (5 best tightwd + 3 best pearson), no attention overhead",
        "models": ["gru_p1_s47", "gru_tw2_s50", "gru_tw2_s48", "gru_p1_s45", "gru_p1_s50",
                    "gru_tw2_s51", "gru_tw2_s53", "gru_p1_s46"],
    },
    "fast8_gru_7030": {
        "desc": "8 GRU-only top by val, 70/30 (tightwd/pearson mix)",
        "models": ["gru_p1_s47", "gru_tw2_s50", "gru_tw2_s48", "gru_p1_s45", "gru_p1_s50",
                    "gru_tw2_s51", "gru_tw2_s53", "gru_p1_s46"],
    },
    "top5_new_gru_2attn": {
        "desc": "Top 5 new GRU (by val) + 2 old combined attn, 70/30",
        "models": ["gru_p1_s47", "gru_tw2_s50", "gru_tw2_s48", "gru_p1_s45", "gru_p1_s50",
                    "attn_comb_s42", "attn_comb_s43"],
        "weights": [0.14, 0.14, 0.14, 0.14, 0.14, 0.15, 0.15],
    },
    "balanced7_new": {
        "desc": "5 new GRU + 2 combined attn (45-46), 70/30",
        "models": ["gru_p1_s47", "gru_tw2_s50", "gru_tw2_s48", "gru_p1_s45", "gru_p1_s50",
                    "attn_comb_s45", "attn_comb_s46"],
        "weights": [0.14, 0.14, 0.14, 0.14, 0.14, 0.15, 0.15],
    },
    "champion_v3_46swap": {
        "desc": "Champion GRUs + attn_nb07_s46 replacing attn_comb_s43, 70/30",
        "models": ["gru_p1_s47", "gru_tw2_s50", "gru_tw2_s48", "gru_p1_s45", "gru_p1_s50",
                    "attn_comb_s42", "attn_nb07_s46"],
        "weights": [0.14, 0.14, 0.14, 0.14, 0.14, 0.15, 0.15],
    },
    "champion_v3_both_new": {
        "desc": "Champion GRUs + 2 best nb07 attn (46+45), 70/30",
        "models": ["gru_p1_s47", "gru_tw2_s50", "gru_tw2_s48", "gru_p1_s45", "gru_p1_s50",
                    "attn_nb07_s46", "attn_nb07_s45"],
        "weights": [0.14, 0.14, 0.14, 0.14, 0.14, 0.15, 0.15],
    },
    "champion_v4_top2attn": {
        "desc": "Champion GRUs + 2 best nb07 attn (50+48), 70/30",
        "models": ["gru_p1_s47", "gru_tw2_s50", "gru_tw2_s48", "gru_p1_s45", "gru_p1_s50",
                    "attn_nb07_s50", "attn_nb07_s48"],
        "weights": [0.14, 0.14, 0.14, 0.14, 0.14, 0.15, 0.15],
    },
    "champion_v4_s50swap": {
        "desc": "Champion GRUs + attn_nb07_s50 (val 0.2752!) + old s42, 70/30",
        "models": ["gru_p1_s47", "gru_tw2_s50", "gru_tw2_s48", "gru_p1_s45", "gru_p1_s50",
                    "attn_comb_s42", "attn_nb07_s50"],
        "weights": [0.14, 0.14, 0.14, 0.14, 0.14, 0.15, 0.15],
    },
}

# Model group shortcuts for greedy search
MODEL_GROUPS = {
    "all": list(MODEL_REGISTRY.keys()),
    "gru_all": [k for k in MODEL_REGISTRY if k.startswith("gru_")],
    "gru_tw2": [k for k in MODEL_REGISTRY if k.startswith("gru_tw2_")],
    "gru_p1": [k for k in MODEL_REGISTRY if k.startswith("gru_p1_")],
    "attn_all": [k for k in MODEL_REGISTRY if k.startswith("attn_")],
    "attn_comb": [k for k in MODEL_REGISTRY if k.startswith("attn_comb_")],
    "attn_pear": [k for k in MODEL_REGISTRY if k.startswith("attn_pear_")],
}


# ---------------------------------------------------------------------------
# Online inference runner (adapted from score_ensemble_candidates.py)
# ---------------------------------------------------------------------------

from src.models.gru_attention import GRUAttentionModel
from src.models.gru_baseline import GRUBaseline
from src.data.preprocessing import DerivedFeatureBuilder, Normalizer


class OnlineModelRunner:
    """Run a single model through validation data with online inference."""

    def __init__(self, config_path: str, checkpoint_path: str, normalizer_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.normalizer = Normalizer.load(normalizer_path)
        self.device = torch.device("cpu")  # Always CPU for local validation

        data_cfg = self.config.get("data", {})
        self.derived_features = bool(data_cfg.get("derived_features", False))

        model_type = self.config.get("model", {}).get("type", "gru")
        if model_type == "gru_attention":
            self.model = GRUAttentionModel(self.config)
        else:
            self.model = GRUBaseline(self.config)

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def run(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run online inference, return (predictions, targets, seq_indices).

        All arrays are aligned and filtered to need_prediction=True rows only.
        """
        predictions = []
        targets = []
        seq_indices = []

        current_seq_ix = None
        hidden = None
        n_rows = len(df)
        t_start = time.time()

        for row_idx, row in enumerate(df.values):
            seq_ix = int(row[0])
            need_prediction = bool(row[2])
            lob_data = row[3:35]
            labels = row[35:]

            if current_seq_ix != seq_ix:
                current_seq_ix = seq_ix
                hidden = None

            raw = lob_data.reshape(1, -1).astype(np.float32)
            if self.derived_features:
                derived = DerivedFeatureBuilder.compute(raw)
                raw = np.concatenate([raw, derived], axis=-1)

            features = self.normalizer.transform(raw)
            x = torch.from_numpy(features).to(self.device)

            with torch.no_grad():
                pred, hidden = self.model.forward_step(x, hidden)
                pred = pred.cpu().numpy().squeeze()

            pred = np.clip(pred, -6, 6)

            if need_prediction:
                predictions.append(pred)
                targets.append(labels)
                seq_indices.append(seq_ix)

            if (row_idx + 1) % 200000 == 0:
                elapsed = time.time() - t_start
                pct = 100.0 * (row_idx + 1) / n_rows
                print(f"    {pct:.0f}% ({row_idx+1}/{n_rows}) in {elapsed:.0f}s")

        return (
            np.array(predictions),
            np.array(targets, dtype=np.float64),
            np.array(seq_indices),
        )


# ---------------------------------------------------------------------------
# Scoring utilities
# ---------------------------------------------------------------------------

def score_ensemble(
    model_preds: List[np.ndarray],
    targets: np.ndarray,
    weights: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Score an ensemble of model predictions with uniform or custom weights."""
    n = len(model_preds)
    if weights is None:
        weights = [1.0 / n] * n
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    ensemble_pred = sum(w * p for w, p in zip(weights, model_preds))
    ensemble_pred = np.clip(ensemble_pred, -6, 6)

    t0 = weighted_pearson_correlation(targets[:, 0], ensemble_pred[:, 0])
    t1 = weighted_pearson_correlation(targets[:, 1], ensemble_pred[:, 1])
    avg = (t0 + t1) / 2.0
    return {"t0": t0, "t1": t1, "avg": avg}


def bootstrap_score(
    model_preds: List[np.ndarray],
    targets: np.ndarray,
    seq_indices: np.ndarray,
    n_bootstrap: int = 200,
    seed: int = 42,
    weights: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Bootstrap ensemble score at sequence level."""
    rng = np.random.RandomState(seed)
    unique_seqs = np.unique(seq_indices)
    n_seqs = len(unique_seqs)

    seq_to_rows = {}
    for idx, s in enumerate(seq_indices):
        if s not in seq_to_rows:
            seq_to_rows[s] = []
        seq_to_rows[s].append(idx)

    boot_avgs = []
    for _ in range(n_bootstrap):
        sampled_seqs = rng.choice(unique_seqs, size=n_seqs, replace=True)
        sampled_rows = []
        for s in sampled_seqs:
            sampled_rows.extend(seq_to_rows[s])
        sampled_rows = np.array(sampled_rows)

        sampled_preds = [p[sampled_rows] for p in model_preds]
        sampled_targets = targets[sampled_rows]
        sc = score_ensemble(sampled_preds, sampled_targets, weights=weights)
        boot_avgs.append(sc["avg"])

    arr = np.array(boot_avgs)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def cache_path(model_name: str) -> Path:
    return CACHE_DIR / f"{model_name}.npz"


def is_cached(model_name: str) -> bool:
    return cache_path(model_name).exists()


def save_cache(model_name: str, preds: np.ndarray, targets: np.ndarray, seq_indices: np.ndarray):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path(model_name), preds=preds, targets=targets, seq_indices=seq_indices)


def load_cache(model_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(cache_path(model_name))
    return data["preds"], data["targets"], data["seq_indices"]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model_names: List[str], data_path: Path, force: bool = False):
    """Run online inference for specified models and cache predictions."""
    print(f"Loading validation data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"  {len(df)} rows, {df['seq_ix'].nunique()} sequences\n")

    for name in model_names:
        if name not in MODEL_REGISTRY:
            print(f"  SKIP {name}: not in registry")
            continue

        if is_cached(name) and not force:
            print(f"  SKIP {name}: already cached")
            continue

        spec = MODEL_REGISTRY[name]
        zip_path = ZIPS[spec["zip"]]
        if not zip_path.exists():
            print(f"  SKIP {name}: zip not found: {zip_path}")
            continue

        config_path = ROOT / "configs" / f"{spec['config']}.yaml"
        if not config_path.exists():
            print(f"  SKIP {name}: config not found: {config_path}")
            continue

        print(f"  Running {name} ({spec['type']})...")

        # Extract checkpoint and normalizer to temp dir
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extract(spec["ckpt"], tmpdir)
                zf.extract(spec["norm"], tmpdir)

            t0 = time.time()
            runner = OnlineModelRunner(
                config_path=str(config_path),
                checkpoint_path=str(tmpdir / spec["ckpt"]),
                normalizer_path=str(tmpdir / spec["norm"]),
            )
            preds, targets, seq_ix = runner.run(df)
            elapsed = time.time() - t0

            # Individual model score
            t0_score = weighted_pearson_correlation(targets[:, 0], preds[:, 0])
            t1_score = weighted_pearson_correlation(targets[:, 1], preds[:, 1])
            avg_score = (t0_score + t1_score) / 2.0

            save_cache(name, preds, targets, seq_ix)
            print(f"    {len(preds)} predictions in {elapsed:.0f}s")
            print(f"    Individual: t0={t0_score:.4f}  t1={t1_score:.4f}  avg={avg_score:.4f}")
            print(f"    Cached to {cache_path(name)}\n")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_infer(args):
    data_path = ROOT / args.data
    if args.all:
        model_names = list(MODEL_REGISTRY.keys())
    else:
        model_names = args.models or []

    if not model_names:
        print("No models specified. Use --models or --all")
        return

    run_inference(model_names, data_path, force=args.force)


def cmd_list(args):
    print(f"{'Name':<20} {'Type':<16} {'Config':<30} {'Val':>7} {'Cached':>7}")
    print("-" * 85)
    for name, spec in sorted(MODEL_REGISTRY.items()):
        cached = "YES" if is_cached(name) else "no"
        val_str = f"{spec['val']:.4f}" if spec.get("val") else "  —"
        print(f"{name:<20} {spec['type']:<16} {spec['config']:<30} {val_str:>7} {cached:>7}")

    print(f"\nTotal: {len(MODEL_REGISTRY)} models, {sum(1 for n in MODEL_REGISTRY if is_cached(n))} cached")
    print(f"Cache dir: {CACHE_DIR}")

    # Show groups
    print(f"\nModel groups (for --pool):")
    for gname, members in MODEL_GROUPS.items():
        print(f"  {gname}: {len(members)} models")


def cmd_score(args):
    model_names = args.models
    if not model_names:
        print("No models specified.")
        return

    # Check all cached
    missing = [n for n in model_names if not is_cached(n)]
    if missing:
        print(f"Missing cached predictions for: {', '.join(missing)}")
        print("Run `infer` first for these models.")
        return

    # Load predictions
    preds_list = []
    targets = None
    seq_indices = None
    for name in model_names:
        p, t, s = load_cache(name)
        preds_list.append(p)
        if targets is None:
            targets = t
            seq_indices = s

    weights = args.weights
    sc = score_ensemble(preds_list, targets, weights=weights)
    print(f"\nEnsemble ({len(model_names)} models):")
    print(f"  Models: {', '.join(model_names)}")
    if weights:
        print(f"  Weights: {weights}")
    print(f"  Score: t0={sc['t0']:.4f}  t1={sc['t1']:.4f}  avg={sc['avg']:.4f}")

    if args.bootstrap > 0:
        boot = bootstrap_score(preds_list, targets, seq_indices,
                               n_bootstrap=args.bootstrap, weights=weights)
        print(f"  Bootstrap ({args.bootstrap}x): mean={boot['mean']:.4f}  "
              f"std={boot['std']:.4f}  p10={boot['p10']:.4f}  p90={boot['p90']:.4f}")


def cmd_preset(args):
    name = args.name
    if name not in PRESETS:
        print(f"Unknown preset: {name}")
        print(f"Available: {', '.join(PRESETS.keys())}")
        return

    preset = PRESETS[name]
    print(f"Preset: {name}")
    print(f"  {preset['desc']}")

    model_names = preset["models"]
    weights = preset.get("weights")

    missing = [n for n in model_names if not is_cached(n)]
    if missing:
        print(f"\n  Missing cached predictions for: {', '.join(missing)}")
        print("  Run `infer` first for these models.")
        return

    preds_list = []
    targets = None
    seq_indices = None
    for mname in model_names:
        p, t, s = load_cache(mname)
        preds_list.append(p)
        if targets is None:
            targets = t
            seq_indices = s

    sc = score_ensemble(preds_list, targets, weights=weights)
    print(f"\n  Score: t0={sc['t0']:.4f}  t1={sc['t1']:.4f}  avg={sc['avg']:.4f}")

    boot = bootstrap_score(preds_list, targets, seq_indices, n_bootstrap=200, weights=weights)
    print(f"  Bootstrap (200x): mean={boot['mean']:.4f}  std={boot['std']:.4f}  "
          f"p10={boot['p10']:.4f}  p90={boot['p90']:.4f}")


def cmd_presets(args):
    """List all presets."""
    print(f"{'Name':<25} {'Models':>6} {'Description'}")
    print("-" * 80)
    for name, preset in PRESETS.items():
        n = len(preset["models"])
        print(f"{name:<25} {n:>6} {preset['desc']}")


def cmd_greedy(args):
    """Greedy forward selection: start with best single model, add one at a time."""
    # Resolve pool
    pool_name = args.pool
    if pool_name in MODEL_GROUPS:
        pool = MODEL_GROUPS[pool_name]
    else:
        pool = pool_name.split(",")

    # Filter to cached only
    pool = [n for n in pool if is_cached(n)]
    if not pool:
        print("No cached models in pool. Run `infer` first.")
        return

    max_models = args.max_models
    use_7030 = args.weighted_attn
    print(f"Greedy search over {len(pool)} cached models (max {max_models})")
    if use_7030:
        print("  Using 70/30 GRU/attn weighting")
    print()

    # Load all predictions
    all_preds = {}
    targets = None
    seq_indices = None
    for name in pool:
        p, t, s = load_cache(name)
        all_preds[name] = p
        if targets is None:
            targets = t
            seq_indices = s

    def compute_weights(selected_names):
        """Compute weights: uniform or 70/30 GRU/attn split."""
        if not use_7030:
            return None  # uniform
        n_gru = sum(1 for n in selected_names if n.startswith("gru_"))
        n_attn = len(selected_names) - n_gru
        if n_attn == 0 or n_gru == 0:
            return None  # uniform if all same type
        w_gru = 0.70 / n_gru
        w_attn = 0.30 / n_attn
        return [w_gru if n.startswith("gru_") else w_attn for n in selected_names]

    # Find best single model
    print("Step 1: Best single model")
    best_single = None
    best_score = -1
    for name in pool:
        sc = score_ensemble([all_preds[name]], targets)
        if sc["avg"] > best_score:
            best_score = sc["avg"]
            best_single = name
    print(f"  {best_single}: {best_score:.4f}")

    selected = [best_single]
    remaining = [n for n in pool if n != best_single]

    # Greedy forward selection
    for step in range(2, max_models + 1):
        if not remaining:
            break

        print(f"\nStep {step}: Adding model {step}/{max_models}")
        best_add = None
        best_add_score = -1

        for candidate in remaining:
            trial = selected + [candidate]
            trial_preds = [all_preds[n] for n in trial]
            weights = compute_weights(trial)
            sc = score_ensemble(trial_preds, targets, weights=weights)
            if sc["avg"] > best_add_score:
                best_add_score = sc["avg"]
                best_add = candidate

        # Check if adding helps
        prev_preds = [all_preds[n] for n in selected]
        prev_weights = compute_weights(selected)
        prev_score = score_ensemble(prev_preds, targets, weights=prev_weights)["avg"]

        delta = best_add_score - prev_score
        selected.append(best_add)
        remaining.remove(best_add)

        print(f"  + {best_add}: {best_add_score:.4f} (delta: {delta:+.4f})")

        if delta < 0:
            print(f"  WARNING: adding {best_add} DECREASED score by {delta:.4f}")

    # Final summary
    print(f"\n{'='*70}")
    print("GREEDY SEARCH RESULTS")
    print(f"{'='*70}")
    final_preds = [all_preds[n] for n in selected]
    final_weights = compute_weights(selected)
    final_sc = score_ensemble(final_preds, targets, weights=final_weights)

    print(f"Selected {len(selected)} models:")
    for i, name in enumerate(selected):
        spec = MODEL_REGISTRY.get(name, {})
        val_str = f"(val={spec['val']:.4f})" if spec.get("val") else ""
        w_str = f"w={final_weights[i]:.3f}" if final_weights else "uniform"
        print(f"  {i+1}. {name} [{w_str}] {val_str}")

    print(f"\nEnsemble: t0={final_sc['t0']:.4f}  t1={final_sc['t1']:.4f}  avg={final_sc['avg']:.4f}")

    boot = bootstrap_score(final_preds, targets, seq_indices, n_bootstrap=200, weights=final_weights)
    print(f"Bootstrap (200x): mean={boot['mean']:.4f}  std={boot['std']:.4f}  "
          f"p10={boot['p10']:.4f}  p90={boot['p90']:.4f}")

    # Also show intermediate ensemble sizes
    print(f"\nProgression:")
    print(f"  {'N':>3} {'Avg':>7} {'Delta':>7} {'Model Added'}")
    print(f"  {'-'*40}")
    running = []
    prev = 0.0
    for i, name in enumerate(selected):
        running.append(all_preds[name])
        w = compute_weights(selected[:i+1])
        sc = score_ensemble(running, targets, weights=w)
        delta = sc["avg"] - prev
        prev = sc["avg"]
        print(f"  {i+1:>3} {sc['avg']:>7.4f} {delta:>+7.4f} {name}")


def cmd_diversity(args):
    """Show prediction correlation matrix between cached models."""
    pool_name = args.pool
    if pool_name in MODEL_GROUPS:
        pool = MODEL_GROUPS[pool_name]
    else:
        pool = pool_name.split(",")

    pool = [n for n in pool if is_cached(n)]
    if len(pool) < 2:
        print("Need at least 2 cached models.")
        return

    # Load all predictions
    all_preds = {}
    for name in pool:
        p, _, _ = load_cache(name)
        all_preds[name] = p

    n = len(pool)
    # Compute correlation matrix on flattened predictions (both targets)
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            pi = all_preds[pool[i]].flatten()
            pj = all_preds[pool[j]].flatten()
            corr_matrix[i, j] = np.corrcoef(pi, pj)[0, 1]

    print(f"Prediction correlation matrix ({n} models):\n")
    # Header
    header = f"{'':>16}" + "".join(f"{pool[j][:10]:>12}" for j in range(n))
    print(header)
    for i in range(n):
        row = f"{pool[i]:>16}"
        for j in range(n):
            val = corr_matrix[i, j]
            row += f"{val:>12.4f}"
        print(row)

    # Show pairs with lowest correlation (most diverse)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((corr_matrix[i, j], pool[i], pool[j]))
    pairs.sort()

    print(f"\nMost diverse pairs (lowest correlation):")
    for corr, a, b in pairs[:10]:
        print(f"  {corr:.4f}  {a} <-> {b}")

    # Average correlation per model (lower = more diverse)
    print(f"\nMean correlation per model (lower = more unique):")
    for i in range(n):
        avg_corr = (corr_matrix[i].sum() - 1.0) / (n - 1)
        print(f"  {avg_corr:.4f}  {pool[i]}")


def cmd_exhaustive(args):
    """Exhaustive search over all k-model combos from a pool."""
    pool_name = args.pool
    if pool_name in MODEL_GROUPS:
        pool = MODEL_GROUPS[pool_name]
    else:
        pool = pool_name.split(",")

    pool = [n for n in pool if is_cached(n)]
    k = args.k
    top_n = args.top

    n_combos = 1
    for i in range(k):
        n_combos = n_combos * (len(pool) - i) // (i + 1)

    if n_combos > 50000:
        print(f"Too many combinations: {n_combos} (C({len(pool)},{k})). Use smaller pool or k.")
        return

    print(f"Exhaustive search: C({len(pool)},{k}) = {n_combos} combinations")

    # Load all predictions
    all_preds = {}
    targets = None
    seq_indices = None
    for name in pool:
        p, t, s = load_cache(name)
        all_preds[name] = p
        if targets is None:
            targets = t
            seq_indices = s

    results = []
    for combo in itertools.combinations(pool, k):
        preds_list = [all_preds[n] for n in combo]

        # Use 70/30 if requested
        weights = None
        if args.weighted_attn:
            n_gru = sum(1 for n in combo if n.startswith("gru_"))
            n_attn = len(combo) - n_gru
            if n_attn > 0 and n_gru > 0:
                w_gru = 0.70 / n_gru
                w_attn = 0.30 / n_attn
                weights = [w_gru if n.startswith("gru_") else w_attn for n in combo]

        sc = score_ensemble(preds_list, targets, weights=weights)
        results.append((sc["avg"], sc["t0"], sc["t1"], combo))

    results.sort(reverse=True)
    print(f"\nTop {top_n} ensembles (k={k}):")
    print(f"  {'Avg':>7} {'t0':>7} {'t1':>7}  Models")
    print(f"  {'-'*60}")
    for avg, t0, t1, combo in results[:top_n]:
        print(f"  {avg:>7.4f} {t0:>7.4f} {t1:>7.4f}  {', '.join(combo)}")

    if len(results) > top_n:
        worst = results[-1]
        print(f"\n  Worst: {worst[0]:.4f}  {', '.join(worst[3])}")
        print(f"  Range: {results[-1][0]:.4f} - {results[0][0]:.4f} (spread: {results[0][0]-results[-1][0]:.4f})")


def cmd_compare_presets(args):
    """Score all presets that have cached models and rank them."""
    results = []
    for name, preset in PRESETS.items():
        model_names = preset["models"]
        missing = [n for n in model_names if not is_cached(n)]
        if missing:
            continue

        preds_list = []
        targets = None
        seq_indices = None
        for mname in model_names:
            p, t, s = load_cache(mname)
            preds_list.append(p)
            if targets is None:
                targets = t
                seq_indices = s

        weights = preset.get("weights")
        sc = score_ensemble(preds_list, targets, weights=weights)
        boot = bootstrap_score(preds_list, targets, seq_indices, n_bootstrap=200, weights=weights)
        results.append({
            "name": name,
            "n_models": len(model_names),
            **sc,
            "boot_mean": boot["mean"],
            "boot_p10": boot["p10"],
            "boot_std": boot["std"],
        })

    if not results:
        print("No presets have all models cached. Run `infer` first.")
        return

    results.sort(key=lambda x: x["avg"], reverse=True)
    print(f"{'Preset':<25} {'N':>3} {'Avg':>7} {'t0':>7} {'t1':>7} {'bMean':>7} {'bP10':>7} {'bStd':>6}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<25} {r['n_models']:>3} {r['avg']:>7.4f} {r['t0']:>7.4f} "
              f"{r['t1']:>7.4f} {r['boot_mean']:>7.4f} {r['boot_p10']:>7.4f} {r['boot_std']:>6.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Local ensemble validation with prediction caching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # infer
    p_infer = subparsers.add_parser("infer", help="Run inference and cache predictions")
    p_infer.add_argument("--models", nargs="+", default=None, help="Model names to infer")
    p_infer.add_argument("--all", action="store_true", help="Infer all models in registry")
    p_infer.add_argument("--force", action="store_true", help="Re-run even if cached")
    p_infer.add_argument("--data", default="datasets/valid.parquet", help="Validation data path")

    # list
    subparsers.add_parser("list", help="List all models and cache status")

    # score
    p_score = subparsers.add_parser("score", help="Score a specific ensemble")
    p_score.add_argument("--models", nargs="+", required=True, help="Model names")
    p_score.add_argument("--weights", nargs="+", type=float, default=None, help="Custom weights")
    p_score.add_argument("--bootstrap", type=int, default=200, help="Bootstrap resamples (0 to skip)")

    # preset
    p_preset = subparsers.add_parser("preset", help="Score a named preset")
    p_preset.add_argument("--name", required=True, help="Preset name")

    # presets
    subparsers.add_parser("presets", help="List all presets")

    # compare
    subparsers.add_parser("compare", help="Score and rank all presets with cached models")

    # greedy
    p_greedy = subparsers.add_parser("greedy", help="Greedy forward ensemble selection")
    p_greedy.add_argument("--pool", default="all", help="Model group or comma-separated names")
    p_greedy.add_argument("--max-models", type=int, default=8, help="Max models in ensemble")
    p_greedy.add_argument("--weighted-attn", action="store_true",
                          help="Use 70/30 GRU/attn weighting instead of uniform")

    # diversity
    p_div = subparsers.add_parser("diversity", help="Show prediction correlation matrix")
    p_div.add_argument("--pool", default="all", help="Model group or comma-separated names")

    # exhaustive
    p_exh = subparsers.add_parser("exhaustive", help="Exhaustive search over k-model combos")
    p_exh.add_argument("--pool", default="all", help="Model group or comma-separated names")
    p_exh.add_argument("--k", type=int, required=True, help="Ensemble size to search")
    p_exh.add_argument("--top", type=int, default=10, help="Show top N results")
    p_exh.add_argument("--weighted-attn", action="store_true",
                        help="Use 70/30 GRU/attn weighting")

    args = parser.parse_args()

    if args.command == "infer":
        cmd_infer(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "score":
        cmd_score(args)
    elif args.command == "preset":
        cmd_preset(args)
    elif args.command == "presets":
        cmd_presets(args)
    elif args.command == "compare":
        cmd_compare_presets(args)
    elif args.command == "greedy":
        cmd_greedy(args)
    elif args.command == "diversity":
        cmd_diversity(args)
    elif args.command == "exhaustive":
        cmd_exhaustive(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
