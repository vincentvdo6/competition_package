#!/usr/bin/env python
"""Score ensemble candidates with online inference + bootstrap ranking.

Runs online step-by-step inference (matching competition semantics) for each
model, caches per-sequence predictions, then assembles and scores candidate
ensembles with bootstrap confidence intervals.

Usage:
    python scripts/score_ensemble_candidates.py \
        --models model1.json model2.json ... \
        --candidates candidates.json \
        --data datasets/valid.parquet \
        --bootstrap 200

Model JSON format (one per model):
    {"config": "configs/X.yaml", "checkpoint": "logs/X.pt",
     "normalizer": "logs/normalizer_X.npz", "name": "gru_seed42"}

Candidates JSON format:
    {
      "champion": {"model_indices": [0,1,2,3,4], "weights": "uniform"},
      "candidates": [
        {"name": "gru5_attn5_uniform", "model_indices": [0,1,2,3,4,5,6,7,8,9],
         "weights": "uniform"},
        {"name": "gru5_attn5_70_30", "model_indices": [0,1,2,3,4,5,6,7,8,9],
         "weights_t0": [0.14,0.14,0.14,0.14,0.14,0.06,0.06,0.06,0.06,0.06],
         "weights_t1": [0.06,0.06,0.06,0.06,0.06,0.14,0.14,0.14,0.14,0.14]}
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import weighted_pearson_correlation, DataPoint
from src.models.gru_attention import GRUAttentionModel
from src.models.gru_baseline import GRUBaseline
from src.models.lstm_model import LSTMModel
from src.data.preprocessing import (
    DerivedFeatureBuilder,
    InteractionFeatureBuilder,
    Normalizer,
    TemporalBuffer,
)


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

class OnlineModelRunner:
    """Run a single model through validation data with online inference."""

    def __init__(self, config_path: str, checkpoint_path: str, normalizer_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.normalizer = Normalizer.load(normalizer_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data_cfg = self.config.get("data", {})
        self.derived_features = bool(data_cfg.get("derived_features", False))
        self.temporal_features = bool(
            data_cfg.get("temporal_features", False) and self.derived_features
        )
        self.interaction_features = bool(data_cfg.get("interaction_features", False))

        model_type = self.config.get("model", {}).get("type", "gru")
        if model_type == "lstm":
            self.model = LSTMModel(self.config)
        elif model_type == "gru_attention":
            self.model = GRUAttentionModel(self.config)
        else:
            self.model = GRUBaseline(self.config)

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def run(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run online inference, return (predictions, targets, seq_indices).

        All arrays are aligned and filtered to need_prediction=True rows only.
        seq_indices is the seq_ix for each prediction row (for bootstrap).
        """
        predictions = []
        targets = []
        seq_indices = []

        current_seq_ix = None
        hidden = None
        temporal_buffer = TemporalBuffer() if self.temporal_features else None

        for row in df.values:
            seq_ix = int(row[0])
            need_prediction = bool(row[2])
            lob_data = row[3:35]
            labels = row[35:]

            # Reset on new sequence
            if current_seq_ix != seq_ix:
                current_seq_ix = seq_ix
                hidden = None
                if temporal_buffer is not None:
                    temporal_buffer.reset()

            # Build features
            raw = lob_data.reshape(1, -1).astype(np.float32)
            if self.derived_features:
                derived = DerivedFeatureBuilder.compute(raw)
                raw = np.concatenate([raw, derived], axis=-1)
            if self.temporal_features:
                raw = temporal_buffer.compute_step(raw.squeeze(0)).reshape(1, -1)
            if self.interaction_features:
                interactions = InteractionFeatureBuilder.compute(
                    raw, has_derived=self.derived_features
                )
                raw = np.concatenate([raw, interactions], axis=-1)

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

        return (
            np.array(predictions),
            np.array(targets, dtype=np.float64),
            np.array(seq_indices),
        )


# ---------------------------------------------------------------------------
# Ensemble scoring
# ---------------------------------------------------------------------------

def score_ensemble(
    model_preds: List[np.ndarray],
    targets: np.ndarray,
    weights: Optional[List[float]] = None,
    weights_t0: Optional[List[float]] = None,
    weights_t1: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Score an ensemble of model predictions.

    Args:
        model_preds: List of (N, 2) prediction arrays
        targets: (N, 2) ground truth
        weights: Uniform per-model weights (ignored if per-target given)
        weights_t0: Per-model weights for t0
        weights_t1: Per-model weights for t1

    Returns:
        Dict with t0, t1, avg scores
    """
    n_models = len(model_preds)

    if weights_t0 is not None and weights_t1 is not None:
        # Per-target weighting — validate counts
        if len(weights_t0) != n_models:
            raise ValueError(
                f"weights_t0 length ({len(weights_t0)}) != model count ({n_models})"
            )
        if len(weights_t1) != n_models:
            raise ValueError(
                f"weights_t1 length ({len(weights_t1)}) != model count ({n_models})"
            )
        w0 = np.array(weights_t0) / np.sum(weights_t0)
        w1 = np.array(weights_t1) / np.sum(weights_t1)
        ensemble_pred = np.zeros_like(targets)
        for i, p in enumerate(model_preds):
            ensemble_pred[:, 0] += w0[i] * p[:, 0]
            ensemble_pred[:, 1] += w1[i] * p[:, 1]
    else:
        if weights is None:
            weights = [1.0 / n_models] * n_models
        else:
            if len(weights) != n_models:
                raise ValueError(
                    f"weights length ({len(weights)}) != model count ({n_models})"
                )
            w_total = sum(weights)
            weights = [w / w_total for w in weights]
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
    **weight_kwargs,
) -> Dict[str, Any]:
    """Bootstrap ensemble score at sequence level.

    Returns dict with:
        mean_t0, mean_t1, mean_avg: bootstrap means
        p10_t0, p10_t1, p10_avg: 10th percentile
        std_avg: bootstrap std of avg
        scores: list of all bootstrap avg scores
    """
    rng = np.random.RandomState(seed)
    unique_seqs = np.unique(seq_indices)
    n_seqs = len(unique_seqs)

    # Build index mapping: seq_ix -> row indices
    seq_to_rows = {}
    for idx, s in enumerate(seq_indices):
        if s not in seq_to_rows:
            seq_to_rows[s] = []
        seq_to_rows[s].append(idx)

    boot_scores = {"t0": [], "t1": [], "avg": []}

    for _ in range(n_bootstrap):
        # Resample sequences with replacement
        sampled_seqs = rng.choice(unique_seqs, size=n_seqs, replace=True)
        sampled_rows = []
        for s in sampled_seqs:
            sampled_rows.extend(seq_to_rows[s])
        sampled_rows = np.array(sampled_rows)

        sampled_preds = [p[sampled_rows] for p in model_preds]
        sampled_targets = targets[sampled_rows]

        scores = score_ensemble(sampled_preds, sampled_targets, **weight_kwargs)
        for k in boot_scores:
            boot_scores[k].append(scores[k])

    result = {}
    for k in ["t0", "t1", "avg"]:
        arr = np.array(boot_scores[k])
        result[f"mean_{k}"] = float(np.mean(arr))
        result[f"p10_{k}"] = float(np.percentile(arr, 10))
        result[f"std_{k}"] = float(np.std(arr))
    result["scores"] = boot_scores["avg"]
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Score ensemble candidates with bootstrap ranking"
    )
    parser.add_argument(
        "--models", type=str, nargs="+", required=True,
        help="JSON files defining individual models (config/checkpoint/normalizer/name)"
    )
    parser.add_argument(
        "--candidates", type=str, required=True,
        help="JSON file defining champion + candidate ensembles"
    )
    parser.add_argument(
        "--data", type=str, default="datasets/valid.parquet",
        help="Path to validation parquet"
    )
    parser.add_argument("--bootstrap", type=int, default=200, help="Number of bootstrap resamples")
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap random seed")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    print("=" * 70)
    print("ENSEMBLE CANDIDATE SCORER")
    print("=" * 70)

    # Load validation data
    data_path = ROOT / args.data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"  {len(df)} rows, {df['seq_ix'].nunique()} sequences")

    # Load and run each model
    all_preds = []
    model_names = []
    targets = None
    seq_indices = None

    for model_json_path in args.models:
        with open(model_json_path, "r") as f:
            model_def = json.load(f)

        name = model_def.get("name", Path(model_json_path).stem)
        model_names.append(name)
        print(f"\n--- Running model: {name} ---")

        t0 = time.time()
        runner = OnlineModelRunner(
            config_path=str(ROOT / model_def["config"]),
            checkpoint_path=str(ROOT / model_def["checkpoint"]),
            normalizer_path=str(ROOT / model_def["normalizer"]),
        )
        preds, tgts, seq_ix = runner.run(df)
        elapsed = time.time() - t0
        print(f"  {len(preds)} predictions in {elapsed:.1f}s")

        # Individual model score
        t0_score = weighted_pearson_correlation(tgts[:, 0], preds[:, 0])
        t1_score = weighted_pearson_correlation(tgts[:, 1], preds[:, 1])
        print(f"  Individual: t0={t0_score:.4f}  t1={t1_score:.4f}  avg={(t0_score+t1_score)/2:.4f}")

        all_preds.append(preds)
        if targets is None:
            targets = tgts
            seq_indices = seq_ix

    # Load candidate definitions
    with open(args.candidates, "r") as f:
        candidates_def = json.load(f)

    champion_def = candidates_def.get("champion", None)
    candidate_list = candidates_def.get("candidates", [])

    # Score champion if defined
    champion_avg = None
    if champion_def is not None:
        idx = champion_def["model_indices"]
        champ_preds = [all_preds[i] for i in idx]
        wkw = _extract_weight_kwargs(champion_def, len(idx))
        champ_score = score_ensemble(champ_preds, targets, **wkw)
        champ_boot = bootstrap_score(
            champ_preds, targets, seq_indices, args.bootstrap, args.seed, **wkw
        )
        champion_avg = champ_score["avg"]
        print(f"\n{'='*70}")
        print(f"CHAMPION: avg={champ_score['avg']:.4f}  t0={champ_score['t0']:.4f}  t1={champ_score['t1']:.4f}")
        print(f"  Bootstrap: mean={champ_boot['mean_avg']:.4f}  p10={champ_boot['p10_avg']:.4f}  std={champ_boot['std_avg']:.4f}")

    # Score each candidate
    results = []
    for cand in candidate_list:
        name = cand["name"]
        idx = cand["model_indices"]
        cand_preds = [all_preds[i] for i in idx]
        wkw = _extract_weight_kwargs(cand, len(idx))

        score = score_ensemble(cand_preds, targets, **wkw)
        boot = bootstrap_score(
            cand_preds, targets, seq_indices, args.bootstrap, args.seed, **wkw
        )

        delta_avg = score["avg"] - champion_avg if champion_avg is not None else 0.0
        delta_p10 = boot["p10_avg"] - (champ_boot["p10_avg"] if champion_avg is not None else 0.0)
        delta_mean = boot["mean_avg"] - (champ_boot["mean_avg"] if champion_avg is not None else 0.0)

        results.append({
            "name": name,
            "avg": score["avg"],
            "t0": score["t0"],
            "t1": score["t1"],
            "boot_mean": boot["mean_avg"],
            "boot_p10": boot["p10_avg"],
            "boot_std": boot["std_avg"],
            "delta_avg": delta_avg,
            "delta_mean": delta_mean,
            "delta_p10": delta_p10,
        })

    # Sort by bootstrap mean delta (best first)
    results.sort(key=lambda x: x["delta_mean"], reverse=True)

    # Print ranked table
    print(f"\n{'='*70}")
    print("CANDIDATE RANKING (sorted by bootstrap mean delta)")
    print(f"{'='*70}")
    print(f"{'Name':<40} {'Avg':>7} {'t0':>7} {'t1':>7} {'dMean':>7} {'dP10':>7}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['name']:<40} {r['avg']:>7.4f} {r['t0']:>7.4f} {r['t1']:>7.4f} "
            f"{r['delta_mean']:>+7.4f} {r['delta_p10']:>+7.4f}"
        )

    # Recommend slots
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    safe = [r for r in results if r["delta_p10"] >= 0]
    if safe:
        # Slot 1 (conservative): highest p10 delta (best worst-case)
        slot1 = max(safe, key=lambda x: x["delta_p10"])
        print(f"  Slot 1 (conservative): {slot1['name']} (p10 delta: {slot1['delta_p10']:+.4f})")
        # Slot 2 (upside): highest mean delta, but prefer a different candidate
        safe_by_mean = sorted(safe, key=lambda x: x["delta_mean"], reverse=True)
        slot2 = safe_by_mean[0]
        if slot2["name"] == slot1["name"] and len(safe_by_mean) > 1:
            slot2 = safe_by_mean[1]
        if slot2["name"] == slot1["name"]:
            print(f"  Slot 2 (upside):       (same as Slot 1 — only one safe candidate)")
        else:
            print(f"  Slot 2 (upside):       {slot2['name']} (mean delta: {slot2['delta_mean']:+.4f})")
    else:
        print("  WARNING: No candidate has non-negative p10 delta vs champion.")
        print(f"  Best available: {results[0]['name']} (mean delta: {results[0]['delta_mean']:+.4f})")

    # Save CSV
    if args.output:
        out_path = ROOT / args.output
        pd.DataFrame(results).to_csv(out_path, index=False)
        print(f"\nResults saved to {out_path}")

    print()


def _extract_weight_kwargs(cand_def: dict, n_models: int) -> dict:
    """Extract weight keyword args from a candidate definition."""
    if "weights_t0" in cand_def and "weights_t1" in cand_def:
        return {
            "weights_t0": cand_def["weights_t0"],
            "weights_t1": cand_def["weights_t1"],
        }
    w = cand_def.get("weights", "uniform")
    if w == "uniform" or w is None:
        return {"weights": [1.0 / n_models] * n_models}
    if isinstance(w, list):
        return {"weights": w}
    return {"weights": [1.0 / n_models] * n_models}


if __name__ == "__main__":
    main()
