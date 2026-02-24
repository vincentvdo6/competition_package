#!/usr/bin/env python
"""Analyze kfold predictions per-target and find optimal vanilla+kfold mixed ensembles.

Usage:
  python scripts/analyze_kfold_pertarget.py --vanilla-cache cache/vanilla_preds --kfold-cache cache/kfold_preds

Outputs:
  1. Per-target scores for all kfold models
  2. Best kfold model rankings by t0 and t1
  3. Mixed vanilla+kfold ensemble scores with different kfold subsets
  4. Recommended submission configurations
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def weighted_corr(y, x):
    """Weighted Pearson with weights |y|."""
    y = y.astype(np.float64, copy=False)
    x = x.astype(np.float64, copy=False)
    w = np.abs(y)
    sw = float(np.sum(w))
    if sw <= 0:
        return 0.0
    mx = float(np.sum(w * x)) / sw
    my = float(np.sum(w * y)) / sw
    vx = max(float(np.sum(w * x * x)) / sw - mx * mx, 0.0)
    vy = max(float(np.sum(w * y * y)) / sw - my * my, 0.0)
    if vx <= 1e-18 or vy <= 1e-18:
        return 0.0
    cov = float(np.sum(w * x * y)) / sw - mx * my
    return float(cov / np.sqrt(vx * vy))


def load_all_preds(cache_dir, glob_pattern="*.npz"):
    """Load all predictions from a cache directory."""
    cache_path = Path(cache_dir)
    preds = {}
    for npz_path in sorted(cache_path.glob(glob_pattern)):
        if npz_path.stem.startswith("_"):
            continue
        data = np.load(npz_path)
        preds[npz_path.stem] = data["preds"].astype(np.float32)
    return preds


def load_targets(cache_dir):
    """Load shared targets."""
    targets_path = Path(cache_dir) / "_targets.npz"
    return np.load(targets_path)["targets"].astype(np.float64)


def score_single(targets, preds):
    """Score a single model's predictions."""
    t0 = weighted_corr(targets[:, 0], preds[:, 0])
    t1 = weighted_corr(targets[:, 1], preds[:, 1])
    return t0, t1, (t0 + t1) / 2


def score_weighted_ensemble(targets, pred_dict, model_names, weights_t0, weights_t1):
    """Score a weighted ensemble with per-target weights."""
    n = targets.shape[0]
    out = np.zeros((n, 2), dtype=np.float64)

    w0_sum = sum(weights_t0)
    w1_sum = sum(weights_t1)

    for name, w0, w1 in zip(model_names, weights_t0, weights_t1):
        p = pred_dict[name].astype(np.float64, copy=False)
        out[:, 0] += (w0 / w0_sum) * p[:, 0]
        out[:, 1] += (w1 / w1_sum) * p[:, 1]

    out = np.clip(out, -6.0, 6.0)
    t0 = weighted_corr(targets[:, 0], out[:, 0])
    t1 = weighted_corr(targets[:, 1], out[:, 1])
    return t0, t1, (t0 + t1) / 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vanilla-cache", default="cache/vanilla_preds")
    parser.add_argument("--kfold-cache", default="cache/kfold_preds")
    args = parser.parse_args()

    # Load targets (from vanilla cache, same validation set)
    targets = load_targets(args.vanilla_cache)
    print(f"Targets: {targets.shape[0]} samples\n")

    # Load vanilla predictions
    vanilla_preds = load_all_preds(args.vanilla_cache, "gru_parity_v1_seed*.npz")
    print(f"Vanilla models loaded: {len(vanilla_preds)}")

    # Load kfold predictions
    kfold_preds = load_all_preds(args.kfold_cache, "gru_kfold_v1_*.npz")
    print(f"Kfold models loaded: {len(kfold_preds)}")

    # ================================================================
    # 1. Per-target scores for all kfold models
    # ================================================================
    print("\n" + "=" * 70)
    print("KFOLD MODEL SCORES (per-target)")
    print("=" * 70)
    kf_scores = {}
    for name in sorted(kfold_preds.keys()):
        t0, t1, avg = score_single(targets, kfold_preds[name])
        kf_scores[name] = {"t0": t0, "t1": t1, "avg": avg}
        print(f"  {name:40s}  avg={avg:.4f}  t0={t0:.4f}  t1={t1:.4f}")

    # Rank by t0 and t1
    print("\n--- Ranked by t0 ---")
    for name in sorted(kf_scores, key=lambda n: kf_scores[n]["t0"], reverse=True):
        s = kf_scores[name]
        print(f"  {name:40s}  t0={s['t0']:.4f}")

    print("\n--- Ranked by t1 ---")
    for name in sorted(kf_scores, key=lambda n: kf_scores[n]["t1"], reverse=True):
        s = kf_scores[name]
        print(f"  {name:40s}  t1={s['t1']:.4f}")

    # ================================================================
    # 2. Vanilla model per-target scores (top 10)
    # ================================================================
    print("\n" + "=" * 70)
    print("VANILLA MODEL SCORES (per-target, top 10 by avg)")
    print("=" * 70)
    v_scores = {}
    for name in sorted(vanilla_preds.keys()):
        t0, t1, avg = score_single(targets, vanilla_preds[name])
        v_scores[name] = {"t0": t0, "t1": t1, "avg": avg}

    for name in sorted(v_scores, key=lambda n: v_scores[n]["avg"], reverse=True)[:10]:
        s = v_scores[name]
        print(f"  {name:40s}  avg={s['avg']:.4f}  t0={s['t0']:.4f}  t1={s['t1']:.4f}")

    # ================================================================
    # 3. Anchor: current PB ensemble (10v + 4kf equal)
    # ================================================================
    print("\n" + "=" * 70)
    print("ANCHOR ENSEMBLE (Current PB: 10v + 4kf equal)")
    print("=" * 70)

    # Top 10 vanilla seeds
    vanilla_top10 = sorted(v_scores, key=lambda n: v_scores[n]["avg"], reverse=True)[:10]

    # Merge all preds into one dict
    all_preds = {}
    all_preds.update(vanilla_preds)
    all_preds.update(kfold_preds)

    # Current PB: 10v + 4kf(seed42, folds 0/1/3/4) equal weight
    anchor_kf = [n for n in sorted(kfold_preds.keys())
                 if "seed42" in n and "fold2" not in n]
    anchor_models = vanilla_top10 + anchor_kf
    anchor_w0 = [1.0] * len(vanilla_top10) + [1.0] * len(anchor_kf)
    anchor_w1 = [1.0] * len(vanilla_top10) + [1.0] * len(anchor_kf)

    t0, t1, avg = score_weighted_ensemble(targets, all_preds, anchor_models, anchor_w0, anchor_w1)
    print(f"  Anchor: avg={avg:.4f}  t0={t0:.4f}  t1={t1:.4f}")
    print(f"  Models: {len(anchor_models)} ({len(vanilla_top10)}v + {len(anchor_kf)}kf)")

    # ================================================================
    # 4. Per-target kfold selection experiments
    # ================================================================
    print("\n" + "=" * 70)
    print("PER-TARGET KFOLD EXPERIMENTS")
    print("=" * 70)

    # Experiment: select kfold models by t0 score for t0, by t1 score for t1
    kf_names = sorted(kfold_preds.keys())
    kf_by_t0 = sorted(kf_names, key=lambda n: kf_scores[n]["t0"], reverse=True)
    kf_by_t1 = sorted(kf_names, key=lambda n: kf_scores[n]["t1"], reverse=True)

    for n_kf_t0 in [2, 3, 4, 5, 6]:
        for n_kf_t1 in [2, 3, 4, 5, 6]:
            kf_t0_set = set(kf_by_t0[:n_kf_t0])
            kf_t1_set = set(kf_by_t1[:n_kf_t1])
            all_kf_in_ensemble = kf_t0_set | kf_t1_set

            models = vanilla_top10 + sorted(all_kf_in_ensemble)
            w0 = [1.0] * len(vanilla_top10) + [1.0 if n in kf_t0_set else 0.0 for n in sorted(all_kf_in_ensemble)]
            w1 = [1.0] * len(vanilla_top10) + [1.0 if n in kf_t1_set else 0.0 for n in sorted(all_kf_in_ensemble)]

            # Need non-zero weights for both targets
            if sum(w0) == 0 or sum(w1) == 0:
                continue

            t0, t1, avg = score_weighted_ensemble(targets, all_preds, models, w0, w1)
            n_total = len(models)
            tag = f"t0_top{n_kf_t0}_t1_top{n_kf_t1}"
            delta = avg - 0.2895  # delta vs PB LB (not local val!)
            print(f"  {tag:20s}  avg={avg:.4f}  t0={t0:.4f}  t1={t1:.4f}  "
                  f"models={n_total}  kf_in_ens={len(all_kf_in_ensemble)}")

    # ================================================================
    # 5. Kfold weight sweep with per-target selection
    # ================================================================
    print("\n" + "=" * 70)
    print("KFOLD WEIGHT SWEEP (per-target selection)")
    print("=" * 70)

    # Best config: top-4 kf by t0, top-4 kf by t1 (may overlap)
    for kf_w in [0.5, 1.0, 1.5, 2.0, 3.0]:
        for n_kf in [3, 4, 5, 6]:
            kf_t0_set = set(kf_by_t0[:n_kf])
            kf_t1_set = set(kf_by_t1[:n_kf])
            all_kf = kf_t0_set | kf_t1_set

            models = vanilla_top10 + sorted(all_kf)
            w0 = [1.0] * len(vanilla_top10) + [kf_w if n in kf_t0_set else 0.0 for n in sorted(all_kf)]
            w1 = [1.0] * len(vanilla_top10) + [kf_w if n in kf_t1_set else 0.0 for n in sorted(all_kf)]

            if sum(w0) == 0 or sum(w1) == 0:
                continue

            t0, t1, avg = score_weighted_ensemble(targets, all_preds, models, w0, w1)
            tag = f"kfw{kf_w:.1f}_top{n_kf}"
            print(f"  {tag:20s}  avg={avg:.4f}  t0={t0:.4f}  t1={t1:.4f}  "
                  f"models={len(models)}")

    # ================================================================
    # 6. Prediction correlation: kfold vs vanilla
    # ================================================================
    print("\n" + "=" * 70)
    print("PREDICTION CORRELATION (kfold vs vanilla top-10 average)")
    print("=" * 70)

    # Vanilla average predictions
    v_avg = np.mean([vanilla_preds[n] for n in vanilla_top10], axis=0)

    for name in sorted(kfold_preds.keys()):
        p = kfold_preds[name]
        corr_t0 = np.corrcoef(v_avg[:, 0], p[:, 0])[0, 1]
        corr_t1 = np.corrcoef(v_avg[:, 1], p[:, 1])[0, 1]
        print(f"  {name:40s}  corr_t0={corr_t0:.4f}  corr_t1={corr_t1:.4f}")

    # Kfold-kfold correlations (abbreviated)
    print("\n--- Kfold inter-model correlation (t0) ---")
    kf_list = sorted(kfold_preds.keys())
    for i, n1 in enumerate(kf_list):
        for n2 in kf_list[i+1:]:
            corr = np.corrcoef(kfold_preds[n1][:, 0], kfold_preds[n2][:, 0])[0, 1]
            if corr < 0.95:  # only show interesting pairs
                print(f"  {n1} vs {n2}: {corr:.4f}")


if __name__ == "__main__":
    main()
