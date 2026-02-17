#!/usr/bin/env python
"""Compute prediction neutralization beta for vanilla GRU ensemble.

Generates val predictions from vanilla parity_v1 checkpoints, fits ridge
regression beta, and tests different neutralization strengths (alpha).

Usage:
    python scripts/compute_neutralization_beta.py
    python scripts/compute_neutralization_beta.py --n-models 10 --alphas 0.0 0.3 0.5 0.7 1.0
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


class VanillaGRU(nn.Module):
    """Minimal vanilla GRU matching parity_v1 config."""
    def __init__(self, input_size=32, hidden_size=64, num_layers=3, output_size=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,
                          dropout=0.0, bidirectional=False)
        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = torch.zeros(self.num_layers, x.shape[0], self.hidden_size,
                                 device=x.device)
        output, hidden = self.gru(x, hidden)
        return self.output_proj(output), hidden


def load_val_data(val_path):
    """Load validation parquet and return features, targets, metadata."""
    df = pd.read_parquet(val_path)
    seq_ix = df['seq_ix'].values
    step_in_seq = df['step_in_seq'].values
    need_pred = df['need_prediction'].values.astype(bool)

    feature_cols = df.columns[3:35]  # 32 raw features
    target_cols = df.columns[35:]    # t0, t1

    features = df[feature_cols].values.astype(np.float32)
    targets = df[target_cols].values.astype(np.float32)

    return features, targets, seq_ix, step_in_seq, need_pred


def run_batch_inference(model, features, seq_ix):
    """Run batch inference over all sequences. Returns predictions for all steps."""
    unique_seqs = np.unique(seq_ix)
    all_preds = np.zeros((len(features), 2), dtype=np.float32)

    with torch.no_grad():
        for s in unique_seqs:
            mask = seq_ix == s
            seq_features = features[mask]  # (1000, 32)
            x = torch.from_numpy(seq_features).unsqueeze(0)  # (1, 1000, 32)
            preds, _ = model(x)  # (1, 1000, 2)
            all_preds[mask] = preds.squeeze(0).numpy()

    return all_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-path', default=str(ROOT / 'datasets' / 'valid.parquet'))
    parser.add_argument('--ckpt-dir', default=str(ROOT / 'logs' / 'vanilla_all'))
    parser.add_argument('--n-models', type=int, default=10,
                        help='Number of top models to use')
    parser.add_argument('--alphas', nargs='+', type=float,
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
    parser.add_argument('--ridge-lambda', type=float, default=1.0,
                        help='Ridge regularization strength')
    parser.add_argument('--output', default=str(ROOT / 'logs' / 'neutralization_beta.npz'))
    args = parser.parse_args()

    # Top seeds by val score (from MEMORY.md)
    top_seeds = [43, 59, 46, 63, 55, 50, 48, 47, 57, 44,
                 49, 56, 52, 51, 45, 53, 54, 60, 58, 42,
                 64, 62, 61]
    seeds_to_use = top_seeds[:args.n_models]

    print(f"=== Prediction Neutralization Beta Computation ===")
    print(f"Using top {args.n_models} seeds: {seeds_to_use}")
    print(f"Ridge lambda: {args.ridge_lambda}")
    print()

    # Load validation data
    print("Loading validation data...")
    t0 = time.time()
    features, targets, seq_ix, step_in_seq, need_pred = load_val_data(args.val_path)
    print(f"  Loaded {len(features)} steps, {len(np.unique(seq_ix))} sequences in {time.time()-t0:.1f}s")

    # Scored mask (need_prediction=True)
    scored_mask = need_pred
    print(f"  Scored steps: {scored_mask.sum()} / {len(scored_mask)}")

    # Run inference for each model
    print(f"\nRunning inference for {args.n_models} models...")
    ensemble_preds = np.zeros((len(features), 2), dtype=np.float32)

    for i, seed in enumerate(seeds_to_use):
        ckpt_path = os.path.join(args.ckpt_dir, f'gru_parity_v1_seed{seed}.pt')
        if not os.path.exists(ckpt_path):
            print(f"  WARNING: {ckpt_path} not found, skipping")
            continue

        t0 = time.time()
        model = VanillaGRU()
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        preds = run_batch_inference(model, features, seq_ix)
        ensemble_preds += preds
        elapsed = time.time() - t0
        print(f"  [{i+1}/{args.n_models}] seed {seed}: {elapsed:.1f}s")

    ensemble_preds /= args.n_models
    print(f"\nEnsemble predictions computed. Shape: {ensemble_preds.shape}")

    # Cache predictions for fast follow-up experiments
    cache_path = str(ROOT / 'logs' / 'neutralization_preds.npz')
    np.savez(cache_path, preds=ensemble_preds[scored_mask],
             features=features[scored_mask], targets=targets[scored_mask])
    print(f"Cached scored predictions to {cache_path}")

    # Score baseline (no neutralization)
    scored_preds = np.clip(ensemble_preds[scored_mask], -6, 6)
    scored_targets = targets[scored_mask]
    scored_features = features[scored_mask]

    base_t0 = weighted_pearson_correlation(scored_targets[:, 0], scored_preds[:, 0])
    base_t1 = weighted_pearson_correlation(scored_targets[:, 1], scored_preds[:, 1])
    base_avg = (base_t0 + base_t1) / 2
    print(f"\nBaseline (alpha=0): t0={base_t0:.4f}, t1={base_t1:.4f}, avg={base_avg:.4f}")

    # Fit ridge regression: beta = (X'X + lambda*I)^-1 X'y
    # X = scored_features (N, 32), y = scored_preds (N, 2)
    print(f"\nFitting ridge regression (lambda={args.ridge_lambda})...")
    X = scored_features  # (N, 32)
    y = scored_preds      # (N, 2)

    XtX = X.T @ X  # (32, 32)
    reg = args.ridge_lambda * np.eye(X.shape[1])
    XtX_inv = np.linalg.inv(XtX + reg)
    beta = XtX_inv @ (X.T @ y)  # (32, 2)

    print(f"  Beta shape: {beta.shape}")
    print(f"  Beta norm (t0): {np.linalg.norm(beta[:, 0]):.4f}")
    print(f"  Beta norm (t1): {np.linalg.norm(beta[:, 1]):.4f}")

    # Feature exposure before neutralization
    exposure_t0 = np.corrcoef(scored_preds[:, 0], X @ beta[:, 0])[0, 1]
    exposure_t1 = np.corrcoef(scored_preds[:, 1], X @ beta[:, 1])[0, 1]
    print(f"  Feature exposure (pred-vs-X@beta correlation): t0={exposure_t0:.4f}, t1={exposure_t1:.4f}")

    # Test different alpha values
    print(f"\n{'Alpha':>6} | {'t0':>7} | {'t1':>7} | {'Avg':>7} | {'Delta':>7}")
    print("-" * 50)

    best_alpha = 0.0
    best_avg = base_avg
    results = []

    for alpha in args.alphas:
        exposure = X @ beta  # (N, 2)
        neutralized = scored_preds - alpha * exposure
        neutralized = np.clip(neutralized, -6, 6)

        t0_score = weighted_pearson_correlation(scored_targets[:, 0], neutralized[:, 0])
        t1_score = weighted_pearson_correlation(scored_targets[:, 1], neutralized[:, 1])
        avg_score = (t0_score + t1_score) / 2
        delta = avg_score - base_avg

        results.append({'alpha': alpha, 't0': t0_score, 't1': t1_score,
                        'avg': avg_score, 'delta': delta})
        print(f"{alpha:6.2f} | {t0_score:7.4f} | {t1_score:7.4f} | {avg_score:7.4f} | {delta:+7.4f}")

        if avg_score > best_avg:
            best_avg = avg_score
            best_alpha = alpha

    print(f"\nBest alpha: {best_alpha} (avg={best_avg:.4f}, delta={best_avg - base_avg:+.4f})")

    # Save beta
    np.savez(args.output,
             beta=beta,
             best_alpha=best_alpha,
             ridge_lambda=args.ridge_lambda,
             seeds_used=np.array(seeds_to_use),
             results=np.array([(r['alpha'], r['t0'], r['t1'], r['avg'], r['delta'])
                               for r in results]))
    print(f"\nSaved beta to {args.output}")

    # Also try per-feature analysis
    print(f"\n=== Top feature exposures (|beta| magnitude) ===")
    feature_names = [f'p{i}' for i in range(12)] + [f'v{i}' for i in range(12)] + \
                    [f'dp{i}' for i in range(4)] + [f'dv{i}' for i in range(4)]
    for target_idx, target_name in enumerate(['t0', 't1']):
        print(f"\n{target_name}:")
        sorted_idx = np.argsort(np.abs(beta[:, target_idx]))[::-1]
        for j in range(min(10, len(sorted_idx))):
            fi = sorted_idx[j]
            print(f"  {feature_names[fi]:>4}: beta={beta[fi, target_idx]:+.6f}")


if __name__ == '__main__':
    main()
