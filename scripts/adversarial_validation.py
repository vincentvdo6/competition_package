#!/usr/bin/env python
"""Adversarial validation: detect train/val distribution shift.

If AUC > 0.60, meaningful shift exists. Compute density-ratio weights
that could be used to retrain GRU with emphasis on test-like sequences.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def compute_sequence_summaries(df, feature_cols):
    """Compute per-sequence summary statistics (mean, std, min, max per feature)."""
    grouped = df.groupby('seq_ix')[feature_cols]
    means = grouped.mean().values
    stds = grouped.std().values
    mins = grouped.min().values
    maxs = grouped.max().values
    seq_ids = grouped.mean().index.values
    summaries = np.concatenate([means, stds, mins, maxs], axis=1)
    return seq_ids, summaries


def main():
    feature_cols = (
        [f'p{i}' for i in range(12)] +
        [f'v{i}' for i in range(12)] +
        [f'dp{i}' for i in range(4)] +
        [f'dv{i}' for i in range(4)]
    )

    # Load data
    print("Loading train data...")
    train_df = pd.read_parquet(ROOT / 'datasets' / 'train.parquet')
    print("Loading val data...")
    val_df = pd.read_parquet(ROOT / 'datasets' / 'valid.parquet')

    # Compute per-sequence summaries
    print("Computing sequence summaries...")
    train_ids, train_summ = compute_sequence_summaries(train_df, feature_cols)
    val_ids, val_summ = compute_sequence_summaries(val_df, feature_cols)

    print(f"  Train sequences: {len(train_ids)}")
    print(f"  Val sequences:   {len(val_ids)}")
    print(f"  Features per seq: {train_summ.shape[1]} (32 x 4 summary stats)")

    # Create binary classification: train=0, val=1
    X = np.vstack([train_summ, val_summ])
    y = np.concatenate([np.zeros(len(train_summ)), np.ones(len(val_summ))])

    # Train adversarial classifier
    from sklearn.model_selection import cross_val_predict
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    # Standardize for logistic regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Method 1: Logistic Regression
    print("\n=== Logistic Regression ===")
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr_probs = cross_val_predict(lr, X_scaled, y, cv=5, method='predict_proba')[:, 1]
    lr_auc = roc_auc_score(y, lr_probs)
    print(f"  AUC-ROC: {lr_auc:.4f}")

    # Method 2: HistGradientBoosting
    print("\n=== HistGradientBoosting ===")
    hgb = HistGradientBoostingClassifier(
        max_iter=200, max_depth=4, learning_rate=0.05,
        min_samples_leaf=20, random_state=42,
    )
    hgb_probs = cross_val_predict(hgb, X, y, cv=5, method='predict_proba')[:, 1]
    hgb_auc = roc_auc_score(y, hgb_probs)
    print(f"  AUC-ROC: {hgb_auc:.4f}")

    best_auc = max(lr_auc, hgb_auc)
    best_method = "LogReg" if lr_auc > hgb_auc else "HGB"
    best_probs = lr_probs if lr_auc > hgb_auc else hgb_probs

    print(f"\n=== VERDICT ===")
    print(f"Best AUC: {best_auc:.4f} ({best_method})")

    if best_auc < 0.55:
        print(f">>> KILL: AUC {best_auc:.4f} < 0.55. No meaningful train/val shift.")
        return

    if best_auc < 0.60:
        print(f">>> MARGINAL: AUC {best_auc:.4f}. Weak shift signal.")
    else:
        print(f">>> SHIFT DETECTED: AUC {best_auc:.4f} > 0.60.")

    # Compute density-ratio weights for train sequences
    train_probs = best_probs[:len(train_summ)]
    # w(x) = p(val|x) / (1 - p(val|x)) = odds ratio
    eps = 1e-6
    weights = np.clip(train_probs / (1 - train_probs + eps), 0.2, 5.0)

    print(f"\n=== TRAIN SEQUENCE WEIGHTS ===")
    print(f"  Mean:   {weights.mean():.3f}")
    print(f"  Median: {np.median(weights):.3f}")
    print(f"  Std:    {weights.std():.3f}")
    print(f"  Min:    {weights.min():.3f}")
    print(f"  Max:    {weights.max():.3f}")
    print(f"  >2.0:   {(weights > 2.0).sum()} sequences ({(weights > 2.0).mean()*100:.1f}%)")
    print(f"  <0.5:   {(weights < 0.5).sum()} sequences ({(weights < 0.5).mean()*100:.1f}%)")

    # Save weights for potential GRU retraining
    out_path = ROOT / 'logs' / 'adversarial_weights.npz'
    np.savez(out_path, seq_ids=train_ids, weights=weights, auc=best_auc)
    print(f"\nSaved weights to {out_path}")

    # Feature importance (which features distinguish train from val?)
    if best_method == "HGB":
        # Refit on full data for feature importance
        hgb_full = HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=20, random_state=42,
        )
        hgb_full.fit(X, y)

        # HistGB doesn't have feature_importances_ directly; use permutation importance
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(hgb_full, X, y, n_repeats=5, random_state=42, n_jobs=-1)

        summary_names = []
        for stat in ['mean', 'std', 'min', 'max']:
            for f in feature_cols:
                summary_names.append(f"{f}_{stat}")

        imp = list(zip(summary_names, perm.importances_mean))
        imp.sort(key=lambda x: -x[1])
        print(f"\n=== TOP 15 DISCRIMINATING FEATURES ===")
        for name, importance in imp[:15]:
            print(f"  {name:<20s} {importance:.4f}")


if __name__ == '__main__':
    main()
