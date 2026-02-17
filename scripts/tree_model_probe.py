#!/usr/bin/env python
"""Tree model blend probe: train snapshot-based tree model, check decorrelation with GRU.

This script trains a HistGradientBoostingRegressor on raw 32 features (per-step snapshot,
no sequential context) and evaluates whether it provides decorrelated signal that could
improve the GRU ensemble via blending.

Phase A: Train tree model, evaluate standalone performance
Phase B: Generate GRU ensemble predictions, check correlation, test blend weights
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import weighted_pearson_correlation


def load_data(split='valid'):
    """Load parquet data, return per-step arrays (features, targets, masks)."""
    path = ROOT / 'datasets' / f'{split}.parquet'
    print(f"Loading {path}...")
    df = pd.read_parquet(path)

    feature_cols = (
        [f'p{i}' for i in range(12)] +
        [f'v{i}' for i in range(12)] +
        [f'dp{i}' for i in range(4)] +
        [f'dv{i}' for i in range(4)]
    )
    target_cols = ['t0', 't1']

    features = df[feature_cols].values.astype(np.float32)
    targets = df[target_cols].values.astype(np.float32)
    mask = df['need_prediction'].values.astype(bool)

    print(f"  Total steps: {len(features):,}, Scored: {mask.sum():,}")
    return features, targets, mask


def train_tree_model(X_train, y_train, target_name='t0'):
    """Train HistGradientBoostingRegressor for one target."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    print(f"\nTraining tree model for {target_name}...")
    print(f"  Train rows: {len(X_train):,}")

    model = HistGradientBoostingRegressor(
        max_iter=500,
        max_depth=6,
        learning_rate=0.05,
        min_samples_leaf=100,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        verbose=0,
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Trained in {elapsed:.1f}s, {model.n_iter_} iterations")
    return model


def evaluate_tree(model, X_val, y_val_target, target_name='t0'):
    """Evaluate tree model with weighted Pearson."""
    preds = model.predict(X_val).astype(np.float32)
    score = weighted_pearson_correlation(y_val_target, preds)
    print(f"  {target_name} weighted Pearson: {score:.4f}")
    return preds, score


def generate_gru_predictions(n_models=10):
    """Generate GRU ensemble predictions on val data."""
    import torch
    from src.models.gru_baseline import GRUBaseline

    # Find top-N parity_v1 checkpoints by val score
    ckpt_dir = ROOT / 'logs' / 'vanilla_all'
    ckpts = []
    for pt in sorted(ckpt_dir.glob('gru_parity_v1_seed*.pt')):
        if '_epoch' in pt.name or '_ckptdiv' in pt.name:
            continue
        state = torch.load(pt, map_location='cpu', weights_only=False)
        score = float(state.get('best_score', 0))
        ckpts.append((pt, score))

    ckpts.sort(key=lambda x: -x[1])
    ckpts = ckpts[:n_models]
    print(f"\nUsing top {len(ckpts)} GRU models:")
    for pt, score in ckpts:
        print(f"  {pt.name}: val={score:.4f}")

    # Load val data as sequences for GRU
    val_path = ROOT / 'datasets' / 'valid.parquet'
    df = pd.read_parquet(val_path)
    feature_cols = [f'p{i}' for i in range(12)] + [f'v{i}' for i in range(12)] + \
                   [f'dp{i}' for i in range(4)] + [f'dv{i}' for i in range(4)]

    seq_ids = np.sort(df['seq_ix'].unique())
    n_seqs = len(seq_ids)
    grouped = df.sort_values('step_in_seq').groupby('seq_ix')

    features_3d = np.empty((n_seqs, 1000, 32), dtype=np.float32)
    targets_3d = np.empty((n_seqs, 1000, 2), dtype=np.float32)
    masks_3d = np.empty((n_seqs, 1000), dtype=bool)
    seq_map = {s: i for i, s in enumerate(seq_ids)}

    for seq_ix, group in grouped:
        idx = seq_map[seq_ix]
        features_3d[idx] = group[feature_cols].values.astype(np.float32)
        targets_3d[idx] = group[['t0', 't1']].values.astype(np.float32)
        masks_3d[idx] = group['need_prediction'].values

    # Run each model
    ensemble_preds = np.zeros((n_seqs, 1000, 2), dtype=np.float32)
    features_t = torch.from_numpy(features_3d)

    for pt, score in ckpts:
        state = torch.load(pt, map_location='cpu', weights_only=False)
        config = state.get('config', {})
        model = GRUBaseline(config)
        model.load_state_dict(state['model_state_dict'], strict=False)
        model.eval()

        print(f"  Running {pt.name}...", end='', flush=True)
        t0 = time.time()
        with torch.no_grad():
            # Process in batches
            batch_size = 64
            for i in range(0, n_seqs, batch_size):
                batch = features_t[i:i+batch_size]
                preds, _ = model(batch)
                ensemble_preds[i:i+batch_size] += preds.numpy()
        elapsed = time.time() - t0
        print(f" {elapsed:.1f}s")

    ensemble_preds /= len(ckpts)

    # Flatten to scored steps only
    scored_mask = masks_3d.flatten()
    flat_preds = ensemble_preds.reshape(-1, 2)[scored_mask]
    flat_targets = targets_3d.reshape(-1, 2)[scored_mask]

    return flat_preds, flat_targets


def main():
    parser = argparse.ArgumentParser(description='Tree model blend probe')
    parser.add_argument('--skip-gru', action='store_true',
                        help='Skip GRU prediction generation (standalone tree eval only)')
    parser.add_argument('--n-models', type=int, default=10,
                        help='Number of GRU models for ensemble (default: 10)')
    args = parser.parse_args()

    # ===== PHASE A: Train tree model =====
    print("=" * 60)
    print("PHASE A: Tree Model Standalone Evaluation")
    print("=" * 60)

    # Load data
    train_feat, train_tgt, train_mask = load_data('train')
    val_feat, val_tgt, val_mask = load_data('valid')

    # Use only scored steps
    X_train = train_feat[train_mask]
    y_train = train_tgt[train_mask]
    X_val = val_feat[val_mask]
    y_val = val_tgt[val_mask]

    print(f"\nScored steps: train={len(X_train):,}, val={len(X_val):,}")

    # Train separate models for t0 and t1
    model_t0 = train_tree_model(X_train, y_train[:, 0], 't0')
    model_t1 = train_tree_model(X_train, y_train[:, 1], 't1')

    # Evaluate
    tree_preds_t0, score_t0 = evaluate_tree(model_t0, X_val, y_val[:, 0], 't0')
    tree_preds_t1, score_t1 = evaluate_tree(model_t1, X_val, y_val[:, 1], 't1')
    tree_avg = (score_t0 + score_t1) / 2

    print(f"\n=== TREE STANDALONE RESULTS ===")
    print(f"  t0: {score_t0:.4f}")
    print(f"  t1: {score_t1:.4f}")
    print(f"  avg: {tree_avg:.4f}")

    if tree_avg < 0.10:
        print(f"\n>>> KILL: tree avg {tree_avg:.4f} < 0.10 threshold. Too weak to blend.")
        return

    if args.skip_gru:
        print("\n(Skipping GRU correlation check, use without --skip-gru for full analysis)")
        return

    # ===== PHASE B: GRU correlation + blend =====
    print("\n" + "=" * 60)
    print("PHASE B: GRU Correlation & Blend Analysis")
    print("=" * 60)

    gru_preds, gru_targets = generate_gru_predictions(args.n_models)

    # Verify targets match
    assert len(gru_preds) == len(X_val), f"Length mismatch: GRU={len(gru_preds)}, tree={len(X_val)}"

    # GRU standalone scores
    gru_t0 = weighted_pearson_correlation(gru_targets[:, 0], gru_preds[:, 0])
    gru_t1 = weighted_pearson_correlation(gru_targets[:, 1], gru_preds[:, 1])
    gru_avg = (gru_t0 + gru_t1) / 2
    print(f"\nGRU ensemble (val): t0={gru_t0:.4f}, t1={gru_t1:.4f}, avg={gru_avg:.4f}")

    # Correlation between tree and GRU predictions
    tree_preds_2d = np.column_stack([tree_preds_t0, tree_preds_t1])
    corr_t0 = np.corrcoef(gru_preds[:, 0], tree_preds_t0)[0, 1]
    corr_t1 = np.corrcoef(gru_preds[:, 1], tree_preds_t1)[0, 1]
    print(f"\nTree-GRU correlation: t0={corr_t0:.4f}, t1={corr_t1:.4f}")

    if corr_t0 > 0.85 and corr_t1 > 0.85:
        print(f"\n>>> KILL: correlation too high (>{0.85}). No diversity value.")
        return

    # Blend sweep
    print(f"\n=== BLEND SWEEP ===")
    print(f"{'Alpha':>6} | {'t0':>8} | {'t1':>8} | {'avg':>8} | {'delta':>8}")
    print("-" * 50)

    best_alpha = 0.0
    best_blend_avg = gru_avg

    for alpha in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        blend_t0 = (1 - alpha) * gru_preds[:, 0] + alpha * tree_preds_t0
        blend_t1 = (1 - alpha) * gru_preds[:, 1] + alpha * tree_preds_t1

        b_t0 = weighted_pearson_correlation(gru_targets[:, 0], blend_t0)
        b_t1 = weighted_pearson_correlation(gru_targets[:, 1], blend_t1)
        b_avg = (b_t0 + b_t1) / 2
        delta = b_avg - gru_avg

        marker = " <-- BEST" if b_avg > best_blend_avg else ""
        print(f"{alpha:>6.2f} | {b_t0:>8.4f} | {b_t1:>8.4f} | {b_avg:>8.4f} | {delta:>+8.4f}{marker}")

        if b_avg > best_blend_avg:
            best_blend_avg = b_avg
            best_alpha = alpha

    print(f"\n=== VERDICT ===")
    if best_alpha == 0.0:
        print(f">>> KILL: No blend alpha improves over GRU-only ({gru_avg:.4f})")
    else:
        delta = best_blend_avg - gru_avg
        print(f">>> PROMISING: best alpha={best_alpha:.2f}, "
              f"blend avg={best_blend_avg:.4f} (delta={delta:+.4f})")
        print(f"    Tree-GRU correlation: t0={corr_t0:.4f}, t1={corr_t1:.4f}")
        print(f"    Next: export tree to ONNX, build submission, verify")


if __name__ == '__main__':
    main()
