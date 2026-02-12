#!/usr/bin/env python
"""Evaluate GRU checkpoints with windowed inference (like the official baseline).

Compares two inference modes on validation data:
  1. Step-by-step: stateful hidden state across all 1000 steps
  2. Windowed: feed last W steps as context, fresh hidden each prediction

Usage:
    python scripts/eval_windowed.py \
        --checkpoint logs/gru_baseline_match_v1_seed42.pt \
        --normalizer logs/normalizer_gru_baseline_match_v1_seed42.npz \
        --window 100
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.models.gru_baseline import GRUBaseline
from utils import weighted_pearson_correlation


def load_val_data(val_path: str):
    """Load validation data as numpy arrays."""
    df = pd.read_parquet(val_path)
    feature_cols = (
        [f'p{i}' for i in range(12)] +
        [f'v{i}' for i in range(12)] +
        [f'dp{i}' for i in range(4)] +
        [f'dv{i}' for i in range(4)]
    )
    seq_ids = np.sort(df['seq_ix'].unique())
    n_seqs = len(seq_ids)

    features = np.empty((n_seqs, 1000, 32), dtype=np.float32)
    targets = np.empty((n_seqs, 1000, 2), dtype=np.float32)
    masks = np.empty((n_seqs, 1000), dtype=np.bool_)

    grouped = df.sort_values('step_in_seq').groupby('seq_ix')
    seq_map = {s: i for i, s in enumerate(seq_ids)}

    for seq_ix, group in grouped:
        idx = seq_map[seq_ix]
        features[idx] = group[feature_cols].values.astype(np.float32)
        targets[idx] = group[['t0', 't1']].values.astype(np.float32)
        masks[idx] = group['need_prediction'].values

    return features, targets, masks, n_seqs


def eval_step_by_step(model, features, targets, masks, normalizer, n_seqs):
    """Standard step-by-step inference with persistent hidden state."""
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i in range(n_seqs):
            hidden = None
            for t in range(1000):
                x = features[i, t].copy()
                x = normalizer.transform(x.reshape(1, -1)).flatten()
                x_t = torch.from_numpy(x).unsqueeze(0)
                pred, hidden = model.forward_step(x_t, hidden)
                if masks[i, t]:
                    all_preds.append(pred.squeeze(0).numpy())
                    all_targets.append(targets[i, t])

    all_preds = np.clip(np.array(all_preds), -6, 6)
    all_targets = np.array(all_targets)
    return all_preds, all_targets


def eval_windowed(model, features, targets, masks, normalizer, n_seqs, window_size):
    """Windowed inference: feed last W steps for each prediction."""
    all_preds = []
    all_targets = []

    # Pre-normalize all features
    flat = features.reshape(-1, 32)
    flat_norm = normalizer.transform(flat).reshape(n_seqs, 1000, 32)

    with torch.no_grad():
        for i in range(n_seqs):
            for t in range(1000):
                if not masks[i, t]:
                    continue

                # Build context window: last W steps up to and including t
                start = max(0, t + 1 - window_size)
                window = flat_norm[i, start:t+1]  # shape (<=W, 32)

                # Pad if needed
                if window.shape[0] < window_size:
                    pad = np.zeros((window_size - window.shape[0], 32), dtype=np.float32)
                    window = np.concatenate([pad, window], axis=0)

                # Forward pass: (1, W, 32) -> take last timestep output
                x_t = torch.from_numpy(window).unsqueeze(0)  # (1, W, 32)
                preds, _ = model(x_t)  # (1, W, 2)
                pred = preds[0, -1].numpy()  # last timestep

                all_preds.append(pred)
                all_targets.append(targets[i, t])

    all_preds = np.clip(np.array(all_preds), -6, 6)
    all_targets = np.array(all_targets)
    return all_preds, all_targets


def eval_windowed_batched(model, features, targets, masks, normalizer, n_seqs, window_size):
    """Windowed inference, batched per sequence for speed."""
    all_preds = []
    all_targets = []

    # Pre-normalize all features
    flat = features.reshape(-1, 32)
    flat_norm = normalizer.transform(flat).reshape(n_seqs, 1000, 32)

    with torch.no_grad():
        for i in range(n_seqs):
            pred_steps = np.where(masks[i])[0]
            if len(pred_steps) == 0:
                continue

            # Build all windows for this sequence at once
            windows = []
            for t in pred_steps:
                start = max(0, t + 1 - window_size)
                w = flat_norm[i, start:t+1]
                if w.shape[0] < window_size:
                    pad = np.zeros((window_size - w.shape[0], 32), dtype=np.float32)
                    w = np.concatenate([pad, w], axis=0)
                windows.append(w)

            # Stack into batch: (N_pred, W, 32)
            batch = torch.from_numpy(np.stack(windows))
            preds, _ = model(batch)  # (N_pred, W, 2)
            preds_last = preds[:, -1].numpy()  # (N_pred, 2)

            all_preds.append(preds_last)
            all_targets.append(targets[i, pred_steps])

    all_preds = np.clip(np.concatenate(all_preds), -6, 6)
    all_targets = np.concatenate(all_targets)
    return all_preds, all_targets


def compute_scores(preds, tgts):
    """Compute per-target and average weighted Pearson."""
    t0 = weighted_pearson_correlation(tgts[:, 0], preds[:, 0])
    t1 = weighted_pearson_correlation(tgts[:, 1], preds[:, 1])
    avg = (t0 + t1) / 2
    return t0, t1, avg


def main():
    parser = argparse.ArgumentParser(description='Windowed inference evaluation')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--normalizer', type=str, required=True)
    parser.add_argument('--val-path', type=str, default='datasets/valid.parquet')
    parser.add_argument('--window', type=int, default=100)
    parser.add_argument('--skip-stepwise', action='store_true',
                        help='Skip step-by-step eval (slow)')
    args = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = ckpt['config']
    model = GRUBaseline(config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Model: {os.path.basename(args.checkpoint)}")
    print(f"  Params: {model.count_parameters():,}")
    print(f"  Config: h={config['model']['hidden_size']}, "
          f"layers={config['model']['num_layers']}, "
          f"output={config['model'].get('output_type', 'mlp')}")
    print(f"  Saved val score: {ckpt.get('best_score', 'N/A')}")

    # Load normalizer
    from src.data.preprocessing import Normalizer
    normalizer = Normalizer.load(args.normalizer)
    print(f"Normalizer: {args.normalizer} (dim={normalizer.mean.shape[0]})")

    # Load validation data
    print(f"\nLoading validation data from {args.val_path}...")
    features, targets, masks, n_seqs = load_val_data(args.val_path)
    n_scored = masks.sum()
    print(f"  {n_seqs} sequences, {n_scored:,} scored steps")

    # Windowed inference
    print(f"\n--- Windowed inference (window={args.window}) ---")
    t0_start = time.time()
    w_preds, w_tgts = eval_windowed_batched(
        model, features, targets, masks, normalizer, n_seqs, args.window
    )
    w_time = time.time() - t0_start
    w_t0, w_t1, w_avg = compute_scores(w_preds, w_tgts)
    print(f"  t0={w_t0:.4f}  t1={w_t1:.4f}  avg={w_avg:.4f}  ({w_time:.1f}s)")

    # Step-by-step inference
    if not args.skip_stepwise:
        print(f"\n--- Step-by-step inference ---")
        t0_start = time.time()
        s_preds, s_tgts = eval_step_by_step(
            model, features, targets, masks, normalizer, n_seqs
        )
        s_time = time.time() - t0_start
        s_t0, s_t1, s_avg = compute_scores(s_preds, s_tgts)
        print(f"  t0={s_t0:.4f}  t1={s_t1:.4f}  avg={s_avg:.4f}  ({s_time:.1f}s)")

        print(f"\n--- Comparison ---")
        print(f"  Windowed: {w_avg:.4f}")
        print(f"  Step-by-step: {s_avg:.4f}")
        print(f"  Delta (window - step): {w_avg - s_avg:+.4f}")

    # Test multiple window sizes
    print(f"\n--- Window size sweep ---")
    for ws in [50, 75, 100, 150, 200, 500, 1000]:
        wp, wt = eval_windowed_batched(
            model, features, targets, masks, normalizer, n_seqs, ws
        )
        _, _, wa = compute_scores(wp, wt)
        print(f"  window={ws:4d}: avg={wa:.4f}")


if __name__ == '__main__':
    main()
