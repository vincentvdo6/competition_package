#!/usr/bin/env python
"""Fast batch-mode evaluation of vanilla GRU checkpoints on validation.

Supports single models and model soups (weight-averaged checkpoints).
Uses batch forward pass (not step-by-step) for speed.

Usage:
    # Evaluate single model
    python scripts/eval_soup.py --checkpoint logs/vanilla_all/gru_parity_v1_seed43.pt

    # Create and evaluate soup from multiple checkpoints
    python scripts/eval_soup.py --checkpoints logs/vanilla_all/gru_parity_v1_seed43.pt \
        logs/vanilla_all/gru_parity_v1_seed59.pt logs/vanilla_all/gru_parity_v1_seed46.pt \
        --soup-output logs/soup_top3.pt

    # Evaluate ensemble (prediction averaging) from multiple checkpoints
    python scripts/eval_soup.py --checkpoints ckpt1.pt ckpt2.pt ... --ensemble
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

from utils import weighted_pearson_correlation
from src.models.gru_baseline import GRUBaseline


def load_val_data(path="datasets/valid.parquet"):
    """Load validation data as (features, targets, mask) arrays."""
    df = pd.read_parquet(path)
    feat_cols = [f'p{i}' for i in range(12)] + [f'v{i}' for i in range(12)] + \
                [f'dp{i}' for i in range(4)] + [f'dv{i}' for i in range(4)]
    target_cols = ['t0', 't1']

    seq_ids = df['seq_ix'].unique()
    n_seqs = len(seq_ids)
    seq_len = 1000
    n_feat = len(feat_cols)

    features = np.zeros((n_seqs, seq_len, n_feat), dtype=np.float32)
    targets = np.zeros((n_seqs, seq_len, 2), dtype=np.float32)
    mask = np.zeros((n_seqs, seq_len), dtype=bool)

    for i, sid in enumerate(sorted(seq_ids)):
        seq = df[df['seq_ix'] == sid].sort_values('step_in_seq')
        steps = seq['step_in_seq'].values
        features[i, steps] = seq[feat_cols].values.astype(np.float32)
        targets[i, steps] = seq[target_cols].values.astype(np.float32)
        mask[i, steps] = seq['need_prediction'].values.astype(bool)

    return features, targets, mask


def load_model(checkpoint_path, config=None):
    """Load vanilla GRU model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if config is None:
        config = ckpt.get('config', {})
    if not config:
        # Default vanilla config
        config = {
            'model': {
                'type': 'gru', 'input_size': 32, 'hidden_size': 64,
                'num_layers': 3, 'dropout': 0.0, 'output_size': 2,
                'vanilla': True, 'output_type': 'linear'
            }
        }
    model = GRUBaseline(config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def average_state_dicts(checkpoint_paths):
    """Average model weights from multiple checkpoints."""
    states = []
    for p in checkpoint_paths:
        ckpt = torch.load(p, map_location='cpu', weights_only=False)
        states.append(ckpt['model_state_dict'])

    avg = {}
    for key in states[0]:
        avg[key] = sum(s[key].float() for s in states) / len(states)
    return avg


def evaluate_model(model, features, targets, mask, batch_size=256):
    """Run batch-mode forward pass and compute weighted Pearson."""
    n_seqs = features.shape[0]
    all_preds = []

    with torch.no_grad():
        for start in range(0, n_seqs, batch_size):
            end = min(start + batch_size, n_seqs)
            x = torch.from_numpy(features[start:end])
            pred, _ = model(x, hidden=None)
            all_preds.append(pred.numpy())

    preds = np.concatenate(all_preds, axis=0)  # (n_seqs, 1000, 2)

    # Extract scored predictions (where mask is True)
    scored_preds = preds[mask]  # (N_scored, 2)
    scored_targets = targets[mask]  # (N_scored, 2)

    # Compute per-target weighted Pearson
    t0_score = weighted_pearson_correlation(scored_targets[:, 0], scored_preds[:, 0])
    t1_score = weighted_pearson_correlation(scored_targets[:, 1], scored_preds[:, 1])
    avg_score = (t0_score + t1_score) / 2

    return {'avg': avg_score, 't0': t0_score, 't1': t1_score}


def main():
    parser = argparse.ArgumentParser(description='Fast vanilla GRU evaluation')
    parser.add_argument('--checkpoint', type=str, help='Single checkpoint to evaluate')
    parser.add_argument('--checkpoints', nargs='+', help='Multiple checkpoints for soup or ensemble')
    parser.add_argument('--soup-output', type=str, help='Save weight-averaged soup to this path')
    parser.add_argument('--ensemble', action='store_true', help='Prediction averaging instead of weight averaging')
    parser.add_argument('--data', type=str, default='datasets/valid.parquet')
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()

    print("Loading validation data...")
    t0 = time.time()
    features, targets, mask = load_val_data(args.data)
    print(f"  {features.shape[0]} sequences, {mask.sum()} scored steps ({time.time()-t0:.1f}s)")

    if args.checkpoint:
        # Single model eval
        print(f"\nEvaluating: {os.path.basename(args.checkpoint)}")
        model = load_model(args.checkpoint)
        scores = evaluate_model(model, features, targets, mask, args.batch_size)
        print(f"  avg={scores['avg']:.4f}  t0={scores['t0']:.4f}  t1={scores['t1']:.4f}")

    elif args.checkpoints and args.ensemble:
        # Prediction ensemble (average predictions, not weights)
        print(f"\nPrediction ensemble ({len(args.checkpoints)} models):")
        all_preds = []
        for cp in args.checkpoints:
            model = load_model(cp)
            n_seqs = features.shape[0]
            preds = []
            with torch.no_grad():
                for start in range(0, n_seqs, args.batch_size):
                    end = min(start + args.batch_size, n_seqs)
                    x = torch.from_numpy(features[start:end])
                    pred, _ = model(x, hidden=None)
                    preds.append(pred.numpy())
            all_preds.append(np.concatenate(preds, axis=0))
            ckpt = torch.load(cp, map_location='cpu', weights_only=False)
            val = ckpt.get('best_score', 0)
            print(f"  {os.path.basename(cp)}: val={val:.4f}")

        ens_preds = np.mean(all_preds, axis=0)
        scored_preds = ens_preds[mask]
        scored_targets = targets[mask]
        t0s = weighted_pearson_correlation(scored_targets[:, 0], scored_preds[:, 0])
        t1s = weighted_pearson_correlation(scored_targets[:, 1], scored_preds[:, 1])
        avg = (t0s + t1s) / 2
        print(f"\n  ENSEMBLE: avg={avg:.4f}  t0={t0s:.4f}  t1={t1s:.4f}")

    elif args.checkpoints:
        # Model soup (weight averaging)
        n = len(args.checkpoints)
        print(f"\nModel soup ({n} models):")
        for cp in args.checkpoints:
            ckpt = torch.load(cp, map_location='cpu', weights_only=False)
            val = ckpt.get('best_score', 0)
            print(f"  {os.path.basename(cp)}: val={val:.4f}")

        avg_state = average_state_dicts(args.checkpoints)

        # Build model with averaged weights
        ckpt0 = torch.load(args.checkpoints[0], map_location='cpu', weights_only=False)
        config = ckpt0.get('config', {})
        model = GRUBaseline(config) if config else load_model(args.checkpoints[0])
        if config:
            model.load_state_dict(avg_state)
        else:
            model.load_state_dict(avg_state)
        model.eval()

        scores = evaluate_model(model, features, targets, mask, args.batch_size)
        print(f"\n  SOUP: avg={scores['avg']:.4f}  t0={scores['t0']:.4f}  t1={scores['t1']:.4f}")

        if args.soup_output:
            torch.save({
                'model_state_dict': avg_state,
                'config': config,
                'best_score': scores['avg'],
                'soup_from': [os.path.basename(p) for p in args.checkpoints],
            }, args.soup_output)
            print(f"  Saved soup to {args.soup_output}")

    else:
        parser.error("Provide --checkpoint or --checkpoints")


if __name__ == '__main__':
    main()
