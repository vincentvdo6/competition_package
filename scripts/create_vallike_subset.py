#!/usr/bin/env python
"""Create val-like subset of training data based on adversarial validation weights.

Uses adversarial_weights.npz (from adversarial_validation.py) to identify
the top N% of training sequences that most resemble the validation distribution.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top-pct', type=float, default=30.0,
                        help='Percentage of val-like sequences to keep (default: 30)')
    parser.add_argument('--weights-file', type=str,
                        default=str(ROOT / 'logs' / 'adversarial_weights.npz'))
    parser.add_argument('--train-parquet', type=str,
                        default=str(ROOT / 'datasets' / 'train.parquet'))
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: datasets/train_vallike{pct}.parquet)')
    args = parser.parse_args()

    # Load adversarial weights
    data = np.load(args.weights_file)
    seq_ids = data['seq_ids']
    weights = data['weights']
    print(f"Loaded {len(seq_ids)} sequence weights from {args.weights_file}")
    print(f"  Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
    print(f"  Mean: {weights.mean():.3f}, Median: {np.median(weights):.3f}")

    # Select top N% by weight (highest weight = most val-like)
    n_keep = int(len(seq_ids) * args.top_pct / 100)
    top_indices = np.argsort(weights)[-n_keep:]
    selected_ids = set(seq_ids[top_indices])
    selected_weights = weights[top_indices]

    print(f"\nSelected top {args.top_pct:.0f}%: {n_keep} sequences")
    print(f"  Weight range: [{selected_weights.min():.3f}, {selected_weights.max():.3f}]")
    print(f"  Mean: {selected_weights.mean():.3f}")

    # Filter train parquet
    print(f"\nLoading {args.train_parquet}...")
    df = pd.read_parquet(args.train_parquet)
    original_seqs = df['seq_ix'].nunique()
    original_rows = len(df)

    df_filtered = df[df['seq_ix'].isin(selected_ids)]
    filtered_seqs = df_filtered['seq_ix'].nunique()
    filtered_rows = len(df_filtered)

    print(f"  Original: {original_seqs} sequences, {original_rows:,} rows")
    print(f"  Filtered: {filtered_seqs} sequences, {filtered_rows:,} rows")

    # Save
    if args.output is None:
        pct_str = f"{int(args.top_pct)}"
        output_path = ROOT / 'datasets' / f'train_vallike{pct_str}.parquet'
    else:
        output_path = Path(args.output)

    df_filtered.to_parquet(output_path, index=False)
    size_mb = output_path.stat().st_size / 1e6
    print(f"\nSaved to {output_path} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
