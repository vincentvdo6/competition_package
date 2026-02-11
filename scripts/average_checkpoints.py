"""Average multiple model checkpoints for improved generalization.

Usage:
    python scripts/average_checkpoints.py --dir logs/ --prefix gru_large_s42 --last 5 --output averaged.pt
    python scripts/average_checkpoints.py --checkpoints ckpt1.pt ckpt2.pt ckpt3.pt --output averaged.pt
"""

import argparse
import glob
import os
import re
import torch


def average_checkpoints(checkpoint_paths: list, output_path: str):
    """Average model_state_dict across multiple checkpoints."""
    assert len(checkpoint_paths) >= 2, f"Need at least 2 checkpoints, got {len(checkpoint_paths)}"

    print(f"Averaging {len(checkpoint_paths)} checkpoints:")
    for p in checkpoint_paths:
        print(f"  {p}")

    # Load first checkpoint as base
    avg_state = {}
    first = torch.load(checkpoint_paths[0], map_location='cpu', weights_only=False)
    base_state = first['model_state_dict']
    config = first.get('config', None)
    seed = first.get('seed', None)

    for key, val in base_state.items():
        avg_state[key] = val.float().clone()

    # Accumulate remaining checkpoints
    for path in checkpoint_paths[1:]:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        state = ckpt['model_state_dict']
        for key in avg_state:
            avg_state[key] += state[key].float()

    # Divide by count
    n = len(checkpoint_paths)
    for key in avg_state:
        avg_state[key] /= n

    # Save averaged checkpoint
    torch.save({
        'model_state_dict': avg_state,
        'config': config,
        'seed': seed,
        'averaged_from': [os.path.basename(p) for p in checkpoint_paths],
        'num_averaged': n,
    }, output_path)

    print(f"Saved averaged checkpoint to {output_path}")


def find_epoch_checkpoints(log_dir: str, prefix: str, last_k: int) -> list:
    """Find the last K epoch checkpoints matching a prefix."""
    pattern = os.path.join(log_dir, f"{prefix}_epoch*.pt")
    paths = sorted(glob.glob(pattern))

    # Extract epoch numbers and sort
    epoch_paths = []
    for p in paths:
        match = re.search(r'_epoch(\d+)\.pt$', p)
        if match:
            epoch_paths.append((int(match.group(1)), p))

    epoch_paths.sort(key=lambda x: x[0])

    if last_k > 0:
        epoch_paths = epoch_paths[-last_k:]

    return [p for _, p in epoch_paths]


def main():
    parser = argparse.ArgumentParser(description='Average model checkpoints')
    parser.add_argument('--checkpoints', nargs='+', help='Explicit checkpoint paths')
    parser.add_argument('--dir', type=str, help='Directory to search for epoch checkpoints')
    parser.add_argument('--prefix', type=str, help='Checkpoint prefix (e.g., gru_large_s42)')
    parser.add_argument('--last', type=int, default=5, help='Average last K checkpoints')
    parser.add_argument('--output', type=str, required=True, help='Output path for averaged checkpoint')
    args = parser.parse_args()

    if args.checkpoints:
        paths = args.checkpoints
    elif args.dir and args.prefix:
        paths = find_epoch_checkpoints(args.dir, args.prefix, args.last)
    else:
        parser.error("Either --checkpoints or (--dir + --prefix) required")

    if len(paths) < 2:
        print(f"Only found {len(paths)} checkpoints, need at least 2. Skipping averaging.")
        return

    average_checkpoints(paths, args.output)


if __name__ == '__main__':
    main()
