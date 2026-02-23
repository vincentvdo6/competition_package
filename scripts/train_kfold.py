#!/usr/bin/env python
"""K-fold training script for Wunderfund Predictorium.

Merges train.parquet + valid.parquet, splits by seq_ix into K folds.
Trains on K-1 folds, validates on the held-out fold.
Each fold model sees val-like data during training, addressing distribution shift.

Usage:
    python scripts/train_kfold.py --config configs/gru_kfold_v1.yaml --fold 0 --seed 42 --device cuda
    python scripts/train_kfold.py --config configs/gru_kfold_v1.yaml --fold 0 --seed 42 --eval-original-val
"""

import argparse
import random
import yaml
import torch
import numpy as np
import pandas as pd
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.dataset import create_dataloaders
from src.models.gru_baseline import GRUBaseline
from src.training.trainer import Trainer, setup_cpu_performance
from src.training.losses import get_loss_function


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_and_split(train_path, valid_path, n_folds, fold_idx):
    """Merge train+val, assign folds by seq_ix, return (train_df, val_df)."""
    print(f"Loading {train_path}...")
    train_df = pd.read_parquet(train_path)
    print(f"Loading {valid_path}...")
    valid_df = pd.read_parquet(valid_path)

    # Offset valid seq_ix to avoid collisions
    offset = int(train_df['seq_ix'].max()) + 1
    valid_df = valid_df.copy()
    valid_df['seq_ix'] = valid_df['seq_ix'] + offset

    combined = pd.concat([train_df, valid_df], ignore_index=True)
    del train_df, valid_df

    # Get unique seq_ix values and assign folds deterministically
    all_seqs = np.sort(combined['seq_ix'].unique())
    n_total = len(all_seqs)
    fold_assignments = np.arange(n_total) % n_folds

    val_seqs = set(all_seqs[fold_assignments == fold_idx])
    train_seqs = set(all_seqs[fold_assignments != fold_idx])

    val_df = combined[combined['seq_ix'].isin(val_seqs)].copy()
    train_df = combined[combined['seq_ix'].isin(train_seqs)].copy()
    del combined

    # Renumber seq_ix to be contiguous (required by dataset)
    train_seq_map = {s: i for i, s in enumerate(sorted(train_seqs))}
    val_seq_map = {s: i for i, s in enumerate(sorted(val_seqs))}
    train_df['seq_ix'] = train_df['seq_ix'].map(train_seq_map)
    val_df['seq_ix'] = val_df['seq_ix'].map(val_seq_map)

    print(f"K-fold split: {n_folds} folds, fold {fold_idx} as val")
    print(f"  Train: {len(train_seqs)} sequences ({len(train_df)} rows)")
    print(f"  Val:   {len(val_seqs)} sequences ({len(val_df)} rows)")

    return train_df, val_df


def main():
    parser = argparse.ArgumentParser(description='K-fold training for Wunderfund Predictorium')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fold', type=int, required=True, help='Which fold to hold out (0 to n_folds-1)')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of folds (default: 5)')
    parser.add_argument('--eval-original-val', action='store_true',
                        help='Also evaluate on original valid.parquet after training')
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Random seed: {args.seed}")

    config = load_config(args.config)
    config['seed'] = args.seed
    print(f"Loaded config from {args.config}")

    # Run name includes fold
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    run_name = f"{config_name}_fold{args.fold}_seed{args.seed}"
    config.setdefault('logging', {})['checkpoint_prefix'] = run_name

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    if device.type == 'cpu':
        setup_cpu_performance()
        print(f"CPU threads: {torch.get_num_threads()}")

    # Data config
    data_cfg = config.get('data', {})
    train_path = data_cfg.get('train_path', 'datasets/train.parquet')
    valid_path = data_cfg.get('valid_path', 'datasets/valid.parquet')
    batch_size = int(config.get('training', {}).get('batch_size', 192))

    feature_indices_raw = data_cfg.get('feature_indices', None)
    feature_indices = [int(x) for x in feature_indices_raw] if feature_indices_raw else None

    # Merge and split
    train_df, val_df = merge_and_split(train_path, valid_path, args.n_folds, args.fold)

    # Write to temp parquets for LOBSequenceDataset
    log_dir = config.get('logging', {}).get('log_dir', 'logs')
    os.makedirs(log_dir, exist_ok=True)

    tmp_train = tempfile.NamedTemporaryFile(suffix='.parquet', delete=False, dir=log_dir)
    tmp_val = tempfile.NamedTemporaryFile(suffix='.parquet', delete=False, dir=log_dir)
    tmp_train_path = tmp_train.name
    tmp_val_path = tmp_val.name
    tmp_train.close()
    tmp_val.close()

    train_df.to_parquet(tmp_train_path)
    val_df.to_parquet(tmp_val_path)
    del train_df, val_df
    print(f"Temp parquets: {tmp_train_path}, {tmp_val_path}")

    # Create dataloaders using the standard pipeline
    use_gpu = device.type == 'cuda'
    train_loader, valid_loader, normalizer = create_dataloaders(
        train_path=tmp_train_path,
        valid_path=tmp_val_path,
        batch_size=batch_size,
        normalize=data_cfg.get('normalize', False),
        pin_memory=use_gpu,
        derived_features=data_cfg.get('derived_features', False),
        feature_indices=feature_indices,
    )
    print(f"Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")

    # Clean up temp files
    try:
        os.unlink(tmp_train_path)
        os.unlink(tmp_val_path)
    except OSError:
        pass

    # Save normalizer if used
    if normalizer is not None:
        normalizer_path = os.path.join(log_dir, f"normalizer_{run_name}.npz")
        normalizer.save(normalizer_path)
        config.setdefault('logging', {})['normalizer_path'] = os.path.basename(normalizer_path)
        print(f"Saved normalizer to {normalizer_path}")

    # Create model (always vanilla GRU for K-fold)
    model = GRUBaseline(config)
    param_count = model.count_parameters()
    print(f"Created GRU model with {param_count:,} parameters")

    # Loss
    loss_fn = get_loss_function(config)
    print(f"Using loss: {type(loss_fn).__name__}")

    # Train
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        valid_loader=valid_loader,
        loss_fn=loss_fn,
        device=device,
    )

    print(f"\n{'='*60}")
    print(f"K-FOLD TRAINING: fold {args.fold}/{args.n_folds}, seed {args.seed}")
    print(f"{'='*60}")
    history = trainer.train()
    print(f"{'='*60}")
    print(f"Fold {args.fold} complete. Best val score: {trainer.best_score:.4f} (epoch {trainer.best_epoch})")

    # Optional: evaluate on original valid.parquet
    if args.eval_original_val:
        print(f"\n{'='*60}")
        print(f"Evaluating on original valid.parquet...")
        from src.data.dataset import LOBSequenceDataset
        from utils import weighted_pearson_correlation

        orig_val = LOBSequenceDataset(
            valid_path,
            normalize=data_cfg.get('normalize', False),
            normalizer=normalizer,
            derived_features=data_cfg.get('derived_features', False),
            feature_indices=feature_indices,
        )
        orig_loader = torch.utils.data.DataLoader(
            orig_val, batch_size=batch_size, shuffle=False, pin_memory=use_gpu,
        )

        # Load best checkpoint
        ckpt_path = os.path.join(log_dir, f"{run_name}.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"Loaded best checkpoint: {ckpt_path}")

        model.to(device)
        model.eval()
        all_preds, all_targets, all_masks = [], [], []
        with torch.no_grad():
            for features, targets, masks in orig_loader:
                features = features.to(device)
                preds, _ = model(features)
                all_preds.append(preds.cpu())
                all_targets.append(targets)
                all_masks.append(masks)

        preds = torch.cat(all_preds, dim=0).numpy()
        targets = torch.cat(all_targets, dim=0).numpy()
        masks = torch.cat(all_masks, dim=0).numpy()

        # Clip predictions
        clip_range = config.get('evaluation', {}).get('clip_range', [-6, 6])
        preds = np.clip(preds, clip_range[0], clip_range[1])

        # Flatten and apply mask
        preds_flat = preds.reshape(-1, 2)
        targets_flat = targets.reshape(-1, 2)
        masks_flat = masks.reshape(-1).astype(bool)
        scored_preds = preds_flat[masks_flat]
        scored_targets = targets_flat[masks_flat]

        t0_corr = weighted_pearson_correlation(scored_targets[:, 0], scored_preds[:, 0])
        t1_corr = weighted_pearson_correlation(scored_targets[:, 1], scored_preds[:, 1])
        score = (t0_corr + t1_corr) / 2
        print(f"Original val: t0={t0_corr:.4f}, t1={t1_corr:.4f}, avg={score:.4f}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
