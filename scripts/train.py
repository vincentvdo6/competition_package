#!/usr/bin/env python
"""Training script for Wunderfund Predictorium models."""

import argparse
import yaml
import torch
import os
import sys

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.dataset import create_dataloaders
from src.models.gru_baseline import GRUBaseline
from src.models.lstm_model import LSTMModel
from src.training.trainer import Trainer, setup_cpu_performance
from src.training.losses import get_loss_function


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_model(config: dict) -> torch.nn.Module:
    """Factory for creating models from config."""
    model_type = config.get('model', {}).get('type', 'gru')

    if model_type == 'gru':
        return GRUBaseline(config)
    elif model_type == 'lstm':
        return LSTMModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser(description='Train model for Wunderfund Predictorium')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # CPU performance: use all cores
    if device.type == 'cpu':
        setup_cpu_performance()
        print(f"CPU threads: {torch.get_num_threads()}")

    # Create data loaders
    data_cfg = config.get('data', {})
    train_path = data_cfg.get('train_path', 'datasets/train.parquet')
    valid_path = data_cfg.get('valid_path', 'datasets/valid.parquet')
    batch_size = int(config.get('training', {}).get('batch_size', 32))
    normalize = data_cfg.get('normalize', True)
    derived_features = data_cfg.get('derived_features', False)
    use_gpu = device.type == 'cuda'

    print(f"Loading data from {train_path} and {valid_path}...")
    if derived_features:
        print("Derived features ENABLED (42 total = 32 raw + 10 derived)")
    train_loader, valid_loader, normalizer = create_dataloaders(
        train_path=train_path,
        valid_path=valid_path,
        batch_size=batch_size,
        normalize=normalize,
        pin_memory=use_gpu,
        derived_features=derived_features,
    )
    print(f"Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")

    # Save normalizer for inference
    log_dir = config.get('logging', {}).get('log_dir', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    if normalizer is not None:
        normalizer_path = os.path.join(log_dir, 'normalizer.npz')
        normalizer.save(normalizer_path)
        print(f"Saved normalizer to {normalizer_path}")

    # Create model
    model = get_model(config)
    param_count = model.count_parameters()
    print(f"Created {config.get('model', {}).get('type', 'gru').upper()} model with {param_count:,} parameters")

    # Create loss function
    loss_fn = get_loss_function(config)
    print(f"Using loss function: {type(loss_fn).__name__}")

    # Create trainer and train
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        valid_loader=valid_loader,
        loss_fn=loss_fn,
        device=device
    )

    print("\n" + "=" * 60)
    history = trainer.train()
    print("=" * 60)
    print(f"\nTraining complete. Best validation score: {trainer.best_score:.4f}")


if __name__ == '__main__':
    main()
