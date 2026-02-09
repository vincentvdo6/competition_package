#!/usr/bin/env python
"""Training script for Wunderfund Predictorium models."""

import argparse
import random
import yaml
import torch
import numpy as np
import os
import sys

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.dataset import create_dataloaders
from src.models.gru_attention import GRUAttentionModel
from src.models.gru_baseline import GRUBaseline
from src.models.lstm_model import LSTMModel
from src.models.tcn_model import TCNModel
from src.training.trainer import Trainer, setup_cpu_performance
from src.training.losses import get_loss_function


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_model(config: dict) -> torch.nn.Module:
    """Factory for creating models from config."""
    model_type = config.get('model', {}).get('type', 'gru')

    if model_type == 'gru':
        return GRUBaseline(config)
    elif model_type == 'gru_attention':
        return GRUAttentionModel(config)
    elif model_type == 'lstm':
        return LSTMModel(config)
    elif model_type == 'tcn':
        return TCNModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser(description='Train model for Wunderfund Predictorium')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)
    print(f"Random seed: {args.seed}")

    # Load config
    config = load_config(args.config)
    config['seed'] = args.seed
    print(f"Loaded config from {args.config}")

    # Derive run name for per-seed checkpoints
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    run_name = f"{config_name}_seed{args.seed}"
    config.setdefault('logging', {})['checkpoint_prefix'] = run_name

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
    temporal_features = data_cfg.get('temporal_features', False)
    interaction_features = data_cfg.get('interaction_features', False)
    use_gpu = device.type == 'cuda'

    print(f"Loading data from {train_path} and {valid_path}...")
    feature_dim = 32
    if derived_features:
        feature_dim += 10
    if temporal_features and derived_features:
        feature_dim += 3
    if interaction_features:
        feature_dim += 3
    microstructure_features = data_cfg.get('microstructure_features', False)
    if microstructure_features:
        feature_dim += 6
    print(
        f"Feature pipeline: derived={derived_features}, temporal={temporal_features}, "
        f"interaction={interaction_features}, microstructure={microstructure_features} "
        f"-> input_size={feature_dim}"
    )
    train_loader, valid_loader, normalizer = create_dataloaders(
        train_path=train_path,
        valid_path=valid_path,
        batch_size=batch_size,
        normalize=normalize,
        pin_memory=use_gpu,
        derived_features=derived_features,
        temporal_features=temporal_features,
        interaction_features=interaction_features,
        microstructure_features=microstructure_features,
    )
    print(f"Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")

    # Save normalizer for inference
    log_dir = config.get('logging', {}).get('log_dir', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    if normalizer is not None:
        # Seed-safe path for reproducibility, plus legacy path for compatibility.
        normalizer_seed_name = f"normalizer_{run_name}.npz"
        normalizer_seed_path = os.path.join(log_dir, normalizer_seed_name)
        normalizer_legacy_path = os.path.join(log_dir, 'normalizer.npz')
        normalizer.save(normalizer_seed_path)
        normalizer.save(normalizer_legacy_path)
        config.setdefault('logging', {})['normalizer_path'] = normalizer_seed_name
        print(f"Saved normalizer to {normalizer_seed_path}")
        print(f"Updated compatibility normalizer at {normalizer_legacy_path}")

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
