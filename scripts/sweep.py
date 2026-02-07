#!/usr/bin/env python
"""Hyperparameter sweep launcher.

Generates config variants from a base config + sweep definition,
runs them sequentially, and writes a summary table at the end.

Usage:
    # Run a predefined sweep
    python scripts/sweep.py --base configs/gru_derived_v1.yaml --sweep lr

    # Run all predefined sweeps
    python scripts/sweep.py --base configs/gru_derived_v1.yaml --sweep all

    # Custom sweep via JSON overrides
    python scripts/sweep.py --base configs/gru_derived_v1.yaml \
        --override '{"training.lr": [0.0005, 0.001, 0.002]}'
"""

import argparse
import copy
import json
import os
import sys
import time
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── Predefined sweeps ────────────────────────────────────────────────

SWEEPS = {
    'lr': {
        'description': 'Learning rate sweep',
        'param': 'training.lr',
        'values': [0.0003, 0.0005, 0.001, 0.002, 0.003],
    },
    'hidden_size': {
        'description': 'Hidden size sweep',
        'param': 'model.hidden_size',
        'values': [64, 128, 192, 256],
    },
    'num_layers': {
        'description': 'Number of RNN layers',
        'param': 'model.num_layers',
        'values': [1, 2, 3],
    },
    'dropout': {
        'description': 'Dropout rate sweep',
        'param': 'model.dropout',
        'values': [0.1, 0.2, 0.3, 0.4],
    },
    'batch_size': {
        'description': 'Batch size sweep',
        'param': 'training.batch_size',
        'values': [128, 256, 512],
    },
    'weighted_ratio': {
        'description': 'Combined loss weighting ratio',
        'param': 'training.weighted_ratio',
        'values': [0.3, 0.5, 0.6, 0.7, 0.8],
    },
    'target_weights': {
        'description': 'Per-target loss weights [t0, t1]',
        'param': 'training.target_weights',
        'values': [[0.5, 0.5], [0.4, 0.6], [0.3, 0.7], [0.35, 0.65]],
    },
}


def set_nested(d: dict, key: str, value) -> dict:
    """Set a value in a nested dict using dot-separated key."""
    keys = key.split('.')
    current = d
    for k in keys[:-1]:
        current = current.setdefault(k, {})
    current[keys[-1]] = value
    return d


def get_nested(d: dict, key: str, default=None):
    """Get a value from a nested dict using dot-separated key."""
    keys = key.split('.')
    current = d
    for k in keys:
        if not isinstance(current, dict) or k not in current:
            return default
        current = current[k]
    return current


def generate_configs(base_config: dict, param: str, values: list) -> list:
    """Generate config variants by sweeping one parameter."""
    configs = []
    for val in values:
        cfg = copy.deepcopy(base_config)
        set_nested(cfg, param, val)
        # Fix input_size if hidden_size changed (input_proj depends on it)
        configs.append((param, val, cfg))
    return configs


def run_single(config: dict, run_name: str, device: str) -> dict:
    """Run a single training experiment and return results."""
    import torch
    from src.data.dataset import create_dataloaders
    from src.training.trainer import Trainer, setup_cpu_performance
    from src.training.losses import get_loss_function

    # Inline model factory (same as train.py)
    from src.models.gru_attention import GRUAttentionModel
    from src.models.gru_baseline import GRUBaseline
    from src.models.lstm_model import LSTMModel

    def get_model(cfg):
        mt = cfg.get('model', {}).get('type', 'gru')
        if mt == 'gru':
            return GRUBaseline(cfg)
        elif mt == 'gru_attention':
            return GRUAttentionModel(cfg)
        elif mt == 'lstm':
            return LSTMModel(cfg)
        else:
            raise ValueError(f"Unknown model type: {mt}")

    # Device
    if device == 'auto':
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        dev = torch.device(device)

    if dev.type == 'cpu':
        setup_cpu_performance()

    # Data
    data_cfg = config.get('data', {})
    train_cfg = config.get('training', {})
    train_loader, valid_loader, normalizer = create_dataloaders(
        train_path=data_cfg.get('train_path', 'datasets/train.parquet'),
        valid_path=data_cfg.get('valid_path', 'datasets/valid.parquet'),
        batch_size=int(train_cfg.get('batch_size', 256)),
        normalize=data_cfg.get('normalize', True),
        pin_memory=dev.type == 'cuda',
        derived_features=data_cfg.get('derived_features', False),
        temporal_features=data_cfg.get('temporal_features', False),
        interaction_features=data_cfg.get('interaction_features', False),
    )

    # Model, loss, trainer
    model = get_model(config)
    loss_fn = get_loss_function(config)

    # Override log_dir to avoid overwriting best_model.pt from other runs
    sweep_log_dir = os.path.join(
        config.get('logging', {}).get('log_dir', 'logs'),
        'sweeps',
        run_name,
    )
    config_copy = copy.deepcopy(config)
    config_copy.setdefault('logging', {})['log_dir'] = sweep_log_dir
    os.makedirs(sweep_log_dir, exist_ok=True)

    trainer = Trainer(
        model=model,
        config=config_copy,
        train_loader=train_loader,
        valid_loader=valid_loader,
        loss_fn=loss_fn,
        device=dev,
    )

    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    return {
        'run_name': run_name,
        'best_score': trainer.best_score,
        'best_epoch': trainer.best_epoch,
        'elapsed_s': elapsed,
        'params': model.count_parameters(),
        'log_dir': sweep_log_dir,
    }


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter sweep')
    parser.add_argument('--base', type=str, required=True,
                        help='Base config YAML path')
    parser.add_argument('--sweep', type=str, default=None,
                        help=f'Predefined sweep name ({", ".join(SWEEPS.keys())}, all)')
    parser.add_argument('--override', type=str, default=None,
                        help='JSON dict: {"param.path": [val1, val2, ...]}')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda/cpu/auto)')
    args = parser.parse_args()

    with open(args.base, 'r') as f:
        base_config = yaml.safe_load(f)

    # Build list of (param, values) to sweep
    sweep_list = []

    if args.sweep:
        if args.sweep == 'all':
            for name, s in SWEEPS.items():
                sweep_list.append((s['param'], s['values'], name))
        elif args.sweep in SWEEPS:
            s = SWEEPS[args.sweep]
            sweep_list.append((s['param'], s['values'], args.sweep))
        else:
            print(f"Unknown sweep: {args.sweep}")
            print(f"Available: {', '.join(SWEEPS.keys())}, all")
            sys.exit(1)

    if args.override:
        overrides = json.loads(args.override)
        for param, values in overrides.items():
            sweep_list.append((param, values, param.replace('.', '_')))

    if not sweep_list:
        print("No sweep specified. Use --sweep or --override.")
        sys.exit(1)

    # Run sweeps
    all_results = []
    total_runs = sum(len(vals) for _, vals, _ in sweep_list)
    run_idx = 0

    for param, values, sweep_name in sweep_list:
        print(f"\n{'='*60}")
        print(f"SWEEP: {sweep_name} ({param})")
        print(f"Values: {values}")
        print(f"{'='*60}\n")

        configs = generate_configs(base_config, param, values)

        for param_path, val, cfg in configs:
            run_idx += 1
            val_str = str(val).replace(' ', '').replace(',', '_').replace('[', '').replace(']', '')
            run_name = f"{sweep_name}_{val_str}"
            print(f"\n--- Run {run_idx}/{total_runs}: {run_name} ({param_path}={val}) ---\n")

            result = run_single(cfg, run_name, args.device)
            result['param'] = param_path
            result['value'] = val
            all_results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"{'Run':<35} {'Score':>8} {'Epoch':>6} {'Params':>10} {'Time':>8}")
    print("-" * 70)

    best = max(all_results, key=lambda r: r['best_score'])
    for r in all_results:
        marker = " *" if r is best else ""
        print(f"{r['run_name']:<35} {r['best_score']:>8.4f} {r['best_epoch']:>6d} "
              f"{r['params']:>10,} {r['elapsed_s']:>7.0f}s{marker}")

    print("-" * 70)
    print(f"Best: {best['run_name']} -> {best['best_score']:.4f} at epoch {best['best_epoch']}")
    print(f"Checkpoint: {best['log_dir']}/best_model.pt")

    # Save results to JSON
    log_dir = base_config.get('logging', {}).get('log_dir', 'logs')
    results_path = os.path.join(log_dir, 'sweeps', 'sweep_results.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
