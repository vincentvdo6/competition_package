#!/usr/bin/env python
"""Export an ensemble of trained models as a competition submission zip.

Generates a self-contained solution.py that loads N models, runs them all
on each step, and averages their predictions.

Usage:
    python scripts/export_ensemble.py \
        --config configs/gru_derived_tightwd_v2.yaml \
        --checkpoints logs/model_seed42.pt logs/model_seed43.pt logs/model_seed44.pt \
        --normalizer logs/normalizer.npz \
        --output submissions/ensemble.zip

    # With custom weights:
    python scripts/export_ensemble.py \
        --config configs/gru_derived_tightwd_v2.yaml \
        --checkpoints logs/model_seed42.pt logs/model_seed43.pt \
        --normalizer logs/normalizer.npz \
        --weights 0.6 0.4 \
        --output submissions/ensemble.zip

    # Heterogeneous ensemble (different configs per model):
    python scripts/export_ensemble.py \
        --configs configs/gru_derived_tightwd_v2.yaml configs/gru_attention_v1.yaml \
        --checkpoints logs/gru_seed42.pt logs/attn_seed42.pt \
        --normalizer logs/normalizer.npz \
        --output submissions/ensemble.zip
"""

import argparse
import json
import yaml
import os
import sys
import zipfile
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _gen_gru_class(name, input_size, hidden_size, num_layers, dropout, output_size):
    """Generate GRU model class code."""
    return textwrap.dedent(f'''\
        class {name}(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden_size = {hidden_size}
                self.num_layers = {num_layers}
                self.input_proj = nn.Linear({input_size}, {hidden_size})
                self.input_norm = nn.LayerNorm({hidden_size})
                self.input_dropout = nn.Dropout({dropout})
                self.gru = nn.GRU(
                    input_size={hidden_size}, hidden_size={hidden_size},
                    num_layers={num_layers}, batch_first=True,
                    dropout={dropout if num_layers > 1 else 0.0}, bidirectional=False,
                )
                self.output_proj = nn.Sequential(
                    nn.Linear({hidden_size}, {hidden_size // 2}), nn.ReLU(),
                    nn.Dropout({dropout}), nn.Linear({hidden_size // 2}, {output_size}),
                )

            def forward_step(self, x, hidden=None):
                if hidden is None:
                    hidden = torch.zeros({num_layers}, 1, {hidden_size})
                x = x.unsqueeze(0).unsqueeze(0)
                x = self.input_proj(x)
                x = self.input_norm(x)
                x = self.input_dropout(x)
                out, hidden = self.gru(x, hidden)
                pred = self.output_proj(out.squeeze(0)).squeeze(0)
                return pred, hidden

    ''')


def _gen_lstm_class(name, input_size, hidden_size, num_layers, dropout, output_size):
    """Generate LSTM model class code."""
    return textwrap.dedent(f'''\
        class {name}(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden_size = {hidden_size}
                self.num_layers = {num_layers}
                self.input_proj = nn.Linear({input_size}, {hidden_size})
                self.input_norm = nn.LayerNorm({hidden_size})
                self.input_dropout = nn.Dropout({dropout})
                self.lstm = nn.LSTM(
                    input_size={hidden_size}, hidden_size={hidden_size},
                    num_layers={num_layers}, batch_first=True,
                    dropout={dropout if num_layers > 1 else 0.0}, bidirectional=False,
                )
                self.output_proj = nn.Sequential(
                    nn.Linear({hidden_size}, {hidden_size // 2}), nn.ReLU(),
                    nn.Dropout({dropout}), nn.Linear({hidden_size // 2}, {output_size}),
                )

            def forward_step(self, x, hidden=None):
                if hidden is None:
                    h = torch.zeros({num_layers}, 1, {hidden_size})
                    c = torch.zeros({num_layers}, 1, {hidden_size})
                    hidden = (h, c)
                x = x.unsqueeze(0).unsqueeze(0)
                x = self.input_proj(x)
                x = self.input_norm(x)
                x = self.input_dropout(x)
                out, hidden = self.lstm(x, hidden)
                pred = self.output_proj(out.squeeze(0)).squeeze(0)
                return pred, hidden

    ''')


MODEL_CLASS_GENERATORS = {
    'gru': _gen_gru_class,
    'lstm': _gen_lstm_class,
}


def generate_ensemble_solution(configs, checkpoint_names, normalizer_name, weights):
    """Generate self-contained ensemble solution.py."""

    n_models = len(configs)

    # Determine which unique model architectures we need
    model_classes = {}  # class_name -> code
    model_assignments = []  # index -> class_name

    for i, cfg in enumerate(configs):
        mcfg = cfg.get('model', {})
        model_type = mcfg.get('type', 'gru')
        input_size = mcfg.get('input_size', 32)
        hidden_size = mcfg.get('hidden_size', 128)
        num_layers = mcfg.get('num_layers', 2)
        dropout = mcfg.get('dropout', 0.2)
        output_size = mcfg.get('output_size', 2)

        # Create unique class name based on architecture params
        class_key = f"{model_type}_{input_size}_{hidden_size}_{num_layers}"
        class_name = f"Model_{model_type}_{hidden_size}h_{num_layers}L"

        if class_key not in model_classes:
            gen = MODEL_CLASS_GENERATORS.get(model_type)
            if gen is None:
                raise ValueError(f"Unsupported model type: {model_type}")
            model_classes[class_key] = gen(
                class_name, input_size, hidden_size, num_layers, dropout, output_size
            )

        model_assignments.append(class_name)

    # Check if all models use derived features
    use_derived = any(c.get('data', {}).get('derived_features', False) for c in configs)

    # Preamble
    code = textwrap.dedent(f'''\
        import numpy as np
        import torch
        import torch.nn as nn
        import json
        import os

        USE_DERIVED_FEATURES = {use_derived}

        def compute_derived(features, eps=1e-8):
            """Compute 10 derived features from raw 32-feature vector."""
            spreads = features[6:12] - features[0:6]
            trade_intensity = features[28:32].sum(keepdims=True)
            bid_pressure = features[12:18].sum(keepdims=True)
            ask_pressure = features[18:24].sum(keepdims=True)
            pressure_imbalance = (bid_pressure - ask_pressure) / (bid_pressure + ask_pressure + eps)
            return np.concatenate([spreads, trade_intensity, bid_pressure, ask_pressure, pressure_imbalance]).astype(np.float32)

        class Normalizer:
            def __init__(self, path):
                data = np.load(path)
                self.mean = data['mean']
                self.std = data['std']

            def transform(self, x):
                return ((x - self.mean) / self.std).astype(np.float32)

    ''')

    # Add unique model class definitions
    for class_code in model_classes.values():
        code += class_code

    # Build model registry string
    registry_items = []
    for i in range(n_models):
        registry_items.append(f'        "{checkpoint_names[i]}": {model_assignments[i]},')
    registry_str = '\n'.join(registry_items)

    # Ensemble config
    ensemble_cfg = {
        'checkpoints': checkpoint_names,
        'normalizer': normalizer_name,
        'weights': weights,
    }

    # Competition interface
    code += textwrap.dedent(f'''\
        # ── Ensemble Competition Interface ────────────────────────────────

        MODEL_REGISTRY = {{
    {registry_str}
        }}

        class PredictionModel:
            def __init__(self):
                base_dir = os.path.dirname(os.path.abspath(__file__))

                # Load ensemble config
                with open(os.path.join(base_dir, 'ensemble_config.json')) as f:
                    self.cfg = json.load(f)

                # Load normalizer (shared across all models)
                self.normalizer = Normalizer(
                    os.path.join(base_dir, self.cfg['normalizer'])
                )

                # Load all models
                self.models = []
                self.hiddens = []
                self.weights = self.cfg['weights']
                for ckpt_name in self.cfg['checkpoints']:
                    model_cls = MODEL_REGISTRY[ckpt_name]
                    model = model_cls()
                    checkpoint = torch.load(
                        os.path.join(base_dir, ckpt_name),
                        map_location='cpu', weights_only=False,
                    )
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.eval()
                    self.models.append(model)
                    self.hiddens.append(None)

                self.current_seq_ix = None

            def predict(self, data_point) -> np.ndarray:
                # Reset all hidden states on new sequence
                if self.current_seq_ix != data_point.seq_ix:
                    self.current_seq_ix = data_point.seq_ix
                    self.hiddens = [None] * len(self.models)

                # Build features
                raw = data_point.state.astype(np.float32)
                if USE_DERIVED_FEATURES:
                    derived = compute_derived(raw)
                    raw = np.concatenate([raw, derived])
                features = self.normalizer.transform(raw.reshape(1, -1)).squeeze(0)
                x = torch.from_numpy(features)

                # Run all models and collect predictions
                preds = []
                for i, model in enumerate(self.models):
                    with torch.no_grad():
                        pred, self.hiddens[i] = model.forward_step(x, self.hiddens[i])
                        preds.append(pred.numpy())

                # Weighted average
                ensemble_pred = sum(
                    w * p for w, p in zip(self.weights, preds)
                )
                ensemble_pred = np.clip(ensemble_pred, -6, 6)

                if not data_point.need_prediction:
                    return None

                return ensemble_pred
    ''')

    return code, ensemble_cfg


def main():
    parser = argparse.ArgumentParser(description='Export ensemble submission zip')
    parser.add_argument('--config', type=str, default=None,
                        help='Single config for homogeneous ensemble (all models same arch)')
    parser.add_argument('--configs', type=str, nargs='+', default=None,
                        help='Per-model configs for heterogeneous ensemble')
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True,
                        help='Paths to model checkpoints')
    parser.add_argument('--normalizer', type=str, default='logs/normalizer.npz',
                        help='Path to normalizer')
    parser.add_argument('--weights', type=float, nargs='+', default=None,
                        help='Per-model weights (default: uniform)')
    parser.add_argument('--output', type=str, default='submissions/ensemble.zip',
                        help='Output zip path')
    args = parser.parse_args()

    n_models = len(args.checkpoints)

    # Resolve configs
    if args.configs:
        if len(args.configs) != n_models:
            print(f"ERROR: --configs count ({len(args.configs)}) != --checkpoints count ({n_models})")
            sys.exit(1)
        config_paths = args.configs
    elif args.config:
        config_paths = [args.config] * n_models
    else:
        print("ERROR: provide --config (homogeneous) or --configs (heterogeneous)")
        sys.exit(1)

    # Load all configs
    configs = []
    for p in config_paths:
        with open(p) as f:
            configs.append(yaml.safe_load(f))

    # Resolve weights
    if args.weights:
        if len(args.weights) != n_models:
            print(f"ERROR: --weights count ({len(args.weights)}) != --checkpoints count ({n_models})")
            sys.exit(1)
        weights = args.weights
    else:
        weights = [1.0 / n_models] * n_models

    # Validate files exist
    for path in args.checkpoints:
        if not os.path.exists(path):
            print(f"ERROR: checkpoint not found: {path}")
            sys.exit(1)
    if not os.path.exists(args.normalizer):
        print(f"ERROR: normalizer not found: {args.normalizer}")
        sys.exit(1)

    # Generate checkpoint names for inside the zip
    checkpoint_names = [f"model_{i}.pt" for i in range(n_models)]
    normalizer_name = "normalizer.npz"

    # Generate solution code
    solution_code, ensemble_cfg = generate_ensemble_solution(
        configs, checkpoint_names, normalizer_name, weights
    )

    # Create output directory
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # Package into zip
    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('solution.py', solution_code)
        zf.writestr('ensemble_config.json', json.dumps(ensemble_cfg, indent=2))
        zf.write(args.normalizer, normalizer_name)
        for i, ckpt_path in enumerate(args.checkpoints):
            zf.write(ckpt_path, checkpoint_names[i])

    zip_size = os.path.getsize(args.output)
    print(f"Ensemble submission exported to {args.output}")
    print(f"  Models: {n_models}")
    print(f"  Weights: {weights}")
    print(f"  Size: {zip_size / 1024:.1f} KB")
    print(f"  Contents: solution.py, ensemble_config.json, normalizer.npz, {n_models} checkpoints")
    for i, p in enumerate(args.checkpoints):
        print(f"    model_{i}.pt <- {p}")
    print(f"\nTo validate, upload to https://wundernn.io/predictorium")


if __name__ == '__main__':
    main()
