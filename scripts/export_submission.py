#!/usr/bin/env python
"""Export trained model as a competition submission zip.

Generates a self-contained solution.py (with embedded model definition)
and packages it with model weights and normalizer into a zip file.

Usage:
    python scripts/export_submission.py \
        --config configs/gru_baseline.yaml \
        --checkpoint logs/best_model.pt \
        --normalizer logs/normalizer.npz \
        --output submissions/submission.zip
"""

import argparse
import yaml
import os
import sys
import zipfile
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def generate_solution_py(config: dict) -> str:
    """Generate a self-contained solution.py for the competition.

    The generated file embeds the model definition (GRU or LSTM), normalizer
    loading, and optional derived feature computation so it has no dependency
    on src/.
    """
    model_cfg = config.get('model', {})
    data_cfg = config.get('data', {})

    model_type = model_cfg.get('type', 'gru')
    input_size = model_cfg.get('input_size', 32)
    hidden_size = model_cfg.get('hidden_size', 128)
    num_layers = model_cfg.get('num_layers', 2)
    dropout = model_cfg.get('dropout', 0.2)
    output_size = model_cfg.get('output_size', 2)
    derived_features = data_cfg.get('derived_features', False)

    # Shared preamble: imports, derived features, normalizer
    preamble = textwrap.dedent(f'''\
        import numpy as np
        import torch
        import torch.nn as nn
        import os

        # ── Derived feature computation ──────────────────────────────────

        USE_DERIVED_FEATURES = {derived_features}

        def compute_derived(features, eps=1e-8):
            """Compute 10 derived features from raw 32-feature vector."""
            spreads = features[6:12] - features[0:6]
            trade_intensity = features[28:32].sum(keepdims=True)
            bid_pressure = features[12:18].sum(keepdims=True)
            ask_pressure = features[18:24].sum(keepdims=True)
            pressure_imbalance = (bid_pressure - ask_pressure) / (bid_pressure + ask_pressure + eps)
            return np.concatenate([spreads, trade_intensity, bid_pressure, ask_pressure, pressure_imbalance]).astype(np.float32)


        # ── Normalizer ───────────────────────────────────────────────────

        class Normalizer:
            def __init__(self, path):
                data = np.load(path)
                self.mean = data['mean']
                self.std = data['std']

            def transform(self, x):
                return ((x - self.mean) / self.std).astype(np.float32)

    ''')

    # Model-specific code
    if model_type == 'lstm':
        model_code = textwrap.dedent(f'''\
        # ── LSTM Model (self-contained, no src/ dependency) ──────────────

        class SequenceModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden_size = {hidden_size}
                self.num_layers = {num_layers}

                self.input_proj = nn.Linear({input_size}, {hidden_size})
                self.input_norm = nn.LayerNorm({hidden_size})
                self.input_dropout = nn.Dropout({dropout})

                self.lstm = nn.LSTM(
                    input_size={hidden_size},
                    hidden_size={hidden_size},
                    num_layers={num_layers},
                    batch_first=True,
                    dropout={dropout if num_layers > 1 else 0.0},
                    bidirectional=False,
                )

                self.output_proj = nn.Sequential(
                    nn.Linear({hidden_size}, {hidden_size // 2}),
                    nn.ReLU(),
                    nn.Dropout({dropout}),
                    nn.Linear({hidden_size // 2}, {output_size}),
                )

            def forward_step(self, x, hidden=None):
                """Single-step forward for online inference."""
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
    else:
        model_code = textwrap.dedent(f'''\
        # ── GRU Model (self-contained, no src/ dependency) ───────────────

        class SequenceModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden_size = {hidden_size}
                self.num_layers = {num_layers}

                self.input_proj = nn.Linear({input_size}, {hidden_size})
                self.input_norm = nn.LayerNorm({hidden_size})
                self.input_dropout = nn.Dropout({dropout})

                self.gru = nn.GRU(
                    input_size={hidden_size},
                    hidden_size={hidden_size},
                    num_layers={num_layers},
                    batch_first=True,
                    dropout={dropout if num_layers > 1 else 0.0},
                    bidirectional=False,
                )

                self.output_proj = nn.Sequential(
                    nn.Linear({hidden_size}, {hidden_size // 2}),
                    nn.ReLU(),
                    nn.Dropout({dropout}),
                    nn.Linear({hidden_size // 2}, {output_size}),
                )

            def forward_step(self, x, hidden=None):
                """Single-step forward for online inference."""
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

    # Shared prediction interface
    interface_code = textwrap.dedent(f'''\
        # ── Competition interface ─────────────────────────────────────────

        class PredictionModel:
            def __init__(self):
                base_dir = os.path.dirname(os.path.abspath(__file__))

                # Load normalizer
                self.normalizer = Normalizer(os.path.join(base_dir, 'normalizer.npz'))

                # Load model
                self.model = SequenceModel()
                checkpoint = torch.load(
                    os.path.join(base_dir, 'best_model.pt'),
                    map_location='cpu',
                    weights_only=False,
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()

                # State
                self.current_seq_ix = None
                self.hidden = None

            def predict(self, data_point) -> np.ndarray:
                # Reset hidden on new sequence
                if self.current_seq_ix != data_point.seq_ix:
                    self.current_seq_ix = data_point.seq_ix
                    self.hidden = None

                # Build features
                raw = data_point.state.astype(np.float32)
                if USE_DERIVED_FEATURES:
                    derived = compute_derived(raw)
                    raw = np.concatenate([raw, derived])
                features = self.normalizer.transform(raw.reshape(1, -1)).squeeze(0)
                x = torch.from_numpy(features)

                # Forward step
                with torch.no_grad():
                    pred, self.hidden = self.model.forward_step(x, self.hidden)
                    pred = pred.numpy()

                pred = np.clip(pred, -6, 6)

                if not data_point.need_prediction:
                    return None

                return pred
    ''')

    return preamble + model_code + interface_code


def main():
    parser = argparse.ArgumentParser(description='Export submission zip')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML')
    parser.add_argument('--checkpoint', type=str, default='logs/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--normalizer', type=str, default='logs/normalizer.npz',
                        help='Path to normalizer')
    parser.add_argument('--output', type=str, default='submissions/submission.zip',
                        help='Output zip path')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Validate files exist
    for path, name in [(args.checkpoint, 'checkpoint'), (args.normalizer, 'normalizer')]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found at {path}")
            sys.exit(1)

    # Generate solution.py
    solution_code = generate_solution_py(config)

    # Create output directory
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # Package into zip
    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('solution.py', solution_code)
        zf.write(args.checkpoint, 'best_model.pt')
        zf.write(args.normalizer, 'normalizer.npz')

    zip_size = os.path.getsize(args.output)
    print(f"Submission exported to {args.output}")
    print(f"  Size: {zip_size / 1024:.1f} KB")
    print(f"  Contents: solution.py, best_model.pt, normalizer.npz")
    print(f"  Model type: {config.get('model', {}).get('type', 'gru')}")
    print(f"  Derived features: {config.get('data', {}).get('derived_features', False)}")
    print(f"\nTo validate, upload to https://wundernn.io/predictorium")


if __name__ == '__main__':
    main()
