#!/usr/bin/env python
"""Export trained model as a competition submission zip.

Generates a self-contained solution.py (embedded model + preprocessing)
and packages it with model checkpoint and normalizer.
"""

import argparse
import os
import sys
import textwrap
import zipfile

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def generate_solution_py(config: dict) -> str:
    """Generate self-contained solution.py compatible with competition runtime."""
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})

    model_type = model_cfg.get("type", "gru")
    input_size = int(model_cfg.get("input_size", 32))
    hidden_size = int(model_cfg.get("hidden_size", 128))
    num_layers = int(model_cfg.get("num_layers", 2))
    dropout = float(model_cfg.get("dropout", 0.2))
    output_size = int(model_cfg.get("output_size", 2))

    attention_heads = int(model_cfg.get("attention_heads", 4))
    attention_dropout = float(model_cfg.get("attention_dropout", 0.1))
    attention_window = int(model_cfg.get("attention_window", 128))

    derived_features = bool(data_cfg.get("derived_features", False))
    temporal_features = bool(data_cfg.get("temporal_features", False) and derived_features)
    interaction_features = bool(data_cfg.get("interaction_features", False))

    preamble = textwrap.dedent(
        f"""\
        import numpy as np
        import torch
        import torch.nn as nn
        import os

        USE_DERIVED_FEATURES = {derived_features}
        USE_TEMPORAL_FEATURES = {temporal_features}
        USE_INTERACTION_FEATURES = {interaction_features}

        def compute_derived(features, eps=1e-8):
            spreads = features[6:12] - features[0:6]
            trade_intensity = features[28:32].sum(keepdims=True)
            bid_pressure = features[12:18].sum(keepdims=True)
            ask_pressure = features[18:24].sum(keepdims=True)
            pressure_imbalance = (bid_pressure - ask_pressure) / (bid_pressure + ask_pressure + eps)
            return np.concatenate([spreads, trade_intensity, bid_pressure, ask_pressure, pressure_imbalance]).astype(np.float32)

        def compute_interactions(features, has_derived=True):
            p0 = features[0]
            v2 = features[14]
            v8 = features[20]
            spread_0 = features[32] if has_derived and features.shape[0] >= 33 else (features[6] - features[0])
            return np.array([v8 * p0, spread_0 * p0, spread_0 * v2], dtype=np.float32)

        class TemporalBuffer:
            def __init__(self):
                self.reset()

            def reset(self):
                self.step = 0
                self.spread0_history = []
                self.trade_int_history = []

            def compute_step(self, features_42):
                spread_0 = float(features_42[32])
                trade_int = float(features_42[38])
                self.spread0_history.append(spread_0)
                self.trade_int_history.append(trade_int)

                roc1 = spread_0 - self.spread0_history[-2] if self.step >= 1 else 0.0
                roc5 = spread_0 - self.spread0_history[-6] if self.step >= 5 else 0.0
                roll_mean = sum(self.trade_int_history[-5:]) / len(self.trade_int_history[-5:])
                self.step += 1
                return np.concatenate([features_42, np.array([roc1, roc5, roll_mean], dtype=np.float32)]).astype(np.float32)

        class Normalizer:
            def __init__(self, path):
                data = np.load(path)
                self.mean = data['mean']
                self.std = data['std']

            def transform(self, x):
                return ((x - self.mean) / self.std).astype(np.float32)

        """
    )

    if model_type == "lstm":
        model_code = textwrap.dedent(
            f"""\
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

            """
        )
    elif model_type == "gru_attention":
        model_code = textwrap.dedent(
            f"""\
            class SequenceModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.hidden_size = {hidden_size}
                    self.num_layers = {num_layers}
                    self.attention_window = {attention_window}

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
                    self.attn = nn.MultiheadAttention(
                        embed_dim={hidden_size},
                        num_heads={attention_heads},
                        dropout={attention_dropout},
                        batch_first=True,
                    )
                    self.attn_norm = nn.LayerNorm({hidden_size})
                    self.output_proj = nn.Sequential(
                        nn.Linear({hidden_size}, {hidden_size // 2}),
                        nn.ReLU(),
                        nn.Dropout({dropout}),
                        nn.Linear({hidden_size // 2}, {output_size}),
                    )

                def forward_step(self, x, hidden=None):
                    if hidden is None:
                        gru_hidden = None
                        attn_buffer = None
                    else:
                        gru_hidden, attn_buffer = hidden

                    x = x.unsqueeze(0).unsqueeze(0)
                    x = self.input_proj(x)
                    x = self.input_norm(x)
                    x = self.input_dropout(x)
                    out, gru_hidden = self.gru(x, gru_hidden)

                    cur = out.squeeze(0)
                    if attn_buffer is None:
                        context = cur.unsqueeze(0)
                    else:
                        context = torch.cat([attn_buffer, cur.unsqueeze(0)], dim=1)
                        if context.size(1) > self.attention_window:
                            context = context[:, -self.attention_window:, :]

                    q = cur.unsqueeze(0)
                    attn_out, _ = self.attn(q, context, context, need_weights=False)
                    fused = self.attn_norm(cur + attn_out.squeeze(0))
                    pred = self.output_proj(fused)
                    return pred, (gru_hidden, context.detach())

            """
        )
    else:
        model_code = textwrap.dedent(
            f"""\
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
                    if hidden is None:
                        hidden = torch.zeros({num_layers}, 1, {hidden_size})
                    x = x.unsqueeze(0).unsqueeze(0)
                    x = self.input_proj(x)
                    x = self.input_norm(x)
                    x = self.input_dropout(x)
                    out, hidden = self.gru(x, hidden)
                    pred = self.output_proj(out.squeeze(0)).squeeze(0)
                    return pred, hidden

            """
        )

    interface_code = textwrap.dedent(
        """\
        class PredictionModel:
            def __init__(self):
                base_dir = os.path.dirname(os.path.abspath(__file__))
                self.normalizer = Normalizer(os.path.join(base_dir, 'normalizer.npz'))
                self.model = SequenceModel()
                checkpoint = torch.load(
                    os.path.join(base_dir, 'best_model.pt'),
                    map_location='cpu',
                    weights_only=False,
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()

                self.current_seq_ix = None
                self.hidden = None
                self.temporal_buffer = TemporalBuffer() if USE_TEMPORAL_FEATURES else None

            def predict(self, data_point) -> np.ndarray:
                if self.current_seq_ix != data_point.seq_ix:
                    self.current_seq_ix = data_point.seq_ix
                    self.hidden = None
                    if self.temporal_buffer is not None:
                        self.temporal_buffer.reset()

                raw = data_point.state.astype(np.float32)
                if USE_DERIVED_FEATURES:
                    raw = np.concatenate([raw, compute_derived(raw)])
                if USE_TEMPORAL_FEATURES:
                    raw = self.temporal_buffer.compute_step(raw)
                if USE_INTERACTION_FEATURES:
                    inter = compute_interactions(raw, has_derived=USE_DERIVED_FEATURES)
                    raw = np.concatenate([raw, inter])

                x = torch.from_numpy(self.normalizer.transform(raw.reshape(1, -1)).squeeze(0))

                with torch.no_grad():
                    pred, self.hidden = self.model.forward_step(x, self.hidden)
                    pred = pred.numpy()

                pred = np.clip(pred, -6, 6)
                if not data_point.need_prediction:
                    return None
                return pred
        """
    )

    return preamble + model_code + interface_code


def main():
    parser = argparse.ArgumentParser(description="Export submission zip")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default="logs/best_model.pt", help="Path to checkpoint")
    parser.add_argument("--normalizer", type=str, default="logs/normalizer.npz", help="Path to normalizer")
    parser.add_argument("--output", type=str, default="submissions/submission.zip", help="Output zip path")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for path, name in [(args.checkpoint, "checkpoint"), (args.normalizer, "normalizer")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found at {path}")
            sys.exit(1)

    solution_code = generate_solution_py(config)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("solution.py", solution_code)
        zf.write(args.checkpoint, "best_model.pt")
        zf.write(args.normalizer, "normalizer.npz")

    zip_size = os.path.getsize(args.output)
    print(f"Submission exported to {args.output}")
    print(f"  Size: {zip_size / 1024:.1f} KB")
    print("  Contents: solution.py, best_model.pt, normalizer.npz")
    print(f"  Model type: {config.get('model', {}).get('type', 'gru')}")
    print(f"  Derived features: {config.get('data', {}).get('derived_features', False)}")
    print(f"  Temporal features: {config.get('data', {}).get('temporal_features', False)}")
    print(f"  Interaction features: {config.get('data', {}).get('interaction_features', False)}")


if __name__ == "__main__":
    main()
