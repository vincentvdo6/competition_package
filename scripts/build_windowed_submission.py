#!/usr/bin/env python
"""Build a windowed inference submission zip.

Uses the same model weights but feeds a rolling window of the last N steps
to the GRU each prediction step (like the official baseline approach),
instead of step-by-step stateful inference.

Usage:
    python scripts/build_windowed_submission.py \
        --checkpoint logs/_staging/gru_derived_tightwd_v2_seed42.pt \
        --normalizer logs/_staging/normalizer_gru_derived_tightwd_v2_seed42.npz \
        --config configs/gru_derived_tightwd_v2.yaml \
        --window-size 100 \
        --output submissions/windowed_tw2_s42.zip
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class _GRUWindowWrapper(nn.Module):
    """Wrapper for ONNX export: full-sequence forward pass (no hidden state I/O)."""

    def __init__(self, input_proj, input_norm, gru, output_proj):
        super().__init__()
        self.input_proj = input_proj
        self.input_norm = input_norm
        self.gru = gru
        self.output_proj = output_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, seq_len, input_size)
        x = self.input_proj(x)
        x = self.input_norm(x)
        gru_out, _ = self.gru(x)
        predictions = self.output_proj(gru_out)
        return predictions  # (1, seq_len, 2)


def export_windowed_onnx(ckpt_path: Path, model_cfg: dict, onnx_path: Path, window_size: int) -> None:
    """Export a GRU checkpoint to ONNX for windowed inference."""
    from src.models.gru_baseline import GRUBaseline

    config = {"model": model_cfg}
    model = GRUBaseline(config)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    wrapper = _GRUWindowWrapper(
        model.input_proj, model.input_norm, model.gru, model.output_proj
    )
    wrapper.eval()

    input_size = model_cfg["input_size"]
    dummy_x = torch.randn(1, window_size, input_size)

    torch.onnx.export(
        wrapper,
        (dummy_x,),
        str(onnx_path),
        input_names=["input"],
        output_names=["predictions"],
        dynamic_axes={"input": {1: "seq_len"}, "predictions": {1: "seq_len"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Exported windowed ONNX to {onnx_path}")


def generate_windowed_solution(window_size: int, has_derived: bool) -> str:
    """Generate solution.py for windowed inference."""
    return f'''import json
import os

import numpy as np
import onnxruntime as ort

WINDOW_SIZE = {window_size}
HAS_DERIVED = {has_derived}


def compute_derived(features, eps=1e-8):
    spreads = features[6:12] - features[0:6]
    trade_intensity = features[28:32].sum(keepdims=True)
    bid_pressure = features[12:18].sum(keepdims=True)
    ask_pressure = features[18:24].sum(keepdims=True)
    pressure_imbalance = (bid_pressure - ask_pressure) / (bid_pressure + ask_pressure + eps)
    return np.concatenate([
        spreads,
        trade_intensity,
        bid_pressure,
        ask_pressure,
        pressure_imbalance,
    ]).astype(np.float32)


class Normalizer:
    def __init__(self, path):
        data = np.load(path)
        self.mean = data["mean"]
        self.std = data["std"]

    def transform(self, x):
        return ((x - self.mean) / self.std).astype(np.float32)


class PredictionModel:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Load ONNX model
        onnx_path = os.path.join(base_dir, "model.onnx")
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(onnx_path, sess_options, providers=["CPUExecutionProvider"])

        # Load normalizer
        self.normalizer = Normalizer(os.path.join(base_dir, "normalizer.npz"))

        # State
        self.current_seq_ix = None
        self.history = []  # list of normalized feature vectors

    def predict(self, data_point):
        # Reset on new sequence
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.history = []

        # Compute features
        raw = data_point.state.astype(np.float32)
        if HAS_DERIVED:
            features = np.concatenate([raw, compute_derived(raw)])
        else:
            features = raw

        # Normalize and store
        normalized = self.normalizer.transform(features.reshape(1, -1)).squeeze(0)
        self.history.append(normalized)

        # Skip if no prediction needed
        if not data_point.need_prediction:
            return None

        # Build window (last WINDOW_SIZE steps, zero-padded if needed)
        window = self.history[-WINDOW_SIZE:]
        if len(window) < WINDOW_SIZE:
            pad = [np.zeros_like(window[0])] * (WINDOW_SIZE - len(window))
            window = pad + window

        # Run inference: (1, WINDOW_SIZE, features)
        data_arr = np.stack(window, axis=0).astype(np.float32)
        data_tensor = np.expand_dims(data_arr, axis=0)

        predictions = self.sess.run(["predictions"], {{"input": data_tensor}})[0]
        # Take last timestep: (1, WINDOW_SIZE, 2) -> (2,)
        pred = predictions[0, -1, :]

        return np.clip(pred, -6, 6)
'''


def main():
    parser = argparse.ArgumentParser(description="Build windowed inference submission")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint (.pt)")
    parser.add_argument("--normalizer", required=True, help="Normalizer (.npz)")
    parser.add_argument("--config", required=True, help="Model config (.yaml)")
    parser.add_argument("--window-size", type=int, default=100, help="Window size (default: 100)")
    parser.add_argument("--output", required=True, help="Output zip path")
    args = parser.parse_args()

    # Load config
    cfg_path = ROOT / args.config
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    data_cfg = cfg.get("data", {})
    has_derived = bool(data_cfg.get("derived_features", False))

    ckpt_path = ROOT / args.checkpoint
    norm_path = ROOT / args.normalizer
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Export ONNX
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "model.onnx"
        export_windowed_onnx(ckpt_path, model_cfg, onnx_path, args.window_size)

        # Generate solution.py
        solution_code = generate_windowed_solution(args.window_size, has_derived)

        # Build zip
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("solution.py", solution_code)
            zf.write(str(onnx_path), "model.onnx")
            zf.write(str(norm_path), "normalizer.npz")

    size_kb = out_path.stat().st_size / 1024.0
    print(f"\nWindowed submission exported: {out_path}")
    print(f"  Window size: {args.window_size}")
    print(f"  Derived features: {has_derived}")
    print(f"  Size: {size_kb:.1f} KB ({size_kb/1024:.1f} MB)")


if __name__ == "__main__":
    main()
