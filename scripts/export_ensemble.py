#!/usr/bin/env python
"""Export an ensemble of trained models as a competition submission zip.

Supports both homogeneous and heterogeneous ensembles:
- per-model configs (`--config` or `--configs`)
- per-model normalizers (`--normalizer` or `--normalizers`)
- global weights (`--weights`) or per-target weights (`--weights-t0/--weights-t1`)

The generated `solution.py` reproduces each model's preprocessing pipeline:
raw -> derived -> temporal -> interaction -> normalize -> model.forward_step
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class _GRUStepWrapper(nn.Module):
    """Wrapper making GRU forward_step ONNX-exportable with explicit hidden state."""

    def __init__(self, input_proj, input_norm, gru, output_proj):
        super().__init__()
        self.input_proj = input_proj
        self.input_norm = input_norm
        self.gru = gru
        self.output_proj = output_proj

    def forward(self, x: torch.Tensor, hidden: torch.Tensor):
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = x.unsqueeze(1)
        gru_out, new_hidden = self.gru(x, hidden)
        prediction = self.output_proj(gru_out.squeeze(1))
        return prediction, new_hidden


def _export_gru_to_onnx(ckpt_path: Path, model_cfg: dict, onnx_path: Path) -> None:
    """Export a GRU checkpoint to ONNX for step-by-step inference."""
    from src.models.gru_baseline import GRUBaseline

    config = {"model": model_cfg}
    model = GRUBaseline(config)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    wrapper = _GRUStepWrapper(
        model.input_proj, model.input_norm, model.gru, model.output_proj
    )
    wrapper.eval()

    input_size = model_cfg["input_size"]
    hidden_size = model_cfg["hidden_size"]
    num_layers = model_cfg["num_layers"]

    dummy_x = torch.randn(1, input_size)
    dummy_h = torch.zeros(num_layers, 1, hidden_size)

    torch.onnx.export(
        wrapper,
        (dummy_x, dummy_h),
        str(onnx_path),
        input_names=["input", "hidden_in"],
        output_names=["prediction", "hidden_out"],
        dynamic_axes=None,
        opset_version=17,
        do_constant_folding=True,
    )


def _normalize_weights(weights: List[float], n_models: int, label: str) -> List[float]:
    if len(weights) != n_models:
        raise ValueError(
            f"{label} count ({len(weights)}) must match number of checkpoints ({n_models})"
        )
    total = float(sum(weights))
    if total <= 0:
        raise ValueError(f"{label} must sum to > 0")
    return [float(w) / total for w in weights]


def _load_weights_from_json(path: Path) -> Tuple[Optional[List[float]], Optional[Dict[str, List[float]]]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    # Common schemas:
    # 1) {"weights": [...]}  (global)
    # 2) {"target_weights": {"t0": [...], "t1": [...]}}
    # 3) {"t0": [...], "t1": [...]} 
    global_weights = None
    target_weights = None

    if isinstance(payload, dict):
        if "weights" in payload and isinstance(payload["weights"], list):
            global_weights = payload["weights"]

        if "target_weights" in payload and isinstance(payload["target_weights"], dict):
            tw = payload["target_weights"]
            if "t0" in tw and "t1" in tw:
                target_weights = {"t0": tw["t0"], "t1": tw["t1"]}

        if target_weights is None and "t0" in payload and "t1" in payload:
            if isinstance(payload["t0"], list) and isinstance(payload["t1"], list):
                target_weights = {"t0": payload["t0"], "t1": payload["t1"]}

    return global_weights, target_weights


def _resolve_config_paths(args: argparse.Namespace, n_models: int) -> List[Path]:
    if args.configs:
        if len(args.configs) != n_models:
            raise ValueError(
                f"--configs count ({len(args.configs)}) != --checkpoints count ({n_models})"
            )
        return [ROOT / p for p in args.configs]

    if args.config:
        return [ROOT / args.config] * n_models

    raise ValueError("Provide --config (homogeneous) or --configs (heterogeneous)")


def _resolve_normalizer_paths(args: argparse.Namespace, n_models: int) -> List[Path]:
    if args.normalizers:
        if len(args.normalizers) != n_models:
            raise ValueError(
                f"--normalizers count ({len(args.normalizers)}) != --checkpoints count ({n_models})"
            )
        return [ROOT / p for p in args.normalizers]

    if args.normalizer:
        return [ROOT / args.normalizer] * n_models

    raise ValueError("Provide --normalizer (shared) or --normalizers (per-model)")


def _resolve_weights(args: argparse.Namespace, n_models: int) -> Tuple[List[float], Optional[Dict[str, List[float]]]]:
    global_weights = None
    target_weights = None

    # 1) Optional JSON source
    if args.weights_json:
        gw, tw = _load_weights_from_json(ROOT / args.weights_json)
        global_weights = gw
        target_weights = tw

    # 2) Explicit CLI overrides JSON
    if args.weights is not None:
        if args.weights_t0 is not None or args.weights_t1 is not None:
            raise ValueError("Use either --weights OR --weights-t0/--weights-t1, not both")
        global_weights = args.weights
        target_weights = None

    if args.weights_t0 is not None or args.weights_t1 is not None:
        if args.weights_t0 is None or args.weights_t1 is None:
            raise ValueError("Provide both --weights-t0 and --weights-t1")
        target_weights = {
            "t0": args.weights_t0,
            "t1": args.weights_t1,
        }
        global_weights = None

    # 3) Normalize and validate
    if target_weights is not None:
        t0 = _normalize_weights(target_weights["t0"], n_models, "weights_t0")
        t1 = _normalize_weights(target_weights["t1"], n_models, "weights_t1")
        return [1.0 / n_models] * n_models, {"t0": t0, "t1": t1}

    if global_weights is None:
        global_weights = [1.0 / n_models] * n_models

    global_weights = _normalize_weights(global_weights, n_models, "weights")
    return global_weights, None


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _expected_input_size(data_cfg: dict) -> int:
    derived = bool(data_cfg.get("derived_features", False))
    temporal = bool(data_cfg.get("temporal_features", False) and derived)
    interaction = bool(data_cfg.get("interaction_features", False))

    n = 32
    if derived:
        n += 10
    if temporal:
        n += 3
    if interaction:
        n += 3
    microstructure = bool(data_cfg.get("microstructure_features", False))
    if microstructure:
        n += 6
    return n


def _prepare_model_specs(
    config_paths: List[Path],
    checkpoint_paths: List[Path],
    normalizer_paths: List[Path],
    strict_input_size: bool,
    use_onnx: bool = False,
) -> Tuple[List[dict], Dict[Path, str]]:
    model_specs: List[dict] = []

    # Deduplicate normalizers in zip
    unique_norm_names: Dict[Path, str] = {}

    for i, (cfg_path, ckpt_path, norm_path) in enumerate(
        zip(config_paths, checkpoint_paths, normalizer_paths)
    ):
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        if not norm_path.exists():
            raise FileNotFoundError(f"Normalizer not found: {norm_path}")

        cfg = _load_yaml(cfg_path)
        model_cfg = cfg.get("model", {})
        data_cfg = cfg.get("data", {})

        model_type = model_cfg.get("type", "gru")
        if model_type not in {"gru", "lstm", "gru_attention", "tcn"}:
            raise ValueError(f"Unsupported model type '{model_type}' in {cfg_path}")

        model_cfg_clean = {
            "type": model_type,
            "input_size": int(model_cfg.get("input_size", 32)),
            "hidden_size": int(model_cfg.get("hidden_size", 128)),
            "num_layers": int(model_cfg.get("num_layers", 2)),
            "dropout": float(model_cfg.get("dropout", 0.2)),
            "output_size": int(model_cfg.get("output_size", 2)),
            "attention_heads": int(model_cfg.get("attention_heads", 4)),
            "attention_dropout": float(model_cfg.get("attention_dropout", 0.1)),
            "attention_window": int(model_cfg.get("attention_window", 128)),
            "output_type": model_cfg.get("output_type", "mlp"),
            "vanilla": bool(model_cfg.get("vanilla", False)),
            # TCN-specific
            "hidden_channels": int(model_cfg.get("hidden_channels", 32)),
            "kernel_size": int(model_cfg.get("kernel_size", 3)),
            "dilations": model_cfg.get("dilations", [1, 2, 4, 8, 16, 32]),
        }

        # Use ONNX for GRU models (7x faster inference)
        is_onnx = use_onnx and model_type == "gru"
        if is_onnx:
            model_cfg_clean["inference"] = "onnx"

        derived = bool(data_cfg.get("derived_features", False))
        temporal = bool(data_cfg.get("temporal_features", False) and derived)
        interaction = bool(data_cfg.get("interaction_features", False))
        microstructure = bool(data_cfg.get("microstructure_features", False))

        data_cfg_clean = {
            "derived_features": derived,
            "temporal_features": temporal,
            "interaction_features": interaction,
            "microstructure_features": microstructure,
        }

        expected_size = _expected_input_size(data_cfg_clean)
        if strict_input_size and model_cfg_clean["input_size"] != expected_size:
            raise ValueError(
                f"Input size mismatch for model {i}: config has {model_cfg_clean['input_size']} "
                f"but preprocessing implies {expected_size} "
                f"(derived={derived}, temporal={temporal}, interaction={interaction})"
            )

        if norm_path not in unique_norm_names:
            norm_name = "normalizer.npz" if len(unique_norm_names) == 0 else f"normalizer_{len(unique_norm_names)}.npz"
            unique_norm_names[norm_path] = norm_name

        file_ext = ".onnx" if is_onnx else ".pt"
        model_specs.append(
            {
                "checkpoint": f"model_{i}{file_ext}",
                "checkpoint_src": str(ckpt_path),
                "is_onnx": is_onnx,
                "normalizer": unique_norm_names[norm_path],
                "normalizer_src": str(norm_path),
                "model": model_cfg_clean,
                "data": data_cfg_clean,
            }
        )

    return model_specs, unique_norm_names


def generate_ensemble_solution() -> str:
    """Generate optimized self-contained solution.py.

    Optimizations vs original:
    1. Feature caching: compute_derived once per step, cache normalized tensors
       by (data_config, normalizer) key so identical pipelines run once.
    2. Lazy prediction: skip output_proj (and attention) when need_prediction
       is False, saving compute on non-scoring steps.
    3. Single torch.no_grad() context wrapping the full model loop.
    4. Pre-allocated prediction array instead of list append + np.stack.
    """
    return """import json
import os

import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort


def _safe_torch_load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


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


def compute_interactions(features, has_derived=True):
    p0 = features[0]
    v2 = features[14]
    v8 = features[20]
    if has_derived and features.shape[0] >= 33:
        spread_0 = features[32]
    else:
        spread_0 = features[6] - features[0]
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

        return np.concatenate([
            features_42,
            np.array([roc1, roc5, roll_mean], dtype=np.float32),
        ]).astype(np.float32)


class MicrostructureBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.prev_p = None
        self.prev_v = None

    def compute_step(self, features, eps=1e-8):
        p = features[0:12].astype(np.float64)
        v = features[12:24].astype(np.float64)
        dp = features[24:28].astype(np.float64)
        dv = features[28:32].astype(np.float64)
        micro = np.zeros(6, dtype=np.float32)
        if self.prev_p is None:
            self.prev_p = p.copy()
            self.prev_v = v.copy()
        else:
            pp, pv = self.prev_p, self.prev_v
            ofi = np.zeros(6, dtype=np.float64)
            for l in range(6):
                eb = (v[l] if p[l] >= pp[l] else 0) - (pv[l] if p[l] <= pp[l] else 0)
                ea = (v[l+6] if p[l+6] <= pp[l+6] else 0) - (pv[l+6] if p[l+6] >= pp[l+6] else 0)
                ofi[l] = eb - ea
            dw = np.array([1.0 / (l + 1) for l in range(6)])
            micro[0] = float((ofi * dw).sum())
            micro[1] = float(ofi[:3].mean() - ofi[3:].mean())
            s0 = p[6] - p[0]; s0p = pp[6] - pp[0]; ds = s0 - s0p
            vi = (v[6] - v[0]) / (v[0] + v[6] + eps)
            micro[3] = float(ds * vi)
            s1 = p[7] - p[1]; s1p = pp[7] - pp[1]
            micro[4] = float((s1 - s0) - (s1p - s0p))
            self.prev_p = p.copy()
            self.prev_v = v.copy()
        qi = np.zeros(6, dtype=np.float64)
        for l in range(6):
            qi[l] = (v[l] - v[l+6]) / (v[l] + v[l+6] + eps)
        micro[2] = float(np.diff(qi).mean() * 5.0)
        micro[5] = float(np.sqrt((dp ** 2 * (1 + np.abs(dv))).sum() + eps))
        return np.concatenate([features, micro]).astype(np.float32)


class Normalizer:
    def __init__(self, path):
        data = np.load(path)
        self.mean = data['mean']
        self.std = data['std']

    def transform(self, x):
        return ((x - self.mean) / self.std).astype(np.float32)


class GRUModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.hidden_size = int(cfg.get('hidden_size', 128))
        self.num_layers = int(cfg.get('num_layers', 2))
        self.dropout = float(cfg.get('dropout', 0.2))
        self.vanilla = bool(cfg.get('vanilla', False))

        input_size = int(cfg.get('input_size', 32))
        output_size = int(cfg.get('output_size', 2))

        if not self.vanilla:
            self.input_proj = nn.Linear(input_size, self.hidden_size)
            self.input_norm = nn.LayerNorm(self.hidden_size)
            self.input_dropout = nn.Dropout(self.dropout)
            gru_input_size = self.hidden_size
        else:
            gru_input_size = input_size

        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            bidirectional=False,
        )

        output_type = cfg.get('output_type', 'mlp')
        if output_type == 'linear':
            self.output_proj = nn.Linear(self.hidden_size, output_size)
        else:
            self.output_proj = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size // 2, output_size),
            )

    def forward_step(self, x, hidden=None, need_pred=True):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.unsqueeze(1)

        if hidden is None:
            hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)

        if not self.vanilla:
            x = self.input_proj(x)
            x = self.input_norm(x)
            x = self.input_dropout(x)

        out, hidden = self.gru(x, hidden)
        if not need_pred:
            return None, hidden
        pred = self.output_proj(out.squeeze(1))
        return pred, hidden


class LSTMModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.hidden_size = int(cfg.get('hidden_size', 128))
        self.num_layers = int(cfg.get('num_layers', 2))
        self.dropout = float(cfg.get('dropout', 0.2))

        input_size = int(cfg.get('input_size', 32))
        output_size = int(cfg.get('output_size', 2))

        self.input_proj = nn.Linear(input_size, self.hidden_size)
        self.input_norm = nn.LayerNorm(self.hidden_size)
        self.input_dropout = nn.Dropout(self.dropout)

        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, output_size),
        )

    def forward_step(self, x, hidden=None, need_pred=True):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.unsqueeze(1)

        if hidden is None:
            h = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
            c = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
            hidden = (h, c)

        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)

        out, hidden = self.lstm(x, hidden)
        if not need_pred:
            return None, hidden
        pred = self.output_proj(out.squeeze(1))
        return pred, hidden


class GRUAttentionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.hidden_size = int(cfg.get('hidden_size', 144))
        self.num_layers = int(cfg.get('num_layers', 2))
        self.dropout = float(cfg.get('dropout', 0.22))
        self.attention_heads = int(cfg.get('attention_heads', 4))
        self.attention_dropout = float(cfg.get('attention_dropout', 0.1))
        self.attention_window = int(cfg.get('attention_window', 128))

        input_size = int(cfg.get('input_size', 42))
        output_size = int(cfg.get('output_size', 2))

        self.input_proj = nn.Linear(input_size, self.hidden_size)
        self.input_norm = nn.LayerNorm(self.hidden_size)
        self.input_dropout = nn.Dropout(self.dropout)

        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.attention_heads,
            dropout=self.attention_dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(self.hidden_size)

        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, output_size),
        )

    def forward_step(self, x, hidden=None, need_pred=True):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.unsqueeze(1)

        if hidden is None:
            gru_hidden = None
            attn_buffer = None
        else:
            gru_hidden, attn_buffer = hidden

        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)

        gru_out, gru_hidden = self.gru(x, gru_hidden)
        cur = gru_out.squeeze(1)

        if attn_buffer is None:
            context = cur.unsqueeze(1)
        else:
            context = torch.cat([attn_buffer, cur.unsqueeze(1)], dim=1)
            if context.size(1) > self.attention_window:
                context = context[:, -self.attention_window:, :]

        if not need_pred:
            return None, (gru_hidden, context.detach())

        q = cur.unsqueeze(1)
        attn_out, _ = self.attn(q, context, context, need_weights=False)
        fused = self.attn_norm(cur + attn_out.squeeze(1))
        pred = self.output_proj(fused)
        return pred, (gru_hidden, context.detach())


class TCNResBlock(nn.Module):
    def __init__(self, ch, ks, dil):
        super().__init__()
        self.ch = ch
        self.ks = ks
        self.dil = dil
        self.buf_len = (ks - 1) * dil + 1
        self.dw_conv = nn.Conv1d(ch, ch, ks, dilation=dil, groups=ch, bias=True)
        self.activation = nn.SiLU()
        self.pw_conv = nn.Conv1d(ch, ch, 1, bias=True)

    def forward_step(self, x, buf, ptr):
        buf[:, :, ptr] = x
        taps = []
        for i in range(self.ks):
            taps.append(buf[:, :, (ptr - i * self.dil) % self.buf_len])
        stack = torch.stack(list(reversed(taps)), dim=2)
        out = torch.nn.functional.conv1d(
            stack, self.dw_conv.weight, self.dw_conv.bias, groups=self.ch
        ).squeeze(2)
        out = self.activation(out)
        out = torch.nn.functional.conv1d(
            out.unsqueeze(2), self.pw_conv.weight, self.pw_conv.bias
        ).squeeze(2)
        return out + x, buf, (ptr + 1) % self.buf_len


class TCNModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ch = int(cfg.get('hidden_channels', 32))
        ks = int(cfg.get('kernel_size', 3))
        dils = cfg.get('dilations', [1, 2, 4, 8, 16, 32])
        input_size = int(cfg.get('input_size', 42))
        output_size = int(cfg.get('output_size', 2))
        self.ch = ch
        self.input_proj = nn.Linear(input_size, ch)
        self.blocks = nn.ModuleList([TCNResBlock(ch, ks, d) for d in dils])
        self.output_norm = nn.LayerNorm(ch)
        self.output_head = nn.Linear(ch, output_size)

    def forward_step(self, x, hidden=None, need_pred=True):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        bs = x.shape[0]
        if hidden is None:
            hidden = [
                (torch.zeros(bs, self.ch, b.buf_len, device=x.device), 0)
                for b in self.blocks
            ]
        x = self.input_proj(x)
        new_hidden = []
        for i, block in enumerate(self.blocks):
            buf, ptr = hidden[i]
            x, buf, new_ptr = block.forward_step(x, buf, ptr)
            new_hidden.append((buf, new_ptr))
        if not need_pred:
            return None, new_hidden
        x = self.output_norm(x)
        pred = self.output_head(x)
        return pred, new_hidden


class TCNFast:
    \"\"\"Numpy-only TCN inference — eliminates PyTorch dispatch overhead.\"\"\"

    def __init__(self, pt):
        ch = pt.ch
        self.ch = ch
        self.inp_w = pt.input_proj.weight.data.numpy().copy()
        self.inp_b = pt.input_proj.bias.data.numpy().copy()
        self.n_blocks = len(pt.blocks)
        self.dw_w = []; self.dw_b = []; self.pw_w = []; self.pw_b = []
        self.buf_lens = []; self.dils = []; self.ks_list = []
        for b in pt.blocks:
            self.dw_w.append(b.dw_conv.weight.data.numpy().reshape(ch, b.ks).copy())
            self.dw_b.append(b.dw_conv.bias.data.numpy().copy())
            self.pw_w.append(b.pw_conv.weight.data.numpy().reshape(ch, ch).copy())
            self.pw_b.append(b.pw_conv.bias.data.numpy().copy())
            self.buf_lens.append(b.buf_len)
            self.dils.append(b.dil)
            self.ks_list.append(b.ks)
        self.norm_w = pt.output_norm.weight.data.numpy().copy()
        self.norm_b = pt.output_norm.bias.data.numpy().copy()
        self.head_w = pt.output_head.weight.data.numpy().copy()
        self.head_b = pt.output_head.bias.data.numpy().copy()

    def eval(self):
        return self

    def forward_step(self, x_tensor, hidden=None, need_pred=True):
        x = x_tensor.numpy().ravel() if hasattr(x_tensor, 'numpy') else np.asarray(x_tensor, dtype=np.float32).ravel()
        h = self.inp_w @ x + self.inp_b
        if hidden is None:
            hidden = [(np.zeros((self.ch, bl), dtype=np.float32), 0) for bl in self.buf_lens]
        new_hidden = []
        for i in range(self.n_blocks):
            buf, ptr = hidden[i]
            buf[:, ptr] = h
            ks = self.ks_list[i]; dil = self.dils[i]; bl = self.buf_lens[i]
            taps = np.empty((self.ch, ks), dtype=np.float32)
            for j in range(ks):
                taps[:, j] = buf[:, (ptr - (ks - 1 - j) * dil) % bl]
            out = (taps * self.dw_w[i]).sum(1) + self.dw_b[i]
            out = out / (1.0 + np.exp(-np.clip(out, -15, 15)))
            out = self.pw_w[i] @ out + self.pw_b[i]
            h = out + h
            new_hidden.append((buf, (ptr + 1) % bl))
        if not need_pred:
            return None, new_hidden
        m = h.mean(); v = ((h - m) ** 2).mean()
        h = (h - m) / np.sqrt(v + 1e-5) * self.norm_w + self.norm_b
        pred = self.head_w @ h + self.head_b
        return pred, new_hidden


class OnnxGRU:
    \"\"\"ONNX Runtime GRU for step-by-step inference — 7x faster than PyTorch.\"\"\"

    def __init__(self, onnx_path, hidden_size, num_layers):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])

    def eval(self):
        return self

    def forward_step(self, x, hidden=None, need_pred=True):
        if hidden is None:
            hidden = np.zeros((self.num_layers, 1, self.hidden_size), dtype=np.float32)
        x_np = x.numpy().reshape(1, -1) if hasattr(x, 'numpy') else np.asarray(x, dtype=np.float32).reshape(1, -1)
        h_np = hidden if isinstance(hidden, np.ndarray) else hidden.numpy()
        prediction, new_hidden = self.sess.run(
            ['prediction', 'hidden_out'],
            {'input': x_np, 'hidden_in': h_np}
        )
        if not need_pred:
            return None, new_hidden
        return prediction, new_hidden


def build_model(model_cfg):
    model_type = model_cfg.get('type', 'gru')
    if model_type == 'gru':
        return GRUModel(model_cfg)
    if model_type == 'lstm':
        return LSTMModel(model_cfg)
    if model_type == 'gru_attention':
        return GRUAttentionModel(model_cfg)
    if model_type == 'tcn':
        return TCNModel(model_cfg)
    raise ValueError(f'Unsupported model type: {model_type}')


class PredictionModel:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(base_dir, 'ensemble_config.json'), 'r', encoding='utf-8') as f:
            self.cfg = json.load(f)

        self.models = []
        self.hiddens = []
        self.normalizers = []
        self.data_cfgs = []
        self.temporal_buffers = []
        self.micro_buffers = []

        for spec in self.cfg['models']:
            inference = spec['model'].get('inference', 'pytorch')
            if inference == 'onnx':
                onnx_path = os.path.join(base_dir, spec['checkpoint'])
                model = OnnxGRU(
                    onnx_path,
                    hidden_size=int(spec['model']['hidden_size']),
                    num_layers=int(spec['model']['num_layers']),
                )
            else:
                model = build_model(spec['model'])
                ckpt = _safe_torch_load(os.path.join(base_dir, spec['checkpoint']))
                model.load_state_dict(ckpt['model_state_dict'])
                model.eval()

            self.models.append(model)
            self.hiddens.append(None)
            self.normalizers.append(Normalizer(os.path.join(base_dir, spec['normalizer'])))
            self.data_cfgs.append(spec['data'])

            if bool(spec['data'].get('temporal_features', False)):
                self.temporal_buffers.append(TemporalBuffer())
            else:
                self.temporal_buffers.append(None)

            if bool(spec['data'].get('microstructure_features', False)):
                self.micro_buffers.append(MicrostructureBuffer())
            else:
                self.micro_buffers.append(None)

        for i, spec in enumerate(self.cfg['models']):
            if spec['model'].get('type') == 'tcn' and spec['model'].get('inference') != 'onnx':
                self.models[i] = TCNFast(self.models[i])

        self.n_models = len(self.models)

        self.weights = np.array(
            self.cfg.get('weights', [1.0 / self.n_models] * self.n_models),
            dtype=np.float32,
        )

        target_weights = self.cfg.get('target_weights', None)
        if target_weights is None:
            self.target_weights = None
        else:
            self.target_weights = {
                't0': np.array(target_weights['t0'], dtype=np.float32),
                't1': np.array(target_weights['t1'], dtype=np.float32),
            }

        self.current_seq_ix = None

        # Pre-compute feature cache keys per model.
        # Models with identical (derived, temporal, interaction, normalizer)
        # produce identical input tensors and can share cached results.
        # Temporal models are never cached (stateful per-model buffer).
        self._feat_cache_keys = []
        for spec in self.cfg['models']:
            dcfg = spec['data']
            if bool(dcfg.get('temporal_features', False)) or bool(dcfg.get('microstructure_features', False)):
                self._feat_cache_keys.append(None)
            else:
                self._feat_cache_keys.append((
                    bool(dcfg.get('derived_features', False)),
                    bool(dcfg.get('interaction_features', False)),
                    spec['normalizer'],
                ))

    def predict(self, data_point):
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.hiddens = [None] * self.n_models
            for tb in self.temporal_buffers:
                if tb is not None:
                    tb.reset()
            for mb in self.micro_buffers:
                if mb is not None:
                    mb.reset()

        raw = data_point.state
        need_pred = data_point.need_prediction

        # Pre-compute shared raw features once
        raw_f32 = raw.astype(np.float32)
        raw_with_derived = None
        feat_tensor_cache = {}
        pred_arr = np.empty((self.n_models, 2), dtype=np.float32)

        with torch.no_grad():
            for i in range(self.n_models):
                cache_key = self._feat_cache_keys[i]
                if cache_key is not None and cache_key in feat_tensor_cache:
                    x = feat_tensor_cache[cache_key]
                else:
                    cfg = self.data_cfgs[i]
                    use_derived = bool(cfg.get('derived_features', False))
                    use_temporal = bool(cfg.get('temporal_features', False))
                    use_interaction = bool(cfg.get('interaction_features', False))
                    use_micro = bool(cfg.get('microstructure_features', False))

                    if use_derived:
                        if raw_with_derived is None:
                            raw_with_derived = np.concatenate([raw_f32, compute_derived(raw_f32)])
                        features = raw_with_derived
                    else:
                        features = raw_f32

                    if use_temporal:
                        features = self.temporal_buffers[i].compute_step(features.copy())

                    if use_interaction:
                        interactions = compute_interactions(features, has_derived=use_derived)
                        features = np.concatenate([features, interactions])

                    if use_micro:
                        features = self.micro_buffers[i].compute_step(features.copy())

                    features = self.normalizers[i].transform(features.reshape(1, -1)).squeeze(0)
                    x = torch.from_numpy(features)

                    if cache_key is not None:
                        feat_tensor_cache[cache_key] = x

                pred, self.hiddens[i] = self.models[i].forward_step(
                    x, self.hiddens[i], need_pred=need_pred,
                )
                if need_pred:
                    if isinstance(pred, np.ndarray):
                        pred_arr[i] = pred
                    else:
                        pred_arr[i] = pred.squeeze(0).numpy()

        if not need_pred:
            return None

        if self.target_weights is not None:
            p0 = float(np.dot(self.target_weights['t0'], pred_arr[:, 0]))
            p1 = float(np.dot(self.target_weights['t1'], pred_arr[:, 1]))
            out = np.array([p0, p1], dtype=np.float32)
        else:
            out = (pred_arr * self.weights.reshape(-1, 1)).sum(axis=0).astype(np.float32)

        out = np.clip(out, -6, 6)
        return out
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ensemble submission zip")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Single config for homogeneous ensemble",
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=None,
        help="Per-model configs for heterogeneous ensemble",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="Model checkpoints",
    )
    parser.add_argument(
        "--normalizer",
        type=str,
        default="logs/normalizer.npz",
        help="Shared normalizer path",
    )
    parser.add_argument(
        "--normalizers",
        type=str,
        nargs="+",
        default=None,
        help="Per-model normalizer paths",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="Global model weights (same for t0/t1)",
    )
    parser.add_argument(
        "--weights-t0",
        type=float,
        nargs="+",
        default=None,
        help="Per-model weights for target t0",
    )
    parser.add_argument(
        "--weights-t1",
        type=float,
        nargs="+",
        default=None,
        help="Per-model weights for target t1",
    )
    parser.add_argument(
        "--weights-json",
        type=str,
        default=None,
        help="Optional JSON with weights, e.g. notebooks/artifacts/.../optimal_weights.json",
    )
    parser.add_argument(
        "--no-strict-input-size",
        action="store_true",
        help="Disable strict validation of model.input_size vs preprocessing flags",
    )
    parser.add_argument(
        "--onnx",
        action="store_true",
        help="Export GRU models to ONNX for 7x faster inference",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submissions/ensemble.zip",
        help="Output zip path",
    )
    args = parser.parse_args()

    checkpoint_paths = [ROOT / p for p in args.checkpoints]
    n_models = len(checkpoint_paths)

    config_paths = _resolve_config_paths(args, n_models)
    normalizer_paths = _resolve_normalizer_paths(args, n_models)
    weights, target_weights = _resolve_weights(args, n_models)

    model_specs, unique_norm_names = _prepare_model_specs(
        config_paths=config_paths,
        checkpoint_paths=checkpoint_paths,
        normalizer_paths=normalizer_paths,
        strict_input_size=not args.no_strict_input_size,
        use_onnx=args.onnx,
    )

    ensemble_cfg = {
        "models": [
            {
                "checkpoint": m["checkpoint"],
                "normalizer": m["normalizer"],
                "model": m["model"],
                "data": m["data"],
            }
            for m in model_specs
        ],
        "weights": weights,
        "target_weights": target_weights,
    }

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    solution_code = generate_ensemble_solution()

    # Create temp directory for ONNX exports
    tmpdir = tempfile.mkdtemp() if args.onnx else None
    n_onnx = 0

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("solution.py", solution_code)
        zf.writestr("ensemble_config.json", json.dumps(ensemble_cfg, indent=2))

        # Write unique normalizers
        for norm_path, norm_name in unique_norm_names.items():
            zf.write(norm_path, norm_name)

        # Write checkpoints (or ONNX exports)
        for m in model_specs:
            if m.get("is_onnx"):
                onnx_file = os.path.join(tmpdir, m["checkpoint"])
                print(f"  Exporting {m['checkpoint']} to ONNX...")
                _export_gru_to_onnx(
                    Path(m["checkpoint_src"]),
                    m["model"],
                    Path(onnx_file),
                )
                zf.write(onnx_file, m["checkpoint"])
                n_onnx += 1
            else:
                zf.write(m["checkpoint_src"], m["checkpoint"])

    # Clean up temp dir
    if tmpdir:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    size_kb = out_path.stat().st_size / 1024.0
    print(f"Ensemble submission exported: {out_path}")
    print(f"  Models: {n_models} ({n_onnx} ONNX, {n_models - n_onnx} PyTorch/TCNFast)")
    print(f"  Global weights: {weights}")
    if target_weights is not None:
        print(f"  Per-target weights t0: {target_weights['t0']}")
        print(f"  Per-target weights t1: {target_weights['t1']}")
    print(f"  Unique normalizers: {len(unique_norm_names)}")
    print(f"  Size: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
