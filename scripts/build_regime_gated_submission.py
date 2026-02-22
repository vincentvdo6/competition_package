#!/usr/bin/env python
"""Build a regime-gated vanilla GRU ONNX submission.

This creates a two-expert submission:
1) Base ensemble (stable vanilla GRU seeds)
2) Specialist ensemble (alternate recipe/checkpoints)

At inference time, a lightweight gate is computed once per sequence from the
warmup region (first N steps, default 99). The gate uses a logistic classifier
trained on train-vs-valid sequence summaries and chooses specialist weight via
piecewise rule:
  if p_val_like < threshold: use gate_weight_low_(t0/t1)
  else:                      use gate_weight_high_(t0/t1)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

FEATURE_COLS = (
    [f"p{i}" for i in range(12)]
    + [f"v{i}" for i in range(12)]
    + [f"dp{i}" for i in range(4)]
    + [f"dv{i}" for i in range(4)]
)


class VanillaGRUStep(nn.Module):
    """Minimal wrapper for ONNX export: raw features -> GRU -> linear."""

    def __init__(self, gru: nn.GRU, fc: nn.Linear):
        super().__init__()
        self.gru = gru
        self.fc = fc

    def forward(self, x: torch.Tensor, hidden: torch.Tensor):
        x = x.unsqueeze(1)  # (1, 32) -> (1, 1, 32)
        out, new_hidden = self.gru(x, hidden)
        return self.fc(out.squeeze(1)), new_hidden


def export_vanilla_to_onnx(
    ckpt_path: str,
    hidden_size: int,
    num_layers: int,
    onnx_path: str,
) -> None:
    """Export a vanilla GRU checkpoint to ONNX step model."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]

    gru = nn.GRU(
        input_size=32,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
        dropout=0.0,
        bidirectional=False,
    )
    fc = nn.Linear(hidden_size, 2)

    gru_sd = {
        k.replace("gru.", ""): v
        for k, v in state_dict.items()
        if k.startswith("gru.")
    }
    fc_sd = {
        k.replace("output_proj.", ""): v
        for k, v in state_dict.items()
        if k.startswith("output_proj.")
    }
    gru.load_state_dict(gru_sd)
    fc.load_state_dict(fc_sd)

    wrapper = VanillaGRUStep(gru, fc)
    wrapper.eval()

    dummy_x = torch.randn(1, 32)
    dummy_h = torch.zeros(num_layers, 1, hidden_size)

    torch.onnx.export(
        wrapper,
        (dummy_x, dummy_h),
        onnx_path,
        input_names=["input", "hidden_in"],
        output_names=["prediction", "hidden_out"],
        dynamic_axes=None,
        opset_version=17,
        do_constant_folding=True,
    )


def compute_sequence_summaries(df: pd.DataFrame, warmup_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-sequence summary stats on warmup window (mean/std/min/max)."""
    df_warm = df[df["step_in_seq"] < warmup_steps]
    grouped = df_warm.groupby("seq_ix")[FEATURE_COLS]

    means = grouped.mean().values
    stds = grouped.std().fillna(0.0).values
    mins = grouped.min().values
    maxs = grouped.max().values
    seq_ids = grouped.mean().index.values

    summaries = np.concatenate([means, stds, mins, maxs], axis=1).astype(np.float32)
    return seq_ids, summaries


def fit_gate_from_data(
    train_path: str,
    valid_path: str,
    warmup_steps: int,
) -> dict:
    """Train train-vs-valid logistic gate on warmup sequence summaries."""
    cols = ["seq_ix", "step_in_seq"] + FEATURE_COLS

    print(f"Loading train data: {train_path}")
    train_df = pd.read_parquet(train_path, columns=cols)
    print(f"Loading valid data: {valid_path}")
    valid_df = pd.read_parquet(valid_path, columns=cols)

    print(f"Computing warmup summaries (steps < {warmup_steps})...")
    train_ids, train_X = compute_sequence_summaries(train_df, warmup_steps)
    valid_ids, valid_X = compute_sequence_summaries(valid_df, warmup_steps)

    X = np.vstack([train_X, valid_X])
    y = np.concatenate([np.zeros(len(train_X)), np.ones(len(valid_X))])

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    clf.fit(Xs, y)

    probs_all = clf.predict_proba(Xs)[:, 1]
    auc = roc_auc_score(y, probs_all)
    valid_probs = clf.predict_proba(scaler.transform(valid_X))[:, 1]

    print(f"Gate train-vs-valid AUC (in-sample): {auc:.4f}")
    print(
        "Valid prob stats: "
        f"min={valid_probs.min():.4f}, "
        f"mean={valid_probs.mean():.4f}, "
        f"max={valid_probs.max():.4f}"
    )

    quantile_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    quantiles = {str(q): float(np.quantile(valid_probs, q)) for q in quantile_grid}
    print("Valid prob quantiles:")
    for q in quantile_grid:
        print(f"  q{int(q * 100):02d}: {quantiles[str(q)]:.6f}")

    return {
        "warmup_steps": int(warmup_steps),
        "feature_order": FEATURE_COLS,
        "scaler_mean": scaler.mean_.astype(np.float32).tolist(),
        "scaler_scale": scaler.scale_.astype(np.float32).tolist(),
        "coef": clf.coef_[0].astype(np.float32).tolist(),
        "intercept": float(clf.intercept_[0]),
        "auc_in_sample": float(auc),
        "valid_prob_quantiles": quantiles,
        "n_train_sequences": int(len(train_ids)),
        "n_valid_sequences": int(len(valid_ids)),
        "dim": int(train_X.shape[1]),
    }


SOLUTION_TEMPLATE = '''"""Regime-gated vanilla GRU ONNX ensemble."""

import os
import numpy as np
import onnxruntime as ort


# Base and specialist model configs: (filename, hidden_size, num_layers, weight)
BASE_MODELS = {base_models}
SPECIAL_MODELS = {special_models}

# Gate parameters (logistic train-vs-valid on warmup sequence summaries)
GATE_SCALER_MEAN = np.array({gate_scaler_mean}, dtype=np.float32)
GATE_SCALER_SCALE = np.array({gate_scaler_scale}, dtype=np.float32)
GATE_COEF = np.array({gate_coef}, dtype=np.float32)
GATE_INTERCEPT = float({gate_intercept})

WARMUP_STEPS = int({warmup_steps})
GATE_THRESHOLD = float({gate_threshold})
GATE_WEIGHT_T0_LOW = float({gate_weight_t0_low})
GATE_WEIGHT_T1_LOW = float({gate_weight_t1_low})
GATE_WEIGHT_T0_HIGH = float({gate_weight_t0_high})
GATE_WEIGHT_T1_HIGH = float({gate_weight_t1_high})
DEFAULT_EXPERT_WEIGHT_T0 = float({default_expert_weight_t0})
DEFAULT_EXPERT_WEIGHT_T1 = float({default_expert_weight_t1})


class OnnxVanillaGRU:
    def __init__(self, onnx_path, hidden_size, num_layers):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            onnx_path, opts, providers=["CPUExecutionProvider"]
        )

    def init_hidden(self):
        return np.zeros((self.num_layers, 1, self.hidden_size), dtype=np.float32)

    def run_step(self, x_np, hidden):
        pred, new_hidden = self.sess.run(
            ["prediction", "hidden_out"],
            {{"input": x_np, "hidden_in": hidden}},
        )
        return pred[0], new_hidden


class RunningSummary:
    """Running mean/std/min/max for 32-dim warmup features."""

    def __init__(self, dim):
        self.dim = dim
        self.reset()

    def reset(self):
        self.count = 0
        self.sum = np.zeros(self.dim, dtype=np.float64)
        self.sumsq = np.zeros(self.dim, dtype=np.float64)
        self.min = np.full(self.dim, np.inf, dtype=np.float64)
        self.max = np.full(self.dim, -np.inf, dtype=np.float64)

    def update(self, x):
        x64 = x.astype(np.float64, copy=False)
        self.count += 1
        self.sum += x64
        self.sumsq += x64 * x64
        self.min = np.minimum(self.min, x64)
        self.max = np.maximum(self.max, x64)

    def feature_vector(self):
        if self.count <= 0:
            mean = np.zeros(self.dim, dtype=np.float32)
            std = np.zeros(self.dim, dtype=np.float32)
            mins = np.zeros(self.dim, dtype=np.float32)
            maxs = np.zeros(self.dim, dtype=np.float32)
        else:
            mean64 = self.sum / self.count
            var64 = np.maximum(self.sumsq / self.count - mean64 * mean64, 1e-8)
            std64 = np.sqrt(var64)
            mean = mean64.astype(np.float32)
            std = std64.astype(np.float32)
            mins = self.min.astype(np.float32)
            maxs = self.max.astype(np.float32)
        return np.concatenate([mean, std, mins, maxs], axis=0)


def compute_expert_weights(summary_vec):
    x = (summary_vec - GATE_SCALER_MEAN) / (GATE_SCALER_SCALE + 1e-8)
    z = float(np.dot(GATE_COEF, x) + GATE_INTERCEPT)
    z = max(min(z, 60.0), -60.0)
    p_val_like = 1.0 / (1.0 + np.exp(-z))
    if p_val_like < GATE_THRESHOLD:
        return GATE_WEIGHT_T0_LOW, GATE_WEIGHT_T1_LOW
    return GATE_WEIGHT_T0_HIGH, GATE_WEIGHT_T1_HIGH


class PredictionModel:
    def __init__(self, model_path=""):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        self.base_models = []
        self.base_hiddens = []
        self.base_weights = []
        for filename, h, nl, w in BASE_MODELS:
            model = OnnxVanillaGRU(os.path.join(base_dir, filename), h, nl)
            self.base_models.append(model)
            self.base_hiddens.append(model.init_hidden())
            self.base_weights.append(w)
        self.base_weights = np.array(self.base_weights, dtype=np.float32)
        self.base_weights /= np.sum(self.base_weights)

        self.special_models = []
        self.special_hiddens = []
        self.special_weights = []
        for filename, h, nl, w in SPECIAL_MODELS:
            model = OnnxVanillaGRU(os.path.join(base_dir, filename), h, nl)
            self.special_models.append(model)
            self.special_hiddens.append(model.init_hidden())
            self.special_weights.append(w)
        self.special_weights = np.array(self.special_weights, dtype=np.float32)
        self.special_weights /= np.sum(self.special_weights)

        self.prev_seq_ix = None
        self.summary = RunningSummary(dim=32)
        self.gate_ready = False
        self.expert_weight_t0 = DEFAULT_EXPERT_WEIGHT_T0
        self.expert_weight_t1 = DEFAULT_EXPERT_WEIGHT_T1

    def _reset_sequence(self):
        self.base_hiddens = [m.init_hidden() for m in self.base_models]
        self.special_hiddens = [m.init_hidden() for m in self.special_models]
        self.summary.reset()
        self.gate_ready = False
        self.expert_weight_t0 = DEFAULT_EXPERT_WEIGHT_T0
        self.expert_weight_t1 = DEFAULT_EXPERT_WEIGHT_T1

    def predict(self, data_point):
        seq_ix = data_point.seq_ix
        if seq_ix != self.prev_seq_ix:
            self._reset_sequence()
            self.prev_seq_ix = seq_ix

        raw = data_point.state.astype(np.float32)[:32]
        self.summary.update(raw)

        # Gate becomes available as soon as warmup window is collected.
        if (not self.gate_ready) and (data_point.step_in_seq >= (WARMUP_STEPS - 1)):
            self.expert_weight_t0, self.expert_weight_t1 = compute_expert_weights(
                self.summary.feature_vector()
            )
            self.gate_ready = True

        x_np = raw.reshape(1, -1)

        base_pred = np.zeros(2, dtype=np.float32)
        for i, model in enumerate(self.base_models):
            pred, self.base_hiddens[i] = model.run_step(x_np, self.base_hiddens[i])
            base_pred += self.base_weights[i] * pred

        special_pred = np.zeros(2, dtype=np.float32)
        for i, model in enumerate(self.special_models):
            pred, self.special_hiddens[i] = model.run_step(x_np, self.special_hiddens[i])
            special_pred += self.special_weights[i] * pred

        pred = np.empty(2, dtype=np.float32)
        pred[0] = (
            (1.0 - self.expert_weight_t0) * base_pred[0]
            + self.expert_weight_t0 * special_pred[0]
        )
        pred[1] = (
            (1.0 - self.expert_weight_t1) * base_pred[1]
            + self.expert_weight_t1 * special_pred[1]
        )

        if not data_point.need_prediction:
            return None
        return pred.clip(-6, 6)
'''


def parse_weights(raw_weights: list[float] | None, n_models: int, label: str) -> list[float]:
    if raw_weights is None or len(raw_weights) == 0:
        return [1.0] * n_models
    if len(raw_weights) != n_models:
        raise ValueError(f"{label} weights mismatch: got {len(raw_weights)} for {n_models} models")
    return list(raw_weights)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build regime-gated vanilla GRU ONNX submission")
    parser.add_argument("--base-checkpoints", nargs="+", required=True, help="Base ensemble checkpoints (.pt)")
    parser.add_argument("--expert-checkpoints", nargs="+", required=True, help="Specialist ensemble checkpoints (.pt)")
    parser.add_argument("--base-weights", nargs="*", type=float, default=None, help="Base model weights")
    parser.add_argument("--expert-weights", nargs="*", type=float, default=None, help="Specialist model weights")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden size (all models)")
    parser.add_argument("--num-layers", type=int, default=3, help="GRU layers (all models)")
    parser.add_argument("--train-data", default="datasets/train.parquet", help="Train parquet for gate fitting")
    parser.add_argument("--valid-data", default="datasets/valid.parquet", help="Valid parquet for gate fitting")
    parser.add_argument("--warmup-steps", type=int, default=99, help="Warmup window for gate stats")
    parser.add_argument("--gate-threshold", type=float, default=0.50, help="Threshold on p_val_like")
    parser.add_argument("--gate-weight-low", type=float, default=0.10, help="Default specialist weight when p<threshold (used if per-target weights are omitted)")
    parser.add_argument("--gate-weight-high", type=float, default=0.00, help="Default specialist weight when p>=threshold (used if per-target weights are omitted)")
    parser.add_argument("--gate-weight-low-t0", type=float, default=None, help="Specialist weight for t0 when p<threshold")
    parser.add_argument("--gate-weight-low-t1", type=float, default=None, help="Specialist weight for t1 when p<threshold")
    parser.add_argument("--gate-weight-high-t0", type=float, default=None, help="Specialist weight for t0 when p>=threshold")
    parser.add_argument("--gate-weight-high-t1", type=float, default=None, help="Specialist weight for t1 when p>=threshold")
    parser.add_argument("--default-expert-weight", type=float, default=0.00, help="Default fallback specialist weight before gate is ready (used if per-target defaults are omitted)")
    parser.add_argument("--default-expert-weight-t0", type=float, default=None, help="Fallback specialist weight for t0 before gate is ready")
    parser.add_argument("--default-expert-weight-t1", type=float, default=None, help="Fallback specialist weight for t1 before gate is ready")
    parser.add_argument("--gate-params-json", default=None, help="Load gate params from JSON instead of fitting")
    parser.add_argument("--save-gate-params", default=None, help="Save fitted gate params JSON")
    parser.add_argument("--output", default="submissions/regime_gated_v1.zip", help="Output zip path")
    args = parser.parse_args()

    base_weights = parse_weights(args.base_weights, len(args.base_checkpoints), "base")
    expert_weights = parse_weights(args.expert_weights, len(args.expert_checkpoints), "expert")

    gate_weight_t0_low = args.gate_weight_low if args.gate_weight_low_t0 is None else args.gate_weight_low_t0
    gate_weight_t1_low = args.gate_weight_low if args.gate_weight_low_t1 is None else args.gate_weight_low_t1
    gate_weight_t0_high = args.gate_weight_high if args.gate_weight_high_t0 is None else args.gate_weight_high_t0
    gate_weight_t1_high = args.gate_weight_high if args.gate_weight_high_t1 is None else args.gate_weight_high_t1
    default_expert_weight_t0 = args.default_expert_weight if args.default_expert_weight_t0 is None else args.default_expert_weight_t0
    default_expert_weight_t1 = args.default_expert_weight if args.default_expert_weight_t1 is None else args.default_expert_weight_t1

    print(f"Building regime-gated ONNX submission")
    print(f"  Base models: {len(args.base_checkpoints)}")
    print(f"  Specialist models: {len(args.expert_checkpoints)}")
    print(
        f"  Gate threshold: p_val_like < {args.gate_threshold:.4f}\n"
        f"    low weights  (t0/t1): {gate_weight_t0_low:.4f} / {gate_weight_t1_low:.4f}\n"
        f"    high weights (t0/t1): {gate_weight_t0_high:.4f} / {gate_weight_t1_high:.4f}\n"
        f"    default      (t0/t1): {default_expert_weight_t0:.4f} / {default_expert_weight_t1:.4f}"
    )

    # Gate params
    if args.gate_params_json:
        with open(args.gate_params_json, "r", encoding="utf-8") as f:
            gate = json.load(f)
        print(f"Loaded gate params: {args.gate_params_json}")
    else:
        gate = fit_gate_from_data(args.train_data, args.valid_data, args.warmup_steps)
        if args.save_gate_params:
            out_gate = Path(args.save_gate_params)
            out_gate.parent.mkdir(parents=True, exist_ok=True)
            with open(out_gate, "w", encoding="utf-8") as f:
                json.dump(gate, f, indent=2)
            print(f"Saved gate params: {out_gate}")

    # Show member checkpoint scores
    for i, ckpt_path in enumerate(args.base_checkpoints):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        print(
            f"  [B{i}] {os.path.basename(ckpt_path)} "
            f"val={ckpt.get('best_score', 0):.4f} epoch={ckpt.get('best_epoch', '?')} "
            f"w={base_weights[i]:.3f}"
        )
    for i, ckpt_path in enumerate(args.expert_checkpoints):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        print(
            f"  [S{i}] {os.path.basename(ckpt_path)} "
            f"val={ckpt.get('best_score', 0):.4f} epoch={ckpt.get('best_epoch', '?')} "
            f"w={expert_weights[i]:.3f}"
        )

    n_models = len(args.base_checkpoints) + len(args.expert_checkpoints)
    est_total = n_models * 74
    margin = 1.0 - est_total / 4200
    print(f"Estimated runtime: {n_models} x 74s = {est_total}s ({margin * 100:.0f}% margin)")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Export ONNX models
        base_models = []
        for i, ckpt_path in enumerate(args.base_checkpoints):
            filename = f"base_{i}.onnx"
            onnx_path = os.path.join(tmpdir, filename)
            print(f"  Exporting {filename} ...", end=" ", flush=True)
            export_vanilla_to_onnx(ckpt_path, args.hidden_size, args.num_layers, onnx_path)
            print(f"({os.path.getsize(onnx_path) / 1024:.0f}KB)")
            base_models.append((filename, args.hidden_size, args.num_layers, float(base_weights[i])))

        special_models = []
        for i, ckpt_path in enumerate(args.expert_checkpoints):
            filename = f"special_{i}.onnx"
            onnx_path = os.path.join(tmpdir, filename)
            print(f"  Exporting {filename} ...", end=" ", flush=True)
            export_vanilla_to_onnx(ckpt_path, args.hidden_size, args.num_layers, onnx_path)
            print(f"({os.path.getsize(onnx_path) / 1024:.0f}KB)")
            special_models.append((filename, args.hidden_size, args.num_layers, float(expert_weights[i])))

        # Write solution.py
        solution = SOLUTION_TEMPLATE.format(
            base_models=repr(base_models),
            special_models=repr(special_models),
            gate_scaler_mean=repr([float(x) for x in gate["scaler_mean"]]),
            gate_scaler_scale=repr([float(x) for x in gate["scaler_scale"]]),
            gate_coef=repr([float(x) for x in gate["coef"]]),
            gate_intercept=float(gate["intercept"]),
            warmup_steps=int(args.warmup_steps),
            gate_threshold=float(args.gate_threshold),
            gate_weight_t0_low=float(gate_weight_t0_low),
            gate_weight_t1_low=float(gate_weight_t1_low),
            gate_weight_t0_high=float(gate_weight_t0_high),
            gate_weight_t1_high=float(gate_weight_t1_high),
            default_expert_weight_t0=float(default_expert_weight_t0),
            default_expert_weight_t1=float(default_expert_weight_t1),
        )
        with open(os.path.join(tmpdir, "solution.py"), "w", encoding="utf-8") as f:
            f.write(solution)

        # Build zip
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        base = str(out_path).replace(".zip", "")
        shutil.make_archive(base, "zip", tmpdir)

    size_mb = os.path.getsize(args.output) / 1e6
    print(f"Created: {args.output} ({size_mb:.2f}MB)")
    if size_mb > 20:
        print("WARNING: exceeds 20MB limit")
    else:
        print("Size OK")


if __name__ == "__main__":
    main()
