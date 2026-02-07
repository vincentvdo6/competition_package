#!/usr/bin/env python
"""Run online-inference parity checks for sequence models.

Checks:
1) forward vs forward_step parity on normalized sequence tensors
2) hidden-state reset correctness across sequence boundaries
3) preprocessing parity (batch vs step) for derived+temporal+interaction path
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import LOBSequenceDataset
from src.data.preprocessing import (
    DerivedFeatureBuilder,
    InteractionFeatureBuilder,
    TemporalBuffer,
    TemporalDerivedFeatureBuilder,
)
from src.models.gru_attention import GRUAttentionModel
from src.models.gru_baseline import GRUBaseline
from src.models.lstm_model import LSTMModel


RAW_COLS = (
    [f"p{i}" for i in range(12)]
    + [f"v{i}" for i in range(12)]
    + [f"dp{i}" for i in range(4)]
    + [f"dv{i}" for i in range(4)]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate online inference parity")
    parser.add_argument("--config", type=str, required=True, help="Config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--data", type=str, default="datasets/valid.parquet", help="Parquet path")
    parser.add_argument("--n-seqs", type=int, default=2, help="Number of sequences for parity checks")
    return parser.parse_args()


def build_model(cfg: dict) -> torch.nn.Module:
    mtype = cfg.get("model", {}).get("type", "gru")
    if mtype == "gru":
        return GRUBaseline(cfg)
    if mtype == "lstm":
        return LSTMModel(cfg)
    if mtype == "gru_attention":
        return GRUAttentionModel(cfg)
    raise ValueError(f"Unsupported model type: {mtype}")


def forward_step_parity(model: torch.nn.Module, features: torch.Tensor) -> float:
    with torch.no_grad():
        batched, _ = model(features)

    max_diff = 0.0
    for b in range(features.size(0)):
        hidden = None
        step_preds = []
        with torch.no_grad():
            for t in range(features.size(1)):
                pred, hidden = model.forward_step(features[b, t : t + 1], hidden)
                step_preds.append(pred)
        step = torch.cat(step_preds, dim=0)
        diff = float((step - batched[b]).abs().max().item())
        max_diff = max(max_diff, diff)
    return max_diff


def hidden_reset_test(model: torch.nn.Module, seq_a: torch.Tensor, seq_b: torch.Tensor) -> tuple[float, float]:
    # seq_b with fresh hidden
    hidden = None
    pred_b_fresh = []
    with torch.no_grad():
        for t in range(seq_b.size(0)):
            pred, hidden = model.forward_step(seq_b[t : t + 1], hidden)
            pred_b_fresh.append(pred)
    pred_b_fresh = torch.cat(pred_b_fresh, dim=0)

    # seq_a then seq_b without reset
    hidden = None
    pred_b_noreset = []
    with torch.no_grad():
        for t in range(seq_a.size(0)):
            _, hidden = model.forward_step(seq_a[t : t + 1], hidden)
        for t in range(seq_b.size(0)):
            pred, hidden = model.forward_step(seq_b[t : t + 1], hidden)
            pred_b_noreset.append(pred)
    pred_b_noreset = torch.cat(pred_b_noreset, dim=0)

    # seq_a then reset then seq_b
    hidden = None
    pred_b_reset = []
    with torch.no_grad():
        for t in range(seq_a.size(0)):
            _, hidden = model.forward_step(seq_a[t : t + 1], hidden)
        hidden = None
        for t in range(seq_b.size(0)):
            pred, hidden = model.forward_step(seq_b[t : t + 1], hidden)
            pred_b_reset.append(pred)
    pred_b_reset = torch.cat(pred_b_reset, dim=0)

    max_diff_reset = float((pred_b_reset - pred_b_fresh).abs().max().item())
    max_diff_noreset = float((pred_b_noreset - pred_b_fresh).abs().max().item())
    return max_diff_reset, max_diff_noreset


def preprocessing_parity(data_path: Path, n_steps: int = 64) -> float:
    df = pd.read_parquet(data_path).sort_values(["seq_ix", "step_in_seq"]).reset_index(drop=True)
    seq = df[df["seq_ix"] == df["seq_ix"].iloc[0]].iloc[:n_steps]
    raw = seq[list(RAW_COLS)].to_numpy(np.float32)

    # Batch pipeline
    base42 = np.concatenate([raw, DerivedFeatureBuilder.compute(raw)], axis=-1)
    temporal = TemporalDerivedFeatureBuilder.compute_batch(base42[None, ...])[0]
    batch45 = np.concatenate([base42, temporal], axis=-1)
    inter = InteractionFeatureBuilder.compute(batch45, has_derived=True)
    batch48 = np.concatenate([batch45, inter], axis=-1)

    # Step pipeline
    buffer = TemporalBuffer()
    step_rows = []
    for i in range(raw.shape[0]):
        feat = np.concatenate([raw[i], DerivedFeatureBuilder.compute(raw[i : i + 1])[0]])
        feat = buffer.compute_step(feat)
        i3 = InteractionFeatureBuilder.compute(feat.reshape(1, -1), has_derived=True)[0]
        feat = np.concatenate([feat, i3])
        step_rows.append(feat)
    step48 = np.stack(step_rows, axis=0)

    return float(np.max(np.abs(batch48 - step48)))


def main() -> None:
    args = parse_args()
    cfg_path = ROOT / args.config
    ckpt_path = ROOT / args.checkpoint
    data_path = ROOT / args.data

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = build_model(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    data_cfg = cfg.get("data", {})
    ds = LOBSequenceDataset(
        str(data_path),
        normalize=bool(data_cfg.get("normalize", True)),
        derived_features=bool(data_cfg.get("derived_features", False)),
        temporal_features=bool(data_cfg.get("temporal_features", False)),
        interaction_features=bool(data_cfg.get("interaction_features", False)),
    )

    n = max(2, int(args.n_seqs))
    feats = ds.features[:n].float()
    forward_step_diff = forward_step_parity(model, feats)
    reset_diff, noreset_diff = hidden_reset_test(model, ds.features[0].float(), ds.features[1].float())
    prep_diff = preprocessing_parity(data_path)

    print("forward_step_parity_max_abs_diff:", forward_step_diff)
    print("hidden_reset_max_diff_vs_fresh:", reset_diff)
    print("hidden_noreset_max_diff_vs_fresh:", noreset_diff)
    print("preprocessing_batch_vs_step_max_abs_diff:", prep_diff)


if __name__ == "__main__":
    main()
