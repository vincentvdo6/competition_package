#!/usr/bin/env python
"""Benchmark dynamic quantization on GRU and GRU+Attention models.

Tests three variants per model:
  1. FP32 (baseline)
  2. Linear-only INT8 (safe fallback)
  3. Linear+GRU INT8 (full quantization)

Measures:
  - Per-model inference speedup (averaged over N sequences)
  - Weighted Pearson correlation delta vs FP32 (primary go/no-go gate)
  - Per-step max absolute prediction error (hidden state drift diagnostic)

Go/no-go gates (from CLAUDE.md):
  - Max val drift: -0.0005 (weighted Pearson)
  - Min speedup: 1.3x (for 3-attn goal) or 1.1x (for timeout safety)

Usage:
  python scripts/benchmark_quantization.py
  python scripts/benchmark_quantization.py --models gru_p1_s47 attn_comb_s43
  python scripts/benchmark_quantization.py --num-seqs 20
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import weighted_pearson_correlation
from src.models.gru_baseline import GRUBaseline
from src.models.gru_attention import GRUAttentionModel
from src.data.preprocessing import DerivedFeatureBuilder, Normalizer

# Reuse registry and zip paths from validate_ensemble_local
from scripts.validate_ensemble_local import MODEL_REGISTRY, ZIPS, CACHE_DIR


# ---------------------------------------------------------------------------
# Quantization variants
# ---------------------------------------------------------------------------

QUANT_VARIANTS = {
    "fp32": None,  # No quantization
    "linear_int8": {nn.Linear},
    "linear_gru_int8": {nn.Linear, nn.GRU},
}


def apply_quantization(model: nn.Module, variant: str) -> nn.Module:
    """Apply dynamic quantization to a model based on variant name."""
    target_modules = QUANT_VARIANTS[variant]
    if target_modules is None:
        return model
    return torch.quantization.quantize_dynamic(
        model, target_modules, dtype=torch.qint8
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(name: str) -> Tuple[nn.Module, Normalizer, dict]:
    """Load a model from the registry, return (model, normalizer, config)."""
    spec = MODEL_REGISTRY[name]
    zip_path = ZIPS[spec["zip"]]
    config_path = ROOT / "configs" / f"{spec['config']}.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extract(spec["ckpt"], tmpdir)
            zf.extract(spec["norm"], tmpdir)

        normalizer = Normalizer.load(str(tmpdir / spec["norm"]))

        model_type = config.get("model", {}).get("type", "gru")
        if model_type == "gru_attention":
            model = GRUAttentionModel(config)
        else:
            model = GRUBaseline(config)

        try:
            ckpt = torch.load(str(tmpdir / spec["ckpt"]), map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(str(tmpdir / spec["ckpt"]), map_location="cpu")

        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        model.eval()

    return model, normalizer, config


# ---------------------------------------------------------------------------
# Online inference runner (per-step, matches scoring server)
# ---------------------------------------------------------------------------

def run_online_inference(
    model: nn.Module,
    normalizer: Normalizer,
    df_seqs: List[pd.DataFrame],
    derived_features: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run online inference on a list of sequence DataFrames.

    Returns (predictions, targets, elapsed_seconds).
    predictions and targets only include need_prediction=True rows.
    """
    all_preds = []
    all_targets = []

    t_start = time.perf_counter()

    for seq_df in df_seqs:
        hidden = None
        for row in seq_df.values:
            need_prediction = bool(row[2])
            lob_data = row[3:35]
            labels = row[35:]

            raw = lob_data.reshape(1, -1).astype(np.float32)
            if derived_features:
                derived = DerivedFeatureBuilder.compute(raw)
                raw = np.concatenate([raw, derived], axis=-1)

            features = normalizer.transform(raw)
            x = torch.from_numpy(features)

            with torch.no_grad():
                pred, hidden = model.forward_step(x, hidden)
                if need_prediction:
                    pred_np = pred.cpu().numpy().squeeze()
                    pred_np = np.clip(pred_np, -6, 6)
                    all_preds.append(pred_np)
                    all_targets.append(labels)

    elapsed = time.perf_counter() - t_start
    return np.array(all_preds), np.array(all_targets, dtype=np.float64), elapsed


def run_online_inference_with_drift(
    model: nn.Module,
    ref_model: nn.Module,
    normalizer: Normalizer,
    df_seqs: List[pd.DataFrame],
    derived_features: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Run online inference and track per-step drift vs reference model.

    Returns (predictions, targets, elapsed_seconds, max_abs_drift).
    """
    all_preds = []
    all_targets = []
    max_drift = 0.0

    t_start = time.perf_counter()

    for seq_df in df_seqs:
        hidden_q = None
        hidden_ref = None
        for row in seq_df.values:
            need_prediction = bool(row[2])
            lob_data = row[3:35]
            labels = row[35:]

            raw = lob_data.reshape(1, -1).astype(np.float32)
            if derived_features:
                derived = DerivedFeatureBuilder.compute(raw)
                raw = np.concatenate([raw, derived], axis=-1)

            features = normalizer.transform(raw)
            x = torch.from_numpy(features)

            with torch.no_grad():
                pred_q, hidden_q = model.forward_step(x, hidden_q)
                pred_ref, hidden_ref = ref_model.forward_step(x, hidden_ref)

                if need_prediction:
                    pq = pred_q.cpu().numpy().squeeze()
                    pr = pred_ref.cpu().numpy().squeeze()
                    pq = np.clip(pq, -6, 6)
                    pr = np.clip(pr, -6, 6)

                    step_drift = np.max(np.abs(pq - pr))
                    if step_drift > max_drift:
                        max_drift = step_drift

                    all_preds.append(pq)
                    all_targets.append(labels)

    elapsed = time.perf_counter() - t_start
    return np.array(all_preds), np.array(all_targets, dtype=np.float64), elapsed, float(max_drift)


# ---------------------------------------------------------------------------
# Benchmark one model
# ---------------------------------------------------------------------------

def benchmark_model(
    name: str,
    df: pd.DataFrame,
    num_seqs: int = 10,
    warmup_seqs: int = 3,
) -> Dict:
    """Benchmark all quantization variants for a single model."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")

    spec = MODEL_REGISTRY[name]
    model_type = spec["type"]
    print(f"  Type: {model_type}")

    # Load the base FP32 model
    model_fp32, normalizer, config = load_model(name)
    derived = bool(config.get("data", {}).get("derived_features", False))

    # Select sequences
    unique_seqs = df["seq_ix"].unique()
    total_seqs = warmup_seqs + num_seqs
    if total_seqs > len(unique_seqs):
        total_seqs = len(unique_seqs)
        num_seqs = total_seqs - warmup_seqs

    selected_seqs = unique_seqs[:total_seqs]
    seq_dfs = [df[df["seq_ix"] == s] for s in selected_seqs]
    warmup_dfs = seq_dfs[:warmup_seqs]
    bench_dfs = seq_dfs[warmup_seqs:]

    results = {}

    # --- FP32 baseline ---
    print(f"\n  [FP32] Warming up ({warmup_seqs} seqs)...")
    run_online_inference(model_fp32, normalizer, warmup_dfs, derived)

    print(f"  [FP32] Benchmarking ({num_seqs} seqs)...")
    preds_fp32, targets_fp32, time_fp32 = run_online_inference(
        model_fp32, normalizer, bench_dfs, derived
    )

    t0_fp32 = weighted_pearson_correlation(targets_fp32[:, 0], preds_fp32[:, 0])
    t1_fp32 = weighted_pearson_correlation(targets_fp32[:, 1], preds_fp32[:, 1])
    avg_fp32 = (t0_fp32 + t1_fp32) / 2.0

    print(f"  [FP32] Time: {time_fp32:.2f}s | Val: {avg_fp32:.4f} (t0={t0_fp32:.4f}, t1={t1_fp32:.4f})")
    results["fp32"] = {
        "time": time_fp32,
        "val_avg": avg_fp32,
        "val_t0": t0_fp32,
        "val_t1": t1_fp32,
        "speedup": 1.0,
        "val_drift": 0.0,
        "max_abs_drift": 0.0,
    }

    # --- Quantized variants ---
    for variant in ["linear_int8", "linear_gru_int8"]:
        label = variant.upper()
        print(f"\n  [{label}] Applying quantization...")

        # Load fresh model (quantization modifies in-place)
        model_fresh, _, _ = load_model(name)
        try:
            model_q = apply_quantization(model_fresh, variant)
        except Exception as e:
            print(f"  [{label}] FAILED: {e}")
            results[variant] = {"error": str(e)}
            continue

        # Warmup
        print(f"  [{label}] Warming up ({warmup_seqs} seqs)...")
        run_online_inference(model_q, normalizer, warmup_dfs, derived)

        # Benchmark timing (quantized model only, no drift tracking)
        print(f"  [{label}] Benchmarking ({num_seqs} seqs)...")
        preds_q, targets_q, time_q = run_online_inference(
            model_q, normalizer, bench_dfs, derived
        )

        # Measure drift separately (reload FP32 for fair comparison)
        model_fp32_fresh, _, _ = load_model(name)
        _, _, _, max_drift = run_online_inference_with_drift(
            model_q, model_fp32_fresh, normalizer, bench_dfs[:3], derived  # only 3 seqs for drift
        )

        t0_q = weighted_pearson_correlation(targets_q[:, 0], preds_q[:, 0])
        t1_q = weighted_pearson_correlation(targets_q[:, 1], preds_q[:, 1])
        avg_q = (t0_q + t1_q) / 2.0

        speedup = time_fp32 / time_q if time_q > 0 else 0
        val_drift = avg_q - avg_fp32

        print(f"  [{label}] Time: {time_q:.2f}s | Speedup: {speedup:.2f}x")
        print(f"  [{label}] Val: {avg_q:.4f} | Drift: {val_drift:+.5f} | Max step drift: {max_drift:.6f}")

        # Go/no-go assessment
        if val_drift < -0.0005:
            print(f"  [{label}] FAIL: val drift {val_drift:+.5f} exceeds -0.0005 gate")
        elif speedup < 1.1:
            print(f"  [{label}] MARGINAL: speedup {speedup:.2f}x below 1.1x minimum")
        elif speedup < 1.3:
            print(f"  [{label}] PASS (timeout safety): speedup {speedup:.2f}x (below 1.3x 3-attn goal)")
        else:
            print(f"  [{label}] PASS (full): speedup {speedup:.2f}x meets 1.3x 3-attn goal")

        results[variant] = {
            "time": time_q,
            "val_avg": avg_q,
            "val_t0": t0_q,
            "val_t1": t1_q,
            "speedup": speedup,
            "val_drift": val_drift,
            "max_abs_drift": max_drift,
        }

    return results


# ---------------------------------------------------------------------------
# Full-validation scoring (optional, uses cached predictions as ground truth)
# ---------------------------------------------------------------------------

def validate_against_cache(name: str, variant: str, df: pd.DataFrame) -> Optional[float]:
    """Run full-validation quantized inference and compare to cached FP32 predictions.

    Returns weighted Pearson delta (quantized - cached).
    """
    cache_file = CACHE_DIR / f"{name}.npz"
    if not cache_file.exists():
        print(f"  No cache for {name}, skipping full-val comparison")
        return None

    cached = np.load(cache_file)
    cached_preds = cached["preds"]
    cached_targets = cached["targets"]

    # Cached score
    t0_cached = weighted_pearson_correlation(cached_targets[:, 0], cached_preds[:, 0])
    t1_cached = weighted_pearson_correlation(cached_targets[:, 1], cached_preds[:, 1])
    avg_cached = (t0_cached + t1_cached) / 2.0

    # Run quantized inference on full val set
    model, normalizer, config = load_model(name)
    model = apply_quantization(model, variant)
    derived = bool(config.get("data", {}).get("derived_features", False))

    unique_seqs = df["seq_ix"].unique()
    seq_dfs = [df[df["seq_ix"] == s] for s in unique_seqs]

    preds_q, targets_q, elapsed = run_online_inference(
        model, normalizer, seq_dfs, derived
    )

    t0_q = weighted_pearson_correlation(targets_q[:, 0], preds_q[:, 0])
    t1_q = weighted_pearson_correlation(targets_q[:, 1], preds_q[:, 1])
    avg_q = (t0_q + t1_q) / 2.0

    drift = avg_q - avg_cached
    print(f"  Full-val {variant}: {avg_q:.4f} vs cached {avg_cached:.4f} = drift {drift:+.5f} ({elapsed:.0f}s)")
    return drift


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark dynamic quantization")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Model names to benchmark (default: champion's 7)")
    parser.add_argument("--num-seqs", type=int, default=15,
                        help="Number of sequences for timing (default: 15)")
    parser.add_argument("--warmup-seqs", type=int, default=3,
                        help="Number of warmup sequences (default: 3)")
    parser.add_argument("--full-val", action="store_true",
                        help="Also run full validation set comparison (slow)")
    parser.add_argument("--data", default="datasets/valid.parquet",
                        help="Path to validation data")
    args = parser.parse_args()

    # Default: champion s2_s43_swap models
    if args.models is None:
        args.models = [
            "gru_p1_s47", "gru_tw2_s50", "gru_tw2_s48",
            "gru_p1_s45", "gru_p1_s50",
            "attn_comb_s43", "attn_nb07_s50",
        ]

    # Verify all models exist
    for name in args.models:
        if name not in MODEL_REGISTRY:
            print(f"ERROR: {name} not in MODEL_REGISTRY")
            sys.exit(1)

    # Load data
    data_path = ROOT / args.data
    print(f"Loading validation data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"  {len(df)} rows, {df['seq_ix'].nunique()} sequences")

    # Run benchmarks
    all_results = {}
    for name in args.models:
        all_results[name] = benchmark_model(
            name, df, num_seqs=args.num_seqs, warmup_seqs=args.warmup_seqs
        )

    # Full validation comparison (optional)
    if args.full_val:
        print(f"\n{'='*60}")
        print("Full validation comparison (vs cached FP32 predictions)")
        print(f"{'='*60}")
        for name in args.models:
            best_variant = None
            best_speedup = 0
            for v in ["linear_int8", "linear_gru_int8"]:
                r = all_results[name].get(v, {})
                if "error" not in r and r.get("val_drift", -1) >= -0.0005:
                    if r.get("speedup", 0) > best_speedup:
                        best_variant = v
                        best_speedup = r["speedup"]
            if best_variant:
                validate_against_cache(name, best_variant, df)

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Variant':<18} {'Time(s)':>8} {'Speedup':>8} {'Val':>8} {'Drift':>10} {'MaxStep':>10} {'Gate':>8}")
    print("-" * 100)

    for name in args.models:
        for variant in ["fp32", "linear_int8", "linear_gru_int8"]:
            r = all_results[name].get(variant, {})
            if "error" in r:
                print(f"{name:<20} {variant:<18} {'ERROR':>8} {r['error']}")
                continue
            if not r:
                continue

            gate = "BASE"
            if variant != "fp32":
                if r["val_drift"] < -0.0005:
                    gate = "FAIL"
                elif r["speedup"] >= 1.3:
                    gate = "PASS"
                elif r["speedup"] >= 1.1:
                    gate = "OK"
                else:
                    gate = "SLOW"

            print(f"{name:<20} {variant:<18} {r['time']:>8.2f} {r['speedup']:>7.2f}x "
                  f"{r['val_avg']:>8.4f} {r['val_drift']:>+10.5f} {r['max_abs_drift']:>10.6f} {gate:>8}")

    # Overall recommendation
    print(f"\n{'='*60}")
    print("RECOMMENDATION")
    print(f"{'='*60}")

    gru_models = [n for n in args.models if not n.startswith("attn_")]
    attn_models = [n for n in args.models if n.startswith("attn_")]

    for model_group, group_name in [(gru_models, "GRU"), (attn_models, "Attention")]:
        if not model_group:
            continue

        best_variant = "fp32"
        best_speedup = 1.0
        for variant in ["linear_int8", "linear_gru_int8"]:
            speedups = []
            all_pass = True
            for name in model_group:
                r = all_results[name].get(variant, {})
                if "error" in r or r.get("val_drift", -1) < -0.0005:
                    all_pass = False
                    break
                speedups.append(r.get("speedup", 0))

            if all_pass and speedups:
                avg_speedup = np.mean(speedups)
                if avg_speedup > best_speedup:
                    best_variant = variant
                    best_speedup = avg_speedup

        if best_variant == "fp32":
            print(f"  {group_name}: No quantization variant passes all gates. Stay with FP32.")
        else:
            print(f"  {group_name}: Best variant = {best_variant} (avg {best_speedup:.2f}x speedup)")

            if best_speedup >= 1.3:
                print(f"    -> PROCEED to 3-attn ensemble experiments")
            elif best_speedup >= 1.1:
                print(f"    -> USE for timeout safety on 5G+2A (not enough for 3-attn)")
            else:
                print(f"    -> MARGINAL benefit, consider skipping")


if __name__ == "__main__":
    main()
