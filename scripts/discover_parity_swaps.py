#!/usr/bin/env python
"""Discover and build two orthogonal parity-seed swap submissions.

Workflow:
1) Load cached per-seed validation predictions from greedy_vanilla_ensemble cache
2) Reconstruct anchor per-target ensemble
3) Identify weakest pure t1 and pure t0 active members via leave-one-out
4) Simulate one-seed swaps from inactive pool
5) Bootstrap sequence-level deltas (p10 gate)
6) Build two ONNX submissions:
   - Slot A: best t1 swap
   - Slot B: best t0 swap
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]


DEFAULT_REQUIRED_SEEDS = list(range(42, 65))
DEFAULT_ANCHOR_SEEDS = [43, 44, 45, 46, 50, 54, 55, 57, 58, 59, 60, 61, 63, 64]
DEFAULT_ANCHOR_W0 = [1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
DEFAULT_ANCHOR_W1 = [1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0.25, 0, 1.75]


SOLUTION_TEMPLATE = '''"""Per-target vanilla GRU ONNX ensemble with shared model pool."""

import os
import numpy as np
import onnxruntime as ort

MODEL_CONFIGS = {model_configs}


class OnnxVanillaGRU:
    def __init__(self, onnx_path, hidden_size, num_layers):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])

    def init_hidden(self):
        return np.zeros((self.num_layers, 1, self.hidden_size), dtype=np.float32)

    def run_step(self, x_np, hidden):
        pred, new_hidden = self.sess.run(
            ["prediction", "hidden_out"],
            {{"input": x_np, "hidden_in": hidden}},
        )
        return pred[0], new_hidden


class PredictionModel:
    def __init__(self, model_path=""):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        self.models = []
        self.hiddens = []
        self.weights_t0 = []
        self.weights_t1 = []

        for filename, h, nl, w0, w1 in MODEL_CONFIGS:
            model = OnnxVanillaGRU(os.path.join(base_dir, filename), h, nl)
            self.models.append(model)
            self.hiddens.append(model.init_hidden())
            self.weights_t0.append(w0)
            self.weights_t1.append(w1)

        sum_t0 = float(sum(self.weights_t0))
        sum_t1 = float(sum(self.weights_t1))
        self.weights_t0 = np.array([w / sum_t0 for w in self.weights_t0], dtype=np.float32)
        self.weights_t1 = np.array([w / sum_t1 for w in self.weights_t1], dtype=np.float32)

        self.prev_seq_ix = None

    def predict(self, data_point) -> np.ndarray:
        seq_ix = data_point.seq_ix
        if seq_ix != self.prev_seq_ix:
            self.hiddens = [m.init_hidden() for m in self.models]
            self.prev_seq_ix = seq_ix

        x = data_point.state.astype(np.float32)[:32].reshape(1, -1)

        t0_sum = 0.0
        t1_sum = 0.0
        for i, model in enumerate(self.models):
            pred, self.hiddens[i] = model.run_step(x, self.hiddens[i])
            t0_sum += float(self.weights_t0[i] * pred[0])
            t1_sum += float(self.weights_t1[i] * pred[1])

        if not data_point.need_prediction:
            return None

        out = np.array([t0_sum, t1_sum], dtype=np.float32)
        return out.clip(-6, 6)
'''


class VanillaGRUStep(nn.Module):
    """Minimal wrapper for ONNX export: raw features -> GRU -> linear."""

    def __init__(self, gru: nn.Module, fc: nn.Module):
        super().__init__()
        self.gru = gru
        self.fc = fc

    def forward(self, x: torch.Tensor, hidden: torch.Tensor):
        x = x.unsqueeze(1)
        out, new_hidden = self.gru(x, hidden)
        return self.fc(out.squeeze(1)), new_hidden


@dataclass
class Score:
    t0: float
    t1: float
    avg: float


def parse_seed_from_stem(stem: str) -> int:
    m = re.search(r"seed(\d+)$", stem)
    if not m:
        raise ValueError(f"Cannot parse seed from cache name: {stem}")
    return int(m.group(1))


def weighted_corr(y: np.ndarray, x: np.ndarray) -> float:
    """Weighted Pearson with weights |y|."""
    y = y.astype(np.float64, copy=False)
    x = x.astype(np.float64, copy=False)
    w = np.abs(y)
    sw = float(np.sum(w))
    if sw <= 0:
        return 0.0
    swx = float(np.sum(w * x))
    swy = float(np.sum(w * y))
    swxx = float(np.sum(w * x * x))
    swyy = float(np.sum(w * y * y))
    swxy = float(np.sum(w * x * y))
    mx = swx / sw
    my = swy / sw
    vx = max(swxx / sw - mx * mx, 0.0)
    vy = max(swyy / sw - my * my, 0.0)
    if vx <= 1e-18 or vy <= 1e-18:
        return 0.0
    cov = swxy / sw - mx * my
    return float(cov / np.sqrt(vx * vy))


def score_predictions(targets: np.ndarray, preds: np.ndarray) -> Score:
    t0 = weighted_corr(targets[:, 0], preds[:, 0])
    t1 = weighted_corr(targets[:, 1], preds[:, 1])
    return Score(t0=t0, t1=t1, avg=(t0 + t1) / 2.0)


def normalize_weights(weights: Sequence[float]) -> np.ndarray:
    w = np.array(weights, dtype=np.float64)
    s = float(w.sum())
    if s <= 0:
        raise ValueError("Weights must sum to > 0")
    return w / s


def build_ensemble_predictions(
    seed_order: Sequence[int],
    weights_t0: Sequence[float],
    weights_t1: Sequence[float],
    seed_preds: Dict[int, np.ndarray],
) -> np.ndarray:
    if len(seed_order) != len(weights_t0) or len(seed_order) != len(weights_t1):
        raise ValueError("seed_order and weights lengths must match")

    w0 = normalize_weights(weights_t0)
    w1 = normalize_weights(weights_t1)
    n = next(iter(seed_preds.values())).shape[0]
    out = np.zeros((n, 2), dtype=np.float64)

    for i, seed in enumerate(seed_order):
        p = seed_preds[seed].astype(np.float64, copy=False)
        out[:, 0] += w0[i] * p[:, 0]
        out[:, 1] += w1[i] * p[:, 1]

    return np.clip(out, -6.0, 6.0).astype(np.float32)


def load_cached_seed_preds(cache_dir: Path, required_seeds: Sequence[int]) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache dir not found: {cache_dir}")

    seed_preds: Dict[int, np.ndarray] = {}
    for npz_path in sorted(cache_dir.glob("gru_parity_v1_seed*.npz")):
        seed = parse_seed_from_stem(npz_path.stem)
        data = np.load(npz_path)
        seed_preds[seed] = data["preds"].astype(np.float32)

    missing = [s for s in required_seeds if s not in seed_preds]
    if missing:
        raise RuntimeError(
            f"Missing cached seeds: {missing}. "
            f"Run cache first for all required seeds."
        )

    targets_path = cache_dir / "_targets.npz"
    if targets_path.exists():
        targets = np.load(targets_path)["targets"].astype(np.float64)
    else:
        targets = next(iter(seed_preds.values())).astype(np.float64)
        raise RuntimeError(f"Missing {targets_path}; rerun cache to write targets")

    expected_n = targets.shape[0]
    for s in required_seeds:
        if seed_preds[s].shape[0] != expected_n:
            raise RuntimeError(
                f"Seed {s} rows mismatch: {seed_preds[s].shape[0]} vs targets {expected_n}"
            )
    return seed_preds, targets


def seq_inverse_indices(data_path: Path, expected_n: int) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(data_path, columns=["seq_ix", "need_prediction"])
    mask = df["need_prediction"].to_numpy().astype(bool)
    seq = df.loc[mask, "seq_ix"].to_numpy(np.int64)
    if seq.shape[0] != expected_n:
        raise RuntimeError(
            f"Scored rows mismatch: seq_ix rows={seq.shape[0]} expected={expected_n}"
        )
    unique_seq, inv = np.unique(seq, return_inverse=True)
    return unique_seq, inv


def per_seq_stats(targets: np.ndarray, preds: np.ndarray, seq_inv: np.ndarray, n_seq: int) -> np.ndarray:
    """Return stats shape (n_seq, 2, 6): [sw, swx, swy, swxx, swyy, swxy]."""
    stats = np.zeros((n_seq, 2, 6), dtype=np.float64)
    for t in (0, 1):
        y = targets[:, t].astype(np.float64, copy=False)
        x = preds[:, t].astype(np.float64, copy=False)
        w = np.abs(y)
        stats[:, t, 0] = np.bincount(seq_inv, weights=w, minlength=n_seq)
        stats[:, t, 1] = np.bincount(seq_inv, weights=w * x, minlength=n_seq)
        stats[:, t, 2] = np.bincount(seq_inv, weights=w * y, minlength=n_seq)
        stats[:, t, 3] = np.bincount(seq_inv, weights=w * x * x, minlength=n_seq)
        stats[:, t, 4] = np.bincount(seq_inv, weights=w * y * y, minlength=n_seq)
        stats[:, t, 5] = np.bincount(seq_inv, weights=w * x * y, minlength=n_seq)
    return stats


def score_from_agg_stats(agg: np.ndarray) -> Score:
    target_scores = []
    for t in (0, 1):
        sw, swx, swy, swxx, swyy, swxy = [float(v) for v in agg[t]]
        if sw <= 0:
            target_scores.append(0.0)
            continue
        mx = swx / sw
        my = swy / sw
        vx = max(swxx / sw - mx * mx, 0.0)
        vy = max(swyy / sw - my * my, 0.0)
        if vx <= 1e-18 or vy <= 1e-18:
            target_scores.append(0.0)
            continue
        cov = swxy / sw - mx * my
        target_scores.append(float(cov / np.sqrt(vx * vy)))
    return Score(t0=target_scores[0], t1=target_scores[1], avg=(target_scores[0] + target_scores[1]) / 2.0)


def bootstrap_delta_stats(
    anchor_stats: np.ndarray,
    cand_stats: np.ndarray,
    n_bootstrap: int,
    rng_seed: int,
) -> Dict[str, float]:
    n_seq = anchor_stats.shape[0]
    rng = np.random.RandomState(rng_seed)
    deltas = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        sample = rng.randint(0, n_seq, size=n_seq)
        counts = np.bincount(sample, minlength=n_seq).astype(np.float64)
        agg_anchor = np.tensordot(counts, anchor_stats, axes=(0, 0))
        agg_cand = np.tensordot(counts, cand_stats, axes=(0, 0))
        s_anchor = score_from_agg_stats(agg_anchor)
        s_cand = score_from_agg_stats(agg_cand)
        deltas[i] = s_cand.avg - s_anchor.avg

    return {
        "mean": float(np.mean(deltas)),
        "std": float(np.std(deltas)),
        "p10": float(np.quantile(deltas, 0.10)),
        "p50": float(np.quantile(deltas, 0.50)),
        "p90": float(np.quantile(deltas, 0.90)),
    }


def find_weakest_index(
    target: str,
    seed_order: Sequence[int],
    w0: Sequence[float],
    w1: Sequence[float],
    seed_preds: Dict[int, np.ndarray],
    targets: np.ndarray,
    anchor_score: Score,
) -> Tuple[int, List[dict]]:
    if target not in {"t0", "t1"}:
        raise ValueError("target must be t0 or t1")

    rows: List[dict] = []
    if target == "t0":
        idxs = [i for i in range(len(seed_order)) if w0[i] > 0 and w1[i] == 0]
        if not idxs:
            idxs = [i for i in range(len(seed_order)) if w0[i] > 0]
    else:
        idxs = [i for i in range(len(seed_order)) if w1[i] > 0 and w0[i] == 0]
        if not idxs:
            idxs = [i for i in range(len(seed_order)) if w1[i] > 0]

    for idx in idxs:
        tw0 = list(w0)
        tw1 = list(w1)
        if target == "t0":
            tw0[idx] = 0.0
        else:
            tw1[idx] = 0.0
        pred = build_ensemble_predictions(seed_order, tw0, tw1, seed_preds)
        sc = score_predictions(targets, pred)
        rows.append(
            {
                "index": idx,
                "seed": int(seed_order[idx]),
                "loo_score": asdict(sc),
                "delta_t0": float(sc.t0 - anchor_score.t0),
                "delta_t1": float(sc.t1 - anchor_score.t1),
                "delta_avg": float(sc.avg - anchor_score.avg),
            }
        )

    key = "delta_t0" if target == "t0" else "delta_t1"
    weakest = max(rows, key=lambda r: (r[key], r["delta_avg"]))
    return int(weakest["index"]), rows


def export_vanilla_to_onnx(ckpt_path: Path, hidden_size: int, num_layers: int, onnx_path: Path) -> None:
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

    gru_sd = {k.replace("gru.", ""): v for k, v in state_dict.items() if k.startswith("gru.")}
    fc_sd = {k.replace("output_proj.", ""): v for k, v in state_dict.items() if k.startswith("output_proj.")}
    gru.load_state_dict(gru_sd)
    fc.load_state_dict(fc_sd)

    wrapper = VanillaGRUStep(gru, fc)
    wrapper.eval()

    dummy_x = torch.randn(1, 32)
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


def build_submission_zip(
    output_zip: Path,
    seed_order: Sequence[int],
    weights_t0: Sequence[float],
    weights_t1: Sequence[float],
    checkpoint_dir: Path,
    checkpoint_pattern: str,
    hidden_size: int,
    num_layers: int,
) -> None:
    model_configs = []
    for i, seed in enumerate(seed_order):
        model_configs.append((f"model_{i}.onnx", hidden_size, num_layers, float(weights_t0[i]), float(weights_t1[i])))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        solution_code = SOLUTION_TEMPLATE.format(model_configs=repr(model_configs))
        (tmpdir_path / "solution.py").write_text(solution_code, encoding="utf-8")

        for i, seed in enumerate(seed_order):
            ckpt = checkpoint_dir / checkpoint_pattern.format(seed=seed)
            if not ckpt.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
            export_vanilla_to_onnx(ckpt, hidden_size, num_layers, tmpdir_path / f"model_{i}.onnx")

        output_zip.parent.mkdir(parents=True, exist_ok=True)
        base = output_zip.with_suffix("")
        shutil.make_archive(str(base), "zip", tmpdir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover parity swaps and optionally build two ONNX submissions")
    parser.add_argument("--cache-dir", default="cache/parity23_valid_preds")
    parser.add_argument("--data", default="datasets/valid.parquet")
    parser.add_argument("--required-seeds", nargs="+", type=int, default=DEFAULT_REQUIRED_SEEDS)
    parser.add_argument("--anchor-seeds", nargs="+", type=int, default=DEFAULT_ANCHOR_SEEDS)
    parser.add_argument("--anchor-w0", nargs="+", type=float, default=DEFAULT_ANCHOR_W0)
    parser.add_argument("--anchor-w1", nargs="+", type=float, default=DEFAULT_ANCHOR_W1)
    parser.add_argument("--delta-min", type=float, default=0.00015)
    parser.add_argument("--p10-max-drop", type=float, default=0.00005)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--output-report", default="logs/parity_swap_discovery_feb21_b2.json")
    parser.add_argument("--checkpoint-dir", default="logs/vanilla_all")
    parser.add_argument("--checkpoint-pattern", default="gru_parity_v1_seed{seed}.pt")
    parser.add_argument("--output-dir", default="submissions/ready")
    parser.add_argument("--slot-a-name", default="feb21-b2-t1swap-a-onnx.zip")
    parser.add_argument("--slot-b-name", default="feb21-b2-t0swap-a-onnx.zip")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()

    if not (
        len(args.anchor_seeds) == len(args.anchor_w0) == len(args.anchor_w1)
    ):
        raise ValueError("Anchor seeds and weights lengths must match")

    cache_dir = ROOT / args.cache_dir
    seed_preds, targets = load_cached_seed_preds(cache_dir, args.required_seeds)
    unique_seq, seq_inv = seq_inverse_indices(ROOT / args.data, targets.shape[0])
    n_seq = len(unique_seq)

    single_rows = []
    for seed in sorted(args.required_seeds):
        sc = score_predictions(targets, np.clip(seed_preds[seed], -6, 6))
        single_rows.append({"seed": int(seed), "t0": sc.t0, "t1": sc.t1, "avg": sc.avg})

    anchor_pred = build_ensemble_predictions(args.anchor_seeds, args.anchor_w0, args.anchor_w1, seed_preds)
    anchor_score = score_predictions(targets, anchor_pred)
    anchor_stats = per_seq_stats(targets, anchor_pred, seq_inv, n_seq)

    weak_t1_idx, loo_t1 = find_weakest_index(
        "t1", args.anchor_seeds, args.anchor_w0, args.anchor_w1, seed_preds, targets, anchor_score
    )
    weak_t0_idx, loo_t0 = find_weakest_index(
        "t0", args.anchor_seeds, args.anchor_w0, args.anchor_w1, seed_preds, targets, anchor_score
    )

    active_set = set(args.anchor_seeds)
    inactive = [s for s in sorted(args.required_seeds) if s not in active_set]

    def run_swaps(target: str, weak_idx: int) -> List[dict]:
        rows: List[dict] = []
        replaced_seed = int(args.anchor_seeds[weak_idx])
        for cand_seed in inactive:
            cand_order = list(args.anchor_seeds)
            cand_order[weak_idx] = cand_seed
            cand_pred = build_ensemble_predictions(cand_order, args.anchor_w0, args.anchor_w1, seed_preds)
            cand_score = score_predictions(targets, cand_pred)
            cand_stats = per_seq_stats(targets, cand_pred, seq_inv, n_seq)
            boot = bootstrap_delta_stats(anchor_stats, cand_stats, args.bootstrap, args.bootstrap_seed + cand_seed)
            delta_avg = float(cand_score.avg - anchor_score.avg)
            keep_gate = bool(delta_avg >= args.delta_min and boot["p10"] >= -args.p10_max_drop)
            rows.append(
                {
                    "target": target,
                    "replace_index": int(weak_idx),
                    "replace_seed": replaced_seed,
                    "candidate_seed": int(cand_seed),
                    "seed_order": [int(x) for x in cand_order],
                    "score": asdict(cand_score),
                    "delta_t0": float(cand_score.t0 - anchor_score.t0),
                    "delta_t1": float(cand_score.t1 - anchor_score.t1),
                    "delta_avg": delta_avg,
                    "bootstrap_delta_vs_anchor": boot,
                    "keep_gate": keep_gate,
                }
            )
        rows.sort(key=lambda r: (r["delta_avg"], r["bootstrap_delta_vs_anchor"]["p10"]), reverse=True)
        return rows

    t1_swaps = run_swaps("t1", weak_t1_idx)
    t0_swaps = run_swaps("t0", weak_t0_idx)

    def choose_best(rows: List[dict]) -> dict:
        gated = [r for r in rows if r["keep_gate"]]
        pool = gated if gated else rows
        return pool[0]

    best_t1 = choose_best(t1_swaps)
    best_t0 = choose_best(t0_swaps)

    report = {
        "anchor": {
            "seeds": [int(x) for x in args.anchor_seeds],
            "weights_t0": [float(x) for x in args.anchor_w0],
            "weights_t1": [float(x) for x in args.anchor_w1],
            "score": asdict(anchor_score),
        },
        "settings": {
            "required_seeds": [int(x) for x in args.required_seeds],
            "delta_min": args.delta_min,
            "p10_max_drop": args.p10_max_drop,
            "bootstrap": args.bootstrap,
            "bootstrap_seed": args.bootstrap_seed,
            "n_sequences": int(n_seq),
        },
        "single_seed_ranking": single_rows,
        "leave_one_out": {
            "t1": loo_t1,
            "t0": loo_t0,
            "weak_t1_seed": int(args.anchor_seeds[weak_t1_idx]),
            "weak_t0_seed": int(args.anchor_seeds[weak_t0_idx]),
        },
        "swap_results": {
            "t1_swaps": t1_swaps,
            "t0_swaps": t0_swaps,
        },
        "selected": {
            "slot_a_t1": best_t1,
            "slot_b_t0": best_t0,
        },
    }

    out_report = ROOT / args.output_report
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote discovery report: {out_report}")

    if not args.no_build:
        output_dir = ROOT / args.output_dir
        slot_a = output_dir / args.slot_a_name
        slot_b = output_dir / args.slot_b_name
        build_submission_zip(
            slot_a,
            best_t1["seed_order"],
            args.anchor_w0,
            args.anchor_w1,
            ROOT / args.checkpoint_dir,
            args.checkpoint_pattern,
            args.hidden_size,
            args.num_layers,
        )
        build_submission_zip(
            slot_b,
            best_t0["seed_order"],
            args.anchor_w0,
            args.anchor_w1,
            ROOT / args.checkpoint_dir,
            args.checkpoint_pattern,
            args.hidden_size,
            args.num_layers,
        )
        print(f"Built: {slot_a}")
        print(f"Built: {slot_b}")

    print("Selected slot A (t1 swap):")
    print(json.dumps(best_t1, indent=2))
    print("Selected slot B (t0 swap):")
    print(json.dumps(best_t0, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
