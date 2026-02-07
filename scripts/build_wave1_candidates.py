#!/usr/bin/env python
"""Build Wave-1 ensemble candidates (A/B/C) from seed-diversity artifacts.

Candidates (from plan):
- A: best subset by local valid score
- B: optimal SLSQP weights
- C: shrinked optimal weights: 0.7 * w_opt + 0.3 * w_uniform
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Wave-1 ensemble candidate zips")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Model config used by all seed checkpoints",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="Seed checkpoints in the same order used for run_04 analysis",
    )
    parser.add_argument(
        "--normalizer",
        type=str,
        required=True,
        help="Normalizer path used by checkpoints",
    )
    parser.add_argument(
        "--artifacts",
        type=str,
        default="notebooks/artifacts/04_seed_diversity",
        help="Artifact directory from run_04_seed_diversity_analysis.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="submissions",
        help="Where candidate zips are written",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="wave1",
        help="Output file prefix",
    )
    return parser.parse_args()


def _run(cmd: List[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def _normalize(weights: List[float]) -> List[float]:
    s = float(sum(weights))
    if s <= 0:
        raise ValueError("Weights must sum to > 0")
    return [float(w) / s for w in weights]


def main() -> None:
    args = parse_args()

    artifacts_dir = ROOT / args.artifacts
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    optimal_path = artifacts_dir / "optimal_weights.json"
    ens_n_path = artifacts_dir / "ensemble_vs_n_models.csv"
    if not optimal_path.exists():
        raise FileNotFoundError(f"Missing artifact: {optimal_path}")
    if not ens_n_path.exists():
        raise FileNotFoundError(f"Missing artifact: {ens_n_path}")

    with optimal_path.open("r", encoding="utf-8") as f:
        optimal = json.load(f)
    weights_opt = [float(w) for w in optimal.get("weights", [])]

    checkpoints = [str(ROOT / p) for p in args.checkpoints]
    n_models = len(checkpoints)
    if len(weights_opt) != n_models:
        status_path = artifacts_dir / "status.json"
        status_hint = ""
        if status_path.exists():
            try:
                with status_path.open("r", encoding="utf-8") as f:
                    status = json.load(f)
                if status.get("status") == "missing_checkpoints":
                    status_hint = (
                        "\nArtifacts are placeholders from a prior run with missing checkpoints. "
                        "Run notebooks/run_04_seed_diversity_analysis.py first, then rerun this script."
                    )
            except Exception:
                status_hint = ""
        raise ValueError(
            f"optimal_weights has {len(weights_opt)} entries, but --checkpoints has {n_models}."
            f"{status_hint}"
        )

    df = pd.read_csv(ens_n_path)
    if len(df) == 0:
        raise ValueError(f"{ens_n_path} is empty. Run seed-diversity analysis first.")
    best_row = df.loc[df["score_avg"].idxmax()]
    subset_indices = ast.literal_eval(best_row["subset_indices"])
    subset_indices = [int(i) for i in subset_indices]

    # Candidate A: best subset by valid, uniform weights
    ckpt_a = [checkpoints[i] for i in subset_indices]
    w_a = _normalize([1.0] * len(ckpt_a))
    out_a = output_dir / f"{args.prefix}_candidateA_best_subset.zip"
    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_ensemble.py"),
            "--config",
            args.config,
            "--checkpoints",
            *ckpt_a,
            "--normalizer",
            args.normalizer,
            "--weights",
            *[str(w) for w in w_a],
            "--output",
            str(out_a),
        ]
    )

    # Candidate B: optimal weights over all models
    out_b = output_dir / f"{args.prefix}_candidateB_optimal.zip"
    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_ensemble.py"),
            "--config",
            args.config,
            "--checkpoints",
            *checkpoints,
            "--normalizer",
            args.normalizer,
            "--weights",
            *[str(w) for w in weights_opt],
            "--output",
            str(out_b),
        ]
    )

    # Candidate C: shrinked optimal
    uniform = [1.0 / n_models] * n_models
    weights_shrink = _normalize(
        [0.7 * w_opt + 0.3 * w_uni for w_opt, w_uni in zip(weights_opt, uniform)]
    )
    out_c = output_dir / f"{args.prefix}_candidateC_shrinked.zip"
    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_ensemble.py"),
            "--config",
            args.config,
            "--checkpoints",
            *checkpoints,
            "--normalizer",
            args.normalizer,
            "--weights",
            *[str(w) for w in weights_shrink],
            "--output",
            str(out_c),
        ]
    )

    manifest = {
        "candidate_a": {
            "path": str(out_a.relative_to(ROOT)),
            "subset_indices": subset_indices,
            "weights": w_a,
            "source_score_avg": float(best_row["score_avg"]),
        },
        "candidate_b": {
            "path": str(out_b.relative_to(ROOT)),
            "weights": weights_opt,
            "source_score_avg": float(optimal["scores"]["avg"]),
        },
        "candidate_c": {
            "path": str(out_c.relative_to(ROOT)),
            "weights": weights_shrink,
        },
        "submission_order": [
            "candidate_a",
            "candidate_b",
            "candidate_c",
        ],
    }
    manifest_path = output_dir / f"{args.prefix}_candidate_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nWave-1 candidates built successfully:")
    print(f"  A: {out_a}")
    print(f"  B: {out_b}")
    print(f"  C: {out_c}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
