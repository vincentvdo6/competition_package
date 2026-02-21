#!/usr/bin/env python
"""Colab-friendly runner for parity cache + swap discovery + zip build.

Designed to be run from repo root in Colab with streaming subprocess logs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_stream(cmd: list[str], cwd: Path) -> None:
    print(f"\n$ {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed (exit {rc}): {' '.join(cmd)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Colab parity swap pipeline")
    parser.add_argument("--repo", default=".", help="Repo root path")
    parser.add_argument("--cache-dir", default="cache/parity23_valid_preds")
    parser.add_argument("--data", default="datasets/valid.parquet")
    parser.add_argument("--output-report", default="logs/parity_swap_discovery_feb21_b2.json")
    parser.add_argument("--output-dir", default="submissions/ready")
    parser.add_argument("--slot-a-name", default="feb21-b2-t1swap-a-onnx.zip")
    parser.add_argument("--slot-b-name", default="feb21-b2-t0swap-a-onnx.zip")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--delta-min", type=float, default=0.00015)
    parser.add_argument("--p10-max-drop", type=float, default=0.00005)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    if not (repo / "scripts" / "greedy_vanilla_ensemble.py").exists():
        raise FileNotFoundError(f"Not a repo root or missing scripts: {repo}")

    checkpoints = [
        str((repo / "logs" / "vanilla_all" / f"gru_parity_v1_seed{seed}.pt").resolve())
        for seed in range(42, 65)
    ]
    for ckpt in checkpoints:
        if not Path(ckpt).exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    if not (repo / args.data).exists():
        raise FileNotFoundError(f"Missing valid parquet: {repo / args.data}")

    run_stream(
        [
            sys.executable,
            "scripts/greedy_vanilla_ensemble.py",
            "cache",
            "--checkpoints",
            *checkpoints,
            "--data",
            args.data,
            "--cache-dir",
            args.cache_dir,
        ],
        cwd=repo,
    )

    run_stream(
        [
            sys.executable,
            "scripts/discover_parity_swaps.py",
            "--cache-dir",
            args.cache_dir,
            "--data",
            args.data,
            "--bootstrap",
            str(args.bootstrap),
            "--delta-min",
            str(args.delta_min),
            "--p10-max-drop",
            str(args.p10_max_drop),
            "--output-report",
            args.output_report,
            "--output-dir",
            args.output_dir,
            "--slot-a-name",
            args.slot_a_name,
            "--slot-b-name",
            args.slot_b_name,
        ],
        cwd=repo,
    )

    print("\nPipeline complete.")
    print(f"Report: {repo / args.output_report}")
    print(f"Slot A: {repo / args.output_dir / args.slot_a_name}")
    print(f"Slot B: {repo / args.output_dir / args.slot_b_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
