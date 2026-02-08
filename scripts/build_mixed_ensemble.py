#!/usr/bin/env python
"""Build a mixed ensemble from old (combined-loss) and new (pearson-loss) checkpoints.

Usage:
  1. Place slim_checkpoints_pearson.zip (from Kaggle) in Downloads or project root
  2. Run: python scripts/build_mixed_ensemble.py --new-zip <path_to_pearson_zip>

This script:
  - Extracts old checkpoints from the existing gru5_attn3_uniform8.zip
  - Extracts new checkpoints from the pearson training zip
  - Runs export_ensemble.py to build the final submission
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Old checkpoints source
OLD_ZIP_CANDIDATES = [
    Path(r"C:\Users\Vincent\Downloads\gru5_attn3_uniform8.zip"),
    ROOT / "gru5_attn3_uniform8.zip",
]

# Mapping from old zip model_N.pt to config + normalizer name
# Based on ensemble_config.json: models 0-4 are GRU, models 5-7 are GRU attention
OLD_MODELS = {
    # 5x GRU tightwd_v2 (seeds 42-46, combined loss)
    "model_0.pt": ("gru_derived_tightwd_v2", "normalizer.npz"),
    "model_1.pt": ("gru_derived_tightwd_v2", "normalizer_1.npz"),
    "model_2.pt": ("gru_derived_tightwd_v2", "normalizer_2.npz"),
    "model_3.pt": ("gru_derived_tightwd_v2", "normalizer_3.npz"),
    "model_4.pt": ("gru_derived_tightwd_v2", "normalizer_4.npz"),
    # 3x GRU attention clean (seeds 42-44, combined loss)
    "model_5.pt": ("gru_attention_clean_v1", "normalizer_5.npz"),
    "model_6.pt": ("gru_attention_clean_v1", "normalizer_6.npz"),
    "model_7.pt": ("gru_attention_clean_v1", "normalizer_7.npz"),
}

# New pearson models expected in the new zip
NEW_MODELS = {
    # 3x GRU pearson (seeds 42-44)
    "gru_pearson_v1_seed42.pt": ("gru_pearson_v1", "normalizer_gru_pearson_v1_seed42.npz"),
    "gru_pearson_v1_seed43.pt": ("gru_pearson_v1", "normalizer_gru_pearson_v1_seed43.npz"),
    "gru_pearson_v1_seed44.pt": ("gru_pearson_v1", "normalizer_gru_pearson_v1_seed44.npz"),
    # 2x GRU attention clean (seeds 45-46)
    "gru_attention_clean_v1_seed45.pt": ("gru_attention_clean_v1", "normalizer_gru_attention_clean_v1_seed45.npz"),
    "gru_attention_clean_v1_seed46.pt": ("gru_attention_clean_v1", "normalizer_gru_attention_clean_v1_seed46.npz"),
    # 2x GRU attention pearson (seeds 42-43)
    "gru_attention_pearson_v1_seed42.pt": ("gru_attention_pearson_v1", "normalizer_gru_attention_pearson_v1_seed42.npz"),
    "gru_attention_pearson_v1_seed43.pt": ("gru_attention_pearson_v1", "normalizer_gru_attention_pearson_v1_seed43.npz"),
}

# Ensemble presets
PRESETS = {
    "full15": {
        "desc": "All 15 models (5 GRU + 3 attn + 3 GRU pearson + 2 attn clean + 2 attn pearson)",
        "old": list(OLD_MODELS.keys()),
        "new": list(NEW_MODELS.keys()),
    },
    "diverse10": {
        "desc": "10 models: 5 GRU combined + 3 GRU pearson + 2 attn pearson (loss+arch diversity, no old attn)",
        "old": ["model_0.pt", "model_1.pt", "model_2.pt", "model_3.pt", "model_4.pt"],
        "new": [
            "gru_pearson_v1_seed42.pt", "gru_pearson_v1_seed43.pt", "gru_pearson_v1_seed44.pt",
            "gru_attention_pearson_v1_seed42.pt", "gru_attention_pearson_v1_seed43.pt",
        ],
    },
    "fast8_gru": {
        "desc": "8 GRU-only models: 5 combined + 3 pearson (fastest, no attention overhead)",
        "old": ["model_0.pt", "model_1.pt", "model_2.pt", "model_3.pt", "model_4.pt"],
        "new": ["gru_pearson_v1_seed42.pt", "gru_pearson_v1_seed43.pt", "gru_pearson_v1_seed44.pt"],
    },
    "balanced7": {
        "desc": "7 models: 5 GRU combined + 2 attn pearson (arch+loss diversity, budget-friendly)",
        "old": ["model_0.pt", "model_1.pt", "model_2.pt", "model_3.pt", "model_4.pt"],
        "new": ["gru_attention_pearson_v1_seed42.pt", "gru_attention_pearson_v1_seed43.pt"],
    },
    "diverse12": {
        "desc": "12 models: 5 GRU + 2 attn combined + 3 GRU pearson + 2 attn pearson (max diversity, tight timing)",
        "old": list(OLD_MODELS.keys())[:7],  # 5 GRU + 2 of 3 attn combined
        "new": [
            "gru_pearson_v1_seed42.pt", "gru_pearson_v1_seed43.pt", "gru_pearson_v1_seed44.pt",
            "gru_attention_pearson_v1_seed42.pt", "gru_attention_pearson_v1_seed43.pt",
        ],
    },
}


def find_old_zip() -> Path:
    for p in OLD_ZIP_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Cannot find gru5_attn3_uniform8.zip. Looked in:\n"
        + "\n".join(f"  {p}" for p in OLD_ZIP_CANDIDATES)
    )


def extract_old_models(old_zip: Path, staging: Path, model_keys: list[str]) -> None:
    with zipfile.ZipFile(old_zip, "r") as zf:
        for key in model_keys:
            _, norm_name = OLD_MODELS[key]
            zf.extract(key, staging)
            zf.extract(norm_name, staging)
            print(f"  Extracted old: {key}, {norm_name}")


def extract_new_models(new_zip: Path, staging: Path, model_keys: list[str]) -> None:
    with zipfile.ZipFile(new_zip, "r") as zf:
        for key in model_keys:
            _, norm_name = NEW_MODELS[key]
            zf.extract(key, staging)
            zf.extract(norm_name, staging)
            print(f"  Extracted new: {key}, {norm_name}")


def main():
    parser = argparse.ArgumentParser(description="Build mixed ensemble submission")
    parser.add_argument("--new-zip", type=str, required=True, help="Path to slim_checkpoints_pearson.zip")
    parser.add_argument("--old-zip", type=str, default=None, help="Path to gru5_attn3_uniform8.zip (auto-detected if not given)")
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(PRESETS.keys()),
        default="diverse10",
        help="Ensemble preset (default: diverse10)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output zip name")
    parser.add_argument("--list-presets", action="store_true", help="List available presets and exit")
    args = parser.parse_args()

    if args.list_presets:
        for name, preset in PRESETS.items():
            n = len(preset["old"]) + len(preset["new"])
            print(f"  {name:15s} ({n:2d} models): {preset['desc']}")
        return

    new_zip = Path(args.new_zip)
    if not new_zip.exists():
        print(f"Error: new zip not found: {new_zip}")
        sys.exit(1)

    old_zip = Path(args.old_zip) if args.old_zip else find_old_zip()
    preset = PRESETS[args.preset]
    output_name = args.output or f"submissions/{args.preset}_mixed.zip"

    n_total = len(preset["old"]) + len(preset["new"])
    print(f"Preset: {args.preset} ({n_total} models)")
    print(f"  {preset['desc']}")
    print(f"  Old zip: {old_zip}")
    print(f"  New zip: {new_zip}")
    print()

    # Stage all checkpoints into a temp dir
    staging = ROOT / "logs" / "_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    print("Extracting checkpoints...")
    extract_old_models(old_zip, staging, preset["old"])
    extract_new_models(new_zip, staging, preset["new"])

    # Build args for export_ensemble.py
    checkpoints = []
    configs = []
    normalizers = []

    for key in preset["old"]:
        cfg_name, norm_name = OLD_MODELS[key]
        checkpoints.append(str(staging / key))
        configs.append(f"configs/{cfg_name}.yaml")
        normalizers.append(str(staging / norm_name))

    for key in preset["new"]:
        cfg_name, norm_name = NEW_MODELS[key]
        checkpoints.append(str(staging / key))
        configs.append(f"configs/{cfg_name}.yaml")
        normalizers.append(str(staging / norm_name))

    print(f"\nRunning export_ensemble.py with {n_total} models...")
    cmd = [
        sys.executable, str(ROOT / "scripts" / "export_ensemble.py"),
        "--configs", *configs,
        "--checkpoints", *checkpoints,
        "--normalizers", *normalizers,
        "--output", output_name,
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        sys.exit(1)

    # Cleanup staging
    shutil.rmtree(staging)
    print("Done! Staging cleaned up.")


if __name__ == "__main__":
    main()
