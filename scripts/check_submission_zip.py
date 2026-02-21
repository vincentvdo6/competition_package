"""Generic pre-submit checker for competition zips.

This checker is format-agnostic and works for both:
- export_ensemble.py style zips
- lightweight vanilla zips without ensemble_config.json

Checks:
1) size under 20MB
2) `solution.py` exists at zip root
3) model files present (`.onnx` / `.pt` / `.pth`)
4) `PredictionModel` class exists
5) prints model/weight mapping summary when parseable
"""

from __future__ import annotations

import argparse
import ast
import zipfile
from pathlib import Path
from typing import Any


MAX_SIZE_MB = 20.0


def _extract_literal_assignment(tree: ast.AST, name: str) -> Any | None:
    for node in getattr(tree, "body", []):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        if node.targets[0].id != name:
            continue
        try:
            return ast.literal_eval(node.value)
        except Exception:
            return None
    return None


def _has_prediction_model(tree: ast.AST) -> tuple[bool, str]:
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.ClassDef) and node.name == "PredictionModel":
            init_sig = "unknown"
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    arg_names = [a.arg for a in item.args.args]
                    defaults = item.args.defaults
                    if len(arg_names) >= 2 and arg_names[1] == "model_path":
                        if defaults:
                            init_sig = "__init__(self, model_path='...')"
                        else:
                            init_sig = "__init__(self, model_path)"
                    else:
                        init_sig = f"__init__ args={arg_names}"
            return True, init_sig
    return False, "missing"


def _summarize_model_configs(solution_tree: ast.AST) -> list[str]:
    lines: list[str] = []

    cfg = _extract_literal_assignment(solution_tree, "MODEL_CONFIGS")
    cfg_t0 = _extract_literal_assignment(solution_tree, "MODEL_CONFIGS_T0")
    cfg_t1 = _extract_literal_assignment(solution_tree, "MODEL_CONFIGS_T1")

    if isinstance(cfg, list):
        lines.append(f"MODEL_CONFIGS entries: {len(cfg)}")
        if cfg and isinstance(cfg[0], tuple):
            tlen = len(cfg[0])
            lines.append(f"MODEL_CONFIGS tuple_len: {tlen}")
            if tlen >= 5:
                try:
                    w0 = [float(x[3]) for x in cfg]
                    w1 = [float(x[4]) for x in cfg]
                    n0 = sum(1 for v in w0 if abs(v) > 0)
                    n1 = sum(1 for v in w1 if abs(v) > 0)
                    lines.append(f"Per-target active models: t0={n0}, t1={n1}")
                except Exception:
                    pass
            elif tlen >= 4:
                try:
                    w = [float(x[3]) for x in cfg]
                    n = sum(1 for v in w if abs(v) > 0)
                    lines.append(f"Weighted active models: {n}")
                except Exception:
                    pass
        return lines

    if isinstance(cfg_t0, list) or isinstance(cfg_t1, list):
        n0 = len(cfg_t0) if isinstance(cfg_t0, list) else 0
        n1 = len(cfg_t1) if isinstance(cfg_t1, list) else 0
        lines.append(f"MODEL_CONFIGS_T0 entries: {n0}")
        lines.append(f"MODEL_CONFIGS_T1 entries: {n1}")
        if n0 and isinstance(cfg_t0[0], tuple) and len(cfg_t0[0]) >= 4:
            try:
                wt0 = [float(x[3]) for x in cfg_t0]
                lines.append(
                    f"T0 active models: {sum(1 for v in wt0 if abs(v) > 0)}"
                )
            except Exception:
                pass
        if n1 and isinstance(cfg_t1[0], tuple) and len(cfg_t1[0]) >= 4:
            try:
                wt1 = [float(x[3]) for x in cfg_t1]
                lines.append(
                    f"T1 active models: {sum(1 for v in wt1 if abs(v) > 0)}"
                )
            except Exception:
                pass
        return lines

    lines.append("Model config summary: not parseable (non-literal or dynamic)")
    return lines


def check_one(path: Path) -> int:
    failures = 0
    print("=" * 72)
    print(f"CHECK: {path}")

    if not path.exists():
        print("FAIL: file not found")
        return 1

    size_mb = path.stat().st_size / 1e6
    print(f"Size: {size_mb:.2f} MB")
    if size_mb > MAX_SIZE_MB:
        print(f"FAIL: size exceeds {MAX_SIZE_MB:.0f} MB")
        failures += 1
    else:
        print("PASS: size under limit")

    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            root_names = {n for n in names if "/" not in n.rstrip("/")}
            has_solution = "solution.py" in root_names
            model_files = [
                n for n in root_names if n.endswith(".onnx") or n.endswith(".pt") or n.endswith(".pth")
            ]

            if has_solution:
                print("PASS: solution.py at zip root")
            else:
                print("FAIL: solution.py missing at zip root")
                failures += 1

            print(f"Model files at root: {len(model_files)}")
            if not model_files:
                print("FAIL: no model files found at zip root")
                failures += 1

            if has_solution:
                source = zf.read("solution.py").decode("utf-8", errors="ignore")
                try:
                    tree = ast.parse(source)
                    ok_cls, init_sig = _has_prediction_model(tree)
                    if ok_cls:
                        print(f"PASS: PredictionModel found ({init_sig})")
                    else:
                        print("FAIL: PredictionModel class missing")
                        failures += 1

                    for line in _summarize_model_configs(tree):
                        print(f"INFO: {line}")
                except SyntaxError as exc:
                    print(f"FAIL: solution.py syntax error: {exc}")
                    failures += 1
    except zipfile.BadZipFile:
        print("FAIL: invalid zip file")
        failures += 1

    if failures == 0:
        print("RESULT: PASS")
    else:
        print(f"RESULT: FAIL ({failures} issue(s))")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Generic submission zip checker")
    parser.add_argument("zips", nargs="+", help="Zip paths to check")
    args = parser.parse_args()

    total_failures = 0
    for p in args.zips:
        total_failures += check_one(Path(p))

    print("=" * 72)
    if total_failures == 0:
        print("ALL CHECKS PASSED")
        return 0
    print(f"CHECKS FAILED: {total_failures} total issue(s)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
