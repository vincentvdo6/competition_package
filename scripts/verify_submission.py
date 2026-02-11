"""Verify a submission zip won't fail on the scoring server.

Checks:
1. State dict key matching for every model
2. Normalizer dimensions match feature pipeline and model input_size
3. TCNFast conversion + numerical correctness vs PyTorch
4. Full PredictionModel integration (warmup, predict, seq reset)
5. Timing estimate
"""

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
import zipfile

import numpy as np
import torch


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_submission.py <path_to_zip>")
        sys.exit(1)

    zip_path = sys.argv[1]
    if not os.path.exists(zip_path):
        print(f"File not found: {zip_path}")
        sys.exit(1)

    print(f"Verifying: {zip_path}")
    print(f"Size: {os.path.getsize(zip_path) / 1e6:.2f} MB (limit 20MB)")
    if os.path.getsize(zip_path) > 20 * 1e6:
        print("FAIL: Zip exceeds 20MB limit!")
        sys.exit(1)
    print()

    tmpdir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmpdir)

        # Load solution module
        sys.path.insert(0, tmpdir)
        spec = importlib.util.spec_from_file_location(
            "solution", os.path.join(tmpdir, "solution.py")
        )
        sol = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sol)

        with open(os.path.join(tmpdir, "ensemble_config.json")) as f:
            cfg = json.load(f)

        n_models = len(cfg["models"])
        all_pass = True

        # CHECK 1: State dict key matching / ONNX loading
        print("=" * 60)
        print("CHECK 1: State dict key matching / ONNX loading")
        print("=" * 60)
        for i, model_spec in enumerate(cfg["models"]):
            model_type = model_spec["model"]["type"]
            inference = model_spec["model"].get("inference", "pytorch")
            ckpt_path = os.path.join(tmpdir, model_spec["checkpoint"])

            if inference == "onnx":
                # Verify ONNX model loads with onnxruntime
                try:
                    import onnxruntime as ort
                    sess = ort.InferenceSession(
                        ckpt_path,
                        providers=["CPUExecutionProvider"],
                    )
                    inputs = {inp.name: inp for inp in sess.get_inputs()}
                    outputs = {out.name: out for out in sess.get_outputs()}
                    has_io = "input" in inputs and "hidden_in" in inputs
                    has_io = has_io and "prediction" in outputs and "hidden_out" in outputs
                    if has_io:
                        print(f"  [{i}] {model_type} (ONNX): OK")
                    else:
                        print(f"  [{i}] {model_type} (ONNX): BAD IO names")
                        all_pass = False
                except Exception as e:
                    print(f"  [{i}] {model_type} (ONNX): FAIL - {e}")
                    all_pass = False
                continue

            model = sol.build_model(model_spec["model"])
            ckpt = sol._safe_torch_load(ckpt_path)
            sd = ckpt["model_state_dict"]

            model_keys = set(model.state_dict().keys())
            ckpt_keys = set(sd.keys())
            missing = model_keys - ckpt_keys
            unexpected = ckpt_keys - model_keys

            if missing or unexpected:
                print(f"  [{i}] {model_type}: MISMATCH!")
                if missing:
                    print(f"    Missing: {missing}")
                if unexpected:
                    print(f"    Unexpected: {unexpected}")
                all_pass = False
            else:
                model.load_state_dict(sd)
                # Check shapes
                shape_ok = True
                for key in model_keys:
                    if model.state_dict()[key].shape != sd[key].shape:
                        print(f"    SHAPE MISMATCH: {key}")
                        shape_ok = False
                        all_pass = False
                if shape_ok:
                    print(f"  [{i}] {model_type}: OK ({len(model_keys)} keys)")
        print()

        # CHECK 2: Normalizer dimensions
        print("=" * 60)
        print("CHECK 2: Normalizer dimensions")
        print("=" * 60)
        for i, model_spec in enumerate(cfg["models"]):
            norm_path = os.path.join(tmpdir, model_spec["normalizer"])
            data = np.load(norm_path)
            mean_len = len(data["mean"])

            dcfg = model_spec["data"]
            expected_dim = 32
            if dcfg.get("derived_features"):
                expected_dim += 10
            if dcfg.get("temporal_features"):
                expected_dim += 3
            if dcfg.get("interaction_features"):
                expected_dim += 3
            if dcfg.get("microstructure_features"):
                expected_dim += 6

            input_size = model_spec["model"]["input_size"]
            match = mean_len == expected_dim == input_size
            status = "OK" if match else "MISMATCH!"
            if not match:
                all_pass = False
            print(
                f"  [{i}] norm={mean_len}, pipeline={expected_dim}, "
                f"model={input_size}: {status}"
            )
        print()

        # CHECK 3: TCNFast conversion
        print("=" * 60)
        print("CHECK 3: TCNFast conversion & numerical correctness")
        print("=" * 60)
        has_tcn = False
        for i, model_spec in enumerate(cfg["models"]):
            if model_spec["model"]["type"] != "tcn":
                continue
            has_tcn = True
            ckpt_path = os.path.join(tmpdir, model_spec["checkpoint"])
            ckpt = sol._safe_torch_load(ckpt_path)

            pt_model = sol.build_model(model_spec["model"])
            pt_model.load_state_dict(ckpt["model_state_dict"])
            pt_model.eval()

            if not hasattr(sol, "TCNFast"):
                print(f"  [{i}] FAIL: TCNFast class not found in solution.py!")
                all_pass = False
                continue

            fast_model = sol.TCNFast(pt_model)

            pt_hidden = None
            fast_hidden = None
            max_diff = 0.0

            for step in range(20):
                x = torch.randn(1, 42)
                with torch.no_grad():
                    pt_pred, pt_hidden = pt_model.forward_step(
                        x, pt_hidden, need_pred=True
                    )
                fast_pred, fast_hidden = fast_model.forward_step(
                    x, fast_hidden, need_pred=True
                )

                pt_np = pt_pred.squeeze(0).numpy()
                fast_np = (
                    fast_pred
                    if isinstance(fast_pred, np.ndarray)
                    else fast_pred.numpy()
                )
                diff = np.abs(pt_np - fast_np).max()
                max_diff = max(max_diff, diff)

            if max_diff < 1e-4:
                print(f"  [{i}] TCN max diff: {max_diff:.2e} PASS")
            else:
                print(f"  [{i}] TCN max diff: {max_diff:.2e} WARNING")

        if not has_tcn:
            print("  No TCN models in ensemble (skip)")
        print()

        # CHECK 4: Full PredictionModel integration
        print("=" * 60)
        print("CHECK 4: Full PredictionModel integration")
        print("=" * 60)
        model = sol.PredictionModel()
        print(f"  Models loaded: {model.n_models}")

        # Verify model types are correct (ONNX, TCNFast, etc.)
        for i in range(model.n_models):
            cls = type(model.models[i]).__name__
            mtype = cfg["models"][i]["model"]["type"]
            inference = cfg["models"][i]["model"].get("inference", "pytorch")
            if inference == "onnx" and cls != "OnnxGRU":
                print(f"  [{i}] FAIL: ONNX model not loaded as OnnxGRU (is {cls})")
                all_pass = False
            elif mtype == "tcn" and inference != "onnx" and cls != "TCNFast":
                print(f"  [{i}] FAIL: TCN not converted to TCNFast (is {cls})")
                all_pass = False
            else:
                print(f"  [{i}] {cls}")

        class FakeDP:
            def __init__(self, seq_ix, state, need_pred):
                self.seq_ix = seq_ix
                self.state = state
                self.need_prediction = need_pred

        # Warmup
        for step in range(99):
            dp = FakeDP(0, np.random.randn(32).astype(np.float32), False)
            result = model.predict(dp)
            if result is not None:
                print(f"  FAIL: warmup step {step} returned {result}")
                all_pass = False
                break
        else:
            print("  Warmup (99 steps): OK")

        # Predictions
        pred_ok = True
        for step in range(100):
            dp = FakeDP(0, np.random.randn(32).astype(np.float32), True)
            result = model.predict(dp)
            if result is None:
                print(f"  FAIL: prediction step {step} returned None")
                pred_ok = False
                all_pass = False
                break
            if result.shape != (2,):
                print(f"  FAIL: wrong shape {result.shape}")
                pred_ok = False
                all_pass = False
                break
            if not np.all(np.isfinite(result)):
                print(f"  FAIL: non-finite {result}")
                pred_ok = False
                all_pass = False
                break
            if not np.all(np.abs(result) <= 6.0):
                print(f"  FAIL: outside clip range {result}")
                pred_ok = False
                all_pass = False
                break
        if pred_ok:
            print("  Predictions (100 steps): OK")

        # Seq reset
        dp = FakeDP(1, np.random.randn(32).astype(np.float32), True)
        result = model.predict(dp)
        if result is not None and result.shape == (2,):
            print("  Seq reset: OK")
        else:
            print("  FAIL: seq reset")
            all_pass = False

        # Multi-sequence
        multi_ok = True
        for seq in range(5):
            for step in range(10):
                need = step >= 3
                dp = FakeDP(
                    seq + 10, np.random.randn(32).astype(np.float32), need
                )
                result = model.predict(dp)
                if not need and result is not None:
                    multi_ok = False
                if need and (result is None or result.shape != (2,)):
                    multi_ok = False
        print(f"  Multi-sequence (5 seqs): {'OK' if multi_ok else 'FAIL'}")
        if not multi_ok:
            all_pass = False
        print()

        # CHECK 5: Timing
        print("=" * 60)
        print("CHECK 5: Timing estimate")
        print("=" * 60)
        model2 = sol.PredictionModel()
        n_steps = 500
        t0 = time.time()
        for step in range(n_steps):
            need = step >= 5
            dp = FakeDP(0, np.random.randn(32).astype(np.float32), need)
            model2.predict(dp)
        elapsed = time.time() - t0
        us_per_step = elapsed / n_steps * 1e6
        print(f"  Local: {n_steps} steps in {elapsed * 1000:.0f}ms ({us_per_step:.0f}us/step)")

        # Component-based server time estimate
        # Per-model estimates (from LB calibration):
        # PyTorch GRU: ~320s, ONNX GRU: ~45s, PyTorch Attn: ~800s, TCNFast: ~40s
        n_onnx_gru = sum(1 for m in cfg["models"]
                         if m["model"]["type"] == "gru" and m["model"].get("inference") == "onnx")
        n_pt_gru = sum(1 for m in cfg["models"]
                       if m["model"]["type"] == "gru" and m["model"].get("inference") != "onnx")
        n_attn = sum(1 for m in cfg["models"] if m["model"]["type"] == "gru_attention")
        n_tcn = sum(1 for m in cfg["models"] if m["model"]["type"] == "tcn")
        est_server = n_onnx_gru * 45 + n_pt_gru * 320 + n_attn * 800 + n_tcn * 40
        margin = (4200 - est_server) / 4200 * 100
        print(f"  Model breakdown: {n_onnx_gru} ONNX GRU, {n_pt_gru} PT GRU, {n_attn} Attn, {n_tcn} TCN")
        print(f"  Estimated server time: ~{est_server:.0f}s")
        print(f"  Time limit: 4200s")
        print(f"  Margin: {margin:.1f}%")
        if margin > 20:
            print("  SAFE")
        elif margin > 10:
            print("  ACCEPTABLE")
        else:
            print("  WARNING: tight margin!")
            all_pass = False
        print()

        # FINAL
        print("=" * 60)
        if all_pass:
            print("ALL 5 CHECKS PASSED - SAFE TO SUBMIT")
        else:
            print("SOME CHECKS FAILED - DO NOT SUBMIT")
        print("=" * 60)

    finally:
        try:
            shutil.rmtree(tmpdir)
        except PermissionError:
            pass  # Windows file lock on .npz; harmless


if __name__ == "__main__":
    main()
