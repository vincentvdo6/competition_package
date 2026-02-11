"""Test ONNX export of GRU model for step-by-step inference.

Exports a GRU model to ONNX, then benchmarks ONNX vs PyTorch
for step-by-step (stateful) inference speed.
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.models.gru_baseline import GRUBaseline


class GRUStepWrapper(nn.Module):
    """Wrapper that makes GRU forward_step ONNX-exportable.

    Takes explicit hidden state input (no None) and returns
    (prediction, new_hidden) with fixed shapes.
    """

    def __init__(self, model: GRUBaseline):
        super().__init__()
        self.input_proj = model.input_proj
        self.input_norm = model.input_norm
        self.gru = model.gru
        self.output_proj = model.output_proj

    def forward(self, x: torch.Tensor, hidden: torch.Tensor):
        """
        Args:
            x: (1, input_size) - single step input
            hidden: (num_layers, 1, hidden_size) - GRU hidden state
        Returns:
            prediction: (1, 2)
            new_hidden: (num_layers, 1, hidden_size)
        """
        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x)
        # No dropout in eval mode

        # Add seq dim: (1, input_size) -> (1, 1, hidden_size)
        x = x.unsqueeze(1)

        # GRU step
        gru_out, new_hidden = self.gru(x, hidden)

        # Output projection: (1, 1, hidden_size) -> (1, 2)
        prediction = self.output_proj(gru_out.squeeze(1))

        return prediction, new_hidden


def export_to_onnx(model, input_size, hidden_size, num_layers, onnx_path):
    """Export GRU model to ONNX for step-by-step inference."""
    wrapper = GRUStepWrapper(model)
    wrapper.eval()

    # Dummy inputs
    dummy_x = torch.randn(1, input_size)
    dummy_h = torch.zeros(num_layers, 1, hidden_size)

    # Export
    torch.onnx.export(
        wrapper,
        (dummy_x, dummy_h),
        onnx_path,
        input_names=["input", "hidden_in"],
        output_names=["prediction", "hidden_out"],
        dynamic_axes=None,  # Fixed shapes for maximum optimization
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Exported to {onnx_path} ({os.path.getsize(onnx_path)/1024:.1f} KB)")


def benchmark_pytorch(model, input_size, n_steps=2000):
    """Benchmark PyTorch step-by-step inference."""
    model.eval()
    hidden = None

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_steps):
            x = torch.randn(1, input_size)
            pred, hidden = model.forward_step(x, hidden)
    elapsed = time.perf_counter() - t0

    us_per_step = elapsed / n_steps * 1e6
    print(f"PyTorch: {n_steps} steps in {elapsed*1000:.0f}ms ({us_per_step:.0f}us/step)")
    return us_per_step


def benchmark_onnx(onnx_path, input_size, hidden_size, num_layers, n_steps=2000):
    """Benchmark ONNX step-by-step inference."""
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = ort.InferenceSession(onnx_path, sess_options, providers=["CPUExecutionProvider"])

    hidden = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)

    t0 = time.perf_counter()
    for _ in range(n_steps):
        x = np.random.randn(1, input_size).astype(np.float32)
        prediction, hidden = sess.run(
            ["prediction", "hidden_out"],
            {"input": x, "hidden_in": hidden}
        )
    elapsed = time.perf_counter() - t0

    us_per_step = elapsed / n_steps * 1e6
    print(f"ONNX:    {n_steps} steps in {elapsed*1000:.0f}ms ({us_per_step:.0f}us/step)")
    return us_per_step


def verify_correctness(model, onnx_path, input_size, hidden_size, num_layers, n_steps=50):
    """Verify ONNX output matches PyTorch."""
    import onnxruntime as ort

    model.eval()
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess = ort.InferenceSession(onnx_path, sess_options, providers=["CPUExecutionProvider"])

    pt_hidden = None
    ort_hidden = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
    max_diff = 0.0

    for step in range(n_steps):
        x_np = np.random.randn(1, input_size).astype(np.float32)
        x_pt = torch.from_numpy(x_np)

        # PyTorch
        with torch.no_grad():
            pt_pred, pt_hidden = model.forward_step(x_pt, pt_hidden)
        pt_pred_np = pt_pred.numpy()

        # ONNX
        ort_pred, ort_hidden = sess.run(
            ["prediction", "hidden_out"],
            {"input": x_np, "hidden_in": ort_hidden}
        )

        diff = np.abs(pt_pred_np - ort_pred).max()
        max_diff = max(max_diff, diff)

    print(f"Max diff over {n_steps} steps: {max_diff:.2e}")
    return max_diff


def main():
    # Use tightwd_v2 config (our best GRU)
    config = {
        "model": {
            "type": "gru",
            "input_size": 42,
            "hidden_size": 144,
            "num_layers": 2,
            "dropout": 0.22,
            "output_size": 2,
        }
    }

    model = GRUBaseline(config)

    # Try to load a real checkpoint
    ckpt_path = os.path.join(ROOT, "logs", "_staging", "gru_derived_tightwd_v2_seed50.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("No checkpoint found, using random weights")

    model.eval()

    input_size = config["model"]["input_size"]
    hidden_size = config["model"]["hidden_size"]
    num_layers = config["model"]["num_layers"]

    print(f"Model: GRU, input={input_size}, hidden={hidden_size}, layers={num_layers}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Export to ONNX
    onnx_path = os.path.join(ROOT, "logs", "gru_step_test.onnx")
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    export_to_onnx(model, input_size, hidden_size, num_layers, onnx_path)
    print()

    # Verify correctness
    print("=== Correctness ===")
    verify_correctness(model, onnx_path, input_size, hidden_size, num_layers)
    print()

    # Benchmark
    n_steps = 5000
    print(f"=== Benchmark ({n_steps} steps) ===")
    pt_us = benchmark_pytorch(model, input_size, n_steps)
    ort_us = benchmark_onnx(onnx_path, input_size, hidden_size, num_layers, n_steps)

    speedup = pt_us / ort_us
    print(f"Speedup: {speedup:.1f}x")
    print()

    # Timing projections
    steps_total = 1_440_000  # ~1.44M steps for 1500 sequences
    pt_time = pt_us * steps_total / 1e6
    ort_time = ort_us * steps_total / 1e6
    print("=== Server Time Projections (1.44M steps) ===")
    print(f"PyTorch per model: {pt_time:.0f}s")
    print(f"ONNX per model:    {ort_time:.0f}s")
    print(f"5 ONNX GRU models: {5 * ort_time:.0f}s")
    print(f"10 ONNX GRU models: {10 * ort_time:.0f}s")
    print(f"15 ONNX GRU models: {15 * ort_time:.0f}s")

    # Clean up
    os.remove(onnx_path)


if __name__ == "__main__":
    main()
