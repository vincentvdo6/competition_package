"""Build a submission zip for vanilla GRU ensemble (multiple seeds, uniform weights).

Supports both PyTorch and ONNX inference modes:
  --onnx: Export to ONNX for ~2.5x faster inference (fit 20+ models in budget)
"""

import argparse
import os
import shutil
import tempfile
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# ONNX export helper
# ---------------------------------------------------------------------------
class VanillaGRUStep(nn.Module):
    """Minimal wrapper for ONNX export: raw features -> GRU -> linear."""

    def __init__(self, gru, fc):
        super().__init__()
        self.gru = gru
        self.fc = fc

    def forward(self, x: torch.Tensor, hidden: torch.Tensor):
        x = x.unsqueeze(1)  # (1, input_size) -> (1, 1, input_size)
        out, new_hidden = self.gru(x, hidden)
        return self.fc(out.squeeze(1)), new_hidden


def _export_vanilla_to_onnx(ckpt_path, hidden_size, num_layers, onnx_path):
    """Export a vanilla GRU checkpoint to ONNX for step-by-step inference."""
    from torch.nn import GRU, Linear

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]

    # Build vanilla GRU and load weights
    gru = GRU(input_size=32, hidden_size=hidden_size, num_layers=num_layers,
              batch_first=True, dropout=0.0, bidirectional=False)
    fc = Linear(hidden_size, 2)

    gru_sd = {k.replace("gru.", ""): v for k, v in state_dict.items() if k.startswith("gru.")}
    fc_sd = {}
    for k, v in state_dict.items():
        if k.startswith("output_proj."):
            fc_sd[k.replace("output_proj.", "")] = v
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


# ---------------------------------------------------------------------------
# Solution templates
# ---------------------------------------------------------------------------
SOLUTION_TEMPLATE = '''"""Vanilla GRU ensemble: {n_models} models, raw 32 features, no normalization."""

import os
import numpy as np
import torch
import torch.nn as nn


class VanillaGRU(nn.Module):
    def __init__(self, input_size=32, hidden_size=64, num_layers=3, output_size=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        output, hidden = self.gru(x, hidden)
        predictions = self.fc(output)
        return predictions, hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)


# Model configs: (filename, hidden_size, num_layers, weight)
MODEL_CONFIGS = {model_configs}


class PredictionModel:
    def __init__(self, model_path=""):
        self.device = torch.device("cpu")
        torch.set_num_threads(1)

        base_dir = os.path.dirname(os.path.abspath(__file__))

        self.models = []
        self.hiddens = []
        self.weights = []

        for filename, h, nl, w in MODEL_CONFIGS:
            ckpt = torch.load(
                os.path.join(base_dir, filename),
                map_location="cpu",
                weights_only=False,
            )
            state_dict = ckpt["model_state_dict"]

            model = VanillaGRU(input_size=32, hidden_size=h, num_layers=nl)

            # Filter state_dict keys
            filtered = {{}}
            for k, v in state_dict.items():
                if k.startswith("gru."):
                    filtered[k] = v
                elif k.startswith("output_proj."):
                    new_key = k.replace("output_proj.", "fc.")
                    filtered[new_key] = v
            model.load_state_dict(filtered)
            model.eval()

            self.models.append(model)
            self.hiddens.append(model.init_hidden(1))
            self.weights.append(w)

        # Normalize weights
        total_w = sum(self.weights)
        self.weights = [w / total_w for w in self.weights]

        self.prev_seq_ix = None

    @torch.no_grad()
    def predict(self, data_point) -> np.ndarray:
        seq_ix = data_point.seq_ix
        if seq_ix != self.prev_seq_ix:
            self.hiddens = [m.init_hidden(1) for m in self.models]
            self.prev_seq_ix = seq_ix

        features = data_point.state.astype(np.float32)[:32]
        x = torch.from_numpy(features).unsqueeze(0).unsqueeze(0)

        pred_sum = np.zeros(2, dtype=np.float32)
        for i, model in enumerate(self.models):
            pred, self.hiddens[i] = model(x, self.hiddens[i])
            pred_sum += self.weights[i] * pred.squeeze().numpy()

        if not data_point.need_prediction:
            return None
        return pred_sum.clip(-6, 6)
'''


SOLUTION_TEMPLATE_ONNX = '''"""Vanilla GRU ONNX ensemble: {n_models} models, raw 32 features, no normalization."""

import os
import numpy as np
import onnxruntime as ort


# Model configs: (filename, hidden_size, num_layers, weight)
MODEL_CONFIGS = {model_configs}


class OnnxVanillaGRU:
    """ONNX Runtime GRU for step-by-step inference."""

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
        return pred[0], new_hidden  # pred shape: (1, 2) -> (2,)


class PredictionModel:
    def __init__(self, model_path=""):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        self.models = []
        self.hiddens = []
        self.weights = []

        for filename, h, nl, w in MODEL_CONFIGS:
            model = OnnxVanillaGRU(
                os.path.join(base_dir, filename), h, nl
            )
            self.models.append(model)
            self.hiddens.append(model.init_hidden())
            self.weights.append(w)

        # Normalize weights
        total_w = sum(self.weights)
        self.weights = np.array([w / total_w for w in self.weights], dtype=np.float32)

        self.prev_seq_ix = None

    def predict(self, data_point) -> np.ndarray:
        seq_ix = data_point.seq_ix
        if seq_ix != self.prev_seq_ix:
            self.hiddens = [m.init_hidden() for m in self.models]
            self.prev_seq_ix = seq_ix

        features = data_point.state.astype(np.float32)[:32].reshape(1, -1)

        pred_sum = np.zeros(2, dtype=np.float32)
        for i, model in enumerate(self.models):
            pred, self.hiddens[i] = model.run_step(features, self.hiddens[i])
            pred_sum += self.weights[i] * pred

        if not data_point.need_prediction:
            return None
        return pred_sum.clip(-6, 6)
'''


def main():
    parser = argparse.ArgumentParser(description="Build vanilla GRU ensemble submission")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Paths to model checkpoints (.pt)")
    parser.add_argument("--weights", nargs="*", type=float, default=None,
                        help="Model weights (default: uniform)")
    parser.add_argument("--hidden-size", type=int, default=64,
                        help="Hidden size (same for all models)")
    parser.add_argument("--num-layers", type=int, default=3,
                        help="Num layers (same for all models)")
    parser.add_argument("--output", default="submissions/vanilla_ensemble.zip",
                        help="Output zip path")
    parser.add_argument("--onnx", action="store_true",
                        help="Export models to ONNX for ~2.5x faster inference")
    args = parser.parse_args()

    n_models = len(args.checkpoints)
    weights = args.weights if args.weights else [1.0] * n_models
    if len(weights) != n_models:
        raise ValueError(f"Got {len(weights)} weights for {n_models} models")

    mode = "ONNX" if args.onnx else "PyTorch"
    print(f"Building vanilla ensemble: {n_models} models ({mode})")
    print(f"Architecture: h={args.hidden_size}, L={args.num_layers}")

    # Load and validate checkpoints
    ext = ".onnx" if args.onnx else ".pt"
    model_configs = []
    for i, ckpt_path in enumerate(args.checkpoints):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        score = ckpt.get("best_score", 0)
        epoch = ckpt.get("best_epoch", "?")
        filename = f"model_{i}{ext}"
        model_configs.append((filename, args.hidden_size, args.num_layers, weights[i]))
        print(f"  [{i}] {os.path.basename(ckpt_path)}: val={score:.4f}, epoch={epoch}, w={weights[i]:.3f}")

    # Estimate runtime
    est_per_model = 74 if args.onnx else 184  # seconds on scoring server
    est_total = n_models * est_per_model
    margin = 1.0 - est_total / 4200
    print(f"\nEstimated runtime: {n_models} x {est_per_model}s = {est_total}s ({margin*100:.0f}% margin)")
    if margin < 0.20:
        print("WARNING: Tight margin! Consider fewer models or ONNX export.")

    # Build zip
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write solution.py
        template = SOLUTION_TEMPLATE_ONNX if args.onnx else SOLUTION_TEMPLATE
        solution_code = template.format(
            n_models=n_models,
            model_configs=repr(model_configs),
        )
        with open(os.path.join(tmpdir, "solution.py"), "w") as f:
            f.write(solution_code)

        # Write model files
        for i, ckpt_path in enumerate(args.checkpoints):
            if args.onnx:
                onnx_path = os.path.join(tmpdir, f"model_{i}.onnx")
                print(f"  Exporting model_{i}.onnx ...", end=" ", flush=True)
                _export_vanilla_to_onnx(
                    ckpt_path, args.hidden_size, args.num_layers, onnx_path
                )
                size_kb = os.path.getsize(onnx_path) / 1024
                print(f"({size_kb:.0f}KB)")
            else:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                slim = {"model_state_dict": ckpt["model_state_dict"]}
                torch.save(slim, os.path.join(tmpdir, f"model_{i}.pt"))

        # Create zip
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        base = args.output.replace(".zip", "")
        shutil.make_archive(base, "zip", tmpdir)

    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\nCreated: {args.output} ({size_mb:.1f}MB)")
    if size_mb > 20:
        print("WARNING: Exceeds 20MB submission limit!")
    else:
        print(f"Size OK ({size_mb:.1f}MB < 20MB limit)")

    # Summary
    scores = []
    for ckpt_path in args.checkpoints:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        scores.append(ckpt.get("best_score", 0))
    mean_val = sum(scores) / len(scores)
    print(f"\nMean val of ensemble members: {mean_val:.4f}")
    print(f"Best single val: {max(scores):.4f}")


if __name__ == "__main__":
    main()
