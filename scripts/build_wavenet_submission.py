"""Build submission zip for WaveNet ensemble (PyTorch step-by-step with ring buffers).

Usage:
  Single canary:
    python scripts/build_wavenet_submission.py --checkpoints logs/wavenet_v1_seed42.pt --output submissions/wavenet_canary.zip

  Ensemble:
    python scripts/build_wavenet_submission.py --checkpoints logs/wavenet_v1_seed*.pt --output submissions/wavenet_ens10.zip
"""

import argparse
import os
import shutil
import tempfile
import torch


# ---------------------------------------------------------------------------
# Solution template: pure numpy for max speed on 1-vCPU scoring server
# ---------------------------------------------------------------------------
SOLUTION_TEMPLATE = '''"""WaveNet-lite ensemble: {n_models} models, ring buffer inference."""

import os
import numpy as np


class WaveNetNumpy:
    """Pure numpy WaveNet for step-by-step inference with ring buffers."""

    def __init__(self, state_dict, channels=64, skip_channels=96,
                 dilations=None, res_scale=0.3):
        self.channels = channels
        self.skip_channels = skip_channels
        self.dilations = dilations or [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        self.res_scale = res_scale
        self.n_blocks = len(self.dilations)

        # Extract weights as numpy arrays
        # Input projection (Conv1d 32->C, kernel=1): weight (C, 32, 1), bias (C,)
        self.inp_w = state_dict["input_proj.weight"].numpy().squeeze(-1)  # (C, 32)
        self.inp_b = state_dict["input_proj.bias"].numpy()  # (C,)

        # Per-block weights
        self.filter_w = []  # (C, C, 2) for kernel_size=2
        self.filter_b = []
        self.gate_w = []
        self.gate_b = []
        self.res_w = []    # (C, C, 1) -> squeeze to (C, C)
        self.res_b = []
        self.skip_w = []   # (skip_C, C, 1) -> squeeze to (skip_C, C)
        self.skip_b = []

        for i in range(self.n_blocks):
            pf = f"blocks.{i}"
            self.filter_w.append(state_dict[f"{pf}.filter_conv.conv.weight"].numpy())  # (C,C,2)
            self.filter_b.append(state_dict[f"{pf}.filter_conv.conv.bias"].numpy())
            self.gate_w.append(state_dict[f"{pf}.gate_conv.conv.weight"].numpy())
            self.gate_b.append(state_dict[f"{pf}.gate_conv.conv.bias"].numpy())
            self.res_w.append(state_dict[f"{pf}.res_conv.weight"].numpy().squeeze(-1))  # (C,C)
            self.res_b.append(state_dict[f"{pf}.res_conv.bias"].numpy())
            self.skip_w.append(state_dict[f"{pf}.skip_conv.weight"].numpy().squeeze(-1))  # (skip,C)
            self.skip_b.append(state_dict[f"{pf}.skip_conv.bias"].numpy())

        # Output head: ReLU -> Conv1d(skip,C,1) -> ReLU -> Conv1d(C,2,1)
        self.head1_w = state_dict["output_head.1.weight"].numpy().squeeze(-1)  # (C, skip)
        self.head1_b = state_dict["output_head.1.bias"].numpy()
        self.head3_w = state_dict["output_head.3.weight"].numpy().squeeze(-1)  # (2, C)
        self.head3_b = state_dict["output_head.3.bias"].numpy()

    def init_buffers(self):
        """Initialize ring buffers: list of (C, dilation) arrays."""
        return [np.zeros((self.channels, d), dtype=np.float32) for d in self.dilations]

    def step(self, x, buffers):
        """Single step inference.

        Args:
            x: (32,) raw features
            buffers: list of ring buffer arrays

        Returns:
            prediction: (2,)
            new_buffers: updated ring buffers
        """
        # Input projection: (32,) -> (C,)
        current = self.inp_w @ x + self.inp_b

        skip_sum = np.zeros(self.skip_channels, dtype=np.float32)
        new_buffers = []

        for i in range(self.n_blocks):
            buf = buffers[i]  # (C, dilation)
            delayed = buf[:, 0]  # oldest value in ring buffer

            # Causal conv kernel_size=2: W[:,:,0] @ delayed + W[:,:,1] @ current + bias
            fw = self.filter_w[i]  # (C, C, 2)
            f_val = fw[:, :, 0] @ delayed + fw[:, :, 1] @ current + self.filter_b[i]
            gw = self.gate_w[i]
            g_val = gw[:, :, 0] @ delayed + gw[:, :, 1] @ current + self.gate_b[i]

            z = np.tanh(f_val) * (1.0 / (1.0 + np.exp(-g_val)))  # tanh * sigmoid

            skip_sum += self.skip_w[i] @ z + self.skip_b[i]
            res = self.res_w[i] @ z + self.res_b[i]
            current = current + self.res_scale * res

            # Update ring buffer: shift left, append current
            new_buf = np.empty_like(buf)
            if buf.shape[1] > 1:
                new_buf[:, :-1] = buf[:, 1:]
            new_buf[:, -1] = current
            new_buffers.append(new_buf)

        # Output head: ReLU -> Linear -> ReLU -> Linear
        h = np.maximum(skip_sum, 0)
        h = np.maximum(self.head1_w @ h + self.head1_b, 0)
        out = self.head3_w @ h + self.head3_b

        return out, new_buffers


# Model configs: (filename, weight)
MODEL_CONFIGS = {model_configs}


class PredictionModel:
    def __init__(self, model_path=""):
        import torch as _torch
        base_dir = os.path.dirname(os.path.abspath(__file__))

        self.models = []
        self.buffers = []
        self.weights = []

        for filename, w in MODEL_CONFIGS:
            ckpt = _torch.load(
                os.path.join(base_dir, filename),
                map_location="cpu",
                weights_only=False,
            )
            sd = ckpt["model_state_dict"]
            model = WaveNetNumpy(sd)
            self.models.append(model)
            self.buffers.append(model.init_buffers())
            self.weights.append(w)

        total_w = sum(self.weights)
        self.weights = np.array([w / total_w for w in self.weights], dtype=np.float32)
        self.prev_seq_ix = None

    def predict(self, data_point) -> np.ndarray:
        seq_ix = data_point.seq_ix
        if seq_ix != self.prev_seq_ix:
            self.buffers = [m.init_buffers() for m in self.models]
            self.prev_seq_ix = seq_ix

        features = data_point.state.astype(np.float32)[:32]

        pred_sum = np.zeros(2, dtype=np.float32)
        for i, model in enumerate(self.models):
            pred, self.buffers[i] = model.step(features, self.buffers[i])
            pred_sum += self.weights[i] * pred

        if not data_point.need_prediction:
            return None
        return pred_sum.clip(-6, 6)
'''


def main():
    parser = argparse.ArgumentParser(description="Build WaveNet ensemble submission")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Paths to wavenet checkpoints (.pt)")
    parser.add_argument("--weights", nargs="*", type=float, default=None,
                        help="Model weights (default: uniform)")
    parser.add_argument("--output", default="submissions/wavenet_ensemble.zip",
                        help="Output zip path")
    args = parser.parse_args()

    n_models = len(args.checkpoints)
    weights = args.weights if args.weights else [1.0] * n_models
    if len(weights) != n_models:
        raise ValueError(f"Got {len(weights)} weights for {n_models} models")

    print(f"Building WaveNet ensemble: {n_models} models (pure numpy inference)")

    # Load and validate checkpoints
    model_configs = []
    scores = []
    for i, ckpt_path in enumerate(args.checkpoints):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        score = ckpt.get("best_score", 0)
        epoch = ckpt.get("best_epoch", "?")
        filename = f"model_{i}.pt"
        model_configs.append((filename, weights[i]))
        scores.append(score)
        print(f"  [{i}] {os.path.basename(ckpt_path)}: val={score:.4f}, epoch={epoch}, w={weights[i]:.3f}")

    # Estimate runtime: WaveNet step-by-step is ~2-3x slower than GRU due to 10 blocks
    est_per_model = 400  # conservative estimate (seconds)
    est_total = n_models * est_per_model
    margin = 1.0 - est_total / 4200
    print(f"\nEstimated runtime: {n_models} x {est_per_model}s = {est_total}s ({margin*100:.0f}% margin)")
    if margin < 0.20:
        print("WARNING: Tight margin! Consider fewer models.")

    # Build zip
    with tempfile.TemporaryDirectory() as tmpdir:
        solution_code = SOLUTION_TEMPLATE.format(
            n_models=n_models,
            model_configs=repr(model_configs),
        )
        with open(os.path.join(tmpdir, "solution.py"), "w") as f:
            f.write(solution_code)

        for i, ckpt_path in enumerate(args.checkpoints):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            slim = {"model_state_dict": ckpt["model_state_dict"]}
            torch.save(slim, os.path.join(tmpdir, f"model_{i}.pt"))

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        base = args.output.replace(".zip", "")
        shutil.make_archive(base, "zip", tmpdir)

    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\nCreated: {args.output} ({size_mb:.1f}MB)")
    if size_mb > 20:
        print("WARNING: Exceeds 20MB submission limit!")
    else:
        print(f"Size OK ({size_mb:.1f}MB < 20MB limit)")

    mean_val = sum(scores) / len(scores)
    print(f"\nMean val of ensemble members: {mean_val:.4f}")
    print(f"Best single val: {max(scores):.4f}")


if __name__ == "__main__":
    main()
