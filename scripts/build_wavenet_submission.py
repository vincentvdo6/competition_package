"""Build submission zip for WaveNet ensemble.

Supports both PyTorch (pure numpy) and ONNX inference modes:
  --onnx: Export to ONNX for faster inference (recommended for 10+ models)

Usage:
  Single canary:
    python scripts/build_wavenet_submission.py --checkpoints logs/wavenet_v1_seed42.pt --output submissions/wavenet_canary.zip

  ONNX ensemble:
    python scripts/build_wavenet_submission.py --checkpoints logs/wavenet_v1_seed*.pt --onnx --output submissions/wavenet_ens10_onnx.zip
"""

import argparse
import os
import shutil
import tempfile
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# ONNX export: pack all ring buffers into single flat tensor
# ---------------------------------------------------------------------------
DILATIONS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
BUF_TOTAL = sum(DILATIONS)  # 1023


def _compute_offsets(dilations):
    """Cumulative offsets for packing ring buffers into single tensor."""
    offsets = []
    pos = 0
    for d in dilations:
        offsets.append(pos)
        pos += d
    return offsets


class WaveNetStep(nn.Module):
    """WaveNet single-step wrapper for ONNX export.

    Packs all ring buffers into a single (1, C, 1023) tensor.
    Uses manual matmul for dilated convs (Conv1d doesn't work on width-2
    input when dilation > 1). 1x1 convs are stored as Linear layers.
    """

    def __init__(self, state_dict, channels=64, skip_channels=96,
                 dilations=None, res_scale=0.3):
        super().__init__()
        self.channels = channels
        self.skip_channels = skip_channels
        self.dilations = dilations or DILATIONS
        self.res_scale = res_scale
        self.n_blocks = len(self.dilations)
        self.offsets = _compute_offsets(self.dilations)

        # Store offsets as buffer for ONNX tracing
        self.register_buffer('_offsets', torch.tensor(self.offsets, dtype=torch.long))
        self.register_buffer('_dilations', torch.tensor(self.dilations, dtype=torch.long))

        # Input projection weight (Conv1d 32->C, kernel=1) stored as (C, 32)
        self.inp_w = nn.Parameter(state_dict["input_proj.weight"].squeeze(-1).clone())
        self.inp_b = nn.Parameter(state_dict["input_proj.bias"].clone())

        # Per-block: kernel_size=2 conv weights stored as two (C, C) matrices
        # filter_w0[i], filter_w1[i]: W[:,:,0] and W[:,:,1] for filter conv
        self.filter_w0 = nn.ParameterList()
        self.filter_w1 = nn.ParameterList()
        self.filter_b = nn.ParameterList()
        self.gate_w0 = nn.ParameterList()
        self.gate_w1 = nn.ParameterList()
        self.gate_b = nn.ParameterList()
        self.res_w = nn.ParameterList()
        self.res_b = nn.ParameterList()
        self.skip_w = nn.ParameterList()
        self.skip_b = nn.ParameterList()

        for i in range(self.n_blocks):
            pf = f"blocks.{i}"
            fw = state_dict[f"{pf}.filter_conv.conv.weight"]  # (C, C, 2)
            self.filter_w0.append(nn.Parameter(fw[:, :, 0].clone()))
            self.filter_w1.append(nn.Parameter(fw[:, :, 1].clone()))
            self.filter_b.append(nn.Parameter(state_dict[f"{pf}.filter_conv.conv.bias"].clone()))

            gw = state_dict[f"{pf}.gate_conv.conv.weight"]
            self.gate_w0.append(nn.Parameter(gw[:, :, 0].clone()))
            self.gate_w1.append(nn.Parameter(gw[:, :, 1].clone()))
            self.gate_b.append(nn.Parameter(state_dict[f"{pf}.gate_conv.conv.bias"].clone()))

            self.res_w.append(nn.Parameter(state_dict[f"{pf}.res_conv.weight"].squeeze(-1).clone()))
            self.res_b.append(nn.Parameter(state_dict[f"{pf}.res_conv.bias"].clone()))
            self.skip_w.append(nn.Parameter(state_dict[f"{pf}.skip_conv.weight"].squeeze(-1).clone()))
            self.skip_b.append(nn.Parameter(state_dict[f"{pf}.skip_conv.bias"].clone()))

        # Output head weights
        self.head1_w = nn.Parameter(state_dict["output_head.1.weight"].squeeze(-1).clone())
        self.head1_b = nn.Parameter(state_dict["output_head.1.bias"].clone())
        self.head2_w = nn.Parameter(state_dict["output_head.3.weight"].squeeze(-1).clone())
        self.head2_b = nn.Parameter(state_dict["output_head.3.bias"].clone())

    def forward(self, x: torch.Tensor, buf_state: torch.Tensor):
        """Single step with packed ring buffer.

        Args:
            x: (1, 32) raw features
            buf_state: (1, C, 1023) packed ring buffers

        Returns:
            prediction: (1, 2)
            new_buf_state: (1, C, 1023)
        """
        # Input projection: (1, 32) @ (32, C) -> (1, C)
        current = x @ self.inp_w.T + self.inp_b

        skip_sum = torch.zeros(1, self.skip_channels, device=x.device)

        # Build new buffer pieces (avoid in-place slice assignment for ONNX)
        buf_pieces = []

        for i in range(self.n_blocks):
            d = int(self._dilations[i])
            off = int(self._offsets[i])

            # Get delayed value from ring buffer (oldest = index 0)
            delayed = buf_state[:, :, off]  # (1, C)

            # Manual matmul: output = W0 @ delayed + W1 @ current + bias
            f = delayed @ self.filter_w0[i].T + current @ self.filter_w1[i].T + self.filter_b[i]
            g = delayed @ self.gate_w0[i].T + current @ self.gate_w1[i].T + self.gate_b[i]
            z = torch.tanh(f) * torch.sigmoid(g)

            # 1x1 convs as matmul
            skip_sum = skip_sum + (z @ self.skip_w[i].T + self.skip_b[i])
            res = z @ self.res_w[i].T + self.res_b[i]

            # Build new buffer slice: shift left + append current (INPUT before residual)
            if d > 1:
                shifted = buf_state[:, :, off + 1:off + d]  # (1, C, d-1)
                buf_pieces.append(shifted)
            buf_pieces.append(current.unsqueeze(-1))  # (1, C, 1)

            # Apply residual to get input for next block
            current = current + self.res_scale * res

        # Concat all buffer pieces to form new_buf_state
        new_buf_state = torch.cat(buf_pieces, dim=-1)  # (1, C, 1023)

        # Output head: ReLU -> Linear -> ReLU -> Linear
        h = torch.relu(skip_sum)
        h = torch.relu(h @ self.head1_w.T + self.head1_b)
        out = h @ self.head2_w.T + self.head2_b  # (1, 2)

        return out, new_buf_state


def _export_wavenet_to_onnx(ckpt_path, onnx_path):
    """Export a WaveNet checkpoint to ONNX."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]

    # Read config from checkpoint if available
    cfg = ckpt.get("config", {}).get("model", {})
    channels = cfg.get("channels", 64)
    skip_channels = cfg.get("skip_channels", 96)
    dilations = cfg.get("dilations", DILATIONS)
    res_scale = cfg.get("res_scale", 0.3)

    wrapper = WaveNetStep(sd, channels, skip_channels, dilations, res_scale)
    wrapper.eval()

    buf_total = sum(dilations)
    dummy_x = torch.randn(1, 32)
    dummy_buf = torch.zeros(1, channels, buf_total)

    torch.onnx.export(
        wrapper,
        (dummy_x, dummy_buf),
        onnx_path,
        input_names=["input", "buf_state"],
        output_names=["prediction", "new_buf_state"],
        dynamic_axes=None,
        opset_version=17,
        do_constant_folding=True,
    )


# ---------------------------------------------------------------------------
# Solution template: pure numpy (PyTorch mode)
# ---------------------------------------------------------------------------
SOLUTION_TEMPLATE_NUMPY = '''"""WaveNet-lite ensemble: {n_models} models, ring buffer inference."""

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

        self.inp_w = state_dict["input_proj.weight"].numpy().squeeze(-1)
        self.inp_b = state_dict["input_proj.bias"].numpy()

        self.filter_w = []
        self.filter_b = []
        self.gate_w = []
        self.gate_b = []
        self.res_w = []
        self.res_b = []
        self.skip_w = []
        self.skip_b = []

        for i in range(self.n_blocks):
            pf = f"blocks.{{i}}"
            self.filter_w.append(state_dict[f"{{pf}}.filter_conv.conv.weight"].numpy())
            self.filter_b.append(state_dict[f"{{pf}}.filter_conv.conv.bias"].numpy())
            self.gate_w.append(state_dict[f"{{pf}}.gate_conv.conv.weight"].numpy())
            self.gate_b.append(state_dict[f"{{pf}}.gate_conv.conv.bias"].numpy())
            self.res_w.append(state_dict[f"{{pf}}.res_conv.weight"].numpy().squeeze(-1))
            self.res_b.append(state_dict[f"{{pf}}.res_conv.bias"].numpy())
            self.skip_w.append(state_dict[f"{{pf}}.skip_conv.weight"].numpy().squeeze(-1))
            self.skip_b.append(state_dict[f"{{pf}}.skip_conv.bias"].numpy())

        self.head1_w = state_dict["output_head.1.weight"].numpy().squeeze(-1)
        self.head1_b = state_dict["output_head.1.bias"].numpy()
        self.head3_w = state_dict["output_head.3.weight"].numpy().squeeze(-1)
        self.head3_b = state_dict["output_head.3.bias"].numpy()

    def init_buffers(self):
        return [np.zeros((self.channels, d), dtype=np.float32) for d in self.dilations]

    def step(self, x, buffers):
        current = self.inp_w @ x + self.inp_b
        skip_sum = np.zeros(self.skip_channels, dtype=np.float32)
        new_buffers = []

        for i in range(self.n_blocks):
            buf = buffers[i]
            delayed = buf[:, 0]

            fw = self.filter_w[i]
            f_val = fw[:, :, 0] @ delayed + fw[:, :, 1] @ current + self.filter_b[i]
            gw = self.gate_w[i]
            g_val = gw[:, :, 0] @ delayed + gw[:, :, 1] @ current + self.gate_b[i]

            z = np.tanh(f_val) * (1.0 / (1.0 + np.exp(-g_val)))

            skip_sum += self.skip_w[i] @ z + self.skip_b[i]
            res = self.res_w[i] @ z + self.res_b[i]

            # Store INPUT to this block in buffer (before residual)
            new_buf = np.empty_like(buf)
            if buf.shape[1] > 1:
                new_buf[:, :-1] = buf[:, 1:]
            new_buf[:, -1] = current
            new_buffers.append(new_buf)

            # Apply residual for next block
            current = current + self.res_scale * res

        h = np.maximum(skip_sum, 0)
        h = np.maximum(self.head1_w @ h + self.head1_b, 0)
        out = self.head3_w @ h + self.head3_b
        return out, new_buffers


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
            cfg = ckpt.get("config", {{}}).get("model", {{}})
            model = WaveNetNumpy(sd, res_scale=cfg.get("res_scale", 0.3))
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


# ---------------------------------------------------------------------------
# Solution template: ONNX
# ---------------------------------------------------------------------------
SOLUTION_TEMPLATE_ONNX = '''"""WaveNet-lite ONNX ensemble: {n_models} models, ring buffer inference."""

import os
import numpy as np
import onnxruntime as ort


BUF_TOTAL = {buf_total}  # sum of dilations
CHANNELS = {channels}

MODEL_CONFIGS = {model_configs}


class OnnxWaveNet:
    """ONNX Runtime WaveNet for step-by-step inference."""

    def __init__(self, onnx_path, channels, buf_total):
        self.channels = channels
        self.buf_total = buf_total
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            onnx_path, opts, providers=["CPUExecutionProvider"]
        )

    def init_buf(self):
        return np.zeros((1, self.channels, self.buf_total), dtype=np.float32)

    def run_step(self, x_np, buf_state):
        pred, new_buf = self.sess.run(
            ["prediction", "new_buf_state"],
            {{"input": x_np, "buf_state": buf_state}},
        )
        return pred[0], new_buf


class PredictionModel:
    def __init__(self, model_path=""):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        self.models = []
        self.bufs = []
        self.weights = []

        for filename, w in MODEL_CONFIGS:
            model = OnnxWaveNet(
                os.path.join(base_dir, filename), CHANNELS, BUF_TOTAL
            )
            self.models.append(model)
            self.bufs.append(model.init_buf())
            self.weights.append(w)

        total_w = sum(self.weights)
        self.weights = np.array([w / total_w for w in self.weights], dtype=np.float32)
        self.prev_seq_ix = None

    def predict(self, data_point) -> np.ndarray:
        seq_ix = data_point.seq_ix
        if seq_ix != self.prev_seq_ix:
            self.bufs = [m.init_buf() for m in self.models]
            self.prev_seq_ix = seq_ix

        features = data_point.state.astype(np.float32)[:32].reshape(1, -1)

        pred_sum = np.zeros(2, dtype=np.float32)
        for i, model in enumerate(self.models):
            pred, self.bufs[i] = model.run_step(features, self.bufs[i])
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
    parser.add_argument("--onnx", action="store_true",
                        help="Export models to ONNX for faster inference")
    args = parser.parse_args()

    n_models = len(args.checkpoints)
    weights = args.weights if args.weights else [1.0] * n_models
    if len(weights) != n_models:
        raise ValueError(f"Got {len(weights)} weights for {n_models} models")

    mode = "ONNX" if args.onnx else "numpy"
    print(f"Building WaveNet ensemble: {n_models} models ({mode} inference)")

    # Load and validate checkpoints
    ext = ".onnx" if args.onnx else ".pt"
    model_configs = []
    scores = []
    for i, ckpt_path in enumerate(args.checkpoints):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        score = ckpt.get("best_score", 0)
        epoch = ckpt.get("best_epoch", "?")
        filename = f"model_{i}{ext}"
        model_configs.append((filename, weights[i]))
        scores.append(score)
        print(f"  [{i}] {os.path.basename(ckpt_path)}: val={score:.4f}, epoch={epoch}, w={weights[i]:.3f}")

    # Estimate runtime
    if args.onnx:
        est_per_model = 100  # ONNX should be ~3-4x faster than numpy
    else:
        est_per_model = 400
    est_total = n_models * est_per_model
    margin = 1.0 - est_total / 4200
    print(f"\nEstimated runtime: {n_models} x {est_per_model}s = {est_total}s ({margin*100:.0f}% margin)")
    if margin < 0.20:
        print("WARNING: Tight margin! Consider fewer models or --onnx.")

    # Read channels/dilations from first checkpoint config
    first_ckpt = torch.load(args.checkpoints[0], map_location="cpu", weights_only=False)
    cfg = first_ckpt.get("config", {}).get("model", {})
    channels = cfg.get("channels", 64)
    dilations = cfg.get("dilations", DILATIONS)
    buf_total = sum(dilations)

    # Build zip
    with tempfile.TemporaryDirectory() as tmpdir:
        if args.onnx:
            solution_code = SOLUTION_TEMPLATE_ONNX.format(
                n_models=n_models,
                model_configs=repr(model_configs),
                buf_total=buf_total,
                channels=channels,
            )
        else:
            solution_code = SOLUTION_TEMPLATE_NUMPY.format(
                n_models=n_models,
                model_configs=repr(model_configs),
            )
        with open(os.path.join(tmpdir, "solution.py"), "w") as f:
            f.write(solution_code)

        for i, ckpt_path in enumerate(args.checkpoints):
            if args.onnx:
                onnx_path = os.path.join(tmpdir, f"model_{i}.onnx")
                print(f"  Exporting model_{i}.onnx ...", end=" ", flush=True)
                _export_wavenet_to_onnx(ckpt_path, onnx_path)
                size_kb = os.path.getsize(onnx_path) / 1024
                print(f"({size_kb:.0f}KB)")
            else:
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
