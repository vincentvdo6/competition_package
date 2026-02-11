"""Quick verification: TCNFast (numpy) vs TCNModel (PyTorch) correctness + speed."""
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

# Load config
with open(ROOT / "configs" / "tcn_base_v1.yaml") as f:
    config = yaml.safe_load(f)

model_cfg = config["model"]

# ---- Build PyTorch model ----
from src.models.tcn_model import TCNModel

pt_model = TCNModel(config)

# Load a checkpoint if available, otherwise use random weights
ckpt_path = ROOT / "logs" / "_staging" / "tcn_base_v1_seed42.pt"
if ckpt_path.exists():
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pt_model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: {ckpt_path.name}")
else:
    print("Using random weights (no checkpoint found)")

pt_model.eval()

# ---- Build TCNFast (numpy) ----
# Import the solution template's TCNFast by exec'ing the relevant portion
# Instead, let's replicate the TCNFast class here for testing

class TCNFast:
    """Numpy-only TCN inference — eliminates PyTorch dispatch overhead."""

    def __init__(self, pt):
        ch = pt.ch if hasattr(pt, 'ch') else pt.hidden_channels
        self.ch = ch
        self.inp_w = pt.input_proj.weight.data.numpy().copy()
        self.inp_b = pt.input_proj.bias.data.numpy().copy()
        self.n_blocks = len(pt.blocks)
        self.dw_w = []; self.dw_b = []; self.pw_w = []; self.pw_b = []
        self.buf_lens = []; self.dils = []; self.ks_list = []
        for b in pt.blocks:
            ks = b.kernel_size
            self.dw_w.append(b.dw_conv.weight.data.numpy().reshape(ch, ks).copy())
            self.dw_b.append(b.dw_conv.bias.data.numpy().copy())
            self.pw_w.append(b.pw_conv.weight.data.numpy().reshape(ch, ch).copy())
            self.pw_b.append(b.pw_conv.bias.data.numpy().copy())
            self.buf_lens.append(b.buf_len)
            self.dils.append(b.dilation)
            self.ks_list.append(ks)
        self.norm_w = pt.output_norm.weight.data.numpy().copy()
        self.norm_b = pt.output_norm.bias.data.numpy().copy()
        self.head_w = pt.output_head.weight.data.numpy().copy()
        self.head_b = pt.output_head.bias.data.numpy().copy()

    def forward_step(self, x_tensor, hidden=None, need_pred=True):
        x = x_tensor.numpy().ravel() if hasattr(x_tensor, 'numpy') else np.asarray(x_tensor, dtype=np.float32).ravel()
        h = self.inp_w @ x + self.inp_b
        if hidden is None:
            hidden = [(np.zeros((self.ch, bl), dtype=np.float32), 0) for bl in self.buf_lens]
        new_hidden = []
        for i in range(self.n_blocks):
            buf, ptr = hidden[i]
            buf[:, ptr] = h
            ks = self.ks_list[i]; dil = self.dils[i]; bl = self.buf_lens[i]
            taps = np.empty((self.ch, ks), dtype=np.float32)
            for j in range(ks):
                taps[:, j] = buf[:, (ptr - (ks - 1 - j) * dil) % bl]
            out = (taps * self.dw_w[i]).sum(1) + self.dw_b[i]
            out = out / (1.0 + np.exp(-np.clip(out, -15, 15)))
            out = self.pw_w[i] @ out + self.pw_b[i]
            h = out + h
            new_hidden.append((buf, (ptr + 1) % bl))
        if not need_pred:
            return None, new_hidden
        m = h.mean(); v = ((h - m) ** 2).mean()
        h = (h - m) / np.sqrt(v + 1e-5) * self.norm_w + self.norm_b
        pred = self.head_w @ h + self.head_b
        return pred, new_hidden


fast_model = TCNFast(pt_model)

# ---- Correctness test: compare predictions over 200 steps ----
print("\n=== Correctness Test (200 steps) ===")
np.random.seed(42)
seq_features = np.random.randn(200, 42).astype(np.float32)

pt_hidden = None
fast_hidden = None
max_diff = 0.0

with torch.no_grad():
    for step in range(200):
        x = torch.from_numpy(seq_features[step])
        need_pred = step >= 99

        pt_pred, pt_hidden = pt_model.forward_step(x.unsqueeze(0), pt_hidden)
        fast_pred, fast_hidden = fast_model.forward_step(x, fast_hidden, need_pred=need_pred)

        if need_pred:
            pt_np = pt_pred.squeeze(0).numpy()
            fast_np = fast_pred
            diff = np.abs(pt_np - fast_np).max()
            max_diff = max(max_diff, diff)
            if step < 102 or step == 199:
                print(f"  Step {step:3d}: PT={pt_np}, NP={fast_np}, diff={diff:.2e}")

print(f"\n  Max absolute difference: {max_diff:.2e}")
if max_diff < 1e-5:
    print("  PASS: Predictions match within float32 precision")
else:
    print("  WARNING: Predictions diverge — check implementation!")

# ---- Speed benchmark: 10000 steps ----
print("\n=== Speed Benchmark (10000 steps) ===")
N = 10000
features_batch = np.random.randn(N, 42).astype(np.float32)

# PyTorch forward_step
pt_hidden = None
t0 = time.perf_counter()
with torch.no_grad():
    for i in range(N):
        x = torch.from_numpy(features_batch[i]).unsqueeze(0)
        pred, pt_hidden = pt_model.forward_step(x, pt_hidden)
pt_time = time.perf_counter() - t0
pt_us = pt_time / N * 1e6

# Numpy forward_step
fast_hidden = None
t0 = time.perf_counter()
for i in range(N):
    x = torch.from_numpy(features_batch[i])
    pred, fast_hidden = fast_model.forward_step(x, fast_hidden)
np_time = time.perf_counter() - t0
np_us = np_time / N * 1e6

print(f"  PyTorch: {pt_us:.1f} us/step ({pt_time:.2f}s total)")
print(f"  Numpy:   {np_us:.1f} us/step ({np_time:.2f}s total)")
print(f"  Speedup: {pt_us/np_us:.1f}x")

# Extrapolate to full dataset
full_steps = 1_444_000
pt_est = pt_us * full_steps / 1e6
np_est = np_us * full_steps / 1e6
print(f"\n  Estimated per-model on 1.44M steps:")
print(f"    PyTorch: {pt_est:.0f}s")
print(f"    Numpy:   {np_est:.0f}s")
print(f"  Estimated 2-TCN total:")
print(f"    PyTorch: {pt_est*2:.0f}s")
print(f"    Numpy:   {np_est*2:.0f}s")
