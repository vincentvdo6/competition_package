"""Build submission zips locally from a parity swap discovery report.

Usage:
    python scripts/build_from_report.py \
        --report logs/parity_swap_discovery_feb22-b1.json \
        --checkpoints-zip "C:/Users/Vincent/Downloads/vanilla_seeds (1).zip" \
        --output-dir submissions/ready
"""

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import torch
import torch.nn as nn


SOLUTION_TEMPLATE = '''"""Per-target vanilla GRU ONNX ensemble with shared model pool."""

import os
import numpy as np
import onnxruntime as ort

MODEL_CONFIGS = {model_configs}


class OnnxVanillaGRU:
    def __init__(self, onnx_path, hidden_size, num_layers):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])

    def init_hidden(self):
        return np.zeros((self.num_layers, 1, self.hidden_size), dtype=np.float32)

    def run_step(self, x_np, hidden):
        pred, new_hidden = self.sess.run(
            ["prediction", "hidden_out"],
            {{"input": x_np, "hidden_in": hidden}},
        )
        return pred[0], new_hidden


class PredictionModel:
    def __init__(self, model_path=""):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        self.models = []
        self.hiddens = []
        self.weights_t0 = []
        self.weights_t1 = []

        for filename, h, nl, w0, w1 in MODEL_CONFIGS:
            model = OnnxVanillaGRU(os.path.join(base_dir, filename), h, nl)
            self.models.append(model)
            self.hiddens.append(model.init_hidden())
            self.weights_t0.append(w0)
            self.weights_t1.append(w1)

        sum_t0 = float(sum(self.weights_t0))
        sum_t1 = float(sum(self.weights_t1))
        self.weights_t0 = np.array([w / sum_t0 for w in self.weights_t0], dtype=np.float32)
        self.weights_t1 = np.array([w / sum_t1 for w in self.weights_t1], dtype=np.float32)

        self.prev_seq_ix = None

    def predict(self, data_point) -> np.ndarray:
        seq_ix = data_point.seq_ix
        if seq_ix != self.prev_seq_ix:
            self.hiddens = [m.init_hidden() for m in self.models]
            self.prev_seq_ix = seq_ix

        x = data_point.state.astype(np.float32)[:32].reshape(1, -1)

        t0_sum = 0.0
        t1_sum = 0.0
        for i, model in enumerate(self.models):
            pred, self.hiddens[i] = model.run_step(x, self.hiddens[i])
            t0_sum += float(self.weights_t0[i] * pred[0])
            t1_sum += float(self.weights_t1[i] * pred[1])

        if not data_point.need_prediction:
            return None

        return np.array([t0_sum, t1_sum], dtype=np.float32).clip(-6, 6)
'''


class VanillaGRUStep(nn.Module):
    def __init__(self, gru, fc):
        super().__init__()
        self.gru = gru
        self.fc = fc

    def forward(self, x, hidden):
        x = x.unsqueeze(1)
        out, new_hidden = self.gru(x, hidden)
        return self.fc(out.squeeze(1)), new_hidden


def export_onnx(ckpt_path: Path, hidden_size: int, num_layers: int, onnx_path: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]

    gru = nn.GRU(input_size=32, hidden_size=hidden_size, num_layers=num_layers,
                 batch_first=True, dropout=0.0, bidirectional=False)
    fc = nn.Linear(hidden_size, 2)

    gru_sd = {k.replace("gru.", ""): v for k, v in state_dict.items() if k.startswith("gru.")}
    fc_sd = {k.replace("output_proj.", ""): v for k, v in state_dict.items() if k.startswith("output_proj.")}
    gru.load_state_dict(gru_sd)
    fc.load_state_dict(fc_sd)

    wrapper = VanillaGRUStep(gru, fc)
    wrapper.eval()

    dummy_x = torch.randn(1, 32)
    dummy_h = torch.zeros(num_layers, 1, hidden_size)
    torch.onnx.export(
        wrapper, (dummy_x, dummy_h), str(onnx_path),
        input_names=["input", "hidden_in"],
        output_names=["prediction", "hidden_out"],
        dynamic_axes=None, opset_version=17, do_constant_folding=True,
    )


def build_zip(output_zip: Path, seed_order, w0, w1, ckpt_dir: Path,
              hidden_size: int, num_layers: int) -> None:
    model_configs = [
        (f"model_{i}.onnx", hidden_size, num_layers, float(w0[i]), float(w1[i]))
        for i in range(len(seed_order))
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        (tmp / "solution.py").write_text(
            SOLUTION_TEMPLATE.format(model_configs=repr(model_configs)),
            encoding="utf-8"
        )
        for i, seed in enumerate(seed_order):
            ckpt = ckpt_dir / f"gru_parity_v1_seed{seed}.pt"
            if not ckpt.exists():
                raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
            print(f"  Exporting model_{i}.onnx (seed {seed})...", flush=True)
            export_onnx(ckpt, hidden_size, num_layers, tmp / f"model_{i}.onnx")

        output_zip.parent.mkdir(parents=True, exist_ok=True)
        shutil.make_archive(str(output_zip.with_suffix("")), "zip", tmpdir)

    size_mb = output_zip.stat().st_size / 1e6
    print(f"  -> {output_zip.name} ({size_mb:.1f}MB)")
    if size_mb > 20:
        print("  WARNING: exceeds 20MB limit!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, help="Path to parity_swap_discovery_*.json")
    parser.add_argument("--checkpoints-zip", required=True, help="Path to vanilla_seeds.zip")
    parser.add_argument("--output-dir", default="submissions/ready")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text())
    session_tag = Path(args.report).stem.replace("parity_swap_discovery_", "")
    out_dir = Path(args.output_dir)

    anchor = report["anchor"]
    w0 = anchor["weights_t0"]
    w1 = anchor["weights_t1"]
    anchor_score = anchor["score"]["avg"]
    print(f"Anchor score: {anchor_score:.4f}")

    slot_a = report["selected"]["slot_a_t1"]
    slot_b = report["selected"]["slot_b_t0"]
    print(f"Slot A (t1): replace s{slot_a['replace_seed']} -> s{slot_a['candidate_seed']}  "
          f"delta={slot_a['delta_avg']:+.5f}  gate={'PASS' if slot_a['keep_gate'] else 'FAIL'}")
    print(f"Slot B (t0): replace s{slot_b['replace_seed']} -> s{slot_b['candidate_seed']}  "
          f"delta={slot_b['delta_avg']:+.5f}  gate={'PASS' if slot_b['keep_gate'] else 'FAIL'}")

    # Extract checkpoints to temp dir
    with tempfile.TemporaryDirectory() as ckpt_tmp:
        ckpt_dir = Path(ckpt_tmp)
        print(f"\nExtracting checkpoints from {args.checkpoints_zip}...")
        with zipfile.ZipFile(args.checkpoints_zip) as zf:
            zf.extractall(ckpt_dir)
        extracted = list(ckpt_dir.glob("*.pt"))
        print(f"Extracted {len(extracted)} checkpoints")

        print(f"\nBuilding Slot A: {session_tag}-t1swap-a-onnx.zip")
        build_zip(
            out_dir / f"{session_tag}-t1swap-a-onnx.zip",
            slot_a["seed_order"], w0, w1, ckpt_dir,
            args.hidden_size, args.num_layers,
        )

        print(f"\nBuilding Slot B: {session_tag}-t0swap-b-onnx.zip")
        build_zip(
            out_dir / f"{session_tag}-t0swap-b-onnx.zip",
            slot_b["seed_order"], w0, w1, ckpt_dir,
            args.hidden_size, args.num_layers,
        )

    print("\nDone. Run check_submission_zip.py to verify before submitting.")


if __name__ == "__main__":
    main()
