"""Build per-target asymmetric weight ensemble from pre-exported ONNX models.

Reuses ONNX files from an existing submission zip, generates a new solution.py
with separate t0/t1 weights for each model.

Usage:
    python scripts/build_pertarget_ensemble.py \
        --source submissions/archive/scored/feb25_batch/mix_v10_kf5_fold2w7_onnx.zip \
        --weights-t0 1 1 1 1 1 1 1 1 1 1 1 1 7 1 1 \
        --weights-t1 1 1 1 1 1 1 1 1 1 1 1 1 20 1 1 \
        --output submissions/ready/pertarget_t0w7_t1w20_onnx.zip
"""
import argparse
import os
import shutil
import tempfile
import zipfile

SOLUTION_TEMPLATE = '''"""Vanilla GRU ONNX ensemble: {n_models} models, per-target weights."""

import os
import numpy as np
import onnxruntime as ort


# Model configs: (filename, hidden_size, num_layers, weight_t0, weight_t1)
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
        return pred[0], new_hidden


class PredictionModel:
    def __init__(self, model_path=""):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        self.models = []
        self.hiddens = []
        self.weights_t0 = []
        self.weights_t1 = []

        for filename, h, nl, w0, w1 in MODEL_CONFIGS:
            model = OnnxVanillaGRU(
                os.path.join(base_dir, filename), h, nl
            )
            self.models.append(model)
            self.hiddens.append(model.init_hidden())
            self.weights_t0.append(w0)
            self.weights_t1.append(w1)

        # Normalize weights per target
        total_t0 = sum(self.weights_t0)
        total_t1 = sum(self.weights_t1)
        self.weights_t0 = np.array([w / total_t0 for w in self.weights_t0], dtype=np.float32)
        self.weights_t1 = np.array([w / total_t1 for w in self.weights_t1], dtype=np.float32)

        self.prev_seq_ix = None

    def predict(self, data_point) -> np.ndarray:
        seq_ix = data_point.seq_ix
        if seq_ix != self.prev_seq_ix:
            self.hiddens = [m.init_hidden() for m in self.models]
            self.prev_seq_ix = seq_ix

        features = data_point.state.astype(np.float32)[:32].reshape(1, -1)

        pred_t0 = np.float32(0.0)
        pred_t1 = np.float32(0.0)
        for i, model in enumerate(self.models):
            pred, self.hiddens[i] = model.run_step(features, self.hiddens[i])
            pred_t0 += self.weights_t0[i] * pred[0]
            pred_t1 += self.weights_t1[i] * pred[1]

        if not data_point.need_prediction:
            return None
        return np.array([pred_t0, pred_t1], dtype=np.float32).clip(-6, 6)
'''


def main():
    parser = argparse.ArgumentParser(description="Build per-target asymmetric ensemble")
    parser.add_argument("--source", required=True, help="Source zip with ONNX models")
    parser.add_argument("--weights-t0", nargs="+", type=float, required=True)
    parser.add_argument("--weights-t1", nargs="+", type=float, required=True)
    parser.add_argument("--output", required=True, help="Output zip path")
    args = parser.parse_args()

    # Extract ONNX files from source
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(args.source, 'r') as zf:
            onnx_files = sorted([f for f in zf.namelist() if f.endswith('.onnx')])
            for f in onnx_files:
                zf.extract(f, tmpdir)

        n_models = len(onnx_files)
        if len(args.weights_t0) != n_models or len(args.weights_t1) != n_models:
            raise ValueError(
                f"Got {len(args.weights_t0)} t0 weights and {len(args.weights_t1)} t1 weights "
                f"for {n_models} models"
            )

        # Build MODEL_CONFIGS with per-target weights
        model_configs = []
        for i, fname in enumerate(onnx_files):
            model_configs.append((fname, 64, 3, args.weights_t0[i], args.weights_t1[i]))

        print(f"Building per-target ensemble: {n_models} models")
        print(f"  fold2_seed42 (model_12): t0_w={args.weights_t0[12]}, t1_w={args.weights_t1[12]}")
        print(f"  Other models: t0_w={args.weights_t0[0]}, t1_w={args.weights_t1[0]}")

        # Write solution.py
        solution_code = SOLUTION_TEMPLATE.format(
            n_models=n_models,
            model_configs=repr(model_configs),
        )
        with open(os.path.join(tmpdir, "solution.py"), "w") as f:
            f.write(solution_code)

        # Create output zip
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        base = args.output.replace(".zip", "")
        shutil.make_archive(base, "zip", tmpdir)

        size_mb = os.path.getsize(args.output) / 1e6
        print(f"\nCreated: {args.output} ({size_mb:.1f}MB)")
        if size_mb > 20:
            print("ERROR: Exceeds 20MB limit!")
        else:
            print(f"Size OK ({size_mb:.1f}MB < 20MB limit)")


if __name__ == "__main__":
    main()
