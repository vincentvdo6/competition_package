"""Build a submission zip for vanilla GRU parity models (no normalizer, no derived features)."""

import argparse
import os
import shutil
import tempfile
import torch


SOLUTION_TEMPLATE = '''"""Parity model: vanilla GRU, raw 32 features, no normalization."""

import os
import numpy as np
import torch
import torch.nn as nn


class VanillaGRU(nn.Module):
    def __init__(self, input_size={input_size}, hidden_size={hidden_size},
                 num_layers={num_layers}, output_size=2):
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


class PredictionModel:
    def __init__(self, model_path=""):
        self.device = torch.device("cpu")
        torch.set_num_threads(1)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        ckpt = torch.load(
            os.path.join(base_dir, "model.pt"),
            map_location="cpu",
            weights_only=False,
        )
        state_dict = ckpt["model_state_dict"]

        self.model = VanillaGRU(
            input_size={input_size},
            hidden_size={hidden_size},
            num_layers={num_layers},
        )
        # Filter state_dict keys to match VanillaGRU
        filtered = {{}}
        for k, v in state_dict.items():
            # Map gru_baseline keys to vanilla keys
            if k.startswith("gru."):
                filtered[k] = v
            elif k.startswith("output_proj."):
                # output_proj is a Linear in vanilla mode
                new_key = k.replace("output_proj.", "fc.")
                filtered[new_key] = v
        self.model.load_state_dict(filtered)
        self.model.eval()

        self.hidden = None
        self.prev_seq_ix = None

    @torch.no_grad()
    def predict(self, data_point) -> np.ndarray:
        seq_ix = data_point.seq_ix
        if seq_ix != self.prev_seq_ix:
            self.hidden = self.model.init_hidden(1)
            self.prev_seq_ix = seq_ix

        features = data_point.state.astype(np.float32)[:32]  # raw 32 only
        x = torch.from_numpy(features).unsqueeze(0).unsqueeze(0)  # (1, 1, 32)

        pred, self.hidden = self.model(x, self.hidden)
        pred = pred.squeeze().numpy().clip(-6, 6)

        if not data_point.need_prediction:
            return None
        return pred
'''


def main():
    parser = argparse.ArgumentParser(description="Build parity model submission")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--output", default="submissions/parity_v1.zip", help="Output zip path")
    parser.add_argument("--input-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    args = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"Loaded checkpoint: {args.checkpoint}")
    if "best_score" in ckpt:
        print(f"  Val score: {ckpt['best_score']:.4f}")
    if "best_epoch" in ckpt:
        print(f"  Best epoch: {ckpt['best_epoch']}")

    # Build zip
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write solution.py
        solution_code = SOLUTION_TEMPLATE.format(
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
        )
        with open(os.path.join(tmpdir, "solution.py"), "w") as f:
            f.write(solution_code)

        # Write model checkpoint
        torch.save(ckpt, os.path.join(tmpdir, "model.pt"))

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


if __name__ == "__main__":
    main()
