#!/usr/bin/env python
"""SSL pretraining for vanilla GRU via masked timestep reconstruction.

Stage 1 of two-stage SSL training:
  1. This script: pretrain on masked feature reconstruction (unsupervised)
  2. scripts/train.py --resume: fine-tune on supervised targets

The saved checkpoint contains ONLY GRU + output_proj weights (ssl_reconstruction_head
is excluded) so it loads cleanly into gru_parity_v1.yaml via --resume.

Usage:
    python scripts/train_ssl.py \\
        --config configs/gru_ssl_pretrain_v1.yaml \\
        --seed 42 --device cuda

Output:
    logs/gru_ssl_pretrain_v1_seed42.pt
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.dataset import LOBSequenceDataset
from src.models.gru_baseline import GRUBaseline


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config(args.config)
    config["seed"] = args.seed

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Seed: {args.seed} | Device: {device}")

    # Data
    data_cfg = config.get("data", {})
    train_path = data_cfg.get("train_path", "datasets/train.parquet")
    batch_size = int(config.get("training", {}).get("batch_size", 192))

    print(f"Loading {train_path}...")
    dataset = LOBSequenceDataset(
        train_path,
        normalize=data_cfg.get("normalize", False),
        derived_features=data_cfg.get("derived_features", False),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Train batches: {len(loader)}")

    # Model (with ssl_pretrain: true)
    model = GRUBaseline(config)
    model = model.to(device)
    param_count = model.count_parameters()
    print(f"Model parameters: {param_count:,}")
    assert model.has_ssl_head, "ssl_pretrain must be true in config"

    # Optimizer
    train_cfg = config.get("training", {})
    lr = float(train_cfg.get("lr", 0.001))
    wd = float(train_cfg.get("weight_decay", 0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    epochs = int(train_cfg.get("epochs", 15))
    mask_ratio = float(train_cfg.get("ssl_mask_ratio", 0.25))
    use_amp = train_cfg.get("use_amp", True) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    clip = float(train_cfg.get("gradient_clip", 1.0))
    mse = nn.MSELoss()

    print(f"\nSSL pretraining: {epochs} epochs, mask_ratio={mask_ratio}")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            # batch is (features, targets, masks) from LOBSequenceDataset.__getitem__
            features = batch[0].to(device)  # (B, 1000, 32)

            # Create random mask: True = masked out
            mask = torch.rand(features.shape[0], features.shape[1], device=device) < mask_ratio

            # Zero masked positions
            x_masked = features.clone()
            x_masked[mask] = 0.0

            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    _, _, aux = model(x_masked, return_aux=True)
                    recon = aux["ssl_recon"]  # (B, 1000, 32)
                    # Loss only on masked positions
                    loss = mse(recon[mask], features[mask])
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                _, _, aux = model(x_masked, return_aux=True)
                recon = aux["ssl_recon"]
                loss = mse(recon[mask], features[mask])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch:3d}/{epochs} | SSL loss: {avg_loss:.6f}", flush=True)

    print("=" * 60)

    # Save checkpoint WITHOUT ssl_reconstruction_head (compatible with gru_parity_v1)
    log_dir = config.get("logging", {}).get("log_dir", "logs")
    os.makedirs(log_dir, exist_ok=True)

    config_name = os.path.splitext(os.path.basename(args.config))[0]
    ckpt_name = f"{config_name}_seed{args.seed}.pt"
    ckpt_path = os.path.join(log_dir, ckpt_name)

    state = {
        k: v for k, v in model.state_dict().items()
        if "ssl_reconstruction_head" not in k
    }
    torch.save({"model_state_dict": state, "config": config}, ckpt_path)
    print(f"\nSaved SSL checkpoint (ssl head excluded): {ckpt_path}")
    print(f"\nNext: python scripts/train.py --config configs/gru_parity_v1.yaml "
          f"--seed {args.seed} --device {args.device} --resume {ckpt_path}")


if __name__ == "__main__":
    main()
