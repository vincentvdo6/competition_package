"""Training loop with validation and logging.

Supports both CPU and GPU (CUDA/ROCm):
- CPU: uses all cores via set_num_threads
- GPU: pin_memory transfers, mixed precision (AMP) via GradScaler
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import json
import os
import time
from datetime import datetime
from tqdm import tqdm
from typing import Dict, Optional, Tuple, List

from src.models.base import BaseModel
from src.evaluation.metrics import compute_weighted_pearson


def setup_cpu_performance():
    """Configure PyTorch for maximum CPU throughput."""
    n_cores = os.cpu_count() or 8
    # Use all physical cores for compute threads
    torch.set_num_threads(n_cores)
    # Use remaining threads for inter-op parallelism
    torch.set_num_interop_threads(max(1, n_cores // 4))


class Trainer:
    """Training loop with validation, early stopping, and logging."""

    def __init__(
        self,
        model: BaseModel,
        config: dict,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        loss_fn: nn.Module,
        device: torch.device
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.loss_fn = loss_fn.to(device)
        self.device = device

        # Training parameters
        train_cfg = config.get('training', {})
        self.epochs = int(train_cfg.get('epochs', 50))
        self.gradient_clip = float(train_cfg.get('gradient_clip', 1.0))
        self.patience = int(train_cfg.get('early_stopping_patience', 10))

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=float(train_cfg.get('lr', 0.001)),
            weight_decay=float(train_cfg.get('weight_decay', 1e-5))
        )

        # Learning rate scheduler
        sched_cfg = train_cfg.get('scheduler', {})
        sched_type = sched_cfg.get('type', 'reduce_on_plateau')

        if sched_type == 'one_cycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=float(train_cfg.get('lr', 0.001)) * 10,
                epochs=self.epochs,
                steps_per_epoch=len(train_loader),
            )
            self.sched_type = 'one_cycle'
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=float(sched_cfg.get('factor', 0.5)),
                patience=int(sched_cfg.get('patience', 5)),
                min_lr=float(sched_cfg.get('min_lr', 1e-6))
            )
            self.sched_type = 'reduce_on_plateau'

        # Mixed precision (AMP) — only for CUDA/ROCm
        self.use_amp = (
            bool(train_cfg.get('use_amp', False))
            and device.type == 'cuda'
        )
        self.scaler = GradScaler('cuda', enabled=self.use_amp)

        # Logging
        log_cfg = config.get('logging', {})
        self.log_dir = log_cfg.get('log_dir', 'logs')
        self.experiment_log = os.path.join(self.log_dir, 'experiments.jsonl')
        os.makedirs(self.log_dir, exist_ok=True)

        # Early stopping state
        self.best_score = -float('inf')
        self.epochs_without_improvement = 0
        self.best_epoch = 0

    def train_epoch(self) -> float:
        """Train for one epoch. Supports CPU, CUDA, and mixed precision."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for features, targets, mask in pbar:
            features = features.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=self.use_amp):
                predictions, _ = self.model(features)
                loss = self.loss_fn(predictions, targets, mask)

            self.scaler.scale(loss).backward()

            if self.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.sched_type == 'one_cycle':
                self.scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """Validate and compute weighted Pearson scores.

        Vectorized: collects all predictions, converts to numpy once at end.
        """
        self.model.eval()

        all_preds = []
        all_targets = []

        for features, targets, mask in self.valid_loader:
            features = features.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)

            with autocast('cuda', enabled=self.use_amp):
                predictions, _ = self.model(features)

            predictions = predictions.float().clamp_(-6, 6)

            # Move mask-selected results to CPU for numpy conversion
            all_preds.append(predictions[mask].cpu())
            all_targets.append(targets[mask.cpu()])

        all_preds_arr = torch.cat(all_preds, dim=0).numpy()
        all_targets_arr = torch.cat(all_targets, dim=0).numpy()

        score_t0 = compute_weighted_pearson(all_targets_arr, all_preds_arr, target_idx=0)
        score_t1 = compute_weighted_pearson(all_targets_arr, all_preds_arr, target_idx=1)
        avg_score = (score_t0 + score_t1) / 2

        return avg_score, {'t0': score_t0, 't1': score_t1, 'avg': avg_score}

    def train(self) -> Dict:
        """Full training loop with early stopping and per-epoch timing."""
        print(f"Starting training for up to {self.epochs} epochs")
        print(f"Device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Mixed precision (AMP): {self.use_amp}")
        else:
            print(f"CPU threads: {torch.get_num_threads()}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"Train batches: {len(self.train_loader)}, Valid batches: {len(self.valid_loader)}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Scheduler: {self.sched_type}")
        print("-" * 60)

        history = {
            'train_loss': [],
            'val_scores': [],
            'learning_rates': []
        }

        total_start = time.time()

        for epoch in range(self.epochs):
            epoch_start = time.time()

            train_loss = self.train_epoch()
            history['train_loss'].append(train_loss)

            avg_score, scores = self.validate()
            history['val_scores'].append(scores)

            # Scheduler step (ReduceLROnPlateau steps per epoch, OneCycle steps per batch)
            if self.sched_type == 'reduce_on_plateau':
                self.scheduler.step(avg_score)
            current_lr = self.optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)

            epoch_time = time.time() - epoch_start

            print(f"Epoch {epoch+1:3d}/{self.epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Val t0: {scores['t0']:.4f} | "
                  f"Val t1: {scores['t1']:.4f} | "
                  f"Val avg: {scores['avg']:.4f} | "
                  f"LR: {current_lr:.2e} | "
                  f"{epoch_time:.1f}s")

            if avg_score > self.best_score:
                self.best_score = avg_score
                self.best_epoch = epoch + 1
                self.epochs_without_improvement = 0
                self._save_checkpoint('best_model.pt')
                print(f"  -> New best model saved!")
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {self.patience} epochs)")
                    break

        total_time = time.time() - total_start
        print("-" * 60)
        print(f"Training complete in {total_time:.0f}s ({total_time/60:.1f}min)")
        print(f"Best score: {self.best_score:.4f} at epoch {self.best_epoch}")

        self._log_experiment(history)

        return history

    def _save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        path = os.path.join(self.log_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch
        }, path)

    def _log_experiment(self, history: Dict) -> None:
        """Append experiment to JSONL log.

        Args:
            history: Training history dictionary
        """
        final_scores = history['val_scores'][-1] if history['val_scores'] else {}

        entry = {
            'timestamp': datetime.now().isoformat(),
            'model': self.config.get('model', {}).get('type', 'unknown'),
            'config': self.config,
            'val_score_t0': final_scores.get('t0'),
            'val_score_t1': final_scores.get('t1'),
            'val_score_avg': final_scores.get('avg'),
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'epochs_trained': len(history['train_loss']),
            'final_lr': history['learning_rates'][-1] if history['learning_rates'] else None,
            'notes': ''
        }

        with open(self.experiment_log, 'a') as f:
            f.write(json.dumps(entry) + '\n')

        print(f"Experiment logged to {self.experiment_log}")
