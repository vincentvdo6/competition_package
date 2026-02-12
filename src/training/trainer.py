"""Training loop with validation and logging.

Supports both CPU and GPU (CUDA/ROCm):
- CPU: uses all cores via set_num_threads
- GPU: pin_memory transfers, mixed precision (AMP) via GradScaler
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau, LambdaLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import json
import math
import os
import time
import sys
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


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply(self, model: nn.Module) -> dict:
        """Swap model params with EMA params. Returns backup for restore."""
        backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        return backup

    def restore(self, model: nn.Module, backup: dict):
        """Restore model params from backup."""
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])


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
        self.device = device

        # Wrap loss with AuxHeadLoss if model has aux heads
        self.use_aux = hasattr(model, 'has_aux_heads') and model.has_aux_heads
        if self.use_aux:
            from src.evaluation.metrics import AuxHeadLoss
            train_cfg_tmp = config.get('training', {})
            self.loss_fn = AuxHeadLoss(
                base_loss=loss_fn,
                delta_weight=float(train_cfg_tmp.get('aux_delta_weight', 0.2)),
                sign_weight=float(train_cfg_tmp.get('aux_sign_weight', 0.1)),
                total_epochs=int(train_cfg_tmp.get('epochs', 50)),
            ).to(device)
        else:
            self.loss_fn = loss_fn.to(device)

        # Training parameters
        train_cfg = config.get('training', {})
        self.epochs = int(train_cfg.get('epochs', 50))
        self.gradient_clip = float(train_cfg.get('gradient_clip', 1.0))
        self.patience = int(train_cfg.get('early_stopping_patience', 10))
        self.batch_log_interval = int(train_cfg.get('batch_log_interval', 10))
        self.use_tqdm = bool(train_cfg.get('use_tqdm', True)) and sys.stdout.isatty()

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
        elif sched_type == 'cosine_warmup':
            base_lr = float(train_cfg.get('lr', 0.001))
            warmup_frac = float(sched_cfg.get('warmup_fraction', 0.1))
            min_lr = float(sched_cfg.get('min_lr', 2e-5))
            min_lr_ratio = min_lr / base_lr
            warmup_epochs = max(1, int(warmup_frac * self.epochs))
            total_epochs = self.epochs

            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / warmup_epochs
                progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
                return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

            self.scheduler = LambdaLR(self.optimizer, lr_lambda)
            self.sched_type = 'cosine_warmup'
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

        # EMA (Exponential Moving Average)
        self.use_ema = bool(train_cfg.get('ema', False))
        if self.use_ema:
            ema_decay = float(train_cfg.get('ema_decay', 0.999))
            self.ema = EMA(model, decay=ema_decay)
        else:
            self.ema = None

        # Periodic checkpoint saving (for checkpoint averaging)
        self.save_every = int(config.get('logging', {}).get('save_every', 0))

        # Logging
        log_cfg = config.get('logging', {})
        self.log_dir = log_cfg.get('log_dir', 'logs')
        self.experiment_log = os.path.join(self.log_dir, 'experiments.jsonl')
        os.makedirs(self.log_dir, exist_ok=True)

        # Data augmentation (variance stretch/compress)
        aug_cfg = train_cfg.get('augmentation', {})
        self.use_augmentation = bool(aug_cfg.get('enabled', False))
        if self.use_augmentation:
            self.aug_scale_low = float(aug_cfg.get('scale_range', [0.8, 1.2])[0])
            self.aug_scale_high = float(aug_cfg.get('scale_range', [0.8, 1.2])[1])
            print(f"Augmentation: scale [{self.aug_scale_low}, {self.aug_scale_high}]",
                  flush=True)

        # Recency weighting: compute temporal weight tensor once
        recency_cfg = train_cfg.get('recency_weighting', {})
        if recency_cfg.get('enabled', False):
            self.temporal_weights = self._compute_temporal_weights(recency_cfg).to(device)
            print(f"Recency weighting: {recency_cfg.get('type', 'linear')} "
                  f"w=[{recency_cfg.get('w_min', 1.0)}, {recency_cfg.get('w_max', 2.0)}] "
                  f"steps [{recency_cfg.get('start_step', 99)}, {recency_cfg.get('end_step', 999)}]",
                  flush=True)
        else:
            self.temporal_weights = None

        # SWA (Stochastic Weight Averaging) — average last N epoch checkpoints
        swa_cfg = train_cfg.get('swa', {})
        self.swa_enabled = bool(swa_cfg.get('enabled', False))
        self.swa_start_epoch = int(swa_cfg.get('start_epoch', self.epochs - 5))
        self.swa_lr = float(swa_cfg.get('lr', float(train_cfg.get('lr', 0.001)) * 0.1))
        if self.swa_enabled:
            print(f"SWA: enabled from epoch {self.swa_start_epoch}, lr={self.swa_lr}",
                  flush=True)

        # Truncated BPTT (windowed training)
        tbptt_cfg = train_cfg.get('tbptt', {})
        self.tbptt_enabled = bool(tbptt_cfg.get('enabled', False))
        self.tbptt_len = int(tbptt_cfg.get('length', 100))
        self.tbptt_random_offset = bool(tbptt_cfg.get('random_offset', True))
        if self.tbptt_enabled:
            print(f"TBPTT: length={self.tbptt_len}, random_offset={self.tbptt_random_offset}",
                  flush=True)

        # Early stopping state
        self.best_score = -float('inf')
        self.epochs_without_improvement = 0
        self.best_epoch = 0

    def train_epoch(self) -> float:
        """Train for one epoch. Supports CPU, CUDA, and mixed precision."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        if self.use_tqdm:
            batch_iter = tqdm(self.train_loader, desc="Training", leave=False)
        else:
            batch_iter = self.train_loader

        for batch_idx, (features, targets, mask) in enumerate(batch_iter, start=1):
            features = features.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)

            # Data augmentation: per-sequence variance stretch/compress
            if self.use_augmentation:
                batch_size = features.shape[0]
                scale = torch.empty(batch_size, 1, 1, device=self.device).uniform_(
                    self.aug_scale_low, self.aug_scale_high
                )
                features = features * scale

            if self.tbptt_enabled:
                # Truncated BPTT: process sequence in chunks
                seq_len = features.shape[1]
                offset = torch.randint(0, self.tbptt_len, (1,)).item() if self.tbptt_random_offset else 0
                hidden = None
                chunk_loss_sum = 0.0
                n_chunks = 0

                for t_start in range(offset, seq_len, self.tbptt_len):
                    t_end = min(t_start + self.tbptt_len, seq_len)
                    feat_chunk = features[:, t_start:t_end]
                    tgt_chunk = targets[:, t_start:t_end]
                    mask_chunk = mask[:, t_start:t_end]

                    if hidden is not None:
                        hidden = hidden.detach()

                    self.optimizer.zero_grad(set_to_none=True)

                    with autocast('cuda', enabled=self.use_amp):
                        pred_chunk, hidden = self.model(feat_chunk, hidden)
                        loss = self.loss_fn(pred_chunk, tgt_chunk, mask_chunk)

                    self.scaler.scale(loss).backward()

                    if self.gradient_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    chunk_loss_sum += loss.item()
                    n_chunks += 1

                if self.ema is not None:
                    self.ema.update(self.model)

                total_loss += chunk_loss_sum / max(n_chunks, 1)
            else:
                self.optimizer.zero_grad(set_to_none=True)

                with autocast('cuda', enabled=self.use_amp):
                    if self.use_aux:
                        predictions, _, aux = self.model(features, return_aux=True)
                        loss = self.loss_fn(predictions, targets, mask,
                                            temporal_weights=self.temporal_weights,
                                            aux_predictions=aux)
                    else:
                        predictions, _ = self.model(features)
                        if self.temporal_weights is not None:
                            loss = self.loss_fn(predictions, targets, mask,
                                                temporal_weights=self.temporal_weights)
                        else:
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

                if self.ema is not None:
                    self.ema.update(self.model)

                if self.sched_type == 'one_cycle':
                    self.scheduler.step()

                total_loss += loss.item()
            if self.use_tqdm:
                batch_iter.set_postfix({'loss': f'{loss.item():.4f}'})
            elif (
                batch_idx % max(1, self.batch_log_interval) == 0
                or batch_idx == num_batches
            ):
                print(
                    f"    Batch {batch_idx:4d}/{num_batches} | "
                    f"loss: {loss.item():.4f}",
                    flush=True,
                )

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
        print(f"Starting training for up to {self.epochs} epochs", flush=True)
        print(f"Device: {self.device}", flush=True)
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
            print(f"Mixed precision (AMP): {self.use_amp}", flush=True)
        else:
            print(f"CPU threads: {torch.get_num_threads()}", flush=True)
        print(f"Model parameters: {self.model.count_parameters():,}", flush=True)
        print(
            f"Train batches: {len(self.train_loader)}, Valid batches: {len(self.valid_loader)}",
            flush=True,
        )
        print(f"Batch size: {self.train_loader.batch_size}", flush=True)
        print(f"Scheduler: {self.sched_type}", flush=True)
        if self.use_ema:
            print(f"EMA: decay={self.ema.decay}", flush=True)
        print(f"Progress mode: {'tqdm' if self.use_tqdm else 'batch logs'}", flush=True)
        print("-" * 60, flush=True)

        history = {
            'train_loss': [],
            'val_scores': [],
            'learning_rates': []
        }

        total_start = time.time()

        # SWA state
        swa_model = None
        swa_scheduler = None
        swa_active = False
        swa_n_averaged = 0

        for epoch in range(self.epochs):
            epoch_start = time.time()

            # Activate SWA at the designated epoch
            if self.swa_enabled and epoch == self.swa_start_epoch and not swa_active:
                swa_model = AveragedModel(self.model, device=self.device)
                swa_scheduler = SWALR(self.optimizer, swa_lr=self.swa_lr)
                swa_active = True
                self.epochs_without_improvement = 0  # reset early stopping
                print(f"\n>>> SWA activated at epoch {epoch+1}, lr={self.swa_lr}", flush=True)

            if hasattr(self.loss_fn, 'set_epoch'):
                self.loss_fn.set_epoch(epoch)

            train_loss = self.train_epoch()
            history['train_loss'].append(train_loss)

            # Swap to EMA weights for validation if enabled
            ema_backup = None
            if self.ema is not None:
                ema_backup = self.ema.apply(self.model)

            avg_score, scores = self.validate()
            history['val_scores'].append(scores)

            # Restore training weights after validation
            if ema_backup is not None:
                self.ema.restore(self.model, ema_backup)

            # SWA: update averaged model and use SWA scheduler
            if swa_active:
                swa_model.update_parameters(self.model)
                swa_scheduler.step()
                swa_n_averaged += 1
            elif self.sched_type == 'reduce_on_plateau':
                self.scheduler.step(avg_score)
            elif self.sched_type == 'cosine_warmup':
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)

            epoch_time = time.time() - epoch_start

            swa_tag = " [SWA]" if swa_active else ""
            print(f"Epoch {epoch+1:3d}/{self.epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Val t0: {scores['t0']:.4f} | "
                  f"Val t1: {scores['t1']:.4f} | "
                  f"Val avg: {scores['avg']:.4f} | "
                  f"LR: {current_lr:.2e} | "
                  f"{epoch_time:.1f}s{swa_tag}", flush=True)

            if avg_score > self.best_score:
                self.best_score = avg_score
                self.best_epoch = epoch + 1
                self.epochs_without_improvement = 0
                prefix = self.config.get('logging', {}).get('checkpoint_prefix', None)
                ckpt_name = f"{prefix}.pt" if prefix else 'best_model.pt'
                self._save_checkpoint('best_model.pt', use_ema=True)
                print(f"  -> New best model saved! ({ckpt_name})", flush=True)
            else:
                self.epochs_without_improvement += 1
                # Don't early stop during SWA — let all SWA epochs run
                if not swa_active and self.epochs_without_improvement >= self.patience:
                    print(
                        f"\nEarly stopping at epoch {epoch+1} "
                        f"(no improvement for {self.patience} epochs)",
                        flush=True,
                    )
                    break

            # Periodic checkpoint saving (for checkpoint averaging)
            if self.save_every > 0 and (epoch + 1) % self.save_every == 0:
                self._save_epoch_checkpoint(epoch + 1, use_ema=True)

        # SWA finalization: update BN stats and save SWA model
        if swa_active and swa_model is not None and swa_n_averaged >= 2:
            print(f"\nFinalizing SWA (averaged {swa_n_averaged} checkpoints)...", flush=True)
            # Update batch normalization statistics with SWA model
            update_bn(self.train_loader, swa_model, device=self.device)
            # Validate SWA model
            # Temporarily swap model for validation
            orig_model = self.model
            self.model = swa_model.module
            swa_score, swa_scores = self.validate()
            self.model = orig_model
            print(f"SWA val: t0={swa_scores['t0']:.4f} | t1={swa_scores['t1']:.4f} | "
                  f"avg={swa_scores['avg']:.4f}", flush=True)
            # Save SWA model if better
            if swa_score > self.best_score:
                print(f"SWA improved: {swa_score:.4f} > {self.best_score:.4f}", flush=True)
                self.best_score = swa_score
                self.best_epoch = -1  # mark as SWA
                # Save SWA weights
                orig_model = self.model
                self.model = swa_model.module
                self._save_checkpoint('best_model.pt', use_ema=False)
                self.model = orig_model
            else:
                print(f"SWA not better: {swa_score:.4f} <= {self.best_score:.4f} (keeping best)",
                      flush=True)

        total_time = time.time() - total_start
        print("-" * 60, flush=True)
        print(f"Training complete in {total_time:.0f}s ({total_time/60:.1f}min)", flush=True)
        print(f"Best score: {self.best_score:.4f} at epoch {self.best_epoch}", flush=True)

        self._log_experiment(history)
        self._save_history(history)

        return history

    def _save_checkpoint(self, filename: str, use_ema: bool = False) -> None:
        """Save model checkpoint.

        Args:
            filename: Checkpoint filename (may be overridden by checkpoint_prefix)
            use_ema: If True and EMA is active, save EMA weights instead of training weights
        """
        prefix = self.config.get('logging', {}).get('checkpoint_prefix', None)
        if prefix:
            filename = f"{prefix}.pt"
        path = os.path.join(self.log_dir, filename)

        # Use EMA weights if requested and available
        if use_ema and self.ema is not None:
            backup = self.ema.apply(self.model)
            state_dict = self.model.state_dict()
            self.ema.restore(self.model, backup)
        else:
            state_dict = self.model.state_dict()

        torch.save({
            'model_state_dict': state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'seed': self.config.get('seed', None)
        }, path)

    def _save_epoch_checkpoint(self, epoch: int, use_ema: bool = False) -> None:
        """Save lightweight checkpoint for later averaging (model weights only)."""
        prefix = self.config.get('logging', {}).get('checkpoint_prefix', 'model')

        if use_ema and self.ema is not None:
            backup = self.ema.apply(self.model)
            state_dict = self.model.state_dict()
            self.ema.restore(self.model, backup)
        else:
            state_dict = self.model.state_dict()

        path = os.path.join(self.log_dir, f"{prefix}_epoch{epoch}.pt")
        torch.save({'model_state_dict': state_dict}, path)

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
            'seed': self.config.get('seed', None),
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

    def _save_history(self, history: Dict) -> None:
        """Save per-epoch training history to JSON for curve diagnostics."""
        prefix = self.config.get('logging', {}).get('checkpoint_prefix', None)
        if prefix:
            history_name = f"training_history_{prefix}.json"
        else:
            history_name = "training_history.json"
        history_path = os.path.join(self.log_dir, history_name)
        serializable = {
            'train_loss': history['train_loss'],
            'val_scores': history['val_scores'],
            'learning_rates': history['learning_rates'],
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'seed': self.config.get('seed', None),
        }
        with open(history_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"Training history saved to {history_path}")

    @staticmethod
    def _compute_temporal_weights(recency_cfg: dict) -> torch.Tensor:
        """Compute per-timestep weights for recency-weighted loss.

        Args:
            recency_cfg: dict with keys: type, w_min, w_max, start_step, end_step

        Returns:
            Tensor of shape (1000,) with weights per timestep.
        """
        seq_len = 1000
        w_min = float(recency_cfg.get('w_min', 1.0))
        w_max = float(recency_cfg.get('w_max', 2.0))
        start_step = int(recency_cfg.get('start_step', 99))
        end_step = int(recency_cfg.get('end_step', 999))
        ramp_type = recency_cfg.get('type', 'linear')

        weights = torch.ones(seq_len)
        span = max(end_step - start_step, 1)

        for t in range(seq_len):
            if t < start_step:
                # Masked steps get weight 1.0 (mask handles exclusion)
                weights[t] = 1.0
            else:
                u = min((t - start_step) / span, 1.0)
                if ramp_type == 'exponential':
                    # Exponential ramp: w_min at u=0, w_max at u=1
                    weights[t] = w_min * (w_max / w_min) ** u
                elif ramp_type == 'step':
                    # Step function: w_min for first half, w_max for second half
                    weights[t] = w_max if u >= 0.5 else w_min
                else:
                    # Linear ramp (default)
                    weights[t] = w_min + (w_max - w_min) * u

        return weights
