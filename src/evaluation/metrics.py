"""Evaluation metrics and loss functions for the competition."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
import sys
import os

# Import competition metric
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import weighted_pearson_correlation


def compute_weighted_pearson(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    target_idx: Optional[int] = None
) -> float:
    """Compute weighted Pearson correlation using competition metric.
    
    Args:
        y_true: Ground truth, shape (N,) or (N, 2)
        y_pred: Predictions, shape (N,) or (N, 2)
        target_idx: If provided, compute for specific target (0 or 1)
    
    Returns:
        Weighted Pearson correlation coefficient
    """
    if target_idx is not None:
        if len(y_true.shape) > 1:
            y_true = y_true[:, target_idx]
        if len(y_pred.shape) > 1:
            y_pred = y_pred[:, target_idx]
    
    return weighted_pearson_correlation(y_true.flatten(), y_pred.flatten())


class WeightedMSELoss(nn.Module):
    """MSE loss weighted by target amplitude.

    Emphasizes large price movements to align with competition metric
    which uses |target| as weight in weighted Pearson correlation.
    """

    def __init__(self, eps: float = 1e-8, target_weights: Optional[list] = None):
        """Initialize loss.

        Args:
            eps: Small constant to prevent division by zero
            target_weights: Per-target weights [w_t0, w_t1]. None = equal weight.
        """
        super().__init__()
        self.eps = eps
        if target_weights is not None:
            self.register_buffer(
                'target_weights',
                torch.tensor(target_weights, dtype=torch.float32),
            )
        else:
            self.target_weights = None
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        temporal_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute weighted MSE loss.

        Args:
            predictions: (batch, seq_len, 2) predicted values
            targets: (batch, seq_len, 2) true values
            mask: (batch, seq_len) bool mask for valid predictions
            temporal_weights: (seq_len,) per-step weights for recency weighting

        Returns:
            Scalar loss value
        """
        # Compute weights from target amplitude
        weights = torch.abs(targets) + self.eps

        # Apply per-target weights (e.g. upweight t1)
        if self.target_weights is not None:
            weights = weights * self.target_weights

        # Compute squared errors
        sq_errors = (predictions - targets) ** 2

        # Apply mask if provided
        if mask is not None:
            # Expand mask to match target dimensions
            mask = mask.unsqueeze(-1).expand_as(weights)
            weights = weights * mask.float()

        # Apply temporal weights for recency weighting
        if temporal_weights is not None:
            tw = temporal_weights.view(1, -1, 1).expand_as(weights)
            weights = weights * tw

        # Weighted mean
        loss = (weights * sq_errors).sum() / (weights.sum() + self.eps)

        return loss


class CombinedLoss(nn.Module):
    """Combination of MSE and weighted MSE for training.
    
    Blends standard MSE (good gradient signal everywhere) with
    weighted MSE (emphasizes large movements like competition metric).
    """
    
    def __init__(
        self,
        weighted_ratio: float = 0.5,
        eps: float = 1e-8,
        target_weights: Optional[list] = None,
    ):
        """Initialize loss.

        Args:
            weighted_ratio: Blend ratio. 0 = pure MSE, 1 = pure weighted MSE
            eps: Small constant for numerical stability
            target_weights: Per-target weights [w_t0, w_t1]. None = equal weight.
        """
        super().__init__()
        self.weighted_ratio = weighted_ratio
        self.eps = eps
        self.mse = nn.MSELoss(reduction='none')
        self.weighted_mse = WeightedMSELoss(eps=eps, target_weights=target_weights)
        if target_weights is not None:
            self.register_buffer(
                'target_weights',
                torch.tensor(target_weights, dtype=torch.float32),
            )
        else:
            self.target_weights = None
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        temporal_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute combined loss.

        Args:
            predictions: (batch, seq_len, 2) predicted values
            targets: (batch, seq_len, 2) true values
            mask: (batch, seq_len) bool mask for valid predictions
            temporal_weights: (seq_len,) per-step weights for recency weighting

        Returns:
            Scalar loss value
        """
        # Plain MSE component (with optional per-target weighting)
        mse_raw = self.mse(predictions, targets)
        if self.target_weights is not None:
            mse_raw = mse_raw * self.target_weights
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(mse_raw)
            if temporal_weights is not None:
                tw = temporal_weights.view(1, -1, 1).expand_as(mse_raw)
                step_weights = mask_expanded.float() * tw
                plain_mse = (mse_raw * step_weights).sum() / (step_weights.sum() + 1e-8)
            else:
                plain_mse = mse_raw[mask_expanded].mean()
        else:
            if temporal_weights is not None:
                tw = temporal_weights.view(1, -1, 1).expand_as(mse_raw)
                plain_mse = (mse_raw * tw).sum() / (tw.sum() + 1e-8)
            else:
                plain_mse = mse_raw.mean()

        # Weighted MSE component
        weighted = self.weighted_mse(predictions, targets, mask, temporal_weights)

        # Combine
        return (1 - self.weighted_ratio) * plain_mse + self.weighted_ratio * weighted


class WeightedPearsonLoss(nn.Module):
    """Differentiable weighted Pearson correlation loss.

    Directly optimizes the competition metric: weighted Pearson correlation
    where weights = |target|. Loss = -(corr_t0 + corr_t1) / 2.

    Matches the competition implementation in utils.py:
    - weights = max(|y_true|, eps)
    - predictions clipped to [-6, 6]
    - weighted mean, covariance, variance
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def _weighted_pearson(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted Pearson correlation for a single target.

        Args:
            pred: (N,) predictions (already clipped)
            target: (N,) ground truth
            weights: (N,) sample weights (|target| + eps)

        Returns:
            Scalar correlation in [-1, 1]
        """
        sum_w = weights.sum()

        # Weighted means
        mean_pred = (weights * pred).sum() / sum_w
        mean_target = (weights * target).sum() / sum_w

        # Weighted deviations
        dev_pred = pred - mean_pred
        dev_target = target - mean_target

        # Weighted covariance and variances
        cov = (weights * dev_pred * dev_target).sum() / sum_w
        var_pred = (weights * dev_pred ** 2).sum() / sum_w
        var_target = (weights * dev_target ** 2).sum() / sum_w

        # Stability: floor variances to avoid division by zero
        std_pred = torch.sqrt(var_pred.clamp(min=self.eps))
        std_target = torch.sqrt(var_target.clamp(min=self.eps))

        corr = cov / (std_pred * std_target)
        return corr

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        temporal_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute negative weighted Pearson correlation loss.

        Args:
            predictions: (batch, seq_len, 2) predicted values
            targets: (batch, seq_len, 2) true values
            mask: (batch, seq_len) bool mask for valid predictions
            temporal_weights: accepted for interface compatibility (unused)

        Returns:
            Scalar loss = -(corr_t0 + corr_t1) / 2
        """
        # Clip predictions to match competition metric
        predictions = predictions.clamp(-6.0, 6.0)

        if mask is not None:
            mask_flat = mask.reshape(-1)
            pred_flat = predictions.reshape(-1, 2)[mask_flat]
            target_flat = targets.reshape(-1, 2)[mask_flat]
        else:
            pred_flat = predictions.reshape(-1, 2)
            target_flat = targets.reshape(-1, 2)

        # Need enough samples for meaningful correlation
        if pred_flat.shape[0] < 2:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        corrs = []
        for t in range(2):
            w = torch.abs(target_flat[:, t]).clamp(min=self.eps)
            corr = self._weighted_pearson(pred_flat[:, t], target_flat[:, t], w)
            corrs.append(corr)

        avg_corr = (corrs[0] + corrs[1]) / 2.0

        # Negate: we minimize loss, but want to maximize correlation
        return -avg_corr


class PearsonCombinedLoss(nn.Module):
    """Hybrid loss: alpha * CombinedLoss + (1 - alpha) * (1 - weighted_corr).

    Blends stable MSE-based gradients with metric-aligned Pearson signal.
    Default alpha=0.6 gives 60% CombinedLoss + 40% Pearson alignment.
    """

    def __init__(
        self,
        alpha: float = 0.6,
        weighted_ratio: float = 0.62,
        eps: float = 1e-6,
        target_weights: Optional[list] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.combined = CombinedLoss(
            weighted_ratio=weighted_ratio, target_weights=target_weights,
        )
        self.pearson = WeightedPearsonLoss(eps=eps)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        temporal_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Guard against empty mask (zero valid predictions in batch)
        if mask is not None and mask.sum() == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        combined_loss = self.combined(predictions, targets, mask, temporal_weights)
        pearson_loss = self.pearson(predictions, targets, mask)
        # pearson_loss is -corr, so (1 - corr) = 1 + pearson_loss
        loss = self.alpha * combined_loss + (1.0 - self.alpha) * (1.0 + pearson_loss)
        # Final NaN guard (should not happen, but prevents training crash)
        if not loss.isfinite():
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        return loss


class PearsonPrimaryLoss(nn.Module):
    """Metric-aligned loss: Pearson primary with Huber stabilizer.

    Loss = alpha * pearson_component + (1 - alpha) * huber_component

    Where:
    - pearson_component = target_ratio*(1-rho_t0) + (1-target_ratio)*(1-rho_t1)
    - huber_component = target_ratio*huber_t0 + (1-target_ratio)*huber_t1
    - alpha linearly ramps from warmup_alpha to alpha over warmup_epochs

    Key improvements over PearsonCombinedLoss:
    1. Asymmetric t0/t1 weighting (0.62/0.38 matches competition metric)
    2. Pearson-dominant after warmup (0.80 vs 0.40)
    3. Huber stabilizer instead of CombinedLoss (more robust to outliers)
    """

    def __init__(
        self,
        alpha: float = 0.80,
        warmup_alpha: float = 0.4,
        warmup_epochs: int = 3,
        huber_delta: float = 1.0,
        target_ratio: float = 0.62,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.alpha = alpha
        self.warmup_alpha = warmup_alpha
        self.warmup_epochs = warmup_epochs
        self.huber_delta = huber_delta
        self.target_ratio = target_ratio
        self.eps = eps
        self.current_epoch = 0
        self.huber = nn.HuberLoss(reduction='none', delta=huber_delta)

    def set_epoch(self, epoch: int):
        """Update current epoch for warmup scheduling."""
        self.current_epoch = epoch

    @property
    def current_alpha(self) -> float:
        """Get alpha for current epoch (linear ramp during warmup)."""
        if self.current_epoch >= self.warmup_epochs:
            return self.alpha
        progress = self.current_epoch / max(1, self.warmup_epochs)
        return self.warmup_alpha + (self.alpha - self.warmup_alpha) * progress

    def _weighted_pearson(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted Pearson correlation for a single target."""
        sum_w = weights.sum()
        mean_pred = (weights * pred).sum() / sum_w
        mean_target = (weights * target).sum() / sum_w
        dev_pred = pred - mean_pred
        dev_target = target - mean_target
        cov = (weights * dev_pred * dev_target).sum() / sum_w
        var_pred = (weights * dev_pred ** 2).sum() / sum_w
        var_target = (weights * dev_target ** 2).sum() / sum_w
        std_pred = torch.sqrt(var_pred.clamp(min=self.eps))
        std_target = torch.sqrt(var_target.clamp(min=self.eps))
        return cov / (std_pred * std_target)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        temporal_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Pearson-primary loss with Huber stabilizer.

        Args:
            predictions: (batch, seq_len, 2) predicted values
            targets: (batch, seq_len, 2) true values
            mask: (batch, seq_len) bool mask for valid predictions
            temporal_weights: accepted for interface compatibility (unused)

        Returns:
            Scalar loss value
        """
        if mask is not None and mask.sum() == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        predictions = predictions.clamp(-6.0, 6.0)

        if mask is not None:
            mask_flat = mask.reshape(-1)
            pred_flat = predictions.reshape(-1, 2)[mask_flat]
            target_flat = targets.reshape(-1, 2)[mask_flat]
        else:
            pred_flat = predictions.reshape(-1, 2)
            target_flat = targets.reshape(-1, 2)

        if pred_flat.shape[0] < 2:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        # Pearson component: asymmetric t0/t1 weighting
        w_t0 = torch.abs(target_flat[:, 0]).clamp(min=self.eps)
        w_t1 = torch.abs(target_flat[:, 1]).clamp(min=self.eps)
        rho_t0 = self._weighted_pearson(pred_flat[:, 0], target_flat[:, 0], w_t0)
        rho_t1 = self._weighted_pearson(pred_flat[:, 1], target_flat[:, 1], w_t1)
        pearson_loss = (
            self.target_ratio * (1.0 - rho_t0)
            + (1.0 - self.target_ratio) * (1.0 - rho_t1)
        )

        # Huber component: per-target with same asymmetric weighting
        huber_t0 = self.huber(pred_flat[:, 0], target_flat[:, 0]).mean()
        huber_t1 = self.huber(pred_flat[:, 1], target_flat[:, 1]).mean()
        huber_loss = (
            self.target_ratio * huber_t0
            + (1.0 - self.target_ratio) * huber_t1
        )

        # Combine with warmup-ramped alpha
        alpha = self.current_alpha
        loss = alpha * pearson_loss + (1.0 - alpha) * huber_loss

        if not loss.isfinite():
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        return loss


class HuberWeightedLoss(nn.Module):
    """Weighted Huber loss for robust training.

    Combines robustness to outliers (Huber) with emphasis
    on large movements (amplitude weighting).
    """
    
    def __init__(self, delta: float = 1.0, eps: float = 1e-8):
        """Initialize loss.
        
        Args:
            delta: Threshold for Huber loss (MSE below, MAE above)
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.delta = delta
        self.eps = eps
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        temporal_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute weighted Huber loss."""
        # Compute weights from target amplitude
        weights = torch.abs(targets) + self.eps

        # Compute absolute errors
        errors = torch.abs(predictions - targets)

        # Huber loss: MSE where |error| < delta, MAE otherwise
        huber = torch.where(
            errors < self.delta,
            0.5 * errors ** 2,
            self.delta * (errors - 0.5 * self.delta)
        )

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(-1).expand_as(weights)
            weights = weights * mask.float()

        # Apply temporal weights for recency weighting
        if temporal_weights is not None:
            tw = temporal_weights.view(1, -1, 1).expand_as(weights)
            weights = weights * tw

        # Weighted mean
        loss = (weights * huber).sum() / (weights.sum() + self.eps)

        return loss


class AuxHeadLoss(nn.Module):
    """Wraps a base loss with auxiliary head losses (delta, sign_t0, sign_t1).

    Aux heads regularize the shared encoder during training. They are dropped
    at inference, so there is zero runtime cost.

    Loss = L_main + alpha * (delta_w * L_delta + sign_w * (L_sign0 + L_sign1))
    where alpha anneals linearly to 0 in the last 20% of epochs.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        delta_weight: float = 0.2,
        sign_weight: float = 0.1,
        total_epochs: int = 35,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.delta_weight = delta_weight
        self.sign_weight = sign_weight
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
        if hasattr(self.base_loss, 'set_epoch'):
            self.base_loss.set_epoch(epoch)

    @property
    def aux_alpha(self) -> float:
        """1.0 for first 80% of epochs, linear decay to 0 in last 20%."""
        anneal_start = int(0.8 * self.total_epochs)
        if self.current_epoch < anneal_start:
            return 1.0
        progress = (self.current_epoch - anneal_start) / max(1, self.total_epochs - anneal_start)
        return max(0.0, 1.0 - progress)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        temporal_weights: Optional[torch.Tensor] = None,
        aux_predictions: Optional[dict] = None,
    ) -> torch.Tensor:
        main_loss = self.base_loss(predictions, targets, mask, temporal_weights)

        if aux_predictions is None or self.aux_alpha == 0:
            return main_loss

        # Flatten and mask
        if mask is not None:
            mask_flat = mask.reshape(-1)
            target_flat = targets.reshape(-1, 2)[mask_flat]
            delta_pred = aux_predictions['delta'].reshape(-1)[mask_flat]
            sign_t0_pred = aux_predictions['sign_t0'].reshape(-1)[mask_flat]
            sign_t1_pred = aux_predictions['sign_t1'].reshape(-1)[mask_flat]
        else:
            target_flat = targets.reshape(-1, 2)
            delta_pred = aux_predictions['delta'].reshape(-1)
            sign_t0_pred = aux_predictions['sign_t0'].reshape(-1)
            sign_t1_pred = aux_predictions['sign_t1'].reshape(-1)

        # Aux targets
        delta_target = target_flat[:, 1] - target_flat[:, 0]
        sign_t0_target = (target_flat[:, 0] > 0).float()
        sign_t1_target = (target_flat[:, 1] > 0).float()

        # Aux losses
        l_delta = self.mse(delta_pred, delta_target)
        l_sign0 = self.bce(sign_t0_pred, sign_t0_target)
        l_sign1 = self.bce(sign_t1_pred, sign_t1_target)

        alpha = self.aux_alpha
        aux_loss = self.delta_weight * l_delta + self.sign_weight * (l_sign0 + l_sign1)

        return main_loss + alpha * aux_loss
