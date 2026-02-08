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
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute weighted MSE loss.
        
        Args:
            predictions: (batch, seq_len, 2) predicted values
            targets: (batch, seq_len, 2) true values
            mask: (batch, seq_len) bool mask for valid predictions
        
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
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute combined loss.
        
        Args:
            predictions: (batch, seq_len, 2) predicted values
            targets: (batch, seq_len, 2) true values
            mask: (batch, seq_len) bool mask for valid predictions
        
        Returns:
            Scalar loss value
        """
        # Plain MSE component (with optional per-target weighting)
        mse_raw = self.mse(predictions, targets)
        if self.target_weights is not None:
            mse_raw = mse_raw * self.target_weights
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(mse_raw)
            plain_mse = mse_raw[mask_expanded].mean()
        else:
            plain_mse = mse_raw.mean()
        
        # Weighted MSE component
        weighted = self.weighted_mse(predictions, targets, mask)
        
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
    ) -> torch.Tensor:
        """Compute negative weighted Pearson correlation loss.

        Args:
            predictions: (batch, seq_len, 2) predicted values
            targets: (batch, seq_len, 2) true values
            mask: (batch, seq_len) bool mask for valid predictions

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
    ) -> torch.Tensor:
        # Guard against empty mask (zero valid predictions in batch)
        if mask is not None and mask.sum() == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        combined_loss = self.combined(predictions, targets, mask)
        pearson_loss = self.pearson(predictions, targets, mask)
        # pearson_loss is -corr, so (1 - corr) = 1 + pearson_loss
        loss = self.alpha * combined_loss + (1.0 - self.alpha) * (1.0 + pearson_loss)
        # Final NaN guard (should not happen, but prevents training crash)
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
        mask: Optional[torch.Tensor] = None
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
        
        # Weighted mean
        loss = (weights * huber).sum() / (weights.sum() + self.eps)
        
        return loss
