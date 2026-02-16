"""Loss function factory for training."""

import torch
import torch.nn as nn
from typing import Optional
from src.evaluation.metrics import (
    WeightedMSELoss,
    CombinedLoss,
    HuberWeightedLoss,
    WeightedPearsonLoss,
    PearsonCombinedLoss,
    PearsonPrimaryLoss,
    CoReWrapper,
)


class MaskedMSELoss(nn.Module):
    """Simple MSE loss that supports the mask argument from the trainer."""

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if mask is not None:
            mask_3d = mask.unsqueeze(-1).expand_as(predictions)
            return torch.nn.functional.mse_loss(predictions[mask_3d], targets[mask_3d])
        return torch.nn.functional.mse_loss(predictions, targets)


class MaskedHuberLoss(nn.Module):
    """Plain Huber loss with mask support. No amplitude weighting.

    Unlike HuberWeightedLoss which weights by |target|, this uses uniform weighting.
    Huber is quadratic for |error| < delta, linear for |error| >= delta.
    This changes the error shape vs MSE, producing different prediction distributions.
    """

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if mask is not None:
            mask_3d = mask.unsqueeze(-1).expand_as(predictions)
            return torch.nn.functional.huber_loss(
                predictions[mask_3d], targets[mask_3d],
                reduction='mean', delta=self.delta)
        return torch.nn.functional.huber_loss(
            predictions, targets, reduction='mean', delta=self.delta)


def get_loss_function(config: dict) -> nn.Module:
    """Factory function for loss functions.

    Args:
        config: Configuration dictionary with 'training' section containing:
            - loss: Loss type ('mse', 'weighted_mse', 'combined', 'huber')
            - weighted_ratio: Blend ratio for combined loss (default: 0.5)

    Returns:
        Loss function module
    """
    training_cfg = config.get('training', {})
    loss_type = training_cfg.get('loss', 'combined')
    target_weights = training_cfg.get('target_weights', None)

    if loss_type == 'mse':
        loss_fn = MaskedMSELoss()
    elif loss_type == 'masked_huber':
        delta = training_cfg.get('huber_delta', 1.0)
        loss_fn = MaskedHuberLoss(delta=delta)
    elif loss_type == 'weighted_mse':
        loss_fn = WeightedMSELoss(target_weights=target_weights)
    elif loss_type == 'combined':
        ratio = training_cfg.get('weighted_ratio', 0.5)
        loss_fn = CombinedLoss(weighted_ratio=ratio, target_weights=target_weights)
    elif loss_type == 'huber':
        delta = training_cfg.get('huber_delta', 1.0)
        loss_fn = HuberWeightedLoss(delta=delta)
    elif loss_type == 'weighted_pearson':
        eps = float(training_cfg.get('pearson_eps', 1e-6))
        loss_fn = WeightedPearsonLoss(eps=eps)
    elif loss_type == 'pearson_combined':
        alpha = float(training_cfg.get('pearson_alpha', 0.6))
        ratio = float(training_cfg.get('weighted_ratio', 0.62))
        eps = float(training_cfg.get('pearson_eps', 1e-6))
        loss_fn = PearsonCombinedLoss(
            alpha=alpha, weighted_ratio=ratio, eps=eps,
            target_weights=target_weights,
        )
    elif loss_type == 'pearson_primary':
        alpha = float(training_cfg.get('pearson_primary_alpha', 0.80))
        warmup_alpha = float(training_cfg.get('warmup_alpha', 0.4))
        warmup_epochs = int(training_cfg.get('warmup_epochs', 3))
        huber_delta = float(training_cfg.get('huber_delta', 1.0))
        target_ratio = float(training_cfg.get('target_ratio', 0.62))
        eps = float(training_cfg.get('pearson_eps', 1e-6))
        loss_fn = PearsonPrimaryLoss(
            alpha=alpha, warmup_alpha=warmup_alpha,
            warmup_epochs=warmup_epochs, huber_delta=huber_delta,
            target_ratio=target_ratio, eps=eps,
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Wrap with CoRe (Coherency Regularization) if configured
    core_lambda = float(training_cfg.get('core_lambda', 0.0))
    if core_lambda > 0:
        warmup = int(training_cfg.get('core_warmup_epochs', 5))
        loss_fn = CoReWrapper(loss_fn, core_lambda=core_lambda,
                              warmup_epochs=warmup)

    return loss_fn
