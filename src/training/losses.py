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
)


class MaskedMSELoss(nn.Module):
    """Simple MSE loss that supports the mask argument from the trainer."""

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if mask is not None:
            mask_3d = mask.unsqueeze(-1).expand_as(predictions)
            return torch.nn.functional.mse_loss(predictions[mask_3d], targets[mask_3d])
        return torch.nn.functional.mse_loss(predictions, targets)


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
        return MaskedMSELoss()
    elif loss_type == 'weighted_mse':
        return WeightedMSELoss(target_weights=target_weights)
    elif loss_type == 'combined':
        ratio = training_cfg.get('weighted_ratio', 0.5)
        return CombinedLoss(weighted_ratio=ratio, target_weights=target_weights)
    elif loss_type == 'huber':
        delta = training_cfg.get('huber_delta', 1.0)
        return HuberWeightedLoss(delta=delta)
    elif loss_type == 'weighted_pearson':
        eps = float(training_cfg.get('pearson_eps', 1e-6))
        return WeightedPearsonLoss(eps=eps)
    elif loss_type == 'pearson_combined':
        alpha = float(training_cfg.get('pearson_alpha', 0.6))
        ratio = float(training_cfg.get('weighted_ratio', 0.62))
        eps = float(training_cfg.get('pearson_eps', 1e-6))
        return PearsonCombinedLoss(
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
        return PearsonPrimaryLoss(
            alpha=alpha, warmup_alpha=warmup_alpha,
            warmup_epochs=warmup_epochs, huber_delta=huber_delta,
            target_ratio=target_ratio, eps=eps,
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
