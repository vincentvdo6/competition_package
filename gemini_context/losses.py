"""Loss function factory for training.

Available loss types (all tested, only 'mse' works well with vanilla GRU):
- mse: Plain MSE (WINNING - used in current best)
- masked_huber: Huber loss (CATASTROPHIC on vanilla: -0.025 val delta)
- weighted_mse: MSE weighted by |target| (part of complex pipeline that HURTS)
- combined: Blend of MSE + weighted MSE (part of complex pipeline that HURTS)
- huber: Huber weighted by |target| (NEGATIVE)
- weighted_pearson: Differentiable weighted Pearson correlation (training instability)
- pearson_combined: alpha*Combined + (1-alpha)*(1-pearson) (genuine diversity but WRONG direction on LB)
- pearson_primary: Ramps from MSE to Pearson over warmup (UNSTABLE, early stopping epoch 8-9)
"""

import torch
import torch.nn as nn
from typing import Optional


class MaskedMSELoss(nn.Module):
    """Simple MSE loss that supports the mask argument from the trainer."""
    def forward(self, predictions, targets, mask=None, **kwargs):
        if mask is not None:
            mask_3d = mask.unsqueeze(-1).expand_as(predictions)
            return torch.nn.functional.mse_loss(predictions[mask_3d], targets[mask_3d])
        return torch.nn.functional.mse_loss(predictions, targets)


class MaskedHuberLoss(nn.Module):
    """Plain Huber loss with mask support. No amplitude weighting."""
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, predictions, targets, mask=None, **kwargs):
        if mask is not None:
            mask_3d = mask.unsqueeze(-1).expand_as(predictions)
            return torch.nn.functional.huber_loss(
                predictions[mask_3d], targets[mask_3d], reduction='mean', delta=self.delta)
        return torch.nn.functional.huber_loss(
            predictions, targets, reduction='mean', delta=self.delta)


def get_loss_function(config):
    """Factory function for loss functions."""
    training_cfg = config.get('training', {})
    loss_type = training_cfg.get('loss', 'combined')

    if loss_type == 'mse':
        return MaskedMSELoss()
    elif loss_type == 'masked_huber':
        return MaskedHuberLoss(delta=training_cfg.get('huber_delta', 1.0))
    # ... other loss types available but all KILLED for vanilla GRU:
    # weighted_mse, combined, huber, weighted_pearson, pearson_combined, pearson_primary
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
