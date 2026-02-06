"""Loss function factory for training."""

import torch.nn as nn
from src.evaluation.metrics import WeightedMSELoss, CombinedLoss, HuberWeightedLoss


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
        return nn.MSELoss()
    elif loss_type == 'weighted_mse':
        return WeightedMSELoss(target_weights=target_weights)
    elif loss_type == 'combined':
        ratio = training_cfg.get('weighted_ratio', 0.5)
        return CombinedLoss(weighted_ratio=ratio, target_weights=target_weights)
    elif loss_type == 'huber':
        delta = training_cfg.get('huber_delta', 1.0)
        return HuberWeightedLoss(delta=delta)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
