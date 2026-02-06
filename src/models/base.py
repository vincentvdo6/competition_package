"""Base model interface for all sequence models."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional


class BaseModel(nn.Module, ABC):
    """Base class for all sequence models in this competition.
    
    All models must implement:
    - forward(x, hidden) -> (predictions, new_hidden)
    - init_hidden(batch_size) -> initial hidden state
    - forward_step(x, hidden) -> (prediction, new_hidden) for online inference
    
    This interface ensures models can be used in both:
    1. Batch training mode (full sequences)
    2. Online inference mode (step-by-step)
    """
    
    def __init__(self, config: dict):
        """Initialize model.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        super().__init__()
        self.config = config
    
    @abstractmethod
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Optional[Any] = None
    ) -> Tuple[torch.Tensor, Any]:
        """Forward pass for full sequences.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size=32)
            hidden: Model-specific hidden state (None = fresh sequence)
        
        Returns:
            predictions: Tensor of shape (batch, seq_len, 2) for t0, t1
            new_hidden: Updated hidden state
        """
        pass
    
    @abstractmethod
    def init_hidden(self, batch_size: int) -> Any:
        """Initialize fresh hidden state for new sequences.
        
        Args:
            batch_size: Batch size
        
        Returns:
            Initial hidden state (type depends on model architecture)
        """
        pass
    
    @abstractmethod
    def forward_step(
        self, 
        x: torch.Tensor, 
        hidden: Optional[Any] = None
    ) -> Tuple[torch.Tensor, Any]:
        """Forward pass for single timestep (online inference).
        
        Args:
            x: Input tensor of shape (batch, input_size=32) for single step
            hidden: Model-specific hidden state (None = fresh sequence)
        
        Returns:
            prediction: Tensor of shape (batch, 2) for t0, t1
            new_hidden: Updated hidden state
        """
        pass
    
    def get_device(self) -> torch.device:
        """Get the device this model is on."""
        return next(self.parameters()).device
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
