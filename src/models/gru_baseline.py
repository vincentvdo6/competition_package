"""GRU baseline model for LOB sequence prediction."""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from .base import BaseModel


class GRUBaseline(BaseModel):
    """Simple GRU model for sequence-to-sequence prediction.
    
    Architecture:
    - Input projection layer with LayerNorm
    - Multi-layer GRU with dropout
    - Output projection to 2 targets (t0, t1)
    
    Supports both batch training and step-by-step online inference.
    """
    
    def __init__(self, config: dict):
        """Initialize GRU model.
        
        Args:
            config: Configuration dictionary with 'model' section containing:
                - input_size: Input feature dimension (default: 32)
                - hidden_size: GRU hidden size (default: 128)
                - num_layers: Number of GRU layers (default: 2)
                - dropout: Dropout rate (default: 0.2)
                - output_size: Number of targets (default: 2)
        """
        super().__init__(config)
        
        model_cfg = config.get('model', {})
        self.input_size = model_cfg.get('input_size', 32)
        self.hidden_size = model_cfg.get('hidden_size', 128)
        self.num_layers = model_cfg.get('num_layers', 2)
        self.dropout = model_cfg.get('dropout', 0.2)
        self.output_size = model_cfg.get('output_size', 2)
        
        # Input projection
        self.input_proj = nn.Linear(self.input_size, self.hidden_size)
        self.input_norm = nn.LayerNorm(self.hidden_size)
        self.input_dropout = nn.Dropout(self.dropout)
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            bidirectional=False
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, self.output_size)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for full sequences.
        
        Args:
            x: Input features of shape (batch, seq_len, 32)
            hidden: Hidden state of shape (num_layers, batch, hidden_size)
                   or None to initialize fresh
        
        Returns:
            predictions: Shape (batch, seq_len, 2)
            new_hidden: Shape (num_layers, batch, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size)
            hidden = hidden.to(x.device)
        
        # Input projection: (batch, seq_len, 32) -> (batch, seq_len, hidden_size)
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)
        
        # GRU forward
        gru_out, new_hidden = self.gru(x, hidden)
        
        # Output projection: (batch, seq_len, hidden_size) -> (batch, seq_len, 2)
        predictions = self.output_proj(gru_out)
        
        return predictions, new_hidden
    
    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """Initialize hidden state with zeros.
        
        Args:
            batch_size: Batch size
        
        Returns:
            Hidden state of shape (num_layers, batch_size, hidden_size)
        """
        return torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size
        )
    
    def forward_step(
        self, 
        x: torch.Tensor, 
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for single timestep (online inference).
        
        Args:
            x: Input features of shape (batch, 32) for single step
            hidden: Hidden state of shape (num_layers, batch, hidden_size)
                   or None to initialize fresh
        
        Returns:
            prediction: Shape (batch, 2)
            new_hidden: Shape (num_layers, batch, hidden_size)
        """
        # Add sequence dimension: (batch, 32) -> (batch, 1, 32)
        x = x.unsqueeze(1)
        
        # Use full forward pass
        predictions, new_hidden = self.forward(x, hidden)
        
        # Remove sequence dimension: (batch, 1, 2) -> (batch, 2)
        return predictions.squeeze(1), new_hidden
