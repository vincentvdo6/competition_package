"""LSTM model for LOB sequence prediction."""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from .base import BaseModel


class LSTMModel(BaseModel):
    """LSTM model for sequence-to-sequence prediction.

    Architecture mirrors GRUBaseline but uses LSTM cells, which add a
    separate cell state for long-range memory — useful for the strong
    temporal autocorrelation observed in the data (t1 lag-1 = 0.976).

    Hidden state is a tuple (h, c) rather than a single tensor.
    """

    def __init__(self, config: dict):
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

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            bidirectional=False,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, self.output_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for full sequences.

        Args:
            x: Input features of shape (batch, seq_len, input_size)
            hidden: Tuple (h, c) each of shape (num_layers, batch, hidden_size)
                   or None to initialise fresh

        Returns:
            predictions: Shape (batch, seq_len, 2)
            new_hidden: Tuple (h, c)
        """
        batch_size = x.shape[0]

        if hidden is None:
            hidden = self.init_hidden(batch_size)
            hidden = (hidden[0].to(x.device), hidden[1].to(x.device))

        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)

        # LSTM forward
        lstm_out, new_hidden = self.lstm(x, hidden)

        # Output projection
        predictions = self.output_proj(lstm_out)

        return predictions, new_hidden

    def init_hidden(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialise (h_0, c_0) with zeros."""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return (h, c)

    def forward_step(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Single-timestep forward for online inference.

        Args:
            x: Input of shape (batch, input_size)
            hidden: Tuple (h, c) or None

        Returns:
            prediction: Shape (batch, 2)
            new_hidden: Tuple (h, c)
        """
        x = x.unsqueeze(1)  # (batch, 1, input_size)
        predictions, new_hidden = self.forward(x, hidden)
        return predictions.squeeze(1), new_hidden
