"""Temporal Convolutional Network (TCN) model for sequence prediction.

Lightweight alternative to GRU/LSTM using causal 1D convolutions with
dilated residual blocks. Designed for fast CPU inference with ring buffer
caching for online step-by-step prediction.

Architecture (Codex-designed):
- Input projection: Linear(input_size -> hidden_channels)
- N residual blocks with exponentially increasing dilation
- Output head: LayerNorm + Linear(hidden_channels -> 2)
- Receptive field: 1 + (kernel_size-1) * sum(dilations) steps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, List, Optional, Tuple

from .base import BaseModel


class TCNResidualBlock(nn.Module):
    """Single residual block: depthwise causal conv + pointwise conv + skip."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float = 0.15):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation  # causal left-padding
        self.buf_len = (kernel_size - 1) * dilation + 1  # ring buffer size

        # Depthwise separable convolution
        self.dw_conv = nn.Conv1d(
            channels, channels, kernel_size,
            dilation=dilation, groups=channels, bias=True,
        )
        self.activation = nn.SiLU()
        self.pw_conv = nn.Conv1d(channels, channels, 1, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full sequence forward with causal padding.

        Args:
            x: (batch, channels, seq_len)
        Returns:
            (batch, channels, seq_len)
        """
        residual = x
        x = F.pad(x, (self.padding, 0))  # left-pad for causality
        x = self.dw_conv(x)
        x = self.activation(x)
        x = self.pw_conv(x)
        x = self.dropout(x)
        return x + residual

    def forward_step(
        self, x: torch.Tensor, buf: torch.Tensor, ptr: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Single step with ring buffer.

        Args:
            x: (batch, channels) current input
            buf: (batch, channels, buf_len) ring buffer
            ptr: current write position
        Returns:
            output: (batch, channels)
            buf: updated buffer (modified in-place)
            new_ptr: next write position
        """
        # Write current input to buffer
        buf[:, :, ptr] = x

        # Read kernel_size taps at correct dilation spacing
        taps = []
        for i in range(self.kernel_size):
            tap_idx = (ptr - i * self.dilation) % self.buf_len
            taps.append(buf[:, :, tap_idx])

        # Stack in chronological order (oldest first) for conv
        tap_stack = torch.stack(list(reversed(taps)), dim=2)  # (batch, channels, kernel_size)

        # Depthwise conv: (batch, channels, k) -> (batch, channels, 1) -> (batch, channels)
        out = F.conv1d(
            tap_stack, self.dw_conv.weight, self.dw_conv.bias,
            groups=self.channels,
        ).squeeze(2)

        out = self.activation(out)

        # Pointwise conv: (batch, channels) -> (batch, channels)
        out = F.conv1d(
            out.unsqueeze(2), self.pw_conv.weight, self.pw_conv.bias,
        ).squeeze(2)

        # Residual connection
        out = out + x

        new_ptr = (ptr + 1) % self.buf_len
        return out, buf, new_ptr


class TCNModel(BaseModel):
    """Temporal Convolutional Network for LOB prediction.

    ~9K parameters with default config (vs ~150K for GRU).
    Receptive field of 127 steps with dilations [1,2,4,8,16,32].
    Online inference uses preallocated ring buffers for O(1) per-step cost.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        model_cfg = config.get('model', {})
        self.input_size = int(model_cfg.get('input_size', 42))
        self.hidden_channels = int(model_cfg.get('hidden_channels', 32))
        self.kernel_size = int(model_cfg.get('kernel_size', 3))
        self.dropout = float(model_cfg.get('dropout', 0.15))
        self.output_size = int(model_cfg.get('output_size', 2))
        dilations = model_cfg.get('dilations', [1, 2, 4, 8, 16, 32])

        # Input projection
        self.input_proj = nn.Linear(self.input_size, self.hidden_channels)

        # Residual blocks
        self.blocks = nn.ModuleList([
            TCNResidualBlock(self.hidden_channels, self.kernel_size, d, self.dropout)
            for d in dilations
        ])

        # Output head
        self.output_norm = nn.LayerNorm(self.hidden_channels)
        self.output_head = nn.Linear(self.hidden_channels, self.output_size)

    def forward(
        self, x: torch.Tensor, hidden: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Any]:
        """Full sequence forward for training.

        Args:
            x: (batch, seq_len, input_size)
            hidden: ignored (interface compatibility)
        Returns:
            predictions: (batch, seq_len, 2)
            hidden: None
        """
        # Input projection: (batch, seq_len, input_size) -> (batch, seq_len, channels)
        x = self.input_proj(x)

        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)

        for block in self.blocks:
            x = block(x)

        # Back to (batch, seq_len, channels)
        x = x.transpose(1, 2)
        x = self.output_norm(x)
        preds = self.output_head(x)

        return preds, None

    def init_hidden(self, batch_size: int) -> List[Tuple[torch.Tensor, int]]:
        """Initialize ring buffers for online inference.

        Returns:
            List of (buffer, pointer) tuples, one per residual block.
        """
        buffers = []
        for block in self.blocks:
            buf = torch.zeros(batch_size, self.hidden_channels, block.buf_len)
            buffers.append((buf, 0))
        return buffers

    def forward_step(
        self, x: torch.Tensor, hidden: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Any]:
        """Single step forward with ring buffer state.

        Args:
            x: (batch, input_size)
            hidden: list of (buffer, pointer) tuples or None
        Returns:
            prediction: (batch, 2)
            new_hidden: updated list of (buffer, pointer) tuples
        """
        batch_size = x.shape[0]

        if hidden is None:
            hidden = self.init_hidden(batch_size)
            hidden = [(buf.to(x.device), ptr) for buf, ptr in hidden]

        # Input projection: (batch, input_size) -> (batch, hidden_channels)
        x = self.input_proj(x)

        new_hidden = []
        for i, block in enumerate(self.blocks):
            buf, ptr = hidden[i]
            x, buf, new_ptr = block.forward_step(x, buf, ptr)
            new_hidden.append((buf, new_ptr))

        x = self.output_norm(x)
        pred = self.output_head(x)  # (batch, 2)

        return pred, new_hidden
