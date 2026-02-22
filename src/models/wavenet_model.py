"""WaveNet-lite: dilated causal CNN for LOB sequence prediction.

Architecture:
- Input projection: Linear(32, C)
- N residual gated blocks with exponentially increasing dilations
- Skip connections aggregated through 2-layer head -> (2,) output

Online inference uses ring buffers per block (no recurrence).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
from .base import BaseModel


class CausalConv1d(nn.Module):
    """Causal convolution: left-padded so output[t] depends only on input[<=t]."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq_len)
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class GatedResBlock(nn.Module):
    """Single gated residual block with skip connection."""

    def __init__(self, channels: int, skip_channels: int, kernel_size: int,
                 dilation: int, res_scale: float = 0.3):
        super().__init__()
        self.filter_conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.gate_conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.res_conv = nn.Conv1d(channels, channels, 1)
        self.skip_conv = nn.Conv1d(channels, skip_channels, 1)
        self.res_scale = res_scale
        self.dilation = dilation
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = torch.tanh(self.filter_conv(x))
        g = torch.sigmoid(self.gate_conv(x))
        z = f * g
        skip = self.skip_conv(z)
        res = self.res_conv(z)
        return x + self.res_scale * res, skip


class WaveNetModel(BaseModel):
    """WaveNet-lite for LOB prediction.

    Dilated causal CNN with gated residual blocks and skip connections.
    Receptive field = sum of dilations * (kernel_size - 1) + 1.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        model_cfg = config.get('model', {})

        self.input_size = model_cfg.get('input_size', 32)
        self.channels = model_cfg.get('channels', 64)
        self.skip_channels = model_cfg.get('skip_channels', 96)
        self.kernel_size = model_cfg.get('kernel_size', 2)
        self.output_size = model_cfg.get('output_size', 2)
        self.res_scale = model_cfg.get('res_scale', 0.3)

        dilations = model_cfg.get('dilations', [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
        self.n_blocks = len(dilations)

        # Input projection
        self.input_proj = nn.Conv1d(self.input_size, self.channels, 1)

        # Residual gated blocks
        self.blocks = nn.ModuleList([
            GatedResBlock(self.channels, self.skip_channels, self.kernel_size,
                          d, self.res_scale)
            for d in dilations
        ])

        # Output head (skip aggregation)
        self.output_head = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(self.skip_channels, self.channels, 1),
            nn.ReLU(),
            nn.Conv1d(self.channels, self.output_size, 1),
        )

        # Store dilations for ring buffer init
        self.dilations = dilations

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Any] = None,
        return_aux: bool = False,
    ) -> Tuple[torch.Tensor, Any]:
        """Forward pass for full sequences.

        Args:
            x: (batch, seq_len, input_size)
            hidden: ignored for CNN (stateless in batch mode)

        Returns:
            predictions: (batch, seq_len, 2)
            hidden: None (CNN has no recurrent state in batch mode)
        """
        # (batch, seq_len, C) -> (batch, C, seq_len)
        x = x.transpose(1, 2)
        x = self.input_proj(x)

        skip_sum = 0
        for block in self.blocks:
            x, skip = block(x)
            skip_sum = skip_sum + skip

        out = self.output_head(skip_sum)  # (batch, 2, seq_len)
        predictions = out.transpose(1, 2)  # (batch, seq_len, 2)
        return predictions, None

    def init_hidden(self, batch_size: int) -> Dict[str, List[torch.Tensor]]:
        """Initialize ring buffers for online inference.

        Each block needs a buffer of length dilation (for kernel_size=2).
        """
        buffers = []
        for d in self.dilations:
            # Buffer stores the last `d` activations for this block's input
            buf = torch.zeros(batch_size, self.channels, d)
            buffers.append(buf)
        # Also need input projection buffer (not needed, input proj is 1x1)
        return {'block_buffers': buffers}

    def forward_step(
        self,
        x: torch.Tensor,
        hidden: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Online inference: single step using ring buffers.

        Args:
            x: (batch, input_size) single timestep
            hidden: dict with 'block_buffers' list of ring buffers

        Returns:
            prediction: (batch, 2)
            new_hidden: updated ring buffers
        """
        batch_size = x.shape[0]
        device = x.device

        if hidden is None:
            hidden = self.init_hidden(batch_size)
            hidden = {k: [b.to(device) for b in v] for k, v in hidden.items()}

        buffers = hidden['block_buffers']

        # Input projection: (batch, 32) -> (batch, C)
        x_proj = self.input_proj(x.unsqueeze(-1)).squeeze(-1)  # (batch, C)

        current = x_proj  # (batch, C)
        skip_sum = torch.zeros(batch_size, self.skip_channels, device=device)

        new_buffers = []
        for i, block in enumerate(self.blocks):
            buf = buffers[i]  # (batch, C, dilation)
            d = self.dilations[i]

            # The delayed input is buf[:, :, 0] (oldest in ring buffer)
            delayed = buf[:, :, 0]  # (batch, C)

            # Manual matmul for kernel_size=2 dilated causal conv:
            # output = W[:,:,0] @ delayed + W[:,:,1] @ current + bias
            # (Conv1d with dilation>1 on width-2 input is incorrect)
            fw = block.filter_conv.conv.weight  # (C, C, 2)
            fb = block.filter_conv.conv.bias     # (C,)
            f = (delayed @ fw[:, :, 0].T) + (current @ fw[:, :, 1].T) + fb

            gw = block.gate_conv.conv.weight
            gb = block.gate_conv.conv.bias
            g = (delayed @ gw[:, :, 0].T) + (current @ gw[:, :, 1].T) + gb

            z = torch.tanh(f) * torch.sigmoid(g)

            # 1x1 convs work fine as matmul
            skip = block.skip_conv(z.unsqueeze(-1)).squeeze(-1)  # (batch, skip_C)
            res = block.res_conv(z.unsqueeze(-1)).squeeze(-1)  # (batch, C)

            skip_sum = skip_sum + skip

            # Update ring buffer with INPUT to this block (before residual)
            new_buf = torch.cat([buf[:, :, 1:], current.unsqueeze(-1)], dim=-1)
            new_buffers.append(new_buf)

            # Apply residual to get input for next block
            current = current + block.res_scale * res

        # Output head
        out = skip_sum.unsqueeze(-1)  # (batch, skip_C, 1)
        out = self.output_head(out).squeeze(-1)  # (batch, 2)

        return out, {'block_buffers': new_buffers}
