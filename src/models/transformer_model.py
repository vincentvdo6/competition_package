"""Small Causal Transformer model for LOB sequence prediction."""

import math
import torch
import torch.nn as nn
from typing import Any, List, Optional, Tuple
from .base import BaseModel


class CausalTransformerBlock(nn.Module):
    """Pre-norm causal attention + FFN block with KV-cache support."""

    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """Full-sequence forward with causal mask.

        Args:
            x: (batch, seq_len, d_model)
            attn_mask: (seq_len, seq_len) additive mask
        """
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x

    def forward_step(
        self, x: torch.Tensor, context_buf: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-step forward with growing context buffer.

        Args:
            x: (batch, d_model) — current step embedding
            context_buf: (batch, <=window, d_model) or None
        Returns:
            output: (batch, d_model)
            new_context_buf: (batch, <=window+1, d_model)
        """
        h = self.norm1(x).unsqueeze(1)  # (batch, 1, d_model)

        if context_buf is None:
            context = h
        else:
            context = torch.cat([context_buf, h], dim=1)

        q = h  # (batch, 1, d_model)
        attn_out, _ = self.attn(q, context, context, need_weights=False)

        x = x + attn_out.squeeze(1)  # (batch, d_model)
        h2 = self.norm2(x)
        x = x + self.ffn(h2)
        return x, context


class CausalTransformerModel(BaseModel):
    """Small causal transformer for LOB prediction.

    Architecture:
    - Input projection: Linear(input_size -> d_model) + LayerNorm
    - Sinusoidal positional encoding (cyclic, window-limited)
    - N causal transformer blocks (pre-norm, multi-head attention + FFN)
    - Output: LayerNorm + Linear(d_model -> output_size)
    ~70K parameters with default config (d=64, 2 blocks, 4 heads, ff=128).
    """

    def __init__(self, config: dict):
        super().__init__(config)
        model_cfg = config.get("model", {})
        self.input_size = int(model_cfg.get("input_size", 42))
        self.d_model = int(model_cfg.get("d_model", 64))
        self.nhead = int(model_cfg.get("nhead", 4))
        self.num_blocks = int(model_cfg.get("num_blocks", 2))
        self.dim_feedforward = int(model_cfg.get("dim_feedforward", 128))
        self.dropout_rate = float(model_cfg.get("dropout", 0.15))
        self.window_size = int(model_cfg.get("window_size", 128))
        self.output_size = int(model_cfg.get("output_size", 2))

        # Input projection
        self.input_proj = nn.Linear(self.input_size, self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.input_dropout = nn.Dropout(self.dropout_rate)

        # Sinusoidal positional encoding (not learnable)
        self.register_buffer(
            "pos_encoding",
            self._build_sinusoidal_pe(self.window_size, self.d_model),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                CausalTransformerBlock(
                    self.d_model, self.nhead, self.dim_feedforward, self.dropout_rate
                )
                for _ in range(self.num_blocks)
            ]
        )

        # Output head
        self.output_norm = nn.LayerNorm(self.d_model)
        self.output_head = nn.Linear(self.d_model, self.output_size)

    @staticmethod
    def _build_sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
        """Build sinusoidal positional encoding table."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # (max_len, d_model)

    @staticmethod
    def _build_windowed_causal_mask(
        seq_len: int, window_size: int, device: torch.device
    ) -> torch.Tensor:
        """Build windowed causal attention mask.

        Position i can attend to positions max(0, i-window_size+1) .. i.
        Returns additive mask: 0 = attend, -inf = block.
        """
        # Causal: block future positions
        causal = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()
        # Window: block positions more than window_size steps in the past
        past = torch.tril(
            torch.ones(seq_len, seq_len, device=device), diagonal=-window_size
        ).bool()
        mask = torch.zeros(seq_len, seq_len, device=device)
        mask.masked_fill_(causal | past, float("-inf"))
        return mask

    def forward(
        self, x: torch.Tensor, hidden: Optional[Any] = None
    ) -> Tuple[torch.Tensor, Any]:
        """Full-sequence forward with windowed causal attention.

        Args:
            x: (batch, seq_len, input_size)
            hidden: Ignored (transformer is stateless in training mode)

        Returns:
            predictions: (batch, seq_len, output_size)
            hidden: None
        """
        batch_size, seq_len, _ = x.shape

        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)

        # Add sinusoidal positional encoding (cyclic within window)
        positions = torch.arange(seq_len, device=x.device) % self.window_size
        x = x + self.pos_encoding[positions]

        # Build windowed causal mask
        attn_mask = self._build_windowed_causal_mask(
            seq_len, self.window_size, x.device
        )

        for block in self.blocks:
            x = block(x, attn_mask)

        x = self.output_norm(x)
        preds = self.output_head(x)
        return preds, None

    def init_hidden(self, batch_size: int) -> List[Optional[torch.Tensor]]:
        """Initialize empty context buffers (one per block)."""
        return [None] * self.num_blocks

    def forward_step(
        self, x: torch.Tensor, hidden: Optional[List[Optional[torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]]]:
        """Single-step online inference with growing context buffers.

        Args:
            x: (batch, input_size) — single timestep features
            hidden: List of per-layer context buffers, or None

        Returns:
            prediction: (batch, output_size)
            new_hidden: Updated list of per-layer context buffers
        """
        if hidden is None:
            hidden = self.init_hidden(x.shape[0])
            self._step_count = 0

        if not hasattr(self, "_step_count"):
            self._step_count = 0

        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x)
        # No dropout during inference (model.eval() handles this)

        # Positional encoding for current step
        pos_idx = self._step_count % self.window_size
        x = x + self.pos_encoding[pos_idx]

        # Process through blocks with growing context buffers
        new_hidden = []
        for i, block in enumerate(self.blocks):
            buf = hidden[i]
            # Truncate to window_size-1 before adding current step,
            # so context has at most window_size entries (matches training mask)
            if buf is not None and buf.size(1) >= self.window_size:
                buf = buf[:, -(self.window_size - 1):, :]
            x, new_buf = block.forward_step(x, buf)
            new_hidden.append(new_buf)

        # Output
        x = self.output_norm(x)
        pred = self.output_head(x)

        self._step_count += 1
        return pred, new_hidden
