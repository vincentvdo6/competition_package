"""GRU + causal self-attention model for sequence prediction.

The model keeps a GRU backbone for stable online inference and applies
self-attention on top of GRU states:
- During training (`forward`): causal attention over the full sequence.
- During online inference (`forward_step`): query = current GRU state,
  key/value = rolling buffer of past GRU states (ring-buffer behavior).
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from .base import BaseModel


class GRUAttentionModel(BaseModel):
    """GRU backbone with attention head for architecture diversity."""

    def __init__(self, config: dict):
        super().__init__(config)

        model_cfg = config.get("model", {})
        self.input_size = int(model_cfg.get("input_size", 42))
        self.hidden_size = int(model_cfg.get("hidden_size", 144))
        self.num_layers = int(model_cfg.get("num_layers", 2))
        self.dropout = float(model_cfg.get("dropout", 0.22))
        self.output_size = int(model_cfg.get("output_size", 2))
        self.attention_heads = int(model_cfg.get("attention_heads", 4))
        self.attention_dropout = float(model_cfg.get("attention_dropout", 0.1))
        self.attention_window = int(model_cfg.get("attention_window", 128))

        self.input_proj = nn.Linear(self.input_size, self.hidden_size)
        self.input_norm = nn.LayerNorm(self.hidden_size)
        self.input_dropout = nn.Dropout(self.dropout)

        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.attention_heads,
            dropout=self.attention_dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(self.hidden_size)

        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, self.output_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Any]:
        """Forward for batched training using causal full-sequence attention."""
        batch_size, seq_len, _ = x.shape

        if hidden is None:
            gru_hidden = self.init_hidden(batch_size)[0].to(x.device)
        elif isinstance(hidden, tuple):
            gru_hidden = hidden[0]
        else:
            gru_hidden = hidden

        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)

        gru_out, new_gru_hidden = self.gru(x, gru_hidden)

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=gru_out.device),
            diagonal=1,
        )
        attn_out, _ = self.attn(
            gru_out,
            gru_out,
            gru_out,
            attn_mask=causal_mask,
            need_weights=False,
        )

        fused = self.attn_norm(gru_out + attn_out)
        preds = self.output_proj(fused)
        return preds, (new_gru_hidden, None)

    def init_hidden(self, batch_size: int) -> Any:
        """Return model hidden state tuple for compatibility with forward_step."""
        gru_hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return (gru_hidden, None)

    def forward_step(
        self,
        x: torch.Tensor,
        hidden: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Any]:
        """Online single-step forward with rolling attention context."""
        if hidden is None:
            gru_hidden = None
            attn_buffer = None
        else:
            gru_hidden, attn_buffer = hidden

        x = x.unsqueeze(1)
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)

        gru_out, new_gru_hidden = self.gru(x, gru_hidden)
        cur = gru_out.squeeze(1)

        if attn_buffer is None:
            context = cur.unsqueeze(1)
        else:
            context = torch.cat([attn_buffer, cur.unsqueeze(1)], dim=1)
            if context.size(1) > self.attention_window:
                context = context[:, -self.attention_window :, :]

        q = cur.unsqueeze(1)
        attn_out, _ = self.attn(q, context, context, need_weights=False)
        fused = self.attn_norm(cur + attn_out.squeeze(1))
        pred = self.output_proj(fused)

        new_hidden = (new_gru_hidden, context.detach())
        return pred, new_hidden
