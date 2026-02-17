"""GRU baseline model for LOB sequence prediction."""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Union


class GRUBaseline(nn.Module):
    """GRU model for sequence-to-sequence prediction.

    Supports multiple modes:
    - vanilla=True: Raw features -> GRU -> Linear (WINNING config)
    - vanilla=False: Features -> Linear+LayerNorm -> GRU -> MLP (HURTS generalization)
    - rnn_type='lstm': LSTM instead of GRU (KILLED: -0.011 vs GRU)
    - cvml=True: Learnable feature mixing MLP before GRU (KILLED)
    - feature_gate=True: SE-Net per-timestep feature gating (KILLED)
    """

    def __init__(self, config: dict):
        super().__init__()

        model_cfg = config.get('model', {})
        self.input_size = model_cfg.get('input_size', 32)
        self.hidden_size = model_cfg.get('hidden_size', 128)
        self.num_layers = model_cfg.get('num_layers', 2)
        self.dropout = model_cfg.get('dropout', 0.2)
        self.output_size = model_cfg.get('output_size', 2)
        self.vanilla = model_cfg.get('vanilla', False)
        self.rnn_type = model_cfg.get('rnn_type', 'gru')

        # CVML block: learnable nonlinear feature mixing with residual (KILLED)
        self.use_cvml = model_cfg.get('cvml', False)
        if self.use_cvml:
            self.cvml = nn.Sequential(
                nn.Linear(self.input_size, self.input_size * 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.input_size * 2, self.input_size),
            )
            nn.init.zeros_(self.cvml[3].weight)
            nn.init.zeros_(self.cvml[3].bias)

        # SE-Net feature gate (KILLED)
        self.use_feature_gate = model_cfg.get('feature_gate', False)
        if self.use_feature_gate:
            bottleneck = max(self.input_size // 4, 4)
            self.feat_gate = nn.Sequential(
                nn.Linear(self.input_size, bottleneck),
                nn.ReLU(),
                nn.Linear(bottleneck, self.input_size),
                nn.Sigmoid(),
            )

        # Input projection (skipped in vanilla mode)
        if not self.vanilla:
            self.input_proj = nn.Linear(self.input_size, self.hidden_size)
            self.input_norm = nn.LayerNorm(self.hidden_size)
            self.input_dropout = nn.Dropout(self.dropout)
            gru_input_size = self.hidden_size
        else:
            gru_input_size = self.input_size

        # RNN layers
        rnn_dropout = self.dropout if self.num_layers > 1 else 0.0
        if self.rnn_type == 'lstm':
            self.lstm = nn.LSTM(
                input_size=gru_input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=rnn_dropout,
                bidirectional=False
            )
        else:
            self.gru = nn.GRU(
                input_size=gru_input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=rnn_dropout,
                bidirectional=False
            )

        # Output projection
        output_type = model_cfg.get('output_type', 'mlp')
        if output_type == 'linear':
            self.output_proj = nn.Linear(self.hidden_size, self.output_size)
        else:
            self.output_proj = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size // 2, self.output_size)
            )

    def forward(self, x, hidden=None):
        """Forward pass for full sequences.
        x: (batch, seq_len, input_size) -> predictions: (batch, seq_len, 2), new_hidden
        """
        batch_size = x.shape[0]

        if hidden is None:
            hidden = self.init_hidden(batch_size)
            if self.rnn_type == 'lstm':
                hidden = (hidden[0].to(x.device), hidden[1].to(x.device))
            else:
                hidden = hidden.to(x.device)

        if self.use_cvml:
            x = x + self.cvml(x)
        if self.use_feature_gate:
            x = x * self.feat_gate(x)
        if not self.vanilla:
            x = self.input_proj(x)
            x = self.input_norm(x)
            x = self.input_dropout(x)

        if self.rnn_type == 'lstm':
            rnn_out, new_hidden = self.lstm(x, hidden)
        else:
            rnn_out, new_hidden = self.gru(x, hidden)

        predictions = self.output_proj(rnn_out)
        return predictions, new_hidden

    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        if self.rnn_type == 'lstm':
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            return (h0, c0)
        return h0

    def forward_step(self, x, hidden=None):
        """Single timestep for online inference. x: (batch, 32) -> pred: (batch, 2)"""
        x = x.unsqueeze(1)
        predictions, new_hidden = self.forward(x, hidden)
        return predictions.squeeze(1), new_hidden
