"""GRU baseline model for LOB sequence prediction."""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Union
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
        self.vanilla = model_cfg.get('vanilla', False)

        # CVML block: learnable nonlinear feature mixing with residual
        self.use_cvml = model_cfg.get('cvml', False)
        if self.use_cvml:
            self.cvml = nn.Sequential(
                nn.Linear(self.input_size, self.input_size * 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.input_size * 2, self.input_size),
            )
            # Zero-init second linear so block starts as identity
            nn.init.zeros_(self.cvml[3].weight)
            nn.init.zeros_(self.cvml[3].bias)

        # SE-Net feature gate: per-timestep feature weighting
        self.use_feature_gate = model_cfg.get('feature_gate', False)
        if self.use_feature_gate:
            bottleneck = max(self.input_size // 4, 4)
            self.feat_gate = nn.Sequential(
                nn.Linear(self.input_size, bottleneck),
                nn.ReLU(),
                nn.Linear(bottleneck, self.input_size),
                nn.Sigmoid(),
            )

        # Input projection (skipped in vanilla mode — raw features go directly to GRU)
        if not self.vanilla:
            self.input_proj = nn.Linear(self.input_size, self.hidden_size)
            self.input_norm = nn.LayerNorm(self.hidden_size)
            self.input_dropout = nn.Dropout(self.dropout)
            gru_input_size = self.hidden_size
        else:
            gru_input_size = self.input_size

        # GRU layers
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
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

        # Optional auxiliary heads (train-only, stripped at inference)
        self.has_aux_heads = model_cfg.get('aux_heads', False)
        if self.has_aux_heads:
            self.aux_delta = nn.Linear(self.hidden_size, 1)
            self.aux_sign_t0 = nn.Linear(self.hidden_size, 1)
            self.aux_sign_t1 = nn.Linear(self.hidden_size, 1)

        # Chrono initialization for GRU gates (biases update gate toward persistence)
        if model_cfg.get('chrono_init', False):
            max_period = model_cfg.get('chrono_max_period', 100)
            self._apply_chrono_init(max_period)
    
    def _apply_chrono_init(self, max_period: int = 10):
        """Chrono initialization: bias GRU update gate toward persistence.

        Sets update gate (z) bias_hh only to log(U(1, max_period)), making z_t
        default to moderately high values so h_t leans toward h_{t-1}.
        Only bias_hh is set (not bias_ih) to avoid double-bias saturation.
        PyTorch GRU gate order: [reset, update, new], each of size hidden_size.
        """
        for layer_idx in range(self.num_layers):
            h = self.hidden_size
            with torch.no_grad():
                # Only set bias_hh (recurrent bias), leave bias_ih at default
                bias_hh = getattr(self.gru, f'bias_hh_l{layer_idx}')
                # log(U(1, T)) gives values in [0, log(T)] ≈ [0, 2.3] for T=10
                bias_hh[h:2*h].uniform_(1, max_period).log_()
                # Zero the corresponding bias_ih update gate to avoid double-bias
                bias_ih = getattr(self.gru, f'bias_ih_l{layer_idx}')
                bias_ih[h:2*h].zero_()

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor],
               Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass for full sequences.

        Args:
            x: Input features of shape (batch, seq_len, input_size)
            hidden: Hidden state or None to initialize fresh
            return_aux: If True and model has aux heads, return aux predictions

        Returns:
            predictions: Shape (batch, seq_len, 2)
            new_hidden: Shape (num_layers, batch, hidden_size)
            aux (optional): Dict with 'delta', 'sign_t0', 'sign_t1' predictions
        """
        batch_size, seq_len, _ = x.shape

        if hidden is None:
            hidden = self.init_hidden(batch_size)
            hidden = hidden.to(x.device)

        # CVML: learnable feature mixing (residual)
        if self.use_cvml:
            x = x + self.cvml(x)

        # SE-Net: per-timestep feature gating
        if self.use_feature_gate:
            x = x * self.feat_gate(x)

        if not self.vanilla:
            x = self.input_proj(x)
            x = self.input_norm(x)
            x = self.input_dropout(x)

        gru_out, new_hidden = self.gru(x, hidden)

        predictions = self.output_proj(gru_out)

        if return_aux and self.has_aux_heads:
            aux = {
                'delta': self.aux_delta(gru_out),
                'sign_t0': self.aux_sign_t0(gru_out),
                'sign_t1': self.aux_sign_t1(gru_out),
            }
            return predictions, new_hidden, aux

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
