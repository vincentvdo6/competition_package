"""Preprocessing utilities for LOB sequence data."""

import numpy as np
from typing import Optional


class DerivedFeatureBuilder:
    """Compute derived LOB features from raw 32-feature data.

    Adds 10 derived features:
    - spread_0..spread_5: ask_price_i - bid_price_i (6)
    - trade_intensity: sum of trade volumes dv0..dv3 (1)
    - bid_pressure: sum of bid volumes v0..v5 (1)
    - ask_pressure: sum of ask volumes v6..v11 (1)
    - pressure_imbalance: (bid - ask) / (bid + ask + eps) (1)
    """

    N_DERIVED = 10
    DERIVED_COLS = [
        'spread_0', 'spread_1', 'spread_2', 'spread_3', 'spread_4', 'spread_5',
        'trade_intensity', 'bid_pressure', 'ask_pressure', 'pressure_imbalance',
    ]

    @staticmethod
    def compute(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Compute derived features from raw LOB features.

        Args:
            features: Raw features of shape (..., 32)
            eps: Small constant for numerical stability

        Returns:
            Derived features of shape (..., 10)
        """
        # Raw layout: p0-p11 (0:12), v0-v11 (12:24), dp0-dp3 (24:28), dv0-dv3 (28:32)
        # Bid prices: p0-p5 (0:6), Ask prices: p6-p11 (6:12)
        # Bid volumes: v0-v5 (12:18), Ask volumes: v6-v11 (18:24)

        # Spreads: ask - bid at each of 6 levels
        spreads = features[..., 6:12] - features[..., 0:6]

        # Trade intensity: sum of trade volumes
        trade_intensity = features[..., 28:32].sum(axis=-1, keepdims=True)

        # Bid/ask pressure
        bid_pressure = features[..., 12:18].sum(axis=-1, keepdims=True)
        ask_pressure = features[..., 18:24].sum(axis=-1, keepdims=True)

        # Imbalance
        pressure_imbalance = (
            (bid_pressure - ask_pressure)
            / (bid_pressure + ask_pressure + eps)
        )

        return np.concatenate(
            [spreads, trade_intensity, bid_pressure, ask_pressure, pressure_imbalance],
            axis=-1,
        ).astype(np.float32)

    @staticmethod
    def compute_single(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Compute derived features for a single 1-D feature vector (online inference).

        Args:
            features: Raw features of shape (32,)
            eps: Small constant

        Returns:
            Augmented features of shape (42,) = raw (32) + derived (10)
        """
        derived = DerivedFeatureBuilder.compute(features, eps)
        return np.concatenate([features, derived], axis=-1).astype(np.float32)


class TemporalDerivedFeatureBuilder:
    """Compute temporal derived features that depend on previous timesteps.

    Requires the 42-feature input (32 raw + 10 derived from DerivedFeatureBuilder).
    Adds 3 temporal features:
    - spread0_roc1: spread_0[t] - spread_0[t-1]  (rate of change, lag 1)
    - spread0_roc5: spread_0[t] - spread_0[t-5]  (rate of change, lag 5)
    - trade_intensity_roll_mean_5: rolling mean of trade_intensity over 5 steps

    Derived feature layout (indices 32-41 in the 42-feature vector):
      32: spread_0, 33-37: spread_1..5, 38: trade_intensity,
      39: bid_pressure, 40: ask_pressure, 41: pressure_imbalance
    """

    N_TEMPORAL = 3
    TEMPORAL_COLS = ['spread0_roc1', 'spread0_roc5', 'trade_intensity_roll_mean_5']

    @staticmethod
    def compute_batch(features: np.ndarray) -> np.ndarray:
        """Compute temporal features for a batch of sequences.

        Args:
            features: Shape (n_seqs, seq_len, 42) — raw + derived features

        Returns:
            Temporal features of shape (n_seqs, seq_len, 3)
        """
        spread_0 = features[..., 32]   # (n_seqs, seq_len)
        trade_int = features[..., 38]  # (n_seqs, seq_len)

        # Rate of change lag 1
        roc1 = np.zeros_like(spread_0)
        roc1[:, 1:] = spread_0[:, 1:] - spread_0[:, :-1]

        # Rate of change lag 5
        roc5 = np.zeros_like(spread_0)
        roc5[:, 5:] = spread_0[:, 5:] - spread_0[:, :-5]

        # Rolling mean of trade_intensity over window of 5
        roll_mean = np.zeros_like(trade_int)
        cumsum = np.cumsum(trade_int, axis=1)
        # Positions 0-3: expanding mean
        for i in range(min(4, features.shape[1])):
            roll_mean[:, i] = cumsum[:, i] / (i + 1)
        # Positions >= 4: full 5-step window
        if features.shape[1] > 4:
            shifted = np.zeros_like(cumsum)
            shifted[:, 5:] = cumsum[:, :-5]
            roll_mean[:, 4:] = (cumsum[:, 4:] - shifted[:, 4:]) / 5.0

        return np.stack([roc1, roc5, roll_mean], axis=-1).astype(np.float32)


class TemporalBuffer:
    """Maintains state for online (step-by-step) temporal feature computation.

    Used during competition inference where we see one DataPoint at a time.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.step = 0
        self.spread0_history = []
        self.trade_int_history = []

    def compute_step(self, features_42: np.ndarray) -> np.ndarray:
        """Compute temporal features for a single step, maintaining buffer.

        Args:
            features_42: Shape (42,) — raw + derived features for current step

        Returns:
            Shape (45,) — original 42 + 3 temporal features appended
        """
        spread_0 = float(features_42[32])
        trade_int = float(features_42[38])

        self.spread0_history.append(spread_0)
        self.trade_int_history.append(trade_int)

        # spread0_roc1
        roc1 = spread_0 - self.spread0_history[-2] if self.step >= 1 else 0.0

        # spread0_roc5
        roc5 = spread_0 - self.spread0_history[-6] if self.step >= 5 else 0.0

        # trade_intensity_roll_mean_5
        window = self.trade_int_history[-5:]
        roll_mean = sum(window) / len(window)

        self.step += 1

        temporal = np.array([roc1, roc5, roll_mean], dtype=np.float32)
        return np.concatenate([features_42, temporal]).astype(np.float32)


class Normalizer:
    """Z-score normalizer for LOB features.
    
    Computes mean and standard deviation from training data,
    then applies normalization to transform features to zero mean
    and unit variance.
    """
    
    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.eps = 1e-8
    
    def fit(self, X: np.ndarray) -> 'Normalizer':
        """Compute mean and std from training data.
        
        Args:
            X: Training features array of shape (n_samples, n_features)
               or (n_samples, seq_len, n_features)
        
        Returns:
            self for method chaining
        """
        # Flatten to (n_samples, n_features) if needed
        if len(X.shape) == 3:
            X = X.reshape(-1, X.shape[-1])
        
        self.mean = X.mean(axis=0).astype(np.float32)
        self.std = X.std(axis=0).astype(np.float32)
        
        # Prevent division by zero for constant features
        self.std[self.std < self.eps] = 1.0
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply normalization.
        
        Args:
            X: Features array of shape (n_samples, n_features)
               or (seq_len, n_features) or (batch, seq_len, n_features)
        
        Returns:
            Normalized features with same shape as input
        """
        if self.mean is None:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        
        return ((X - self.mean) / self.std).astype(np.float32)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse normalization.
        
        Args:
            X: Normalized features
        
        Returns:
            Original scale features
        """
        if self.mean is None:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        
        return (X * self.std + self.mean).astype(np.float32)
    
    def save(self, path: str) -> None:
        """Save normalizer parameters to file.
        
        Args:
            path: Path to save .npz file
        """
        if self.mean is None:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        
        np.savez(path, mean=self.mean, std=self.std)
    
    @classmethod
    def load(cls, path: str) -> 'Normalizer':
        """Load normalizer from file.
        
        Args:
            path: Path to .npz file
        
        Returns:
            Loaded Normalizer instance
        """
        data = np.load(path)
        norm = cls()
        norm.mean = data['mean']
        norm.std = data['std']
        return norm
