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
