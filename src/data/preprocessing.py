"""Preprocessing utilities for LOB sequence data."""

from typing import Optional

import numpy as np


class DerivedFeatureBuilder:
    """Compute static derived LOB features from the raw 32-feature vector."""

    N_DERIVED = 10
    DERIVED_COLS = [
        "spread_0",
        "spread_1",
        "spread_2",
        "spread_3",
        "spread_4",
        "spread_5",
        "trade_intensity",
        "bid_pressure",
        "ask_pressure",
        "pressure_imbalance",
    ]

    @staticmethod
    def compute(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Compute derived features from raw features.

        Args:
            features: Raw features of shape (..., 32)
            eps: Numerical stability epsilon

        Returns:
            Array of shape (..., 10)
        """
        spreads = features[..., 6:12] - features[..., 0:6]
        trade_intensity = features[..., 28:32].sum(axis=-1, keepdims=True)
        bid_pressure = features[..., 12:18].sum(axis=-1, keepdims=True)
        ask_pressure = features[..., 18:24].sum(axis=-1, keepdims=True)
        pressure_imbalance = (bid_pressure - ask_pressure) / (bid_pressure + ask_pressure + eps)

        return np.concatenate(
            [spreads, trade_intensity, bid_pressure, ask_pressure, pressure_imbalance],
            axis=-1,
        ).astype(np.float32)

    @staticmethod
    def compute_single(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Compute derived features for a single raw state."""
        derived = DerivedFeatureBuilder.compute(features, eps)
        return np.concatenate([features, derived], axis=-1).astype(np.float32)


class TemporalDerivedFeatureBuilder:
    """Compute temporal derived features from base 42-dim (raw + derived) features."""

    N_TEMPORAL = 3
    TEMPORAL_COLS = ["spread0_roc1", "spread0_roc5", "trade_intensity_roll_mean_5"]

    @staticmethod
    def compute_batch(features: np.ndarray) -> np.ndarray:
        """Compute temporal features for a batch of full sequences.

        Args:
            features: Shape (n_seqs, seq_len, >=42), where first 42 columns are raw+derived.

        Returns:
            Array of shape (n_seqs, seq_len, 3)
        """
        spread_0 = features[..., 32]
        trade_int = features[..., 38]

        roc1 = np.zeros_like(spread_0)
        roc1[:, 1:] = spread_0[:, 1:] - spread_0[:, :-1]

        roc5 = np.zeros_like(spread_0)
        roc5[:, 5:] = spread_0[:, 5:] - spread_0[:, :-5]

        roll_mean = np.zeros_like(trade_int)
        cumsum = np.cumsum(trade_int, axis=1)
        for i in range(min(4, features.shape[1])):
            roll_mean[:, i] = cumsum[:, i] / (i + 1)
        if features.shape[1] > 4:
            shifted = np.zeros_like(cumsum)
            shifted[:, 5:] = cumsum[:, :-5]
            roll_mean[:, 4:] = (cumsum[:, 4:] - shifted[:, 4:]) / 5.0

        return np.stack([roc1, roc5, roll_mean], axis=-1).astype(np.float32)


class InteractionFeatureBuilder:
    """Compute cross-feature interactions used by refined GRU/attention variants.

    Adds 3 interaction features:
    - v8_p0: v8 * p0
    - spread0_p0: spread_0 * p0
    - spread0_v2: spread_0 * v2
    """

    N_INTERACTION = 3
    INTERACTION_COLS = ["v8_p0", "spread0_p0", "spread0_v2"]

    @staticmethod
    def compute(features: np.ndarray, has_derived: bool = True) -> np.ndarray:
        """Compute interaction features.

        Args:
            features: Feature tensor of shape (..., D). D can be 32 raw, 42 raw+derived,
                or larger where the first 42 columns preserve raw+derived layout.
            has_derived: Whether spread_0 is already available at index 32.

        Returns:
            Array of shape (..., 3)
        """
        p0 = features[..., 0]
        v2 = features[..., 14]
        v8 = features[..., 20]

        if has_derived and features.shape[-1] >= 33:
            spread_0 = features[..., 32]
        else:
            spread_0 = features[..., 6] - features[..., 0]

        interactions = np.stack(
            [v8 * p0, spread_0 * p0, spread_0 * v2],
            axis=-1,
        )
        return interactions.astype(np.float32)


class TemporalBuffer:
    """Stateful temporal feature computation for online inference."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.step = 0
        self.spread0_history = []
        self.trade_int_history = []

    def compute_step(self, features_42: np.ndarray) -> np.ndarray:
        """Compute temporal features incrementally.

        Args:
            features_42: Array of shape (>=42,), where first 42 columns are raw+derived.

        Returns:
            Input with 3 temporal columns appended.
        """
        spread_0 = float(features_42[32])
        trade_int = float(features_42[38])

        self.spread0_history.append(spread_0)
        self.trade_int_history.append(trade_int)

        roc1 = spread_0 - self.spread0_history[-2] if self.step >= 1 else 0.0
        roc5 = spread_0 - self.spread0_history[-6] if self.step >= 5 else 0.0
        roll_mean = sum(self.trade_int_history[-5:]) / len(self.trade_int_history[-5:])
        self.step += 1

        temporal = np.array([roc1, roc5, roll_mean], dtype=np.float32)
        return np.concatenate([features_42, temporal]).astype(np.float32)


class Normalizer:
    """Z-score normalizer."""

    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.eps = 1e-8

    def fit(self, X: np.ndarray) -> "Normalizer":
        if len(X.shape) == 3:
            X = X.reshape(-1, X.shape[-1])
        self.mean = X.mean(axis=0).astype(np.float32)
        self.std = X.std(axis=0).astype(np.float32)
        self.std[self.std < self.eps] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        return ((X - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        return (X * self.std + self.mean).astype(np.float32)

    def save(self, path: str) -> None:
        if self.mean is None:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: str) -> "Normalizer":
        data = np.load(path)
        norm = cls()
        norm.mean = data["mean"]
        norm.std = data["std"]
        return norm
