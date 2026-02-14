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


class MicrostructureFeatureBuilder:
    """Compute LOB microstructure features from raw 32-feature vector.

    Features (6 total):
    0: wOFI_6 — depth-weighted multi-level order flow imbalance
    1: OFI_slope — front-of-book vs deep OFI concentration
    2: QI_slope — queue-imbalance term-structure slope (stateless)
    3: DirSpreadVel — directional spread velocity
    4: SpreadTwist — L1 vs L2 spread dynamics change
    5: RV_vw_4 — volume-weighted short-horizon realized volatility (stateless)

    Features 0,1,3,4 require previous-step values (stateful in online inference).
    Features 2,5 are stateless.
    """

    N_MICROSTRUCTURE = 6
    MICROSTRUCTURE_COLS = [
        "wOFI_6", "OFI_slope", "QI_slope",
        "DirSpreadVel", "SpreadTwist", "RV_vw_4",
    ]

    @staticmethod
    def compute_batch(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Compute microstructure features for full sequences (training).

        Args:
            features: (n_seqs, seq_len, >=32) raw features
        Returns:
            (n_seqs, seq_len, 6) microstructure features
        """
        n_seqs, seq_len, _ = features.shape
        result = np.zeros((n_seqs, seq_len, 6), dtype=np.float32)

        # Extract raw columns
        p = features[..., 0:12]    # prices p0-p11
        v = features[..., 12:24]   # volumes v0-v11
        dp = features[..., 24:28]  # trade price changes
        dv = features[..., 28:32]  # trade volume changes

        # Previous step values (shift right by 1, first step copies current)
        p_prev = np.zeros_like(p)
        p_prev[:, 1:] = p[:, :-1]
        p_prev[:, 0] = p[:, 0]

        v_prev = np.zeros_like(v)
        v_prev[:, 1:] = v[:, :-1]
        v_prev[:, 0] = v[:, 0]

        # Per-level OFI (needed for features 0 and 1)
        ofi_per_level = np.zeros((n_seqs, seq_len, 6), dtype=np.float32)
        for lev in range(6):
            e_bid = (np.where(p[..., lev] >= p_prev[..., lev], v[..., lev], 0.0)
                     - np.where(p[..., lev] <= p_prev[..., lev], v_prev[..., lev], 0.0))
            e_ask = (np.where(p[..., lev + 6] <= p_prev[..., lev + 6], v[..., lev + 6], 0.0)
                     - np.where(p[..., lev + 6] >= p_prev[..., lev + 6], v_prev[..., lev + 6], 0.0))
            ofi_per_level[..., lev] = e_bid - e_ask

        # Feature 0: Depth-weighted multi-level OFI
        depth_weights = np.array([1.0 / (lev + 1) for lev in range(6)], dtype=np.float32)
        result[..., 0] = (ofi_per_level * depth_weights).sum(axis=-1)

        # Feature 1: OFI slope (front vs deep)
        result[..., 1] = ofi_per_level[..., :3].mean(axis=-1) - ofi_per_level[..., 3:].mean(axis=-1)

        # Feature 2: Queue-imbalance term-structure slope (stateless)
        qi = np.zeros((n_seqs, seq_len, 6), dtype=np.float32)
        for lev in range(6):
            qi[..., lev] = (v[..., lev] - v[..., lev + 6]) / (v[..., lev] + v[..., lev + 6] + eps)
        qi_diffs = np.diff(qi, axis=-1)  # (n_seqs, seq_len, 5)
        result[..., 2] = qi_diffs.mean(axis=-1) * 5.0  # scale for visibility

        # Feature 3: Directional spread velocity
        spread_0 = p[..., 6] - p[..., 0]
        spread_0_prev = p_prev[..., 6] - p_prev[..., 0]
        delta_spread = spread_0 - spread_0_prev
        vol_imb = (v[..., 6] - v[..., 0]) / (v[..., 0] + v[..., 6] + eps)
        result[..., 3] = delta_spread * vol_imb

        # Feature 4: Spread curve twist (L1 vs L2 dynamics)
        spread_1 = p[..., 7] - p[..., 1]
        spread_1_prev = p_prev[..., 7] - p_prev[..., 1]
        result[..., 4] = (spread_1 - spread_0) - (spread_1_prev - spread_0_prev)

        # Feature 5: Volume-weighted short-horizon realized volatility (stateless)
        result[..., 5] = np.sqrt((dp ** 2 * (1.0 + np.abs(dv))).sum(axis=-1) + eps)

        return result


class MicrostructureBuffer:
    """Stateful microstructure feature computation for online inference."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.prev_p: Optional[np.ndarray] = None
        self.prev_v: Optional[np.ndarray] = None

    def compute_step(self, features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Compute 6 microstructure features for a single step.

        Args:
            features: (>=32,) raw (or raw+derived) features
        Returns:
            Input array with 6 microstructure columns appended
        """
        p = features[0:12].astype(np.float64)
        v = features[12:24].astype(np.float64)
        dp = features[24:28].astype(np.float64)
        dv = features[28:32].astype(np.float64)

        micro = np.zeros(6, dtype=np.float32)

        if self.prev_p is None:
            # First step: stateful features are 0, stateless computed
            self.prev_p = p.copy()
            self.prev_v = v.copy()
        else:
            p_prev = self.prev_p
            v_prev = self.prev_v

            # Per-level OFI
            ofi_levels = np.zeros(6, dtype=np.float64)
            for lev in range(6):
                e_bid = (v[lev] if p[lev] >= p_prev[lev] else 0.0) - \
                        (v_prev[lev] if p[lev] <= p_prev[lev] else 0.0)
                e_ask = (v[lev + 6] if p[lev + 6] <= p_prev[lev + 6] else 0.0) - \
                        (v_prev[lev + 6] if p[lev + 6] >= p_prev[lev + 6] else 0.0)
                ofi_levels[lev] = e_bid - e_ask

            # Feature 0: wOFI_6
            depth_w = np.array([1.0 / (lev + 1) for lev in range(6)])
            micro[0] = float((ofi_levels * depth_w).sum())

            # Feature 1: OFI slope
            micro[1] = float(ofi_levels[:3].mean() - ofi_levels[3:].mean())

            # Feature 3: Directional spread velocity
            spread_0 = p[6] - p[0]
            spread_0_prev = p_prev[6] - p_prev[0]
            delta_spread = spread_0 - spread_0_prev
            vol_imb = (v[6] - v[0]) / (v[0] + v[6] + eps)
            micro[3] = float(delta_spread * vol_imb)

            # Feature 4: Spread curve twist
            spread_1 = p[7] - p[1]
            spread_1_prev = p_prev[7] - p_prev[1]
            micro[4] = float((spread_1 - spread_0) - (spread_1_prev - spread_0_prev))

            self.prev_p = p.copy()
            self.prev_v = v.copy()

        # Feature 2: QI slope (stateless)
        qi = np.zeros(6, dtype=np.float64)
        for lev in range(6):
            qi[lev] = (v[lev] - v[lev + 6]) / (v[lev] + v[lev + 6] + eps)
        micro[2] = float(np.diff(qi).mean() * 5.0)

        # Feature 5: RV (stateless)
        micro[5] = float(np.sqrt((dp ** 2 * (1.0 + np.abs(dv))).sum() + eps))

        return np.concatenate([features, micro]).astype(np.float32)


class LagFeatureBuilder:
    """Compute multi-horizon lag-diff features from raw LOB features.

    For each selected feature, computes x_t - x_{t-k} for k in LAG_WINDOWS.
    Pads initial steps with 0. All lags are causal (no lookahead).
    """

    # Raw feature indices to lag
    LAG_INDICES = [0, 6, 12, 24]  # p0 (best bid), p6 (best ask), v0 (bid vol), dp0 (trade price)
    LAG_WINDOWS = [1, 4, 16]      # short, medium, long horizon
    N_LAG = 12  # len(LAG_INDICES) * len(LAG_WINDOWS)
    LAG_COLS = [
        f"{name}_diff{k}"
        for name in ["p0", "p6", "v0", "dp0"]
        for k in [1, 4, 16]
    ]

    @staticmethod
    def compute_batch(features: np.ndarray) -> np.ndarray:
        """Compute lag-diff features for full sequences (training).

        Args:
            features: (n_seqs, seq_len, >=32) — raw features must be in first 32 cols
        Returns:
            (n_seqs, seq_len, 12) lag-diff features
        """
        n_seqs, seq_len, _ = features.shape
        result = np.zeros((n_seqs, seq_len, LagFeatureBuilder.N_LAG), dtype=np.float32)
        idx = 0
        for feat_idx in LagFeatureBuilder.LAG_INDICES:
            col = features[:, :, feat_idx]  # (n_seqs, seq_len)
            for lag in LagFeatureBuilder.LAG_WINDOWS:
                if lag < seq_len:
                    result[:, lag:, idx] = col[:, lag:] - col[:, :-lag]
                idx += 1
        return result


class LagBuffer:
    """Stateful lag feature computation for online inference."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.history: list = []

    def compute_step(self, features: np.ndarray) -> np.ndarray:
        """Compute lag-diff features for a single step.

        Args:
            features: (>=32,) current features (raw cols in first 32)
        Returns:
            Input with 12 lag-diff columns appended
        """
        current = np.array([features[i] for i in LagFeatureBuilder.LAG_INDICES], dtype=np.float32)
        self.history.append(current)
        t = len(self.history) - 1

        lags = np.zeros(LagFeatureBuilder.N_LAG, dtype=np.float32)
        idx = 0
        for i in range(len(LagFeatureBuilder.LAG_INDICES)):
            for lag in LagFeatureBuilder.LAG_WINDOWS:
                if t >= lag:
                    lags[idx] = self.history[t][i] - self.history[t - lag][i]
                idx += 1
        return np.concatenate([features, lags]).astype(np.float32)


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
