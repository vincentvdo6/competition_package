"""PyTorch Dataset for LOB sequence data.

Optimized for CPU training: pre-normalizes and pre-tensorizes all data
during __init__ so __getitem__ is a zero-copy tensor slice.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List

from .preprocessing import Normalizer, DerivedFeatureBuilder, TemporalDerivedFeatureBuilder


class LOBSequenceDataset(Dataset):
    """PyTorch Dataset for LOB sequence data.

    All data is pre-normalized and stored as contiguous tensors in RAM.
    __getitem__ is a simple index operation with no copies or transforms.
    """

    FEATURE_COLS = (
        [f'p{i}' for i in range(12)] +
        [f'v{i}' for i in range(12)] +
        [f'dp{i}' for i in range(4)] +
        [f'dv{i}' for i in range(4)]
    )
    TARGET_COLS = ['t0', 't1']

    def __init__(
        self,
        parquet_path: str,
        normalize: bool = True,
        normalizer: Optional[Normalizer] = None,
        derived_features: bool = False,
        temporal_features: bool = False,
    ):
        self.derived_features = derived_features
        self.temporal_features = temporal_features and derived_features  # temporal requires derived
        self.df = pd.read_parquet(parquet_path)
        self.seq_ids = np.sort(self.df['seq_ix'].unique())
        n_seqs = len(self.seq_ids)

        n_base = 32 + (DerivedFeatureBuilder.N_DERIVED if derived_features else 0)

        # Pre-allocate contiguous arrays for all sequences (base features first)
        features_all = np.empty((n_seqs, 1000, n_base), dtype=np.float32)
        targets_all = np.empty((n_seqs, 1000, 2), dtype=np.float32)
        masks_all = np.empty((n_seqs, 1000), dtype=np.bool_)

        # Group once then fill arrays (much faster than repeated df filtering)
        grouped = self.df.sort_values('step_in_seq').groupby('seq_ix')
        seq_ix_to_idx = {s: i for i, s in enumerate(self.seq_ids)}

        for seq_ix, group in grouped:
            idx = seq_ix_to_idx[seq_ix]
            raw = group[self.FEATURE_COLS].values.astype(np.float32)
            if derived_features:
                derived = DerivedFeatureBuilder.compute(raw)
                features_all[idx] = np.concatenate([raw, derived], axis=-1)
            else:
                features_all[idx] = raw
            targets_all[idx] = group[self.TARGET_COLS].values
            masks_all[idx] = group['need_prediction'].values

        # Compute temporal features BEFORE normalization (on raw derived values)
        if self.temporal_features:
            temporal = TemporalDerivedFeatureBuilder.compute_batch(features_all)
            features_all = np.concatenate([features_all, temporal], axis=-1)

        n_features = features_all.shape[-1]

        # Handle normalization (over all features including derived + temporal)
        self.normalizer = normalizer
        if normalize:
            if normalizer is None:
                flat = features_all.reshape(-1, n_features)
                self.normalizer = Normalizer()
                self.normalizer.fit(flat)
            # Apply normalization in-place across all data at once
            flat = features_all.reshape(-1, n_features)
            flat[:] = self.normalizer.transform(flat)

        # Convert to contiguous tensors (zero-copy from numpy)
        self.features = torch.from_numpy(np.ascontiguousarray(features_all))
        self.targets = torch.from_numpy(np.ascontiguousarray(targets_all))
        self.masks = torch.from_numpy(np.ascontiguousarray(masks_all))

        # Free the DataFrame - no longer needed
        del self.df

    def __len__(self) -> int:
        return len(self.seq_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx], self.masks[idx]

    def get_seq_ix(self, idx: int) -> int:
        return self.seq_ids[idx]


def create_dataloaders(
    train_path: str,
    valid_path: str,
    batch_size: int = 32,
    num_workers: int = 0,
    normalize: bool = True,
    pin_memory: bool = False,
    derived_features: bool = False,
    temporal_features: bool = False,
) -> Tuple[DataLoader, DataLoader, Optional[Normalizer]]:
    """Create train and validation dataloaders with shared normalizer.

    Since all data is pre-loaded as tensors, num_workers=0 is optimal
    (avoids multiprocessing serialization overhead for simple index ops).
    Set pin_memory=True when using GPU for faster CPU->GPU transfers.
    """
    train_dataset = LOBSequenceDataset(
        train_path, normalize=normalize, derived_features=derived_features,
        temporal_features=temporal_features,
    )

    valid_dataset = LOBSequenceDataset(
        valid_path,
        normalize=normalize,
        normalizer=train_dataset.normalizer if normalize else None,
        derived_features=derived_features,
        temporal_features=temporal_features,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    return train_loader, valid_loader, train_dataset.normalizer
