"""PyTorch Dataset for LOB sequence data.

Optimized for CPU training: pre-normalizes and pre-tensorizes all data
during __init__ so __getitem__ is a zero-copy tensor slice.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List

from .preprocessing import (
    DerivedFeatureBuilder,
    InteractionFeatureBuilder,
    MicrostructureFeatureBuilder,
    Normalizer,
    TemporalDerivedFeatureBuilder,
)


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
        interaction_features: bool = False,
        microstructure_features: bool = False,
    ):
        self.derived_features = derived_features
        self.temporal_features = temporal_features and derived_features  # temporal requires derived
        self.interaction_features = interaction_features
        self.microstructure_features = microstructure_features
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

        # Interaction features are stateless and computed from raw/derived columns.
        if self.interaction_features:
            interactions = InteractionFeatureBuilder.compute(
                features_all, has_derived=self.derived_features
            )
            features_all = np.concatenate([features_all, interactions], axis=-1)

        # Microstructure features (OFI, QI slope, spread dynamics, realized vol)
        if self.microstructure_features:
            micro = MicrostructureFeatureBuilder.compute_batch(features_all)
            features_all = np.concatenate([features_all, micro], axis=-1)

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


class WindowedDataset(Dataset):
    """Returns random fixed-length windows from full sequences.

    Matches the official baseline's training/inference mode:
    each sample is a W-step crop with zero initial hidden state.
    """

    def __init__(self, base_dataset: LOBSequenceDataset, window_size: int = 100):
        self.features = base_dataset.features  # (N, 1000, F)
        self.targets = base_dataset.targets    # (N, 1000, 2)
        self.masks = base_dataset.masks        # (N, 1000)
        self.window_size = window_size
        self.n_seqs = len(base_dataset)
        self.max_start = 1000 - window_size

    def __len__(self) -> int:
        return self.n_seqs

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = torch.randint(0, self.max_start + 1, (1,)).item()
        end = start + self.window_size
        return (
            self.features[idx, start:end],
            self.targets[idx, start:end],
            self.masks[idx, start:end],
        )


def create_dataloaders(
    train_path: str,
    valid_path: str,
    batch_size: int = 32,
    num_workers: int = 0,
    normalize: bool = True,
    pin_memory: bool = False,
    derived_features: bool = False,
    temporal_features: bool = False,
    interaction_features: bool = False,
    microstructure_features: bool = False,
    window_size: int = 0,
) -> Tuple[DataLoader, DataLoader, Optional[Normalizer]]:
    """Create train and validation dataloaders with shared normalizer.

    Args:
        window_size: If >0, wrap train dataset in WindowedDataset (random crops).
                     Validation always uses full sequences for fair comparison.
    """
    train_dataset = LOBSequenceDataset(
        train_path, normalize=normalize, derived_features=derived_features,
        temporal_features=temporal_features,
        interaction_features=interaction_features,
        microstructure_features=microstructure_features,
    )

    valid_dataset = LOBSequenceDataset(
        valid_path,
        normalize=normalize,
        normalizer=train_dataset.normalizer if normalize else None,
        derived_features=derived_features,
        temporal_features=temporal_features,
        interaction_features=interaction_features,
        microstructure_features=microstructure_features,
    )

    # Wrap train in windowed sampling if requested
    effective_train = train_dataset
    if window_size > 0:
        effective_train = WindowedDataset(train_dataset, window_size=window_size)

    train_loader = DataLoader(
        effective_train,
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


def create_fulldata_loader(
    train_path: str,
    valid_path: str,
    batch_size: int = 32,
    normalize: bool = True,
    pin_memory: bool = False,
    derived_features: bool = False,
    temporal_features: bool = False,
    interaction_features: bool = False,
    microstructure_features: bool = False,
) -> Tuple[DataLoader, 'Normalizer']:
    """Create a single dataloader from train+val combined.

    Concatenates train and valid parquets with seq_ix offset to avoid collisions.
    Used for full-data training (no validation-based early stopping).
    """
    train_df = pd.read_parquet(train_path)
    valid_df = pd.read_parquet(valid_path)

    # Dynamic offset: max(train_seq_ix) + 1 to guarantee no collisions
    offset = int(train_df['seq_ix'].max()) + 1
    valid_df = valid_df.copy()
    valid_df['seq_ix'] = valid_df['seq_ix'] + offset
    print(f"Full-data: train seqs={train_df['seq_ix'].nunique()}, "
          f"val seqs={valid_df['seq_ix'].nunique()}, offset={offset}")

    # Concatenate
    combined_df = pd.concat([train_df, valid_df], ignore_index=True)
    del train_df, valid_df

    # Write to temp parquet for LOBSequenceDataset
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix='.parquet', delete=False)
    tmp_path = tmp.name
    tmp.close()
    combined_df.to_parquet(tmp_path)
    del combined_df

    dataset = LOBSequenceDataset(
        tmp_path,
        normalize=normalize,
        derived_features=derived_features,
        temporal_features=temporal_features,
        interaction_features=interaction_features,
        microstructure_features=microstructure_features,
    )

    # Clean up temp file
    import os as _os
    try:
        _os.unlink(tmp_path)
    except OSError:
        pass

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )

    return loader, dataset.normalizer
