#!/usr/bin/env python
"""Evaluate model using competition scorer."""

import argparse
import torch
import yaml
import os
import sys
import numpy as np

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils import DataPoint, ScorerStepByStep
from src.models.gru_baseline import GRUBaseline
from src.models.lstm_model import LSTMModel
from src.data.preprocessing import Normalizer, DerivedFeatureBuilder


class PyTorchPredictionModel:
    """Wrapper for trained PyTorch model to match competition interface.

    This class wraps a trained PyTorch model to work with the competition's
    ScorerStepByStep evaluation harness. It handles:
    - Model loading from checkpoint
    - Normalization using saved normalizer
    - Hidden state management (reset on new sequence)
    - Online inference (step-by-step predictions)
    """

    def __init__(self, checkpoint_path: str, config_path: str, normalizer_path: str):
        """Initialize model wrapper.

        Args:
            checkpoint_path: Path to model checkpoint (.pt file)
            config_path: Path to config YAML file
            normalizer_path: Path to normalizer (.npz file)
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load normalizer
        self.normalizer = Normalizer.load(normalizer_path)

        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Check if model uses derived features
        self.derived_features = self.config.get('data', {}).get('derived_features', False)

        # Load model
        model_type = self.config.get('model', {}).get('type', 'gru')
        if model_type == 'lstm':
            self.model = LSTMModel(self.config)
        else:
            self.model = GRUBaseline(self.config)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"Loaded model from {checkpoint_path}")
        print(f"Best score from training: {checkpoint.get('best_score', 'N/A')}")
        if self.derived_features:
            print("Derived features: ENABLED (42 input features)")

        # State management for online inference
        self.current_seq_ix = None
        self.hidden = None

    def predict(self, data_point: DataPoint) -> np.ndarray:
        """Make prediction matching competition interface.

        Args:
            data_point: DataPoint with seq_ix, step_in_seq, need_prediction, state

        Returns:
            np.ndarray of shape (2,) with predictions for t0, t1
            or None if need_prediction is False
        """
        # Reset hidden state on new sequence
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.hidden = None

        # Compute derived features if needed, then normalize
        raw = data_point.state.reshape(1, -1).astype(np.float32)
        if self.derived_features:
            derived = DerivedFeatureBuilder.compute(raw)
            raw = np.concatenate([raw, derived], axis=-1)
        features = self.normalizer.transform(raw)
        x = torch.from_numpy(features).to(self.device)

        # Forward step (updates hidden state)
        with torch.no_grad():
            pred, self.hidden = self.model.forward_step(x, self.hidden)
            pred = pred.cpu().numpy().squeeze()

        # Clip predictions to competition range
        pred = np.clip(pred, -6, 6)

        # Return None for warm-up steps
        if not data_point.need_prediction:
            return None

        return pred


def main():
    parser = argparse.ArgumentParser(description='Evaluate model using competition scorer')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--normalizer', type=str, default='logs/normalizer.npz',
                        help='Path to normalizer file')
    parser.add_argument('--data', type=str, default='datasets/valid.parquet',
                        help='Path to evaluation data')
    args = parser.parse_args()

    print(f"Evaluating model from {args.checkpoint}")
    print(f"Using data from {args.data}")
    print("-" * 60)

    # Create model wrapper
    model = PyTorchPredictionModel(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        normalizer_path=args.normalizer
    )

    # Create scorer
    scorer = ScorerStepByStep(args.data)

    # Run evaluation
    print("\nRunning evaluation (this may take a while)...")
    results = scorer.score(model)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Weighted Pearson Correlation (avg): {results['weighted_pearson']:.6f}")
    print(f"  t0: {results['t0']:.6f}")
    print(f"  t1: {results['t1']:.6f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
