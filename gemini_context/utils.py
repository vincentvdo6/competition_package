"""Official competition scoring utilities.

The metric is Weighted Pearson Correlation:
- Weights = |y_true| (emphasizes large price movements)
- Predictions clipped to [-6, 6]
- Final score = average of weighted_pearson(t0) and weighted_pearson(t1)
"""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from dataclasses import dataclass


def weighted_pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the Weighted Pearson Correlation Coefficient.

    Emphasizes performance on data points with larger target amplitudes
    by using |target| as sample weight. Predictions clipped to [-6, 6].
    """
    y_pred_clipped = np.clip(y_pred, -6.0, 6.0)
    weights = np.abs(y_true)
    weights = np.maximum(weights, 1e-8)

    sum_w = np.sum(weights)
    if sum_w == 0:
        return 0.0

    mean_true = np.sum(y_true * weights) / sum_w
    mean_pred = np.sum(y_pred_clipped * weights) / sum_w

    dev_true = y_true - mean_true
    dev_pred = y_pred_clipped - mean_pred

    cov = np.sum(weights * dev_true * dev_pred) / sum_w
    var_true = np.sum(weights * dev_true**2) / sum_w
    var_pred = np.sum(weights * dev_pred**2) / sum_w

    if var_true <= 0 or var_pred <= 0:
        return 0.0

    return float(cov / (np.sqrt(var_true) * np.sqrt(var_pred)))


@dataclass
class DataPoint:
    seq_ix: int
    step_in_seq: int
    need_prediction: bool
    state: np.ndarray


class PredictionModel:
    def predict(self, data_point: DataPoint) -> np.ndarray:
        return np.zeros(2)


class ScorerStepByStep:
    """Scores a model using step-by-step online inference (matching competition)."""

    def __init__(self, dataset_path: str):
        self.dataset = pd.read_parquet(dataset_path)
        self.dim = 2
        self.features = self.dataset.columns[3:35]
        self.targets = self.dataset.columns[35:]

    def score(self, model: PredictionModel) -> dict:
        predictions = []
        targets = []

        for row in tqdm(self.dataset.values):
            seq_ix, step_in_seq, need_prediction = row[0], row[1], row[2]
            lob_data = row[3:35]  # 32 features
            labels = row[35:]     # 2 targets

            data_point = DataPoint(seq_ix, step_in_seq, need_prediction, lob_data)
            prediction = model.predict(data_point)

            if prediction is not None:
                predictions.append(prediction)
                targets.append(labels)

        return self.calc_metrics(np.array(predictions), np.array(targets))

    def calc_metrics(self, predictions, targets):
        scores = {}
        for ix_target, target_name in enumerate(self.targets):
            scores[target_name] = weighted_pearson_correlation(
                targets[:, ix_target], predictions[:, ix_target]
            )
        scores["weighted_pearson"] = np.mean(list(scores.values()))
        return scores
