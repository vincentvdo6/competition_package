"""Vanilla GRU ensemble: 10 models, raw 32 features, no normalization."""

import os
import numpy as np
import torch
import torch.nn as nn


class VanillaGRU(nn.Module):
    def __init__(self, input_size=32, hidden_size=64, num_layers=3, output_size=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        output, hidden = self.gru(x, hidden)
        predictions = self.fc(output)
        return predictions, hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)


# Model configs: (filename, hidden_size, num_layers, weight)
MODEL_CONFIGS = [('model_0.pt', 64, 3, 1.0), ('model_1.pt', 64, 3, 1.0), ('model_2.pt', 64, 3, 1.0), ('model_3.pt', 64, 3, 1.0), ('model_4.pt', 64, 3, 1.0), ('model_5.pt', 64, 3, 1.0), ('model_6.pt', 64, 3, 1.0), ('model_7.pt', 64, 3, 1.0), ('model_8.pt', 64, 3, 1.0), ('model_9.pt', 64, 3, 1.0)]


class PredictionModel:
    def __init__(self, model_path=""):
        self.device = torch.device("cpu")
        torch.set_num_threads(1)

        base_dir = os.path.dirname(os.path.abspath(__file__))

        self.models = []
        self.hiddens = []
        self.weights = []

        for filename, h, nl, w in MODEL_CONFIGS:
            ckpt = torch.load(
                os.path.join(base_dir, filename),
                map_location="cpu",
                weights_only=False,
            )
            state_dict = ckpt["model_state_dict"]

            model = VanillaGRU(input_size=32, hidden_size=h, num_layers=nl)

            # Filter state_dict keys
            filtered = {}
            for k, v in state_dict.items():
                if k.startswith("gru."):
                    filtered[k] = v
                elif k.startswith("output_proj."):
                    new_key = k.replace("output_proj.", "fc.")
                    filtered[new_key] = v
            model.load_state_dict(filtered)
            model.eval()

            self.models.append(model)
            self.hiddens.append(model.init_hidden(1))
            self.weights.append(w)

        # Normalize weights
        total_w = sum(self.weights)
        self.weights = [w / total_w for w in self.weights]

        self.prev_seq_ix = None

    @torch.no_grad()
    def predict(self, data_point) -> np.ndarray:
        seq_ix = data_point.seq_ix
        if seq_ix != self.prev_seq_ix:
            self.hiddens = [m.init_hidden(1) for m in self.models]
            self.prev_seq_ix = seq_ix

        features = data_point.state.astype(np.float32)[:32]
        x = torch.from_numpy(features).unsqueeze(0).unsqueeze(0)

        pred_sum = np.zeros(2, dtype=np.float32)
        for i, model in enumerate(self.models):
            pred, self.hiddens[i] = model(x, self.hiddens[i])
            pred_sum += self.weights[i] * pred.squeeze().numpy()

        if not data_point.need_prediction:
            return None
        return pred_sum.clip(-6, 6)
