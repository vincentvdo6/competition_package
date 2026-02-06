# Wunderfund Predictorium RNN Challenge — Project Specification

## Agent Workflow Protocol

> **This document is the single source of truth for all AI agents working on this project.**
> Both Claude Code and Codex (VS Code) should read this file before starting any task.

---

## 1. Project Overview

### Competition Summary
- **Host**: Wunder Fund (global HFT prop trading firm)
- **URL**: https://wundernn.io/predictorium
- **Task**: Predict two target variables (`t0`, `t1`) representing future price movements from Limit Order Book (LOB) sequences
- **Metric**: Weighted Pearson Correlation (weighted by |target|, predictions clipped to [-6, 6])
- **Prize Pool**: $13,600

### Problem Type
- **Sequence-to-value regression** (not classification)
- **Online/streaming inference** — model receives one `DataPoint` at a time, must maintain hidden state
- **Independent sequences** — must reset hidden state on new `seq_ix`

### Data Format
```
train.parquet  — 10,721 sequences × 1000 steps each
valid.parquet  — 1,444 sequences × 1000 steps each
```

Each row = one market state with columns:
- `seq_ix` (int): Sequence ID (independent sequences, randomly shuffled)
- `step_in_seq` (int): Step number 0–999
- `need_prediction` (bool): True for steps 99–999

**Features (32 total):**
| Group | Columns | Description |
|-------|---------|-------------|
| Bid Prices | `p0`–`p5` | Anonymized LOB bid price features (6 levels) |
| Ask Prices | `p6`–`p11` | Anonymized LOB ask price features (6 levels) |
| Bid Volumes | `v0`–`v5` | Anonymized LOB bid volume features (6 levels) |
| Ask Volumes | `v6`–`v11` | Anonymized LOB ask volume features (6 levels) |
| Trade Prices | `dp0`–`dp3` | Anonymized recent trade price features |
| Trade Volumes | `dv0`–`dv3` | Anonymized recent trade volume features |

**Targets:**
- `t0`, `t1`: Two different future price movement indicators (continuous, anonymized)

### Submission Format
```python
import numpy as np
from utils import DataPoint

class PredictionModel:
    def __init__(self):
        # Load weights, initialize hidden state, etc.
        pass

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        # data_point.seq_ix      — sequence ID
        # data_point.step_in_seq — step number (0-999)
        # data_point.need_prediction — bool
        # data_point.state       — np.ndarray of shape (32,) — the 32 features

        if not data_point.need_prediction:
            # Still process the data to update hidden state!
            return None

        return np.array([pred_t0, pred_t1])  # shape (2,)
```

**Critical constraints:**
- Must reset hidden state when `seq_ix` changes
- Must still process warm-up steps (0–98) even though `need_prediction=False`
- Predictions clipped to [-6, 6] during evaluation
- All model weights must be included in the submission zip
- `solution.py` must be at the root of the zip archive

---

## 2. Repository Structure

```
wunderfund-predictorium/
├── WUNDERFUND_PROJECT_SPEC.md     ← THIS FILE (read first, always)
├── README.md
│
├── data/                          ← Raw competition data (gitignored)
│   ├── train.parquet
│   └── valid.parquet
│
├── notebooks/                     ← EDA & experimentation (CODEX territory)
│   ├── 01_eda.ipynb               — Data exploration, distributions, correlations
│   ├── 02_feature_analysis.ipynb  — Feature importance, autocorrelation, stationarity
│   ├── 03_baseline_experiments.ipynb — Quick model iterations
│   └── 04_ablation_studies.ipynb  — Systematic component testing
│
├── src/                           ← Core library code (SHARED — careful coordination)
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py             — PyTorch Dataset for sequence loading
│   │   ├── preprocessing.py       — Normalization, feature engineering
│   │   └── dataloader.py          — Custom DataLoader with sequence batching
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gru_baseline.py        — GRU model
│   │   ├── lstm_model.py          — LSTM model
│   │   ├── transformer_model.py   — Transformer model
│   │   ├── mamba_model.py         — Mamba/SSM model (stretch goal)
│   │   └── ensemble.py            — Model ensembling logic
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py             — Training loop with logging
│   │   ├── losses.py              — Custom loss functions (weighted MSE, etc.)
│   │   └── scheduler.py           — LR scheduling strategies
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py             — Weighted Pearson implementation
│       └── validator.py           — Local validation pipeline
│
├── configs/                       ← Experiment configs (YAML)
│   ├── gru_baseline.yaml
│   ├── lstm_v1.yaml
│   ├── transformer_v1.yaml
│   └── best_model.yaml
│
├── scripts/                       ← CLI tools (CLAUDE CODE territory)
│   ├── train.py                   — Main training entrypoint
│   ├── evaluate.py                — Run validation locally
│   ├── export_submission.py       — Package solution.py + weights into zip
│   └── sweep.py                   — Hyperparameter sweep launcher
│
├── submissions/                   ← Packaged submissions
│   └── solution.py                — Final submission file
│
├── logs/                          ← Training logs & checkpoints (gitignored)
├── utils.py                       ← Competition-provided utils (DO NOT MODIFY)
├── requirements.txt
└── .gitignore
```

---

## 3. Agent Role Assignments

### 🔵 Claude Code — "The Architect"
**Primary responsibilities:**
- Project scaffolding, directory structure, and boilerplate
- `src/` core library code: models, training loops, data pipelines
- `scripts/` CLI tools: training, evaluation, export
- Config management (YAML experiment configs)
- Git operations, dependency management
- Code review and refactoring across the codebase
- Debugging training failures, CUDA issues, import errors
- Submission packaging and validation

**Claude Code strengths to exploit:**
- Terminal access for running training jobs, git, pip installs
- Multi-file refactoring across the full project tree
- Long-running debugging sessions with iterative fixes
- System-level tasks (environment setup, CUDA checks)

### 🟢 Codex (VS Code) — "The Experimenter"
**Primary responsibilities:**
- `notebooks/` EDA and rapid experimentation
- Feature engineering research and prototyping
- Model architecture experiments (try ideas fast in notebooks)
- Hyperparameter tuning iterations
- Visualization of results, loss curves, feature distributions
- Ablation studies (what helps, what doesn't)
- Literature review implementation (replicating paper techniques)

**Codex strengths to exploit:**
- Fast inline code generation within notebooks
- Quick iteration on small code blocks
- Autocomplete-driven exploration of APIs
- Side-by-side editing of related cells

---

## 4. Coordination Protocol

### Rule 1: Shared Interface Contract
Both agents write code against these interfaces. **Never change an interface without updating this spec.**

```python
# === DATA INTERFACE ===
# Input to all models during training:
#   x: torch.Tensor of shape (batch, seq_len, 32)  — 32 features
#   targets: torch.Tensor of shape (batch, seq_len, 2)  — t0, t1
#   mask: torch.Tensor of shape (batch, seq_len)  — True where need_prediction=True

# === MODEL INTERFACE ===
# All models in src/models/ must implement:
class BaseModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        # config loaded from YAML

    def forward(self, x: torch.Tensor, hidden=None) -> tuple[torch.Tensor, Any]:
        """
        Args:
            x: (batch, seq_len, 32) input features
            hidden: model-specific hidden state (None = fresh sequence)
        Returns:
            predictions: (batch, seq_len, 2) predicted t0, t1
            hidden: updated hidden state
        """
        ...

    def init_hidden(self, batch_size: int) -> Any:
        """Return fresh hidden state for new sequences."""
        ...

# === METRIC INTERFACE ===
def weighted_pearson_correlation(y_true, y_pred, target_idx):
    """From utils.py — DO NOT REIMPLEMENT, use the competition version."""
    ...

# === CONFIG INTERFACE ===
# All configs are YAML dicts with at minimum:
# {
#   "model": { "type": "gru|lstm|transformer|mamba", "hidden_size": int, ... },
#   "training": { "lr": float, "epochs": int, "batch_size": int, ... },
#   "data": { "seq_len": int, "normalize": str, ... }
# }
```

### Rule 2: File Ownership
| Path | Owner | Other agent may... |
|------|-------|--------------------|
| `notebooks/*` | Codex | Claude Code: read only |
| `src/models/*` | Claude Code | Codex: import and use, don't modify |
| `src/data/*` | Claude Code | Codex: import and use, don't modify |
| `src/training/*` | Claude Code | Codex: read only |
| `scripts/*` | Claude Code | Codex: read only |
| `configs/*` | Both | Either can create new configs, never modify each other's |
| `submissions/*` | Claude Code | Codex: read only |

### Rule 3: Communication via Files
Agents communicate through structured files, not conversation context:

```
logs/
├── experiments.jsonl        ← Append-only experiment log
│   Format: {"timestamp": "...", "model": "...", "config": "...",
│            "val_score": float, "notes": "..."}
│
├── findings.md              ← Key discoveries (either agent appends)
│   Format: ## [DATE] Finding Title
│           **Source**: notebook/script name
│           **Result**: what was discovered
│           **Action**: what to do next
│
└── todo.md                  ← Task queue (either agent can add/complete)
    Format: - [ ] Task description (@claude-code | @codex)
            - [x] Completed task — result summary
```

### Rule 4: Git Branch Strategy
```
main                ← Stable, validated code only
├── feat/eda        ← Codex: exploratory work
├── feat/gru-v1     ← Claude Code: first GRU implementation
├── feat/lstm-v1    ← Claude Code: LSTM experiments
├── feat/features   ← Codex: feature engineering
└── feat/transformer ← Claude Code: transformer implementation
```

---

## 5. Development Roadmap

### Phase 1: Foundation (Day 1)
**Goal**: Running baseline with local validation score

| Task | Agent | Priority |
|------|-------|----------|
| Set up repo structure, requirements.txt, .gitignore | Claude Code | P0 |
| Download data, verify parquet loading | Claude Code | P0 |
| EDA notebook: distributions, correlations, target analysis | Codex | P0 |
| Implement `src/data/dataset.py` (PyTorch Dataset) | Claude Code | P0 |
| Implement `src/evaluation/metrics.py` (wrap competition metric) | Claude Code | P0 |
| Run competition's example GRU baseline, record score | Claude Code | P0 |

### Phase 2: Strong Baseline (Days 2–3)
**Goal**: Tuned GRU that meaningfully beats the example

| Task | Agent | Priority |
|------|-------|----------|
| Feature analysis: autocorrelation, stationarity, cross-feature | Codex | P0 |
| Implement GRU model in `src/models/gru_baseline.py` | Claude Code | P0 |
| Implement training loop with logging | Claude Code | P0 |
| Feature engineering: rolling stats, bid-ask spread, imbalance | Codex | P1 |
| Hyperparameter sweep on GRU (hidden size, layers, dropout, LR) | Both | P1 |
| Implement proper normalization pipeline | Claude Code | P1 |

### Phase 3: Architecture Exploration (Days 4–6)
**Goal**: Test LSTM, Transformer, and find best single model

| Task | Agent | Priority |
|------|-------|----------|
| LSTM implementation and comparison | Claude Code | P1 |
| Transformer implementation (causal masking for online inference) | Claude Code | P1 |
| Notebook: architecture comparison analysis | Codex | P1 |
| Custom loss function experiments (weighted MSE targeting large moves) | Codex | P2 |
| Sequence length ablation (how much context helps?) | Codex | P2 |
| Mamba/SSM exploration (if time permits) | Claude Code | P2 |

### Phase 4: Optimization & Submission (Days 7–8)
**Goal**: Best possible score, clean submission

| Task | Agent | Priority |
|------|-------|----------|
| Ensemble top 2–3 models | Claude Code | P1 |
| Final hyperparameter fine-tuning | Both | P1 |
| Export and validate submission zip | Claude Code | P0 |
| Verify submission runs correctly in clean environment | Claude Code | P0 |
| Document all findings in logs/findings.md | Both | P1 |

---

## 6. Technical Strategy Notes

### Key Insights from the Competition Structure

1. **Weighted metric = large moves matter most.** The Weighted Pearson Correlation weights by |target|. This means correctly predicting the direction and magnitude of LARGE price movements is worth far more than getting small movements right. Implications:
   - Consider sample weighting during training (upweight samples with large |t0| or |t1|)
   - A loss function like `weight * MSE` where `weight = |target| + epsilon` could help
   - Don't optimize for R² or plain MSE — optimize for the actual competition metric

2. **Online inference constraint shapes architecture choice.** The model sees one step at a time during evaluation. This means:
   - RNNs (GRU/LSTM) are naturally suited — hidden state carries forward
   - Transformers need adaptation: either process the full sequence up to current step (expensive) or use a sliding window / cached KV approach
   - Mamba/SSMs are excellent here — designed for efficient sequential processing

3. **32 features, 1000 steps, ~12K sequences.** This is a moderately sized dataset. Overfitting is a real risk with large models. Regularization strategy matters:
   - Dropout on hidden states
   - Weight decay
   - Early stopping on validation weighted Pearson
   - Consider data augmentation (sequence reversal? noise injection?)

4. **Feature structure has domain meaning.** Even though anonymized:
   - `p0-p5` (bid) and `p6-p11` (ask) are mirrored → bid-ask spread features
   - `v0-v5` (bid vol) and `v6-v11` (ask vol) → order book imbalance features
   - `dp0-dp3` and `dv0-dv3` → trade activity features
   - Cross-feature interactions likely matter (e.g., volume-weighted price changes)

5. **Two targets may have different optimal models.** t0 and t1 represent different types of future price movements. Consider:
   - Shared backbone with separate prediction heads
   - Separate models for each target
   - Multi-task learning with task-specific loss weighting

### Architecture Priority Order
1. **GRU** (start here — fast to train, strong baseline for sequences)
2. **LSTM** (marginal improvement potential, worth testing)
3. **Transformer with causal masking** (better long-range dependencies, but harder to deploy online)
4. **Mamba-2 / S4** (best theoretical fit for online sequence modeling, but implementation complexity)

### Feature Engineering Ideas
```python
# === Derived features to test ===
# Bid-ask spread at each level
spread_i = ask_price_i - bid_price_i

# Order book imbalance
imbalance_i = (bid_vol_i - ask_vol_i) / (bid_vol_i + ask_vol_i + eps)

# Weighted mid price
wmid = (bid_price_0 * ask_vol_0 + ask_price_0 * bid_vol_0) / (bid_vol_0 + ask_vol_0 + eps)

# Volume pressure (total bid vs ask volume)
bid_pressure = sum(v0...v5)
ask_pressure = sum(v6...v11)
pressure_ratio = bid_pressure / (ask_pressure + eps)

# Price momentum (diff features if sequential)
# Note: must compute within sequence, reset on new seq_ix

# Trade intensity
trade_volume_total = sum(dv0...dv3)
```

---

## 7. Environment Setup

```bash
# Python 3.10+ recommended
python -m venv venv
source venv/bin/activate

# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas pyarrow scikit-learn matplotlib seaborn
pip install pyyaml tqdm wandb jupyter

# Optional (for advanced architectures)
pip install mamba-ssm  # Mamba
pip install einops      # Transformer utilities

# Verify
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## 8. Quick Reference Commands

```bash
# Training
python scripts/train.py --config configs/gru_baseline.yaml

# Evaluation
python scripts/evaluate.py --config configs/gru_baseline.yaml --checkpoint logs/best_model.pt

# Export submission
python scripts/export_submission.py --config configs/best_model.yaml --output submissions/solution.zip

# Hyperparameter sweep
python scripts/sweep.py --config configs/gru_baseline.yaml --param training.lr --values 1e-4 3e-4 1e-3
```

---

## 9. Prompt Templates for Each Agent

### When starting a Claude Code session:
```
Read WUNDERFUND_PROJECT_SPEC.md first.
Check logs/todo.md for pending tasks assigned to @claude-code.
Check logs/findings.md for recent discoveries from Codex.
Check logs/experiments.jsonl for latest scores.
Then proceed with your assigned task.
After completing work, update todo.md and experiments.jsonl.
```

### When starting a Codex session:
```
Read WUNDERFUND_PROJECT_SPEC.md first.
Check logs/todo.md for pending tasks assigned to @codex.
Check logs/findings.md for recent discoveries from Claude Code.
Check logs/experiments.jsonl for latest scores.
Work in notebooks/ — import from src/ but don't modify src/.
After completing work, update findings.md with key results.
```

---

## 10. Success Criteria

| Milestone | Target | Status |
|-----------|--------|--------|
| Baseline GRU running | Any positive correlation | ✅ Val avg 0.2578 (t0: 0.3869, t1: 0.1286) |
| Beat example baseline | > example score | ⬜ |
| Strong single model | Top 50% of submissions | ⬜ |
| Optimized submission | Top 25% of submissions | ⬜ |
| Stretch: Top 100 | Top 20% of submissions | ⬜ |

---

*Last updated: 2026-02-05*
*Competition data date: 2025-12-30*
