# Wunderfund Predictorium RNN Challenge — Full Context for Strategic Analysis

## Competition Overview

- **Competition**: Wunderfund Predictorium (HFT prop trading firm), Kaggle-like platform at wunderfund.io
- **Task**: Predict two targets (t0, t1) from Limit Order Book (LOB) sequences
- **Metric**: **Weighted Pearson Correlation** — weighted by |target|, predictions clipped to [-6, 6], final score = average of weighted_pearson(t0) and weighted_pearson(t1)
- **Prize Pool**: $13,600
- **Deadline**: ~March 2026
- **Our rank**: **73/4728** (top 1.5%) with score **0.2885**
- **Final evaluation uses a DIFFERENT dataset** — generalization matters more than val optimization

### Scoring Environment (Docker)
- `python:3.11-slim-bookworm` with PyTorch CPU (**NO GPU**)
- 1 vCPU core, 16 GB RAM, local SSD
- Time limit: ~4200s (70 minutes)
- No internet, offline execution
- Submission: `.zip` with `solution.py` at root + model files, max **20MB**
- **5 submissions per day**

### Data Format
- **Train**: 10,721 sequences x 1000 steps each
- **Valid**: 1,444 sequences x 1000 steps each
- **Test/Final**: ~1,500 sequences each (similar to validation)
- **Features**: 32 raw features — bid/ask prices (p0-p11), volumes (v0-v11), trade features (dp0-dv3)
- **Data is pre-scaled to approximately [-5.2, 5.2]** — NOT raw prices/volumes
- **need_prediction**: True for steps 99-999 (first 99 steps are warmup, no scoring)
- **Two targets**: t0 (easier, ~0.47 weighted Pearson achievable), t1 (harder, best ~0.14-0.17)

### Inference Protocol (Step-by-Step, Stateful)
```python
class PredictionModel:
    def __init__(self, model_path=""):
        # Load models, initialize hidden states
    def predict(self, data_point) -> np.ndarray:
        # data_point.seq_ix: sequence ID (MUST reset hidden state when changes)
        # data_point.step_in_seq: step within sequence (0-999)
        # data_point.need_prediction: True for steps 99-999
        # data_point.state: np.ndarray of 32 features
        # Return: np.ndarray of shape (2,) for (t0, t1) or None if need_prediction=False
```
Each step is processed individually with hidden state carried forward. Must reset when seq_ix changes.

---

## Our Current Best: Vanilla GRU Ensemble

### Architecture (see `gru_baseline.py` and `config.yaml`)
```
Raw 32 features -> 3-layer GRU (hidden_size=64) -> Linear(64, 2)
```
That's it. NO input projection, NO LayerNorm, NO feature engineering, NO normalization, NO MLP output head. **Simplicity IS the regularization.**

### Training Recipe
- Loss: Plain MSE
- Optimizer: AdamW, lr=0.001, weight_decay=0.0
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)
- Batch size: 192, gradient clip 1.0, AMP enabled
- Early stopping: patience=12 (best models stop at ~15-20 epochs)
- Dropout: 0.0

### Ensemble
- 10 models trained with different random seeds (42-64), flat uniform averaging
- Each model produces (t0, t1) predictions, averaged across all 10
- Inference: ~170s per model on CPU (1700s total for 10)
- ONNX alternative: ~42s per model (420s for 10)

---

## Leaderboard Context (Feb 16, 2026)

| Rank | User | Score | Subs | Notes |
|------|------|-------|------|-------|
| 1 | insuperabilehart | **0.3240** | 96 | |
| 2 | Giovanni | **0.3168** | 96 | "99% pure ML" |
| 3 | sultanmunirov | **0.3138** | 78 | "improve generalisation, not validation score" |
| 7 | Prefect | **0.3051** | 88 | Gave ONNX/speed tips |
| ~50 | — | ~0.2920 | — | Our target |
| **73** | **Us** | **0.2885** | 43 | 10-model vanilla GRU ensemble |
| 100 | IoannisM | 0.2834 | 16 | |

### Gap Analysis
- Top-1 is +0.0355 above us (0.3240 vs 0.2885)
- Top-50 target is +0.0035 above us (~0.2920 vs 0.2885)
- Someone scored **0.2921 with 1 submission** and 0.2958 with 12 subs — strong single models exist

### Discord Intelligence (from top competitors)
- **"improve generalisation, not validation score"** (sultanmunirov, #3)
- **"99% pure ML"** (Giovanni, #2) — top competitors use better models, not features
- **"almost all financial features do not provide an increase"** (sultanmunirov, #3)
- **"0.292 with raw features, own model"** with very few submissions (unknown user, Jan 5)
- **t1 is universally hard** — best reported ~0.14
- **Val improvement often leads to LB decline** (multiple users report this)
- **Trust the given validation set** over custom cross-validation (Azure_UnshadeNN)

---

## Val-to-LB Gap Analysis (CRITICAL PHENOMENON)

| Model | Val Score | LB Score | Gap |
|-------|----------|----------|-----|
| vanilla_ens10 (10 models) | 0.2708 | **0.2885** | **+0.0177** |
| parity_v1_s43 (single) | 0.2737 | 0.2814 | +0.0077 |
| vanilla_s59 (single) | 0.2727 | 0.2764 | +0.0037 |
| Official baseline | 0.2595 | 0.2761 | +0.0166 |
| baseline_match (complex pipeline) | 0.2738 | 0.2394 | **-0.0344** |

**Key patterns**:
1. Vanilla GRU has a **POSITIVE gap** — the test set is easier than validation for simple models
2. **Ensemble amplifies the positive gap**: single model +0.004 to +0.008, 10-model ensemble +0.018
3. **Complex models have NEGATIVE gap** — they overfit to training-specific patterns
4. 20 models = 10 models on LB (0.2884 vs 0.2885) — diminishing returns from same recipe
5. **ANY modification to the vanilla recipe hurts LB** even when val improves
6. There is a material **train-valid distribution shift** (KS stat up to 0.33 in price features)

---

## COMPREHENSIVE KILL LIST — Everything We've Tried and Failed

### Architecture Changes (ALL KILLED)
- **Larger hidden sizes**: h=128 (-0.0016 val), h=144 (-0.0029), h=192 (-0.0020), h=192 2L (-0.0037). h=64 IS the sweet spot.
- **input_proj + LayerNorm**: Causes catastrophic -0.034 LB gap (val looks fine, LB collapses)
- **MLP output head**: Part of complex pipeline that fails
- **Attention layers** (GRUAttention with rolling window): Negative on LB
- **TCN (Temporal Convolutional Network)**: LB 0.2683, well below vanilla
- **Small transformer**: Negative
- **CVML (learnable feature mixing MLP with residual)**: Mean val 0.2633-0.2636, KILLED
- **SE-Net feature gate (squeeze-excitation per timestep)**: Negative
- **LSTM instead of GRU**: Mean val 0.2576 (-0.011 vs GRU). Way too weak.

### Feature Engineering (ALL KILLED)
- **Derived 42 features** (spreads, imbalances, trade intensity, pressure): HURT vanilla GRU generalization
- **Microstructure features**: HURT
- **Temporal features** (rolling stats, rate-of-change): HURT
- **Interaction features** (v8*p0, spread*p0): HURT
- **Lag features** (multi-horizon diffs at k=1,4,16): Mean val 0.2582 (-0.008)
- **Z-score normalization**: HURTS vanilla GRU (data is already pre-scaled)
- **RevIN (Reversible Instance Normalization)**:
  - Full-sequence: Val 0.339 — INFLATED by lookahead bias, useless for real inference
  - Causal: Mean val 0.2497 (-0.016). Per-sequence running-stats normalization actively hurts.

### Loss Functions (ALL KILLED on vanilla GRU)
- **Pearson blend** (70% MSE + 30% Pearson): Mean val 0.2695 (on par), but LB 0.2868 when mixed into ensemble (WORSE than 0.2885)
- **Pearson primary** (ramps to 60% Pearson): Mean val 0.2652 (-0.004), training instability
- **PearsonPrimaryLoss**: Negative, unstable
- **Huber loss (delta=1.0)**: CATASTROPHIC mean val 0.2438 (-0.025)
- **CombinedLoss** (weighted MSE blend): Part of complex pipeline that fails
- **Auxiliary heads** (delta prediction, sign classification): Negative
- **Recency weighting** (weight later steps more): Negative

### Training Modifications (ALL KILLED)
- **Full-data training** (train+val combined): 7ep=0.2572, 18ep=0.2455 vs 0.2580 baseline. Worse.
- **Stronger regularization** (WD=2e-4, dropout=0.30, OneCycleLR): -0.005 delta
- **SAM optimizer** (sharpness-aware minimization, rho=0.05): Mean val 0.2611 (-0.005)
- **EMA** (exponential moving average of weights): Negative
- **Cosine warmup scheduler**: As recipe variant, mean val 0.2663 (-0.003)
- **Adam instead of AdamW**: Negative
- **Higher learning rate**: Negative
- **Longer training**: Negative — early stopping at ~15-20 epochs is optimal
- **Bagging (85% data subsets)**: Prediction correlation 0.945 vs base 0.948 — negligible diversity gain

### Ensemble/Diversity Strategies (ALL KILLED)
- **20 models instead of 10**: LB 0.2884 vs 0.2885 (literally no improvement)
- **Greedy forward selection** (pick models that maximize val): LB 0.2856-0.2862 (WORSE than flat average of 0.2885)
- **Recipe diversity** (vary LR/WD/dropout/scheduler):
  - varA (mild reg): mean val 0.2662, pred corr with base 0.942 (no meaningful diversity)
  - varB (cosine schedule): mean val 0.2663, pred corr with base 0.943 (no diversity)
  - varC (Huber loss): CATASTROPHIC val 0.2438
- **LSTM diversity**: -0.011 vs GRU, too weak for ensemble inclusion
- **Checkpoint diversity** (different epoch snapshots): Near-peak corr 0.97 (useless). Early epochs diverse (0.87) but val -0.014 (too weak).
- **Pearson blend loss mixing**: Genuine diversity (0.78 pred corr) but WRONG DIRECTION on LB — mixed ensemble 0.2868 vs pure 0.2885
- **Key lesson**: ONLY same-recipe flat averaging helps LB. Any mixing, selection, or filtering HURTS.

### Inference Modifications (ALL KILLED)
- **Windowed inference** (last 100 steps with fresh hidden state): Same scores as step-by-step on LB

---

## Key Insights and Patterns

1. **Simplicity IS the regularization.** The -0.034 LB gap from our complex pipeline vs +0.017 from vanilla is entirely from architectural complexity. The GRU is powerful enough to learn what it needs from raw features.

2. **The test set is EASIER than validation** for simple models. This positive gap phenomenon means overfitting to val-specific patterns is the main risk.

3. **Ensemble averaging helps but only from same recipe.** The positive gap scales with ensemble size (single +0.004, 10-model +0.018). But mixing different recipes or selecting subsets HURTS.

4. **h=64 is the sweet spot.** Not a budget constraint — larger models (128/144/192) are actively worse even on validation.

5. **ALL feature engineering hurts.** The data is pre-scaled, the GRU learns what it needs from raw features. Top competitors confirm "almost all financial features do not provide an increase."

6. **"99% pure ML"** — the top competitors likely have better model architectures or training paradigms, not better features.

7. **t1 is the bottleneck.** Our best t1: ~0.17, t0: ~0.47. The metric averages both, so t1 improvement has outsized impact.

8. **Prediction correlation between same-recipe seeds: 0.94+.** True diversity hasn't been achieved except with Pearson loss (0.78 corr), but that diversity was in the wrong direction.

9. **Train-val distribution shift is real.** KS stats up to 0.33 in price features (p0, p1, p6, p7). Models that overfit to training distribution fail on val/test.

10. **Target autocorrelation is very high.** t0 lag-1=0.715, t1 lag-1=0.976. Large-move clustering is substantial (12-17x lift).

---

## Available Resources

### Compute
- **Training**: Google Colab (T4 GPU) or Kaggle (T4 GPU), ~12 min per seed
- **Scoring**: 1 vCPU, 16GB RAM, 4200s timeout, no GPU
- **Budget**: 5 subs/day, ~1 month remaining (~150 submissions left)

### Available Trained Models (23 vanilla GRU seeds)
- Seeds 42-64, all h=64 3L vanilla GRU with MSE loss
- Val scores range: 0.2624 to 0.2737 (mean 0.2689)
- Top seeds by val: s43(0.2737), s59(0.2727), s46(0.2716), s63(0.2713), s55(0.2705)
- All checkpoints available for distillation/analysis

### ONNX Runtime
- GRU ONNX: ~42s per model (2.5x faster than PyTorch ~170s), confirmed lossless
- Can fit 40+ ONNX models in time budget vs 10 PyTorch

---

## Codex's 7 Untested Strategic Ideas (For Reference)

1. **Regime-gated experts**: Train identical GRUs on different data regimes (volatility/spread/imbalance), use tiny gating model to weight predictions
2. **Adversarial-validation density-ratio weighting**: Train classifier to detect train-vs-val distribution shift, use predicted probability as sample weight during training
3. **Mega-teacher distillation**: Train 40-100 models offline, average their predictions as soft targets, distill into 1-2 student models
4. **Tree model blend**: LightGBM/CatBoost on sequence summaries or rolling features, blend if low correlation with GRU
5. **Prediction neutralization**: Post-process predictions to remove unstable exposures (regress out nuisance basis functions)
6. **Variance-penalized stacking**: Optimize ensemble weights by `max(mean_fold_corr - beta * std_fold_corr)` instead of greedy val maximization
7. **Chunk-wise inference calibration**: Per-sequence demeaning/winsorizing at test time to reduce prediction bias

---

## Attached Files Reference

| File | Description |
|------|-------------|
| `solution.py` | Current best inference code (10-model vanilla GRU ensemble) |
| `gru_baseline.py` | Full GRU model architecture (vanilla/LSTM/CVML modes) |
| `config.yaml` | Winning training config (vanilla h=64, raw32, MSE) |
| `losses.py` | Loss function factory (MSE, Huber, Pearson, Combined, etc.) |
| `utils.py` | Official scoring metric (weighted Pearson) + DataPoint/Scorer |
| `data_profile.txt` | Validation set feature/target statistics |
| `PROMPT.md` | The Gemini Deep Research prompt |
