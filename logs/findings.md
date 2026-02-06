# Findings Log

## 2026-02-05: Project Initialization

**Source**: WUNDERFUND_PROJECT_SPEC.md, README.md, utils.py

**Key Insights**:
1. Weighted Pearson Correlation emphasizes large price movements (weight = |target|)
2. Online inference constraint: model sees one step at a time, must maintain hidden state
3. Must reset hidden state when seq_ix changes
4. Warm-up steps (0-98) build context but don't need predictions (need_prediction=False)
5. 32 features organized as:
   - Bid prices (p0-p5), Ask prices (p6-p11)
   - Bid volumes (v0-v5), Ask volumes (v6-v11)
   - Trade prices (dp0-dp3), Trade volumes (dv0-dv3)
6. Two targets (t0, t1) representing different price movement indicators
7. Predictions clipped to [-6, 6] during evaluation

**Feature Engineering Ideas**:
- Bid-ask spread at each level
- Order book imbalance ratio
- Volume pressure (bid vs ask ratio)
- Price momentum within sequence

**Action**: Implement GRU baseline following these constraints, then iterate

---

## 2026-02-05: GRU Baseline Results

**Source**: Kaggle training (Tesla T4 GPU), configs/gru_baseline.yaml

**Architecture**: 2-layer GRU, hidden_size=128, dropout=0.2, LayerNorm, 211K parameters
**Loss**: CombinedLoss (50% MSE + 50% WeightedMSE)
**Training**: AdamW lr=0.001, ReduceLROnPlateau, batch_size=256, AMP enabled

**Results**:
- Val avg: **0.2578** (best at epoch 8)
- Val t0: **0.3869** (stronger target)
- Val t1: **0.1286** (weaker target)
- Early stopped at epoch 18 (patience=10)
- Total training time: 62 seconds (~3.4s per epoch)
- Final LR: 0.000125 (decayed 3x from initial)

**Key Observations**:
1. **t0 is much easier to predict than t1** — 3x higher correlation. This is the biggest finding so far.
2. Early stopping triggered well — model was clearly overfitting by epoch 18
3. LR decay helped (reduced 3x), but best epoch was still early (epoch 8)
4. CombinedLoss with 0.5 weighting seems reasonable as starting point

**Open Questions for Codex**:
1. What makes t0 vs t1 different? Feature correlation analysis needed
2. Are there features that are highly predictive for t1 specifically?
3. What's the distribution of |target| weights? (determines competition metric sensitivity)
4. Is there autocorrelation in the targets that the GRU can exploit?
5. Would separate prediction heads with different loss weights for t0/t1 help?

**Action**: Codex should start EDA + feature analysis. Claude Code should build sweep infrastructure and LSTM model.

## [2026-02-06] EDA Completed: Data Integrity + Scale
**Source**: notebooks/01_eda.ipynb, notebooks/artifacts/01_eda/eda_summary.json
**Result**: Data quality is clean (0 NaN, 0 inf), every sequence is exactly 1000 steps, and feature scales are heterogeneous (train feature std max/min ratio = 7.20).
**Action**: Keep strict train-fit normalization in `src/data/preprocessing.py`; add optional per-sequence centering ablation for price/volume features.

## [2026-02-06] Train/Valid Distribution Shift Is Material
**Source**: notebooks/01_eda.ipynb, notebooks/artifacts/01_eda/feature_shift_train_vs_valid.csv
**Result**: Strong train-valid shift in several price features (`p6`, `p0`, `p7`, `p1`, `p3`) with KS stats up to 0.3335 and mean shifts near 1 train std.
**Action**: Add robustness-focused validation in training scripts (shift-aware diagnostics by feature percentile buckets) and prioritize regularization/early-stopping settings resilient to shift.

## [2026-02-06] Target Structure: Correlated but Asymmetric
**Source**: notebooks/01_eda.ipynb, notebooks/artifacts/01_eda/eda_summary.json
**Result**: Targets are moderately related (`corr(t0,t1)` train=0.505, valid=0.440) but raw linear signal differs: top |corr| feature for `t0` is 0.136 (`p0`) vs only 0.036 for `t1` (`v8`).
**Action**: Keep shared backbone, but implement target-specific output heads and tune per-target loss weighting to improve `t1` without hurting `t0`.

## [2026-02-06] Temporal Dependence Is Very Strong (Especially t1)
**Source**: notebooks/01_eda.ipynb, notebooks/artifacts/01_eda/target_autocorrelation.csv
**Result**: High target autocorrelation (`t0` lag-1=0.715, `t1` lag-1=0.976), and large-move clustering is substantial (conditional next-step large-move lift: 12.37x for `t0`, 17.34x for `t1`).
**Action**: Test longer-memory configurations (larger hidden size/sequence handling) and add temporal-consistency auxiliary loss options in experiments.

## [2026-02-06] Metric Weight Concentration + Derived Feature Signal
**Source**: notebooks/01_eda.ipynb, notebooks/artifacts/01_eda/derived_feature_target_corr.csv
**Result**: Weighted metric is concentrated: top 20% of |target| contributes ~50-56% of total weight; derived spreads improve `t0` linear signal (`spread_2` corr=0.145 > best raw 0.136), while `t1` gains are modest.
**Action**: Implement optional derived spread/trade-intensity channels and run weighted-loss sweeps emphasizing high-|target| samples, with separate monitoring for `t0`/`t1`.

---

## [2026-02-06] Phase 2 Experiment Results — Architecture & Config Comparison

**Source**: Kaggle training (Tesla T4 GPU), 5 configs tested

### Full Results Table

| Config | Val Avg | t0 | t1 | Best Epoch | Params | Notes |
|--------|---------|------|------|------------|--------|-------|
| gru_baseline | 0.2578 | 0.3869 | 0.1286 | 8 | 211K | Original baseline |
| **gru_derived_v1** | **0.2614** | **0.3912** | **0.1316** | 7 | 212K | **BEST — derived features help** |
| gru_derived_t1focus_v1 | 0.2536 | 0.3854 | 0.1218 | 9 | 212K | target_weights [0.35, 0.65] HURT |
| gru_long_memory_derived_v1 | 0.2609 | 0.3869 | 0.1348 | 4 | 694K | Big model overfits fast |
| lstm_derived_v1 | 0.2542 | 0.3836 | 0.1247 | 11 | 278K | LSTM underperforms GRU |

### Key Findings

1. **Derived features provide a small but real improvement** (+0.0036 avg over baseline). The 10 extra features (spreads, trade_intensity, bid/ask pressure, pressure_imbalance) add signal without overfitting. This is the ONLY config that improved over baseline.

2. **Target-specific loss weights HURT performance**. `gru_derived_t1focus_v1` used `target_weights: [0.35, 0.65]` to emphasize t1, but BOTH targets got worse (t0: 0.3854 vs 0.3912, t1: 0.1218 vs 0.1316). The equal-weight CombinedLoss is better.

3. **Bigger models overfit faster without improving**. `gru_long_memory_derived_v1` (694K params, hidden=192, 3 layers) peaked at epoch 4 with 0.2609 avg — worse than the smaller 212K model. More capacity ≠ better generalization on this data.

4. **LSTM underperforms GRU** despite theoretical advantage of cell state for long-range memory. 0.2542 avg is worse than both GRU configs. The forget-gate mechanism doesn't help on these LOB sequences.

5. **t1 remains the bottleneck**. Best t1 score is 0.1348 (from the big GRU) but that model has worse avg. The 0.1316 from gru_derived_v1 is the best t1 from a competitive model. The ~3x gap between t0 and t1 persists across all architectures.

6. **Early stopping triggers early** (epochs 4-11 across configs). Models learn quickly but overfit quickly too. The short optimal training window suggests strong regularization is key.

### What Didn't Work
- Per-target loss weighting (any ratio other than equal)
- More layers (3 vs 2)
- More hidden units (192 vs 128)
- LSTM architecture

### What Worked
- Derived features (+0.0036 avg)
- Equal-weight CombinedLoss (MSE + WeightedMSE at 0.5/0.5)
- Moderate model size (2 layers, hidden=128, 212K params)
- Aggressive early stopping

### Open Questions for Next Round
1. **Would a learning rate sweep improve gru_derived_v1 further?** Current lr=0.001, try 0.0005-0.003 range
2. **Does dropout tuning matter?** Current 0.2, try 0.1-0.4
3. **Are there better derived features?** Current 10 are basic. Rolling statistics, rate-of-change features?
4. **Would sequence-level augmentation help?** (e.g., add noise, feature dropout)
5. **Is t1 predictable at all or is it noise?** Need correlation analysis of t1 with lag features
6. **Would a Transformer (causal attention) capture different patterns than RNN?**

**Action**: Run hyperparameter sweeps on gru_derived_v1 (lr, dropout, batch_size). Codex to do deep feature analysis + suggest next feature engineering round.

## [2026-02-06] Feature Analysis: Derived Feature Importance + t1 Deep Dive
**Source**: notebooks/02_feature_analysis.ipynb, notebooks/artifacts/02_feature_analysis/*
**Result**:
- Among current derived features, the strongest contributors are `spread_2`, `trade_intensity`, `spread_0`, and `ask_pressure`.
- Permutation importance (derived-only proxy model) shows largest avg score drops for `spread_2` (0.0289), `trade_intensity` (0.0235), and `spread_0` (0.0148).
- `t1` remains weakly linear overall, but the best stable signals are `spread_0` (valid corr 0.0397), `trade_intensity` (0.0285), `v8` (0.0282), and `dv3` (0.0279).
- Cross-feature interactions add signal for `t1` (`v8*p0`, `spread_0*p0`, `spread_0*v2` were top interaction terms).
- Candidate next-round temporal features show promise for `t1`: `spread0_roc1` (valid corr 0.0521), `spread0_roc5` (0.0366), and `trade_intensity_roll_mean_5` (0.0243).
**Action**: Add the top temporal derivative features (`spread0_roc1`, `spread0_roc5`, `trade_intensity_roll_mean_5`) as optional engineered channels in `src/data/preprocessing.py` behind config flags, then run ablation against current 42-feature baseline.

## [2026-02-06] HP Optimization Analysis: Early Peak Behavior + Sweep Ranges
**Source**: notebooks/03_hp_optimization.ipynb, notebooks/artifacts/03_hp_optimization/*
**Result**:
- Best config remains `gru_derived_v1` at val avg 0.2614.
- Across the 5 key experiments, best epochs are consistently early (range 4-11), indicating fast fit + early overfit rather than undertraining.
- No per-epoch curve files were persisted for these runs; analysis used `best_epoch`, `epochs_trained`, and score outcomes as curve proxies.
- Recommended sweep ranges from observed outcomes:
  - `lr`: 0.0007 to 0.0015
  - `dropout`: 0.15 to 0.28
  - `batch_size`: 128 to 320
  - `weight_decay`: 1e-5 to 8e-5
- Regularization readout: feature dropout and moderately higher weight decay are promising; label smoothing is not a primary fit for this regression objective.
**Action**: Run focused GRU+derived sweeps in the recommended ranges and persist per-epoch history in future runs to enable exact curve diagnostics.
