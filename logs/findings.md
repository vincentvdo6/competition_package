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
