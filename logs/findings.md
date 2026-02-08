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

---

## [2026-02-06] V2 Config Results + Temporal Features

**Source**: Kaggle training runs

### Full Results Table (all experiments to date)

| Config | Val Avg | t0 | t1 | Best Epoch | Params | Notes |
|--------|---------|------|------|------------|--------|-------|
| gru_baseline | 0.2578 | 0.3869 | 0.1286 | 8 | 211K | Original |
| gru_derived_v1 (run 1) | 0.2614 | 0.3912 | 0.1316 | 7 | 212K | Derived +0.0036 |
| gru_derived_v1 (run 2) | 0.2608 | 0.3841 | 0.1374 | 7 | 212K | Variance: 0.0006 |
| gru_derived_t1focus_v1 (run 1) | 0.2536 | 0.3854 | 0.1218 | 9 | 212K | target_weights HURT |
| gru_derived_t1focus_v1 (run 2) | 0.2611 | 0.3909 | 0.1314 | 7 | 212K | Variance: 0.0075! |
| gru_long_memory_derived_v1 | 0.2609 | 0.3869 | 0.1348 | 4 | 694K | Overfits fast |
| lstm_derived_v1 | 0.2542 | 0.3836 | 0.1247 | 11 | 278K | LSTM < GRU |
| gru_temporal_v1 | 0.2512 | 0.3863 | 0.1160 | 7 | 212K | Temporal HURT |
| gru_derived_reg_v2 | 0.2660 | 0.3967 | 0.1352 | 8 | 212K | Strong reg helps |
| **gru_derived_tightwd_v2 (run 1)** | **0.2674** | 0.3955 | 0.1394 | 9 | 267K | **BEST LOCAL** |
| gru_derived_onecycle_v2 | 0.2566 | 0.3804 | 0.1327 | 10 | 212K | LR too aggressive |
| gru_derived_tightwd_v2 (run 2) | 0.2579 | 0.3839 | 0.1318 | 7 | 267K | Variance: 0.0095! |

### Key Findings

1. **Regularization is the #1 lever**: higher dropout (0.22-0.24), higher WD (4-5e-5), faster LR decay → best results. tightwd_v2 and reg_v2 are top 2.
2. **hidden_size=144 helps** when paired with strong regularization. 267K params with dropout 0.22 + WD 5e-5 beats 212K.
3. **Temporal features HURT** (0.2512 vs 0.2614). Lower loss but worse correlation — GRU hidden state already captures temporal patterns.
4. **OneCycleLR too aggressive**: max_lr=lr*10 overshoots. ReduceLROnPlateau is better.
5. **Run-to-run variance is huge**: tightwd_v2 gave 0.2674 then 0.2579 (0.0095 gap). This makes seed ensembling critical.
6. **First submission**: tightwd_v2 single model → LB 0.2580, rank 338/4598.

**Action**: Implement seed-controlled training + ensemble framework. Train 5 seeds of tightwd_v2, average for more stable score. Target: top 100 via ensemble + architecture diversity.

## [2026-02-07] Wave 1 Setup: Seed Diversity Notebook + Runner
**Source**: notebooks/04_seed_diversity_analysis.ipynb, notebooks/run_04_seed_diversity_analysis.py
**Result**:
- Implemented full Wave 1 seed-diversity pipeline with strict online inference semantics:
  - reset hidden state on new `seq_ix`
  - process all timesteps including warm-up
  - emit predictions only when `need_prediction=True`
  - apply derived features before normalization
- Runner generates required outputs: `model_scores.csv`, `pairwise_correlations.csv`, `optimal_weights.json`, `ensemble_vs_n_models.csv`, `diversity_by_target_bucket.csv`, plus plots.
- Current local run is blocked because `logs/gru_derived_tightwd_v2_seed42..46.pt` checkpoints are not present yet; `status.json` + placeholder artifact files were written under `notebooks/artifacts/04_seed_diversity/`.
**Action**: Once 5 seed checkpoints are available locally, rerun `python notebooks/run_04_seed_diversity_analysis.py` to populate full metrics/plots and finalize ensemble weights for submission.

## [2026-02-07] Ensemble + Architecture Infrastructure Implemented (Wave 1/2)
**Source**: `scripts/export_ensemble.py`, `src/models/gru_attention.py`, `src/data/preprocessing.py`, `src/data/dataset.py`, `scripts/train.py`, `scripts/evaluate.py`, `src/training/trainer.py`
**Result**:
- Implemented `GRUAttentionModel` with online-safe rolling attention context (`attention_window`) and `forward_step` support.
- Added interaction feature pipeline (`v8*p0`, `spread_0*p0`, `spread_0*v2`) end-to-end through data preprocessing, dataset loading, train/eval scripts, and export code.
- Upgraded `scripts/export_ensemble.py` to support:
  - heterogeneous ensembles (`--config` or `--configs`)
  - per-model preprocessing flags from config (`derived/temporal/interaction`)
  - per-model normalizers (`--normalizer` or `--normalizers`)
  - global weights (`--weights`) and per-target weights (`--weights-t0` + `--weights-t1`)
  - optional weights import from JSON (`--weights-json`)
- Added `scripts/build_wave1_candidates.py` to generate Candidate A/B/C zips directly from `notebooks/artifacts/04_seed_diversity/*`.
- Upgraded training reproducibility artifacts:
  - per-seed normalizer saved as `logs/normalizer_<config>_seed<seed>.npz`
  - per-run history saved as `logs/training_history_<checkpoint_prefix>.json`
- Added Wave 2 training configs:
  - `configs/gru_interaction_tightwd_v1.yaml`
  - `configs/gru_attention_interaction_v1.yaml`
  - `configs/gru_attention_interaction_v1b.yaml`

**Validation**:
- `forward` vs `forward_step` parity (GRU baseline): max abs diff = `4.768e-07`
- Hidden reset test:
  - with reset: max diff vs fresh sequence = `0.0`
  - without reset: max diff vs fresh sequence = `3.479`
- Preprocessing parity (batch vs step, derived+temporal+interaction): max abs diff = `2.384e-07`
- Ensemble exporter smoke-tested for:
  - uniform/global weights
  - per-target weights
  - generated `solution.py` runtime inference
- Added `scripts/validate_online_parity.py` to rerun these parity checks before export/submission.

**Action**:
- Sync Kaggle seed checkpoints locally, then run Wave 1 script to compute real `optimal_weights.json`.
- Export and submit A/B/C candidates using `best_subset`, `optimal_weights`, and shrinked-optimal rules from plan.

## [2026-02-07] Wave 1 Leaderboard Results — Seed Ensemble Submissions

**Source**: wundernn.io competition leaderboard

### Results

| Submission | LB Score | Strategy |
|-----------|----------|----------|
| submission.zip | 0.2580 | Single tightwd_v2 seed42 |
| ensemble top3 uniform.zip | 0.2604 | Top 3 seeds, uniform weights |
| ensemble weighted5 v1.zip | 0.2607 | All 5 seeds, optimized weights |
| **ensemble.zip** | **0.2614** | All 5 seeds, uniform weights |

### Key Findings

1. **Seed ensembling delivers +0.0034 LB improvement** (0.2580 → 0.2614). Confirms that reducing seed variance is the right strategy.
2. **Uniform averaging > optimized weights on LB**. The simple 5-seed uniform ensemble (0.2614) beat the optimized-weight variant (0.2607). Optimized weights likely overfit to local val noise.
3. **More models > fewer models**. 5-seed uniform (0.2614) > 3-seed uniform (0.2604). Including weaker seeds still helps via variance reduction.
4. **LB vs local val gap persists**. Best local val was 0.2674 (single seed), LB is 0.2614 (5-seed ensemble). The train→valid distribution shift noted in EDA is real and hurts.

### Implications for Wave 2
- Architecture diversity (GRU+Attention) should provide more ensemble lift than more GRU seeds, since GRU seed predictions are correlated.
- Keep uniform weighting as the default — don't over-optimize ensemble weights.
- The 0.2614 LB is the new baseline to beat. Target: 0.2650+ with diverse ensemble.

**Action**: Train Wave 2 configs (gru_attention_interaction_v1, gru_refined_v3) on Kaggle. Fold best into diverse ensemble with the 5 GRU seeds.

## [2026-02-07] Wave 2 Ensemble Result — New Best LB

**Source**: wundernn.io competition leaderboard

### Result

| Submission | LB Score | Composition |
|-----------|----------|-------------|
| ensemble.zip | 0.2614 | 5x GRU tightwd seeds, uniform |
| **gru5_plus_attention2_balanced7030.zip** | **0.2633** | 5x GRU tightwd + 2x attention_clean, family-balanced 70/30 |

### Key Findings

1. Adding architecture diversity improved LB by **+0.0019** over the 5-seed GRU baseline (0.2614 -> 0.2633).
2. A conservative family balance (GRU-heavy at 70%) was effective and reduced downside from weaker attention seeds.
3. Attention models are useful as ensemble diversifiers even when their per-seed variance is high.

### Action

- Keep `gru5_plus_attention2_balanced7030` as the new safe baseline.
- Next submission candidates should stay in the same mixed-family direction (small weight adjustments, or improved attention family members) rather than pure GRU weight tuning.

## [2026-02-07] Phase 1 Infrastructure: Metric-Aligned Pearson Loss + Bootstrap Scorer

**Source**: Plan discussion + implementation

### Changes

1. **WeightedPearsonLoss** added to `src/evaluation/metrics.py`:
   - Differentiable weighted Pearson correlation matching competition metric exactly
   - weights = |target|, predictions clipped to [-6, 6], mask support for warm-up exclusion
   - Stability: variance floor, finite checks, handles degenerate batches

2. **PearsonCombinedLoss** added to `src/evaluation/metrics.py`:
   - Hybrid: `alpha * CombinedLoss + (1-alpha) * (1 - weighted_corr_avg)`
   - Default alpha=0.6 (60% CombinedLoss for stable gradients + 40% Pearson for metric alignment)

3. **Loss factory** updated in `src/training/losses.py`:
   - `loss: weighted_pearson` — pure Pearson loss
   - `loss: pearson_combined` — hybrid (recommended for training stability)
   - Config fields: `pearson_alpha`, `pearson_eps`

4. **New configs**:
   - `configs/gru_pearson_v1.yaml` — tightwd_v2 base + pearson_combined, lr=0.0005
   - `configs/gru_attention_pearson_v1.yaml` — attention_clean base + pearson_combined, lr=0.0005

5. **Bootstrap candidate scorer** — `scripts/score_ensemble_candidates.py`:
   - Online step-by-step inference matching competition semantics
   - Sequence-level bootstrap resampling (configurable, default 200)
   - Outputs: ranked table with delta vs champion, recommendations for conservative + upside slots

### Validation
- All loss functions pass sanity checks: finite output, non-zero gradients, perfect predictions → loss near -1
- Both configs load correctly: model + loss instantiate without errors
- Loss factory backward-compatible with all existing loss types

### Action
- Commit and push for Kaggle training
- Train gru_pearson_v1 seeds 42-44 first (fast signal), then attention seeds 45-46, then attention_pearson if GRU Pearson promising
- Gate: val_avg >= 0.2620 for pool inclusion
