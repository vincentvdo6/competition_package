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

## [2026-02-07] GRU Pearson v1 Training Results

**Source**: Kaggle training (Tesla T4 GPU), configs/gru_pearson_v1.yaml, seeds 42/43/44

### Results

| Seed | Val Avg | Val t0 | Val t1 | Best Epoch |
|------|---------|--------|--------|------------|
| 42 | 0.2595 | 0.3591 | 0.1599 | 10 |
| **43** | **0.2652** | **0.3718** | **0.1586** | 11 |
| 44 | 0.2610 | 0.3624 | 0.1596 | 10 |
| **Avg** | **0.2619** | 0.3644 | 0.1594 | — |

### Key Findings

1. **Seed 43 passes gate** (0.2652 >= 0.2620) — enters ensemble candidate pool.
2. **Seeds 42/44 fail strong gate** (0.2595, 0.2610 < 0.2620). Seed 44 may qualify under conditional gate (t1=0.1596 is strong).
3. **Average 0.2619 is borderline** — essentially equivalent to tightwd_v2 average. Pearson loss is NOT a clear lift for GRU.
4. **High seed variance** (0.0057 range, vs ~0.0095 for tightwd_v2). Pearson training variance is moderate.
5. **Loss spikes** observed (occasional jumps to 10-12 range) — gradient instability from correlation term. Alpha=0.6 blend keeps training viable but not perfectly smooth.
6. **t1 degrades sharply post-peak** (e.g., seed 42: 0.1345→0.0593 between epochs 8-18). Best epoch is consistently 10-11.
7. **t1 values notably higher than tightwd_v2** (~0.159 avg vs ~0.135 avg for tightwd_v2). Pearson loss may help t1 specifically.
8. **t0 values lower than tightwd_v2** (~0.364 avg vs ~0.390 avg). The Pearson term pulls resources from t0 toward t1 equalization.

### Verdict
- Seed 43 is a strong individual model — include in candidate pool
- The primary value is **diversity** for ensembling, not raw score lift
- Proceed with attention Pearson training — attention's richer capacity may benefit more from metric alignment
- t1 improvement is notable and may help ensemble t1 specifically

### Action
- Include seed 43 in candidate pool
- Run attention Pearson (Cell 4) on Kaggle
- Await attention clean seeds 45/46 results and gru5_attn3_uniform8 LB score

---

## [2026-02-14] BREAKTHROUGH: Parity Audit — Vanilla GRU Paradigm

**Source**: Parity testing of official baseline vs our complex pipeline

### The Discovery
- **Official baseline**: h=64 3L plain GRU, raw 32 features, no normalization. Val=0.2595, LB=**0.2761** (gap **+0.017**)
- **Our baseline_match**: same h=64 3L + input_proj+LayerNorm+MLP+derived+zscore. Val=0.2738, LB=**0.2394** (gap **-0.034**)
- **The entire -0.034 gap is from architectural complexity, not data or training**

### Parity v1 Config
Plain GRU + linear output, raw 32 features, no normalization, MSE loss, dropout=0:
- Mean val: **0.2692** across seeds
- **Seed 43 LB: 0.2814** (val 0.2737, gap +0.0077). **NEW PB at the time.**

### Key Insight
**Simplicity IS the regularization.** The test set is EASIER than validation for simple models. Complex models overfit to training-specific patterns that don't transfer.

---

## [2026-02-14-15] Vanilla GRU Scaling & Ensemble

### Scaling Results (ALL KILLED)
| Config | Mean Val | Delta vs h=64 |
|--------|---------|---------------|
| h=64, 3L (base) | 0.2692 | baseline |
| h=128, 3L | 0.2676 | -0.0016 |
| h=144, 3L | 0.2663 | -0.0029 |
| h=192, 3L | 0.2672 | -0.0020 |
| h=192, 2L | 0.2655 | -0.0037 |

**h=64 IS the sweet spot.** Larger models are strictly worse even on validation.

### Mass Seed Training (23 seeds: s42-s64)
- Val range: 0.2624 to 0.2737
- Mean: 0.2689, Top-10 mean: 0.2704

### Ensemble LB Results
| Ensemble | Val | LB | Gap |
|----------|-----|-----|-----|
| Single s43 | 0.2737 | 0.2814 | +0.0077 |
| Single s59 | 0.2727 | 0.2764 | +0.0037 |
| **10-model flat avg** | **0.2708** | **0.2885** | **+0.0177** |
| 20-model ONNX | ~0.2710 | 0.2884 | — |

**10-model ensemble = 0.2885 LB, Rank 73/4728.** 20 models gave no improvement.

---

## [2026-02-15-16] Diversity Strategies — ALL KILLED

### Recipe Diversity
| Variant | Mean Val | Correlation with base |
|---------|---------|----------------------|
| Base (MSE, LR=1e-3, WD=0, drop=0) | 0.2689 | 1.000 |
| varA (LR=5e-4, WD=1e-5, drop=0.05) | 0.2662 | 0.942 |
| varB (LR=2e-3, cosine) | 0.2663 | 0.943 |
| varC (Huber delta=1.0) | 0.2438 | CATASTROPHIC |

**Prediction correlation 0.94+ — no meaningful diversity.** varC catastrophic.

### LSTM Diversity
- Mean val: 0.2576 (-0.011 vs GRU). Seeds: 0.2636, 0.2588, 0.2584, 0.2580, 0.2493
- **KILLED** — too weak for ensemble inclusion

### Checkpoint Diversity
- Near-peak epochs: corr 0.97 with best (useless)
- Early epochs: corr 0.87 but val -0.014 (too weak)
- Within-seed smoothing: +0.0005 (not worth complexity)

### Greedy Ensemble Selection
- **greedy_top5_onnx**: val 0.2802 but LB **0.2862** (WORSE than flat-10's 0.2885)
- **greedy_top3_onnx**: val 0.2830 but LB **0.2856** (even WORSE)
- **Greedy val-optimization HURTS LB.** Positive gap is from averaging, not selection.

### Pearson Blend Loss Diversity
- Trained vanilla GRU with 70% MSE + 30% Pearson loss
- **Genuine diversity achieved**: pred correlation 0.78 with base (vs 0.94 base-base)
- **mixed_ens11_onnx** (10 base + 1 blend): LB **0.2868** — BELOW PB 0.2885
- **Diversity is real but in the WRONG DIRECTION.** Different predictions != better predictions.

### Bagging (85% data subsets)
- Prediction correlation 0.945 vs base 0.948 — negligible diversity gain

---

## [2026-02-16] Strategic Dead End — All Approaches Exhausted

All Codex-agreed strategies have been tested and killed. The ceiling at 0.2885 appears to be a fundamental limit of our current approach (vanilla GRU h=64 + flat ensemble averaging).

### Codex's 7 New Ideas (untested)
1. Regime-gated experts (different data regimes)
2. Adversarial-validation density-ratio weighting
3. Mega-teacher distillation (40-100 models -> 1-2 students)
4. Tree model blend (LightGBM/CatBoost)
5. Prediction neutralization (post-processing)
6. Variance-penalized stacking
7. Chunk-wise inference calibration

### Gemini Context Package Created
Comprehensive context in `gemini_context/` for external strategic consultation. Awaiting Gemini analysis for fundamentally new directions.

---

## [2026-02-18] Top-50 Push: Reweighted Pearson-Blend Family (Scored, KILLED)

**Source**: `cache/vanilla_preds`, `submissions/ready/*.zip`, checkpoint fingerprinting of `submissions/ready/vanilla_ens10.zip`

### Verified Baseline Identity
- Hash-matched the exact `vanilla_ens10` members to:
  - `gru_parity_v1_seed43, 59, 46, 63, 55, 61, 50, 54, 57, 45`
- This confirms all comparisons below use the true PB base family (not a proxy subset).

### New Signal
- `vanilla_pearson_blend_seed44` is both:
  - **Strong** single model (local val `0.2804`)
  - **Diverse** vs base10 (prediction correlation `0.762`)
- Prior failed test `mixed_ens11_onnx` (LB `0.2868`) used only 1/11 uniform blend weight (~9.1%) and likely under-used this direction.

### Local Validation Sweep (base10 + pearson_blend_seed44)
| Blend Weight on seed44 | Local Val Avg |
|---|---|
| 0% (base10) | 0.2790 |
| 10% | 0.2863 |
| 20% | 0.2899 |
| 30% | 0.2911 |

All gains came mainly from higher `t1` correlation while keeping `t0` stable.

### Submission Artifacts Built
- `submissions/ready/top50_mix_p44_w10_onnx.zip` (10% blend weight)
- `submissions/ready/top50_mix_p44_w20_onnx.zip` (20% blend weight)
- `submissions/ready/top50_mix_p44_w30_onnx.zip` (30% blend weight)

All are ONNX, ~2.8MB each, estimated server runtime ~814s (large safety margin under 4200s).

### Leaderboard Results
| Submission | LB Score | Delta vs PB (0.2885) |
|---|---:|---:|
| `top50_mix_p44_w10_onnx.zip` | 0.2866 | -0.0019 |
| `top50_mix_p44_w20_onnx.zip` | 0.2829 | -0.0056 |
| `top50_mix_p44_w30_onnx.zip` | 0.2784 | -0.0101 |

### Verdict
- Strongly confirms prior pattern: Pearson-blend diversity remains in the wrong LB direction.
- Monotonic decay with higher blend weight indicates this family should be treated as **KILLED** for further LB submissions.

---

## [2026-02-18] Regime-Gated Experts — Infrastructure Built, LB Pending

**Source**: `scripts/build_regime_gated_submission.py`, `logs/regime_gate_warm99_logreg.json`

### What Was Implemented
- Added `scripts/build_regime_gated_submission.py`:
  - Exports ONNX models for a **base ensemble** and **specialist ensemble**
  - Fits (or loads) a warmup-only adversarial logistic gate using steps `0..98`
  - Generates `solution.py` that:
    - tracks running mean/std/min/max on raw 32 features during warmup
    - computes `p_val_like` once per sequence
    - applies piecewise specialist weight (`low` / `high`) by threshold
    - blends base and specialist predictions per sequence

### Gate Fitting Snapshot
- Warmup gate (`step_in_seq < 99`) train-vs-valid logistic AUC (in-sample): **0.9175**
- Valid probability quantiles from saved gate params:
  - q20: `0.136645`
  - q50: `0.494322`
  - q80: `0.883797`

### Candidate Artifacts Built
- `submissions/ready/regime_gate_blend44_q20_w10_onnx.zip`
  - threshold `0.136645`, specialist weight `0.10` below threshold else `0.00`
- `submissions/ready/regime_gate_blend44_q50_w10_onnx.zip`
  - threshold `0.50`, specialist weight `0.10` below threshold else `0.00`

Both use:
- Base expert: verified PB `vanilla_ens10` seed set
- Specialist expert: `vanilla_pearson_blend_seed44`
- Runtime estimate: ~814s (11 ONNX models), size ~2.83MB

### Additional Observation
- Finetune specialist (`gru_parity_v1_finetune_seed42/43/44`) showed near-zero gating signal in local gate sweeps and was not selected for these first regime-gated LB tests.

---

## [2026-02-18] Regime-Gated Blend44 v1 Results + Tomorrow Pack Built

### Leaderboard Results (first regime-gated attempts)
| Submission | LB Score | Delta vs PB (0.2885) |
|---|---:|---:|
| `regime_gate_blend44_q20_w10_onnx.zip` | 0.2874 | -0.0011 |
| `regime_gate_blend44_q50_w10_onnx.zip` | 0.2871 | -0.0014 |

**Verdict**:
- Sequence gating with **same t0/t1 specialist weight** reduced damage vs static blend but still under PB.
- This specific formulation is not sufficient.

### New Capability Added
- Upgraded `scripts/build_regime_gated_submission.py` to support **per-target gate weights**:
  - independent `(t0, t1)` weights for low and high branches
  - enables `t1`-only specialist blending while keeping `t0` on stable base ensemble

### Tomorrow Submission Slate (5 zips ready)
All built in `submissions/ready/`:
- `tomorrow_t1only_blend44_a008_onnx.zip`
- `tomorrow_t1only_blend44_a012_onnx.zip`
- `tomorrow_t1only_blend44_a016_onnx.zip`
- `tomorrow_t1only_blend44_a022_onnx.zip`
- `tomorrow_t1only_blend44_a030_onnx.zip`

These are static `t1`-only blends of specialist `vanilla_pearson_blend_seed44`:
- `t0` weight = 0.0 (always base)
- `t1` weight = `alpha` in {0.08, 0.12, 0.16, 0.22, 0.30}
- Base ensemble remains the verified PB 10-seed set.

### Information-Optimized 5-Submission Plan (updated)
Instead of spending all 5 slots on a single static alpha ladder, use:

1. **Static magnitude probe (low)**: `tomorrow_t1only_blend44_a012_onnx.zip`
2. **Static magnitude probe (mid/high)**: `tomorrow_t1only_blend44_a022_onnx.zip`
3. **Static magnitude probe (high)**: `tomorrow_t1only_blend44_a030_onnx.zip`
4. **Regime-direction probe A**: `tomorrow_t1gate_blend44_q20_l025_h005_onnx.zip`
5. **Regime-direction probe B**: `tomorrow_t1gate_blend44_q80_l005_h025_onnx.zip`

Rationale:
- Slots 1–3 estimate the static `t1`-only response curve quickly (slope + saturation).
- Slots 4–5 hold near-similar average blend mass but flip where specialist weight is applied (low vs high `p_val_like`) to test whether regime assignment has directional value.

---

## [2026-02-20] Next Recovery Slate Built (Per-Target Vanilla GRU, ONNX)

### Latest Live Signal
Recent per-target submissions establish:
- `recovery_ptarget_t07_t17_onnx.zip` -> **0.2883** (best in this family, near PB 0.2885)
- `recovery_ptarget_t010_t17_onnx.zip` -> 0.2878
- `recovery_ptarget_t05_t13_onnx.zip` -> 0.2868

Interpretation:
- Adding extra `t0` members beyond top-7 hurt (`t010_t17` < `t07_t17`).
- Very aggressive `t1` pruning hurt (`t05_t13` < `t07_t17`).
- Best next EV is a tight neighborhood around `t07_t17`.

### Candidate Selection Method
- Used cached parity predictions in `cache/vanilla_preds/gru_parity_v1_seed*.npz`.
- Ranked seeds by per-target weighted Pearson on validation.
- Ran sequence-level bootstrap robustness around `t07_t17` on nearby `topK_t0/topK_t1` pairs.
- Prioritized robust neighbors instead of larger architectural changes.

### New Artifacts Built (ready to submit)
1. `submissions/ready/recovery_ptarget_t05_t17_onnx.zip`
2. `submissions/ready/recovery_ptarget_t07_t16_onnx.zip`
3. `submissions/ready/recovery_ptarget_t07_t15_onnx.zip`

All are ONNX vanilla GRU parity-only ensembles and are under 20MB:
- `recovery_ptarget_t05_t17_onnx.zip` -> 2.57MB
- `recovery_ptarget_t07_t16_onnx.zip` -> 2.83MB
- `recovery_ptarget_t07_t15_onnx.zip` -> 2.83MB

### Locked Submission Order
1. `recovery_ptarget_t05_t17_onnx.zip`
2. `recovery_ptarget_t07_t16_onnx.zip`
3. `recovery_ptarget_t07_t15_onnx.zip`

Why this order:
- Start with highest robustness-sweep EV (`t05_t17`).
- Then test mild and stronger `t1` trimming while keeping the strong `t0_top7` core fixed.

### Live Update (2026-02-20 17:33)
- `recovery_ptarget_t05_t17_onnx.zip` scored **0.2879** on public LB.
- Delta vs PB (`0.2885`): **-0.0006**
- Delta vs this branch best (`t07_t17 = 0.2883`): **-0.0004**

Interpretation:
- Reducing `t0` pool to top-5 did not improve over the `t0_top7` branch.
- Keep focus on `t0_top7` variants for remaining slots (`t07_t16`, then `t07_t15` if needed).

### Live Update (2026-02-20 17:46)
- `recovery_ptarget_t07_t16_onnx.zip` scored **0.2879** on public LB.
- Delta vs PB (`0.2885`): **-0.0006**
- Delta vs branch best (`t07_t17 = 0.2883`): **-0.0004**

Interpretation:
- Trimming `t1` from 7 to 6 did not help.
- Current evidence favors `t07_t17` as the local optimum in this family.

### Next Probe Prepared After 17:46 Result
- Built: `submissions/ready/recovery_ptarget_t07_t18_onnx.zip` (3.34MB)
- Design:
  - Keep `t0` fixed at top-7 (`50,63,59,55,43,46,57`)
  - Increase `t1` from top-7 to top-8 by adding seed `64`
- Purpose:
  - After `t07_t16` underperformed, test whether the local optimum is exactly at `t1_top7` or slightly right-shifted.

---

## [2026-02-20] Max-Leverage Workflow Implemented

### What Was Added
1. `scripts/submission_decision_engine.py`
   - Records live `zip -> score` updates.
   - Computes `delta = score - PB`.
   - Classifies score band:
     - `strong_win` (`delta >= +0.0003`)
     - `soft_win` (`0 <= delta < +0.0003`)
     - `near_miss` (`-0.0004 < delta < 0`)
     - `clear_fail` (`delta <= -0.0004`)
   - Enforces family logic:
     - two `clear_fail` => kill family, with one optional mechanistic final probe.
   - Persists branch state to `logs/submission_decision_state.json`.

2. `scripts/check_submission_zip.py`
   - Generic pre-submit gate for all zip formats.
   - Verifies:
     - size `<20MB`
     - `solution.py` at root
     - model files present
     - `PredictionModel` class exists
     - parseable model/weight summary (when literal configs are present)

3. `docs/max_leverage_submission_workflow.md`
   - Decision-complete runbook with exact commands.

4. `AGENTS.md` update
   - Locked defaults and banding rules now codified in repo playbook.

### Validation
- `check_submission_zip.py` PASS on:
  - `recovery_ptarget_t07_t16_onnx.zip`
  - `recovery_ptarget_t07_t15_onnx.zip`
  - `recovery_ptarget_t07_t18_onnx.zip`

### Current Family State (from decision engine)
- Family: `recovery_ptarget_topk`
- Recorded:
  - `recovery_ptarget_t05_t17_onnx.zip` -> `0.2879` (clear_fail)
  - `recovery_ptarget_t07_t16_onnx.zip` -> `0.2879` (clear_fail)
- State: `final_probe_only`
- Next action: `RUN_FINAL_PROBE_ONLY: recovery_ptarget_t07_t18_onnx.zip`

---

## [2026-02-20] Live Docs Audit (wunder-challenge.io) — Source-of-Truth Clarification

### What was checked
- External docs routes for active challenge `predictorium`:
  - `/predictorium/docs/submission_guide`
  - `/predictorium/docs/data_overview`
  - `/predictorium/docs/rules`
- Platform bundle metadata confirms active challenge code:
  - `predictorium` (id `wnn-lob`, status `active`, submissionsLimit `5`)
  - `wunder_challenge` (id `wnn1`, status `finished`)

### Confirmed constraints for active `predictorium` challenge
- Submission format: `.zip` with root `solution.py` and `PredictionModel.predict(...)`.
- Runtime environment: CPU-only, 1 vCPU, 16GB RAM, local SSD, offline/no internet.
- Docker base image listed: `python:3.11-slim-bookworm`.
- Time limit (submission guide): **90 minutes**.
- Data semantics: warm-up first **99 steps** (`0..98`), scored predictions `99..999`.

### Critical inconsistency discovered
- `predictorium/docs/rules` still contains stale/conflicting resource text:
  - mentions **2 hours** and **4 cores / 16GB RAM**.
- `predictorium/docs/submission_guide` gives concrete scorer setup:
  - **1 vCPU, 16GB RAM, 90 minutes**.

### Decision for project operations
- Treat `predictorium/docs/submission_guide` + observed live behavior as primary runtime source.
- Keep conservative runtime budgeting (existing 4200s empirical guard remains useful).

---

## [2026-02-20] Live Result Update — Final Probe Outcome

- Submission: `recovery_ptarget_t07_t18_onnx.zip`
- Score: `0.2885`
- PB reference: `0.2885`
- Delta: `+0.0000`
- Band: `soft_win`

Decision engine state (`recovery_ptarget_topk`):
- `status`: `active` (reopened)
- `final_probe_status`: `used`
- `next_action`: `REOPEN_BRANCH_TIGHT_SWEEP`

Interpretation:
- The mechanistic final probe recovered to PB level and re-opened the per-target topK family.
- Next cycle should run a tight exploit neighborhood around `t07_t18` (small, attributable deltas only).

### Two-slot exploit pack built (same day follow-up)
With 2 submissions left, avoided widening to new seed sets and built a tight weight bracket around the recovered winner `t07_t18`:

1. `submissions/ready/recovery_ptarget_t07_t18_w64x05_onnx.zip`
   - Same `t0_top7` and `t1_top8` membership as `t07_t18`
   - `t1` weight on seed `64` reduced from `1.0` to `0.5`

2. `submissions/ready/recovery_ptarget_t07_t18_w64x15_onnx.zip`
   - Same membership
   - `t1` weight on seed `64` increased from `1.0` to `1.5`

Checks:
- Both zips pass `scripts/check_submission_zip.py`
- Both sizes: ~3.34MB (<20MB)

Locked submit order for remaining two slots:
1. `recovery_ptarget_t07_t18_w64x05_onnx.zip`
2. `recovery_ptarget_t07_t18_w64x15_onnx.zip`

### Live Results (remaining two slots)
- `recovery_ptarget_t07_t18_w64x15_onnx.zip` -> **0.2886**
  - Delta vs prior PB (`0.2885`): `+0.0001`
  - Band: `soft_win`
- `recovery_ptarget_t07_t18_w64x05_onnx.zip` -> **0.2885**
  - Delta vs prior PB (`0.2885`): `+0.0000`
  - Band: `soft_win`

Outcome:
- New branch best / new PB: **0.2886** from `t07_t18_w64x15`.
- Directional signal: increasing seed64 weight on `t1` helped; decreasing it did not.
- Family remains `active`; next action stays `CONTINUE_SMALL_EXPLOIT`.

### Next full slate built (max 5) without further context
Objective: exploit around new PB winner `t07_t18_w64x15` with small, attributable deltas only.

Built in `submissions/ready/`:
1. `next_recovery_t07_t18_w64x125_onnx.zip`
2. `next_recovery_t07_t18_w64x175_onnx.zip`
3. `next_recovery_t07_t18_w64x150_t61x025_onnx.zip`
4. `next_recovery_t07_t18_w64x175_t61x025_onnx.zip`
5. `next_recovery_t07_t18_w64x150_t61x050_onnx.zip`

All pass pre-submit checks:
- size <20MB
- root `solution.py`
- ONNX model files present
- `PredictionModel` present

Locked submit order for next day:
1. `next_recovery_t07_t18_w64x125_onnx.zip`
2. `next_recovery_t07_t18_w64x175_onnx.zip`
3. `next_recovery_t07_t18_w64x150_t61x025_onnx.zip`
4. `next_recovery_t07_t18_w64x175_t61x025_onnx.zip`
5. `next_recovery_t07_t18_w64x150_t61x050_onnx.zip`

---

## [2026-02-20] Operational Hygiene Lock

- Added persistent directive to `AGENTS.md` and `memory/MEMORY.md`:
  - clear `submissions/ready/` at end of every chat by archiving to
    `submissions/archive/unsent/cleanup_<timestamp>/`
  - keep logs/decision state up to date each chat
- Applied immediately:
  - moved 10 items from `submissions/ready/` to
    `submissions/archive/unsent/cleanup_2026-02-20_170541/`
  - `submissions/ready/` is now empty

## [2026-02-21] Submission Naming Cleanup (Short Format)

Standardized `submissions/ready/` zip names to reduce operator error and improve tracking.

New format (locked):
- `dMMDD-b<batch>-<family>-<variant>-ox.zip`

Old -> New mapping for active batch:
- `next_recovery_t07_t18_w64x125_onnx.zip` -> `d0221-b1-pt7t18-w64125-ox.zip`
- `next_recovery_t07_t18_w64x175_onnx.zip` -> `d0221-b1-pt7t18-w64175-ox.zip`
- `next_recovery_t07_t18_w64x150_t61x025_onnx.zip` -> `d0221-b1-pt7t18-w64150-s61025-ox.zip`
- `next_recovery_t07_t18_w64x175_t61x025_onnx.zip` -> `d0221-b1-pt7t18-w64175-s61025-ox.zip`
- `next_recovery_t07_t18_w64x150_t61x050_onnx.zip` -> `d0221-b1-pt7t18-w64150-s61050-ox.zip`

Submit order preserved:
1. `d0221-b1-pt7t18-w64125-ox.zip`
2. `d0221-b1-pt7t18-w64175-ox.zip`
3. conditional third based on winner of #1/#2:
   - if `w64175 >= w64125`: `d0221-b1-pt7t18-w64175-s61025-ox.zip`
   - else: `d0221-b1-pt7t18-w64150-s61025-ox.zip`

## [2026-02-21] Live Result Update (Batch d0221-b1)

User-reported platform scores:
- `next_recovery_t07_t18_w64x175_onnx.zip` -> `0.2886`
- `next_recovery_t07_t18_w64x125_onnx.zip` -> `0.2886`

Mapped short names:
- `d0221-b1-pt7t18-w64175-ox.zip` -> `0.2886`
- `d0221-b1-pt7t18-w64125-ox.zip` -> `0.2886`

Decision engine (PB `0.2886`):
- both are `soft_win` (`delta +0.0000`)
- branch state remains `active`
- next action: `CONTINUE_SMALL_EXPLOIT`

Action taken:
- archived scored zips to `submissions/archive/scored/d0221_batch1_scored/`
- kept unscored candidates in `submissions/ready/`

Next submit (third slot in this cycle):
- `d0221-b1-pt7t18-w64175-s61025-ox.zip`

## [2026-02-21] Naming Convention Humanized

Updated naming convention to be more readable:
- new format: `monDD-b<batch>-<family>-<variant>.onnx.zip`
- example: `feb21-b1-ptarget-t07t18-w64x175-s61x025.onnx.zip`

Renamed active `submissions/ready/` files:
- `d0221-b1-pt7t18-w64175-s61025-ox.zip` -> `feb21-b1-ptarget-t07t18-w64x175-s61x025.onnx.zip`
- `d0221-b1-pt7t18-w64150-s61025-ox.zip` -> `feb21-b1-ptarget-t07t18-w64x150-s61x025.onnx.zip`
- `d0221-b1-pt7t18-w64150-s61050-ox.zip` -> `feb21-b1-ptarget-t07t18-w64x150-s61x050.onnx.zip`

## [2026-02-21] Naming Convention Shortened Again (Readable)

Moved to shorter readable format:
- `monDD-b<batch>-t<ab>-w<nnn>-s<nnn>-onnx.zip`
- Example: `feb21-b1-t718-w175-s025-onnx.zip`

Rename mapping (active ready files):
- `feb21-b1-ptarget-t07t18-w64x175-s61x025.onnx.zip` -> `feb21-b1-t718-w175-s025-onnx.zip`
- `feb21-b1-ptarget-t07t18-w64x150-s61x025.onnx.zip` -> `feb21-b1-t718-w150-s025-onnx.zip`
- `feb21-b1-ptarget-t07t18-w64x150-s61x050.onnx.zip` -> `feb21-b1-t718-w150-s050-onnx.zip`

Correction note (2026-02-21):
- previously listed next submit `d0221-b1-pt7t18-w64175-s61025-ox.zip` is now renamed to `feb21-b1-t718-w175-s025-onnx.zip`.

## [2026-02-21] Live Result Update

- `feb21-b1-t718-w175-s025-onnx.zip` -> `0.2886`
- PB reference: `0.2886`
- delta: `+0.0000` (`soft_win`)
- branch state: `active`
- next action: `CONTINUE_SMALL_EXPLOIT`

Artifact hygiene action:
- archived scored zip to `submissions/archive/scored/feb21_batch1_scored/`
- kept remaining unscored candidates in `submissions/ready/`
