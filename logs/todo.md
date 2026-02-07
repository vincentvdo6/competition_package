# Task Queue

## Phase 1 — Foundation (COMPLETE)

- [x] Set up repo structure, requirements.txt, .gitignore (@claude-code)
- [x] Create logs/ with initial content (@claude-code)
- [x] Implement src/data/dataset.py — pre-tensorized, zero-copy (@claude-code)
- [x] Implement src/evaluation/metrics.py — WeightedMSE, CombinedLoss, HuberWeighted (@claude-code)
- [x] Implement src/models/gru_baseline.py — 2-layer GRU, 211K params (@claude-code)
- [x] Implement src/training/trainer.py — GPU+CPU, AMP, early stopping (@claude-code)
- [x] Implement src/training/losses.py — loss factory (@claude-code)
- [x] Implement scripts/train.py — CLI entrypoint (@claude-code)
- [x] Implement scripts/evaluate.py — local validation with ScorerStepByStep (@claude-code)
- [x] Train GRU baseline — Val avg 0.2578 (t0: 0.3869, t1: 0.1286) (@claude-code)

## Phase 2 — Strong Baseline (IN PROGRESS)

### Completed
- [x] EDA notebook (01_eda.ipynb): feature distributions, target analysis, correlations (@codex)
- [x] Write EDA artifacts to notebooks/artifacts/01_eda for reproducibility (@codex)
- [x] Create EDA-driven training configs (@codex)
- [x] Implement derived features (DerivedFeatureBuilder) in src/data/preprocessing.py (@claude-code)
- [x] Wire derived features through dataset.py, train.py, evaluate.py (@claude-code)
- [x] Implement target-specific loss weights (target_weights config) in losses (@claude-code)
- [x] Implement scripts/export_submission.py — self-contained zip packager (@claude-code)
- [x] Create new configs: gru_derived_v1, gru_derived_t1focus_v1, gru_long_memory_derived_v1 (@claude-code)
- [x] Implement scripts/sweep.py — hyperparameter sweep launcher (@claude-code)
- [x] LSTM implementation in src/models/lstm_model.py + configs (@claude-code)
- [x] Wire LSTM into train.py, evaluate.py, export_submission.py (@claude-code)

### Kaggle Training Results
- [x] Train gru_derived_v1 — Val avg 0.2614 / 0.2608 (t0: 0.3912, t1: 0.1316), best epoch 7 (@user) **BEST**
- [x] Train gru_derived_t1focus_v1 — Val avg 0.2536 (t0: 0.3854, t1: 0.1218), target_weights HURT (@user)
- [x] Train gru_long_memory_derived_v1 — Val avg 0.2609 (t0: 0.3869, t1: 0.1348), overfits fast (@user)
- [x] Train lstm_derived_v1 — Val avg 0.2542 (t0: 0.3836, t1: 0.1247), LSTM < GRU (@user)
- [ ] Train lstm_large_derived_v1 (LSTM 192h/3L + derived + t1) (@user) — LOW PRIORITY given LSTM results
- [ ] Export submission zip from best gru_derived_v1 checkpoint (@claude-code)

### Codex Tasks (READY — assign these next)
- [x] Deep feature analysis notebook (02_feature_analysis.ipynb): (@codex)
  - Analyze WHY gru_derived_v1 is best (which derived features contribute most)
  - t1 predictability analysis: autocorrelation with lag features, cross-feature interactions
  - Feature importance via permutation or gradient analysis
  - Propose next-round feature engineering (rolling stats, rate-of-change, etc.)
- [x] Hyperparameter optimization notebook (03_hp_optimization.ipynb): (@codex)
  - Analyze training curves from all 5 experiments
  - Suggest optimal lr, dropout, batch_size ranges for sweep
  - Investigate why early stopping triggers so early (epoch 4-11)
  - Regularization analysis: would weight decay, label smoothing, or mixup help?
- [x] Create improved configs based on analysis (@codex)

### New Follow-ups from Codex Analysis
- [x] Implement candidate temporal derived features (`spread0_roc1`, `spread0_roc5`, `trade_intensity_roll_mean_5`) (@claude-code) — RESULT: temporal features HURT (0.2512 vs 0.2614), deprioritized
- [x] Run new GRU+derived configs: `gru_derived_reg_v2.yaml` (0.2660), `gru_derived_tightwd_v2.yaml` (**0.2674 BEST**), `gru_derived_onecycle_v2.yaml` (0.2566) (@user on Kaggle)
- [x] Persist per-epoch train/val history (loss + t0/t1/avg scores) (@claude-code)
- [x] First submission: tightwd_v2 single model — LB score **0.2580**, rank **338/4598** (@user)

## Phase 3 — Ensemble & Architecture Diversity (IN PROGRESS)

### Wave 1: Seed Ensemble (highest ROI)
- [x] Add seed control (`--seed` flag) to scripts/train.py (@claude-code)
- [x] Add per-seed checkpoint naming to trainer.py (@claude-code)
- [x] Build scripts/export_ensemble.py — multi-model ensemble packager (@claude-code)
- [x] Train tightwd_v2 x5 seeds (42-46) on Kaggle (@user) — scores: 0.2581/0.2646/0.2641/0.2586/0.2650
- [x] Seed diversity analysis notebook (04_seed_diversity_analysis) (@codex) — runner + artifacts schema ready
- [ ] Execute seed diversity analysis after seed checkpoints are synced locally (generate full metrics/plots/weights) (@codex)
- [ ] Export seed ensemble + submit (@user)
- [x] Upgrade `export_ensemble.py` for heterogeneous pipelines + per-model normalizers + per-target weights (@codex)
- [x] Add `scripts/build_wave1_candidates.py` to auto-generate A/B/C candidate zips from run_04 artifacts (@codex)
- [x] Add `scripts/validate_online_parity.py` for forward-step/reset/preprocessing parity checks (@codex)

### Wave 2: Architecture Diversity
- [x] Implement GRU+Attention model in src/models/gru_attention.py (@codex)
- [x] Add feature interaction features (v8*p0, spread_0*p0, spread_0*v2) (@codex)
- [x] Wire GRU+Attention through train/evaluate/export pipeline (@codex)
- [x] Add Wave-2 configs (`gru_interaction_tightwd_v1`, `gru_attention_interaction_v1`, `gru_attention_interaction_v1b`) (@codex)
- [ ] Attention prototype notebook (05_attention_prototype) (@codex) — go/no-go decision
- [ ] Train gru_refined_v3, v3b, attention_v1 on Kaggle (@user)
- [ ] Build diverse ensemble (GRU seeds + Attention + v3), submit (@user)

### Wave 3: Optimization
- [ ] Add SWA (Stochastic Weight Averaging) support to trainer.py (@claude-code)
- [ ] Add post-processing (EMA, scaling) to ensemble export (@claude-code)
- [ ] Greedy ensemble selection script (@claude-code)
- [ ] Ensemble optimization notebook (06_ensemble_optimization) (@codex)
- [ ] Post-processing analysis notebook (07_postprocessing) (@codex)
- [ ] Final optimized ensemble + submit (@user)

## Phase 4 — Stretch Goals
- [ ] Transformer with causal masking (@claude-code)
- [ ] Mamba/SSM architecture (@claude-code)
- [ ] Sequence length ablation study (@codex)
- [ ] Cross-validation framework (@codex)
