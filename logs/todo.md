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
- [ ] Deep feature analysis notebook (02_feature_analysis.ipynb): (@codex)
  - Analyze WHY gru_derived_v1 is best (which derived features contribute most)
  - t1 predictability analysis: autocorrelation with lag features, cross-feature interactions
  - Feature importance via permutation or gradient analysis
  - Propose next-round feature engineering (rolling stats, rate-of-change, etc.)
- [ ] Hyperparameter optimization notebook (03_hp_optimization.ipynb): (@codex)
  - Analyze training curves from all 5 experiments
  - Suggest optimal lr, dropout, batch_size ranges for sweep
  - Investigate why early stopping triggers so early (epoch 4-11)
  - Regularization analysis: would weight decay, label smoothing, or mixup help?
- [ ] Create improved configs based on analysis (@codex)

## Phase 3 — Architecture Exploration

- [x] LSTM implementation (@claude-code) — RESULT: underperforms GRU, deprioritized
- [ ] Transformer with causal masking (@claude-code)
- [ ] Sequence length ablation study (@codex)
- [ ] Architecture comparison notebook (@codex)

## Phase 4 — Optimization & Submission

- [ ] Run hyperparameter sweeps on gru_derived_v1 (lr, dropout) (@user on Kaggle)
- [ ] Ensemble top models (@claude-code)
- [ ] Final hyperparameter fine-tuning (@both)
- [ ] Export and validate submission zip (@claude-code)
- [ ] Verify submission in clean environment (@claude-code)
