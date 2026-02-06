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

### Kaggle Training Results
- [x] Train gru_derived_v1 — Val avg 0.2614 (t0: 0.3912, t1: 0.1316), best epoch 7 (@user)
- [ ] Train gru_derived_t1focus_v1 (derived features + t1 emphasis) (@user)
- [ ] Train gru_long_memory_derived_v1 (big model + derived + t1 emphasis) (@user)
- [ ] Train lstm_derived_v1 (LSTM + derived + t1 emphasis) (@user)
- [ ] Train lstm_large_derived_v1 (LSTM 192h/3L + derived + t1) (@user)
- [ ] Train Codex configs: gru_long_memory_v1, gru_weighted_focus_v1, gru_huber_shift_robust_v1 (@user)

### Completed (Claude Code)
- [x] Implement scripts/sweep.py — hyperparameter sweep launcher (@claude-code)
- [x] LSTM implementation in src/models/lstm_model.py + configs (@claude-code)
- [x] Wire LSTM into train.py, evaluate.py, export_submission.py (@claude-code)

### Codex Tasks (next round)
- [ ] Feature analysis notebook (02_feature_analysis.ipynb): autocorrelation, stationarity, cross-feature (@codex)
- [ ] Hyperparameter tuning experiments in notebook (@codex)

## Phase 3 — Architecture Exploration

- [x] LSTM implementation (@claude-code)
- [ ] Transformer with causal masking (@claude-code)
- [ ] Sequence length ablation study (@codex)
- [ ] Architecture comparison notebook (@codex)

## Phase 4 — Optimization & Submission

- [ ] Ensemble top models (@claude-code)
- [ ] Final hyperparameter fine-tuning (@both)
- [ ] Export and validate submission zip (@claude-code)
- [ ] Verify submission in clean environment (@claude-code)
