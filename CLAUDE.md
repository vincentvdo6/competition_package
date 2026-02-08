# Wunderfund Predictorium RNN Challenge

## Collaboration Protocol (Claude Code + Codex)

**MANDATORY FOR EVERY TASK:**

1. **Consult**: Before any action, Claude MUST route the task to Codex MCP for joint discussion. Both models analyze the problem independently and share their approach.
2. **Agree**: Both models must agree on the plan before execution. If they disagree, present both perspectives to the user for a decision.
3. **Execute**: Claude Code executes the agreed plan (it has file write access).
4. **Review**: After execution, route the changes back to Codex MCP for review. Codex checks for correctness, missed edge cases, and improvement opportunities.
5. **Iterate**: If Codex flags issues during review, fix them before moving on.

**Rules:**
- Neither model has dominance. Both are equal contributors.
- Do NOT skip the consultation step, even for "simple" tasks.
- Do NOT make assumptions about what the other model would say. Actually ask.
- When context is stale or unclear, read CLAUDE.md and MEMORY.md first.
- All findings, decisions, and results must be recorded in project files for shared access.

---

## Competition Overview

- **Competition**: Wunderfund Predictorium (HFT prop trading firm)
- **Task**: Predict two targets (t0, t1) from Limit Order Book sequences
- **Metric**: Weighted Pearson Correlation (weighted by |target|, clipped to [-6, 6])
- **Prize Pool**: $13,600
- **Goal**: Top 100 placement

### Inference Constraints (Scoring Environment)
- **CPU**: 1 vCPU core (no GPU)
- **RAM**: 16 GB
- **Time limit**: 4200 seconds on full test dataset
- **No internet**: Offline execution
- **Docker**: python:3.11-slim-bookworm with PyTorch CPU
- **Submission format**: zip with solution.py + checkpoints + normalizers, max 20MB

### Data Format
- **Train**: 10,721 sequences x 1000 steps each
- **Valid**: 1,444 sequences x 1000 steps each
- **Features**: 32 raw (bid/ask prices p0-p11, volumes v0-v11, trade dp0-dv3)
- **need_prediction**: True for steps 99-999 (90.1% of steps), False for warm-up 0-98
- **Online inference**: Process one step at a time, reset hidden state on new seq_ix

### Submission Budget
- **5 submissions per day**
- **1+ month remaining** in competition

---

## Current State (as of 2026-02-07)

### Leaderboard Scores
| Submission | Score | Models | Time | Status |
|-----------|-------|--------|------|--------|
| Single tightwd_v2 | 0.2580 | 1 GRU | ~319s | Baseline |
| 5-seed GRU uniform | 0.2614 | 5 GRU | ~1595s | Seed diversity |
| 5 GRU + 2 attn 70/30 | **0.2633** | 7 (5G+2A) | ~3463s | **CHAMPION** |
| 5 GRU + 3 attn uniform | TIMEOUT | 8 (5G+3A) | >4200s | Failed |
| 5 GRU + 3 attn OPTIMIZED | PENDING | 8 (5G+3A) | ~4185s est | Borderline |

### Training In Progress
Kaggle notebook `05_retrain_pearson_models.ipynb` is running (currently on cell 3 of 6):
- Cell 1: 3x gru_pearson_v1 seeds 42-44 (DONE)
- Cell 2: 2x gru_attention_clean_v1 seeds 45-46 (DONE)
- Cell 3: 2x gru_attention_pearson_v1 seeds 42-43 (IN PROGRESS)
- Cell 4-6: Strip checkpoints, zip, print scores (PENDING)

Output will be: `slim_checkpoints_pearson.zip` containing 7 new models.

### Available Checkpoints
**In downloaded zips (C:\Users\Vincent\Downloads\):**
- `gru5_attn3_uniform8.zip` — 5 GRU tightwd_v2 (seeds 42-46) + 3 attn clean (seeds 42-44), all combined loss, slim
- `gru5_plus_attention2_balanced7030.zip` — 5 GRU + 2 attn, 70/30 weights (champion)
- `tightwd5_uniform.zip` — 5 GRU tightwd_v2 only, full checkpoints (not slim)

**Being trained (will download as slim_checkpoints_pearson.zip):**
- 3x gru_pearson_v1 (seeds 42-44) — pearson_combined loss
- 2x gru_attention_clean_v1 (seeds 45-46) — combined loss
- 2x gru_attention_pearson_v1 (seeds 42-43) — pearson_combined loss

---

## Calibrated Timing Data

### Per-Model Inference Cost (from real submissions, OLD non-optimized code)
| Model Type | Cost per Model | Source |
|-----------|---------------|--------|
| GRU | 319s | tightwd5_uniform: 1595s / 5 models |
| GRU+Attention | 934s (2.93x) | champion: (3463s - 1595s) / 2 models |

### need_pred Optimization Impact
- need_prediction=True for **90.1%** of steps (99-999 out of 0-999)
- Optimization only skips output_proj/attention on warm-up steps (9.9%)
- **Actual savings: ~3-5% total** (NOT 50% as originally estimated)
- GRU optimized: ~313s (-2%)
- Attention optimized: ~874s (-6%)

### Safe Ensemble Presets (with optimized inference code)
| Preset | Models | Composition | Est. Time | Margin | Status |
|--------|--------|-------------|-----------|--------|--------|
| fast8_gru | 8 | 8 GRU, 0 attn | 2501s | +40% | **SAFE** |
| balanced7 | 7 | 5 GRU, 2 attn | 3311s | +21% | **SAFE** |
| diverse9 | 9 | 7 GRU, 2 attn | 3937s | +6% | OK |
| diverse10 | 10 | 8 GRU, 2 attn | 4249s | -1% | **TIMEOUT** |
| diverse12 | 12 | 8 GRU, 4 attn | 5997s | -43% | **TIMEOUT** |

### Recommendation
- **Primary**: `balanced7` (5 GRU combined + 2 attn pearson) — same count as champion, adds loss diversity, 21% margin
- **Alternative**: `fast8_gru` (5 GRU combined + 3 GRU pearson) — more models, no attention overhead, 40% margin
- **Risky**: `diverse9` — max diversity but only 6% margin

---

## Key Findings

### What Works
- **GRU > LSTM** consistently on this data
- **Derived features** add +0.0036 avg (spreads, trade intensity, pressure)
- **tightwd_v2** is best GRU config: hidden=144, dropout=0.22, lr=0.0008, WD=5e-5
- **Attention clean** is best attention config: same base + 4 heads, window=128
- **Seed diversity** is critical (0.0095 gap between identical runs)
- **Uniform weights > optimized weights** on LB consistently
- **Architecture diversity** helps (+0.0019 LB from adding attention to GRU ensemble)

### What Doesn't Work
- Temporal features HURT performance
- Interaction features HURT performance
- LSTM underperforms GRU
- Optimized ensemble weights (SLSQP, per-target) underperform uniform on LB
- Ring buffer SLOWER than torch.cat on CPU (indexed write overhead)
- t1-focused loss weighting HURTS

### Untested Hypotheses
- Loss diversity (combined + pearson models in same ensemble) — NEW, being tested
- ONNX Runtime inference — could be 2-5x faster than PyTorch on CPU
- Reduced attention_window (128 -> 64) — halves attention cost
- torch.jit.script compilation — potential 10-30% CPU speedup

---

## Architecture & Key Files

### Active Configs (configs/)
| Config | Type | Loss | Key Diff |
|--------|------|------|----------|
| gru_derived_tightwd_v2.yaml | GRU | combined | Best GRU baseline |
| gru_attention_clean_v1.yaml | GRU+Attn | combined | Best attention model |
| gru_pearson_v1.yaml | GRU | pearson_combined | Metric-aligned loss, lr=0.0005 |
| gru_attention_pearson_v1.yaml | GRU+Attn | pearson_combined | Metric-aligned attention |

### Core Scripts (scripts/)
| Script | Purpose |
|--------|---------|
| train.py | Train single model from config + seed |
| export_ensemble.py | Build submission zip (optimized solution.py with feature cache + need_pred skip) |
| build_mixed_ensemble.py | Combine old + new checkpoints using presets, calls export_ensemble.py |
| evaluate.py | Local evaluation on valid set |
| validate_online_parity.py | Verify step-by-step matches batch inference |
| score_ensemble_candidates.py | Score ensemble combinations |

### Source Code (src/)
- `src/models/gru_baseline.py` — GRU with input projection + LayerNorm + output MLP
- `src/models/gru_attention.py` — GRU + multi-head causal attention, ring buffer for online inference
- `src/models/lstm_model.py` — LSTM variant (unused, underperforms)
- `src/training/trainer.py` — Training loop with AMP, grad clip, early stopping, checkpointing
- `src/training/losses.py` — Loss factory: MSE, Combined, Huber, WeightedPearson, PearsonCombined
- `src/data/preprocessing.py` — DerivedFeatureBuilder (10 features), TemporalBuffer, InteractionBuilder
- `src/data/dataset.py` — PyTorch Dataset from parquet
- `src/evaluation/metrics.py` — Weighted Pearson Correlation, all loss implementations

### Notebooks (notebooks/)
| Notebook | Status |
|----------|--------|
| 01_eda.ipynb | Complete — data analysis |
| 02_feature_analysis.ipynb | Complete — feature importance |
| 03_hp_optimization.ipynb | Complete — HP sweep results |
| 04_seed_diversity_analysis.ipynb | Complete — seed variance analysis |
| 05_retrain_pearson_models.ipynb | **ACTIVE on Kaggle** — training 7 new pearson models |

---

## Immediate Next Steps

1. **Wait** for Kaggle training to finish (~44 min remaining for cell 3, then cells 4-6 are fast)
2. **Download** `slim_checkpoints_pearson.zip` from Kaggle output
3. **Check** optimized 8-model submission result (borderline, may timeout)
4. **Build** ensemble with `build_mixed_ensemble.py`:
   ```
   python scripts/build_mixed_ensemble.py --new-zip <path> --preset balanced7
   ```
5. **Submit** the zip and evaluate LB score
6. **Iterate** based on results — try other presets, investigate ONNX/JIT speedups for more models

---

## Technical Notes
- Kaggle `%%bash` buffers output → use `os.system()` or `!` prefix instead
- `sys.stdout.isatty()` returns False in notebooks → tqdm falls back to batch logs
- Checkpoints must be stripped (remove optimizer/scheduler state) to fit 20MB zip limit
- Kaggle sessions don't persist → train + export + download in single session
- `.gitignore` excludes: *.pt, *.npz, *.zip, logs/slim/, submissions/
- `export_ensemble.py` is modified locally with optimizations but NOT yet committed to GitHub
