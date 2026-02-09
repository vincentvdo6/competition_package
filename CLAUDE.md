# Wunderfund Predictorium RNN Challenge

## Collaboration Protocol (Claude Code + Codex)

Claude Code and Codex are **equal contributors**. Neither has dominance. Codex is connected as an MCP server — Claude routes tasks to it.

### When to ALWAYS consult Codex (non-negotiable)
- **Strategy decisions**: Ensemble composition, which models to train, submission strategy
- **Architecture/model changes**: New model types, loss functions, feature engineering
- **Inference code**: Anything in solution.py or export_ensemble.py (mistakes waste submissions)
- **Debugging failures**: Timeouts, score drops, training issues
- **New experiments**: Before starting any new training run or config
- **Uncertainty**: When Claude isn't confident, ALWAYS ask Codex. Err on the side of asking.

### When Claude can act solo (routine operations)
- Reading files, git operations, status checks
- Simple formatting, typo fixes, comment updates
- Running already-agreed-upon commands
- Reporting results back to the user

### Workflow for important decisions
1. **Consult**: Claude routes the problem to Codex MCP with full context
2. **Agree**: Both models must agree on the plan. Disagreements go to the user.
3. **Execute**: Claude executes the agreed plan
4. **Review**: Route changes back to Codex for review before declaring done
5. **Iterate**: Fix any issues Codex flags

### Codex MCP Call Optimization (MANDATORY)
Codex reads the entire codebase if you let it. Always use these parameters:
- **`sandbox: "read-only"`** — prevents shell commands
- **`base-instructions: "Answer from the prompt only. Do NOT read or explore any files. Be concise."`** — prevents codebase exploration
- **Prompts must be under 150 words.** Include only the essential context inline — don't make Codex go find it.
- **Request specific output format** (e.g., "3-5 bullets", "yes/no + 1 sentence", "table format")

### Rules
- **When in doubt, consult Codex.** Better to over-consult than to miss something.
- Do NOT assume what Codex would say. Actually ask via MCP.
- Read CLAUDE.md and MEMORY.md at the start of every new chat.
- Record all findings and decisions in project files for shared access.

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

## Current State (as of 2026-02-08)

### Leaderboard Scores
| Submission | Score | Models | Time | Status |
|-----------|-------|--------|------|--------|
| Single tightwd_v2 | 0.2580 | 1 GRU | ~319s | Baseline |
| 5-seed GRU uniform | 0.2614 | 5 GRU | ~1595s | Seed diversity |
| 5 GRU + 2 attn 70/30 (old seeds) | 0.2633 | 7 (5G+2A) | ~3463s | Previous champion |
| 5 GRU + 3 attn uniform | TIMEOUT | 8 (5G+3A) | >4200s | Failed |
| 5 GRU + 3 attn OPTIMIZED | 0.2624 | 8 (5G+3A) | ~4185s | Below champion |
| balanced7 (5G+2A pearson attn) | 0.2615 | 7 (5G+2A) | ~3311s | Pearson attn HURTS |
| **champion_clone_v2** | **0.2654** | 7 (5G+2A) | ~3311s | **CHAMPION** — best 5 GRU by val + old combined attn 70/30 |

### Available Checkpoints
**In downloaded zips (C:\Users\Vincent\Downloads\):**
- `gru5_attn3_uniform8.zip` — 5 GRU tightwd_v2 (seeds 42-46) + 3 attn clean (seeds 42-44), all combined loss, slim
- `gru5_plus_attention2_balanced7030.zip` — 5 GRU + 2 attn, 70/30 weights (champion)
- `tightwd5_uniform.zip` — 5 GRU tightwd_v2 only, full checkpoints (not slim)
- `slim_checkpoints_pearson.zip` — 3 GRU pearson (42-44) + 2 attn clean (45-46) + 2 attn pearson (42-43)
- `gru_seed_expansion.zip` — 7 tightwd_v2 (47-53) + 6 pearson_v1 (45-50), all slim

### Expanded GRU Pool (val scores from notebook 06)
| Model | Val Score | Config | Seed |
|-------|----------|--------|------|
| gru_pearson_v1_seed47 | **0.2668** | pearson_combined | 47 |
| gru_tightwd_v2_seed50 | **0.2654** | combined | 50 |
| gru_tightwd_v2_seed48 | **0.2649** | combined | 48 |
| gru_pearson_v1_seed45 | **0.2648** | pearson_combined | 45 |
| gru_pearson_v1_seed50 | **0.2640** | pearson_combined | 50 |
| gru_tightwd_v2_seed51 | 0.2637 | combined | 51 |
| gru_tightwd_v2_seed53 | 0.2636 | combined | 53 |
| gru_pearson_v1_seed46 | 0.2634 | pearson_combined | 46 |
| gru_tightwd_v2_seed47 | 0.2620 | combined | 47 |
| gru_pearson_v1_seed49 | 0.2603 | pearson_combined | 49 |
| gru_pearson_v1_seed48 | 0.2589 | pearson_combined | 48 |
| gru_tightwd_v2_seed49 | 0.2577 | combined | 49 |
| gru_tightwd_v2_seed52 | 0.2537 | combined | 52 |

### Expanded Attention Pool (val scores from notebook 07 + original training)
| Model | Val Score | Source | Seed |
|-------|----------|--------|------|
| **attn_clean_seed50** | **0.2752** | nb07 session 2 | 50 |
| attn_clean_seed48 | 0.2706 | nb07 session 2 | 48 |
| attn_clean_seed46 | 0.2659 | nb07 session 1 | 46 |
| attn_clean_seed52 | 0.2641 | nb07 session 3 | 52 |
| attn_clean_seed51 | 0.2600 | nb07 session 3 | 51 |
| attn_clean_seed45 | 0.2599 | nb07 session 1 | 45 |
| attn_clean_seed47 | 0.2598 | nb07 session 1 | 47 |
| attn_clean_seed49 | 0.2560 | nb07 session 2 | 49 |
| attn_clean_seed42 | — | original | 42 |
| attn_clean_seed43 | — | original | 43 |
| attn_clean_seed44 | — | original | 44 |

All use combined loss (gru_attention_clean_v1 config). Zips: attn_seeds_45_46_47.zip, attn_seeds_48_49_50.zip, attn_seeds_51_52.zip.

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
- **Primary**: `champion_clone_v2` — best 5 GRU (by val) + 2 old combined attn, 70/30 weights (same structure as champion with upgraded GRUs)
- **Alternative**: `fast8_gru` (8 GRU-only, loss diversity) — more models, no attention overhead, 40% margin
- **Avoid**: Pearson attention models in ensembles (tested, hurts)

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
- **GRU-dominant weighting (~70%) is critical** — 8-model uniform (62.5% GRU) scored 0.2624, worse than 7-model 70/30 (71% GRU) at 0.2633

### What Doesn't Work
- Temporal features HURT performance
- Interaction features HURT performance
- LSTM underperforms GRU
- Optimized ensemble weights (SLSQP, per-target) underperform uniform on LB
- Ring buffer SLOWER than torch.cat on CPU (indexed write overhead)
- t1-focused loss weighting HURTS
- **Pearson-loss attention models HURT ensemble** — balanced7 (0.2615) barely beat 5-GRU-only (0.2614)

### Untested Hypotheses
- Dynamic quantization (INT8) — est. 1.2-1.6x speedup on CPU, unlocks +2-3 models
- ONNX Runtime inference — could be 2-5x faster than PyTorch on CPU
- Reduced attention_window (128 -> 64) — halves attention cost
- torch.jit.script compilation — potential 10-30% CPU speedup
- Recency-weighted loss — weight later timesteps more heavily in training
- Distilled student — 1 tiny model trained on ensemble soft targets + true labels

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
| validate_ensemble_local.py | Cache per-model predictions, greedy/exhaustive ensemble search, diversity analysis |

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
| 05_retrain_pearson_models.ipynb | Complete — 7 pearson models trained |
| 06_gru_seed_expansion.ipynb | Complete — 13 new GRU models (7 tightwd + 6 pearson) |

---

## Immediate Next Steps

### champion_clone_v2 = 0.2654 (CHAMPION)
- GRUs (0.14 each, 70% total): pearson_v1_seed47 (0.2668), tightwd_v2_seed50 (0.2654), tightwd_v2_seed48 (0.2649), pearson_v1_seed45 (0.2648), pearson_v1_seed50 (0.2640)
- Attention (0.15 each, 30% total): attn_clean_seed42, attn_clean_seed43 (both combined loss, from gru5_attn3_uniform8.zip model_5/model_6)

### Target: 0.2761 (133rd place, +0.0107 gap)
- 133rd through 305th all share this score

### This Week (Codex-agreed, ROI-ordered)
1. **Latency-aware blend optimization** — local validation running (validate_ensemble_local.py), greedy search with time-budget constraint + low correlation. ~done.
2. **Dynamic quantization** — INT8 quantization on CPU, est. 1.2-1.6x speedup, unlocks +2-3 more models. Fast/low-risk.
3. **Recency-weighted GRU retrain** — weight later timesteps more heavily in loss. Same architecture, new objective.
4. **Scale attention seeds** — DONE! All 11 seeds trained (42-52). Running local inference now (~3h). Best: seed50 val 0.2752.

### Next Week
5. **Proxy microstructure features** — queue imbalance, microprice, spread slope, OFI proxy. Strict ablation, kill fast if no lift.
6. **Distilled student** — 1 tiny GRU trained on ensemble soft targets + true labels. Frees time budget.
7. **Stateful TCN** — tiny causal TCN with cached conv buffers. Moonshot only if time permits.

---

## Technical Notes
- Kaggle `%%bash` buffers output → use `os.system()` or `!` prefix instead
- `sys.stdout.isatty()` returns False in notebooks → tqdm falls back to batch logs
- Checkpoints must be stripped (remove optimizer/scheduler state) to fit 20MB zip limit
- Kaggle sessions don't persist → train + export + download in single session
- `.gitignore` excludes: *.pt, *.npz, *.zip, logs/slim/, submissions/
- `export_ensemble.py` is modified locally with optimizations but NOT yet committed to GitHub
