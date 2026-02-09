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

## Current State (as of 2026-02-10)

### Leaderboard Scores (all submissions ever)
| Submission | Score | Models | Time | Notes |
|-----------|-------|--------|------|-------|
| Single tightwd_v2 | 0.2580 | 1 GRU | 321s | Baseline |
| 5-seed GRU uniform | 0.2614 | 5 GRU | 1595s | Seed diversity |
| balanced7 (pearson attn) | 0.2615 | 5G+2A | 3504s | Pearson attn HURTS |
| 5 GRU + 3 attn OPTIMIZED | 0.2624 | 5G+3A | 3836s | Too much attn weight |
| 5 GRU + 2 attn 70/30 (old) | 0.2633 | 5G+2A | 3463s | Old champion |
| best_gru_combo (exhaustive) | 0.2647 | 5G+2A | 3164s | OVERFITS val |
| champion_clone_v2 | 0.2654 | 5G+2A | 2925s | Top-5 GRU + old attn |
| champion_v4_s50swap | 0.2668 | 5G+2A | 2895s | Old champion |
| **s2_s43_swap** | **0.2675** | **5G+2A** | **3106s** | **CURRENT CHAMPION** |
| champion_v4_top2attn | TIMEOUT | 5G+2A | 4199s | Server variance |
| 5 GRU + 3 attn uniform | TIMEOUT | 5G+3A | 4200s | Too many attn |
| s4_nb07s48_swap | 0.2646 | 5G+2A | 3670s | nb07_s48 worse than s50 |

### Val-to-LB Calibration (8 data points)
| Submission | Val | LB | Gap | # Models |
|-----------|-----|-----|-----|----------|
| Single GRU | 0.2584 | 0.2580 | -0.0004 | 1 |
| 5 GRU uniform | 0.2683 | 0.2614 | -0.0069 | 5 |
| Old champion 5G+2A | 0.2715 | 0.2633 | -0.0082 | 7 |
| champion_clone_v2 | 0.2734 | 0.2654 | -0.0080 | 7 |
| best_gru_combo | 0.2788 | 0.2647 | **-0.0141** | 7 (OVERFIT) |
| champion_v4_s50swap | 0.2758 | 0.2668 | -0.0090 | 7 |
| s2_s43_swap | 0.2770 | 0.2675 | -0.0095 | 7 |
| s4_nb07s48_swap | 0.2749 | 0.2646 | -0.0103 | 7 |
- **7-model gap (excl overfit)**: mean -0.0088, range [-0.0103, -0.0080]
- **Use -0.009 as conservative 7-model gap estimate**

### Champion s2_s43_swap Details
- **LB Score**: 0.2675 (+0.0007 over previous champion v4_s50swap)
- **GRUs** (70% weight, 0.14 each): p1_s47, tw2_s50, tw2_s48, p1_s45, p1_s50
- **Attention** (30% weight, 0.15 each): attn_comb_s43 (old) + attn_nb07_s50 (new)
- **Key insight**: attn_comb_s43 > attn_comb_s42 — old attention seed rotation works
- **S4 result**: s4_nb07s48_swap scored 0.2646, confirming nb07_s50 >> nb07_s48 on LB

### Timing Data
- **Per GRU**: ~320s (stable across all submissions)
- **Per Attention**: 665-952s (HUGE server variance)
- **Server variance**: 45% — identical 5G+2A ran in 2895s, 3164s, and timed out at 4199s
- **MUST budget >30% margin** for timeout safety

---

## Proven Rules (from LB data)

### DO
- Select GRUs by **top individual val score** (robust, gap ~0.009)
- Use **70/30 GRU/attention weighting** (0.14 per GRU, 0.15 per attn)
- Mix **old + new attention seeds** (diversity > same-batch)
- Use **combined-loss** attention only (pearson-loss attn hurts ensembles)
- Test **one variable at a time** per submission (clean A/B tests)

### DON'T
- **NEVER use exhaustive combo search** for model selection (overfits val, gap -0.014)
- Don't use pearson-loss attention in ensembles
- Don't use >2 attention models (timeout risk with 45% variance)
- Don't trust val scores alone — always calibrate with -0.009 gap
- Don't use optimized/SLSQP weights — uniform/fixed 70/30 is more robust

---

## Available Models

### GRU Models — Top Candidates (new top-5 in bold)
| Model | Val | Config | Status |
|-------|-----|--------|--------|
| **gru_tw2_s63** | **0.2736** | tightwd_v2 | NEW — needs cache+staging |
| **gru_p1_s47** | **0.2668** | pearson_v1 | cached |
| **gru_tw2_s60** | **0.2663** | tightwd_v2 | NEW — needs cache+staging |
| **gru_tw2_s50** | **0.2654** | tightwd_v2 | cached |
| **gru_tw2_s48** | **0.2649** | tightwd_v2 | cached |
| gru_p1_s45 | 0.2648 | pearson_v1 | cached |
| gru_p1_s50 | 0.2640 | pearson_v1 | cached |
| gru_tw2_s57 | 0.2641 | tightwd_v2 | NEW — needs cache+staging |
| gru_tw2_s51 | 0.2637 | tightwd_v2 | cached |
| gru_tw2_s53 | 0.2636 | tightwd_v2 | cached |
| gru_tw2_s62 | 0.2634 | tightwd_v2 | NEW — needs cache+staging |
| gru_p1_s46 | 0.2634 | pearson_v1 | cached |
| gru_tw2_s42-46 | various | tightwd_v2 | cached |

### Seed Expansion Progress
- **tw2 COMPLETE**: 32 seeds (s42-73), batches 1+2 done, downloaded
- **p1 batch 1-2 (s51-70)**: In progress on Colab
- **p1 batch 3-4 (s71-90)**: Next on Kaggle
- **Next steps**: Extract to _staging, cache predictions, register in validate_ensemble_local.py

### GRU Models (13 cached in cache/predictions/)

### Attention Models (10 cached)
| Model | Val | Corr with s42 |
|-------|-----|--------------|
| attn_nb07_s50 | **0.2752** | 0.926 |
| attn_nb07_s48 | 0.2706 | 0.929 |
| attn_nb07_s46 | 0.2659 | 0.935 |
| attn_nb07_s52 | 0.2641 | 0.934 |
| attn_nb07_s51 | 0.2600 | 0.917 (most diverse) |
| attn_nb07_s45 | 0.2599 | 0.910 |
| attn_nb07_s47 | 0.2598 | 0.937 |
| attn_nb07_s49 | 0.2560 | 0.932 |
| attn_comb_s42 | — | 1.000 (old, in champion) |
| attn_comb_s43 | — | 0.895 (most diverse old) |

### Not Cached (need inference before use in val)
- attn_comb_s44, attn_comb_s45, attn_comb_s46, attn_pear_s42, attn_pear_s43
- gru_p1_s{42-44,48,49}, gru_tw2_s{47,49,52}

---

## Week Plan (Feb 9-15, Claude+Codex collaborated)

### Target: 0.2761 (top 100). Current: 0.2675. Gap: +0.0086
- Attention seed rotation alone **won't close this gap** — need speed gains + new model types
- ONNX/JIT blocked by Docker constraints → dynamic quantization instead

### Priority Order (revised after quantization FAIL, 2026-02-10)

**Quantization KILLED**: Dynamic quantization makes inference 2-2.5x SLOWER for batch=1 hidden=144 models. Overhead dominates at this scale. Benchmarked on both GRU and Attention models.

| P | Approach | Expected LB Gain | Risk | Effort |
|---|----------|-----------------|------|--------|
| P1 | **Expand GRU seeds** (13→30, re-rank top-5) | +0.003 to +0.006 | Low | 1 day Kaggle |
| P1 | **4G+3A ensemble** (swap weakest GRU for 3rd attn) | +0.002 to +0.004 | Medium (timeout) | 0.5 day |
| P2 | **Recency-weighted loss** (objective change) | +0.001 to +0.003 | Medium | 1 day Kaggle |
| P3 | **Microstructure features** (1-seed kill test ONLY) | -0.002 to +0.003 | High (0/2 record) | 0.5 day |
| KILLED | Dynamic quantization, ONNX, JIT, torch.compile, FP16 | — | — | — |

### Go/No-Go Gates (Codex-agreed)
- **4G+3A**: Must fit within ~3400s estimated time. Only submit if val improvement justifies timeout risk (19% margin vs 30% preferred).
- **Recency-weighted**: Must show val improvement over base config with same seed. Kill if negative after 2 seeds.
- **Microstructure**: Must show val improvement in 1-seed test. Kill immediately if negative. Cannot extend beyond Friday.

### Execution Plan

**Mon (Feb 10): S2/S4 Results + Quantization Benchmark + 4G+3A Evaluation**
- Process S2/S4 results → s2_s43_swap is new champion (0.2675) ✓
- Quantization benchmark → FAILED (2-2.5x slower) ✓
- 4G+3A offline evaluation with cached predictions

**Tue-Wed (Feb 11-12): GRU Seed Expansion (Kaggle)**
- Train ~20 new GRU seeds (10 tightwd_v2, 10 pearson_v1) on Kaggle
- Cache predictions, re-rank ensemble candidates

**Thu (Feb 13): Recency-Weighted Loss Training**
- Train 2-3 recency-weighted variants (different ramp schedules)
- Evaluate against base config

**Fri (Feb 14): Microstructure Kill Test + Review**
- Train 1 microstructure GRU (seed 42). Compare val to base.
- Review all results, build final ensembles

**Sat-Sun (Feb 15-16): Consolidate + Final Submissions**
- Build 2-3 final ensemble archetypes (safe / best / aggressive)
- Submit ranked ladder

### Submission Strategy Rules
1. **One hypothesis per submission** (no multi-change confounding)
2. Keep recurring **control** to detect LB noise/drift
3. **Promotion gates**: offline uplift → runtime safe → LB non-negative → confirm
4. **Reality check**: need ~0.285+ val to reach top 100. Speed + new objectives are the path, not reshuffling.

---

## Architecture & Key Files

### Active Configs (configs/)
| Config | Type | Loss | Key Diff |
|--------|------|------|----------|
| gru_derived_tightwd_v2.yaml | GRU | combined | Best GRU baseline |
| gru_attention_clean_v1.yaml | GRU+Attn | combined | Best attention model |
| gru_pearson_v1.yaml | GRU | pearson_combined | Metric-aligned loss |
| gru_recency_v1.yaml | GRU | combined+recency | Recency-weighted (untested) |
| gru_microstructure_v1.yaml | GRU | combined | +6 microstructure features (untested) |
| tcn_base_v1.yaml | TCN | combined | Causal TCN (untested) |

### Core Scripts (scripts/)
| Script | Purpose |
|--------|---------|
| train.py | Train single model from config + seed |
| export_ensemble.py | Build submission zip (optimized solution.py) |
| build_mixed_ensemble.py | Combine checkpoints using presets |
| validate_ensemble_local.py | Cache predictions, greedy/exhaustive search, diversity analysis |
| evaluate.py | Local evaluation on valid set |

### Source Code (src/)
- `src/models/gru_baseline.py` — GRU with input projection + LayerNorm + output MLP
- `src/models/gru_attention.py` — GRU + multi-head causal attention
- `src/models/tcn_model.py` — Causal TCN with depthwise separable convs (untested)
- `src/training/trainer.py` — Training loop with AMP, grad clip, early stopping
- `src/training/losses.py` — MSE, Combined, Huber, WeightedPearson, PearsonCombined
- `src/data/preprocessing.py` — DerivedFeatures, TemporalBuffer, InteractionBuilder, MicrostructureBuffer
- `src/data/dataset.py` — PyTorch Dataset from parquet

### Notebooks (notebooks/)
| Notebook | Status |
|----------|--------|
| 01-04 | Complete — EDA, features, HP sweep, seed analysis |
| 05_retrain_pearson_models.ipynb | Complete — pearson models |
| 06_gru_seed_expansion.ipynb | Complete — 13 new GRU models |
| 07_attention_seed_expansion.ipynb | Complete — 8 new attention models (seeds 45-52) |
| 08_attn_gpu_inference.ipynb | Complete — GPU batch inference for attention caching |

---

## Technical Notes
- Kaggle `%%bash` buffers output → use `os.system()` or `!` prefix instead
- Checkpoints must be stripped (remove optimizer/scheduler state) to fit 20MB zip
- Kaggle sessions don't persist → train + export + download in single session
- `.gitignore` excludes: *.pt, *.npz, *.zip, logs/slim/, submissions/
- export_ensemble.py generates optimized solution.py (feature cache + need_pred skip)
- validate_ensemble_local.py has 23 cached models for instant ensemble scoring
- Staging dir `logs/_staging/` has extracted checkpoints for submission building
