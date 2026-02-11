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

### Scoring Environment (from docs)
- **Docker**: `python:3.11-slim-bookworm` with PyTorch CPU (no GPU)
- **CPU**: 1 vCPU core | **RAM**: 16 GB | **Storage**: Local SSD
- **Time limit**: Docs say 60 minutes, but actual timeouts observed at **4200s (70 min)**
- **No internet**: Offline execution, all caches redirected to /app/
- **Submission**: `.zip` with `solution.py` at root + model files, max 20MB

### Submission Interface (from docs)
- `solution.py` must define `class PredictionModel` with `predict(self, data_point)` method
- `data_point` attributes: `seq_ix` (int), `step_in_seq` (int), `need_prediction` (bool), `state` (np.ndarray)
- Return `None` when `need_prediction is False`, else `np.ndarray` of shape `(2,)` for (t0, t1)
- Must reset recurrent state when `seq_ix` changes

### Test Data (from docs)
- **Test & Final sets**: ~1,500 sequences each (similar to validation's 1,444)
- **Total steps**: ~1.5M (1,500 seq x 1,000 steps)
- **Scored steps**: 99-999 per sequence (901 steps x ~1,500 seq = ~1.35M scored predictions)
- **Metric**: Weighted Pearson Correlation, averaged across t0 and t1
- **Two leaderboards**: Public (test set) during competition, Private (final set) for prizes

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

## Current State (as of 2026-02-13)

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
| s2_s43_swap | 0.2675 | 5G+2A | 3106s | Previous champion |
| champion_tcn_5g2a2t_v2 | 0.2683 | 5G+2A+2T | 3297s | TCN base s45+s46 |
| onnx_champion_control | 0.2683 | 5G+2A+2T ONNX | 2321s | ONNX lossless confirmed |
| **champion_mixed_tcn_v1** | **0.2689** | **5G+2A+2T** | **3308s** | **CURRENT CHAMPION** |
| champion_tcn_5g2a1t_v2 | 0.2681 | 5G+2A+1T | 3829s | 1T ablation |
| onnx_9g2a2t_v1 | 0.2677 | 9G+2A+2T ONNX | 3186s | 9G HURT, diluted |
| variant_b_2tw2_3p1 | 0.2662 | 5G+2A | 2975s | OVERFIT: top-5 from 81 seeds |
| s4_nb07s48_swap | 0.2646 | 5G+2A | 3670s | nb07_s48 worse than s50 |
| champion_v4_top2attn | TIMEOUT | 5G+2A | 4199s | Server variance |
| 5 GRU + 3 attn uniform | TIMEOUT | 5G+3A | 4200s | Too many attn |
| champion_tcn_5g2a2t v1 | TIMEOUT | 5G+2A+2T | 4200s | PyTorch TCN too slow |

### Val-to-LB Calibration (10 data points)
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
| variant_b_2tw2_3p1 | 0.2801 | 0.2662 | **-0.0139** | 7 (OVERFIT: top-5 of 81) |
| **champion_tcn_5g2a2t_v2** | **0.2808** | **0.2683** | **-0.0125** | **9 (5G+2A+2T, TCNFast)** |
| champion_tcn_5g2a2t v1 | 0.2808 | TIMEOUT | — | 9 (PyTorch TCN too slow) |
- **7-model gap (well-selected)**: mean -0.0088, range [-0.0103, -0.0080]
- **9-model gap (5G+2A+2T)**: -0.0125 — wider than 7-model, expected with more models
- **Cherry-picked from large pool**: gap -0.0139 (same as exhaustive -0.0141)
- **RULE: Selecting top-K from pool >20 seeds OVERFITS. Use contiguous/pre-registered seed ranges.**

### Champion champion_mixed_tcn_v1 Details
- **LB Score**: 0.2689 (+0.0006 over previous champion_tcn_5g2a2t_v2)
- **GRUs** (54% weight, 0.108 each): p1_s47, tw2_s50, tw2_s48, p1_s45, p1_s50
- **Attention** (26% weight, 0.130 each): attn_comb_s43 (old) + attn_nb07_s50 (new)
- **TCN** (20% weight, 0.100 each): tcn_base_s48 + tcn_k5_s42 (mixed k3/k5, TCNFast numpy)
- **Key insight**: Mixed TCN architectures (base k=3 + k5) beat same-config pair (s45+s46). k5 adds diversity.
- **ONNX version built**: onnx_champion_mixed_tcn_v1.zip — same models, ONNX GRU, est ~1905s (55% margin)
- **Val-LB gap**: -0.0125 (wider than 7-model -0.0088, expected with 9 models)

### Timing Data
- **Per GRU**: ~320s (stable across all submissions)
- **Per Attention**: 665-952s (HUGE server variance)
- **Per TCN (TCNFast)**: ~40s estimated (from 3297s total)
- **Server variance**: 45% — identical 5G+2A ran in 2895s, 3164s, and timed out at 4199s
- **5G+2A+2T**: 3297s with 21.5% margin to 4200s budget
- **MUST budget >30% margin** for timeout safety

---

## Proven Rules (from LB data)

### DO
- Use **contiguous/pre-registered seed ranges** for GRU selection (e.g., s42-50)
- Use **70/30 GRU/attention weighting** (0.14 per GRU, 0.15 per attn)
- Mix **old + new attention seeds** (diversity > same-batch)
- Use **combined-loss** attention only (pearson-loss attn hurts ensembles)
- Test **one variable at a time** per submission (clean A/B tests)

### DON'T
- **NEVER cherry-pick top-K from large seed pool** (gap -0.0139 from 81 seeds, same as exhaustive)
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
| **gru_tw2_s63** | **0.2736** | tightwd_v2 | cached + staged |
| **gru_p1_s79** | **0.2690** | pearson_v1 | cached + staged |
| **gru_p1_s63** | **0.2689** | pearson_v1 | cached + staged |
| **gru_p1_s87** | **0.2685** | pearson_v1 | cached + staged |
| **gru_p1_s67** | **0.2683** | pearson_v1 | cached + staged |
| gru_p1_s56 | 0.2681 | pearson_v1 | staged |
| gru_p1_s76 | 0.2677 | pearson_v1 | staged |
| gru_p1_s86 | 0.2672 | pearson_v1 | staged |
| gru_p1_s59 | 0.2672 | pearson_v1 | staged |
| gru_p1_s47 | 0.2668 | pearson_v1 | cached |
| gru_tw2_s60 | 0.2663 | tightwd_v2 | cached + staged |
| gru_tw2_s65 | 0.2655 | tightwd_v2 | staged |
| gru_tw2_s50 | 0.2654 | tightwd_v2 | cached + staged |

### Seed Expansion Progress
- **tw2 COMPLETE**: 32 seeds (s42-73), all extracted to _staging
- **p1 COMPLETE**: 49 seeds (s42-90), all extracted to _staging
- **Total**: 81 GRU seeds trained, 96 models registered in validate_ensemble_local.py
- **Cached**: 28 models in cache/predictions/ (13 original + 6 new expansion + 10 attention)

### Submission Variants — GRU-only (DO NOT SUBMIT — cherry-pick from 81 seeds overfits)
| Variant | GRUs | Val | LB | Gap | Notes |
|---------|------|-----|----|-----|-------|
| B (2tw2+3p1) | tw2_s63/s60 + p1_s79/s63/s87 | 0.2801 | **0.2662** | **-0.0139** | OVERFIT |
| A, C | similar cherry-picked | ~0.2800 | — | — | DO NOT SUBMIT |
| Champion | tw2_s50/s48 + p1_s47/s45/s50 | 0.2770 | **0.2675** | -0.0095 | Contiguous seeds = safe |

**Lesson confirmed**: Selecting top-K GRU from pool of 81 seeds gives -0.014 gap, identical to exhaustive search overfitting. Only contiguous/pre-registered seed ranges are safe.

### GRU Models (28 cached in cache/predictions/)

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

### TCN Models (5 cached in cache/predictions/)
| Model | Val | Status |
|-------|-----|--------|
| tcn_s45 | 0.2688 | cached + staged |
| tcn_s42 | 0.2672 | cached + staged |
| tcn_s43 | 0.2652 | cached + staged |
| tcn_s44 | 0.2637 | cached + staged |
| tcn_s46 | 0.2601 | cached + staged |
- TCN-GRU correlation: ~0.87 (strong diversity). TCN-Attn: ~0.81-0.85 (even more diverse)
- Inference: 70us/step numpy (TCNFast), ~101s per model on 1.44M steps

### Ready-to-Submit Zips
| Zip | Models | Val | p10 | Est LB | Est Time | Status |
|-----|--------|-----|-----|--------|----------|--------|
| **champion_tcn_5g2a2t_v2.zip** | 5G+2A+2T | **0.2808** | **0.2687** | ~0.2718 | ~3303s | **READY** |
| champion_tcn_5g2a1t.zip | 5G+2A+1T | 0.2806 | — | ~0.2716 | ~3200s | backup |

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

| P | Approach | Expected LB Gain | Risk | Effort | Status |
|---|----------|-----------------|------|--------|--------|
| DONE | **Expand GRU seeds** (13→81, re-rank top-5) | +0.003 to +0.006 | Low | 1 day | top-5 of 81 OVERFITS (-0.0139 gap) |
| DONE | **TCN as 3rd architecture** (5 seeds, numpy inference) | +0.003 to +0.006 | Medium | 3 days | val +0.0038, awaiting LB result |
| KILLED | **4G+3A ensemble** — no meaningful val improvement | — | — | — | — |
| KILLED | **Recency-weighted loss** — both seeds negative | — | — | — | — |
| KILLED | **Microstructure features** — 5-seed confirmation FAILED | — | — | — | — |
| KILLED | Dynamic quantization, ONNX, JIT, torch.compile, FP16 | — | — | — | — |

### Go/No-Go Gates (Codex-agreed) — ALL RESOLVED
- **4G+3A**: KILLED — no meaningful val improvement over 5G+2A.
- **Recency-weighted**: KILLED — both seeds (42: 0.2570, 43: 0.2633) underperformed baseline mean (0.2613).
- **Microstructure**: KILLED — 5-seed confirmation failed (mean delta +0.0009, needed +0.003).
- **TCN 3rd architecture**: APPROVED — val +0.0038, strong diversity (corr 0.87 with GRU). Awaiting LB result.
- **Variant B (cherry-picked GRUs)**: KILLED — gap -0.0139, confirmed top-K from large pool overfits.

### Execution Plan

**Mon (Feb 10): S2/S4 Results + Quantization + 4G+3A + Seed Expansion**
- Process S2/S4 results → s2_s43_swap is new champion (0.2675) ✓
- Quantization benchmark → KILLED (2-2.5x slower) ✓
- 4G+3A offline evaluation → KILLED (no meaningful improvement) ✓
- GRU seed expansion: 32 tw2 (Kaggle) + 40 p1 (Colab) = 72 new seeds ✓
- New top-5 GRU: tw2_s63(0.2736), p1_s79(0.2690), p1_s63(0.2689), p1_s87(0.2685), p1_s67(0.2683) ✓
- Built 3 submission variants (A/B/C), best val 0.2801 (+0.003 over champion) ✓
- Recency-weighted loss → KILLED (both seeds negative) ✓

**Tue (Feb 11): Variant B + Correlation-Aware + Micro Kill Test**
- Variant B submitted → 0.2662 OVERFIT (gap -0.0139, same as exhaustive) ✓
- Correlation-aware greedy selection → zero effect (lambda sweep found no improvement) ✓
- Microstructure 1-seed kill test → PASS (+0.0065) but inconclusive ✓

**Wed (Feb 12): Micro Confirmation + TCN Integration**
- Microstructure 5-seed confirmation → FAILED (mean +0.0009, 3/5 positive). KILLED. ✓
- TCN kill test (5 seeds s42-46) → ALL VIABLE. Mean 0.2650, best s45 (0.2688). ✓
- TCN cached (batch inference via model.forward()), diversity confirmed (corr 0.87 with GRU) ✓
- 5G+2A+2T ensemble: val 0.2808 (+0.0038 over champion). Built champion_tcn_5g2a2t.zip ✓
- Submitted champion_tcn_5g2a2t.zip → TIMED OUT (PyTorch TCN forward_step too slow) ✓

**Thu (Feb 13): TCNFast Optimization**
- Built TCNFast class (pure numpy inference) → 26.5x faster (70us vs 1859us per step) ✓
- Rebuilt as champion_tcn_5g2a2t_v2.zip with TCNFast ✓
- Verified correctness (max diff 4.77e-07 vs PyTorch) and timing (est ~3303s total) ✓
- **READY TO SUBMIT**: champion_tcn_5g2a2t_v2.zip (val 0.2808, est LB ~0.2718)
- Also built champion_tcn_5g2a1t.zip (5G+2A+1T backup, val 0.2806)

**Fri (Feb 14): Submit v2 + Evaluate**
- Submit champion_tcn_5g2a2t_v2.zip (5G+2A+2T with numpy TCN)
- If LB improves over champion: try TCN weight variants
- If timeout again: submit champion_tcn_5g2a1t.zip (5G+2A+1T, fewer TCN)

**Sat-Sun (Feb 15-16): Final Submissions**
- Build final ensemble archetypes based on all LB data
- Submit ranked ladder

### IMMEDIATE NEXT STEPS (for next chat session)
1. **Submit champion_tcn_5g2a2t_v2.zip** to Wunderfund platform (NOT Kaggle — this is a Wunderfund competition!)
   - File: `submissions/champion_tcn_5g2a2t_v2.zip` (7.4MB, verified)
   - Models: 5 GRU (champion) + 2 Attention + 2 TCN (numpy inference via TCNFast)
   - Val: 0.2808, est LB: ~0.2718, est timing: ~3303s
2. **If v2 times out**: Submit `submissions/champion_tcn_5g2a1t.zip` (5G+2A+1T backup, fewer TCN)
3. **If v2 succeeds**: Record LB score, compare to champion (0.2675), try TCN weight variants
4. **Decision tree after LB result**:
   - TCN helps (+0.005+): Train more TCN seeds, try higher TCN weight
   - TCN neutral (±0.003): Keep current, focus on other improvements
   - TCN hurts (-0.005+): Drop TCN, revert to 5G+2A champion

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
| gru_recency_v1.yaml | GRU | combined+recency | KILLED — both seeds negative |
| gru_microstructure_v1.yaml | GRU | combined | +6 microstructure features — KILLED (5-seed confirmation failed) |
| tcn_base_v1.yaml | TCN | combined | Causal TCN, 9K params, 5 seeds trained (s42-46) |

### Core Scripts (scripts/)
| Script | Purpose |
|--------|---------|
| train.py | Train single model from config + seed |
| export_ensemble.py | Build submission zip (optimized solution.py) |
| build_mixed_ensemble.py | Combine checkpoints using presets |
| validate_ensemble_local.py | Cache predictions, greedy/exhaustive search, diversity analysis |
| evaluate.py | Local evaluation on valid set |
| test_tcn_fast.py | Benchmark TCNFast (numpy) vs PyTorch speed + correctness |

### Source Code (src/)
- `src/models/gru_baseline.py` — GRU with input projection + LayerNorm + output MLP
- `src/models/gru_attention.py` — GRU + multi-head causal attention
- `src/models/tcn_model.py` — Causal TCN with depthwise separable convs (tested, 5 seeds viable)
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
| 09_gru_seed_expansion_v2.ipynb | Complete — tw2 seeds 54-73 (Kaggle) |
| 09_colab_p1_seeds.ipynb | Complete — p1 seeds 51-90 (Colab) |
| 10_colab_microstructure.ipynb | Complete — micro 5-seed confirmation (KILLED) |

---

## Technical Notes
- Kaggle `%%bash` buffers output → use `os.system()` or `!` prefix instead
- Checkpoints must be stripped (remove optimizer/scheduler state) to fit 20MB zip
- Kaggle sessions don't persist → train + export + download in single session
- `.gitignore` excludes: *.pt, *.npz, *.zip, logs/slim/, submissions/
- export_ensemble.py generates optimized solution.py (feature cache + need_pred skip)
- validate_ensemble_local.py has 96+ registered models (33 cached: 18 GRU + 10 Attn + 5 TCN)
- Staging dir `logs/_staging/` has extracted checkpoints for submission building

### TCN & TCNFast (added Feb 13)
- **TCN architecture**: Causal dilated depthwise-separable Conv1d, 6 blocks, ~9K params
- **TCN forward_step (PyTorch)**: Ring buffer approach, ~1859us/step — TOO SLOW for scoring server
- **TCNFast (numpy)**: Extracts weights to numpy arrays, does all inference in numpy — 70us/step (26.5x faster)
- **TCNFast is auto-created**: In solution.py, after PyTorch model loads weights, it gets replaced by `TCNFast(pt_model)` automatically
- **Correctness verified**: Max abs diff vs PyTorch = 4.77e-07 (float32 precision)
- **export_ensemble.py** now includes TCNFast class and auto-conversion in PredictionModel.__init__
- **validate_ensemble_local.py** uses batch `model.forward()` for TCN (17s per model vs 5600s step-by-step)
- **scripts/test_tcn_fast.py**: Benchmark script comparing PyTorch vs numpy TCN speed + correctness
