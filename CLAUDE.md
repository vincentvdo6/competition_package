# Wunderfund Predictorium RNN Challenge

## CRITICAL: Notebook Rules (NEVER VIOLATE)
- **NEVER use `os.system()` in Colab/Jupyter notebooks** — it swallows ALL output silently.
- **ALWAYS use `subprocess.Popen`** with streaming:
```python
proc = subprocess.Popen(
    [sys.executable, "-u", "scripts/train.py", ...args...],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
)
for line in proc.stdout:
    print(line, end="", flush=True)
proc.wait()
```
- **ALWAYS commit+push before running Colab notebooks** — they `git clone` from remote.

---

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
- **Two inference modes**: Step-by-step (stateful hidden) OR windowed (last N steps, fresh hidden)

### Submission Budget
- **5 submissions per day**
- **~1 month remaining** in competition

---

## Current State (as of 2026-02-17)

### STATUS: RECIPE-LEVEL SATURATION — STUCK IN LOCAL BASIN
- **Current best**: vanilla_ens10 = **0.2885 LB, Rank 73/4728** (PAST TOP 100!)
- **Vanilla GRU** (plain GRU + linear output, raw 32 features, no norm, MSE loss) has POSITIVE val-to-LB gap
- **EVERY modification to vanilla recipe has been tested and killed or neutral** (see full list below)
- **Gap to top-1 (0.3240) is +0.035** — not a tweak, requires fundamentally different approach
- **Codex verdict (Feb 17)**: "yes, you're likely stuck in this local basin"

### Key LB Scores (43 submissions total)
| Submission | Score | Duration | Gap | Notes |
|-----------|-------|----------|-----|-------|
| **vanilla_ens10** | **0.2885** | 1702s | +0.0177 | 10 vanilla h=64 models, PyTorch, **PB** |
| vanilla_ens20_onnx | 0.2884 | 841s | — | 20 models ONNX, no improvement over 10 |
| mixed_ens11_onnx | 0.2868 | 465s | — | 10 base + 1 Pearson blend, WORSE |
| greedy_top5_onnx | 0.2862 | 219s | — | 5 greedy-selected, WORSE |
| greedy_top3_onnx | 0.2856 | 135s | — | 3 greedy-selected, WORSE |
| parity_v1_s43 | 0.2814 | 184s | +0.0077 | Single vanilla h=64 |
| vanilla_s59 | 0.2764 | 380s | +0.0037 | Single vanilla h=64 |
| Official baseline | 0.2761 | 501s | +0.017 | h=64 3L plain GRU, ONNX windowed |
| champion_mixed_tcn_v1 | 0.2689 | 3308s | -0.0125 | OLD champion (complex pipeline) |
| baseline_match_s42 | 0.2394 | 87s | -0.034 | Complex pipeline, CATASTROPHIC |

### Val-to-LB Gap Rules
- **Vanilla GRU**: POSITIVE gap (+0.004 to +0.018) — test set is EASIER than val
- **Complex pipeline**: NEGATIVE gap (-0.008 to -0.034) — complex models overfit
- **Ensemble amplifies positive gap**: single +0.0077, 10-model +0.0177
- **20 models = 10 models**: diminishing returns from same-recipe seeds
- **Mixing recipes HURTS**: Pearson blend diversity (0.78 corr) scored WORSE on LB

### Available Vanilla Models (23 seeds, h=64 3L, gru_parity_v1 config)
- Checkpoints in `logs/vanilla_all/gru_parity_v1_seed*.pt`
- Top seeds by val: s43(0.2737), s59(0.2727), s46(0.2716), s63(0.2713), s55(0.2705)
- Mean val (all 23): 0.2689, Top-10 mean: 0.2704

### Timing & Budget
- **Vanilla h=64 PyTorch**: ~184s/model (but 2x server variance — s59 took 380s)
- **Vanilla h=64 ONNX**: ~42s/model (confirmed: 20 models in 841s)
- **Safe model counts**: PyTorch 10 max, ONNX 40+ possible
- **MUST budget for 2x server variance** (not just 45%)

---

## Strategic Status (Feb 16) — ALL PREVIOUS STRATEGIES EXHAUSTED

### What's been tried and KILLED
Every diversity and modification strategy has been tested and confirmed negative:

**Architecture**: input_proj, LayerNorm, MLP output, attention, TCN, transformer, CVML, SE-Net gate, LSTM
**Features**: derived 42, microstructure, lag diffs, z-score, RevIN (full+causal)
**Loss**: Pearson blend, Huber, PearsonPrimary, aux heads, recency weighting, innovation aux head
**Training**: full-data, stronger reg, SAM, EMA, cosine warmup, Adam, higher LR, bagging, Mixup augmentation
**Scaling**: h=128/144/192/256 (all worse than h=64)
**Ensemble diversity**: recipe variants (varA/B/C), LSTM, checkpoint epochs, greedy selection, Pearson blend mixing, 20 models
**Inference**: windowed (same as step-by-step)
**Distribution shift**: adversarial-weighted fine-tune on val-like 30% subset (zero-delta), tree model blend (decorrelated but too weak)
**Post-processing**: prediction neutralization (signal IS linear, removing it destroys predictions)

### Remaining Untried Ideas (Codex, Feb 17)
1. **Regime-specialist ensemble + test-time gating** — highest EV but mixing has historically hurt
2. **Self-supervised pretraining** (masked/next-step reconstruction) then fine-tune — genuinely untested
3. **Massive candidate library (100+) + LB-aware blend** — contradicts our evidence (20=10, greedy=worse)

---

## Architecture & Key Files

### Winning Config
- **`configs/gru_parity_v1.yaml`** — THE winning recipe (vanilla h=64, raw32, no norm, MSE)

### Build Scripts
| Script | Purpose |
|--------|---------|
| `scripts/build_vanilla_ensemble.py` | Build vanilla ensemble zip (PyTorch or ONNX with `--onnx`) |
| `scripts/build_parity_submission.py` | Build single vanilla model zip |
| `scripts/train.py` | Train single model from config + seed |
| `scripts/export_ensemble.py` | Build complex pipeline ensemble zip (OLD, for tightwd/attn/tcn) |
| `scripts/validate_ensemble_local.py` | Cache predictions, greedy search (OLD pipeline only) |

### Key Source Files
- `src/models/gru_baseline.py` — GRU model (vanilla mode: `vanilla: true` + `output_type: linear`)
- `src/training/trainer.py` — Training loop (supports Mixup, SAM, innovation aux, all KILLED)
- `src/training/losses.py` — MSE, Combined, Huber, WeightedPearson, PearsonCombined
- `src/data/dataset.py` — PyTorch Dataset (supports `lag_features` flag, KILLED)

### Analysis Scripts
- `scripts/adversarial_validation.py` — Train/val distribution shift detection (AUC=0.959)
- `scripts/tree_model_probe.py` — HistGradientBoosting probe + GRU correlation + blend sweep
- `scripts/create_vallike_subset.py` — Create val-like train subset from adversarial weights

### KILLED Approaches (comprehensive — ALL tested and confirmed negative)
- **Architecture**: input_proj, LayerNorm, MLP output, attention, TCN, transformer, CVML, SE-Net gate, LSTM
- **Features**: derived 42, microstructure, lag diffs, raw32+zscore, raw32 no-norm (with complex arch)
- **Loss**: PearsonPrimary, Pearson blend, Huber, aux heads, recency weighting, CVML+CoRe, innovation aux head
- **Training**: full-data, stronger reg, SAM, EMA, cosine warmup, Adam, higher LR, bagging (85%), Mixup (alpha=0.2, -0.0004 neutral)
- **Normalization**: RevIN (full-seq and causal), z-score removal
- **Scaling**: h=128/144/192/256 (all worse than h=64 on vanilla)
- **Inference**: windowed (same as step-by-step on LB)
- **Ensemble diversity**: 20 models (=10), greedy selection (WORSE), recipe variants (varA/B/C all worse), LSTM (-0.011), checkpoint epochs (useless), Pearson blend mixing (0.78 corr but WRONG direction on LB)
- **Distribution shift**: val-like subset fine-tune (zero-delta), tree model blend (decorrelated but every alpha hurts)
- **Post-processing**: prediction neutralization (signal IS linear relationship with features)

### Key Discoveries (Feb 17)
- **Adversarial validation AUC = 0.959**: train and val are almost perfectly separable. Massive distribution shift.
- **Top discriminators**: v3_mean, p6_min, p0_min, p6_mean (level-based summary stats)
- **97% of train seqs** get adversarial weight <0.5, only 72/10721 are "val-like"
- **This explains everything**: why complex models overfit, why vanilla GRU generalizes, why positive val-LB gap exists

---

## Technical Notes
- Kaggle `%%bash` buffers output -> use `os.system()` or `!` prefix instead
- Colab: ALWAYS use `subprocess.Popen` with PIPE for output (never `os.system()`)
- Checkpoints must be stripped (remove optimizer/scheduler state) to fit 20MB zip
- Kaggle sessions don't persist -> train + export + download in single session
- `.gitignore` excludes: *.pt, *.npz, *.zip, logs/slim/, submissions/
- PredictionModel init signature: `__init__(self, model_path="")` — server calls with no args
- ONNX export uses opset 17, `VanillaGRUStep` wrapper (gru + fc, no input_proj/norm)
- Windows cp1252 encoding: avoid unicode arrows in Python template strings

### Submissions Folder Structure
```
submissions/
  ready/              <- Submit next
  baseline/           <- Official baseline reference
  archive/
    scored/           <- Previously scored
    timeout/          <- Timed out (ens15, etc.)
    unsent/           <- Never submitted
```
