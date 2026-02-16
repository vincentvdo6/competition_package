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

## Current State (as of 2026-02-15)

### BREAKTHROUGH: Vanilla GRU Paradigm (Feb 14-15)
- **Parity audit** revealed: our complex pipeline (input_proj+LayerNorm+MLP+derived+zscore) HURTS generalization
- **Vanilla GRU** (plain GRU + linear output, raw 32 features, no norm, MSE loss) has POSITIVE val-to-LB gap
- **Current best**: vanilla_ens10 = **0.2885 LB, Rank 73/4728** (PAST TOP 100!)

### Key LB Scores
| Submission | Score | Duration | Gap | Notes |
|-----------|-------|----------|-----|-------|
| **vanilla_ens10** | **0.2885** | 1702s | +0.0177 | 10 vanilla h=64 models, PyTorch |
| vanilla_ens20_onnx | 0.2884 | 841s | — | 20 models ONNX, no improvement over 10 |
| parity_v1_s43 | 0.2814 | 184s | +0.0077 | Single vanilla h=64 |
| vanilla_s59 | 0.2764 | 380s | +0.0037 | Single vanilla h=64 |
| Official baseline | 0.2761 | 501s | +0.017 | h=64 3L plain GRU, ONNX windowed |
| champion_mixed_tcn_v1 | 0.2689 | 3308s | -0.0125 | OLD champion (complex pipeline) |

### Val-to-LB Gap Rules
- **Vanilla GRU**: POSITIVE gap (+0.004 to +0.018) — test set is EASIER than val
- **Complex pipeline**: NEGATIVE gap (-0.008 to -0.034) — complex models overfit
- **Ensemble amplifies positive gap**: single +0.0077, 10-model +0.0177
- **20 models = 10 models**: diminishing returns from same-recipe seeds

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

## Next Steps (Codex-agreed, Feb 15)

### Bottleneck: Model quality/diversity, NOT ensemble size
20 ONNX models scored 0.2884 (same as 10 PyTorch at 0.2885). More same-recipe seeds don't help.

### Priority Order
| P | Approach | Expected Impact | Status |
|---|----------|----------------|--------|
| 1 | **Recipe diversity** — vary LR/WD/dropout/scheduler on h=64 vanilla | High (decorrelated predictions) | TODO |
| 2 | **Greedy ensemble selection** — forward selection on val predictions | High (better than flat top-K) | TODO |
| 3 | **Vanilla LSTM** — same simplicity, different inductive bias | Medium (architectural diversity) | TODO |
| 4 | **Checkpoint diversity** — keep 2-3 late checkpoints per run | Low-Medium (free diversity) | TODO |

### Specific Recipe Variations to Train
All h=64, 3L, vanilla+linear, raw32, no norm:
- **Base (gru_parity_v1)**: LR=0.001, WD=0, dropout=0, MSE, ReduceOnPlateau
- **Variant A**: LR=0.0005, WD=1e-5, dropout=0.05
- **Variant B**: LR=0.002, WD=0, dropout=0, CosineAnnealing
- **Variant C**: LR=0.001, WD=1e-4, dropout=0.1
- Train 3 seeds each, select by val + diversity (prediction correlation)

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
- `src/training/trainer.py` — Training loop (SAM optimizer added but KILLED)
- `src/training/losses.py` — MSE, Combined, Huber, WeightedPearson, PearsonCombined
- `src/data/dataset.py` — PyTorch Dataset (supports `lag_features` flag, KILLED)

### KILLED Approaches (comprehensive)
Everything below has been tested and confirmed negative:
- **Architecture**: input_proj, LayerNorm, MLP output, attention, TCN, transformer, CVML, feature gate
- **Features**: derived 42, microstructure, lag diffs, raw32+zscore, raw32 no-norm (with complex arch)
- **Loss**: PearsonPrimary, aux heads, recency weighting, CVML+CoRe
- **Training**: full-data, stronger reg, SAM, EMA, cosine warmup, Adam, higher LR
- **Normalization**: RevIN (full-seq and causal), z-score removal
- **Scaling**: h=128/144/192/256 (all worse than h=64 on vanilla)
- **Inference**: windowed (same as step-by-step on LB)
- **Ensemble**: 20 models (same as 10 models, diminishing returns)

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
