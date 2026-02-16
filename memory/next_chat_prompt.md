# Seamless Chat Continuation — Windowed Inference Breakthrough (Feb 13, 2026)

Paste this entire block into your other chat. It contains the full context from the session that just ended.

---

## IMMEDIATE ACTION: Submit 3 zips to wunderfund.io LB

All 3 are already built and sitting in `submissions/`. Submit in this order:

1. **`submissions/official_baseline_control.zip`** (252 KB) — exact repack of official baseline. Safety anchor, should match known 0.2761 LB.
2. **`submissions/windowed_tw2_s42_w100.zip`** (967 KB) — tightwd_v2 (h=144 2L, derived42) with windowed inference (window=100). Stable model, testing if windowed mode helps on LB.
3. **`submissions/windowed_bm_s42_w100.zip`** (283 KB) — baseline_match (h=64 3L, raw32) with windowed inference (window=100). **THE critical A/B test** — same arch as official baseline but our training. Step-by-step scored catastrophic 0.2394; windowed should fix it.

Codex agreed on this submission plan.

---

## CRITICAL DISCOVERY THIS SESSION: Windowed Inference

### The Official Baseline Uses Windowed Inference
The official baseline (`example_solution/solution.py`) does NOT use step-by-step stateful GRU inference. Instead:
- Stores full sequence history in a buffer
- Each prediction step: feeds last 100 timesteps as batch `(1, 100, 32)` to ONNX
- Takes the **last timestep output** `predictions[0, -1, :]`
- GRU hidden state is NOT carried between steps — fresh each call

This is fundamentally different from our step-by-step approach (feed 1 step at a time, carry hidden state).

### Local Validation Results (A/B Comparison)

| Model | Inference Mode | Val Score | Known LB | Gap |
|-------|---------------|-----------|----------|-----|
| Official baseline (h=64 3L raw) | Windowed | 0.2595 | **0.2761** | **+0.0166** |
| baseline_match (h=64 3L raw, OUR training) | **Windowed** | **0.2741** | **???** | **???** |
| baseline_match (h=64 3L raw) | Step-by-step | 0.2738 | **0.2394** | **-0.0344** |
| tightwd_v2 (h=144 2L derived42) | Windowed | 0.2578 | **???** | **???** |
| tightwd_v2 (h=144 2L derived42) | Step-by-step | 0.2584 | 0.2580 | -0.0004 |

### Key Takeaways
1. **Windowed vs step-by-step makes almost NO difference on val** (+0.0003 for baseline_match, -0.0006 for tightwd_v2)
2. **Official baseline has a POSITIVE val-to-LB gap** (+0.0166) — test set is EASIER than val for this model
3. **baseline_match step-by-step has catastrophic NEGATIVE gap** (-0.034) — but windowed baseline_match might fix it on LB
4. **tightwd_v2 has near-zero gap** regardless of inference mode — stable model

### Why This Matters
If `windowed_bm_s42_w100.zip` scores near 0.27+ on LB, it confirms:
- Windowed inference is critical for generalization on the test set
- The catastrophic 0.2394 was caused by step-by-step hidden state drift on test data
- We should rebuild ALL models with windowed inference for LB

If it still scores poorly, the gap is in training recipe (official baseline trained differently).

---

## New Script Created: `scripts/build_windowed_submission.py`

Builds single-model windowed inference submission zips. Usage:
```
python scripts/build_windowed_submission.py \
    --checkpoint logs/_staging/gru_derived_tightwd_v2_seed42.pt \
    --normalizer logs/_staging/normalizer_gru_derived_tightwd_v2_seed42.npz \
    --config configs/gru_derived_tightwd_v2.yaml \
    --window-size 100 \
    --output submissions/windowed_tw2_s42.zip
```

Key: exports ONNX with dynamic seq_len axis (no hidden state I/O), solution.py buffers history and feeds last N steps each prediction.

---

## Round 5 Results This Session — ALL KILLED

Before the windowed discovery, we tested:
1. **Full-data training** (train+val combined): 7ep=0.2572, 18ep=0.2455 vs 0.2580 baseline. KILLED.
2. **Stronger regularization** (WD=2e-4, dropout=0.30, OneCycleLR): -0.0049 delta, 0/3 positive. KILLED.

---

## Updated KILLED List (comprehensive)
Everything that's been tried and failed:
- Derived/temporal/interaction/microstructure features
- Larger GRU h=256/384, EMA, cosine warmup
- PearsonPrimaryLoss, aux heads, small transformer
- Recency loss, 9G ensemble, exhaustive combo, chrono init, augmentation, SWA
- baseline_match architecture with step-by-step inference
- Windowed training, Adam, higher LR, longer training
- Full-data training (train+val combined)
- Stronger regularization (WD=2e-4, dropout=0.30, OneCycleLR)

---

## What Happens After LB Results Come Back

### If windowed_bm scores well (>0.265):
- **Windowed inference is the key.** Rebuild champion ensemble (5G+2A+2T) with windowed inference for all GRU models.
- Need to create `build_windowed_ensemble.py` (multi-model windowed inference).
- May need to retrain with windowed training too (train models expecting windowed context).

### If windowed_bm scores poorly (<0.245):
- Problem is in training recipe, not inference mode.
- Focus on understanding what makes official baseline training different.
- May need to try different optimizer settings, learning rate schedules, or data preprocessing.

### If official_baseline_control confirms 0.2761:
- Good — our repack is correct, server is consistent.

---

## File State
- `scripts/build_windowed_submission.py` — NEW, builds single-model windowed zips
- `memory/MEMORY.md` — updated with full-data + regularization kills
- 3 submission zips ready in `submissions/`
- All other files unchanged from previous sessions

## Submission Budget
- 5 subs/day. Using 3 today for windowed A/B tests. 2 remaining.
