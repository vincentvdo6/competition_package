# Operational Memory (Active)

## Persistent User Directives
- Always read `CLAUDE.md` and this `memory/MEMORY.md` at the start of a new chat.
- Keep repo state up to date after each chat:
  - `logs/findings.md`
  - `logs/submission_decision_state.json` (when score updates are provided)
- Submission hygiene (locked):
  - do not auto-clear `submissions/ready/` at end of chat
  - archive from `submissions/ready/` only after score/result is provided or when artifact is clearly outdated/superseded
  - move archived files to `submissions/archive/unsent/cleanup_<timestamp>/` (never delete)
  - keep unscored, active candidates in `submissions/ready/`
- Submission naming (locked):
  - format: `monDD-b<batch>-t<ab>-w<nnn>-s<nnn>-onnx.zip`
  - keep names short, sortable, and readable
  - include only decision-relevant knobs in `w`/`s` tokens

## Competition Working Defaults
- Daily slot policy: `3 then adapt`
- Family kill policy: `2-strike strict` at `delta <= -0.0004`
- Control mode: high autonomy (assistant builds/ranks, user submits)
- Active challenge docs source-of-truth: `/predictorium/docs/submission_guide`

## Current Anchor (as of Feb 24, 2026)
- **PB: `0.2907`** (`mix_v10_kf5best_fold2` — 10v + 5kf best-per-fold + fold2, 15 models, 33% kf)
- 25 submissions remaining (5 used today), deadline March 1

### Feb 24 Submission Results (ALL scored)
| Zip | Score | Kf Models | Kf Influence | Finding |
|-----|-------|-----------|-------------|---------|
| mix_v10_kf12_equal | 0.2897 | 12 (all s42/43/44) | 55% | Brute force kf helped vs 4kf |
| mix_v10_kf4best | 0.2898 | 4 (best per fold) | 29% | Selection > brute force |
| mix_v10_kf12_vup2 | 0.2896 | 12 (vanilla w=2) | 37.5% | Vanilla upweight didn't help |
| **mix_v10_kf5best_fold2** | **0.2907 PB** | 5 (best/fold + fold2) | 33% | **FOLD2 IS THE SECRET WEAPON** |

### BREAKTHROUGH: Fold2 Diversity (Feb 24)
- **+0.0009 jump** from adding fold2_seed42 to best-per-fold ensemble
- fold2_seed42 has lowest corr with vanilla: t1=0.8339 (vs 0.94+ for other folds)
- fold2 is structurally different — trained on different data split than other folds
- PB config: 10v + f0s44 + f1s43 + f2s42 + f3s42 + f4s44 (15 models, equal weight)
- **Kfold selection + fold2 diversity = winning formula**

### Key Findings (Feb 24)
- **Fold2 is the key diversifier**: corr=0.83 on t1 vs 0.94+ for other folds
- **Kfold selection > brute force**: best-per-fold beats all-12kf
- **~33% kf influence with fold2 is optimal** (was 29% without fold2)
- Best-per-fold picks: f0=s44, f1=s43, f2=s42, f3=s42, f4=s44
- Kfold predictions cached in `cache/kfold_preds/` (14 models)

### Active Direction — Exploit Fold2
- Add fold2_seed43 (second fold2 model) to ensemble
- Try different fold2 weights (upweight since it's the key contributor)
- Per-target kfold selection with fold2 emphasis
- Train fold2 with more seeds (44, 45, 46) for better fold2 pool
- Train fold2_seed44 (missing — only have seed42 and seed43)
