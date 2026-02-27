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

## User Environment
- Google student plan (not free tier) — Antigravity gives access to Gemini 3.1 Pro
- Gemini CLI installed (v0.29.6) but 3.1 Pro NOT available yet on student plan
- Three-way collab (Claude + Codex + Gemini) POSTPONED until 3.1 Pro rolls out to student tier

## Competition Working Defaults
- Daily slot policy: `3 then adapt`
- Family kill policy: `2-strike strict` at `delta <= -0.0004`
- Control mode: high autonomy (assistant builds/ranks, user submits)
- Active challenge docs source-of-truth: `/predictorium/docs/submission_guide`

## Current Anchor (as of Feb 26, 2026)
- **PB: `0.2927`** (`mix_v10_kf5_fold2w7_onnx` — 10v + 5kf, fold2_seed42 w=7)
- **CEILING CONFIRMED**: w=7, w=8, w=10 all score 0.2927 (symmetric weight exhausted)
- ~9 submissions remaining, deadline March 1
- **Per-target asymmetry KILLED** (2 tests: 0.2890, 0.2897 — both well below PB)

### Symmetric Weight Curve (COMPLETE — ceiling found)
| Weight | LB | Delta |
|--------|------|-------|
| w=1 | 0.2907 | — |
| w=3 | 0.2918 | +0.0011 |
| w=4 | 0.2922 | +0.0004 |
| w=5 | 0.2924 | +0.0002 |
| w=6 | 0.2926 | +0.0002 |
| w=7 | **0.2927** | +0.0001 |
| w=8 | 0.2927 | +0.0000 |
| w=10 | 0.2927 | +0.0000 |

### Per-Target Asymmetry — KILLED (Feb 26)
Val grid showed t1 benefits from fold2 weight, but LB says NO:
- t0w7-t1w30: 0.2890 (-0.0037) — aggressive, crashed
- t0w7-t1w15: 0.2897 (-0.0030) — conservative, still crashed
- **Any deviation from symmetric weighting hurts on LB**

### Feb 25-26 Submission Results (ALL scored)
| Zip | Score | Config | Delta |
|-----|-------|--------|-------|
| mix_v10_kf5_fold2w3 | 0.2918 | fold2_s42 w=3 | +0.0011 |
| mix_v10_kf5_fold2w4 | 0.2922 | fold2_s42 w=4 | +0.0004 |
| mix_v10_kf5_fold2w5 | 0.2924 | fold2_s42 w=5 | +0.0002 |
| mix_v10_kf5_fold2w6 | 0.2926 | fold2_s42 w=6 | +0.0002 |
| **mix_v10_kf5_fold2w7** | **0.2927** | fold2_s42 w=7 | **+0.0001** |
| mix_v10_kf5_fold2w8 | 0.2927 | fold2_s42 w=8 | +0.0000 |
| mix_v10_kf5_fold2w10 | 0.2927 | fold2_s42 w=10 | +0.0000 |
| mix_v10_kf6_fold2w4_s45 | 0.2921 | w=4 + fold2_s45 | KILLED (below w=4) |
| feb26-pt-t0w7-t1w30 | 0.2890 | asymmetric t0=7/t1=30 | KILLED (-0.0037) |
| feb26-pt-t0w7-t1w15 | 0.2897 | asymmetric t0=7/t1=15 | KILLED (-0.0030) |

### Fold2 Seed Inventory (all in logs/vanilla_all/)
| Seed | Val | corr_t1 with vanilla |
|------|-----|---------------------|
| 42 | 0.3094 | **0.8339** (key outlier) |
| 43 | 0.3016 | 0.9437 |
| 44 | 0.2908 | 0.9247 |
| 45 | 0.2983 | 0.9301 |
| 46 | 0.3020 | 0.9362 |
- seed42 diversity is SEED-SPECIFIC, not fold-specific. New seeds much less diverse.
- fold2_seed45 LB-confirmed KILLED (hurts when added to ensemble)

### Final Submission Strategy (3 picks for private LB)
| Pick | Submission | Public LB | Rationale |
|------|-----------|-----------|-----------|
| 1 | mix_v10_kf5_fold2w7 | **0.2927** | Current PB |
| 2 | TBD | pending | Need new direction |
| 3 | mix_v10_kf5best_fold2 | 0.2907 | Conservative backup |

### Key LB Scores (top, descending)
| Submission | Score | Date |
|-----------|-------|------|
| **mix_v10_kf5_fold2w7** | **0.2927** | Feb 26 |
| mix_v10_kf5_fold2w8 | 0.2927 | Feb 26 |
| mix_v10_kf5_fold2w10 | 0.2927 | Feb 26 |
| mix_v10_kf5_fold2w6 | 0.2926 | Feb 25 |
| mix_v10_kf5_fold2w5 | 0.2924 | Feb 25 |
| mix_v10_kf5_fold2w4 | 0.2922 | Feb 25 |
| mix_v10_kf5_fold2w3 | 0.2918 | Feb 25 |
| mix_v10_kf5best_fold2 | 0.2907 | Feb 24 |
| mix_v10_kf4best | 0.2898 | Feb 24 |
| mix_v10_kf12_equal | 0.2897 | Feb 24 |
| mix_v10_kf4_equal | 0.2895 | Feb 23 |
| t07_t18_w64x175 | 0.2886 | Feb 21 |
| vanilla_ens10 | 0.2885 | Feb 15 |
