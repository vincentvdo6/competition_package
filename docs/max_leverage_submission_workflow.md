# Max-Leverage Submission Workflow

This runbook implements the daily collaboration system for leaderboard progress.

## Locked Defaults
- Slot policy: `3 then adapt`
- Kill policy: `2-strike strict` (`delta <= -0.0004` twice in same family)
- One optional mechanistic final probe is allowed
- PB baseline default: `0.2885` (update when beaten)

## Core Commands

### 1) Pre-submit gate checks
Run before every submit slate:

```powershell
python scripts/check_submission_zip.py submissions/ready/*.zip
```

What this checks:
- zip size `<20MB`
- `solution.py` exists at root
- model files exist at root
- `PredictionModel` exists
- weight/model config summary when parseable

### 2) Record a leaderboard result and get next action
Use the compact update flow:

```powershell
python scripts/submission_decision_engine.py record `
  --family recovery_ptarget_topk `
  --zip recovery_ptarget_t07_t16_onnx.zip `
  --score 0.2879 `
  --pb 0.2885 `
  --final-probe-zip recovery_ptarget_t07_t18_onnx.zip
```

This outputs:
- `band`
- `delta`
- family state (`active`, `final_probe_only`, `killed`)
- `next_action`
- compact line format for chat logs

### 3) Inspect current family states

```powershell
python scripts/submission_decision_engine.py status
python scripts/submission_decision_engine.py status --family recovery_ptarget_topk
```

### 4) Reset one family (fresh branch start)

```powershell
python scripts/submission_decision_engine.py reset-family --family recovery_ptarget_topk
```

## Banding Rules
- `delta = score - PB`
- `strong_win`: `delta >= +0.0003`
- `soft_win`: `0.0000 <= delta < +0.0003`
- `near_miss`: `-0.0004 < delta < 0.0000`
- `clear_fail`: `delta <= -0.0004`

## Adaptation Rules
- After `strong_win`: exploit tight neighbors with remaining slots
- After `near_miss`: run one directional probe, then decide
- After `clear_fail`: do not widen in same direction
- After two `clear_fail` in same family: kill family, except one optional mechanistic final probe

## Daily Operating Loop
1. Read `CLAUDE.md`, latest `logs/findings.md`, and inspect `submissions/ready/`
2. Build ranked `A/B/C` slate with explicit kill/continue conditions
3. Run pre-submit checks on candidate zips
4. Submit A, log result via decision engine
5. Re-rank B/C from live signal, submit next
6. At end of day, write `continue`, `kill`, and `tomorrow initial 3-pack` in `logs/findings.md`
