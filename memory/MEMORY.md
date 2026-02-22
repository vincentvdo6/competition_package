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

## Current Anchor (as of Feb 20, 2026)
- PB: `0.2886`
- Best current branch: per-target vanilla topK (`recovery_ptarget_topk`) with `t07_t18` and heavier seed64 on `t1`.
