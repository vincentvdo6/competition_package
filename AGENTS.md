# Repository Agent Playbook

## Mission
- Maximize leaderboard progress per day under the 5-submission limit.
- Optimize for real signal and decision value, not local-validation vanity.
- Prefer robust, simple, high-transfer RNN paths unless evidence says otherwise.

## Operational Defaults (Locked)
- Slot policy: `3 then adapt`.
- Kill policy: `2-strike strict` within a family when `delta <= -0.0004`.
- Control mode: high autonomy for build/ranking, user decides submit clicks.
- PB reference default: `0.2885` (update when beaten).

## Primary Workflow Source
- At the start of every new chat or task, read `CLAUDE.md` before taking action.
- Treat `CLAUDE.md` as the main strategy and operating reference for this repository.
- If `CLAUDE.md` is missing or unreadable, state that immediately and continue with best-effort execution.
- For challenge runtime constraints, prefer active challenge docs under `/predictorium/docs/*`, with `submission_guide` as primary when `rules` conflicts.

## Fast Startup Checklist (Every Chat)
1. Read `CLAUDE.md`.
2. Read the latest section of `logs/findings.md`.
3. Inspect `submissions/ready/` and `submissions/archive/` state.
4. Confirm current goal: exploit (beat PB) or explore (maximize information).
5. Execute directly unless user asks for planning only.

## Decision Framework for Submission Slates
- Always anchor around the best recent live score branch.
- Do not spend slots on already-killed families unless explicitly requested.
- Prefer small, attributable deltas between candidates so outcomes are interpretable.
- Use fewer slots (often 3) when uncertainty is high, leaving room for same-day adaptation.
- Use all 5 slots only when the slate is information-efficient and branch-complete.
- When uncertain, choose the option that yields the most decision value per slot.

## Decision Engine (Score Bands)
- Compute `delta = score - PB`.
- `strong_win`: `delta >= +0.0003`
- `soft_win`: `0.0000 <= delta < +0.0003`
- `near_miss`: `-0.0004 < delta < 0.0000`
- `clear_fail`: `delta <= -0.0004`
- Family rule: two `clear_fail` results => kill family, except one optional mechanistic final probe.
- Adaptation:
  - after `strong_win`: exploit tight neighbors
  - after `near_miss`: run one directional probe, then decide
  - after `clear_fail`: do not widen sweep in same direction

## Build and Verification Guardrails
- Use reproducible scripts and explicit seed lists.
- Keep submission zips under 20MB.
- Ensure `solution.py` contract is valid:
  - `PredictionModel(model_path=\"\")`
  - reset state on `seq_ix` change
  - return `None` when `need_prediction=False`
  - return shape `(2,)` otherwise
- Validate archive contents before handoff (`solution.py` + required model files).
- Budget runtime with safety margin and account for server variance.
- Use `scripts/check_submission_zip.py` as the default pre-submit checker.

## Artifact Hygiene
- Keep `submissions/ready/` clean and intentional.
- When cleaning `submissions/ready/`, move files to a timestamped folder under `submissions/archive/unsent/` instead of deleting.
- Preserve reproducibility: do not overwrite prior artifacts silently.
- Cleanup trigger rule (locked): do **not** auto-clear `submissions/ready/` at end of chat.
- Archive files from `submissions/ready/` only when at least one is true:
  - the user provides a score/result for that submission (`zip_name -> score`)
  - the submission is outdated/superseded by new evidence or branch decisions
- Keep unscored, still-viable candidates in `submissions/ready/` across chats.

## Submission Naming Convention (Locked)
- Use readable, sortable names: `monDD-b<batch>-t<ab>-w<nnn>-s<nnn>-onnx.zip`
- Token rules:
  - `monDD`: day code (example: `feb21`)
  - `b<batch>`: batch number for the day (`b1`, `b2`, ...)
  - `t<ab>`: topK pair shorthand (example: `t718` means `t0_top7`, `t1_top18`)
  - `w<nnn>`: primary weight knob (`w175` = 1.75, `w150` = 1.50)
  - `s<nnn>`: secondary weight knob when used (`s025` = 0.25)
- Keep names concise and human-readable; avoid long family prose in filenames.

## Logging Discipline
- After each meaningful build/decision, append a concise record to `logs/findings.md`:
  - what was built
  - why it was chosen
  - expected signal
  - final scores when available
- Record explicit kill/continue decisions for each tested family.
- Score update interface:
  - Input from user: `zip_name -> score`
  - Output from assistant: `band`, `delta`, `branch_state`, `next_action`
- Use `scripts/submission_decision_engine.py` to enforce family state and rules.
- End-of-chat rule (locked): keep `logs/findings.md` and decision state (`logs/submission_decision_state.json`) up to date before final response.

## Communication Style
- Be direct and concise.
- State the chosen path, then execute.
- Surface blockers immediately with the smallest practical fallback.

## Precedence
1. System, developer, and runtime instructions
2. `AGENTS.md`
3. `CLAUDE.md`
