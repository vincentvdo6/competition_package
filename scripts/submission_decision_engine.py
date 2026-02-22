"""Submission decision engine for daily leaderboard workflow.

Implements the "max-leverage" rules:
- 3 then adapt
- strict 2-strike clear-fail kill rule
- one optional mechanistic final probe

Examples
--------
Record a result:
    python scripts/submission_decision_engine.py record \
      --family recovery_ptarget_topk \
      --zip recovery_ptarget_t07_t16_onnx.zip \
      --score 0.2879 \
      --pb 0.2885 \
      --final-probe-zip recovery_ptarget_t07_t18_onnx.zip

Show status:
    python scripts/submission_decision_engine.py status
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_STATE_PATH = Path("logs/submission_decision_state.json")


@dataclass(frozen=True)
class Policy:
    pb: float = 0.2885
    strong_win_margin: float = 0.0003
    clear_fail_margin: float = 0.0004


def classify_band(delta: float, policy: Policy) -> str:
    """Classify a score delta into one of the locked bands."""
    if delta >= policy.strong_win_margin:
        return "strong_win"
    if delta >= 0.0:
        return "soft_win"
    if delta > -policy.clear_fail_margin:
        return "near_miss"
    return "clear_fail"


def load_state(path: Path, policy: Policy) -> dict[str, Any]:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        # Keep persisted policy for reproducibility, but update defaults if missing.
        data.setdefault("policy", {})
        data["policy"].setdefault("pb", policy.pb)
        data["policy"].setdefault("strong_win_margin", policy.strong_win_margin)
        data["policy"].setdefault("clear_fail_margin", policy.clear_fail_margin)
        data.setdefault("families", {})
        return data

    return {
        "policy": {
            "pb": policy.pb,
            "strong_win_margin": policy.strong_win_margin,
            "clear_fail_margin": policy.clear_fail_margin,
        },
        "families": {},
    }


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def ensure_family(state: dict[str, Any], family: str) -> dict[str, Any]:
    families = state.setdefault("families", {})
    fam = families.setdefault(
        family,
        {
            "status": "active",
            "clear_fail_count": 0,
            "final_probe_zip": None,
            "final_probe_status": "none",  # none | reserved | used
            "history": [],
            "last_action": "NONE",
            "updated_at": None,
        },
    )
    return fam


def format_compact(
    zip_name: str,
    score: float,
    delta: float,
    band: str,
    family: str,
    status: str,
    action: str,
) -> str:
    return (
        f"{zip_name} -> {score:.4f} | band={band} | delta={delta:+.4f} | "
        f"family={family} status={status} | next={action}"
    )


def cmd_record(args: argparse.Namespace) -> int:
    policy = Policy(
        pb=args.pb,
        strong_win_margin=args.strong_win_margin,
        clear_fail_margin=args.clear_fail_margin,
    )
    state_path = Path(args.state)
    state = load_state(state_path, policy)
    fam = ensure_family(state, args.family)

    if args.final_probe_zip:
        fam["final_probe_zip"] = args.final_probe_zip

    delta = args.score - policy.pb
    band = classify_band(delta, policy)
    ts = datetime.now().isoformat(timespec="seconds")

    is_final_probe_result = (
        fam.get("final_probe_zip") is not None and args.zip == fam["final_probe_zip"]
    )

    if band == "clear_fail":
        fam["clear_fail_count"] = int(fam.get("clear_fail_count", 0)) + 1

    action = "NONE"

    if is_final_probe_result:
        fam["final_probe_status"] = "used"
        if args.score >= policy.pb:
            fam["status"] = "active"
            fam["clear_fail_count"] = 0
            action = "REOPEN_BRANCH_TIGHT_SWEEP"
        else:
            fam["status"] = "killed"
            action = "KILL_FAMILY"
    elif band == "strong_win":
        fam["status"] = "active"
        fam["clear_fail_count"] = 0
        action = "EXPLOIT_TIGHT_NEIGHBORS"
    elif band == "soft_win":
        fam["status"] = "active"
        action = "CONTINUE_SMALL_EXPLOIT"
    elif band == "near_miss":
        fam["status"] = "active"
        action = "ONE_DIRECTIONAL_PROBE_THEN_DECIDE"
    else:
        clear_fails = int(fam.get("clear_fail_count", 0))
        if clear_fails >= 2:
            if fam.get("final_probe_zip") and fam.get("final_probe_status") == "none":
                fam["status"] = "final_probe_only"
                fam["final_probe_status"] = "reserved"
                action = f"RUN_FINAL_PROBE_ONLY:{fam['final_probe_zip']}"
            elif fam.get("final_probe_zip") and fam.get("final_probe_status") == "reserved":
                fam["status"] = "final_probe_only"
                action = f"RUN_RESERVED_FINAL_PROBE:{fam['final_probe_zip']}"
            else:
                fam["status"] = "killed"
                action = "KILL_FAMILY"
        else:
            fam["status"] = "active"
            action = "DO_NOT_WIDEN_SWEEP"

    record = {
        "timestamp": ts,
        "zip": args.zip,
        "score": args.score,
        "delta": delta,
        "band": band,
        "notes": args.notes or "",
    }
    fam["history"].append(record)
    fam["last_action"] = action
    fam["updated_at"] = ts

    save_state(state_path, state)

    print("RESULT")
    print(f"  zip:    {args.zip}")
    print(f"  score:  {args.score:.4f}")
    print(f"  pb:     {policy.pb:.4f}")
    print(f"  delta:  {delta:+.4f}")
    print(f"  band:   {band}")
    print("")
    print("FAMILY")
    print(f"  name:             {args.family}")
    print(f"  status:           {fam['status']}")
    print(f"  clear_fail_count: {fam['clear_fail_count']}")
    print(f"  final_probe_zip:  {fam.get('final_probe_zip')}")
    print(f"  final_probe_stat: {fam.get('final_probe_status')}")
    print(f"  next_action:      {action}")
    print("")
    print("COMPACT")
    print(
        format_compact(
            args.zip,
            args.score,
            delta,
            band,
            args.family,
            fam["status"],
            action,
        )
    )
    print("")
    print(f"State saved: {state_path}")

    return 0


def cmd_status(args: argparse.Namespace) -> int:
    policy = Policy(
        pb=args.pb,
        strong_win_margin=args.strong_win_margin,
        clear_fail_margin=args.clear_fail_margin,
    )
    state_path = Path(args.state)
    state = load_state(state_path, policy)

    print("POLICY")
    print(json.dumps(state.get("policy", {}), indent=2))
    print("")

    families = state.get("families", {})
    if not families:
        print("No family state recorded yet.")
        return 0

    if args.family:
        fam = families.get(args.family)
        if fam is None:
            print(f"Family not found: {args.family}")
            return 1
        print(f"FAMILY: {args.family}")
        print(json.dumps(fam, indent=2))
        return 0

    print("FAMILIES")
    for name, fam in families.items():
        hist_len = len(fam.get("history", []))
        print(
            f"- {name}: status={fam.get('status')} "
            f"clear_fails={fam.get('clear_fail_count', 0)} "
            f"final_probe={fam.get('final_probe_status')} "
            f"entries={hist_len} "
            f"last_action={fam.get('last_action')}"
        )
    return 0


def cmd_reset_family(args: argparse.Namespace) -> int:
    policy = Policy(
        pb=args.pb,
        strong_win_margin=args.strong_win_margin,
        clear_fail_margin=args.clear_fail_margin,
    )
    state_path = Path(args.state)
    state = load_state(state_path, policy)

    families = state.setdefault("families", {})
    if args.family not in families:
        print(f"Family not found: {args.family}")
        return 1

    del families[args.family]
    save_state(state_path, state)
    print(f"Removed family state: {args.family}")
    print(f"State saved: {state_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    def add_policy_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--state",
            default=str(DEFAULT_STATE_PATH),
            help="Path to JSON state file",
        )
        p.add_argument("--pb", type=float, default=0.2885, help="Current PB score")
        p.add_argument(
            "--strong-win-margin",
            type=float,
            default=0.0003,
            help="Threshold for strong win: delta >= margin",
        )
        p.add_argument(
            "--clear-fail-margin",
            type=float,
            default=0.0004,
            help="Threshold for clear fail: delta <= -margin",
        )

    parser = argparse.ArgumentParser(
        description="Leaderboard submission decision engine"
    )
    add_policy_args(parser)

    sub = parser.add_subparsers(dest="command", required=True)

    p_record = sub.add_parser("record", help="Record one leaderboard result")
    add_policy_args(p_record)
    p_record.add_argument("--family", required=True, help="Family/branch identifier")
    p_record.add_argument("--zip", required=True, help="Submission zip filename")
    p_record.add_argument("--score", required=True, type=float, help="LB score")
    p_record.add_argument(
        "--final-probe-zip",
        default=None,
        help="Optional one-time mechanistic final probe zip",
    )
    p_record.add_argument("--notes", default="", help="Optional notes")
    p_record.set_defaults(func=cmd_record)

    p_status = sub.add_parser("status", help="Show current policy/family state")
    add_policy_args(p_status)
    p_status.add_argument("--family", default=None, help="Optional family filter")
    p_status.set_defaults(func=cmd_status)

    p_reset = sub.add_parser("reset-family", help="Remove one family from state")
    add_policy_args(p_reset)
    p_reset.add_argument("--family", required=True, help="Family/branch identifier")
    p_reset.set_defaults(func=cmd_reset_family)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
