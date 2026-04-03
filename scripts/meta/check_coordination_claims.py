#!/usr/bin/env python3
"""Authoritative coordination registry CLI for multi-agent work.

This CLI is the sanctioned human/agent surface for local active-work claims and
plan reservations. It keeps all runtime coordination state in one local-only
registry under ``~/.claude/coordination/`` rather than split YAML files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plan_reservations import (
    DEFAULT_TTL_HOURS,
    claim_active_work,
    coordination_dir,
    ingest_legacy_state,
    list_registry,
    mark_plan_reservation_historical_unlanded,
    observe_file,
    reconcile_plan_reservations,
    release_active_work,
    release_plan_reservation,
    reserve_next_plan_number,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for coordination operations."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Check active work for a project.")
    group.add_argument("--claim", action="store_true", help="Create or refresh an active-work claim.")
    group.add_argument("--release", action="store_true", help="Release an active-work claim.")
    group.add_argument("--observe-file", metavar="PATH", help="Record one observed file for an active-work claim.")
    group.add_argument("--reserve-plan", action="store_true", help="Reserve the next plan number for a repo.")
    group.add_argument("--release-plan", metavar="PLAN", type=int, help="Release an unconsumed plan reservation.")
    group.add_argument(
        "--mark-historical-unlanded",
        metavar="PLAN",
        type=int,
        help="Mark one consumed reservation as historical-unlanded lineage.",
    )
    group.add_argument(
        "--reconcile-plan-lineage",
        action="store_true",
        help="Reconcile consumed reservation lineage for one repo.",
    )
    group.add_argument("--list", action="store_true", help="List active work and plan reservations.")
    group.add_argument("--ingest-legacy", action="store_true", help="One-way ingest from legacy coordination files.")

    parser.add_argument("--agent", help="Agent brain name (claude-code, codex, openclaw).")
    parser.add_argument("--project", help="Project name. Defaults from --repo-root when provided.")
    parser.add_argument("--scope", help="Scope path or identifier.")
    parser.add_argument("--intent", help="What the agent intends to do.")
    parser.add_argument("--plan", help="Plan reference or number.")
    parser.add_argument("--ttl-hours", type=float, default=DEFAULT_TTL_HOURS, help="Claim TTL in hours.")
    parser.add_argument("--repo-root", help="Repo root used to infer project and plan namespace.")
    parser.add_argument("--branch", help="Branch name for active work or reservation metadata.")
    parser.add_argument("--worktree-path", help="Worktree path for active work or reservation metadata.")
    parser.add_argument("--session-id", help="Optional session identifier.")
    parser.add_argument("--no-fetch", action="store_true", help="Do not fetch origin before reserving a plan number.")
    parser.add_argument("--historical-plan-file", help="Historical worktree-local plan path for cleanup resolution.")
    parser.add_argument("--reason", help="Optional lineage resolution reason text.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args()


def _print_payload(payload: object, *, as_json: bool) -> None:
    """Print either JSON or a readable representation."""
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if isinstance(payload, dict):
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(payload)


def main() -> int:
    """Run the authoritative coordination registry CLI."""
    args = parse_args()

    if args.ingest_legacy:
        payload = ingest_legacy_state(repo_root=args.repo_root)
        _print_payload(payload, as_json=args.json)
        return 0

    if args.check or args.list:
        payload = list_registry(project=args.project)
        if not args.json:
            print(f"Registry: {coordination_dir()}")
            for entry in payload["active_work"]:
                print(
                    f"[active] {entry.get('agent')} {entry.get('project')}:{entry.get('scope')} "
                    f"plan={entry.get('plan')} task={entry.get('task')}"
                )
            for entry in payload["plan_reservations"]:
                lineage_state = entry.get("lineage_state")
                lineage_suffix = f" lineage={lineage_state}" if lineage_state else ""
                print(
                    f"[reserved] {entry.get('project')} Plan #{entry.get('plan')} "
                    f"by {entry.get('agent')} task={entry.get('task')} status={entry.get('status')}{lineage_suffix}"
                )
            if not payload["active_work"] and not payload["plan_reservations"]:
                print("No coordination entries.")
        else:
            _print_payload(payload, as_json=True)
        return 0

    if args.claim:
        if not all([args.agent, args.scope, args.intent]):
            raise SystemExit("--claim requires --agent, --scope, and --intent")
        ok, msg, entry = claim_active_work(
            agent=args.agent,
            project=args.project,
            scope=args.scope,
            intent=args.intent,
            repo_root=args.repo_root,
            branch=args.branch,
            worktree_path=args.worktree_path,
            session_id=args.session_id,
            plan=args.plan,
            ttl_hours=args.ttl_hours,
        )
        _print_payload(entry if args.json else msg, as_json=args.json)
        return 0 if ok else 1

    if args.release:
        if not all([args.agent, args.scope]):
            raise SystemExit("--release requires --agent and --scope")
        ok, msg = release_active_work(
            args.agent,
            args.project,
            args.scope,
            repo_root=args.repo_root,
        )
        _print_payload({"ok": ok, "message": msg} if args.json else msg, as_json=args.json)
        return 0 if ok else 1

    if args.observe_file:
        if not all([args.agent, args.scope]):
            raise SystemExit("--observe-file requires --agent and --scope")
        ok, msg = observe_file(
            agent=args.agent,
            project=args.project,
            scope=args.scope,
            file_path=args.observe_file,
            repo_root=args.repo_root,
        )
        _print_payload({"ok": ok, "message": msg} if args.json else msg, as_json=args.json)
        return 0 if ok else 1

    if args.reserve_plan:
        if not all([args.agent, args.intent, args.repo_root]):
            raise SystemExit("--reserve-plan requires --agent, --intent, and --repo-root")
        entry = reserve_next_plan_number(
            repo_root=args.repo_root,
            agent=args.agent,
            task=args.intent,
            branch=args.branch,
            worktree_path=args.worktree_path,
            session_id=args.session_id,
            fetch=not args.no_fetch,
        )
        _print_payload(entry, as_json=args.json)
        return 0

    if args.release_plan is not None:
        if not args.repo_root:
            raise SystemExit("--release-plan requires --repo-root")
        ok, msg = release_plan_reservation(repo_root=args.repo_root, plan=args.release_plan)
        _print_payload({"ok": ok, "message": msg} if args.json else msg, as_json=args.json)
        return 0 if ok else 1

    if args.mark_historical_unlanded is not None:
        if not args.repo_root:
            raise SystemExit("--mark-historical-unlanded requires --repo-root")
        ok, msg = mark_plan_reservation_historical_unlanded(
            repo_root=args.repo_root,
            plan=args.mark_historical_unlanded,
            historical_plan_file=args.historical_plan_file,
            reason=args.reason,
        )
        _print_payload({"ok": ok, "message": msg} if args.json else msg, as_json=args.json)
        return 0 if ok else 1

    if args.reconcile_plan_lineage:
        if not args.repo_root:
            raise SystemExit("--reconcile-plan-lineage requires --repo-root")
        payload = reconcile_plan_reservations(repo_root=args.repo_root)
        _print_payload(payload, as_json=args.json)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
