#!/usr/bin/env python3
"""Create a coordination-visible review claim for another branch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "enforced_planning").is_dir():
            return parent
    raise RuntimeError("Unable to locate repo root containing enforced_planning/")


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enforced_planning import concern_routing  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--agent", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--target-branch", required=True)
    parser.add_argument("--intent", required=True)
    parser.add_argument("--write-path", action="append", default=[])
    parser.add_argument("--plan")
    parser.add_argument("--ttl-hours", type=float, default=24.0)
    parser.add_argument("--notes")
    parser.add_argument("--scope")
    parser.add_argument("--session-id")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = concern_routing.create_review_claim(
        repo_root=args.repo_root,
        agent=args.agent,
        project=args.project,
        target_branch=args.target_branch,
        intent=args.intent,
        write_paths=args.write_path,
        plan_ref=args.plan,
        ttl_hours=args.ttl_hours,
        notes=args.notes,
        scope=args.scope,
        session_id=args.session_id,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"{payload['message']} -> {payload['scope']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
