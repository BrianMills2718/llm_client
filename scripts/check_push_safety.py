#!/usr/bin/env python3
"""Validate whether the current branch is safe to push."""

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

from enforced_planning import push_safety  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--project")
    parser.add_argument("--branch")
    parser.add_argument("--fail-on-active-decisions", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = push_safety.evaluate_push_safety(
        args.repo_root,
        project=args.project,
        branch=args.branch,
        fail_on_active_decisions=args.fail_on_active_decisions,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"push-check: branch={payload['branch']} default={payload['default_branch']} "
            f"issues={len(payload['issues'])} warnings={len(payload['warnings'])}"
        )
        for issue in payload["issues"]:
            print(f"ERROR {issue['code']}: {issue['message']}")
        for warning in payload["warnings"]:
            print(f"WARN  {warning['code']}: {warning['message']}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
