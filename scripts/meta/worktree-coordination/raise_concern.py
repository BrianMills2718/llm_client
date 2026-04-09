#!/usr/bin/env python3
"""Route a concern to PR review when published, otherwise to the local inbox."""

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
    parser.add_argument("--subject", required=True)
    parser.add_argument("--content")
    parser.add_argument("--content-file")
    parser.add_argument("--recipient")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _load_content(args: argparse.Namespace) -> str:
    if args.content_file:
        return Path(args.content_file).read_text(encoding="utf-8")
    if args.content:
        return args.content
    raise ValueError("Either --content or --content-file is required.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = concern_routing.route_concern(
        repo_root=args.repo_root,
        agent=args.agent,
        project=args.project,
        target_branch=args.target_branch,
        subject=args.subject,
        content=_load_content(args),
        recipient=args.recipient,
        dry_run=args.dry_run,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"{payload['route']}: {payload['destination']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
