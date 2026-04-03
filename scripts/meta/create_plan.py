#!/usr/bin/env python3
"""Create a new implementation plan through the sanctioned reservation path."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan_reservations import (
    consume_plan_reservation,
    preview_next_plan_number,
    release_plan_reservation,
    reserve_next_plan_number,
)


DEFAULT_ROOT = Path.cwd()


def resolve_plan_root(repo_root: str) -> Path:
    """Return the target repo root for plan creation."""
    return Path(repo_root).expanduser().resolve()


def resolve_plan_paths(repo_root: Path) -> tuple[Path, Path, Path]:
    """Return the plan directory, template path, and index path for a repo."""
    plans_dir = repo_root / "docs" / "plans"
    return plans_dir, plans_dir / "TEMPLATE.md", plans_dir / "CLAUDE.md"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for sanctioned plan creation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", required=True, help="Human-readable plan title.")
    parser.add_argument("--priority", default="High", help="Priority value for the plan index.")
    parser.add_argument("--status", default="📋 Planned", help="Status text for the plan index.")
    parser.add_argument("--blocked-by", default="None", help="Blocked By value for the generated plan.")
    parser.add_argument("--blocks", default="None", help="Blocks value for the generated plan.")
    parser.add_argument("--agent", default="codex", help="Agent creating the plan reservation.")
    parser.add_argument("--task", default=None, help="Reservation task text. Defaults to the title.")
    parser.add_argument("--dry-run", action="store_true", help="Preview the reserved number without writing files.")
    parser.add_argument("--no-fetch", action="store_true", help="Do not fetch origin before allocating the next plan number.")
    parser.add_argument(
        "--repo-root",
        default=str(DEFAULT_ROOT),
        help="Target repo root for plan creation. Defaults to the current working directory.",
    )
    return parser.parse_args()


def slugify(title: str) -> str:
    """Return a stable markdown filename slug for the title."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "new-plan"


def render_plan(
    plan_number: int,
    title: str,
    blocked_by: str,
    blocks: str,
    *,
    template_path: Path,
) -> str:
    """Render the new plan file from the canonical template."""
    template = template_path.read_text(encoding="utf-8")
    return (
        template.replace("# Plan #N: [Name]", f"# Plan #{plan_number}: {title}")
        .replace("**Status:** Planned", "**Status:** Planned")
        .replace("**Priority:** High | Medium | Low", "**Priority:** High")
        .replace("**Blocked By:** None", f"**Blocked By:** {blocked_by}")
        .replace("**Blocks:** None", f"**Blocks:** {blocks}")
    )


def _new_index_row(plan_number: int, title: str, priority: str, status: str, blocked_by: str) -> str:
    """Render one markdown table row for the plan index."""
    filename = f"{plan_number:02d}_{slugify(title)}.md"
    blocks = blocked_by if blocked_by != "None" else "-"
    return f"| {plan_number} | [{title}]({filename}) | {priority} | {status} | {blocks} |"


def update_index(index_text: str, row: str, plan_number: int) -> str:
    """Insert one new row into the plan index before the status-key section."""
    if f"| {plan_number} |" in index_text:
        raise ValueError(f"Plan #{plan_number} already exists in docs/plans/CLAUDE.md")

    lines = index_text.splitlines()
    insert_at = None
    for idx, line in enumerate(lines):
        if line.startswith("## Status Key"):
            insert_at = idx
            break
    if insert_at is None:
        raise ValueError("docs/plans/CLAUDE.md is missing the '## Status Key' section")

    rows = []
    for idx, line in enumerate(lines):
        if re.match(r"^\| \d+ \|", line):
            rows.append((idx, int(line.split("|")[1].strip())))
    last_row_index = max((idx for idx, _ in rows), default=insert_at - 1)
    insert_at = last_row_index + 1
    lines.insert(insert_at, row)
    return "\n".join(lines) + "\n"


def create_plan(args: argparse.Namespace) -> dict[str, str | int | bool]:
    """Create a plan file and index row under one reserved number."""
    repo_root = resolve_plan_root(args.repo_root)
    plans_dir, template_path, index_path = resolve_plan_paths(repo_root)
    title = args.title.strip()
    task = (args.task or title).strip()
    if args.dry_run:
        next_plan = preview_next_plan_number(repo_root=repo_root, fetch=not args.no_fetch)
        return {
            "dry_run": True,
            "plan": next_plan,
            "filename": f"{next_plan:02d}_{slugify(title)}.md",
            "title": title,
        }

    reservation = reserve_next_plan_number(
        repo_root=repo_root,
        agent=args.agent,
        task=task,
        fetch=not args.no_fetch,
    )
    plan_number = int(reservation["plan"])
    filename = f"{plan_number:02d}_{slugify(title)}.md"
    plan_path = plans_dir / filename
    if plan_path.exists():
        release_plan_reservation(repo_root=repo_root, plan=plan_number)
        raise ValueError(f"Plan file already exists: {plan_path}")

    original_index = index_path.read_text(encoding="utf-8")
    try:
        plan_path.write_text(
            render_plan(
                plan_number,
                title,
                args.blocked_by,
                args.blocks,
                template_path=template_path,
            ),
            encoding="utf-8",
        )
        index_path.write_text(
            update_index(
                original_index,
                _new_index_row(plan_number, title, args.priority, args.status, args.blocked_by),
                plan_number,
            ),
            encoding="utf-8",
        )
        ok, message = consume_plan_reservation(
            repo_root=repo_root,
            plan=plan_number,
            plan_file=plan_path,
        )
        if not ok:
            raise ValueError(message)
    except Exception:
        if plan_path.exists():
            plan_path.unlink()
        index_path.write_text(original_index, encoding="utf-8")
        release_plan_reservation(repo_root=repo_root, plan=plan_number)
        raise

    return {
        "dry_run": False,
        "filename": filename,
        "plan": plan_number,
        "title": title,
    }


def main() -> int:
    """Run the sanctioned plan-creation CLI."""
    args = parse_args()
    result = create_plan(args)
    if result["dry_run"]:
        print(
            f"Plan #{result['plan']} would be created as {result['filename']} "
            f"for '{result['title']}'"
        )
        return 0
    print(f"Created Plan #{result['plan']}: {result['filename']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
