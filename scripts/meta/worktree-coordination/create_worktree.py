#!/usr/bin/env python3
"""Create a git worktree and fail loud if the fresh checkout is unsafe.

This wrapper exists because raw ``git worktree add`` does not provide any
ecosystem-level guarantee that the created checkout is actually usable for
agent work. A freshly created worktree should be clean. If it is not, that is
already enough to block autonomous execution. The wrapper records the immediate
status, classifies stronger split-brain-like symptoms, and optionally cleans up
the failed worktree instead of leaving an ambiguous checkout in circulation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class StatusEntry:
    """One porcelain status entry from a worktree checkout."""

    code: str
    path: str


@dataclass(frozen=True)
class WorktreeStatusSummary:
    """Summarize the immediate status of a freshly created worktree."""

    branch_line: str | None
    entries: list[StatusEntry]
    deleted_count: int
    untracked_count: int
    split_brain_like: bool
    clean: bool


@dataclass(frozen=True)
class WorktreeCreationResult:
    """Structured result for worktree creation and first-status verification."""

    ok: bool
    repo_root: str
    worktree_path: str
    branch: str
    created_branch: bool
    classification: str
    cleanup_performed: bool
    message: str
    status: WorktreeStatusSummary | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for worktree creation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Git repo root")
    parser.add_argument("--path", required=True, help="Path for the new worktree")
    parser.add_argument("--branch", required=True, help="Branch to create or attach")
    parser.add_argument(
        "--start-point",
        default="HEAD",
        help="Commit-ish to branch from when the branch does not already exist",
    )
    parser.add_argument(
        "--split-brain-threshold",
        type=int,
        default=5,
        help="Deleted/untracked count threshold for split-brain-like classification",
    )
    parser.add_argument(
        "--keep-failed-worktree",
        action="store_true",
        help="Leave a failed worktree on disk for manual diagnosis",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON output",
    )
    return parser.parse_args(argv)


def run_git(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run one git command and capture stdout/stderr for diagnosis."""
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )


def branch_exists(repo_root: Path, branch: str) -> bool:
    """Return whether a local branch already exists in the target repo."""
    result = run_git(["show-ref", "--verify", f"refs/heads/{branch}"], cwd=repo_root)
    return result.returncode == 0


def parse_status_porcelain(
    porcelain: str,
    *,
    split_brain_threshold: int,
) -> WorktreeStatusSummary:
    """Parse porcelain status output into a deterministic summary."""
    branch_line: str | None = None
    entries: list[StatusEntry] = []
    deleted_count = 0
    untracked_count = 0

    for raw_line in porcelain.splitlines():
        if not raw_line:
            continue
        if raw_line.startswith("## "):
            branch_line = raw_line[3:]
            continue
        if raw_line.startswith("?? "):
            code = "??"
            path = raw_line[3:]
            untracked_count += 1
        else:
            code = raw_line[:2]
            path = raw_line[3:]
            if "D" in code:
                deleted_count += 1
        entries.append(StatusEntry(code=code, path=path))

    split_brain_like = (
        deleted_count >= split_brain_threshold and untracked_count >= split_brain_threshold
    )
    return WorktreeStatusSummary(
        branch_line=branch_line,
        entries=entries,
        deleted_count=deleted_count,
        untracked_count=untracked_count,
        split_brain_like=split_brain_like,
        clean=len(entries) == 0,
    )


def inspect_worktree_state(
    worktree_path: Path,
    *,
    split_brain_threshold: int,
) -> WorktreeStatusSummary:
    """Inspect immediate worktree status after creation."""
    result = run_git(
        ["status", "--porcelain", "--untracked-files=all", "--branch"],
        cwd=worktree_path,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Unable to inspect fresh worktree status:\n"
            f"{result.stderr or result.stdout}".strip()
        )
    return parse_status_porcelain(
        result.stdout,
        split_brain_threshold=split_brain_threshold,
    )


def classify_summary(summary: WorktreeStatusSummary) -> str:
    """Classify the immediate worktree state for operator-facing messages."""
    if summary.clean:
        return "clean"
    if summary.split_brain_like:
        return "split-brain-like"
    return "dirty"


def cleanup_failed_worktree(
    repo_root: Path,
    worktree_path: Path,
    *,
    branch: str,
    created_branch: bool,
) -> tuple[bool, str]:
    """Attempt to remove a failed worktree and delete its just-created branch."""
    remove_result = run_git(["worktree", "remove", "--force", str(worktree_path)], cwd=repo_root)
    if remove_result.returncode != 0 and worktree_path.exists():
        return False, (remove_result.stderr or remove_result.stdout).strip()
    if created_branch:
        delete_result = run_git(["branch", "-D", branch], cwd=repo_root)
        if delete_result.returncode != 0:
            return False, (delete_result.stderr or delete_result.stdout).strip()
    return True, "cleanup complete"


def ensure_safe_target_path(worktree_path: Path) -> None:
    """Reject non-empty target paths before invoking git worktree add."""
    if not worktree_path.exists():
        return
    if worktree_path.is_dir() and not any(worktree_path.iterdir()):
        return
    raise ValueError(
        f"Target worktree path already exists and is not empty: {worktree_path}"
    )


def create_worktree(
    *,
    repo_root: Path,
    worktree_path: Path,
    branch: str,
    start_point: str,
    split_brain_threshold: int,
    keep_failed_worktree: bool,
) -> WorktreeCreationResult:
    """Create a worktree, inspect it immediately, and fail loud on unsafe state."""
    ensure_safe_target_path(worktree_path)
    worktree_path.parent.mkdir(parents=True, exist_ok=True)

    repo_root = repo_root.resolve()
    worktree_path = worktree_path.resolve()
    branch_already_exists = branch_exists(repo_root, branch)
    if branch_already_exists and start_point != "HEAD":
        raise ValueError(
            "Cannot combine --start-point with an existing branch; "
            f"branch already exists: {branch}"
        )

    add_args = ["worktree", "add", str(worktree_path)]
    if branch_already_exists:
        add_args.append(branch)
        created_branch = False
    else:
        add_args.extend(["-b", branch, start_point])
        created_branch = True

    add_result = run_git(add_args, cwd=repo_root)
    if add_result.returncode != 0:
        return WorktreeCreationResult(
            ok=False,
            repo_root=str(repo_root),
            worktree_path=str(worktree_path),
            branch=branch,
            created_branch=created_branch,
            classification="git-error",
            cleanup_performed=False,
            message=(add_result.stderr or add_result.stdout).strip(),
            status=None,
        )

    summary = inspect_worktree_state(
        worktree_path,
        split_brain_threshold=split_brain_threshold,
    )
    classification = classify_summary(summary)
    if classification == "clean":
        return WorktreeCreationResult(
            ok=True,
            repo_root=str(repo_root),
            worktree_path=str(worktree_path),
            branch=branch,
            created_branch=created_branch,
            classification=classification,
            cleanup_performed=False,
            message="Worktree created cleanly.",
            status=summary,
        )

    cleanup_performed = False
    cleanup_note = ""
    if not keep_failed_worktree:
        cleanup_ok, cleanup_note = cleanup_failed_worktree(
            repo_root,
            worktree_path,
            branch=branch,
            created_branch=created_branch,
        )
        cleanup_performed = cleanup_ok
        if not cleanup_ok and worktree_path.exists():
            cleanup_note = f" Cleanup failed: {cleanup_note}"

    sample_entries = ", ".join(
        f"{entry.code} {entry.path}" for entry in summary.entries[:8]
    )
    return WorktreeCreationResult(
        ok=False,
        repo_root=str(repo_root),
        worktree_path=str(worktree_path),
        branch=branch,
        created_branch=created_branch,
        classification=classification,
        cleanup_performed=cleanup_performed,
        message=(
            "Fresh worktree was not clean immediately after creation. "
            f"classification={classification}; "
            f"deleted={summary.deleted_count}; "
            f"untracked={summary.untracked_count}; "
            f"sample=[{sample_entries}].{cleanup_note}"
        ),
        status=summary,
    )


def _print_human(result: WorktreeCreationResult) -> None:
    """Print a concise operator summary."""
    state = "OK" if result.ok else "FAIL"
    print(f"{state}: {result.message}")
    print(f"repo: {result.repo_root}")
    print(f"path: {result.worktree_path}")
    print(f"branch: {result.branch}")
    if result.status is not None:
        print(f"classification: {result.classification}")
        print(
            "status-counts: "
            f"deleted={result.status.deleted_count} "
            f"untracked={result.status.untracked_count} "
            f"entries={len(result.status.entries)}"
        )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for safe worktree creation."""
    args = parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve()
    worktree_path = Path(args.path).expanduser().resolve()
    try:
        result = create_worktree(
            repo_root=repo_root,
            worktree_path=worktree_path,
            branch=args.branch,
            start_point=args.start_point,
            split_brain_threshold=args.split_brain_threshold,
            keep_failed_worktree=args.keep_failed_worktree,
        )
    except (RuntimeError, ValueError) as exc:
        error_result = WorktreeCreationResult(
            ok=False,
            repo_root=str(repo_root),
            worktree_path=str(worktree_path),
            branch=args.branch,
            created_branch=False,
            classification="error",
            cleanup_performed=False,
            message=str(exc),
            status=None,
        )
        if args.json:
            print(json.dumps(asdict(error_result), indent=2))
        else:
            _print_human(error_result)
        return 1

    if args.json:
        print(json.dumps(asdict(result), indent=2))
    else:
        _print_human(result)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
