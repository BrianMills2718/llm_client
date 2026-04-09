"""Pre-push coordination checks for governed worktree branches.

The push gate exists to keep publish actions aligned with the coordination
layer. It blocks obvious unsafe cases mechanically and surfaces softer
coordination context for human review before shared state changes.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from enforced_planning import coordination_claims


@dataclass(frozen=True)
class PushCheckFinding:
    """One push-safety issue or warning with enough detail to act on it."""

    code: str
    message: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation for CLI and tests."""

        return asdict(self)


def _run_git(repo_root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one git command inside the repo and capture the result."""

    return subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )


def _git_stdout(repo_root: Path, args: list[str]) -> str:
    """Return stdout for one successful git command or fail loudly."""

    result = _run_git(repo_root, args)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def resolve_repo_root(start_path: str | Path = ".") -> Path:
    """Resolve the canonical git toplevel for the current repo."""

    start = Path(start_path).resolve()
    return Path(_git_stdout(start, ["rev-parse", "--show-toplevel"]))


def resolve_default_branch(repo_root: Path) -> str | None:
    """Resolve the repo's default branch from local refs and origin/HEAD."""

    remote_head = _run_git(repo_root, ["symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD"])
    if remote_head.returncode == 0:
        value = remote_head.stdout.strip()
        if value.startswith("origin/"):
            return value.split("/", 1)[1]
        if value:
            return value
    for candidate in ("main", "master"):
        exists = _run_git(repo_root, ["show-ref", "--verify", f"refs/heads/{candidate}"])
        if exists.returncode == 0:
            return candidate
    return None


def current_branch(repo_root: Path) -> str:
    """Return the current symbolic branch name or fail on detached HEAD."""

    branch = _run_git(repo_root, ["symbolic-ref", "--quiet", "--short", "HEAD"])
    if branch.returncode != 0:
        raise RuntimeError("Detached HEAD is not eligible for push-check.")
    value = branch.stdout.strip()
    if not value:
        raise RuntimeError("Unable to resolve current branch name.")
    return value


def current_upstream(repo_root: Path) -> str | None:
    """Return the configured upstream for HEAD when one exists."""

    upstream = _run_git(repo_root, ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"])
    if upstream.returncode != 0:
        return None
    value = upstream.stdout.strip()
    return value or None


def ahead_behind(repo_root: Path, upstream_ref: str) -> tuple[int, int]:
    """Return ahead/behind counts for HEAD relative to its upstream."""

    counts = _git_stdout(repo_root, ["rev-list", "--left-right", "--count", f"{upstream_ref}...HEAD"])
    behind_text, ahead_text = counts.split()
    return int(ahead_text), int(behind_text)


def changed_paths_since_default(repo_root: Path, default_branch: str) -> list[str]:
    """Return repo-relative paths changed on this branch against default."""

    base = _git_stdout(repo_root, ["merge-base", "HEAD", f"refs/heads/{default_branch}"])
    diff = _git_stdout(repo_root, ["diff", "--name-only", f"{base}..HEAD"])
    if not diff:
        return []
    return [coordination_claims._normalize_repo_path(line) for line in diff.splitlines() if line.strip()]


def _working_tree_dirty(repo_root: Path) -> bool:
    """Return whether the repo has tracked or untracked local dirt."""

    return bool(_git_stdout(repo_root, ["status", "--short"]))


def _branch_claims(project: str, branch: str) -> list[coordination_claims.ClaimRecord]:
    """Return live claims attached to one project branch."""

    return [
        claim
        for claim in coordination_claims.check_claims(project)
        if claim.branch == branch
    ]


def _extract_json_block(raw_text: str) -> str:
    """Strip CLI noise before the first JSON token so parsing stays deterministic."""

    for line in raw_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") or stripped.startswith("{"):
            return stripped
    return raw_text


def load_active_decisions(project: str, *, limit: int = 5) -> list[dict[str, Any]]:
    """Read active architectural decisions from agent_memory as raw JSON."""

    result = subprocess.run(
        [
            "agent-memory",
            "recall",
            "active decisions",
            "--project",
            project,
            "--raw",
            "--limit",
            str(limit),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip() or "agent-memory recall failed")
    payload = _extract_json_block(result.stdout.strip())
    if not payload:
        return []
    decoded = json.loads(payload)
    if not isinstance(decoded, list):
        raise RuntimeError("agent-memory recall --raw must return a JSON array")
    return [item for item in decoded if isinstance(item, dict)]


def _claim_overlap_for_paths(
    changed_paths: list[str],
    claim: coordination_claims.ClaimRecord,
) -> list[str]:
    """Return normalized changed-path overlaps against one active claim."""

    overlaps: list[str] = []
    for changed_path in changed_paths:
        for write_path in claim.write_paths:
            if coordination_claims._paths_overlap(changed_path, write_path):
                overlaps.append(f"{changed_path} <-> {write_path}")
    return sorted(set(overlaps))


def evaluate_push_safety(
    repo_root: str | Path = ".",
    *,
    project: str | None = None,
    branch: str | None = None,
    fail_on_active_decisions: bool = False,
) -> dict[str, Any]:
    """Evaluate whether the current branch is safe to push as-is."""

    resolved_repo_root = resolve_repo_root(repo_root)
    resolved_project = project or resolved_repo_root.name
    resolved_branch = branch or current_branch(resolved_repo_root)
    default_branch = resolve_default_branch(resolved_repo_root)
    if not default_branch:
        raise RuntimeError("Unable to resolve the default branch for push-check.")

    issues: list[PushCheckFinding] = []
    warnings: list[PushCheckFinding] = []

    if _working_tree_dirty(resolved_repo_root):
        issues.append(
            PushCheckFinding(
                code="dirty_worktree",
                message="Push-check requires a clean working tree so the published delta is unambiguous.",
                details={},
            )
        )

    if resolved_branch == default_branch:
        issues.append(
            PushCheckFinding(
                code="default_branch_push",
                message="Direct pushes from the default branch are blocked; use a worktree-backed task branch.",
                details={"branch": resolved_branch, "default_branch": default_branch},
            )
        )

    branch_claims = _branch_claims(resolved_project, resolved_branch)
    if not branch_claims:
        issues.append(
            PushCheckFinding(
                code="missing_branch_claim",
                message="No live coordination claim is attached to the current branch.",
                details={"branch": resolved_branch, "project": resolved_project},
            )
        )

    upstream = current_upstream(resolved_repo_root)
    ahead = 0
    behind = 0
    if upstream is None:
        warnings.append(
            PushCheckFinding(
                code="missing_upstream",
                message="Current branch has no upstream; first push will establish the remote tracking branch.",
                details={"branch": resolved_branch},
            )
        )
    else:
        ahead, behind = ahead_behind(resolved_repo_root, upstream)
        if behind > 0:
            issues.append(
                PushCheckFinding(
                    code="behind_upstream",
                    message="Local branch is behind its upstream; reconcile before pushing new commits.",
                    details={"branch": resolved_branch, "upstream": upstream, "behind": behind},
                )
            )

    changed_paths = changed_paths_since_default(resolved_repo_root, default_branch)
    if not changed_paths:
        warnings.append(
            PushCheckFinding(
                code="no_branch_delta",
                message="Current branch has no file delta relative to the default branch.",
                details={"branch": resolved_branch, "default_branch": default_branch},
            )
        )

    for claim in coordination_claims.check_claims(resolved_project):
        if claim.branch == resolved_branch:
            continue
        overlaps = _claim_overlap_for_paths(changed_paths, claim)
        if not overlaps:
            continue
        claim_details = {
            "other_agent": claim.agent,
            "other_scope": claim.scope,
            "other_branch": claim.branch,
            "other_claim_type": claim.claim_type,
            "overlaps": overlaps,
            "source_file": claim.source_file,
        }
        runtime_status = coordination_claims.claim_runtime_status(claim)
        if runtime_status == "stale":
            warnings.append(
                PushCheckFinding(
                    code="stale_overlapping_claim",
                    message="A stale overlapping claim still exists; prune or resolve it before treating the registry as clean.",
                    details=claim_details,
                )
            )
            continue
        if claim.claim_type == "write":
            issues.append(
                PushCheckFinding(
                    code="overlapping_write_claim",
                    message="Changed files overlap another live write claim.",
                    details=claim_details,
                )
            )
            continue
        if claim.claim_type == "review":
            warnings.append(
                PushCheckFinding(
                    code="overlapping_review_claim",
                    message="Changed files overlap an active review claim.",
                    details=claim_details,
                )
            )

    active_decisions = load_active_decisions(resolved_project)
    if active_decisions:
        decision_finding = PushCheckFinding(
            code="active_decisions_present",
            message="Active architectural decisions exist for this repo; review them before publishing.",
            details={"decision_count": len(active_decisions), "records": active_decisions},
        )
        if fail_on_active_decisions:
            issues.append(decision_finding)
        else:
            warnings.append(decision_finding)

    return {
        "ok": not issues,
        "repo_root": str(resolved_repo_root),
        "project": resolved_project,
        "branch": resolved_branch,
        "default_branch": default_branch,
        "upstream": upstream,
        "ahead": ahead,
        "behind": behind,
        "changed_paths": changed_paths,
        "branch_claim_count": len(branch_claims),
        "issues": [item.to_dict() for item in issues],
        "warnings": [item.to_dict() for item in warnings],
    }
