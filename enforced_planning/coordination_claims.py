#!/usr/bin/env python3
"""Cross-brain coordination claims for multi-agent work.

Manages scope claims across agent brains (claude-code, codex, openclaw).
Claims are YAML files in ``~/.claude/coordination/claims/``.

The v2 model preserves backwards-compatible v1 loading while adding the narrow
write-scope metadata needed for real collision avoidance.

Usage:
    check_coordination_claims.py --check [--project PROJECT]
    check_coordination_claims.py --claim --agent AGENT --project PROJECT --scope SCOPE --intent INTENT [--plan PLAN] [--ttl-hours TTL]
    check_coordination_claims.py --release --agent AGENT --project PROJECT --scope SCOPE
    check_coordination_claims.py --list
    check_coordination_claims.py --prune
"""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

CLAIMS_DIR = Path.home() / ".claude" / "coordination" / "claims"
DEFAULT_TTL_HOURS = 24  # Sprints run 24h; 2h caused false-expiry conflicts mid-sprint
LIVE_STATUSES = {"active", "blocked", "handoff"}
CLAIM_TYPES = {"program", "write", "review", "research"}
STRICT_LIVE_METADATA_CLAIM_TYPES = {"program", "write", "research"}
DEFAULT_HEARTBEAT_STALE_MINUTES = 120
SESSION_ENV_KEYS = {
    "codex": ("CODEX_THREAD_ID",),
    "claude-code": ("CLAUDE_SESSION_ID", "CLAUDE_CODE_SSE_PORT"),
    "openclaw": ("OPENCLAW_SESSION_ID", "OPENCLAW_RUN_ID"),
}


@dataclass(frozen=True)
class ClaimRecord:
    """Normalized coordination claim record used across v1 and v2 schemas."""

    agent: str
    claimed_at: str | None
    expires_at: str | None
    projects: list[str]
    scope: str
    intent: str
    claim_type: str
    write_paths: list[str]
    read_paths: list[str]
    worktree_path: str | None
    repo_root: str | None
    branch: str | None
    session_name: str | None
    broader_goal: str | None
    tracker_path: str | None
    session_id: str | None
    heartbeat_at: str | None
    status: str
    updated_at: str | None
    parent_scope: str | None
    notes: str | None
    plan_ref: str | None
    source_file: str | None
    schema_version: int

    def primary_project(self) -> str | None:
        """Return the first project for CLI compatibility surfaces."""
        return self.projects[0] if self.projects else None

    def is_live(self) -> bool:
        """Return whether the claim should participate in active coordination."""
        return self.status in LIVE_STATUSES

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-safe dictionary for reporting and persistence."""
        data = asdict(self)
        data["project"] = self.primary_project()
        return data


@dataclass(frozen=True)
class ClaimInteraction:
    """Describe how one candidate claim interacts with another active claim."""

    severity: str
    reason: str
    other_agent: str
    other_scope: str
    other_claim_type: str
    projects: list[str]
    overlapping_write_paths: list[str]
    other_source_file: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe interaction summary."""
        return asdict(self)


@dataclass(frozen=True)
class ClaimCheckResult:
    """Structured result for candidate-vs-active-claims evaluation."""

    candidate: ClaimRecord
    interactions: list[ClaimInteraction]

    @property
    def hard_conflicts(self) -> list[ClaimInteraction]:
        """Return hard-conflict interactions only."""
        return [item for item in self.interactions if item.severity == "hard_conflict"]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe check result."""
        return {
            "candidate": self.candidate.to_dict(),
            "interactions": [item.to_dict() for item in self.interactions],
            "has_hard_conflict": bool(self.hard_conflicts),
        }


def claim_health_issues(claim: ClaimRecord) -> list[str]:
    """Return machine-readable health issues for one normalized claim."""
    issues: list[str] = []
    if not claim.projects:
        issues.append("missing_project")
    if claim.claim_type == "write" and not claim.write_paths:
        issues.append("missing_write_paths")
    if claim.is_live() and claim.claim_type in STRICT_LIVE_METADATA_CLAIM_TYPES:
        if not claim.branch:
            issues.append("missing_branch")
        if not claim.worktree_path:
            issues.append("missing_worktree_path")
        if not claim.session_id:
            issues.append("missing_session_id")
    return issues


def claim_health_status(claim: ClaimRecord) -> str:
    """Classify one claim as healthy or weak for registry/reporting surfaces."""
    return "weak" if claim_health_issues(claim) else "healthy"


def _heartbeat_stale_after() -> timedelta:
    """Return the configured heartbeat freshness window."""
    raw = os.environ.get("COORDINATION_HEARTBEAT_STALE_MINUTES", "").strip()
    if not raw:
        return timedelta(minutes=DEFAULT_HEARTBEAT_STALE_MINUTES)
    try:
        minutes = float(raw)
    except ValueError:
        return timedelta(minutes=DEFAULT_HEARTBEAT_STALE_MINUTES)
    if minutes <= 0:
        return timedelta(minutes=DEFAULT_HEARTBEAT_STALE_MINUTES)
    return timedelta(minutes=minutes)


def _run_git(repo_root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one git command for lifecycle diagnostics without throwing."""
    return subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )


def _resolve_repo_root_from_worktree_path(worktree_path: str | None) -> Path | None:
    """Resolve the canonical repo root from a claim worktree path when possible."""
    if not worktree_path:
        return None
    expanded = Path(worktree_path).expanduser()
    if expanded.exists():
        result = _run_git(expanded, ["rev-parse", "--show-toplevel"])
        if result.returncode == 0:
            return Path(result.stdout.strip())
    parent = expanded.parent
    if parent.name.endswith("_worktrees"):
        candidate = parent.parent / parent.name.removesuffix("_worktrees")
        if candidate.exists():
            result = _run_git(candidate, ["rev-parse", "--show-toplevel"])
            if result.returncode == 0:
                return Path(result.stdout.strip())
    return None


def _resolve_default_branch(repo_root: Path) -> str | None:
    """Return the canonical default branch name for one repo when resolvable."""
    remote_head = _run_git(repo_root, ["symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD"])
    if remote_head.returncode == 0:
        value = remote_head.stdout.strip()
        if value.startswith("origin/"):
            return value.split("/", 1)[1]
        if value:
            return value
    for candidate in ("main", "master"):
        branch_check = _run_git(repo_root, ["show-ref", "--verify", f"refs/heads/{candidate}"])
        if branch_check.returncode == 0:
            return candidate
    return None


def claim_lifecycle_issues(claim: ClaimRecord) -> list[str]:
    """Return mechanically provable stale-lifecycle issues for one live claim."""
    if not claim.is_live():
        return []

    issues: list[str] = []
    repo_root = _resolve_repo_root_from_worktree_path(claim.worktree_path)
    worktree_path = Path(claim.worktree_path).expanduser() if claim.worktree_path else None

    if worktree_path is not None and not worktree_path.exists():
        issues.append("missing_worktree_on_disk")

    if claim.branch and repo_root is not None:
        branch_ref = f"refs/heads/{claim.branch}"
        branch_check = _run_git(repo_root, ["show-ref", "--verify", branch_ref])
        branch_exists = branch_check.returncode == 0
        if not branch_exists:
            issues.append("missing_branch_ref")
        else:
            default_branch = _resolve_default_branch(repo_root)
            if default_branch and default_branch != claim.branch:
                branch_sha = _run_git(repo_root, ["rev-parse", branch_ref])
                default_sha = _run_git(repo_root, ["rev-parse", f"refs/heads/{default_branch}"])
                if branch_sha.returncode != 0 or default_sha.returncode != 0:
                    return issues
                if branch_sha.stdout.strip() == default_sha.stdout.strip():
                    return issues
                merged_check = _run_git(
                    repo_root,
                    ["merge-base", "--is-ancestor", branch_ref, f"refs/heads/{default_branch}"],
                )
                if merged_check.returncode == 0:
                    issues.append("branch_merged_to_default")

    return issues


def claim_liveness_issues(
    claim: ClaimRecord,
    *,
    now: datetime | None = None,
) -> list[str]:
    """Return stale-session issues derived from heartbeat freshness.

    Backward compatibility rule: a live claim with no `heartbeat_at` remains
    readable and does not become stale solely because the heartbeat rollout has
    not touched it yet.
    """

    if not claim.is_live():
        return []
    if not claim.session_id:
        return []
    if not claim.heartbeat_at:
        return []
    heartbeat = _parse_iso_datetime(claim.heartbeat_at)
    if heartbeat is None:
        return ["invalid_heartbeat_at"]
    reference_now = now or datetime.now(timezone.utc)
    if reference_now - heartbeat > _heartbeat_stale_after():
        return ["stale_session_heartbeat"]
    return []


def claim_runtime_status(claim: ClaimRecord) -> str:
    """Classify one live claim across stale/weak/healthy states."""
    if claim_lifecycle_issues(claim) or claim_liveness_issues(claim):
        return "stale"
    return claim_health_status(claim)


def validate_claim_for_creation(claim: ClaimRecord) -> None:
    """Reject new claims that omit required ownership metadata for live coordination."""
    issues = claim_health_issues(claim)
    if not issues:
        return
    if not claim.is_live():
        return
    flag_map = {
        "missing_project": "--project",
        "missing_write_paths": "--write-path",
        "missing_branch": "--branch",
        "missing_worktree_path": "--worktree-path",
        "missing_session_id": "--session-id",
    }
    required_flags = [flag_map[item] for item in issues if item in flag_map]
    required_text = ", ".join(required_flags)
    raise ValueError(
        f"Active {claim.claim_type} claims require {required_text}. "
        "Legacy claims remain readable, but new live claims must declare real ownership."
    )


def resolve_session_id(agent: str, explicit_session_id: str | None = None) -> str | None:
    """Return an explicit or environment-derived session identifier.

    The result is scoped to the named agent so one tool runtime does not
    accidentally borrow another tool's ambient session marker.
    """

    if explicit_session_id:
        return explicit_session_id
    for key in SESSION_ENV_KEYS.get(agent, ()):
        raw_value = os.environ.get(key, "").strip()
        if not raw_value:
            continue
        if agent == "claude-code" and key == "CLAUDE_CODE_SSE_PORT":
            return f"claude-code:sse:{raw_value}"
        return f"{agent}:{raw_value}"
    return None


def _safe_string_list(value: Any) -> list[str]:
    """Normalize a scalar-or-list YAML value into a clean string list."""
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = [item for item in value if isinstance(item, str)]
    else:
        return []
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        stripped = item.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        deduped.append(stripped)
    return deduped


def _normalize_repo_path(path: str) -> str:
    """Normalize a repo-relative path for parent/child overlap checks."""
    normalized = posixpath.normpath(path.replace("\\", "/").strip())
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _projects_overlap(left: ClaimRecord, right: ClaimRecord) -> bool:
    """Return whether two claims touch at least one common project."""
    return bool(set(left.projects) & set(right.projects))


def _paths_overlap(left: str, right: str) -> bool:
    """Return whether two normalized repo-relative paths overlap."""
    left_norm = _normalize_repo_path(left)
    right_norm = _normalize_repo_path(right)
    return (
        left_norm == right_norm
        or left_norm.startswith(f"{right_norm}/")
        or right_norm.startswith(f"{left_norm}/")
    )


def _compute_overlapping_write_paths(candidate: ClaimRecord, other: ClaimRecord) -> list[str]:
    """Return normalized write-path overlaps between two claims."""
    overlaps: list[str] = []
    for left in candidate.write_paths:
        for right in other.write_paths:
            if _paths_overlap(left, right):
                overlaps.append(f"{_normalize_repo_path(left)} <-> {_normalize_repo_path(right)}")
    return sorted(set(overlaps))


def _parse_iso_datetime(value: Any) -> datetime | None:
    """Parse an ISO timestamp from claim data if present."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def normalize_claim(data: dict[str, Any], *, source_file: str | None = None) -> ClaimRecord | None:
    """Normalize a raw YAML claim into the v2 in-memory representation."""
    agent = data.get("agent")
    scope = data.get("scope")
    intent = data.get("intent")
    if not all(isinstance(value, str) and value.strip() for value in (agent, scope, intent)):
        return None
    assert isinstance(agent, str)
    assert isinstance(scope, str)
    assert isinstance(intent, str)
    agent_text = agent.strip()
    scope_text = scope.strip()
    intent_text = intent.strip()

    projects = _safe_string_list(data.get("projects"))
    legacy_project = data.get("project")
    if isinstance(legacy_project, str) and legacy_project.strip() and legacy_project.strip() not in projects:
        projects.insert(0, legacy_project.strip())

    write_paths = [_normalize_repo_path(path) for path in _safe_string_list(data.get("write_paths"))]
    read_paths = [_normalize_repo_path(path) for path in _safe_string_list(data.get("read_paths"))]
    raw_claim_type = data.get("claim_type")
    claim_type = raw_claim_type if isinstance(raw_claim_type, str) and raw_claim_type in CLAIM_TYPES else None
    if claim_type is None:
        claim_type = "write" if write_paths else "program"

    raw_status = data.get("status")
    status = raw_status if isinstance(raw_status, str) and raw_status.strip() else "active"
    schema_version = 2 if any(
        key in data
        for key in (
            "claim_type",
            "projects",
            "write_paths",
            "read_paths",
            "worktree_path",
            "repo_root",
            "branch",
            "session_name",
            "broader_goal",
            "tracker_path",
            "session_id",
            "heartbeat_at",
            "status",
            "updated_at",
            "parent_scope",
            "notes",
        )
    ) else 1

    return ClaimRecord(
        agent=agent_text,
        claimed_at=data.get("claimed_at") if isinstance(data.get("claimed_at"), str) else None,
        expires_at=data.get("expires_at") if isinstance(data.get("expires_at"), str) else None,
        projects=projects,
        scope=scope_text,
        intent=intent_text,
        claim_type=claim_type,
        write_paths=write_paths,
        read_paths=read_paths,
        worktree_path=data.get("worktree_path") if isinstance(data.get("worktree_path"), str) else None,
        repo_root=data.get("repo_root") if isinstance(data.get("repo_root"), str) else None,
        branch=data.get("branch") if isinstance(data.get("branch"), str) else None,
        session_name=data.get("session_name") if isinstance(data.get("session_name"), str) else None,
        broader_goal=data.get("broader_goal") if isinstance(data.get("broader_goal"), str) else None,
        tracker_path=data.get("tracker_path") if isinstance(data.get("tracker_path"), str) else None,
        session_id=data.get("session_id") if isinstance(data.get("session_id"), str) else None,
        heartbeat_at=data.get("heartbeat_at") if isinstance(data.get("heartbeat_at"), str) else None,
        status=status,
        updated_at=data.get("updated_at") if isinstance(data.get("updated_at"), str) else None,
        parent_scope=data.get("parent_scope") if isinstance(data.get("parent_scope"), str) else None,
        notes=data.get("notes") if isinstance(data.get("notes"), str) else None,
        plan_ref=data.get("plan_ref") if isinstance(data.get("plan_ref"), str) else None,
        source_file=source_file,
        schema_version=schema_version,
    )


def _load_claims() -> list[ClaimRecord]:
    """Load all live claim files, pruning expired entries on read."""
    if not CLAIMS_DIR.exists():
        return []
    claims: list[ClaimRecord] = []
    now = datetime.now(timezone.utc)
    for claim_file in CLAIMS_DIR.glob("*.yaml"):
        try:
            data = yaml.safe_load(claim_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        expires_at = _parse_iso_datetime(data.get("expires_at"))
        if expires_at is not None and expires_at < now:
            claim_file.unlink()
            continue
        claim = normalize_claim(data, source_file=str(claim_file))
        if claim is not None:
            claims.append(claim)
    return claims


def _claim_filename(agent: str, project: str, scope: str) -> str:
    """Generate a deterministic filename for a claim."""
    def safe(value: str) -> str:
        return value.replace("/", "_").replace(" ", "_").strip("_")

    return f"{safe(agent)}_{safe(project)}_{safe(scope)}.yaml"


def check_claims(project: str | None = None) -> list[ClaimRecord]:
    """Check for active live claims, optionally filtered by project."""
    claims = [claim for claim in _load_claims() if claim.is_live()]
    if project:
        claims = [claim for claim in claims if project in claim.projects]
    return claims


def evaluate_claim(candidate: ClaimRecord, *, active_claims: list[ClaimRecord] | None = None) -> ClaimCheckResult:
    """Classify candidate claim interactions against active live claims."""
    claims = active_claims if active_claims is not None else check_claims()
    interactions: list[ClaimInteraction] = []
    for other in claims:
        if other.agent == candidate.agent:
            continue
        if not _projects_overlap(candidate, other):
            continue

        overlapping_write_paths = _compute_overlapping_write_paths(candidate, other)
        if candidate.claim_type == "write" and other.claim_type == "write" and overlapping_write_paths:
            interactions.append(
                ClaimInteraction(
                    severity="hard_conflict",
                    reason="write_paths overlap across active write claims",
                    other_agent=other.agent,
                    other_scope=other.scope,
                    other_claim_type=other.claim_type,
                    projects=sorted(set(candidate.projects) & set(other.projects)),
                    overlapping_write_paths=overlapping_write_paths,
                    other_source_file=other.source_file,
                )
            )
            continue

        if overlapping_write_paths and {candidate.claim_type, other.claim_type} == {"write", "review"}:
            interactions.append(
                ClaimInteraction(
                    severity="soft_overlap",
                    reason="review claim overlaps an active write claim",
                    other_agent=other.agent,
                    other_scope=other.scope,
                    other_claim_type=other.claim_type,
                    projects=sorted(set(candidate.projects) & set(other.projects)),
                    overlapping_write_paths=overlapping_write_paths,
                    other_source_file=other.source_file,
                )
            )
            continue

        if candidate.scope == other.scope:
            interactions.append(
                ClaimInteraction(
                    severity="informational",
                    reason="same project/scope is already claimed, but no write-path conflict was detected",
                    other_agent=other.agent,
                    other_scope=other.scope,
                    other_claim_type=other.claim_type,
                    projects=sorted(set(candidate.projects) & set(other.projects)),
                    overlapping_write_paths=overlapping_write_paths,
                    other_source_file=other.source_file,
                )
            )
            continue

        if overlapping_write_paths:
            interactions.append(
                ClaimInteraction(
                    severity="informational",
                    reason="write-path overlap exists but the claim types do not require blocking",
                    other_agent=other.agent,
                    other_scope=other.scope,
                    other_claim_type=other.claim_type,
                    projects=sorted(set(candidate.projects) & set(other.projects)),
                    overlapping_write_paths=overlapping_write_paths,
                    other_source_file=other.source_file,
                )
            )
            continue

        interactions.append(
            ClaimInteraction(
                severity="informational",
                reason="same project has another active claim with no overlapping write paths",
                other_agent=other.agent,
                other_scope=other.scope,
                other_claim_type=other.claim_type,
                projects=sorted(set(candidate.projects) & set(other.projects)),
                overlapping_write_paths=[],
                other_source_file=other.source_file,
            )
        )
    return ClaimCheckResult(candidate=candidate, interactions=interactions)


def build_candidate_claim(
    *,
    agent: str,
    project: str,
    scope: str,
    intent: str,
    plan_ref: str | None = None,
    claim_type: str | None = None,
    write_paths: list[str] | None = None,
    read_paths: list[str] | None = None,
    worktree_path: str | None = None,
    repo_root: str | None = None,
    branch: str | None = None,
    session_name: str | None = None,
    broader_goal: str | None = None,
    tracker_path: str | None = None,
    session_id: str | None = None,
    heartbeat_at: str | None = None,
    status: str = "active",
    parent_scope: str | None = None,
    notes: str | None = None,
    claimed_at: str | None = None,
    expires_at: str | None = None,
    updated_at: str | None = None,
) -> ClaimRecord:
    """Build a normalized candidate claim from CLI or test inputs."""
    normalized_write_paths = [_normalize_repo_path(path) for path in (write_paths or [])]
    normalized_read_paths = [_normalize_repo_path(path) for path in (read_paths or [])]
    resolved_session_id = resolve_session_id(agent, session_id)
    resolved_claim_type = claim_type or ("write" if normalized_write_paths else "program")
    if resolved_claim_type not in CLAIM_TYPES:
        raise ValueError(f"Unsupported claim type: {resolved_claim_type}")
    if resolved_claim_type == "write" and not normalized_write_paths:
        raise ValueError("Write claims require at least one --write-path.")
    return ClaimRecord(
        agent=agent,
        claimed_at=claimed_at,
        expires_at=expires_at,
        projects=[project],
        scope=scope,
        intent=intent,
        claim_type=resolved_claim_type,
        write_paths=normalized_write_paths,
        read_paths=normalized_read_paths,
        worktree_path=worktree_path,
        repo_root=repo_root,
        branch=branch,
        session_name=session_name,
        broader_goal=broader_goal,
        tracker_path=tracker_path,
        session_id=resolved_session_id,
        heartbeat_at=heartbeat_at,
        status=status,
        updated_at=updated_at,
        parent_scope=parent_scope,
        notes=notes,
        plan_ref=plan_ref,
        source_file=None,
        schema_version=2,
    )


def create_claim(
    agent: str,
    project: str,
    scope: str,
    intent: str,
    plan_ref: str | None = None,
    ttl_hours: float = DEFAULT_TTL_HOURS,
    claim_type: str | None = None,
    write_paths: list[str] | None = None,
    read_paths: list[str] | None = None,
    worktree_path: str | None = None,
    repo_root: str | None = None,
    branch: str | None = None,
    session_name: str | None = None,
    broader_goal: str | None = None,
    tracker_path: str | None = None,
    session_id: str | None = None,
    status: str = "active",
    parent_scope: str | None = None,
    notes: str | None = None,
) -> tuple[bool, str]:
    """Create a new claim after checking for hard conflicts."""
    now = datetime.now(timezone.utc)
    candidate = build_candidate_claim(
        agent=agent,
        project=project,
        scope=scope,
        intent=intent,
        plan_ref=plan_ref,
        claim_type=claim_type,
        write_paths=write_paths,
        read_paths=read_paths,
        worktree_path=worktree_path,
        repo_root=repo_root,
        branch=branch,
        session_name=session_name,
        broader_goal=broader_goal,
        tracker_path=tracker_path,
        session_id=session_id,
        heartbeat_at=now.isoformat(),
        status=status,
        parent_scope=parent_scope,
        notes=notes,
        claimed_at=now.isoformat(),
        expires_at=(now + timedelta(hours=ttl_hours)).isoformat(),
        updated_at=now.isoformat(),
    )
    validate_claim_for_creation(candidate)

    check_result = evaluate_claim(candidate, active_claims=check_claims(project))
    if check_result.hard_conflicts:
        formatted = "; ".join(
            f"{item.other_agent} ({item.other_scope}: {', '.join(item.overlapping_write_paths)})"
            for item in check_result.hard_conflicts
        )
        return False, f"CONFLICT: active write claim overlap in '{project}' — {formatted}"

    CLAIMS_DIR.mkdir(parents=True, exist_ok=True)
    filename = _claim_filename(agent, project, scope)
    claim_payload = candidate.to_dict()
    claim_payload.pop("source_file", None)
    claim_payload.pop("project", None)
    (CLAIMS_DIR / filename).write_text(
        yaml.safe_dump(claim_payload, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    return True, (
        f"Claimed: {agent} → {project}:{scope} "
        f"[{candidate.claim_type}] (expires in {ttl_hours}h)"
    )


def hydrate_missing_session_ids(
    *,
    agent: str,
    project: str,
    session_id: str | None = None,
    scope: str | None = None,
    branch: str | None = None,
) -> tuple[int, list[str], str]:
    """Fill in missing session IDs for matching live claims.

    This is an explicit remediation tool for older live claims that were created
    before automatic session capture was wired into the v2 claim surface.
    """

    resolved_session_id = resolve_session_id(agent, session_id)
    if not resolved_session_id:
        raise ValueError(
            "Unable to resolve a session ID. Pass --session-id explicitly or run from a supported tool runtime."
        )

    if not CLAIMS_DIR.exists():
        return 0, [], resolved_session_id

    updated_scopes: list[str] = []
    now = datetime.now(timezone.utc).isoformat()
    for claim_file in CLAIMS_DIR.glob("*.yaml"):
        try:
            data = yaml.safe_load(claim_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        claim = normalize_claim(data, source_file=str(claim_file))
        if claim is None or not claim.is_live():
            continue
        if claim.agent != agent:
            continue
        if project not in claim.projects:
            continue
        if scope and claim.scope != scope:
            continue
        if branch and claim.branch != branch:
            continue
        if claim.session_id:
            continue
        data["session_id"] = resolved_session_id
        data["heartbeat_at"] = now
        data["updated_at"] = now
        claim_file.write_text(
            yaml.safe_dump(data, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )
        updated_scopes.append(claim.scope)
    return len(updated_scopes), sorted(updated_scopes), resolved_session_id


def heartbeat_claims(
    *,
    agent: str,
    project: str,
    session_id: str | None = None,
    scope: str | None = None,
    branch: str | None = None,
) -> tuple[int, list[str], str, str]:
    """Refresh heartbeat metadata for matching live claims owned by one session."""

    resolved_session_id = resolve_session_id(agent, session_id)
    if not resolved_session_id:
        raise ValueError(
            "Unable to resolve a session ID. Pass --session-id explicitly or run from a supported tool runtime."
        )

    if not CLAIMS_DIR.exists():
        return 0, [], resolved_session_id, datetime.now(timezone.utc).isoformat()

    heartbeat_at = datetime.now(timezone.utc).isoformat()
    updated_scopes: list[str] = []
    for claim_file in CLAIMS_DIR.glob("*.yaml"):
        try:
            data = yaml.safe_load(claim_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        claim = normalize_claim(data, source_file=str(claim_file))
        if claim is None or not claim.is_live():
            continue
        if claim.agent != agent:
            continue
        if project not in claim.projects:
            continue
        if scope and claim.scope != scope:
            continue
        if branch and claim.branch != branch:
            continue
        if claim.session_id and claim.session_id != resolved_session_id:
            continue
        data["session_id"] = resolved_session_id
        data["heartbeat_at"] = heartbeat_at
        data["updated_at"] = heartbeat_at
        claim_file.write_text(
            yaml.safe_dump(data, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )
        updated_scopes.append(claim.scope)
    return len(updated_scopes), sorted(updated_scopes), resolved_session_id, heartbeat_at


def release_claim(agent: str, project: str, scope: str) -> tuple[bool, str]:
    """Release an existing claim."""
    filename = _claim_filename(agent, project, scope)
    path = CLAIMS_DIR / filename
    if path.exists():
        path.unlink()
        return True, f"Released: {agent} → {project}:{scope}"
    return False, f"No claim found for {agent} → {project}:{scope}"


def complete_claims_for_plan(
    *,
    project: str,
    plan_ref: str,
    note: str | None = None,
) -> tuple[int, list[str]]:
    """Mark matching live claims completed and return the affected scopes.

    This is the lifecycle-closeout path for finished lanes: claims stop being
    active coordination input, but the YAML records remain on disk as audit
    history with an explicit `completed` status.
    """

    if not CLAIMS_DIR.exists():
        return 0, []

    now = datetime.now(timezone.utc).isoformat()
    completed_scopes: list[str] = []
    for claim_file in CLAIMS_DIR.glob("*.yaml"):
        try:
            data = yaml.safe_load(claim_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        claim = normalize_claim(data, source_file=str(claim_file))
        if claim is None or not claim.is_live():
            continue
        if project not in claim.projects:
            continue
        if claim.plan_ref != plan_ref:
            continue
        data["status"] = "completed"
        data["updated_at"] = now
        if note:
            existing_notes = data.get("notes")
            if isinstance(existing_notes, str) and existing_notes.strip():
                if note not in existing_notes:
                    data["notes"] = f"{existing_notes.rstrip()} | {note}"
            else:
                data["notes"] = note
        claim_file.write_text(
            yaml.safe_dump(data, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )
        completed_scopes.append(claim.scope)
    return len(completed_scopes), sorted(completed_scopes)


def prune_expired() -> int:
    """Remove expired claims and return the number pruned."""
    if not CLAIMS_DIR.exists():
        return 0
    now = datetime.now(timezone.utc)
    removed = 0
    for claim_file in CLAIMS_DIR.glob("*.yaml"):
        try:
            data = yaml.safe_load(claim_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        expires_at = _parse_iso_datetime(data.get("expires_at") if isinstance(data, dict) else None)
        if expires_at is not None and expires_at < now:
            claim_file.unlink()
            removed += 1
    return removed


def prune_stale() -> tuple[int, list[str]]:
    """Remove stale live claims and return the removal count plus scope labels."""
    if not CLAIMS_DIR.exists():
        return 0, []
    removed_labels: list[str] = []
    for claim_file in CLAIMS_DIR.glob("*.yaml"):
        try:
            data = yaml.safe_load(claim_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        claim = normalize_claim(data, source_file=str(claim_file))
        if claim is None or not claim.is_live():
            continue
        if not (claim_lifecycle_issues(claim) or claim_liveness_issues(claim)):
            continue
        claim_file.unlink()
        removed_labels.append(f"{claim.primary_project()}:{claim.scope}")
    return len(removed_labels), sorted(removed_labels)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for coordination-claim management."""
    parser = argparse.ArgumentParser(description="Cross-brain coordination claims")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Check for active claims")
    group.add_argument("--claim", action="store_true", help="Create a new claim")
    group.add_argument("--release", action="store_true", help="Release an existing claim")
    group.add_argument("--list", action="store_true", help="List all active claims")
    group.add_argument("--prune", action="store_true", help="Remove expired claims")
    group.add_argument(
        "--prune-stale",
        action="store_true",
        help="Remove mechanically stale live claims whose lifecycle state is no longer truthful.",
    )
    group.add_argument(
        "--hydrate-session-ids",
        action="store_true",
        help="Fill in missing session_id metadata for matching live claims.",
    )
    group.add_argument(
        "--heartbeat",
        action="store_true",
        help="Refresh heartbeat metadata for matching live claims owned by the current session.",
    )

    parser.add_argument("--agent", help="Agent brain name (claude-code, codex, openclaw)")
    parser.add_argument("--project", help="Project name")
    parser.add_argument("--scope", help="Scope path or identifier")
    parser.add_argument("--intent", help="What the agent intends to do")
    parser.add_argument("--plan", help="Plan reference (e.g., Plan #28)")
    parser.add_argument("--ttl-hours", type=float, default=DEFAULT_TTL_HOURS, help="Claim TTL in hours")
    parser.add_argument("--claim-type", choices=sorted(CLAIM_TYPES), help="Claim category")
    parser.add_argument("--write-path", action="append", default=[], help="Repo-relative write path")
    parser.add_argument("--read-path", action="append", default=[], help="Repo-relative read path")
    parser.add_argument("--worktree-path", help="Worktree path for this claim")
    parser.add_argument("--branch", help="Branch for this claim")
    parser.add_argument("--session-id", help="Session identifier")
    parser.add_argument("--status", default="active", help="Claim status (default: active)")
    parser.add_argument("--parent-scope", help="Parent/broad-scope identifier")
    parser.add_argument("--notes", help="Freeform notes")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    return parser.parse_args(argv)


def _render_check_output(
    *,
    claims: list[ClaimRecord],
    project: str | None,
    candidate: ClaimRecord | None,
) -> dict[str, Any]:
    """Build a structured report for list/check operations."""
    payload: dict[str, Any] = {
        "project": project,
        "claims": [
            {
                **claim.to_dict(),
                "health_status": claim_runtime_status(claim),
                "health_issues": claim_health_issues(claim),
                "lifecycle_issues": claim_lifecycle_issues(claim),
                "liveness_issues": claim_liveness_issues(claim),
            }
            for claim in claims
        ],
    }
    if candidate is not None:
        payload["check"] = {
            **evaluate_claim(candidate, active_claims=claims).to_dict(),
            "candidate_health_status": claim_runtime_status(candidate),
            "candidate_health_issues": claim_health_issues(candidate),
            "candidate_lifecycle_issues": claim_lifecycle_issues(candidate),
            "candidate_liveness_issues": claim_liveness_issues(candidate),
        }
    return payload


def main(argv: list[str] | None = None) -> int:
    """Run the CLI for cross-brain coordination claim management."""
    args = parse_args(argv)

    if args.check:
        claims = check_claims(args.project)
        candidate = None
        if args.project and (args.claim_type or args.write_path or args.scope or args.agent):
            candidate = build_candidate_claim(
                agent=args.agent or "candidate",
                project=args.project,
                scope=args.scope or "preview",
                intent=args.intent or "preview active coordination interactions",
                plan_ref=args.plan,
                claim_type=args.claim_type,
                write_paths=args.write_path,
                read_paths=args.read_path,
                worktree_path=args.worktree_path,
                branch=args.branch,
                session_id=args.session_id,
                status=args.status,
                parent_scope=args.parent_scope,
                notes=args.notes,
            )
        if args.json:
            print(json.dumps(_render_check_output(claims=claims, project=args.project, candidate=candidate), indent=2))
            return 0
        if not claims:
            print("No active claims.")
            return 0
        for claim in claims:
            print(
                f"  [{claim.agent}] {claim.primary_project()}:{claim.scope} "
                f"[{claim.claim_type}] — {claim.intent}"
            )
            print(f"    expires: {claim.expires_at}")
            if claim.write_paths:
                print(f"    write_paths: {', '.join(claim.write_paths)}")
        if candidate is not None:
            result = evaluate_claim(candidate, active_claims=claims)
            if not result.interactions:
                print("No interactions for candidate claim.")
            else:
                print("Candidate interactions:")
                for item in result.interactions:
                    overlaps = ", ".join(item.overlapping_write_paths) or "none"
                    print(
                        f"  - {item.severity}: {item.other_agent} {item.other_scope} "
                        f"({item.reason}; overlap={overlaps})"
                    )
        return 0

    if args.list:
        claims = check_claims()
        if args.json:
            print(json.dumps(_render_check_output(claims=claims, project=args.project, candidate=None), indent=2))
            return 0
        if not claims:
            print("No active claims.")
            return 0
        for claim in claims:
            print(
                f"  [{claim.agent}] {claim.primary_project()}:{claim.scope} "
                f"[{claim.claim_type}] — {claim.intent}"
            )
        return 0

    if args.claim:
        if not all([args.agent, args.project, args.scope, args.intent]):
            raise SystemExit("--claim requires --agent, --project, --scope, --intent")
        try:
            ok, msg = create_claim(
                args.agent,
                args.project,
                args.scope,
                args.intent,
                args.plan,
                args.ttl_hours,
                claim_type=args.claim_type,
                write_paths=args.write_path,
                read_paths=args.read_path,
                worktree_path=args.worktree_path,
                branch=args.branch,
                session_id=args.session_id,
                status=args.status,
                parent_scope=args.parent_scope,
                notes=args.notes,
            )
        except ValueError as exc:
            if args.json:
                print(json.dumps({"ok": False, "message": str(exc)}, indent=2))
            else:
                print(str(exc))
            return 1
        if args.json:
            print(json.dumps({"ok": ok, "message": msg}, indent=2))
        else:
            print(msg)
        return 0 if ok else 1

    if args.release:
        if not all([args.agent, args.project, args.scope]):
            raise SystemExit("--release requires --agent, --project, --scope")
        ok, msg = release_claim(args.agent, args.project, args.scope)
        if args.json:
            print(json.dumps({"ok": ok, "message": msg}, indent=2))
        else:
            print(msg)
        return 0 if ok else 1

    if args.prune:
        removed = prune_expired()
        if args.json:
            print(json.dumps({"pruned": removed}, indent=2))
        else:
            print(f"Expired claims pruned: {removed}")
        return 0

    if args.prune_stale:
        removed, removed_scopes = prune_stale()
        payload = {"pruned": removed, "removed_scopes": removed_scopes}
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"Stale claims pruned: {removed}")
            if removed_scopes:
                print("Removed scopes: " + ", ".join(removed_scopes))
        return 0

    if args.hydrate_session_ids:
        if not all([args.agent, args.project]):
            raise SystemExit("--hydrate-session-ids requires --agent and --project")
        try:
            updated_count, updated_scopes, resolved_session_id = hydrate_missing_session_ids(
                agent=args.agent,
                project=args.project,
                session_id=args.session_id,
                scope=args.scope,
                branch=args.branch,
            )
        except ValueError as exc:
            if args.json:
                print(json.dumps({"ok": False, "message": str(exc)}, indent=2))
            else:
                print(str(exc))
            return 1
        payload = {
            "ok": True,
            "updated_count": updated_count,
            "updated_scopes": updated_scopes,
            "session_id": resolved_session_id,
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(
                f"Hydrated {updated_count} claim(s) for {args.agent}:{args.project} "
                f"with session_id={resolved_session_id}"
            )
        return 0

    if args.heartbeat:
        if not all([args.agent, args.project]):
            raise SystemExit("--heartbeat requires --agent and --project")
        try:
            updated_count, updated_scopes, resolved_session_id, heartbeat_at = heartbeat_claims(
                agent=args.agent,
                project=args.project,
                session_id=args.session_id,
                scope=args.scope,
                branch=args.branch,
            )
        except ValueError as exc:
            if args.json:
                print(json.dumps({"ok": False, "message": str(exc)}, indent=2))
            else:
                print(str(exc))
            return 1
        payload = {
            "ok": True,
            "updated_count": updated_count,
            "updated_scopes": updated_scopes,
            "session_id": resolved_session_id,
            "heartbeat_at": heartbeat_at,
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(
                f"Heartbeated {updated_count} claim(s) for {args.agent}:{args.project} "
                f"with session_id={resolved_session_id} at {heartbeat_at}"
            )
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
