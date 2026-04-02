#!/usr/bin/env python3
"""Authoritative local coordination registry for active work and plan reservations.

This module replaces split coordination state with one local-only registry under
``~/.claude/coordination/``. It deliberately keeps runtime state out of git and
provides deterministic locking so multiple local agents cannot reserve the same
plan number or silently diverge on active-work ownership.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml


DEFAULT_TTL_HOURS = 2.0
REGISTRY_VERSION = 1


def coordination_dir() -> Path:
    """Return the local coordination directory."""
    override = os.environ.get("CLAUDE_COORDINATION_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".claude" / "coordination"


def registry_path() -> Path:
    """Return the authoritative registry path."""
    override = os.environ.get("CLAUDE_COORDINATION_REGISTRY")
    if override:
        return Path(override).expanduser()
    return coordination_dir() / "active-work-registry.yaml"


def _legacy_claims_dir() -> Path:
    """Return the legacy cross-brain claims directory for one-way ingest."""
    return coordination_dir() / "claims"


def _now() -> datetime:
    """Return the current UTC time."""
    return datetime.now(timezone.utc)


def _iso_now() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return _now().isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    """Parse an ISO 8601 timestamp if present."""
    if not value:
        return None
    return datetime.fromisoformat(value)


def _default_registry() -> dict[str, Any]:
    """Return an empty authoritative registry payload."""
    return {
        "version": REGISTRY_VERSION,
        "active_work": [],
        "plan_reservations": [],
    }


@contextlib.contextmanager
def locked_registry() -> Iterator[dict[str, Any]]:
    """Yield the authoritative registry while holding an exclusive file lock."""
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        raw = handle.read()
        data = yaml.safe_load(raw) if raw.strip() else None
        registry = data if isinstance(data, dict) else _default_registry()
        registry.setdefault("version", REGISTRY_VERSION)
        registry.setdefault("active_work", [])
        registry.setdefault("plan_reservations", [])
        yield registry
        handle.seek(0)
        handle.truncate()
        yaml.safe_dump(registry, handle, sort_keys=False)
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _prune_expired(registry: dict[str, Any]) -> None:
    """Prune expired active-work entries in place."""
    now = _now()
    kept: list[dict[str, Any]] = []
    for entry in registry.get("active_work", []):
        expires_at = _parse_iso(entry.get("expires_at"))
        status = entry.get("status", "active")
        if status == "released":
            continue
        if expires_at and expires_at < now:
            continue
        kept.append(entry)
    registry["active_work"] = kept


def _normalize_path(value: str | Path | None) -> str | None:
    """Return a stable normalized string path."""
    if value is None:
        return None
    return str(Path(value).expanduser())


def _repo_name(repo_root: str | Path | None, project: str | None) -> str:
    """Return the coordination project key."""
    if project:
        return project
    if repo_root:
        return Path(repo_root).expanduser().name
    raise ValueError("project or repo_root is required")


def _find_active_work(
    registry: dict[str, Any],
    *,
    project: str,
    scope: str,
    agent: str | None = None,
) -> dict[str, Any] | None:
    """Return the active-work entry matching the provided identity."""
    for entry in registry.get("active_work", []):
        if entry.get("project") != project or entry.get("scope") != scope:
            continue
        if agent is not None and entry.get("agent") != agent:
            continue
        return entry
    return None


def list_registry(project: str | None = None) -> dict[str, Any]:
    """Return the authoritative registry, optionally filtered by project."""
    with locked_registry() as registry:
        _prune_expired(registry)
        active = registry.get("active_work", [])
        reservations = registry.get("plan_reservations", [])
        if project:
            active = [entry for entry in active if entry.get("project") == project]
            reservations = [
                entry for entry in reservations if entry.get("project") == project
            ]
        return {
            "version": registry.get("version", REGISTRY_VERSION),
            "active_work": active,
            "plan_reservations": reservations,
        }


def get_active_plan_number(
    *,
    repo_root: str | Path | None,
    branch: str,
    worktree_path: str | Path | None = None,
) -> int | None:
    """Resolve the active plan number from the authoritative registry."""
    if repo_root is None:
        return None
    project = _repo_name(repo_root, None)
    normalized_worktree = _normalize_path(worktree_path)
    normalized_repo = _normalize_path(repo_root)
    payload = list_registry(project=project)
    for entry in payload["active_work"]:
        if entry.get("status", "active") != "active":
            continue
        plan = entry.get("plan")
        if plan in (None, ""):
            continue
        if normalized_worktree and entry.get("worktree_path") == normalized_worktree:
            return int(plan)
    for entry in payload["active_work"]:
        if entry.get("status", "active") != "active":
            continue
        plan = entry.get("plan")
        if plan in (None, ""):
            continue
        if entry.get("repo_root") == normalized_repo and entry.get("branch") == branch:
            return int(plan)
    return None


def ingest_legacy_state(repo_root: str | Path | None = None) -> dict[str, int]:
    """Ingest legacy claim sources into the authoritative registry once."""
    ingested_active = 0
    ingested_reservations = 0
    with locked_registry() as registry:
        _prune_expired(registry)

        # Ingest legacy cross-brain claim YAML files.
        claims_dir = _legacy_claims_dir()
        if claims_dir.exists():
            for claim_file in claims_dir.glob("*.yaml"):
                try:
                    payload = yaml.safe_load(claim_file.read_text(encoding="utf-8"))
                except yaml.YAMLError:
                    continue
                if not isinstance(payload, dict):
                    continue
                project = payload.get("project")
                scope = payload.get("scope")
                agent = payload.get("agent")
                intent = payload.get("intent", "")
                if not project or not scope or not agent:
                    continue
                existing = _find_active_work(
                    registry, project=project, scope=scope, agent=agent
                )
                if existing:
                    continue
                registry["active_work"].append(
                    {
                        "agent": agent,
                        "branch": payload.get("branch"),
                        "claimed_at": payload.get("claimed_at", _iso_now()),
                        "expires_at": payload.get("expires_at"),
                        "intent": intent,
                        "observed_files": [],
                        "plan": payload.get("plan_ref"),
                        "project": project,
                        "repo_root": _normalize_path(repo_root),
                        "scope": scope,
                        "session_id": payload.get("session_id"),
                        "status": "active",
                        "task": intent,
                        "updated_at": _iso_now(),
                        "worktree_path": payload.get("worktree_path"),
                    }
                )
                ingested_active += 1

        # Ingest legacy repo-local active-work state if available.
        if repo_root:
            legacy_file = Path(repo_root).expanduser() / ".claude" / "active-work.yaml"
            if legacy_file.exists():
                try:
                    payload = yaml.safe_load(legacy_file.read_text(encoding="utf-8"))
                except yaml.YAMLError:
                    payload = {}
                claims = payload.get("claims", []) if isinstance(payload, dict) else []
                for claim in claims:
                    if not isinstance(claim, dict):
                        continue
                    project = _repo_name(repo_root, None)
                    scope = claim.get("cc_id")
                    agent = claim.get("agent", "legacy-worktree")
                    if not scope:
                        continue
                    existing = _find_active_work(
                        registry, project=project, scope=scope, agent=agent
                    )
                    if existing:
                        continue
                    registry["active_work"].append(
                        {
                            "agent": agent,
                            "branch": scope,
                            "claimed_at": claim.get("claimed_at", _iso_now()),
                            "expires_at": None,
                            "intent": claim.get("task", ""),
                            "observed_files": [],
                            "plan": claim.get("plan"),
                            "project": project,
                            "repo_root": _normalize_path(repo_root),
                            "scope": scope,
                            "session_id": claim.get("session_id"),
                            "status": "active",
                            "task": claim.get("task", ""),
                            "updated_at": _iso_now(),
                            "worktree_path": None,
                        }
                    )
                    ingested_active += 1

    return {
        "active_work": ingested_active,
        "plan_reservations": ingested_reservations,
    }


def claim_active_work(
    *,
    agent: str,
    project: str | None,
    scope: str,
    intent: str,
    repo_root: str | Path | None = None,
    branch: str | None = None,
    worktree_path: str | Path | None = None,
    session_id: str | None = None,
    plan: str | int | None = None,
    ttl_hours: float = DEFAULT_TTL_HOURS,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Claim active work in the authoritative registry."""
    resolved_project = _repo_name(repo_root, project)
    with locked_registry() as registry:
        _prune_expired(registry)
        for entry in registry.get("active_work", []):
            if entry.get("project") != resolved_project:
                continue
            if entry.get("scope") != scope:
                continue
            if entry.get("agent") == agent:
                continue
            return (
                False,
                (
                    f"CONFLICT: scope '{scope}' in '{resolved_project}' already claimed by "
                    f"{entry.get('agent')} — task: {entry.get('task') or entry.get('intent')}"
                ),
                entry,
            )

        now = _now()
        existing = _find_active_work(
            registry, project=resolved_project, scope=scope, agent=agent
        )
        payload = {
            "agent": agent,
            "branch": branch,
            "claimed_at": existing.get("claimed_at") if existing else now.isoformat(),
            "expires_at": (now + timedelta(hours=ttl_hours)).isoformat(),
            "intent": intent,
            "observed_files": existing.get("observed_files", []) if existing else [],
            "plan": plan,
            "project": resolved_project,
            "repo_root": _normalize_path(repo_root),
            "scope": scope,
            "session_id": session_id,
            "status": "active",
            "task": intent,
            "updated_at": now.isoformat(),
            "worktree_path": _normalize_path(worktree_path),
        }
        if existing:
            existing.update(payload)
        else:
            registry["active_work"].append(payload)
        return True, f"Claimed: {agent} → {resolved_project}:{scope}", payload


def release_active_work(agent: str, project: str | None, scope: str, repo_root: str | Path | None = None) -> tuple[bool, str]:
    """Release one active-work claim."""
    resolved_project = _repo_name(repo_root, project)
    with locked_registry() as registry:
        _prune_expired(registry)
        remaining: list[dict[str, Any]] = []
        removed = False
        for entry in registry.get("active_work", []):
            if (
                entry.get("agent") == agent
                and entry.get("project") == resolved_project
                and entry.get("scope") == scope
            ):
                removed = True
                continue
            remaining.append(entry)
        registry["active_work"] = remaining
        if removed:
            return True, f"Released: {agent} → {resolved_project}:{scope}"
        return False, f"No active work found for {agent} → {resolved_project}:{scope}"


def observe_file(
    *,
    agent: str,
    project: str | None,
    scope: str,
    file_path: str | Path,
    repo_root: str | Path | None = None,
) -> tuple[bool, str]:
    """Record one touched file against an active-work entry."""
    resolved_project = _repo_name(repo_root, project)
    observed = _normalize_path(file_path)
    with locked_registry() as registry:
        _prune_expired(registry)
        entry = _find_active_work(
            registry, project=resolved_project, scope=scope, agent=agent
        )
        if entry is None:
            return False, f"No active work found for {agent} → {resolved_project}:{scope}"
        files = entry.setdefault("observed_files", [])
        if observed not in files:
            files.append(observed)
        entry["updated_at"] = _iso_now()
        return True, f"Observed file: {observed}"


def _git_output(repo_root: Path, args: list[str]) -> str:
    """Run one git command and return stdout or raise."""
    result = subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return result.stdout


def _plan_numbers_from_filenames(paths: list[str]) -> set[int]:
    """Extract plan numbers from filenames."""
    numbers: set[int] = set()
    for path in paths:
        name = Path(path).name
        match = re.match(r"(\d+)_.*\.md$", name)
        if match:
            numbers.add(int(match.group(1)))
    return numbers


def existing_plan_numbers(
    repo_root: str | Path,
    *,
    ref: str = "origin/master",
    fetch: bool = True,
) -> set[int]:
    """Return known plan numbers from live origin/master plus the local checkout."""
    repo = Path(repo_root).expanduser()
    numbers = _plan_numbers_from_filenames(
        [str(path.relative_to(repo)) for path in (repo / "docs" / "plans").glob("[0-9]*_*.md")]
    )

    if fetch:
        subprocess.run(
            ["git", "fetch", "origin", "--prune"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=False,
        )

    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", ref, "docs/plans"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        remote_paths = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        numbers |= _plan_numbers_from_filenames(remote_paths)
    return numbers


def preview_next_plan_number(
    *,
    repo_root: str | Path,
    fetch: bool = True,
) -> int:
    """Return the next free plan number without creating a reservation."""
    repo = Path(repo_root).expanduser()
    project = repo.name
    with locked_registry() as registry:
        _prune_expired(registry)
        numbers = existing_plan_numbers(repo, fetch=fetch)
        reserved = {
            int(entry["plan"])
            for entry in registry.get("plan_reservations", [])
            if entry.get("project") == project and entry.get("status", "reserved") == "reserved"
        }
        return (max(numbers | reserved) + 1) if (numbers or reserved) else 1


def reserve_next_plan_number(
    *,
    repo_root: str | Path,
    agent: str,
    task: str,
    branch: str | None = None,
    worktree_path: str | Path | None = None,
    session_id: str | None = None,
    fetch: bool = True,
) -> dict[str, Any]:
    """Reserve the next free plan number for a repo in the authoritative registry."""
    repo = Path(repo_root).expanduser()
    project = repo.name
    with locked_registry() as registry:
        _prune_expired(registry)
        numbers = existing_plan_numbers(repo, fetch=fetch)
        reserved = {
            int(entry["plan"])
            for entry in registry.get("plan_reservations", [])
            if entry.get("project") == project and entry.get("status", "reserved") == "reserved"
        }
        next_plan = (max(numbers | reserved) + 1) if (numbers or reserved) else 1
        now = _iso_now()
        payload = {
            "agent": agent,
            "branch": branch,
            "project": project,
            "repo_root": _normalize_path(repo),
            "reserved_at": now,
            "session_id": session_id,
            "status": "reserved",
            "task": task,
            "plan": next_plan,
            "worktree_path": _normalize_path(worktree_path),
        }
        registry.setdefault("plan_reservations", []).append(payload)
        return payload


def release_plan_reservation(
    *,
    repo_root: str | Path,
    plan: int,
) -> tuple[bool, str]:
    """Release one reserved plan number."""
    project = Path(repo_root).expanduser().name
    with locked_registry() as registry:
        kept: list[dict[str, Any]] = []
        removed = False
        for entry in registry.get("plan_reservations", []):
            if (
                entry.get("project") == project
                and int(entry.get("plan")) == int(plan)
                and entry.get("status", "reserved") == "reserved"
            ):
                removed = True
                continue
            kept.append(entry)
        registry["plan_reservations"] = kept
        if removed:
            return True, f"Released reservation for Plan #{plan}"
        return False, f"No reservation found for Plan #{plan}"


def consume_plan_reservation(
    *,
    repo_root: str | Path,
    plan: int,
    plan_file: str | Path,
) -> tuple[bool, str]:
    """Mark one plan reservation as consumed."""
    project = Path(repo_root).expanduser().name
    with locked_registry() as registry:
        for entry in registry.get("plan_reservations", []):
            if (
                entry.get("project") == project
                and int(entry.get("plan")) == int(plan)
                and entry.get("status", "reserved") == "reserved"
            ):
                entry["status"] = "consumed"
                entry["consumed_at"] = _iso_now()
                entry["plan_file"] = _normalize_path(plan_file)
                return True, f"Consumed reservation for Plan #{plan}"
        return False, f"No reservation found for Plan #{plan}"
