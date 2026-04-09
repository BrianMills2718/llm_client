"""Session lifecycle operations for sanctioned worktree flows.

The session contract lives partly on the canonical claim and partly in the
linked tracker artifact. This module keeps those surfaces in sync without
inventing a second coordination registry.
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from enforced_planning import coordination_claims, session_contracts
from enforced_planning import doc_authority
from enforced_planning.worktree_paths import resolve_canonical_repo_root


def _split_cli_values(values: list[str] | None) -> list[str]:
    """Normalize repeated or delimiter-packed CLI values into one clean list."""

    items: list[str] = []
    for value in values or []:
        for chunk in value.replace(";", "|").split("|"):
            text = chunk.strip()
            if text:
                items.append(text)
    return items


def _claim_path(agent: str, project: str, scope: str) -> Path:
    """Return the canonical YAML path for one claim."""

    return coordination_claims.CLAIMS_DIR / coordination_claims._claim_filename(agent, project, scope)


def _load_claim_payload(agent: str, project: str, scope: str) -> dict[str, Any] | None:
    """Load one claim payload if it exists."""

    path = _claim_path(agent, project, scope)
    if not path.exists():
        return None
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Claim file at {path} must be a YAML mapping")
    return raw


def _write_claim_payload(path: Path, payload: dict[str, Any]) -> None:
    """Persist one normalized claim payload."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )


def _upsert_session_claim(
    *,
    agent: str,
    project: str,
    scope: str,
    intent: str,
    plan_ref: str | None,
    repo_root: str,
    worktree_path: str,
    branch: str,
    session_id: str,
    broader_goal: str,
    session_name: str,
    tracker_path: str,
    claim_type: str = "program",
    ttl_hours: float = coordination_claims.DEFAULT_TTL_HOURS,
) -> str:
    """Create or update the compact claim-side session contract metadata."""

    path = _claim_path(agent, project, scope)
    now = datetime.now(timezone.utc)
    existing_payload = _load_claim_payload(agent, project, scope)
    if existing_payload is None:
        ok, message = coordination_claims.create_claim(
            agent=agent,
            project=project,
            scope=scope,
            intent=intent,
            plan_ref=plan_ref,
            claim_type=claim_type,
            repo_root=repo_root,
            worktree_path=worktree_path,
            branch=branch,
            session_id=session_id,
            session_name=session_name,
            broader_goal=broader_goal,
            tracker_path=tracker_path,
            ttl_hours=ttl_hours,
        )
        if not ok:
            raise ValueError(message)
        return "created"

    existing = coordination_claims.normalize_claim(existing_payload, source_file=str(path))
    if existing is None:
        raise ValueError(f"Existing claim at {path} is invalid")
    if existing.agent != agent:
        raise ValueError(f"Claim at {path} belongs to {existing.agent}, not {agent}")
    if existing.session_id and existing.session_id != session_id:
        raise ValueError(
            f"Claim at {path} belongs to session {existing.session_id}, not {session_id}"
        )

    expires_at = existing.expires_at or (now + timedelta(hours=ttl_hours)).isoformat()
    payload = {
        **existing_payload,
        "agent": agent,
        "project": project,
        "projects": [project],
        "scope": scope,
        "intent": intent,
        "plan_ref": plan_ref,
        "repo_root": repo_root,
        "worktree_path": worktree_path,
        "branch": branch,
        "session_id": session_id,
        "session_name": session_name,
        "broader_goal": broader_goal,
        "tracker_path": tracker_path,
        "status": "active",
        "heartbeat_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "claimed_at": existing.claimed_at or now.isoformat(),
        "expires_at": expires_at,
        "claim_type": existing.claim_type or claim_type,
    }
    _write_claim_payload(path, payload)
    return "updated"


def _iter_matching_live_claims(
    *,
    agent: str | None = None,
    project: str | None = None,
    scope: str | None = None,
    branch: str | None = None,
) -> list[coordination_claims.ClaimRecord]:
    """Return live claims filtered to one bounded session scope."""

    claims = coordination_claims.check_claims(project)
    filtered: list[coordination_claims.ClaimRecord] = []
    for claim in claims:
        if agent and claim.agent != agent:
            continue
        if project and project not in claim.projects:
            continue
        if scope and claim.scope != scope:
            continue
        if branch and claim.branch != branch:
            continue
        filtered.append(claim)
    return filtered


def _single_matching_live_claim(
    *,
    agent: str,
    project: str,
    scope: str,
) -> coordination_claims.ClaimRecord:
    """Return one live claim for a bounded lane or fail loud."""

    claims = _iter_matching_live_claims(agent=agent, project=project, scope=scope)
    if not claims:
        raise ValueError(f"No live claim found for {agent} → {project}:{scope}")
    if len(claims) > 1:
        raise ValueError(f"Multiple live claims found for {agent} → {project}:{scope}")
    return claims[0]


def _claim_status(claim: coordination_claims.ClaimRecord) -> str:
    """Return the current persisted status string for one claim."""

    return claim.status


def _recovery_action_for_claim(claim: coordination_claims.ClaimRecord) -> str:
    """Return the operator action implied by one claim's lifecycle state."""

    health_status = coordination_claims.claim_runtime_status(claim)
    if claim.status == "handoff":
        return "resume_or_finish_handoff"
    if health_status == "stale":
        return "resume_or_abandon_or_prune"
    return "continue"


def _worktree_is_clean(worktree_path: str) -> tuple[bool, str]:
    """Return whether one worktree has a clean git status."""

    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=worktree_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return (not result.stdout.strip(), result.stdout.strip())


def _claim_record_any_status(
    *,
    agent: str,
    project: str,
    scope: str,
) -> tuple[coordination_claims.ClaimRecord, dict[str, Any], Path]:
    """Load one claim regardless of current lifecycle status."""

    claim_file = _claim_path(agent, project, scope)
    payload = _load_claim_payload(agent, project, scope)
    if payload is None:
        raise ValueError(f"Claim file missing for {agent} → {project}:{scope}")
    claim = coordination_claims.normalize_claim(payload, source_file=str(claim_file))
    if claim is None:
        raise ValueError(f"Claim file invalid for {agent} → {project}:{scope}")
    return claim, payload, claim_file


def _resolve_claim_repo_root(claim: coordination_claims.ClaimRecord) -> Path:
    """Return the canonical repo root for one claim."""

    if claim.repo_root:
        return resolve_canonical_repo_root(Path(claim.repo_root).expanduser())
    if claim.worktree_path:
        return resolve_canonical_repo_root(Path(claim.worktree_path).expanduser())
    raise ValueError(f"Claim {claim.scope} is missing repo_root and worktree_path")


def _cwd_inside(path: Path) -> bool:
    """Return whether the current shell cwd is inside the target path."""

    try:
        current_dir = Path(os.getcwd()).resolve()
    except OSError:
        return False
    try:
        current_dir.relative_to(path.resolve())
        return True
    except ValueError:
        return False


def _branch_exists(repo_root: Path, branch: str) -> bool:
    """Return whether one local branch ref exists."""

    result = subprocess.run(
        ["git", "show-ref", "--verify", f"refs/heads/{branch}"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _remove_worktree_path(repo_root: Path, worktree_path: Path) -> str:
    """Remove one worktree path from a safe root-anchored control session."""

    if not worktree_path.exists():
        return "already_missing"
    if _cwd_inside(worktree_path):
        raise ValueError(
            "Cannot close a session from a shell whose cwd is inside the target worktree. "
            "Run closeout from the canonical repo root session instead."
        )
    result = subprocess.run(
        ["git", "worktree", "remove", str(worktree_path)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return "removed"


def _delete_branch(repo_root: Path, branch: str | None) -> str:
    """Delete one local branch after worktree cleanup."""

    if not branch:
        return "not_requested"
    if not _branch_exists(repo_root, branch):
        return "already_missing"
    result = subprocess.run(
        ["git", "branch", "-D", branch],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return "deleted"


def start_session(
    *,
    agent: str,
    project: str,
    scope: str,
    intent: str,
    repo_root: str,
    worktree_path: str,
    branch: str,
    broader_goal: str,
    current_phase: str,
    plan_ref: str | None = None,
    session_id: str | None = None,
    session_name: str | None = None,
    intended_next_phases: list[str] | None = None,
    depends_on_repos: list[str] | None = None,
    requires_shared_infra_changes: bool = False,
    stop_conditions: list[str] | None = None,
    notes: str | None = None,
    tracker_dir: Path = session_contracts.DEFAULT_SESSION_TRACKERS_DIR,
    allow_unplanned: bool = False,
    allow_parallel: bool = False,
) -> dict[str, Any]:
    """Create or refresh the session contract plus linked tracker artifact."""

    resolved_session_id = coordination_claims.resolve_session_id(agent, session_id)
    if not resolved_session_id:
        raise ValueError(
            "Unable to resolve a session ID. Pass --session-id explicitly or run from a supported tool runtime."
        )

    contract = session_contracts.SessionContract.build(
        agent=agent,
        project=project,
        scope=scope,
        intent=intent,
        plan_ref=plan_ref,
        repo_root=repo_root,
        worktree_path=worktree_path,
        branch=branch,
        session_id=resolved_session_id,
        broader_goal=broader_goal,
        session_name=session_name,
        allow_unplanned=allow_unplanned,
    )
    matching_lane_claims = [
        claim
        for claim in _iter_matching_live_claims(project=project, scope=scope)
        if claim.plan_ref == contract.plan_ref and claim.branch != branch
    ]
    if matching_lane_claims and not allow_parallel:
        branches = ", ".join(sorted({claim.branch or "-" for claim in matching_lane_claims}))
        raise ValueError(
            "A live lane already exists for the same project + plan_ref + scope "
            f"on branch(es): {branches}. Use explicit parallelism if this is intentional."
        )
    tracker_path = session_contracts.session_tracker_path(contract, tracker_dir=tracker_dir)
    contract = contract.with_tracker_path(str(tracker_path))
    tracker = session_contracts.build_session_tracker(
        contract=contract,
        current_phase=current_phase,
        intended_next_phases=intended_next_phases,
        depends_on_repos=depends_on_repos,
        requires_shared_infra_changes=requires_shared_infra_changes,
        stop_conditions=stop_conditions,
        notes=notes,
    )
    tracker_path.parent.mkdir(parents=True, exist_ok=True)
    tracker_path.write_text(
        yaml.safe_dump(tracker.to_dict(), default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    try:
        action = _upsert_session_claim(
            agent=agent,
            project=project,
            scope=scope,
            intent=intent,
            plan_ref=plan_ref,
            repo_root=repo_root,
            worktree_path=worktree_path,
            branch=branch,
            session_id=resolved_session_id,
            broader_goal=contract.broader_goal,
            session_name=contract.session_name,
            tracker_path=str(tracker_path),
        )
    except Exception:
        tracker_path.unlink(missing_ok=True)
        raise
    return {
        "action": action,
        "session_id": resolved_session_id,
        "session_name": contract.session_name,
        "broader_goal": contract.broader_goal,
        "tracker_path": str(tracker_path),
        "plan_ref": contract.plan_ref,
    }


def heartbeat_session(
    *,
    agent: str,
    project: str,
    session_id: str | None = None,
    scope: str | None = None,
    branch: str | None = None,
    current_phase: str | None = None,
    tracker_dir: Path = session_contracts.DEFAULT_SESSION_TRACKERS_DIR,
) -> dict[str, Any]:
    """Refresh claim heartbeat state and the linked tracker timestamp."""

    updated_count, updated_scopes, resolved_session_id, heartbeat_at = coordination_claims.heartbeat_claims(
        agent=agent,
        project=project,
        session_id=session_id,
        scope=scope,
        branch=branch,
    )
    tracker_paths_updated: list[str] = []
    for claim in _iter_matching_live_claims(
        agent=agent,
        project=project,
        scope=scope,
        branch=branch,
    ):
        if claim.session_id != resolved_session_id:
            continue
        if not claim.tracker_path:
            continue
        path = Path(claim.tracker_path).expanduser()
        if not path.exists():
            continue
        session_contracts.update_session_tracker(
            path,
            current_phase=current_phase,
            updated_at=heartbeat_at,
        )
        tracker_paths_updated.append(str(path))
    return {
        "updated_count": updated_count,
        "updated_scopes": updated_scopes,
        "session_id": resolved_session_id,
        "heartbeat_at": heartbeat_at,
        "tracker_paths_updated": sorted(tracker_paths_updated),
    }


def status_sessions(
    *,
    project: str | None = None,
    agent: str | None = None,
    scope: str | None = None,
    branch: str | None = None,
) -> dict[str, Any]:
    """Return live session summaries derived from claims plus linked trackers."""

    sessions: list[dict[str, Any]] = []
    for claim in _iter_matching_live_claims(
        agent=agent,
        project=project,
        scope=scope,
        branch=branch,
    ):
        tracker_payload: dict[str, Any] | None = None
        if claim.tracker_path:
            path = Path(claim.tracker_path).expanduser()
            if path.exists():
                tracker_payload = session_contracts.read_session_tracker(path)
        tracker_section = tracker_payload.get("tracker") if isinstance(tracker_payload, dict) else {}
        timestamps = tracker_payload.get("timestamps") if isinstance(tracker_payload, dict) else {}
        sessions.append(
            {
                "project": claim.primary_project(),
                "scope": claim.scope,
                "agent": claim.agent,
                "branch": claim.branch,
                "worktree_path": claim.worktree_path,
                "plan_ref": claim.plan_ref,
                "session_id": claim.session_id,
                "session_name": claim.session_name,
                "broader_goal": claim.broader_goal,
                "tracker_path": claim.tracker_path,
                "claim_status": claim.status,
                "health_status": coordination_claims.claim_runtime_status(claim),
                "current_phase": tracker_section.get("current_phase") if isinstance(tracker_section, dict) else None,
                "intended_next_phases": tracker_section.get("intended_next_phases") if isinstance(tracker_section, dict) else [],
                "depends_on_repos": tracker_section.get("depends_on_repos") if isinstance(tracker_section, dict) else [],
                "requires_shared_infra_changes": tracker_section.get("requires_shared_infra_changes") if isinstance(tracker_section, dict) else False,
                "stop_conditions": tracker_section.get("stop_conditions") if isinstance(tracker_section, dict) else [],
                "notes": tracker_section.get("notes") if isinstance(tracker_section, dict) else None,
                "tracker_updated_at": timestamps.get("updated_at") if isinstance(timestamps, dict) else None,
                "recovery_action": _recovery_action_for_claim(claim),
            }
        )
    return {
        "session_count": len(sessions),
        "sessions": sessions,
    }


def finish_session(
    *,
    agent: str,
    project: str,
    scope: str,
    worktree_path: str,
    note: str | None = None,
    release_claim: bool = False,
    allow_dirty_handoff: bool = False,
) -> dict[str, Any]:
    """Close out one session or fail loud if the worktree state is unsafe."""

    claim = _single_matching_live_claim(agent=agent, project=project, scope=scope)

    clean, dirty_details = _worktree_is_clean(worktree_path)
    updated_at = datetime.now(timezone.utc).isoformat()

    tracker_path_text = claim.tracker_path
    if tracker_path_text:
        path = Path(tracker_path_text).expanduser()
        if path.exists():
            finish_note = note or ("completed and cleaned up" if clean else "handoff required")
            session_contracts.update_session_tracker(
                path,
                current_phase="completed" if clean else "handoff required",
                notes=finish_note,
                updated_at=updated_at,
            )

    claim_file = _claim_path(agent, project, scope)
    payload = _load_claim_payload(agent, project, scope)
    if payload is None:
        raise ValueError(f"Claim file missing for {agent} → {project}:{scope}")

    if not clean:
        if not allow_dirty_handoff:
            raise ValueError(
                "Worktree is dirty; commit or stash before session-finish, "
                "or pass --allow-dirty-handoff with a handoff note."
            )
        payload["status"] = "handoff"
        payload["updated_at"] = updated_at
        payload["notes"] = note or "handoff required because the worktree still has uncommitted changes"
        _write_claim_payload(claim_file, payload)
        return {
            "action": "handoff",
            "clean": False,
            "dirty_details": dirty_details,
            "tracker_path": tracker_path_text,
        }

    doc_authority.assert_no_unresolved_owned_obligations(claim)

    if release_claim:
        coordination_claims.release_claim(agent, project, scope)
        return {
            "action": "released",
            "clean": True,
            "tracker_path": tracker_path_text,
        }

    payload["status"] = "completed"
    payload["updated_at"] = updated_at
    payload["notes"] = note or "session finished cleanly"
    _write_claim_payload(claim_file, payload)
    return {
        "action": "completed",
        "clean": True,
        "tracker_path": tracker_path_text,
    }


def close_session(
    *,
    agent: str,
    project: str,
    scope: str,
    worktree_path: str | None = None,
    branch: str | None = None,
    note: str | None = None,
    delete_branch: bool = True,
) -> dict[str, Any]:
    """Finish, clean up, and release one claimed lane as a single sanctioned flow.

    This is the canonical closeout path for claimed worktrees. It is intentionally
    idempotent around already-missing worktree and branch state so that rerunning
    a partially completed closeout can still release the claim cleanly.
    """

    claim, payload, claim_file = _claim_record_any_status(agent=agent, project=project, scope=scope)
    resolved_worktree_path = Path(
        worktree_path or claim.worktree_path or ""
    ).expanduser()
    resolved_branch = branch or claim.branch
    repo_root = _resolve_claim_repo_root(claim)
    updated_at = datetime.now(timezone.utc).isoformat()

    if claim.write_paths:
        doc_authority.assert_no_unresolved_owned_obligations(claim)

    if resolved_worktree_path and resolved_worktree_path.exists():
        clean, dirty_details = _worktree_is_clean(str(resolved_worktree_path))
        if not clean:
            raise ValueError(
                "Worktree is dirty; commit or stash before session-close. "
                f"Uncommitted state:\n{dirty_details}"
            )

    payload["status"] = "closing"
    payload["updated_at"] = updated_at
    payload["notes"] = note or "closing claimed lane via canonical session-close flow"
    _write_claim_payload(claim_file, payload)

    tracker_path_text = claim.tracker_path
    if tracker_path_text:
        tracker_path = Path(tracker_path_text).expanduser()
        if tracker_path.exists():
            session_contracts.update_session_tracker(
                tracker_path,
                current_phase="closing",
                notes=payload["notes"],
                updated_at=updated_at,
            )

    worktree_action = "not_requested"
    branch_action = "kept"
    if worktree_path or claim.worktree_path:
        worktree_action = _remove_worktree_path(repo_root, resolved_worktree_path)
    if delete_branch:
        branch_action = _delete_branch(repo_root, resolved_branch)

    coordination_claims.release_claim(agent, project, scope)

    if tracker_path_text:
        tracker_path = Path(tracker_path_text).expanduser()
        if tracker_path.exists():
            session_contracts.update_session_tracker(
                tracker_path,
                current_phase="closed",
                notes=note or "session closed and claimed worktree cleaned up",
                updated_at=datetime.now(timezone.utc).isoformat(),
            )

    return {
        "action": "closed",
        "worktree_action": worktree_action,
        "branch_action": branch_action,
        "released": True,
        "tracker_path": tracker_path_text,
    }


def resume_session(
    *,
    agent: str,
    project: str,
    scope: str,
    worktree_path: str,
    branch: str,
    current_phase: str,
    session_id: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Reattach a new runtime session to an existing plan-bound lane."""

    claim = _single_matching_live_claim(agent=agent, project=project, scope=scope)
    if not claim.plan_ref:
        raise ValueError("Cannot resume a lane with no plan_ref")
    if claim.branch and claim.branch != branch:
        raise ValueError(f"Claim branch is {claim.branch}, not {branch}")
    if claim.worktree_path and claim.worktree_path != worktree_path:
        raise ValueError(f"Claim worktree is {claim.worktree_path}, not {worktree_path}")

    resolved_session_id = coordination_claims.resolve_session_id(agent, session_id)
    if not resolved_session_id:
        raise ValueError("Unable to resolve a session ID for session-resume.")

    updated_at = datetime.now(timezone.utc).isoformat()
    claim_file = _claim_path(agent, project, scope)
    payload = _load_claim_payload(agent, project, scope)
    if payload is None:
        raise ValueError(f"Claim file missing for {agent} → {project}:{scope}")

    payload["status"] = "active"
    payload["session_id"] = resolved_session_id
    payload["heartbeat_at"] = updated_at
    payload["updated_at"] = updated_at
    payload["notes"] = note or "session resumed with a fresh runtime attachment"
    _write_claim_payload(claim_file, payload)

    tracker_path_text = claim.tracker_path
    if tracker_path_text:
        path = Path(tracker_path_text).expanduser()
        if path.exists():
            session_contracts.update_session_tracker(
                path,
                current_phase=current_phase,
                notes=payload["notes"],
                updated_at=updated_at,
            )

    return {
        "action": "resumed",
        "session_id": resolved_session_id,
        "tracker_path": tracker_path_text,
        "plan_ref": claim.plan_ref,
    }


def handoff_session(
    *,
    agent: str,
    project: str,
    scope: str,
    note: str,
    current_phase: str = "handoff required",
) -> dict[str, Any]:
    """Mark one live lane as intentionally handed off."""

    claim = _single_matching_live_claim(agent=agent, project=project, scope=scope)
    updated_at = datetime.now(timezone.utc).isoformat()
    claim_file = _claim_path(agent, project, scope)
    payload = _load_claim_payload(agent, project, scope)
    if payload is None:
        raise ValueError(f"Claim file missing for {agent} → {project}:{scope}")

    payload["status"] = "handoff"
    payload["updated_at"] = updated_at
    payload["notes"] = note.strip()
    _write_claim_payload(claim_file, payload)

    tracker_path_text = claim.tracker_path
    if tracker_path_text:
        path = Path(tracker_path_text).expanduser()
        if path.exists():
            session_contracts.update_session_tracker(
                path,
                current_phase=current_phase,
                notes=payload["notes"],
                updated_at=updated_at,
            )

    return {
        "action": "handoff",
        "tracker_path": tracker_path_text,
    }


def abandon_session(
    *,
    agent: str,
    project: str,
    scope: str,
    note: str,
) -> dict[str, Any]:
    """Mark one live lane as explicitly abandoned."""

    claim = _single_matching_live_claim(agent=agent, project=project, scope=scope)
    updated_at = datetime.now(timezone.utc).isoformat()
    claim_file = _claim_path(agent, project, scope)
    payload = _load_claim_payload(agent, project, scope)
    if payload is None:
        raise ValueError(f"Claim file missing for {agent} → {project}:{scope}")

    payload["status"] = "abandoned"
    payload["updated_at"] = updated_at
    payload["notes"] = note.strip()
    _write_claim_payload(claim_file, payload)

    tracker_path_text = claim.tracker_path
    if tracker_path_text:
        path = Path(tracker_path_text).expanduser()
        if path.exists():
            session_contracts.update_session_tracker(
                path,
                current_phase="abandoned",
                notes=payload["notes"],
                updated_at=updated_at,
            )

    return {
        "action": "abandoned",
        "tracker_path": tracker_path_text,
    }
