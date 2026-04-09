"""Session bootstrap contract and tracker helpers.

This module defines the split between compact claim metadata that belongs on
the canonical coordination claim and richer tracker-only execution context that
should live in a linked per-session artifact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


DEFAULT_SESSION_TRACKERS_DIR = Path.home() / ".claude" / "coordination" / "sessions"
SESSION_TRACKER_SCHEMA_VERSION = 1
UNPLANNED_PLAN_REF = "UNPLANNED"

CLAIM_FIELD_NAMES = (
    "agent",
    "project",
    "scope",
    "intent",
    "plan_ref",
    "repo_root",
    "worktree_path",
    "branch",
    "session_id",
    "session_name",
    "broader_goal",
    "tracker_path",
)

TRACKER_ONLY_FIELD_NAMES = (
    "current_phase",
    "intended_next_phases",
    "depends_on_repos",
    "requires_shared_infra_changes",
    "stop_conditions",
    "notes",
)


def _require_text(value: str, *, field_name: str) -> str:
    """Return one stripped text field or raise a clear contract error."""

    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def normalize_plan_ref(plan_ref: str | None, *, allow_unplanned: bool = False) -> str:
    """Return a normalized plan marker or fail loud if none was declared."""

    if isinstance(plan_ref, str) and plan_ref.strip():
        return plan_ref.strip()
    if allow_unplanned:
        return UNPLANNED_PLAN_REF
    raise ValueError(
        "plan_ref is required for live sessions. "
        "Pass a real numbered plan or explicitly allow unplanned work."
    )


def _clean_string_list(items: list[str] | None) -> list[str]:
    """Return a stable deduplicated string list for tracker serialization."""

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items or []:
        text = item.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned


def derive_session_name(broader_goal: str) -> str:
    """Derive the canonical session name from the broader goal text."""

    tokens = re.findall(r"[a-z0-9]+", broader_goal.lower())
    if not tokens:
        raise ValueError("broader_goal must contain at least one alphanumeric token")
    return "-".join(tokens)


def validate_session_name(*, session_name: str, broader_goal: str) -> str:
    """Require the session name to match the broader-goal-derived canonical slug."""

    expected = derive_session_name(broader_goal)
    normalized = derive_session_name(session_name)
    if normalized != expected:
        raise ValueError(
            "session_name must match the broader-goal-derived canonical name "
            f"'{expected}', not local-task wording"
        )
    return expected


@dataclass(frozen=True)
class SessionContract:
    """Compact session contract fields that belong on the canonical claim."""

    agent: str
    project: str
    scope: str
    intent: str
    plan_ref: str | None
    repo_root: str
    worktree_path: str
    branch: str
    session_id: str
    session_name: str
    broader_goal: str
    tracker_path: str | None = None

    @classmethod
    def build(
        cls,
        *,
        agent: str,
        project: str,
        scope: str,
        intent: str,
        repo_root: str,
        worktree_path: str,
        branch: str,
        session_id: str,
        broader_goal: str,
        plan_ref: str | None = None,
        session_name: str | None = None,
        tracker_path: str | None = None,
        allow_unplanned: bool = False,
    ) -> "SessionContract":
        """Build a validated session contract from bootstrap inputs."""

        broader_goal_text = _require_text(broader_goal, field_name="broader_goal")
        derived_session_name = derive_session_name(broader_goal_text)
        if session_name is None:
            session_name_text = derived_session_name
        else:
            session_name_text = validate_session_name(
                session_name=session_name,
                broader_goal=broader_goal_text,
            )

        return cls(
            agent=_require_text(agent, field_name="agent"),
            project=_require_text(project, field_name="project"),
            scope=_require_text(scope, field_name="scope"),
            intent=_require_text(intent, field_name="intent"),
            plan_ref=normalize_plan_ref(plan_ref, allow_unplanned=allow_unplanned),
            repo_root=_require_text(repo_root, field_name="repo_root"),
            worktree_path=_require_text(worktree_path, field_name="worktree_path"),
            branch=_require_text(branch, field_name="branch"),
            session_id=_require_text(session_id, field_name="session_id"),
            session_name=session_name_text,
            broader_goal=broader_goal_text,
            tracker_path=tracker_path.strip() if isinstance(tracker_path, str) and tracker_path.strip() else None,
        )

    def with_tracker_path(self, tracker_path: str) -> "SessionContract":
        """Return the same contract with a concrete tracker path attached."""

        return replace(self, tracker_path=_require_text(tracker_path, field_name="tracker_path"))

    def claim_fields(self) -> dict[str, str]:
        """Return only the claim-critical contract fields."""

        return {
            "agent": self.agent,
            "project": self.project,
            "scope": self.scope,
            "intent": self.intent,
            "plan_ref": self.plan_ref or "",
            "repo_root": self.repo_root,
            "worktree_path": self.worktree_path,
            "branch": self.branch,
            "session_id": self.session_id,
            "session_name": self.session_name,
            "broader_goal": self.broader_goal,
            "tracker_path": self.tracker_path or "",
        }


@dataclass(frozen=True)
class SessionTrackerRecord:
    """Human-readable tracker artifact linked from one session contract."""

    contract: SessionContract
    current_phase: str
    intended_next_phases: list[str]
    depends_on_repos: list[str]
    requires_shared_infra_changes: bool
    stop_conditions: list[str]
    notes: str | None
    created_at: str
    updated_at: str
    schema_version: int = SESSION_TRACKER_SCHEMA_VERSION

    def tracker_fields(self) -> dict[str, Any]:
        """Return only the tracker-only execution fields."""

        return {
            "current_phase": self.current_phase,
            "intended_next_phases": self.intended_next_phases,
            "depends_on_repos": self.depends_on_repos,
            "requires_shared_infra_changes": self.requires_shared_infra_changes,
            "stop_conditions": self.stop_conditions,
            "notes": self.notes or "",
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the nested contract/tracker representation for YAML output."""

        return {
            "schema_version": self.schema_version,
            "claim": self.contract.claim_fields(),
            "tracker": self.tracker_fields(),
            "timestamps": {
                "created_at": self.created_at,
                "updated_at": self.updated_at,
            },
        }


def session_contract_schema() -> dict[str, list[str]]:
    """Return the formal split between claim and tracker fields."""

    return {
        "claim_fields": list(CLAIM_FIELD_NAMES),
        "tracker_only_fields": list(TRACKER_ONLY_FIELD_NAMES),
    }


def build_session_tracker(
    *,
    contract: SessionContract,
    current_phase: str,
    intended_next_phases: list[str] | None = None,
    depends_on_repos: list[str] | None = None,
    requires_shared_infra_changes: bool = False,
    stop_conditions: list[str] | None = None,
    notes: str | None = None,
    now: datetime | None = None,
) -> SessionTrackerRecord:
    """Build the linked tracker artifact for one active session contract."""

    timestamp = (now or datetime.now(timezone.utc)).isoformat()
    return SessionTrackerRecord(
        contract=contract,
        current_phase=_require_text(current_phase, field_name="current_phase"),
        intended_next_phases=_clean_string_list(intended_next_phases),
        depends_on_repos=_clean_string_list(depends_on_repos),
        requires_shared_infra_changes=requires_shared_infra_changes,
        stop_conditions=_clean_string_list(stop_conditions),
        notes=notes.strip() if isinstance(notes, str) and notes.strip() else None,
        created_at=timestamp,
        updated_at=timestamp,
    )


def session_tracker_path(
    contract: SessionContract,
    *,
    tracker_dir: Path = DEFAULT_SESSION_TRACKERS_DIR,
) -> Path:
    """Return the canonical tracker path for one session contract."""

    safe_session_id = re.sub(r"[^a-zA-Z0-9._-]+", "-", contract.session_id)
    filename = (
        f"{contract.agent}__{contract.project}__{safe_session_id}"
        f"__{contract.session_name}.yaml"
    )
    return tracker_dir / contract.project / filename


def write_session_tracker(
    record: SessionTrackerRecord,
    *,
    tracker_dir: Path = DEFAULT_SESSION_TRACKERS_DIR,
) -> Path:
    """Persist one session tracker artifact and return its concrete path."""

    path = session_tracker_path(record.contract, tracker_dir=tracker_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(record.to_dict(), default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    return path


def read_session_tracker(path: Path) -> dict[str, Any]:
    """Load one tracker artifact and fail loud if the structure is invalid."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Session tracker at {path} must be a YAML mapping")
    return raw


def update_session_tracker(
    path: Path,
    *,
    current_phase: str | None = None,
    intended_next_phases: list[str] | None = None,
    depends_on_repos: list[str] | None = None,
    requires_shared_infra_changes: bool | None = None,
    stop_conditions: list[str] | None = None,
    notes: str | None = None,
    updated_at: str | None = None,
) -> dict[str, Any]:
    """Update one existing tracker artifact in place and return the payload."""

    payload = read_session_tracker(path)
    tracker = payload.get("tracker")
    timestamps = payload.get("timestamps")
    if not isinstance(tracker, dict) or not isinstance(timestamps, dict):
        raise ValueError(f"Session tracker at {path} is missing tracker/timestamps sections")

    if current_phase is not None:
        tracker["current_phase"] = _require_text(current_phase, field_name="current_phase")
    if intended_next_phases is not None:
        tracker["intended_next_phases"] = _clean_string_list(intended_next_phases)
    if depends_on_repos is not None:
        tracker["depends_on_repos"] = _clean_string_list(depends_on_repos)
    if requires_shared_infra_changes is not None:
        tracker["requires_shared_infra_changes"] = requires_shared_infra_changes
    if stop_conditions is not None:
        tracker["stop_conditions"] = _clean_string_list(stop_conditions)
    if notes is not None:
        tracker["notes"] = notes.strip()

    timestamps["updated_at"] = updated_at or datetime.now(timezone.utc).isoformat()
    path.write_text(
        yaml.safe_dump(payload, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    return payload
