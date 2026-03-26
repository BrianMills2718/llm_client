"""Observability event/config adapters.

This module is a boundary layer around ``llm_client.io_log`` while we
incrementally migrate internals out of that monolith.
"""

from __future__ import annotations

from typing import Any

import llm_client.io_log as _io_log
from llm_client.observability.context import (
    ActiveFeatureProfile as _ActiveFeatureProfile,
    activate_feature_profile as _activate_feature_profile,
    configure_agent_spec_enforcement as _configure_agent_spec_enforcement,
    configure_experiment_enforcement as _configure_experiment_enforcement,
    configure_feature_profile as _configure_feature_profile,
    enforce_agent_spec as _enforce_agent_spec,
    get_active_experiment_run_id as _get_active_experiment_run_id,
    get_active_feature_profile as _get_active_feature_profile,
)

ActiveFeatureProfile = _ActiveFeatureProfile


def configure_logging(
    *,
    enabled: bool | None = None,
    data_root: str | Any | None = None,
    project: str | None = None,
    db_path: str | Any | None = None,
) -> None:
    """Forward logging configuration to the legacy ``io_log`` backend."""

    _io_log.configure(enabled=enabled, data_root=data_root, project=project, db_path=db_path)


def configure_experiment_enforcement(
    *,
    mode: str | None = None,
    task_patterns: list[str] | str | None = None,
) -> None:
    """Configure benchmark experiment-context guardrails via the extracted context module."""

    _configure_experiment_enforcement(mode=mode, task_patterns=task_patterns)


def configure_feature_profile(
    *,
    profile: str | dict[str, Any] | None = None,
    mode: str | None = None,
    task_patterns: list[str] | str | None = None,
) -> None:
    """Configure feature-profile guardrails without exposing ``io_log`` internals."""

    _configure_feature_profile(profile=profile, mode=mode, task_patterns=task_patterns)


def configure_agent_spec_enforcement(
    *,
    mode: str | None = None,
    task_patterns: list[str] | str | None = None,
) -> None:
    """Configure AgentSpec guardrails through the observability facade."""

    _configure_agent_spec_enforcement(mode=mode, task_patterns=task_patterns)


def activate_feature_profile(profile: str | dict[str, Any]) -> _ActiveFeatureProfile:
    """Return the context manager that binds a feature profile for nested calls."""

    return _activate_feature_profile(profile)


def get_active_feature_profile() -> dict[str, Any] | None:
    """Return the current feature profile, if one is active in this context."""

    return _get_active_feature_profile()


def get_active_experiment_run_id() -> str | None:
    """Return the current bound experiment run id, if any."""

    return _get_active_experiment_run_id()


def enforce_agent_spec(
    *,
    task: str | None = None,
    has_agent_spec: bool,
    allow_missing: bool = False,
    missing_reason: str | None = None,
    caller: str = "llm_client.observability.events",
) -> None:
    """Enforce AgentSpec declarations for guarded benchmark-like tasks."""

    _enforce_agent_spec(
        task=task,
        has_agent_spec=has_agent_spec,
        allow_missing=allow_missing,
        missing_reason=missing_reason,
        caller=caller,
    )


def log_embedding(**kwargs: Any) -> None:
    """Record an embedding event through the legacy storage backend."""

    _io_log.log_embedding(**kwargs)


def log_foundation_event(
    *,
    event: dict[str, Any],
    caller: str = "foundation",
    task: str | None = None,
    trace_id: str | None = None,
) -> None:
    """Record one foundation-level observability event through ``io_log``."""

    _io_log.log_foundation_event(event=event, caller=caller, task=task, trace_id=trace_id)
