"""Observability event/config adapters.

This module is a boundary layer around ``llm_client.io_log`` while we
incrementally migrate internals out of that monolith.
"""

from __future__ import annotations

from typing import Any

from llm_client import io_log as _io_log

ActiveFeatureProfile = _io_log.ActiveFeatureProfile


def configure_logging(
    *,
    enabled: bool | None = None,
    data_root: str | Any | None = None,
    project: str | None = None,
    db_path: str | Any | None = None,
) -> None:
    _io_log.configure(enabled=enabled, data_root=data_root, project=project, db_path=db_path)


def configure_experiment_enforcement(
    *,
    mode: str | None = None,
    task_patterns: list[str] | str | None = None,
) -> None:
    _io_log.configure_experiment_enforcement(mode=mode, task_patterns=task_patterns)


def configure_feature_profile(
    *,
    profile: str | dict[str, Any] | None = None,
    mode: str | None = None,
    task_patterns: list[str] | str | None = None,
) -> None:
    _io_log.configure_feature_profile(profile=profile, mode=mode, task_patterns=task_patterns)


def configure_agent_spec_enforcement(
    *,
    mode: str | None = None,
    task_patterns: list[str] | str | None = None,
) -> None:
    _io_log.configure_agent_spec_enforcement(mode=mode, task_patterns=task_patterns)


def activate_feature_profile(profile: str | dict[str, Any]) -> ActiveFeatureProfile:
    return _io_log.activate_feature_profile(profile)


def get_active_feature_profile() -> dict[str, Any] | None:
    return _io_log.get_active_feature_profile()


def get_active_experiment_run_id() -> str | None:
    return _io_log.get_active_experiment_run_id()


def enforce_agent_spec(
    *,
    task: str | None = None,
    has_agent_spec: bool,
    allow_missing: bool = False,
    missing_reason: str | None = None,
    caller: str = "llm_client.observability.events",
) -> None:
    _io_log.enforce_agent_spec(
        task=task,
        has_agent_spec=has_agent_spec,
        allow_missing=allow_missing,
        missing_reason=missing_reason,
        caller=caller,
    )


def log_embedding(**kwargs: Any) -> None:
    _io_log.log_embedding(**kwargs)


def log_foundation_event(
    *,
    event: dict[str, Any],
    caller: str = "foundation",
    task: str | None = None,
    trace_id: str | None = None,
) -> None:
    _io_log.log_foundation_event(event=event, caller=caller, task=task, trace_id=trace_id)
