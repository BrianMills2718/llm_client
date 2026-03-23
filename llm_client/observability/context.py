"""Experiment context and benchmark guardrail helpers.

This module owns the execution-context state and enforcement rules that were
previously embedded in ``llm_client.io_log``. The behavior is still reachable
through the historical ``io_log`` surface for compatibility, but the actual
responsibility lives here because it is observability policy, not call-log
storage.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import re
from typing import Any, Literal

logger = logging.getLogger(__name__)

_active_experiment_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "llm_client_active_experiment_run_id",
    default=None,
)
_active_feature_profile: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "llm_client_active_feature_profile",
    default=None,
)

_EXPERIMENT_ENFORCEMENT_ENV = "LLM_CLIENT_EXPERIMENT_ENFORCEMENT"
_EXPERIMENT_TASK_PATTERNS_ENV = "LLM_CLIENT_EXPERIMENT_TASK_PATTERNS"
_DEFAULT_EXPERIMENT_TASK_PATTERNS: tuple[str, ...] = ("benchmark", "eval", "experiment")
_FEATURE_PROFILE_ENV = "LLM_CLIENT_FEATURE_PROFILE"
_FEATURE_PROFILE_ENFORCEMENT_ENV = "LLM_CLIENT_FEATURE_PROFILE_ENFORCEMENT"
_FEATURE_PROFILE_TASK_PATTERNS_ENV = "LLM_CLIENT_FEATURE_PROFILE_TASK_PATTERNS"
_DEFAULT_FEATURE_PROFILE_TASK_PATTERNS: tuple[str, ...] = ("benchmark", "eval", "experiment")
_AGENT_SPEC_ENFORCEMENT_ENV = "LLM_CLIENT_AGENT_SPEC_ENFORCEMENT"
_AGENT_SPEC_TASK_PATTERNS_ENV = "LLM_CLIENT_AGENT_SPEC_TASK_PATTERNS"
_DEFAULT_AGENT_SPEC_TASK_PATTERNS: tuple[str, ...] = ("benchmark", "eval", "experiment")
_BUILTIN_FEATURE_PROFILES: dict[str, dict[str, Any]] = {
    "baseline": {
        "name": "baseline",
        "features": {
            "observability_tags": True,
        },
    },
    "benchmark_strict": {
        "name": "benchmark_strict",
        "features": {
            "experiment_context": True,
            "provenance": True,
            "tool_reasoning": True,
        },
    },
}


def get_active_experiment_run_id() -> str | None:
    """Return the active experiment run id for the current execution context."""

    return _active_experiment_run_id.get()


def _normalize_feature_profile(profile: str | dict[str, Any]) -> dict[str, Any]:
    """Normalize feature-profile declarations into a stable dict form."""

    if isinstance(profile, str):
        value = profile.strip()
        if not value:
            raise ValueError("feature profile name cannot be empty")
        built_in = _BUILTIN_FEATURE_PROFILES.get(value.lower())
        if built_in is not None:
            return dict(built_in)
        if value.startswith("{"):
            parsed = json.loads(value)
            if not isinstance(parsed, dict):
                raise ValueError("feature profile JSON must decode to an object")
            profile_dict = dict(parsed)
        else:
            profile_dict = {"name": value, "features": {}}
    elif isinstance(profile, dict):
        profile_dict = dict(profile)
    else:
        raise TypeError("feature profile must be a profile name or dict")

    name = str(profile_dict.get("name", "custom")).strip() or "custom"
    features_raw = profile_dict.get("features")
    if features_raw is None and isinstance(profile_dict.get("require"), dict):
        features_raw = profile_dict.get("require")
    features = dict(features_raw) if isinstance(features_raw, dict) else {}

    normalized = dict(profile_dict)
    normalized["name"] = name
    normalized["features"] = features
    return normalized


def get_active_feature_profile() -> dict[str, Any] | None:
    """Return the active feature profile for the current execution context."""

    active = _active_feature_profile.get()
    if active is not None:
        return active

    raw = os.environ.get(_FEATURE_PROFILE_ENV)
    if not raw:
        return None
    try:
        return _normalize_feature_profile(raw)
    except Exception:
        logger.warning(
            "Invalid %s value; expected profile name or JSON object.",
            _FEATURE_PROFILE_ENV,
            exc_info=True,
        )
        return None


class ActiveFeatureProfile:
    """Bind a feature profile as the active profile in the current context."""

    def __init__(self, profile: str | dict[str, Any]) -> None:
        self.profile = _normalize_feature_profile(profile)
        self._token: contextvars.Token[dict[str, Any] | None] | None = None

    def __enter__(self) -> "ActiveFeatureProfile":
        self._token = _active_feature_profile.set(self.profile)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> Literal[False]:
        if self._token is not None:
            _active_feature_profile.reset(self._token)
            self._token = None
        return False

    async def __aenter__(self) -> "ActiveFeatureProfile":
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> Literal[False]:
        return self.__exit__(exc_type, exc, tb)


def activate_feature_profile(profile: str | dict[str, Any]) -> ActiveFeatureProfile:
    """Return a context manager that activates one feature profile."""

    return ActiveFeatureProfile(profile)


class ActiveExperimentRun:
    """Bind an existing run id as the active experiment context."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._token: contextvars.Token[str | None] | None = None

    def __enter__(self) -> "ActiveExperimentRun":
        if get_active_experiment_run_id() != self.run_id:
            self._token = _active_experiment_run_id.set(self.run_id)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> Literal[False]:
        if self._token is not None:
            _active_experiment_run_id.reset(self._token)
            self._token = None
        return False

    async def __aenter__(self) -> "ActiveExperimentRun":
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> Literal[False]:
        return self.__exit__(exc_type, exc, tb)


def activate_experiment_run(run_id: str) -> ActiveExperimentRun:
    """Return a context manager that binds one active experiment run id."""

    return ActiveExperimentRun(run_id)


def configure_experiment_enforcement(
    *,
    mode: str | None = None,
    task_patterns: list[str] | str | None = None,
) -> None:
    """Configure experiment-context enforcement via environment variables."""

    if mode is not None:
        os.environ[_EXPERIMENT_ENFORCEMENT_ENV] = str(mode).strip().lower()

    if task_patterns is not None:
        if isinstance(task_patterns, str):
            patterns_str = task_patterns
        else:
            patterns_str = ",".join(str(p).strip() for p in task_patterns if str(p).strip())
        os.environ[_EXPERIMENT_TASK_PATTERNS_ENV] = patterns_str


def configure_feature_profile(
    *,
    profile: str | dict[str, Any] | None = None,
    mode: str | None = None,
    task_patterns: list[str] | str | None = None,
) -> None:
    """Configure feature-profile policy via environment variables."""

    if profile is not None:
        normalized = _normalize_feature_profile(profile)
        os.environ[_FEATURE_PROFILE_ENV] = json.dumps(normalized, sort_keys=True)
    if mode is not None:
        os.environ[_FEATURE_PROFILE_ENFORCEMENT_ENV] = str(mode).strip().lower()
    if task_patterns is not None:
        if isinstance(task_patterns, str):
            patterns_str = task_patterns
        else:
            patterns_str = ",".join(str(p).strip() for p in task_patterns if str(p).strip())
        os.environ[_FEATURE_PROFILE_TASK_PATTERNS_ENV] = patterns_str


def configure_agent_spec_enforcement(
    *,
    mode: str | None = None,
    task_patterns: list[str] | str | None = None,
) -> None:
    """Configure AgentSpec enforcement via environment variables."""

    if mode is not None:
        os.environ[_AGENT_SPEC_ENFORCEMENT_ENV] = str(mode).strip().lower()

    if task_patterns is not None:
        if isinstance(task_patterns, str):
            patterns_str = task_patterns
        else:
            patterns_str = ",".join(str(p).strip() for p in task_patterns if str(p).strip())
        os.environ[_AGENT_SPEC_TASK_PATTERNS_ENV] = patterns_str


def _load_experiment_task_patterns() -> list[str]:
    raw = os.environ.get(_EXPERIMENT_TASK_PATTERNS_ENV, "")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if parts:
        return parts
    return list(_DEFAULT_EXPERIMENT_TASK_PATTERNS)


def _load_feature_profile_task_patterns() -> list[str]:
    raw = os.environ.get(_FEATURE_PROFILE_TASK_PATTERNS_ENV, "")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if parts:
        return parts
    return list(_DEFAULT_FEATURE_PROFILE_TASK_PATTERNS)


def _load_agent_spec_task_patterns() -> list[str]:
    raw = os.environ.get(_AGENT_SPEC_TASK_PATTERNS_ENV, "")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if parts:
        return parts
    return list(_DEFAULT_AGENT_SPEC_TASK_PATTERNS)


def _pattern_matches_task(task: str, pattern: str) -> bool:
    try:
        return re.search(pattern, task, flags=re.IGNORECASE) is not None
    except re.error:
        return pattern.lower() in task.lower()


def enforce_experiment_context(task: str | None, *, caller: str = "llm_client") -> None:
    """Optionally require an active experiment context for benchmark-like tasks."""

    mode = os.environ.get(_EXPERIMENT_ENFORCEMENT_ENV, "warn").strip().lower()
    if mode in {"", "0", "off", "false", "none"}:
        return
    if mode not in {"warn", "error"}:
        logger.warning(
            "Invalid %s=%r; expected off|warn|error. Guard disabled.",
            _EXPERIMENT_ENFORCEMENT_ENV,
            mode,
        )
        return
    if not task:
        return

    patterns = _load_experiment_task_patterns()
    if not any(_pattern_matches_task(task, pattern) for pattern in patterns):
        return

    active_run_id = get_active_experiment_run_id()
    if active_run_id:
        return

    msg = (
        f"{caller}: task={task!r} matched experiment guard patterns {patterns}, "
        "but no active experiment context was found. Wrap this workflow in "
        "llm_client.experiment_run(...) or call llm_client.activate_experiment_run(run_id) "
        "around benchmark/eval calls. Set "
        "LLM_CLIENT_EXPERIMENT_ENFORCEMENT=off to disable."
    )
    if mode == "error":
        raise ValueError(msg)
    logger.warning(msg)


def enforce_feature_profile(task: str | None, *, caller: str = "llm_client") -> None:
    """Optionally require explicit feature profiles for benchmark-like tasks."""

    mode = os.environ.get(_FEATURE_PROFILE_ENFORCEMENT_ENV, "warn").strip().lower()
    if mode in {"", "0", "off", "false", "none"}:
        return
    if mode not in {"warn", "error"}:
        logger.warning(
            "Invalid %s=%r; expected off|warn|error. Guard disabled.",
            _FEATURE_PROFILE_ENFORCEMENT_ENV,
            mode,
        )
        return
    if not task:
        return

    patterns = _load_feature_profile_task_patterns()
    if not any(_pattern_matches_task(task, pattern) for pattern in patterns):
        return

    profile = get_active_feature_profile()
    if not profile:
        msg = (
            f"{caller}: task={task!r} matched feature-profile guard patterns {patterns}, "
            "but no explicit feature profile was declared. Use "
            "llm_client.activate_feature_profile(...) or set "
            "LLM_CLIENT_FEATURE_PROFILE."
        )
        if mode == "error":
            raise ValueError(msg)
        logger.warning(msg)
        return

    features = profile.get("features", {})
    if isinstance(features, dict) and features.get("experiment_context") and not get_active_experiment_run_id():
        msg = (
            f"{caller}: feature profile {profile.get('name', 'unnamed')!r} requires "
            "experiment_context but no active run is bound. Use "
            "llm_client.activate_experiment_run(run_id)."
        )
        if mode == "error":
            raise ValueError(msg)
        logger.warning(msg)


def enforce_agent_spec(
    task: str | None,
    *,
    has_agent_spec: bool,
    allow_missing: bool = False,
    missing_reason: str | None = None,
    caller: str = "llm_client",
) -> None:
    """Optionally require AgentSpec declarations for benchmark-like tasks."""

    mode = os.environ.get(_AGENT_SPEC_ENFORCEMENT_ENV, "error").strip().lower()
    if mode in {"", "0", "off", "false", "none"}:
        return
    if mode not in {"warn", "error"}:
        logger.warning(
            "Invalid %s=%r; expected off|warn|error. Guard disabled.",
            _AGENT_SPEC_ENFORCEMENT_ENV,
            mode,
        )
        return
    if not task:
        return

    patterns = _load_agent_spec_task_patterns()
    if not any(_pattern_matches_task(task, pattern) for pattern in patterns):
        return

    if has_agent_spec:
        return

    reason = (missing_reason or "").strip()
    if allow_missing and reason:
        logger.warning(
            "%s: task=%r proceeding without AgentSpec due to explicit opt-out: %s",
            caller,
            task,
            reason,
        )
        return

    if allow_missing and not reason:
        msg = (
            f"{caller}: task={task!r} requested allow_missing_agent_spec but no "
            "missing_agent_spec_reason was provided. Provide an explicit reason "
            "or declare a valid AgentSpec."
        )
        if mode == "error":
            raise ValueError(msg)
        logger.warning(msg)
        return

    msg = (
        f"{caller}: task={task!r} matched AgentSpec guard patterns {patterns}, "
        "but no AgentSpec was declared. Pass agent_spec=... to start_run() "
        "or explicitly opt out with allow_missing_agent_spec=True and "
        "missing_agent_spec_reason='...'. Set "
        "LLM_CLIENT_AGENT_SPEC_ENFORCEMENT=off to disable."
    )
    if mode == "error":
        raise ValueError(msg)
    logger.warning(msg)
