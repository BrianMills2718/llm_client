"""Persistent I/O logging for LLM calls and embeddings.

Appends one JSONL record per LLM call to:
    {DATA_ROOT}/{PROJECT}/{PROJECT}_llm_client_data/calls.jsonl
Appends one JSONL record per embedding call to:
    {DATA_ROOT}/{PROJECT}/{PROJECT}_llm_client_data/embeddings.jsonl

Optionally writes both to a SQLite database at LLM_CLIENT_DB_PATH
(default: ~/projects/data/llm_observability.db).

Configured via env vars (library convention — llm_client already auto-loads
from ~/.secrets/api_keys.env):

    LLM_CLIENT_LOG_ENABLED  — "1" (default) or "0" to disable
    LLM_CLIENT_DATA_ROOT    — base dir (default: ~/projects/data)
    LLM_CLIENT_PROJECT      — project name (default: basename(os.getcwd()))
    LLM_CLIENT_DB_PATH      — SQLite DB path (default: ~/projects/data/llm_observability.db)

Or override at runtime via configure().
"""

from __future__ import annotations

import contextvars
import json
import logging
import math
import os
import re
import sqlite3
import statistics
import threading
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

_enabled: bool = os.environ.get("LLM_CLIENT_LOG_ENABLED", "1") == "1"
_data_root: Path = Path(os.environ.get("LLM_CLIENT_DATA_ROOT", str(Path.home() / "projects" / "data")))
_project: str | None = os.environ.get("LLM_CLIENT_PROJECT")
_db_path: Path = Path(os.environ.get("LLM_CLIENT_DB_PATH", str(Path.home() / "projects" / "data" / "llm_observability.db")))
_db_conn: sqlite3.Connection | None = None
_db_lock = threading.Lock()
_run_timer_lock = threading.Lock()
_run_timers: dict[str, dict[str, Any]] = {}
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


def _get_project() -> str:
    """Get project name, lazily resolving cwd if not configured."""
    if _project is not None:
        return _project
    return Path.cwd().name


def _log_dir() -> Path:
    return _data_root / _get_project() / f"{_get_project()}_llm_client_data"


def configure(
    *,
    enabled: bool | None = None,
    data_root: str | Path | None = None,
    project: str | None = None,
    db_path: str | Path | None = None,
) -> None:
    """Override logging config at runtime."""
    global _enabled, _data_root, _project, _db_path, _db_conn
    if enabled is not None:
        _enabled = enabled
    if data_root is not None:
        _data_root = Path(data_root)
    if project is not None:
        _project = project
    if db_path is not None:
        with _db_lock:
            if _db_conn is not None:
                _db_conn.close()
                _db_conn = None
            _db_path = Path(db_path)


def _start_run_timer(run_id: str) -> None:
    """Store process-level start clocks for an experiment run."""
    with _run_timer_lock:
        _run_timers[run_id] = {
            "wall_t0": time.monotonic(),
            "cpu_t0": time.process_time(),
            "cpu_times_t0": os.times(),
        }


def _pop_run_timer(run_id: str) -> dict[str, Any] | None:
    """Remove and return cached run timer clocks."""
    with _run_timer_lock:
        return _run_timers.pop(run_id, None)


def _auto_capture_run_timing(
    *,
    run_id: str,
    wall_time_s: float | None,
    cpu_time_s: float | None,
    cpu_user_s: float | None,
    cpu_system_s: float | None,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Fill missing runtime timing fields from start_run clocks when available."""
    timer = _run_timers.get(run_id)
    if timer is None:
        return wall_time_s, cpu_time_s, cpu_user_s, cpu_system_s

    if wall_time_s is None:
        wall_time_s = time.monotonic() - timer["wall_t0"]
    if cpu_time_s is None:
        cpu_time_s = time.process_time() - timer["cpu_t0"]
    if cpu_user_s is None or cpu_system_s is None:
        cpu_times_end = os.times()
        cpu_times_t0 = timer["cpu_times_t0"]
        if cpu_user_s is None:
            cpu_user_s = cpu_times_end.user - cpu_times_t0.user
        if cpu_system_s is None:
            cpu_system_s = cpu_times_end.system - cpu_times_t0.system

    return wall_time_s, cpu_time_s, cpu_user_s, cpu_system_s


def get_active_experiment_run_id() -> str | None:
    """Current active experiment run ID for this execution context."""
    return _active_experiment_run_id.get()


def _normalize_feature_profile(profile: str | dict[str, Any]) -> dict[str, Any]:
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
    """Current active feature profile for this execution context."""
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
    """Bind a feature profile as active context for benchmark/eval calls."""

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
    """Activate a feature profile for current context."""
    return ActiveFeatureProfile(profile)


class ActiveExperimentRun:
    """Bind an existing run_id as the active experiment context."""

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
    """Activate an existing experiment run for enforcement checks."""
    return ActiveExperimentRun(run_id)


def configure_experiment_enforcement(
    *,
    mode: str | None = None,
    task_patterns: list[str] | str | None = None,
) -> None:
    """Configure benchmark/eval experiment guardrails via environment.

    Args:
        mode: ``warn`` (default), ``off``, or ``error``.
        task_patterns: Comma-separated regex patterns (or list of patterns)
            for tasks that should require an active experiment context.
    """
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
    """Configure feature-profile contract behavior via environment."""
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
    """Configure AgentSpec contract enforcement via environment.

    Args:
        mode: ``error`` (default), ``warn``, or ``off``.
        task_patterns: Comma-separated regex patterns (or list of patterns)
            for tasks that require an AgentSpec declaration.
    """
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
    """Optionally require an active experiment run for benchmark/eval tasks.

    Controlled by env var ``LLM_CLIENT_EXPERIMENT_ENFORCEMENT``:
    - ``warn`` (default): emit warning when missing
    - ``off``: disabled
    - ``error``: raise ValueError when missing
    """
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
    """Optionally require explicit feature profiles for benchmark/eval tasks.

    Controlled by env var ``LLM_CLIENT_FEATURE_PROFILE_ENFORCEMENT``:
    - ``warn`` (default): emit warning when missing
    - ``off``: disabled
    - ``error``: raise ValueError when missing
    """
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
    """Optionally require AgentSpec declarations for benchmark/eval tasks.

    Controlled by env var ``LLM_CLIENT_AGENT_SPEC_ENFORCEMENT``:
    - ``error`` (default): raise ValueError when missing
    - ``warn``: emit warning when missing
    - ``off``: disabled
    """
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


def log_call(
    *,
    model: str,
    messages: list[dict[str, Any]] | None = None,
    result: Any = None,
    error: Exception | None = None,
    latency_s: float | None = None,
    caller: str = "call_llm",
    task: str | None = None,
    trace_id: str | None = None,
) -> None:
    """Append one JSONL record. Never raises — logging must not break calls."""
    if not _enabled:
        return
    try:
        d = _log_dir()
        d.mkdir(parents=True, exist_ok=True)

        # Extract fields from result if available
        response_content = None
        usage = None
        cost = None
        cost_source = None
        billing_mode = None
        marginal_cost = None
        cache_hit = 0
        finish_reason = None
        warnings: list[str] | None = None
        n_tool_calls: int | None = None
        if result is not None:
            response_content = getattr(result, "content", None)
            usage_raw = getattr(result, "usage", None)
            usage = usage_raw if isinstance(usage_raw, dict) else None
            cost_raw = getattr(result, "cost", None)
            if isinstance(cost_raw, (int, float)):
                cost = float(cost_raw)
            cost_source_raw = getattr(result, "cost_source", None)
            if isinstance(cost_source_raw, str):
                cost_source = cost_source_raw
            billing_mode_raw = getattr(result, "billing_mode", None)
            if isinstance(billing_mode_raw, str):
                billing_mode = billing_mode_raw
            marginal_raw = getattr(result, "marginal_cost", None)
            if isinstance(marginal_raw, (int, float)):
                marginal_cost = float(marginal_raw)
            elif isinstance(cost, (int, float)):
                marginal_cost = float(cost)
            cache_attr = getattr(result, "cache_hit", False)
            cache_hit = 1 if cache_attr is True else 0
            finish_reason = getattr(result, "finish_reason", None)
            warnings_raw = getattr(result, "warnings", None)
            if isinstance(warnings_raw, list):
                warnings = [str(w) for w in warnings_raw if str(w)]
            tool_calls_raw = getattr(result, "tool_calls", None)
            if isinstance(tool_calls_raw, list):
                n_tool_calls = len(tool_calls_raw)

        timestamp = datetime.now(timezone.utc).isoformat()
        record = {
            "timestamp": timestamp,
            "model": model,
            "messages": _truncate_messages(messages),
            "response": response_content,
            "usage": usage,
            "cost": cost,
            "cost_source": cost_source,
            "billing_mode": billing_mode,
            "marginal_cost": marginal_cost,
            "cache_hit": cache_hit,
            "finish_reason": finish_reason,
            "latency_s": round(latency_s, 3) if latency_s is not None else None,
            "error": str(error) if error else None,
            "error_type": type(error).__name__ if error else None,
            "warnings": warnings,
            "n_tool_calls": n_tool_calls,
            "caller": caller,
            "task": task,
            "trace_id": trace_id,
        }
        with open(d / "calls.jsonl", "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

        # SQLite dual-write
        _write_call_to_db(
            timestamp=timestamp,
            model=model,
            messages=_truncate_messages(messages),
            response=response_content,
            usage=usage,
            cost=cost,
            cost_source=cost_source,
            billing_mode=billing_mode,
            marginal_cost=marginal_cost,
            cache_hit=cache_hit,
            finish_reason=finish_reason,
            latency_s=round(latency_s, 3) if latency_s is not None else None,
            error=str(error) if error else None,
            caller=caller,
            task=task,
            trace_id=trace_id,
        )
    except Exception:
        # Never break LLM calls for logging
        logger.debug("io_log.log_call failed", exc_info=True)


def log_embedding(
    *,
    model: str,
    input_count: int,
    input_chars: int,
    dimensions: int | None = None,
    usage: dict[str, Any] | None = None,
    cost: float | None = None,
    latency_s: float | None = None,
    error: Exception | None = None,
    caller: str = "embed",
    task: str | None = None,
    trace_id: str | None = None,
) -> None:
    """Append one JSONL record for an embedding call. Never raises."""
    if not _enabled:
        return
    try:
        d = _log_dir()
        d.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).isoformat()
        record = {
            "timestamp": timestamp,
            "model": model,
            "input_count": input_count,
            "input_chars": input_chars,
            "dimensions": dimensions,
            "usage": usage,
            "cost": cost,
            "latency_s": round(latency_s, 3) if latency_s is not None else None,
            "error": str(error) if error else None,
            "caller": caller,
            "task": task,
            "trace_id": trace_id,
        }
        with open(d / "embeddings.jsonl", "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

        # SQLite dual-write
        _write_embedding_to_db(
            timestamp=timestamp,
            model=model,
            input_count=input_count,
            input_chars=input_chars,
            dimensions=dimensions,
            usage=usage,
            cost=cost,
            latency_s=round(latency_s, 3) if latency_s is not None else None,
            error=str(error) if error else None,
            caller=caller,
            task=task,
            trace_id=trace_id,
        )
    except Exception:
        logger.debug("io_log.log_embedding failed", exc_info=True)


def log_foundation_event(
    *,
    event: dict[str, Any],
    caller: str = "foundation",
    task: str | None = None,
    trace_id: str | None = None,
) -> None:
    """Append one FOUNDATION event record. Never raises by default."""
    if not _enabled:
        return
    try:
        from llm_client.foundation import validate_foundation_event

        normalized_event = validate_foundation_event(event)
        timestamp = datetime.now(timezone.utc).isoformat()
        run_id = str(normalized_event.get("run_id") or "")
        event_id = str(normalized_event.get("event_id") or "")
        event_type = str(normalized_event.get("event_type") or "")

        d = _log_dir()
        d.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": timestamp,
            "caller": caller,
            "task": task,
            "trace_id": trace_id,
            "run_id": run_id or None,
            "event_id": event_id or None,
            "event_type": event_type or None,
            "event": normalized_event,
        }
        with open(d / "foundation_events.jsonl", "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

        _write_foundation_event_to_db(
            timestamp=timestamp,
            run_id=run_id or None,
            trace_id=trace_id,
            event_id=event_id or None,
            event_type=event_type or None,
            payload=normalized_event,
            caller=caller,
            task=task,
        )
    except Exception:
        logger.debug("io_log.log_foundation_event failed", exc_info=True)


# ---------------------------------------------------------------------------
# SQLite observability database
# ---------------------------------------------------------------------------

_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    project TEXT,
    model TEXT NOT NULL,
    messages TEXT,
    response TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    cost REAL,
    cost_source TEXT,
    billing_mode TEXT,
    marginal_cost REAL,
    cache_hit INTEGER DEFAULT 0,
    finish_reason TEXT,
    latency_s REAL,
    error TEXT,
    caller TEXT,
    task TEXT,
    trace_id TEXT
);

CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    project TEXT,
    model TEXT NOT NULL,
    input_count INTEGER,
    input_chars INTEGER,
    dimensions INTEGER,
    prompt_tokens INTEGER,
    total_tokens INTEGER,
    cost REAL,
    latency_s REAL,
    error TEXT,
    caller TEXT,
    task TEXT,
    trace_id TEXT
);

CREATE TABLE IF NOT EXISTS task_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    project TEXT,
    task TEXT,
    trace_id TEXT,
    rubric TEXT NOT NULL,
    method TEXT NOT NULL,
    overall_score REAL NOT NULL,
    dimensions TEXT,
    reasoning TEXT,
    output_model TEXT,
    judge_model TEXT,
    agent_spec TEXT,
    prompt_id TEXT,
    cost REAL,
    latency_s REAL,
    git_commit TEXT
);

CREATE TABLE IF NOT EXISTS experiment_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    timestamp TEXT NOT NULL,
    project TEXT,
    dataset TEXT NOT NULL,
    model TEXT NOT NULL,
    config TEXT,
    provenance TEXT,
    condition_id TEXT,
    seed INTEGER,
    replicate INTEGER,
    scenario_id TEXT,
    phase TEXT,
    metrics_schema TEXT,
    n_items INTEGER DEFAULT 0,
    n_completed INTEGER DEFAULT 0,
    n_errors INTEGER DEFAULT 0,
    summary_metrics TEXT,
    total_cost REAL DEFAULT 0.0,
    wall_time_s REAL,
    cpu_time_s REAL,
    cpu_user_s REAL,
    cpu_system_s REAL,
    git_commit TEXT,
    status TEXT DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS experiment_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    metrics TEXT NOT NULL,
    predicted TEXT,
    gold TEXT,
    latency_s REAL,
    cost REAL,
    n_tool_calls INTEGER,
    error TEXT,
    extra TEXT,
    trace_id TEXT,
    UNIQUE(run_id, item_id)
);

CREATE TABLE IF NOT EXISTS foundation_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    project TEXT,
    run_id TEXT,
    trace_id TEXT,
    event_id TEXT,
    event_type TEXT,
    payload TEXT NOT NULL,
    caller TEXT,
    task TEXT
);
"""

_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_calls_timestamp ON llm_calls(timestamp);
CREATE INDEX IF NOT EXISTS idx_calls_model ON llm_calls(model);
CREATE INDEX IF NOT EXISTS idx_calls_task ON llm_calls(task);
CREATE INDEX IF NOT EXISTS idx_calls_project ON llm_calls(project);
CREATE INDEX IF NOT EXISTS idx_calls_trace_id ON llm_calls(trace_id);
CREATE INDEX IF NOT EXISTS idx_emb_timestamp ON embeddings(timestamp);
CREATE INDEX IF NOT EXISTS idx_emb_model ON embeddings(model);
CREATE INDEX IF NOT EXISTS idx_emb_task ON embeddings(task);
CREATE INDEX IF NOT EXISTS idx_emb_project ON embeddings(project);
CREATE INDEX IF NOT EXISTS idx_emb_trace_id ON embeddings(trace_id);
CREATE INDEX IF NOT EXISTS idx_scores_task ON task_scores(task);
CREATE INDEX IF NOT EXISTS idx_scores_rubric ON task_scores(rubric);
CREATE INDEX IF NOT EXISTS idx_scores_project ON task_scores(project);
CREATE INDEX IF NOT EXISTS idx_scores_trace_id ON task_scores(trace_id);
CREATE INDEX IF NOT EXISTS idx_scores_timestamp ON task_scores(timestamp);
CREATE INDEX IF NOT EXISTS idx_scores_git_commit ON task_scores(git_commit);
CREATE INDEX IF NOT EXISTS idx_expr_dataset ON experiment_runs(dataset);
CREATE INDEX IF NOT EXISTS idx_expr_model ON experiment_runs(model);
CREATE INDEX IF NOT EXISTS idx_expr_project ON experiment_runs(project);
CREATE INDEX IF NOT EXISTS idx_expr_timestamp ON experiment_runs(timestamp);
CREATE INDEX IF NOT EXISTS idx_expr_git_commit ON experiment_runs(git_commit);
CREATE INDEX IF NOT EXISTS idx_expr_condition_id ON experiment_runs(condition_id);
CREATE INDEX IF NOT EXISTS idx_expr_seed ON experiment_runs(seed);
CREATE INDEX IF NOT EXISTS idx_expr_scenario_id ON experiment_runs(scenario_id);
CREATE INDEX IF NOT EXISTS idx_expr_phase ON experiment_runs(phase);
CREATE INDEX IF NOT EXISTS idx_expr_condition_seed ON experiment_runs(condition_id, seed);
CREATE INDEX IF NOT EXISTS idx_expri_run_id ON experiment_items(run_id);
CREATE INDEX IF NOT EXISTS idx_expri_item_id ON experiment_items(item_id);
CREATE INDEX IF NOT EXISTS idx_expri_trace_id ON experiment_items(trace_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_fevent_event_id ON foundation_events(event_id);
CREATE INDEX IF NOT EXISTS idx_fevent_run_id ON foundation_events(run_id);
CREATE INDEX IF NOT EXISTS idx_fevent_trace_id ON foundation_events(trace_id);
CREATE INDEX IF NOT EXISTS idx_fevent_event_type ON foundation_events(event_type);
"""


def _migrate_db(conn: sqlite3.Connection) -> None:
    """Add missing columns (idempotent). For DBs created before these columns existed."""
    for table in ("llm_calls", "embeddings"):
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if "trace_id" not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN trace_id TEXT")
            prefix = table[:3]
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{prefix}_trace_id ON {table}(trace_id)")

    # llm_calls: add accounting metadata fields if missing
    llm_cols = {r[1] for r in conn.execute("PRAGMA table_info(llm_calls)").fetchall()}
    if "cost_source" not in llm_cols:
        conn.execute("ALTER TABLE llm_calls ADD COLUMN cost_source TEXT")
    if "billing_mode" not in llm_cols:
        conn.execute("ALTER TABLE llm_calls ADD COLUMN billing_mode TEXT")
    if "marginal_cost" not in llm_cols:
        conn.execute("ALTER TABLE llm_calls ADD COLUMN marginal_cost REAL")
    if "cache_hit" not in llm_cols:
        conn.execute("ALTER TABLE llm_calls ADD COLUMN cache_hit INTEGER DEFAULT 0")

    # task_scores: add git_commit if missing
    scores_cols = {r[1] for r in conn.execute("PRAGMA table_info(task_scores)").fetchall()}
    if scores_cols and "git_commit" not in scores_cols:
        conn.execute("ALTER TABLE task_scores ADD COLUMN git_commit TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_scores_git_commit ON task_scores(git_commit)")

    # experiment_runs: add provenance + CPU timing fields if missing
    run_cols = {r[1] for r in conn.execute("PRAGMA table_info(experiment_runs)").fetchall()}
    if run_cols and "provenance" not in run_cols:
        conn.execute("ALTER TABLE experiment_runs ADD COLUMN provenance TEXT")
    if run_cols and "cpu_time_s" not in run_cols:
        conn.execute("ALTER TABLE experiment_runs ADD COLUMN cpu_time_s REAL")
    if run_cols and "cpu_user_s" not in run_cols:
        conn.execute("ALTER TABLE experiment_runs ADD COLUMN cpu_user_s REAL")
    if run_cols and "cpu_system_s" not in run_cols:
        conn.execute("ALTER TABLE experiment_runs ADD COLUMN cpu_system_s REAL")
    if run_cols and "condition_id" not in run_cols:
        conn.execute("ALTER TABLE experiment_runs ADD COLUMN condition_id TEXT")
    if run_cols and "seed" not in run_cols:
        conn.execute("ALTER TABLE experiment_runs ADD COLUMN seed INTEGER")
    if run_cols and "replicate" not in run_cols:
        conn.execute("ALTER TABLE experiment_runs ADD COLUMN replicate INTEGER")
    if run_cols and "scenario_id" not in run_cols:
        conn.execute("ALTER TABLE experiment_runs ADD COLUMN scenario_id TEXT")
    if run_cols and "phase" not in run_cols:
        conn.execute("ALTER TABLE experiment_runs ADD COLUMN phase TEXT")

    # experiment_items: add trace_id if missing
    item_cols = {r[1] for r in conn.execute("PRAGMA table_info(experiment_items)").fetchall()}
    if item_cols and "trace_id" not in item_cols:
        conn.execute("ALTER TABLE experiment_items ADD COLUMN trace_id TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_expri_trace_id ON experiment_items(trace_id)")

    conn.commit()


def _get_db() -> sqlite3.Connection:
    """Lazy singleton DB connection. Creates tables on first call."""
    global _db_conn
    with _db_lock:
        if _db_conn is not None:
            return _db_conn
        _db_path.parent.mkdir(parents=True, exist_ok=True)
        _db_conn = sqlite3.connect(str(_db_path), check_same_thread=False)
        _db_conn.executescript(_TABLES_SQL)
        _migrate_db(_db_conn)
        _db_conn.executescript(_INDEXES_SQL)
        return _db_conn


def _write_call_to_db(
    *,
    timestamp: str,
    model: str,
    messages: list[dict[str, Any]] | None,
    response: str | None,
    usage: dict[str, Any] | None,
    cost: float | None,
    cost_source: str | None,
    billing_mode: str | None,
    marginal_cost: float | None,
    cache_hit: int,
    finish_reason: str | None,
    latency_s: float | None,
    error: str | None,
    caller: str,
    task: str | None,
    trace_id: str | None = None,
) -> None:
    """Insert a call record into SQLite. Never raises."""
    try:
        db = _get_db()
        prompt_tokens = (usage or {}).get("prompt_tokens")
        completion_tokens = (usage or {}).get("completion_tokens")
        total_tokens = (usage or {}).get("total_tokens")
        db.execute(
            """INSERT INTO llm_calls
               (timestamp, project, model, messages, response,
                prompt_tokens, completion_tokens, total_tokens,
                cost, cost_source, billing_mode, marginal_cost, cache_hit,
                finish_reason, latency_s, error, caller, task, trace_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                timestamp, _get_project(), model,
                json.dumps(messages, default=str) if messages else None,
                response,
                prompt_tokens, completion_tokens, total_tokens,
                cost, cost_source, billing_mode, marginal_cost, cache_hit,
                finish_reason, latency_s, error, caller, task, trace_id,
            ),
        )
        db.commit()
    except Exception:
        logger.debug("io_log._write_call_to_db failed", exc_info=True)


def _write_embedding_to_db(
    *,
    timestamp: str,
    model: str,
    input_count: int,
    input_chars: int,
    dimensions: int | None,
    usage: dict[str, Any] | None,
    cost: float | None,
    latency_s: float | None,
    error: str | None,
    caller: str,
    task: str | None,
    trace_id: str | None = None,
) -> None:
    """Insert an embedding record into SQLite. Never raises."""
    try:
        db = _get_db()
        prompt_tokens = (usage or {}).get("prompt_tokens")
        total_tokens = (usage or {}).get("total_tokens")
        db.execute(
            """INSERT INTO embeddings
               (timestamp, project, model, input_count, input_chars, dimensions,
                prompt_tokens, total_tokens, cost, latency_s, error, caller, task, trace_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                timestamp, _get_project(), model,
                input_count, input_chars, dimensions,
                prompt_tokens, total_tokens,
                cost, latency_s, error, caller, task, trace_id,
            ),
        )
        db.commit()
    except Exception:
        logger.debug("io_log._write_embedding_to_db failed", exc_info=True)


def _write_foundation_event_to_db(
    *,
    timestamp: str,
    run_id: str | None,
    trace_id: str | None,
    event_id: str | None,
    event_type: str | None,
    payload: dict[str, Any],
    caller: str,
    task: str | None,
) -> None:
    """Insert a foundation event into SQLite. Never raises."""
    try:
        db = _get_db()
        db.execute(
            """INSERT INTO foundation_events
               (timestamp, project, run_id, trace_id, event_id, event_type, payload, caller, task)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                timestamp,
                _get_project(),
                run_id,
                trace_id,
                event_id,
                event_type,
                json.dumps(payload, default=str),
                caller,
                task,
            ),
        )
        db.commit()
    except Exception:
        logger.debug("io_log._write_foundation_event_to_db failed", exc_info=True)


def import_jsonl(jsonl_path: str | Path, table: str = "llm_calls") -> int:
    """Import existing JSONL records into SQLite. Returns count imported.

    Args:
        jsonl_path: Path to the JSONL file (calls.jsonl or embeddings.jsonl).
        table: Target table — "llm_calls" or "embeddings".

    Returns:
        Number of records imported.
    """
    path = Path(jsonl_path)
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    if table not in ("llm_calls", "embeddings"):
        raise ValueError(f"table must be 'llm_calls' or 'embeddings', got {table!r}")

    db = _get_db()
    count = 0

    # Infer project from path: .../data/{project}/{project}_llm_client_data/calls.jsonl
    project = None
    parts = path.parts
    for i, part in enumerate(parts):
        if part.endswith("_llm_client_data") and i > 0:
            project = parts[i - 1]
            break

    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue

        if table == "llm_calls":
            usage = r.get("usage") or {}
            db.execute(
                """INSERT INTO llm_calls
                   (timestamp, project, model, messages, response,
                    prompt_tokens, completion_tokens, total_tokens,
                    cost, cost_source, billing_mode, marginal_cost, cache_hit,
                    finish_reason, latency_s, error, caller, task, trace_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    r.get("timestamp"), project, r.get("model"),
                    json.dumps(r.get("messages"), default=str) if r.get("messages") else None,
                    r.get("response"),
                    usage.get("prompt_tokens"), usage.get("completion_tokens"),
                    usage.get("total_tokens"),
                    r.get("cost"),
                    r.get("cost_source"),
                    r.get("billing_mode"),
                    r.get("marginal_cost"),
                    r.get("cache_hit", 0),
                    r.get("finish_reason"),
                    r.get("latency_s"), r.get("error"),
                    r.get("caller"), r.get("task"), r.get("trace_id"),
                ),
            )
        else:  # embeddings
            usage = r.get("usage") or {}
            db.execute(
                """INSERT INTO embeddings
                   (timestamp, project, model, input_count, input_chars, dimensions,
                    prompt_tokens, total_tokens, cost, latency_s, error, caller, task, trace_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    r.get("timestamp"), project, r.get("model"),
                    r.get("input_count"), r.get("input_chars"), r.get("dimensions"),
                    usage.get("prompt_tokens"), usage.get("total_tokens"),
                    r.get("cost"), r.get("latency_s"), r.get("error"),
                    r.get("caller"), r.get("task"), r.get("trace_id"),
                ),
            )
        count += 1

    db.commit()
    return count


def log_score(
    *,
    rubric: str,
    method: str,
    overall_score: float,
    dimensions: dict[str, Any] | None = None,
    reasoning: str | None = None,
    output_model: str | None = None,
    judge_model: str | None = None,
    agent_spec: str | None = None,
    prompt_id: str | None = None,
    cost: float | None = None,
    latency_s: float | None = None,
    task: str | None = None,
    trace_id: str | None = None,
    git_commit: str | None = None,
) -> None:
    """Write a rubric score to the observability DB. Never raises."""
    if not _enabled:
        return
    try:
        # Auto-capture git commit if not provided
        if git_commit is None:
            from llm_client.git_utils import get_git_head

            git_commit = get_git_head()

        timestamp = datetime.now(timezone.utc).isoformat()

        # JSONL append
        d = _log_dir()
        d.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": timestamp,
            "rubric": rubric,
            "method": method,
            "overall_score": overall_score,
            "dimensions": dimensions,
            "reasoning": reasoning,
            "output_model": output_model,
            "judge_model": judge_model,
            "agent_spec": agent_spec,
            "prompt_id": prompt_id,
            "cost": cost,
            "latency_s": round(latency_s, 3) if latency_s is not None else None,
            "task": task,
            "trace_id": trace_id,
            "git_commit": git_commit,
        }
        with open(d / "scores.jsonl", "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

        # SQLite write
        db = _get_db()
        db.execute(
            """INSERT INTO task_scores
               (timestamp, project, task, trace_id, rubric, method,
                overall_score, dimensions, reasoning, output_model,
                judge_model, agent_spec, prompt_id, cost, latency_s,
                git_commit)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                timestamp, _get_project(), task, trace_id, rubric, method,
                overall_score,
                json.dumps(dimensions, default=str) if dimensions else None,
                reasoning, output_model, judge_model, agent_spec, prompt_id,
                cost, round(latency_s, 3) if latency_s is not None else None,
                git_commit,
            ),
        )
        db.commit()
    except Exception:
        logger.debug("io_log.log_score failed", exc_info=True)


def _truncate_messages(
    messages: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Deep-copy messages for storage (full content, no truncation)."""
    if messages is None:
        return None
    return [dict(m) for m in messages]


# ---------------------------------------------------------------------------
# Resume / lookup functions
# ---------------------------------------------------------------------------


def lookup_result(trace_id: str) -> dict[str, Any] | None:
    """Compatibility shim: delegate to observability.query.lookup_result."""
    from llm_client.observability.query import lookup_result as _lookup_result

    return _lookup_result(trace_id)


def get_completed_traces(
    *,
    project: str | None = None,
    task: str | None = None,
) -> set[str]:
    """Compatibility shim: delegate to observability.query.get_completed_traces."""
    from llm_client.observability.query import get_completed_traces as _get_completed_traces

    return _get_completed_traces(project=project, task=task)


def get_cost(
    *,
    trace_id: str | None = None,
    trace_prefix: str | None = None,
    task: str | None = None,
    project: str | None = None,
    since: str | date | None = None,
) -> float:
    """Compatibility shim: delegate to observability.query.get_cost."""
    from llm_client.observability.query import get_cost as _get_cost

    return _get_cost(
        trace_id=trace_id,
        trace_prefix=trace_prefix,
        task=task,
        project=project,
        since=since,
    )


def get_trace_tree(
    trace_prefix: str,
    *,
    days: int = 7,
) -> list[dict[str, Any]]:
    """Compatibility shim: delegate to observability.query.get_trace_tree."""
    from llm_client.observability.query import get_trace_tree as _get_trace_tree

    return _get_trace_tree(trace_prefix, days=days)


def get_background_mode_adoption(
    *,
    experiments_path: str | Path | None = None,
    since: str | date | datetime | None = None,
    run_id_prefix: str | None = None,
) -> dict[str, Any]:
    """Compatibility shim: delegate to observability.query.get_background_mode_adoption."""
    from llm_client.observability.query import (
        get_background_mode_adoption as _get_background_mode_adoption,
    )

    return _get_background_mode_adoption(
        experiments_path=experiments_path,
        since=since,
        run_id_prefix=run_id_prefix,
    )


# ---------------------------------------------------------------------------
# Experiment logging
# ---------------------------------------------------------------------------


def _build_auto_run_provenance(*, git_commit: str | None) -> dict[str, Any]:
    """Compatibility shim: delegate to observability.experiments helper."""
    from llm_client.observability.experiments import (
        _build_auto_run_provenance as _build_auto_provenance,
    )

    return _build_auto_provenance(git_commit=git_commit)


def start_run(
    *,
    dataset: str,
    model: str,
    task: str | None = None,
    config: dict[str, Any] | None = None,
    condition_id: str | None = None,
    seed: int | None = None,
    replicate: int | None = None,
    scenario_id: str | None = None,
    phase: str | None = None,
    metrics_schema: list[str] | None = None,
    run_id: str | None = None,
    git_commit: str | None = None,
    provenance: dict[str, Any] | None = None,
    feature_profile: str | dict[str, Any] | None = None,
    agent_spec: str | Path | dict[str, Any] | None = None,
    allow_missing_agent_spec: bool = False,
    missing_agent_spec_reason: str | None = None,
    project: str | None = None,
) -> str:
    """Compatibility shim: delegate to observability.experiments.start_run."""
    from llm_client.observability.experiments import start_run as _start_run

    return _start_run(
        dataset=dataset,
        model=model,
        task=task,
        config=config,
        condition_id=condition_id,
        seed=seed,
        replicate=replicate,
        scenario_id=scenario_id,
        phase=phase,
        metrics_schema=metrics_schema,
        run_id=run_id,
        git_commit=git_commit,
        provenance=provenance,
        feature_profile=feature_profile,
        agent_spec=agent_spec,
        allow_missing_agent_spec=allow_missing_agent_spec,
        missing_agent_spec_reason=missing_agent_spec_reason,
        project=project,
    )


def log_item(
    *,
    run_id: str,
    item_id: str,
    metrics: dict[str, Any],
    predicted: str | None = None,
    gold: str | None = None,
    latency_s: float | None = None,
    cost: float | None = None,
    n_tool_calls: int | None = None,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
    trace_id: str | None = None,
) -> None:
    """Compatibility shim: delegate to observability.experiments.log_item."""
    from llm_client.observability.experiments import log_item as _log_item

    _log_item(
        run_id=run_id,
        item_id=item_id,
        metrics=metrics,
        predicted=predicted,
        gold=gold,
        latency_s=latency_s,
        cost=cost,
        n_tool_calls=n_tool_calls,
        error=error,
        extra=extra,
        trace_id=trace_id,
    )


def finish_run(
    *,
    run_id: str,
    summary_metrics: dict[str, Any] | None = None,
    status: str = "completed",
    wall_time_s: float | None = None,
    cpu_time_s: float | None = None,
    cpu_user_s: float | None = None,
    cpu_system_s: float | None = None,
) -> dict[str, Any]:
    """Compatibility shim: delegate to observability.experiments.finish_run."""
    from llm_client.observability.experiments import finish_run as _finish_run

    return _finish_run(
        run_id=run_id,
        summary_metrics=summary_metrics,
        status=status,
        wall_time_s=wall_time_s,
        cpu_time_s=cpu_time_s,
        cpu_user_s=cpu_user_s,
        cpu_system_s=cpu_system_s,
    )


class ExperimentRun:
    """Compatibility shim for observability.experiments.ExperimentRun."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from llm_client.observability.experiments import ExperimentRun as _ExperimentRun

        return _ExperimentRun(*args, **kwargs)


def experiment_run(
    *,
    dataset: str,
    model: str,
    config: dict[str, Any] | None = None,
    condition_id: str | None = None,
    seed: int | None = None,
    replicate: int | None = None,
    scenario_id: str | None = None,
    phase: str | None = None,
    metrics_schema: list[str] | None = None,
    run_id: str | None = None,
    git_commit: str | None = None,
    provenance: dict[str, Any] | None = None,
    feature_profile: str | dict[str, Any] | None = None,
    project: str | None = None,
    status_on_exception: str = "interrupted",
) -> Any:
    """Compatibility shim: delegate to observability.experiments.experiment_run."""
    from llm_client.observability.experiments import experiment_run as _experiment_run

    return _experiment_run(
        dataset=dataset,
        model=model,
        config=config,
        condition_id=condition_id,
        seed=seed,
        replicate=replicate,
        scenario_id=scenario_id,
        phase=phase,
        metrics_schema=metrics_schema,
        run_id=run_id,
        git_commit=git_commit,
        provenance=provenance,
        feature_profile=feature_profile,
        project=project,
        status_on_exception=status_on_exception,
    )


def get_runs(
    *,
    dataset: str | None = None,
    model: str | None = None,
    project: str | None = None,
    condition_id: str | None = None,
    scenario_id: str | None = None,
    phase: str | None = None,
    seed: int | None = None,
    since: str | date | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Compatibility shim: delegate to observability.experiments.get_runs."""
    from llm_client.observability.experiments import get_runs as _get_runs

    return _get_runs(
        dataset=dataset,
        model=model,
        project=project,
        condition_id=condition_id,
        scenario_id=scenario_id,
        phase=phase,
        seed=seed,
        since=since,
        limit=limit,
    )


def get_run(run_id: str) -> dict[str, Any] | None:
    """Compatibility shim: delegate to observability.experiments.get_run."""
    from llm_client.observability.experiments import get_run as _get_run

    return _get_run(run_id)


def get_run_items(run_id: str) -> list[dict[str, Any]]:
    """Compatibility shim: delegate to observability.experiments.get_run_items."""
    from llm_client.observability.experiments import get_run_items as _get_run_items

    return _get_run_items(run_id)


def compare_runs(run_ids: list[str]) -> dict[str, Any]:
    """Compatibility shim: delegate to observability.experiments.compare_runs."""
    from llm_client.observability.experiments import compare_runs as _compare_runs

    return _compare_runs(run_ids)


def compare_cohorts(
    *,
    condition_ids: list[str] | None = None,
    baseline_condition_id: str | None = None,
    dataset: str | None = None,
    model: str | None = None,
    project: str | None = None,
    scenario_id: str | None = None,
    phase: str | None = None,
    since: str | date | None = None,
    limit: int = 500,
) -> dict[str, Any]:
    """Compatibility shim: delegate to observability.experiments.compare_cohorts."""
    from llm_client.observability.experiments import compare_cohorts as _compare_cohorts

    return _compare_cohorts(
        condition_ids=condition_ids,
        baseline_condition_id=baseline_condition_id,
        dataset=dataset,
        model=model,
        project=project,
        scenario_id=scenario_id,
        phase=phase,
        since=since,
        limit=limit,
    )
