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
import os
import re
import sqlite3
import threading
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

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

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any) -> bool:
        if self._token is not None:
            _active_feature_profile.reset(self._token)
            self._token = None
        return False

    async def __aenter__(self) -> "ActiveFeatureProfile":
        return self.__enter__()

    async def __aexit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any) -> bool:
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

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any) -> bool:
        if self._token is not None:
            _active_experiment_run_id.reset(self._token)
            self._token = None
        return False

    async def __aenter__(self) -> "ActiveExperimentRun":
        return self.__enter__()

    async def __aexit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any) -> bool:
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
    usage: dict | None = None,
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
CREATE INDEX IF NOT EXISTS idx_expri_run_id ON experiment_items(run_id);
CREATE INDEX IF NOT EXISTS idx_expri_item_id ON experiment_items(item_id);
CREATE INDEX IF NOT EXISTS idx_expri_trace_id ON experiment_items(trace_id);
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
    usage: dict | None,
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
    usage: dict | None,
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
    """Look up a successful LLM call by trace_id.

    Returns a dict with response, model, cost, latency_s, finish_reason,
    prompt_tokens, completion_tokens — or None if no successful call found.
    """
    db = _get_db()
    if db is None:
        return None
    row = db.execute(
        "SELECT response, model, cost, latency_s, finish_reason, "
        "prompt_tokens, completion_tokens, timestamp "
        "FROM llm_calls WHERE trace_id = ? AND error IS NULL "
        "ORDER BY timestamp DESC LIMIT 1",
        (trace_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "response": row[0],
        "model": row[1],
        "cost": row[2],
        "latency_s": row[3],
        "finish_reason": row[4],
        "prompt_tokens": row[5],
        "completion_tokens": row[6],
        "timestamp": row[7],
    }


def get_completed_traces(
    *,
    project: str | None = None,
    task: str | None = None,
) -> set[str]:
    """Return all trace_ids that have at least one successful call.

    Filter by project and/or task to scope results.
    """
    db = _get_db()
    if db is None:
        return set()
    clauses = ["error IS NULL", "trace_id IS NOT NULL"]
    params: list[str] = []
    if project is not None:
        clauses.append("project = ?")
        params.append(project)
    if task is not None:
        clauses.append("task = ?")
        params.append(task)
    where = " AND ".join(clauses)
    rows = db.execute(
        f"SELECT DISTINCT trace_id FROM llm_calls WHERE {where}",  # noqa: S608
        params,
    ).fetchall()
    return {r[0] for r in rows}


def get_cost(
    *,
    trace_id: str | None = None,
    trace_prefix: str | None = None,
    task: str | None = None,
    project: str | None = None,
    since: str | date | None = None,
) -> float:
    """Query cumulative cost from the observability DB.

    At least one filter must be provided. Combines LLM call costs and
    embedding costs. Returns total cost in USD (0.0 if no records found).

    Args:
        trace_id: Sum cost for this exact trace_id.
        trace_prefix: Sum cost for this trace_id and all children
            (matched with ``trace_id = prefix OR trace_id LIKE prefix/%``).
            Mutually exclusive with trace_id.
        task: Sum cost for this task name.
        project: Sum cost for this project.
        since: Only include records on or after this date (ISO string or date).
    """
    if trace_id is not None and trace_prefix is not None:
        raise ValueError("trace_id and trace_prefix are mutually exclusive.")
    if not any([trace_id, trace_prefix, task, project, since]):
        raise ValueError("At least one filter (trace_id, trace_prefix, task, project, since) is required.")
    try:
        db = _get_db()
    except Exception:
        return 0.0

    total = 0.0
    for table in ("llm_calls", "embeddings"):
        clauses: list[str] = ["error IS NULL"]
        params: list[str] = []
        if trace_id is not None:
            clauses.append("trace_id = ?")
            params.append(trace_id)
        if trace_prefix is not None:
            clauses.append("(trace_id = ? OR trace_id LIKE ?)")
            params.extend([trace_prefix, trace_prefix + "/%"])
        if task is not None:
            clauses.append("task = ?")
            params.append(task)
        if project is not None:
            clauses.append("project = ?")
            params.append(project)
        if since is not None:
            since_str = since.isoformat() if isinstance(since, date) else since
            clauses.append("timestamp >= ?")
            params.append(since_str)
        where = " AND ".join(clauses)
        sum_expr = "COALESCE(SUM(COALESCE(marginal_cost, cost)), 0)" if table == "llm_calls" else "COALESCE(SUM(cost), 0)"
        row = db.execute(
            f"SELECT {sum_expr} FROM {table} WHERE {where}",  # noqa: S608
            params,
        ).fetchone()
        total += row[0] if row else 0.0
    return total


def get_trace_tree(
    trace_prefix: str,
    *,
    days: int = 7,
) -> list[dict[str, Any]]:
    """Roll up child traces under a parent prefix.

    Hierarchical trace_ids use ``/`` as separator:
    ``"openclaw.morning_brief/sam_gov_research_abc"``

    Given prefix ``"openclaw.morning_brief"``, returns every distinct
    trace_id starting with that prefix, with cost/call/error rollup.

    Args:
        trace_prefix: The parent trace_id prefix (matched with ``LIKE prefix/%``
            and also the exact prefix itself).
        days: Look-back window (default 7).

    Returns:
        List of dicts sorted by last_seen descending. Each dict has:
        trace_id, project, task, total_cost_usd, call_count, error_count,
        first_seen, last_seen, models_used, depth (0 = exact match, 1+ = children).
    """
    db = _get_db()
    if db is None:
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_iso = cutoff.isoformat()

    # Match exact prefix OR anything under it (prefix/...)
    like_pattern = trace_prefix + "/%"

    rows = db.execute(
        """SELECT
            trace_id,
            COALESCE(project, 'unknown') as project,
            COALESCE(task, 'untagged') as task,
            ROUND(SUM(CASE WHEN error IS NULL THEN cost ELSE 0 END), 6) as total_cost,
            COUNT(*) as call_count,
            SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as error_count,
            MIN(timestamp) as first_seen,
            MAX(timestamp) as last_seen,
            GROUP_CONCAT(DISTINCT model) as models_used
        FROM llm_calls
        WHERE trace_id IS NOT NULL
          AND (trace_id = ? OR trace_id LIKE ?)
          AND timestamp >= ?
        GROUP BY trace_id
        ORDER BY MAX(timestamp) DESC""",
        (trace_prefix, like_pattern, cutoff_iso),
    ).fetchall()

    prefix_len = len(trace_prefix)
    result = []
    for r in rows:
        tid = r[0]
        # depth: 0 = exact match, 1 = direct child, 2+ = deeper
        if tid == trace_prefix:
            depth = 0
        else:
            # tid starts with prefix/ — count remaining slashes + 1
            suffix = tid[prefix_len + 1:]  # skip the /
            depth = suffix.count("/") + 1

        result.append({
            "trace_id": tid,
            "depth": depth,
            "project": r[1],
            "task": r[2],
            "total_cost_usd": r[3],
            "call_count": r[4],
            "error_count": r[5],
            "first_seen": r[6],
            "last_seen": r[7],
            "models_used": r[8].split(",") if r[8] else [],
        })

    return result


# ---------------------------------------------------------------------------
# Experiment logging
# ---------------------------------------------------------------------------


def _build_auto_run_provenance(*, git_commit: str | None) -> dict[str, Any]:
    """Build automatic provenance metadata for an experiment run."""
    provenance: dict[str, Any] = {
        "git_dirty": False,
        "changed_files": [],
        "diff_categories": [],
    }
    try:
        from llm_client.git_utils import classify_diff_files, get_working_tree_files, is_git_dirty

        changed_files = get_working_tree_files()
        provenance["git_dirty"] = is_git_dirty()
        provenance["changed_files"] = changed_files
        provenance["diff_categories"] = sorted(classify_diff_files(changed_files))
    except Exception:
        logger.debug("io_log._build_auto_run_provenance failed", exc_info=True)

    if git_commit:
        provenance["git_commit"] = git_commit
    return provenance


def start_run(
    *,
    dataset: str,
    model: str,
    task: str | None = None,
    config: dict[str, Any] | None = None,
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
    """Register an experiment run. Returns run_id.

    Args:
        dataset: Dataset name (e.g. "HotpotQA", "MuSiQue").
        model: Model used (e.g. "gemini/gemini-3-flash").
        task: Optional task label (e.g. "digimon.benchmark") used for
            enforcement policies and provenance.
        config: Run configuration (backend, mode, timeout, etc.).
        metrics_schema: Metric names items will report (e.g. ["em", "f1", "llm_em"]).
        run_id: Explicit run ID. Auto-generated UUID if None.
        git_commit: Git SHA. Auto-captured from HEAD if None.
        provenance: Optional provenance manifest. Auto-captured git state is
            merged in and stored in run metadata.
        feature_profile: Optional explicit feature profile declaration
            (profile name or dict). If omitted, active context/env profile
            is attached when available.
        agent_spec: Optional AgentSpec path or mapping. Required for tasks
            matching AgentSpec enforcement patterns unless explicitly opted out.
        allow_missing_agent_spec: Explicit opt-out flag when no AgentSpec is
            provided. Requires missing_agent_spec_reason.
        missing_agent_spec_reason: Human-readable justification for AgentSpec
            opt-out; recorded in provenance.
        project: Project name override. Uses _get_project() if None.

    Returns:
        The run_id string.
    """
    if run_id is None:
        run_id = uuid.uuid4().hex[:12]
    _start_run_timer(run_id)

    if git_commit is None:
        from llm_client.git_utils import get_git_head
        git_commit = get_git_head()

    timestamp = datetime.now(timezone.utc).isoformat()
    proj = project or _get_project()
    auto_provenance = _build_auto_run_provenance(git_commit=git_commit)
    merged_provenance = dict(auto_provenance)
    if provenance:
        merged_provenance.update(provenance)
    resolved_profile: dict[str, Any] | None = None
    try:
        if feature_profile is not None:
            resolved_profile = _normalize_feature_profile(feature_profile)
        else:
            active_profile = get_active_feature_profile()
            if active_profile is not None:
                resolved_profile = dict(active_profile)
    except Exception:
        logger.warning("Failed to normalize feature profile for run provenance.", exc_info=True)
    if resolved_profile is not None:
        merged_provenance["feature_profile"] = resolved_profile

    agent_spec_payload: dict[str, Any] | None = None
    if agent_spec is not None:
        try:
            from llm_client.agent_spec import load_agent_spec
        except Exception as exc:
            raise ValueError("Failed to import AgentSpec loader from llm_client.agent_spec") from exc
        _, agent_spec_summary = load_agent_spec(agent_spec)
        agent_spec_payload = {
            "summary": agent_spec_summary,
        }
        merged_provenance["agent_spec"] = agent_spec_payload

    enforce_agent_spec(
        task,
        has_agent_spec=agent_spec_payload is not None,
        allow_missing=allow_missing_agent_spec,
        missing_reason=missing_agent_spec_reason,
        caller="llm_client.io_log.start_run",
    )

    if allow_missing_agent_spec:
        reason = (missing_agent_spec_reason or "").strip()
        merged_provenance["agent_spec_opt_out"] = {
            "enabled": True,
            "reason": reason or None,
        }
    if task:
        merged_provenance["task"] = task
    provenance_payload = merged_provenance or None

    # JSONL append
    try:
        d = _log_dir()
        d.mkdir(parents=True, exist_ok=True)
        record = {
            "type": "run_start",
            "run_id": run_id,
            "timestamp": timestamp,
            "project": proj,
            "dataset": dataset,
            "model": model,
            "config": config,
            "metrics_schema": metrics_schema,
            "git_commit": git_commit,
            "provenance": provenance_payload,
        }
        with open(d / "experiments.jsonl", "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        logger.debug("io_log.start_run JSONL write failed", exc_info=True)

    # SQLite write
    try:
        db = _get_db()
        db.execute(
            """INSERT INTO experiment_runs
               (run_id, timestamp, project, dataset, model, config,
                provenance, metrics_schema, git_commit, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'running')""",
            (
                run_id, timestamp, proj, dataset, model,
                json.dumps(config, default=str) if config else None,
                json.dumps(provenance_payload, default=str) if provenance_payload else None,
                json.dumps(metrics_schema) if metrics_schema else None,
                git_commit,
            ),
        )
        db.commit()
    except Exception:
        logger.debug("io_log.start_run DB write failed", exc_info=True)

    return run_id


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
    """Log one item result. Dual-write SQLite + JSONL. Never raises."""
    timestamp = datetime.now(timezone.utc).isoformat()

    # JSONL append
    try:
        d = _log_dir()
        d.mkdir(parents=True, exist_ok=True)
        record = {
            "type": "item",
            "run_id": run_id,
            "item_id": item_id,
            "timestamp": timestamp,
            "metrics": metrics,
            "predicted": predicted,
            "gold": gold,
            "latency_s": round(latency_s, 3) if latency_s is not None else None,
            "cost": cost,
            "n_tool_calls": n_tool_calls,
            "error": error,
            "extra": extra,
            "trace_id": trace_id,
        }
        with open(d / "experiments.jsonl", "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        logger.debug("io_log.log_item JSONL write failed", exc_info=True)

    # SQLite write
    try:
        db = _get_db()
        db.execute(
            """INSERT OR REPLACE INTO experiment_items
               (run_id, item_id, timestamp, metrics, predicted, gold,
                latency_s, cost, n_tool_calls, error, extra, trace_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id, item_id, timestamp,
                json.dumps(metrics, default=str),
                predicted, gold,
                round(latency_s, 3) if latency_s is not None else None,
                cost, n_tool_calls, error,
                json.dumps(extra, default=str) if extra else None,
                trace_id,
            ),
        )
        db.commit()
    except Exception:
        logger.debug("io_log.log_item DB write failed", exc_info=True)


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
    """Finalize run. Auto-aggregates metrics from items if summary_metrics not provided.

    Returns the final run record as a dict.
    """
    db = _get_db()

    # Fetch items for this run
    item_rows = db.execute(
        "SELECT metrics, cost, error FROM experiment_items WHERE run_id = ?",
        (run_id,),
    ).fetchall()

    n_items = len(item_rows)
    n_errors = sum(1 for r in item_rows if r[2] is not None)
    n_completed = n_items - n_errors
    total_cost = sum(r[1] or 0.0 for r in item_rows)

    wall_time_s, cpu_time_s, cpu_user_s, cpu_system_s = _auto_capture_run_timing(
        run_id=run_id,
        wall_time_s=wall_time_s,
        cpu_time_s=cpu_time_s,
        cpu_user_s=cpu_user_s,
        cpu_system_s=cpu_system_s,
    )

    # Auto-aggregate metrics if not provided
    if summary_metrics is None:
        # Get metrics_schema from run
        schema_row = db.execute(
            "SELECT metrics_schema FROM experiment_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        schema: list[str] = []
        if schema_row and schema_row[0]:
            schema = json.loads(schema_row[0])

        summary_metrics = {}
        for metric_name in schema:
            values = []
            for r in item_rows:
                m = json.loads(r[0])
                v = m.get(metric_name)
                if v is not None:
                    values.append(float(v))
            if values:
                summary_metrics[f"avg_{metric_name}"] = round(
                    100.0 * sum(values) / len(values), 2
                )

    # Update run record
    try:
        db.execute(
            """UPDATE experiment_runs
               SET n_items = ?, n_completed = ?, n_errors = ?,
                   summary_metrics = ?, total_cost = ?,
                   wall_time_s = ?, cpu_time_s = ?, cpu_user_s = ?, cpu_system_s = ?,
                   status = ?
               WHERE run_id = ?""",
            (
                n_items, n_completed, n_errors,
                json.dumps(summary_metrics, default=str) if summary_metrics else None,
                round(total_cost, 6),
                round(wall_time_s, 1) if wall_time_s is not None else None,
                round(cpu_time_s, 3) if cpu_time_s is not None else None,
                round(cpu_user_s, 3) if cpu_user_s is not None else None,
                round(cpu_system_s, 3) if cpu_system_s is not None else None,
                status,
                run_id,
            ),
        )
        db.commit()
    except Exception:
        logger.debug("io_log.finish_run DB update failed", exc_info=True)

    # JSONL append
    try:
        d = _log_dir()
        d.mkdir(parents=True, exist_ok=True)
        record = {
            "type": "run_finish",
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_items": n_items,
            "n_completed": n_completed,
            "n_errors": n_errors,
            "summary_metrics": summary_metrics,
            "total_cost": round(total_cost, 6),
            "wall_time_s": round(wall_time_s, 1) if wall_time_s is not None else None,
            "cpu_time_s": round(cpu_time_s, 3) if cpu_time_s is not None else None,
            "cpu_user_s": round(cpu_user_s, 3) if cpu_user_s is not None else None,
            "cpu_system_s": round(cpu_system_s, 3) if cpu_system_s is not None else None,
            "status": status,
        }
        with open(d / "experiments.jsonl", "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        logger.debug("io_log.finish_run JSONL write failed", exc_info=True)

    # Return the full run record
    row = db.execute(
        """SELECT run_id, timestamp, project, dataset, model, config, provenance,
                  metrics_schema, n_items, n_completed, n_errors,
                  summary_metrics, total_cost, wall_time_s,
                  cpu_time_s, cpu_user_s, cpu_system_s,
                  git_commit, status
           FROM experiment_runs WHERE run_id = ?""",
        (run_id,),
    ).fetchone()

    _pop_run_timer(run_id)

    if row is None:
        return {"run_id": run_id, "status": status}

    return {
        "run_id": row[0],
        "timestamp": row[1],
        "project": row[2],
        "dataset": row[3],
        "model": row[4],
        "config": json.loads(row[5]) if row[5] else None,
        "provenance": json.loads(row[6]) if row[6] else None,
        "metrics_schema": json.loads(row[7]) if row[7] else None,
        "n_items": row[8],
        "n_completed": row[9],
        "n_errors": row[10],
        "summary_metrics": json.loads(row[11]) if row[11] else None,
        "total_cost": row[12],
        "wall_time_s": row[13],
        "cpu_time_s": row[14],
        "cpu_user_s": row[15],
        "cpu_system_s": row[16],
        "git_commit": row[17],
        "status": row[18],
    }


class ExperimentRun:
    """Managed experiment run with auto timing, context, and convenience APIs.

    Typical usage:

        with experiment_run(dataset="MuSiQue", model="gpt-5-mini") as run:
            run.log_item(item_id="q1", metrics={"em": 1})
            run.finish(status="completed")  # optional, auto-called on exit
    """

    def __init__(
        self,
        *,
        dataset: str,
        model: str,
        config: dict[str, Any] | None = None,
        metrics_schema: list[str] | None = None,
        run_id: str | None = None,
        git_commit: str | None = None,
        provenance: dict[str, Any] | None = None,
        feature_profile: str | dict[str, Any] | None = None,
        project: str | None = None,
        status_on_exception: str = "interrupted",
    ) -> None:
        self._feature_profile = (
            _normalize_feature_profile(feature_profile)
            if feature_profile is not None
            else None
        )
        self.run_id = start_run(
            dataset=dataset,
            model=model,
            config=config,
            metrics_schema=metrics_schema,
            run_id=run_id,
            git_commit=git_commit,
            provenance=provenance,
            feature_profile=self._feature_profile,
            project=project,
        )
        self._status_on_exception = status_on_exception
        self._finished = False
        self._active_token: contextvars.Token[str | None] | None = None
        self._feature_profile_token: contextvars.Token[dict[str, Any] | None] | None = None

    def __enter__(self) -> "ExperimentRun":
        if self._active_token is None and get_active_experiment_run_id() != self.run_id:
            self._active_token = _active_experiment_run_id.set(self.run_id)
        if self._feature_profile is not None and self._feature_profile_token is None:
            self._feature_profile_token = _active_feature_profile.set(self._feature_profile)
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any) -> bool:
        try:
            if not self._finished:
                final_status = "completed" if exc_type is None else self._status_on_exception
                self.finish(status=final_status)
        finally:
            if self._active_token is not None:
                _active_experiment_run_id.reset(self._active_token)
                self._active_token = None
            if self._feature_profile_token is not None:
                _active_feature_profile.reset(self._feature_profile_token)
                self._feature_profile_token = None
        return False

    async def __aenter__(self) -> "ExperimentRun":
        return self.__enter__()

    async def __aexit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any) -> bool:
        return self.__exit__(exc_type, exc, tb)

    def log_item(
        self,
        *,
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
        """Log one item into this run."""
        log_item(
            run_id=self.run_id,
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

    def finish(
        self,
        *,
        summary_metrics: dict[str, Any] | None = None,
        status: str = "completed",
        wall_time_s: float | None = None,
        cpu_time_s: float | None = None,
        cpu_user_s: float | None = None,
        cpu_system_s: float | None = None,
    ) -> dict[str, Any]:
        """Finalize run (idempotent). Missing timing fields auto-capture."""
        if self._finished:
            existing = get_run(self.run_id)
            if existing is not None:
                return existing
            return {"run_id": self.run_id, "status": status}

        result = finish_run(
            run_id=self.run_id,
            summary_metrics=summary_metrics,
            status=status,
            wall_time_s=wall_time_s,
            cpu_time_s=cpu_time_s,
            cpu_user_s=cpu_user_s,
            cpu_system_s=cpu_system_s,
        )
        self._finished = True
        return result


def experiment_run(
    *,
    dataset: str,
    model: str,
    config: dict[str, Any] | None = None,
    metrics_schema: list[str] | None = None,
    run_id: str | None = None,
    git_commit: str | None = None,
    provenance: dict[str, Any] | None = None,
    feature_profile: str | dict[str, Any] | None = None,
    project: str | None = None,
    status_on_exception: str = "interrupted",
) -> ExperimentRun:
    """Create a managed experiment run context.

    Returns an ``ExperimentRun`` object usable via ``with`` or ``async with``.
    """
    return ExperimentRun(
        dataset=dataset,
        model=model,
        config=config,
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
    since: str | date | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Query experiment runs, newest first."""
    db = _get_db()

    clauses: list[str] = []
    params: list[Any] = []

    if dataset is not None:
        clauses.append("dataset = ?")
        params.append(dataset)
    if model is not None:
        clauses.append("model = ?")
        params.append(model)
    if project is not None:
        clauses.append("project = ?")
        params.append(project)
    if since is not None:
        since_str = since.isoformat() if isinstance(since, date) else since
        clauses.append("timestamp >= ?")
        params.append(since_str)

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)

    rows = db.execute(
        f"""SELECT run_id, timestamp, project, dataset, model,
                   config, provenance,
                   n_items, n_completed, n_errors,
                   summary_metrics, total_cost, wall_time_s,
                   cpu_time_s, cpu_user_s, cpu_system_s,
                   git_commit, status
            FROM experiment_runs
            {where}
            ORDER BY timestamp DESC
            LIMIT ?""",  # noqa: S608
        params,
    ).fetchall()

    results = []
    for r in rows:
        results.append({
            "run_id": r[0],
            "timestamp": r[1],
            "project": r[2],
            "dataset": r[3],
            "model": r[4],
            "config": json.loads(r[5]) if r[5] else None,
            "provenance": json.loads(r[6]) if r[6] else None,
            "n_items": r[7],
            "n_completed": r[8],
            "n_errors": r[9],
            "summary_metrics": json.loads(r[10]) if r[10] else None,
            "total_cost": r[11],
            "wall_time_s": r[12],
            "cpu_time_s": r[13],
            "cpu_user_s": r[14],
            "cpu_system_s": r[15],
            "git_commit": r[16],
            "status": r[17],
        })
    return results


def get_run(run_id: str) -> dict[str, Any] | None:
    """Fetch one experiment run by run_id."""
    db = _get_db()
    row = db.execute(
        """SELECT run_id, timestamp, project, dataset, model,
                  config, provenance, metrics_schema,
                  n_items, n_completed, n_errors, summary_metrics,
                  total_cost, wall_time_s, cpu_time_s, cpu_user_s, cpu_system_s,
                  git_commit, status
           FROM experiment_runs
           WHERE run_id = ?""",
        (run_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "run_id": row[0],
        "timestamp": row[1],
        "project": row[2],
        "dataset": row[3],
        "model": row[4],
        "config": json.loads(row[5]) if row[5] else None,
        "provenance": json.loads(row[6]) if row[6] else None,
        "metrics_schema": json.loads(row[7]) if row[7] else None,
        "n_items": row[8],
        "n_completed": row[9],
        "n_errors": row[10],
        "summary_metrics": json.loads(row[11]) if row[11] else None,
        "total_cost": row[12],
        "wall_time_s": row[13],
        "cpu_time_s": row[14],
        "cpu_user_s": row[15],
        "cpu_system_s": row[16],
        "git_commit": row[17],
        "status": row[18],
    }


def get_run_items(run_id: str) -> list[dict[str, Any]]:
    """All items for a run, ordered by timestamp."""
    db = _get_db()

    rows = db.execute(
        """SELECT item_id, timestamp, metrics, predicted, gold,
                  latency_s, cost, n_tool_calls, error, extra, trace_id
           FROM experiment_items
           WHERE run_id = ?
           ORDER BY timestamp""",
        (run_id,),
    ).fetchall()

    results = []
    for r in rows:
        results.append({
            "item_id": r[0],
            "timestamp": r[1],
            "metrics": json.loads(r[2]) if r[2] else {},
            "predicted": r[3],
            "gold": r[4],
            "latency_s": r[5],
            "cost": r[6],
            "n_tool_calls": r[7],
            "error": r[8],
            "extra": json.loads(r[9]) if r[9] else None,
            "trace_id": r[10],
        })
    return results


def compare_runs(run_ids: list[str]) -> dict[str, Any]:
    """Side-by-side summary_metrics for 2+ runs, with deltas from first run."""
    if len(run_ids) < 2:
        raise ValueError("compare_runs requires at least 2 run_ids.")

    def _to_float(v: Any) -> float | None:
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        try:
            if isinstance(v, str) and v.strip():
                return float(v)
        except ValueError:
            return None
        return None

    def _item_delta(base_items: list[dict[str, Any]], cand_items: list[dict[str, Any]]) -> dict[str, Any]:
        base_map = {str(it["item_id"]): it for it in base_items}
        cand_map = {str(it["item_id"]): it for it in cand_items}
        shared_ids = sorted(set(base_map) & set(cand_map))
        new_ids = sorted(set(cand_map) - set(base_map))
        missing_ids = sorted(set(base_map) - set(cand_map))

        improved: dict[str, list[str]] = {}
        regressed: dict[str, list[str]] = {}
        unchanged_items = 0

        for item_id in shared_ids:
            base_metrics = base_map[item_id].get("metrics", {}) or {}
            cand_metrics = cand_map[item_id].get("metrics", {}) or {}
            changed = False
            for metric_name in sorted(set(base_metrics) | set(cand_metrics)):
                base_v = _to_float(base_metrics.get(metric_name))
                cand_v = _to_float(cand_metrics.get(metric_name))
                if base_v is None or cand_v is None or cand_v == base_v:
                    continue
                changed = True
                if cand_v > base_v:
                    improved.setdefault(metric_name, []).append(item_id)
                else:
                    regressed.setdefault(metric_name, []).append(item_id)
            if not changed:
                unchanged_items += 1

        return {
            "shared_items": len(shared_ids),
            "unchanged_items": unchanged_items,
            "improved": improved,
            "regressed": regressed,
            "new_in_candidate": new_ids,
            "missing_in_candidate": missing_ids,
        }

    db = _get_db()
    runs = []
    for rid in run_ids:
        row = db.execute(
            """SELECT run_id, dataset, model, n_items, n_completed, n_errors,
                      summary_metrics, total_cost, wall_time_s,
                      cpu_time_s, cpu_user_s, cpu_system_s,
                      status, timestamp, git_commit, provenance
               FROM experiment_runs WHERE run_id = ?""",
            (rid,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Run not found: {rid}")
        runs.append({
            "run_id": row[0],
            "dataset": row[1],
            "model": row[2],
            "n_items": row[3],
            "n_completed": row[4],
            "n_errors": row[5],
            "summary_metrics": json.loads(row[6]) if row[6] else {},
            "total_cost": row[7],
            "wall_time_s": row[8],
            "cpu_time_s": row[9],
            "cpu_user_s": row[10],
            "cpu_system_s": row[11],
            "status": row[12],
            "timestamp": row[13],
            "git_commit": row[14],
            "provenance": json.loads(row[15]) if row[15] else None,
        })

    # Compute deltas from first run (baseline)
    baseline = runs[0]["summary_metrics"]
    deltas = []
    baseline_items = get_run_items(run_ids[0])
    item_deltas = []
    for run in runs[1:]:
        d: dict[str, float] = {}
        for k, v in run["summary_metrics"].items():
            if k in baseline and isinstance(v, (int, float)) and isinstance(baseline[k], (int, float)):
                d[k] = round(v - baseline[k], 2)
        deltas.append(d)
        per_item = _item_delta(baseline_items, get_run_items(run["run_id"]))
        per_item["run_id"] = run["run_id"]
        item_deltas.append(per_item)

    return {
        "runs": runs,
        "deltas_from_first": deltas,
        "item_deltas_from_first": item_deltas,
    }
