"""Model registry with task-based selection and performance tracking.

Centralized model matrix — no more hardcoded model strings scattered across
projects. Each model has intelligence/speed/cost attributes. Task profiles
define requirements and sort preferences. ``get_model("extraction")`` returns
the best available model for that task.

Usage::

    from llm_client import get_model, list_models, query_performance

    model = get_model("extraction")      # best model for structured extraction
    model = get_model("bulk_cheap")      # cheapest available model
    models = list_models(task="synthesis")  # all models sorted for synthesis

    perf = query_performance(task="extraction", days=7)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ModelInfo(BaseModel):
    """A model in the registry with its attributes."""

    name: str
    litellm_id: str
    provider: str
    api_key_env: str
    intelligence: int
    speed: int
    cost: float  # blended $/1M tokens
    context: int
    structured_output: bool
    tool_calling: bool = True
    tags: list[str] = []


class TaskRequirements(BaseModel):
    """Hard requirements a model must meet for a task."""

    structured_output: bool = False
    min_intelligence: int = 0
    min_context: int = 0


class TaskProfile(BaseModel):
    """Defines how to select a model for a task category."""

    description: str
    require: TaskRequirements = TaskRequirements()
    prefer: list[str] = []  # sort keys: "intelligence", "-cost" (- = lower is better)


# ---------------------------------------------------------------------------
# Default registry (embedded — no external file needed)
# ---------------------------------------------------------------------------

_DEFAULT_MODELS: list[dict[str, Any]] = [
    {
        "name": "deepseek-chat",
        "litellm_id": "openrouter/deepseek/deepseek-chat",
        "provider": "openrouter",
        "api_key_env": "OPENROUTER_API_KEY",
        "intelligence": 42,
        "speed": 36,
        "cost": 0.32,
        "context": 128_000,
        "structured_output": True,
        "tags": ["open-weight", "cheap"],
    },
    {
        "name": "gemini-3-flash",
        "litellm_id": "gemini/gemini-3-flash",
        "provider": "google",
        "api_key_env": "GEMINI_API_KEY",
        "intelligence": 46,
        "speed": 207,
        "cost": 1.13,
        "context": 1_000_000,
        "structured_output": True,
        "tags": ["mid-tier"],
    },
    {
        "name": "gemini-2.5-flash",
        "litellm_id": "gemini/gemini-2.5-flash",
        "provider": "google",
        "api_key_env": "GEMINI_API_KEY",
        "intelligence": 34,
        "speed": 152,
        "cost": 0.68,
        "context": 1_000_000,
        "structured_output": True,
        "tags": ["free-tier"],
    },
    {
        "name": "gemini-2.5-flash-lite",
        "litellm_id": "gemini/gemini-2.5-flash-lite",
        "provider": "google",
        "api_key_env": "GEMINI_API_KEY",
        "intelligence": 28,
        "speed": 250,
        "cost": 0.175,
        "context": 1_000_000,
        "structured_output": True,
        "tool_calling": False,
        "tags": ["cheapest-google"],
    },
    {
        "name": "gpt-5-mini",
        "litellm_id": "openrouter/openai/gpt-5-mini",
        "provider": "openrouter",
        "api_key_env": "OPENROUTER_API_KEY",
        "intelligence": 41,
        "speed": 127,
        "cost": 0.69,
        "context": 128_000,
        "structured_output": True,
        "tags": ["reliable-structured"],
    },
    {
        "name": "gpt-5",
        "litellm_id": "openrouter/openai/gpt-5",
        "provider": "openrouter",
        "api_key_env": "OPENROUTER_API_KEY",
        "intelligence": 45,
        "speed": 98,
        "cost": 3.44,
        "context": 128_000,
        "structured_output": True,
        "tags": ["frontier"],
    },
    {
        "name": "gpt-5-nano",
        "litellm_id": "openrouter/openai/gpt-5-nano",
        "provider": "openrouter",
        "api_key_env": "OPENROUTER_API_KEY",
        "intelligence": 27,
        "speed": 141,
        "cost": 0.14,
        "context": 128_000,
        "structured_output": True,
        "tags": ["cheapest-openai"],
    },
    {
        "name": "grok-4.1-fast",
        "litellm_id": "openrouter/x-ai/grok-4.1-fast",
        "provider": "openrouter",
        "api_key_env": "OPENROUTER_API_KEY",
        "intelligence": 39,
        "speed": 179,
        "cost": 0.28,
        "context": 2_000_000,
        "structured_output": True,
        "tags": ["2m-context"],
    },
]

_DEFAULT_TASKS: dict[str, dict[str, Any]] = {
    "extraction": {
        "description": "Structured output with Pydantic",
        "require": {"structured_output": True, "min_intelligence": 35},
        "prefer": ["intelligence", "-cost"],
    },
    "bulk_cheap": {
        "description": "High-volume, cost-sensitive",
        "require": {"min_intelligence": 25},
        "prefer": ["-cost", "speed"],
    },
    "synthesis": {
        "description": "Research synthesis, reports",
        "require": {"min_intelligence": 40},
        "prefer": ["intelligence", "-cost"],
    },
    "graph_building": {
        "description": "Knowledge graph construction",
        "require": {"structured_output": True, "min_intelligence": 30},
        "prefer": ["-cost", "speed"],
    },
    "agent_reasoning": {
        "description": "Complex multi-step reasoning",
        "require": {"min_intelligence": 42},
        "prefer": ["intelligence"],
    },
    "code_generation": {
        "description": "Writing and reviewing code",
        "require": {"min_intelligence": 38},
        "prefer": ["intelligence", "speed"],
    },
    "judging": {
        "description": "LLM-as-judge rubric scoring of task outputs",
        "require": {"structured_output": True, "min_intelligence": 30},
        "prefer": ["-cost", "intelligence"],
    },
}

_DEFAULT_CONFIG: dict[str, Any] = {
    "models": _DEFAULT_MODELS,
    "tasks": _DEFAULT_TASKS,
}


# ---------------------------------------------------------------------------
# Config loading (lazy, cached)
# ---------------------------------------------------------------------------

_config_cache: dict[str, Any] | None = None


def _load_config() -> dict[str, Any]:
    """Load config with fallback chain: env var → user file → built-in defaults."""
    global _config_cache  # noqa: PLW0603
    if _config_cache is not None:
        return _config_cache

    # Try env var path first
    env_path = os.environ.get("LLM_CLIENT_MODELS_CONFIG")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            _config_cache = _load_yaml_file(p)
            logger.debug("Loaded model config from %s (env var)", p)
            return _config_cache
        raise RuntimeError(f"LLM_CLIENT_MODELS_CONFIG points to non-existent file: {p}")

    # Try user config file
    user_path = Path.home() / ".config" / "llm_client" / "models.yaml"
    if user_path.is_file():
        _config_cache = _load_yaml_file(user_path)
        logger.debug("Loaded model config from %s", user_path)
        return _config_cache

    # Built-in defaults
    _config_cache = _DEFAULT_CONFIG
    return _config_cache


def _load_yaml_file(path: Path) -> dict[str, Any]:
    """Load a YAML config file. Merges with defaults for missing keys."""
    import yaml  # type: ignore[import-untyped]  # lazy import — only needed if user has a config file

    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise RuntimeError(f"Invalid config file {path}: expected dict, got {type(raw).__name__}")
    # Merge with defaults: user config overrides, but missing sections fall back
    merged: dict[str, Any] = dict(_DEFAULT_CONFIG)
    if "models" in raw:
        merged["models"] = raw["models"]
    if "tasks" in raw:
        merged["tasks"] = raw["tasks"]
    return merged


def _reset_config() -> None:
    """Reset cached config. For testing only."""
    global _config_cache  # noqa: PLW0603
    _config_cache = None


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------


def get_model(
    task: str,
    *,
    available_only: bool = True,
    use_performance: bool = True,
    performance_days: int = 7,
    min_calls: int = 10,
    error_threshold: float = 0.15,
) -> str:
    """Get the best model for a task category.

    Selects based on static attributes (intelligence, cost, speed) from the
    registry, then optionally demotes models with high error rates based on
    real observability data.

    Args:
        task: Task name (e.g., "extraction", "bulk_cheap", "synthesis").
        available_only: Only consider models whose API key env var is set.
        use_performance: Consult observability DB to demote unreliable models.
        performance_days: Look-back window for performance data (default 7).
        min_calls: Minimum call count before performance data affects ranking.
            Models with fewer calls are not penalized (insufficient data).
        error_threshold: Error rate above which a model is considered unreliable
            (default 0.15 = 15%). Unreliable models are ranked after reliable
            ones, preserving relative order within each group.

    Returns:
        The ``litellm_id`` of the winning model.

    Raises:
        KeyError: Unknown task name.
        RuntimeError: No models qualify (after filtering).
    """
    config = _load_config()
    tasks_cfg = config["tasks"]
    if task not in tasks_cfg:
        raise KeyError(f"Unknown task {task!r}. Available: {sorted(tasks_cfg)}")

    profile = TaskProfile(**tasks_cfg[task]) if isinstance(tasks_cfg[task], dict) else tasks_cfg[task]
    models = [ModelInfo(**m) if isinstance(m, dict) else m for m in config["models"]]

    # Filter by hard requirements
    candidates = []
    for m in models:
        req = profile.require
        if req.structured_output and not m.structured_output:
            continue
        if m.intelligence < req.min_intelligence:
            continue
        if m.context < req.min_context:
            continue
        if available_only and not os.environ.get(m.api_key_env):
            continue
        candidates.append(m)

    if not candidates:
        raise RuntimeError(
            f"No models qualify for task {task!r} "
            f"(available_only={available_only}, "
            f"require={profile.require.model_dump()})"
        )

    # Sort by prefer keys (static attributes)
    candidates = _sort_by_prefer(candidates, profile.prefer)

    # Demote unreliable models based on real performance data
    if use_performance and len(candidates) > 1:
        candidates = _demote_unreliable(
            candidates, task,
            days=performance_days,
            min_calls=min_calls,
            error_threshold=error_threshold,
        )

    return candidates[0].litellm_id


def list_models(
    task: str | None = None,
    *,
    available_only: bool = True,
) -> list[dict[str, Any]]:
    """List models, optionally filtered and sorted for a task.

    Returns list of dicts with all model fields plus ``available: bool``.
    """
    config = _load_config()
    models = [ModelInfo(**m) if isinstance(m, dict) else m for m in config["models"]]

    profile: TaskProfile | None = None
    if task is not None:
        tasks_cfg = config["tasks"]
        if task not in tasks_cfg:
            raise KeyError(f"Unknown task {task!r}. Available: {sorted(tasks_cfg)}")
        profile = TaskProfile(**tasks_cfg[task]) if isinstance(tasks_cfg[task], dict) else tasks_cfg[task]

    result = []
    for m in models:
        available = bool(os.environ.get(m.api_key_env))
        if available_only and not available:
            continue
        if profile is not None:
            req = profile.require
            if req.structured_output and not m.structured_output:
                continue
            if m.intelligence < req.min_intelligence:
                continue
            if m.context < req.min_context:
                continue
        d = m.model_dump()
        d["available"] = available
        result.append(d)

    if profile is not None:
        info_list = [ModelInfo(**{k: v for k, v in d.items() if k != "available"}) for d in result]
        sorted_infos = _sort_by_prefer(info_list, profile.prefer)
        # Re-order result to match
        name_order = [m.name for m in sorted_infos]
        result.sort(key=lambda d: name_order.index(d["name"]))

    return result


def query_performance(
    *,
    task: str | None = None,
    model: str | None = None,
    days: int = 30,
) -> list[dict[str, Any]]:
    """Query performance stats from the observability DB.

    Uses SQLite for fast aggregation. Falls back to JSONL parsing if the
    DB doesn't exist yet (backward compat).

    Args:
        task: Filter to this task category (None = all).
        model: Filter to this model (None = all).
        days: Look back this many days.

    Returns:
        List of dicts with: task, model, call_count, total_cost,
        avg_latency_s, error_rate, avg_tokens.
    """
    # Try SQL first (fast). If SQL returns results, use them.
    # If SQL returns empty or fails, fall back to JSONL (handles migration period
    # where historical data may only exist in JSONL files).
    try:
        sql_result = _query_performance_sql(task=task, model=model, days=days)
        if sql_result:
            return sql_result
    except Exception:
        logger.debug("SQL query_performance failed, falling back to JSONL", exc_info=True)
    return _query_performance_jsonl(task=task, model=model, days=days)


def _query_performance_sql(
    *,
    task: str | None = None,
    model: str | None = None,
    days: int = 30,
) -> list[dict[str, Any]]:
    """SQL-backed performance query."""
    from llm_client.io_log import _get_db

    cutoff = datetime.now(timezone.utc).isoformat()[:10]  # date only for rough cutoff
    cutoff_dt = datetime.now(timezone.utc).timestamp() - (days * 86400)
    cutoff_iso = datetime.fromtimestamp(cutoff_dt, tz=timezone.utc).isoformat()

    db = _get_db()
    sql = """
        SELECT
            COALESCE(task, 'untagged') as task,
            model,
            COUNT(*) as call_count,
            ROUND(COALESCE(SUM(COALESCE(marginal_cost, cost)), 0), 4) as total_cost,
            ROUND(COALESCE(AVG(latency_s), 0), 3) as avg_latency_s,
            ROUND(CAST(SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS REAL) / COUNT(*), 3) as error_rate,
            ROUND(COALESCE(AVG(total_tokens), 0)) as avg_tokens
        FROM llm_calls
        WHERE timestamp >= ?
    """
    params: list[Any] = [cutoff_iso]

    if task is not None:
        sql += " AND COALESCE(task, 'untagged') = ?"
        params.append(task)
    if model is not None:
        sql += " AND model = ?"
        params.append(model)

    sql += " GROUP BY COALESCE(task, 'untagged'), model ORDER BY task, model"

    rows = db.execute(sql, params).fetchall()
    return [
        {
            "task": row[0],
            "model": row[1],
            "call_count": row[2],
            "total_cost": row[3],
            "avg_latency_s": row[4],
            "error_rate": row[5],
            "avg_tokens": int(row[6]),
        }
        for row in rows
    ]


def _query_performance_jsonl(
    *,
    task: str | None = None,
    model: str | None = None,
    days: int = 30,
) -> list[dict[str, Any]]:
    """Legacy JSONL-based performance query (fallback)."""
    from llm_client.io_log import _log_dir

    log_file = _log_dir() / "calls.jsonl"
    if not log_file.is_file():
        return []

    cutoff = datetime.now(timezone.utc).timestamp() - (days * 86400)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for line in log_file.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        ts_str = record.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str).timestamp()
        except (ValueError, TypeError):
            continue
        if ts < cutoff:
            continue

        rec_task = record.get("task") or "untagged"
        rec_model = record.get("model", "unknown")

        if task is not None and rec_task != task:
            continue
        if model is not None and rec_model != model:
            continue

        key = (rec_task, rec_model)
        groups.setdefault(key, []).append(record)

    result = []
    for (grp_task, grp_model), records in sorted(groups.items()):
        total_cost = 0.0
        for r in records:
            raw_cost = r.get("marginal_cost")
            if raw_cost is None:
                raw_cost = r.get("cost")
            if isinstance(raw_cost, bool):
                continue
            if isinstance(raw_cost, (int, float)):
                total_cost += float(raw_cost)
        latencies = [r["latency_s"] for r in records if r.get("latency_s") is not None]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        errors = sum(1 for r in records if r.get("error") is not None)
        total_tokens = sum(
            (r.get("usage") or {}).get("total_tokens", 0)
            for r in records
        )
        count = len(records)

        result.append({
            "task": grp_task,
            "model": grp_model,
            "call_count": count,
            "total_cost": round(total_cost, 4),
            "avg_latency_s": round(avg_latency, 3),
            "error_rate": round(errors / count, 3) if count else 0.0,
            "avg_tokens": round(total_tokens / count) if count else 0,
        })

    return result


def supports_tool_calling(litellm_id: str) -> bool:
    """Check if a model supports native tool/function calling.

    Returns False only for models explicitly marked ``tool_calling: False``
    in the registry.  Unknown models (not in registry) are assumed capable.
    """
    config = _load_config()
    for m in config["models"]:
        entry = m if isinstance(m, dict) else m.model_dump()
        if entry.get("litellm_id") == litellm_id:
            return bool(entry.get("tool_calling", True))
    return True  # unknown models assumed capable


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _demote_unreliable(
    candidates: list[ModelInfo],
    task: str,
    *,
    days: int,
    min_calls: int,
    error_threshold: float,
) -> list[ModelInfo]:
    """Stable-partition candidates: reliable first, unreliable last.

    Queries the observability DB for error rates by model+task over the
    look-back window. Models with error_rate > threshold AND call_count >=
    min_calls are moved to the back. Relative order within each group
    (reliable / unreliable) is preserved from the prefer-sorted input.

    Returns candidates unchanged if no performance data or all models
    are equally reliable/unreliable.
    """
    try:
        perf = _query_performance_sql(task=task, days=days)
    except Exception:
        return candidates

    if not perf:
        return candidates

    # Build lookup: litellm_id → (error_rate, call_count)
    # Performance DB stores the model string as-is from the call
    error_by_model: dict[str, tuple[float, int]] = {}
    for row in perf:
        error_by_model[row["model"]] = (row["error_rate"], row["call_count"])

    def is_unreliable(m: ModelInfo) -> bool:
        info = error_by_model.get(m.litellm_id)
        if info is None:
            return False  # no data = no penalty
        error_rate, call_count = info
        return call_count >= min_calls and error_rate > error_threshold

    reliable = [c for c in candidates if not is_unreliable(c)]
    unreliable = [c for c in candidates if is_unreliable(c)]

    if not reliable:
        return candidates  # all unreliable — don't change order

    if unreliable:
        demoted_names = [m.name for m in unreliable]
        chosen = reliable[0].name
        logger.info(
            "get_model(%s): demoted %s (high error rate), selected %s",
            task, demoted_names, chosen,
        )

    return reliable + unreliable


def _sort_by_prefer(models: list[ModelInfo], prefer: list[str]) -> list[ModelInfo]:
    """Sort models by prefer keys. '-cost' means lower is better."""
    if not prefer:
        return models

    def sort_key(m: ModelInfo) -> tuple[Any, ...]:
        parts: list[Any] = []
        for key in prefer:
            descending = not key.startswith("-")
            attr = key.lstrip("-")
            val = getattr(m, attr, 0)
            # For descending (higher is better), negate; for ascending (lower is better), keep as-is
            parts.append(-val if descending else val)
        return tuple(parts)

    return sorted(models, key=sort_key)
