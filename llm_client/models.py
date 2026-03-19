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
from dataclasses import dataclass
from importlib import import_module
from importlib.resources import files as resource_files
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


@dataclass(frozen=True)
class ModelPerformanceObservation:
    """Observed error-rate summary for one model on one task window."""

    litellm_id: str
    error_rate: float
    call_count: int


@dataclass
class PerformanceOverlayDecision:
    """Result of applying empirical reliability demotion to static candidates.

    This keeps the empirical overlay explicit and inspectable without changing
    the public selection API. Static selection happens first; this structure
    then records which candidates stayed in place, which ones were demoted, and
    what observations were available.
    """

    ordered_candidates: list[ModelInfo]
    reliable_candidates: list[ModelInfo]
    demoted_candidates: list[ModelInfo]
    observations: dict[str, ModelPerformanceObservation]


# ---------------------------------------------------------------------------
# Packaged default registry data
# ---------------------------------------------------------------------------

_PACKAGED_DEFAULT_CONFIG_PATH = "data/default_model_registry.json"


def _parse_packaged_default_config(raw_text: str) -> dict[str, Any]:
    """Parse and validate the packaged default model registry payload.

    The packaged defaults are data, not code. Validation stays fail-loud so an
    invalid wheel or source tree raises immediately instead of silently falling
    back to stale embedded literals.
    """

    try:
        loaded = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid packaged model registry JSON {_PACKAGED_DEFAULT_CONFIG_PATH}: {exc}"
        ) from exc

    if not isinstance(loaded, dict):
        raise RuntimeError(
            f"Invalid packaged model registry {_PACKAGED_DEFAULT_CONFIG_PATH}: "
            f"expected top-level object, got {type(loaded).__name__}"
        )

    models = loaded.get("models")
    tasks = loaded.get("tasks")
    if not isinstance(models, list):
        raise RuntimeError(
            f"Invalid packaged model registry {_PACKAGED_DEFAULT_CONFIG_PATH}: "
            f"'models' must be a list"
        )
    if not isinstance(tasks, dict):
        raise RuntimeError(
            f"Invalid packaged model registry {_PACKAGED_DEFAULT_CONFIG_PATH}: "
            f"'tasks' must be a dict"
        )
    return {"models": models, "tasks": tasks}


def _load_packaged_default_config() -> dict[str, Any]:
    """Load the packaged default registry from package data."""

    resource = resource_files("llm_client").joinpath(_PACKAGED_DEFAULT_CONFIG_PATH)
    try:
        raw_text = resource.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Packaged model registry not found: {_PACKAGED_DEFAULT_CONFIG_PATH}"
        ) from exc
    return _parse_packaged_default_config(raw_text)


_DEFAULT_CONFIG: dict[str, Any] = _load_packaged_default_config()


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
    yaml = import_module("yaml")  # lazy import — only needed if user has a config file

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
    profile = _load_task_profile(config, task)
    candidates = _select_static_candidates(
        config=config,
        profile=profile,
        available_only=available_only,
    )

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
        overlay = _apply_performance_overlay(
            candidates, task,
            days=performance_days,
            min_calls=min_calls,
            error_threshold=error_threshold,
        )
        candidates = overlay.ordered_candidates

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
    models = _load_registry_models(config)

    profile: TaskProfile | None = None
    if task is not None:
        profile = _load_task_profile(config, task)
        models = _select_static_candidates(
            config=config,
            profile=profile,
            available_only=available_only,
        )

    result = []
    for m in models:
        available = bool(os.environ.get(m.api_key_env))
        if available_only and not available:
            continue
        d = m.model_dump()
        d["available"] = available
        result.append(d)

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
    """Legacy JSONL-based performance query (fallback).

    Reads from both legacy undated ``calls.jsonl`` and dated
    ``calls_YYYY-MM-DD.jsonl`` files.
    """
    from llm_client.io_log import _log_dir, glob_jsonl_files

    log_dir = _log_dir()
    log_files = glob_jsonl_files(log_dir, "calls")
    if not log_files:
        return []

    cutoff = datetime.now(timezone.utc).timestamp() - (days * 86400)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for log_file in log_files:
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


def _load_registry_models(config: dict[str, Any]) -> list[ModelInfo]:
    """Normalize configured model entries into typed registry objects."""
    return [ModelInfo(**model) if isinstance(model, dict) else model for model in config["models"]]


def _load_task_profile(config: dict[str, Any], task: str) -> TaskProfile:
    """Load one task profile from config and fail loudly on unknown tasks."""
    tasks_cfg = config["tasks"]
    if task not in tasks_cfg:
        raise KeyError(f"Unknown task {task!r}. Available: {sorted(tasks_cfg)}")
    raw_profile = tasks_cfg[task]
    return TaskProfile(**raw_profile) if isinstance(raw_profile, dict) else raw_profile


def _model_qualifies_for_profile(
    model: ModelInfo,
    profile: TaskProfile,
    *,
    available_only: bool,
) -> bool:
    """Apply static requirement and availability checks for one model."""
    req = profile.require
    if req.structured_output and not model.structured_output:
        return False
    if model.intelligence < req.min_intelligence:
        return False
    if model.context < req.min_context:
        return False
    if available_only and not os.environ.get(model.api_key_env):
        return False
    return True


def _select_static_candidates(
    *,
    config: dict[str, Any],
    profile: TaskProfile,
    available_only: bool,
) -> list[ModelInfo]:
    """Return candidates selected by static registry policy only.

    This helper intentionally excludes any observed-performance overlay so the
    static policy and empirical demotion steps remain separable.
    """
    models = _load_registry_models(config)
    candidates = [
        model
        for model in models
        if _model_qualifies_for_profile(model, profile, available_only=available_only)
    ]
    return _sort_by_prefer(candidates, profile.prefer)


def _load_performance_observations(
    *,
    task: str,
    days: int,
) -> dict[str, ModelPerformanceObservation]:
    """Load model performance observations for one task window.

    Returns an empty mapping when performance data is unavailable or when the
    observability query fails. Callers remain responsible for deciding whether
    and how to use the observations.
    """

    try:
        perf = _query_performance_sql(task=task, days=days)
    except Exception:
        return {}

    observations: dict[str, ModelPerformanceObservation] = {}
    for row in perf:
        litellm_id = row["model"]
        observations[litellm_id] = ModelPerformanceObservation(
            litellm_id=litellm_id,
            error_rate=float(row["error_rate"]),
            call_count=int(row["call_count"]),
        )
    return observations


def _apply_performance_overlay(
    candidates: list[ModelInfo],
    task: str,
    *,
    days: int,
    min_calls: int,
    error_threshold: float,
) -> PerformanceOverlayDecision:
    """Apply empirical reliability demotion over an already static candidate order.

    Queries the observability DB for error rates by model+task over the
    look-back window. Models with error_rate > threshold AND call_count >=
    min_calls are moved to the back. Relative order within each group
    (reliable / demoted) is preserved from the static prefer-sorted input.
    """

    observations = _load_performance_observations(task=task, days=days)
    if not observations:
        return PerformanceOverlayDecision(
            ordered_candidates=list(candidates),
            reliable_candidates=list(candidates),
            demoted_candidates=[],
            observations={},
        )

    def is_unreliable(model: ModelInfo) -> bool:
        info = observations.get(model.litellm_id)
        if info is None:
            return False  # no data = no penalty
        return info.call_count >= min_calls and info.error_rate > error_threshold

    reliable = [c for c in candidates if not is_unreliable(c)]
    unreliable = [c for c in candidates if is_unreliable(c)]

    if not reliable:
        return PerformanceOverlayDecision(
            ordered_candidates=list(candidates),
            reliable_candidates=[],
            demoted_candidates=list(candidates),
            observations=observations,
        )

    if unreliable:
        demoted_names = [m.name for m in unreliable]
        chosen = reliable[0].name
        logger.info(
            "get_model(%s): demoted %s (high error rate), selected %s",
            task, demoted_names, chosen,
        )

    return PerformanceOverlayDecision(
        ordered_candidates=reliable + unreliable,
        reliable_candidates=reliable,
        demoted_candidates=unreliable,
        observations=observations,
    )


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
