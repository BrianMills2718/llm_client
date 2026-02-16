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
        "litellm_id": "deepseek/deepseek-chat",
        "provider": "deepseek",
        "api_key_env": "DEEPSEEK_API_KEY",
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
        "tags": ["cheapest-google"],
    },
    {
        "name": "gpt-5-mini",
        "litellm_id": "gpt-5-mini",
        "provider": "openai",
        "api_key_env": "OPENAI_API_KEY",
        "intelligence": 41,
        "speed": 127,
        "cost": 0.69,
        "context": 128_000,
        "structured_output": True,
        "tags": ["reliable-structured"],
    },
    {
        "name": "gpt-5",
        "litellm_id": "gpt-5",
        "provider": "openai",
        "api_key_env": "OPENAI_API_KEY",
        "intelligence": 45,
        "speed": 98,
        "cost": 3.44,
        "context": 128_000,
        "structured_output": True,
        "tags": ["frontier"],
    },
    {
        "name": "gpt-5-nano",
        "litellm_id": "gpt-5-nano",
        "provider": "openai",
        "api_key_env": "OPENAI_API_KEY",
        "intelligence": 27,
        "speed": 141,
        "cost": 0.14,
        "context": 128_000,
        "structured_output": True,
        "tags": ["cheapest-openai"],
    },
    {
        "name": "grok-4.1-fast",
        "litellm_id": "xai/grok-4.1-fast",
        "provider": "xai",
        "api_key_env": "XAI_API_KEY",
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
    import yaml  # lazy import — only needed if user has a config file

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


def get_model(task: str, *, available_only: bool = True) -> str:
    """Get the best model for a task category.

    Args:
        task: Task name (e.g., "extraction", "bulk_cheap", "synthesis")
        available_only: If True, only consider models whose API key env var
            is set in ``os.environ``. Set to False to see the theoretical best.

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

    # Sort by prefer keys
    candidates = _sort_by_prefer(candidates, profile.prefer)
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
    """Query performance stats from I/O logs.

    Reads ``calls.jsonl`` from the io_log data directory, groups by
    (task, model), and returns aggregate stats.

    Args:
        task: Filter to this task category (None = all).
        model: Filter to this model (None = all).
        days: Look back this many days.

    Returns:
        List of dicts with: task, model, call_count, total_cost,
        avg_latency_s, error_rate, avg_tokens.
    """
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

        # Parse timestamp
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
        total_cost = sum(r.get("cost") or 0 for r in records)
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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
