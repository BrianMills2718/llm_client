"""Legacy difficulty-tier model guidance for task-graph compatibility.

This module maps a coarse difficulty tier to the cheapest currently available
model recommendation for that tier. It remains part of `llm_client` because the
simple `task_graph` runner, analyzer/model-floor logic, and compatibility
imports still depend on it.

New project code should prefer task-based selection via `get_model(task)` and
config fields such as `selection_task`. The difficulty router is advisory, not
authoritative policy: it should not become a second evolving product-facing
model-governance system alongside the shared model registry.

Usage:
    from llm_client.difficulty import get_model_for_difficulty, DifficultyTier

    model = get_model_for_difficulty(2)  # → cheapest available tier-2 model
    model = get_model_for_difficulty(1)  # → local ollama if available, else cloud fallback
    model = get_model_for_difficulty(0)  # → None (scripted, no LLM needed)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import warnings
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DifficultyTier(BaseModel):
    """Definition of a difficulty tier."""

    tier: int
    description: str
    models: list[str]  # litellm_id, ordered by preference (cheapest first)
    agent: bool = False  # If True, uses agent SDK (not litellm completion)


# Default tier definitions. Each tier lists models cheapest-first.
# The router picks the first available one.
DEFAULT_TIERS: list[dict[str, Any]] = [
    {
        "tier": 0,
        "description": "Scripted — no LLM needed",
        "models": [],
    },
    {
        "tier": 1,
        "description": "Simple: formatting, template fill, structured extraction",
        "models": [
            "ollama/llama3.1",
            "openrouter/deepseek/deepseek-chat",
            "deepseek/deepseek-chat",
            "gpt-5-nano",
            "gemini/gemini-2.5-flash-lite",
        ],
    },
    {
        "tier": 2,
        "description": "Moderate: entity extraction, classification, analysis",
        "models": [
            "openrouter/deepseek/deepseek-chat",
            "gemini/gemini-2.5-flash",
            "deepseek/deepseek-chat",
            "gpt-5-mini",
            "gemini/gemini-3-flash",
        ],
    },
    {
        "tier": 3,
        "description": "Complex: multi-hop reasoning, synthesis, multi-tool",
        "models": [
            "anthropic/claude-sonnet-4-5-20250929",
            "gemini/gemini-3-flash",
            "gpt-5",
        ],
    },
    {
        "tier": 4,
        "description": "Agent: multi-step autonomous tool use, MCP composition",
        "models": [
            "codex",
            "claude-code",
        ],
        "agent": True,
    },
]

_SELECTION_ADVISORY_EMITTED: set[tuple[int, str, str]] = set()


def _warn_model_selection(
    *,
    tier: int,
    task_id: str | None,
    model: str,
    candidates: list[str],
) -> None:
    key = (tier, model, task_id or "")
    if key in _SELECTION_ADVISORY_EMITTED:
        return
    _SELECTION_ADVISORY_EMITTED.add(key)

    candidates_display = ", ".join(candidates) if candidates else model
    context = f"task {task_id!r}" if task_id else "unkeyed caller"
    msg = (
        f"\n{'=' * 72}\n"
        f"MODEL SELECTION ADVISORY for {context} (tier {tier})\n"
        f"{'=' * 72}\n"
        f"Suggested model: {model}\n"
        f"Candidate models: {candidates_display}\n"
        "This is model-selection guidance, not policy. Treat it as a recommendation "
        "and pin an explicit model in task config when required.\n"
    )
    msg += (
        "For safer operations, prefer explicit pinning and review alternatives "
        f"(including cheaper + higher-quality options from your model policy). "
        "For any model that is not your preferred default, explicitly confirm usage with the declarer before running.\n"
        f"{'=' * 72}\n"
    )
    logger.warning(msg)
    warnings.warn(msg, UserWarning, stacklevel=3)


def _is_ollama_available() -> bool:
    """Check if ollama is running locally."""
    if not shutil.which("ollama"):
        return False
    try:
        import subprocess

        proc = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _is_model_available(model: str) -> bool:
    """Check if a model is available (has API key or is local)."""
    if model.startswith("ollama/"):
        return _is_ollama_available()
    if model in ("codex", "claude-code") or model.startswith("codex/") or model.startswith("claude-code/"):
        return True  # Agent SDKs use their own auth
    # Map model prefix to env var
    prefix_to_env: dict[str, str] = {
        "openrouter/": "OPENROUTER_API_KEY",
        "deepseek/": "DEEPSEEK_API_KEY",
        "gemini/": "GEMINI_API_KEY",
        "gpt-": "OPENAI_API_KEY",
        "anthropic/": "ANTHROPIC_API_KEY",
        "xai/": "XAI_API_KEY",
    }
    for prefix, env_var in prefix_to_env.items():
        if model.startswith(prefix):
            return bool(os.environ.get(env_var))
    return False


_tiers_cache: list[DifficultyTier] | None = None


def _load_tiers() -> list[DifficultyTier]:
    """Load tier definitions. Returns cached list after first call."""
    global _tiers_cache  # noqa: PLW0603
    if _tiers_cache is not None:
        return _tiers_cache
    _tiers_cache = [DifficultyTier(**t) for t in DEFAULT_TIERS]
    return _tiers_cache


def _reset_tiers() -> None:
    """Reset cached tiers. For testing only."""
    global _tiers_cache  # noqa: PLW0603
    _tiers_cache = None


def get_model_for_difficulty(
    tier: int,
    *,
    override_model: str | None = None,
    available_only: bool = True,
    task_id: str | None = None,
) -> str | None:
    """Get a recommended available model for a difficulty tier.

    Args:
        tier: Difficulty level 0-4.
        override_model: If provided, bypass routing and return this model.
        available_only: If False, return first model even if API key is missing.
        task_id: Optional task id included in advisory warnings.

    Returns:
        litellm model ID string, or None for tier 0 (no LLM needed).

    Raises:
        ValueError: Invalid tier.
        RuntimeError: No available models for this tier.
    """
    if override_model:
        return override_model

    tiers = _load_tiers()
    tier_def = next((t for t in tiers if t.tier == tier), None)
    if tier_def is None:
        raise ValueError(f"Invalid difficulty tier: {tier}. Valid: 0-{len(tiers) - 1}")

    if not tier_def.models:
        return None  # Tier 0: scripted

    selected_model: str | None
    if not available_only:
        selected_model = tier_def.models[0]
    else:
        selected_model = None
        for model in tier_def.models:
            if _is_model_available(model):
                selected_model = model
                break
        if selected_model is None:
            raise RuntimeError(
                f"No available models for difficulty tier {tier} ({tier_def.description}). "
                f"Candidates: {tier_def.models}"
            )

    if selected_model is None:
        raise RuntimeError(
            f"No selected model for difficulty tier {tier} ({tier_def.description})."
        )
    _warn_model_selection(
        tier=tier,
        task_id=task_id,
        model=selected_model,
        candidates=tier_def.models,
    )
    return selected_model


def get_model_candidates_for_difficulty(
    tier: int,
    *,
    available_only: bool = True,
) -> list[str]:
    """Return ordered model candidates for a difficulty tier."""
    tiers = _load_tiers()
    tier_def = next((t for t in tiers if t.tier == tier), None)
    if tier_def is None:
        raise ValueError(f"Invalid difficulty tier: {tier}. Valid: 0-{len(tiers) - 1}")
    if not tier_def.models:
        return []
    if not available_only:
        return list(tier_def.models)
    return [model for model in tier_def.models if _is_model_available(model)]


# --- Model floors (cumulative learning) ---


def load_model_floors(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load model_floors.json — the system's learned knowledge about what works.

    Args:
        path: Path to model_floors.json. Defaults to
            ~/projects/data/task_graph/model_floors.json.

    Returns:
        Dict mapping task_id → {floor, ceiling, last_tested, runs}.
    """
    if path is None:
        path = Path.home() / "projects" / "data" / "task_graph" / "model_floors.json"
    else:
        path = Path(path)
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        return {}
    return cast(dict[str, dict[str, Any]], loaded)


def save_model_floors(
    floors: dict[str, dict[str, Any]],
    path: str | Path | None = None,
) -> None:
    """Save model_floors.json.

    Args:
        floors: Dict mapping task_id → {floor, ceiling, last_tested, runs}.
        path: Path to write. Defaults to
            ~/projects/data/task_graph/model_floors.json.
    """
    if path is None:
        path = Path.home() / "projects" / "data" / "task_graph" / "model_floors.json"
    else:
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(floors, indent=2, sort_keys=True) + "\n")


def get_effective_tier(
    task_id: str,
    declared_tier: int,
    floors: dict[str, dict[str, Any]] | None = None,
) -> int:
    """Get the effective difficulty tier, accounting for learned floors.

    If the model_floors data shows this task consistently succeeds at a lower
    tier, use that. Never auto-upgrade above declared — only downgrade.

    Args:
        task_id: The task identifier.
        declared_tier: The tier declared in the task graph YAML.
        floors: Model floors data. If None, loads from disk.

    Returns:
        Effective tier (may be lower than declared, never higher).
    """
    if floors is None:
        floors = load_model_floors()
    if task_id not in floors:
        return declared_tier
    floor_data = floors[task_id]
    floor = int(floor_data.get("floor", declared_tier))
    # Never go above declared tier (upgrades require human approval)
    return min(declared_tier, max(floor, 0))
