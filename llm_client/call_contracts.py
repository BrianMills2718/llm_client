"""Pre-call contract helpers for ``llm_client``.

This module centralizes the invariants that must hold before any provider or
agent SDK dispatch happens:

1. every call has a resolved ``task``,
2. every call has a resolved ``trace_id``,
3. every call has a pre-flight budget check,
4. prompt asset provenance is validated before it enters observability,
5. retry-safety policy is derived consistently for agent SDK calls.

These checks belong to the runtime substrate itself, not to any one transport
backend. Keeping them in one small module makes the boundary easier to reason
about and easier to test without dragging the full client runtime with it.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

from llm_client import io_log as _io_log
from llm_client.errors import LLMBudgetExceededError
from llm_client.prompt_assets import parse_prompt_ref

REQUIRE_TAGS_ENV = "LLM_CLIENT_REQUIRE_TAGS"
AGENT_RETRY_SAFE_ENV = "LLM_CLIENT_AGENT_RETRY_SAFE"


def truthy_env(value: Any) -> bool:
    """Parse common truthy env-style values."""
    if isinstance(value, bool):
        return value
    raw = str(value or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def tags_strict_mode(task: str | None) -> bool:
    """Whether missing task/trace/budget tags should raise instead of defaulting."""
    if truthy_env(os.environ.get(REQUIRE_TAGS_ENV)):
        return True
    if truthy_env(os.environ.get("CI")):
        return True
    normalized_task = str(task or "").strip().lower()
    return normalized_task.startswith(("benchmark", "bench", "eval", "ci"))


def normalize_prompt_ref(prompt_ref: str | None) -> str | None:
    """Validate prompt asset identity before it enters observability."""
    if prompt_ref is None:
        return None
    normalized = str(prompt_ref).strip()
    if not normalized:
        raise ValueError("prompt_ref must not be empty when provided.")
    return parse_prompt_ref(normalized).prompt_ref


def require_tags(
    task: str | None,
    trace_id: str | None,
    max_budget: float | None,
    *,
    caller: str,
) -> tuple[str, str, float, list[str]]:
    """Resolve observability tags and enforce shared guardrails.

    In strict mode, missing values fail loudly. Outside strict mode, the
    substrate fills in conservative defaults and emits warnings so the call is
    still observable and queryable.
    """
    missing: list[str] = []
    if not task:
        missing.append("task")
    if not trace_id:
        missing.append("trace_id")
    if max_budget is None:
        missing.append("max_budget")

    if tags_strict_mode(task) and missing:
        raise ValueError(
            f"Missing required kwargs: {', '.join(missing)}. "
            "Strict tag enforcement is enabled "
            f"(set {REQUIRE_TAGS_ENV}=0 to disable outside CI/benchmark)."
        )

    resolved_task = str(task).strip() if task else "adhoc"
    resolved_trace_id = (
        str(trace_id).strip() if trace_id else f"auto/{caller}/{uuid.uuid4().hex[:12]}"
    )
    if max_budget is None:
        resolved_max_budget = 0.0
    else:
        try:
            resolved_max_budget = float(max_budget)
        except (TypeError, ValueError):
            raise ValueError(f"max_budget must be numeric, got {max_budget!r}") from None

    auto_warnings: list[str] = []
    if not task:
        auto_warnings.append("AUTO_TAG: task=adhoc")
    if not trace_id:
        auto_warnings.append(f"AUTO_TAG: trace_id={resolved_trace_id}")
    if max_budget is None:
        auto_warnings.append("AUTO_TAG: max_budget=0 (unlimited)")

    _io_log.enforce_feature_profile(resolved_task, caller="llm_client.client")
    _io_log.enforce_experiment_context(resolved_task, caller="llm_client.client")
    return resolved_task, resolved_trace_id, resolved_max_budget, auto_warnings


def check_budget(trace_id: str, max_budget: float) -> None:
    """Raise before dispatch if a trace has already spent its budget."""
    if max_budget <= 0:
        return
    spent = _io_log.get_cost(trace_id=trace_id)
    if spent >= max_budget:
        raise LLMBudgetExceededError(
            f"Budget exceeded for trace {trace_id}: "
            f"${spent:.4f} spent >= ${max_budget:.4f} limit"
        )


def agent_retry_safe_enabled(explicit: Any | None) -> bool:
    """Whether retries on agent SDK calls are allowed."""
    if explicit is not None:
        return truthy_env(explicit)
    return truthy_env(os.environ.get(AGENT_RETRY_SAFE_ENV))
