"""Task-based model selection helpers for downstream projects.

This gives projects one small happy path for model governance:

1. resolve a model from the shared task registry,
2. optionally honor an explicit override,
3. enforce deprecated-model blocking in strict lanes.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Literal

from pydantic import BaseModel, Field

from llm_client.core.models import get_model


class ResolvedModelSelection(BaseModel):
    """Resolved model selection metadata."""

    task: str
    model: str
    source: Literal["task", "override"]
    strict_models: bool = True


class ResolvedModelChain(BaseModel):
    """Resolved primary model plus any fallback models."""

    primary: ResolvedModelSelection
    fallback_models: list[str] = Field(default_factory=list)
    fallback_tasks: list[str] = Field(default_factory=list)


def resolve_model_selection(
    task: str,
    *,
    override_model: str | None = None,
    strict_models: bool = True,
    available_only: bool = False,
    use_performance: bool = True,
) -> ResolvedModelSelection:
    """Resolve a model for a task, optionally preserving an explicit override."""

    normalized_task = task.strip()
    if not normalized_task:
        raise ValueError("task must be non-empty")
    if override_model:
        return ResolvedModelSelection(
            task=normalized_task,
            model=override_model,
            source="override",
            strict_models=strict_models,
        )
    model = get_model(
        normalized_task,
        available_only=available_only,
        use_performance=use_performance,
    )
    return ResolvedModelSelection(
        task=normalized_task,
        model=model,
        source="task",
        strict_models=strict_models,
    )


def resolve_model_chain(
    task: str,
    *,
    fallback_tasks: list[str] | None = None,
    override_model: str | None = None,
    fallback_models: list[str] | None = None,
    strict_models: bool = True,
    available_only: bool = False,
    use_performance: bool = True,
) -> ResolvedModelChain:
    """Resolve a primary model plus deduplicated fallback models."""

    primary = resolve_model_selection(
        task,
        override_model=override_model,
        strict_models=strict_models,
        available_only=available_only,
        use_performance=use_performance,
    )
    deduped_fallbacks: list[str] = []
    seen_models = {primary.model}
    resolved_fallback_tasks: list[str] = []

    for fallback_task in fallback_tasks or []:
        selection = resolve_model_selection(
            fallback_task,
            strict_models=strict_models,
            available_only=available_only,
            use_performance=use_performance,
        )
        resolved_fallback_tasks.append(selection.task)
        if selection.model in seen_models:
            continue
        deduped_fallbacks.append(selection.model)
        seen_models.add(selection.model)

    for fallback_model in fallback_models or []:
        normalized = fallback_model.strip()
        if not normalized or normalized in seen_models:
            continue
        deduped_fallbacks.append(normalized)
        seen_models.add(normalized)

    return ResolvedModelChain(
        primary=primary,
        fallback_models=deduped_fallbacks,
        fallback_tasks=resolved_fallback_tasks,
    )


@contextmanager
def strict_model_policy(enabled: bool = True) -> Iterator[None]:
    """Temporarily set deprecated-model blocking for a call site."""

    previous = os.environ.get("LLM_CLIENT_STRICT_MODELS")
    if enabled:
        os.environ["LLM_CLIENT_STRICT_MODELS"] = "1"
    else:
        os.environ.pop("LLM_CLIENT_STRICT_MODELS", None)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("LLM_CLIENT_STRICT_MODELS", None)
        else:
            os.environ["LLM_CLIENT_STRICT_MODELS"] = previous
