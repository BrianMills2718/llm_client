"""Client dispatch, routing, and result-finalization helpers.

This module owns the internal orchestration helpers that sit between
the public ``client.py`` entrypoints and the per-transport runtime modules:

- routing plan resolution and policy labelling,
- result finalization with deprecation-warning injection,
- structured-call result building,
- agent-loop kwargs splitting and result finalization,
- observability call-event logging,
- text/schema utility helpers (``strip_fences``, ``_as_text_content``,
  ``_clean_schema_for_gemini``).

``client.py`` re-exports every name so existing callers (``text_runtime``,
``structured_runtime``, ``stream_runtime``, ``streaming``, etc.) that
access these via ``_client.xxx`` continue to work unchanged.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable, Iterable

from pydantic import BaseModel

from llm_client import io_log as _io_log
from llm_client.call_contracts import (
    ExecutionMode,
    _DEPRECATED_MODEL_EXCEPTIONS,
    _DEPRECATED_MODELS,
    _WARNED_MODELS,
)
from llm_client.core.config import ClientConfig
from llm_client.core.data_types import LLMCallResult
from llm_client.core.model_detection import _resolve_api_base_for_model
from llm_client.openrouter import _openrouter_routing_enabled
from llm_client.result_finalization import finalize_result as _finalize_result_base
from llm_client.result_metadata import (
    build_routing_trace as _build_routing_trace_base,
    warning_record as _warning_record,
)
from llm_client.retry import Hooks, RetryPolicy
from llm_client.core.routing import (
    CallRequest,
    ResolvedCallPlan,
    resolve_call,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing plan and policy
# ---------------------------------------------------------------------------


def _routing_policy_label(config: ClientConfig | None = None) -> str:
    """Return a stable routing-policy label for result tracing."""
    if config is not None:
        return "openrouter_on" if config.routing_policy == "openrouter" else "openrouter_off"
    return "openrouter_on" if _openrouter_routing_enabled() else "openrouter_off"


def _build_routing_trace(
    *,
    requested_model: str,
    attempted_models: list[str] | None = None,
    selected_model: str | None = None,
    requested_api_base: str | None = None,
    effective_api_base: str | None = None,
    sticky_fallback: bool | None = None,
    background_mode: bool | None = None,
    routing_policy: str | None = None,
) -> dict[str, Any]:
    """Build a minimal routing trace for week-1 contract characterization."""
    return _build_routing_trace_base(
        requested_model=requested_model,
        attempted_models=attempted_models,
        selected_model=selected_model,
        requested_api_base=requested_api_base,
        effective_api_base=effective_api_base,
        sticky_fallback=sticky_fallback,
        background_mode=background_mode,
        routing_policy=routing_policy or _routing_policy_label(),
    )


def _resolve_call_plan(
    *,
    model: str,
    fallback_models: list[str] | None,
    api_base: str | None,
    config: ClientConfig | None = None,
) -> ResolvedCallPlan:
    """Resolve and log routing plan once per entrypoint."""
    cfg = config or ClientConfig.from_env()
    plan = resolve_call(
        CallRequest(model=model, fallback_models=fallback_models, api_base=api_base),
        cfg,
    )
    normalization_events = plan.routing_trace.get("normalization_events")
    if isinstance(normalization_events, list):
        for event in normalization_events:
            if not isinstance(event, dict):
                continue
            raw = str(event.get("from", "")).strip()
            normalized = str(event.get("to", "")).strip()
            if raw and normalized and raw != normalized:
                logger.info("ROUTE_MODEL: %s -> %s", raw, normalized)
    return plan


def _build_model_chain(
    model: str,
    fallback_models: list[str] | None,
    config: ClientConfig | None = None,
) -> list[str]:
    """Build primary+fallback model chain with stable de-duplication."""
    plan = _resolve_call_plan(
        model=model,
        fallback_models=fallback_models,
        api_base=None,
        config=config,
    )
    return plan.models


# ---------------------------------------------------------------------------
# Result finalization
# ---------------------------------------------------------------------------


def _model_warning_record(requested_model: str) -> dict[str, Any] | None:
    """Return a warning record if the model is deprecated/outclassed."""
    lower = str(requested_model or "").lower()
    for pattern in _DEPRECATED_MODELS:
        if pattern in lower and not any(
            exc in lower and exc != pattern
            for exc in _DEPRECATED_MODEL_EXCEPTIONS
        ):
            return _warning_record(
                code="LLMC_WARN_MODEL_DEPRECATED",
                category="DeprecationWarning",
                message=f"Model {requested_model} is deprecated/outclassed.",
                field_path="model",
            )
    for pattern in _WARNED_MODELS:
        if pattern in lower and not any(
            exc in lower and exc != pattern
            for exc in _DEPRECATED_MODEL_EXCEPTIONS
        ):
            return _warning_record(
                code="LLMC_WARN_MODEL_OUTCLASSED",
                category="UserWarning",
                message=f"Model {requested_model} is outclassed but still allowed.",
                field_path="model",
            )
    return None


def _finalize_result(
    result: LLMCallResult,
    *,
    requested_model: str,
    resolved_model: str | None = None,
    routing_trace: dict[str, Any] | None = None,
    warning_records: list[dict[str, Any]] | None = None,
    cache_hit: bool = False,
) -> LLMCallResult:
    """Finalize a completed result while keeping warning policy local."""
    extra_warning_records: list[dict[str, Any]] = [
        dict(record) for record in (warning_records or []) if isinstance(record, dict)
    ]
    model_warning = _model_warning_record(requested_model)
    if model_warning is not None:
        extra_warning_records.append(model_warning)
    return _finalize_result_base(
        result,
        requested_model=requested_model,
        resolved_model=resolved_model,
        routing_trace=routing_trace,
        warning_records=extra_warning_records or None,
        cache_hit=cache_hit,
    )


def _build_structured_call_result(
    *,
    parsed: BaseModel,
    usage: dict[str, Any],
    cost: float,
    cost_source: str,
    current_model: str,
    finish_reason: str,
    raw_response: Any,
    warnings: list[str],
    requested_model: str,
    attempted_models: list[str],
    requested_api_base: str | None,
    effective_api_base: str | None,
    background_mode: bool | None = None,
    routing_policy: str,
) -> LLMCallResult:
    """Build and identity-annotate a structured-call result consistently."""
    llm_result = LLMCallResult(
        content=str(parsed.model_dump_json()),
        usage=usage,
        cost=cost,
        model=current_model,
        finish_reason=finish_reason,
        raw_response=raw_response,
        warnings=warnings,
        cost_source=cost_source,
    )
    return _finalize_result(
        llm_result,
        requested_model=requested_model,
        resolved_model=current_model,
        routing_trace=_build_routing_trace(
            requested_model=requested_model,
            attempted_models=attempted_models,
            selected_model=current_model,
            requested_api_base=requested_api_base,
            effective_api_base=effective_api_base,
            background_mode=background_mode,
            routing_policy=routing_policy,
        ),
    )


# ---------------------------------------------------------------------------
# Agent-loop orchestration helpers
# ---------------------------------------------------------------------------


def _build_inner_named_call_kwargs(
    *,
    num_retries: int,
    base_delay: float,
    max_delay: float,
    retry_on: list[str] | None,
    on_retry: Callable[[int, Exception, float], None] | None,
    retry: RetryPolicy | None,
    fallback_models: list[str] | None,
    on_fallback: Callable[[str, Exception, str], None] | None,
    reasoning_effort: str | None,
    api_base: str | None,
    hooks: Hooks | None,
    execution_mode: ExecutionMode,
    config: ClientConfig,
) -> dict[str, Any]:
    """Build kwargs propagated into inner per-turn call paths."""
    inner_named: dict[str, Any] = {}
    if num_retries != 2:
        inner_named["num_retries"] = num_retries
    if base_delay != 1.0:
        inner_named["base_delay"] = base_delay
    if max_delay != 30.0:
        inner_named["max_delay"] = max_delay
    if retry_on is not None:
        inner_named["retry_on"] = retry_on
    if on_retry is not None:
        inner_named["on_retry"] = on_retry
    if retry is not None:
        inner_named["retry"] = retry
    if fallback_models is not None:
        inner_named["fallback_models"] = fallback_models
    if on_fallback is not None:
        inner_named["on_fallback"] = on_fallback
    if reasoning_effort is not None:
        inner_named["reasoning_effort"] = reasoning_effort
    if api_base is not None:
        inner_named["api_base"] = api_base
    if hooks is not None:
        inner_named["hooks"] = hooks
    if execution_mode != "text":
        inner_named["execution_mode"] = execution_mode
    inner_named["config"] = config
    return inner_named


def _split_agent_loop_kwargs(
    *,
    kwargs: dict[str, Any],
    loop_kwargs: Iterable[str],
    task: str | None,
    trace_id: str | None,
    max_budget: float | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split kwargs into loop-specific and remaining values with task metadata."""
    selected: dict[str, Any] = {}
    remaining = dict(kwargs)
    remaining["task"] = task
    remaining["trace_id"] = trace_id
    remaining["max_budget"] = max_budget
    for key in loop_kwargs:
        if key in remaining:
            selected[key] = remaining.pop(key)
    return selected, remaining


def _finalize_agent_loop_result(
    *,
    result: LLMCallResult,
    requested_model: str,
    primary_model: str,
    requested_api_base: str | None,
    config: ClientConfig,
    routing_policy: str,
    caller: str,
    messages: list[dict[str, Any]],
    log_started_at: float,
    task: str | None,
    trace_id: str | None,
    prompt_ref: str | None,
    call_snapshot: dict[str, Any] | None = None,
) -> LLMCallResult:
    """Attach identity/routing trace and emit final call log for loop results."""
    existing_trace = result.routing_trace if isinstance(result.routing_trace, dict) else {}
    existing_attempted = existing_trace.get("attempted_models")
    attempted_models = (
        [str(m).strip() for m in existing_attempted if isinstance(m, str) and str(m).strip()]
        if isinstance(existing_attempted, list)
        else []
    )
    if not attempted_models:
        attempted_models = [primary_model]

    existing_sticky = existing_trace.get("sticky_fallback")
    sticky_fallback = (
        bool(existing_sticky)
        if isinstance(existing_sticky, bool)
        else any("STICKY_FALLBACK" in w for w in (result.warnings or []))
    )
    existing_background = existing_trace.get("background_mode")
    background_mode = (
        bool(existing_background)
        if isinstance(existing_background, bool)
        else None
    )
    selected_model = (
        result.resolved_model
        or (str(result.model).strip() if isinstance(result.model, str) and str(result.model).strip() else None)
    )
    api_base_model = selected_model or primary_model

    finalized = _finalize_result(
        result,
        requested_model=requested_model,
        resolved_model=result.resolved_model,
        routing_trace=_build_routing_trace(
            requested_model=requested_model,
            attempted_models=attempted_models,
            selected_model=selected_model,
            requested_api_base=requested_api_base,
            effective_api_base=_resolve_api_base_for_model(api_base_model, requested_api_base, config),
            sticky_fallback=sticky_fallback,
            background_mode=background_mode,
            routing_policy=routing_policy,
        ),
    )
    _log_call_event(
        model=selected_model or primary_model,
        messages=messages,
        result=finalized,
        latency_s=time.monotonic() - log_started_at,
        caller=caller,
        task=task,
        trace_id=trace_id,
        prompt_ref=prompt_ref,
        call_snapshot=call_snapshot,
    )
    return finalized


def _log_call_event(
    *,
    model: str,
    messages: list[dict[str, Any]] | None = None,
    result: Any = None,
    error: Exception | None = None,
    latency_s: float | None = None,
    caller: str = "call_llm",
    task: str | None = None,
    trace_id: str | None = None,
    prompt_ref: str | None = None,
    call_snapshot: dict[str, Any] | None = None,
) -> None:
    """Write one observability record for an LLM call."""
    from llm_client.observability.replay import snapshot_fingerprint as _snapshot_fingerprint

    call_fingerprint = (
        _snapshot_fingerprint(call_snapshot) if call_snapshot is not None else None
    )
    _io_log.log_call(
        model=model,
        messages=messages,
        result=result,
        error=error,
        latency_s=latency_s,
        caller=caller,
        task=task,
        trace_id=trace_id,
        prompt_ref=prompt_ref,
        call_snapshot=call_snapshot,
        call_fingerprint=call_fingerprint,
    )


# ---------------------------------------------------------------------------
# Text / schema utilities
# ---------------------------------------------------------------------------


def strip_fences(content: str) -> str:
    """Strip markdown code fences from LLM response content.

    Useful when calling call_llm() and parsing JSON manually:
        result = call_llm("gpt-4o", messages)
        clean = strip_fences(result.content)
        data = json.loads(clean)
    """
    content = content.strip()
    content = re.sub(r"^```(?:json|python|xml|text)?\s*\n?", "", content)
    content = re.sub(r"\n?\s*```\s*$", "", content)
    return content.strip()


def _as_text_content(content: Any) -> str:
    """Best-effort conversion of OpenAI-style content into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                if item:
                    parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
                elif "content" in item and isinstance(item["content"], str) and item["content"]:
                    parts.append(item["content"])
                continue
            rendered = str(item)
            if rendered:
                parts.append(rendered)
        return "\n".join(parts)
    return str(content)


def _clean_schema_for_gemini(schema: dict[str, Any]) -> dict[str, Any]:
    """Clean a JSON schema for Gemini compatibility.

    Gemini's function declaration API doesn't support several OpenAI/JSON Schema
    features: additionalProperties, strict, $defs/$ref, anyOf with null (use
    nullable instead), and title fields. This recursively strips unsupported
    fields and converts anyOf-with-null to nullable.

    Used by both the litellm Gemini path (via mcp_agent) and structured output.
    """
    import copy
    schema = copy.deepcopy(schema)

    def _clean(node: Any) -> Any:
        if not isinstance(node, dict):
            return node
        # Remove unsupported top-level fields
        for key in ("additionalProperties", "strict", "title", "$schema"):
            node.pop(key, None)
        # Resolve $defs/$ref inline (simple single-level)
        defs = node.pop("$defs", None) or node.pop("definitions", None)
        if isinstance(defs, dict):
            _inline_refs(node, defs)
        # Convert anyOf-with-null to nullable
        any_of = node.get("anyOf")
        if isinstance(any_of, list) and len(any_of) == 2:
            non_null = [s for s in any_of if s != {"type": "null"}]
            null_present = len(non_null) < len(any_of)
            if null_present and len(non_null) == 1:
                merged = dict(non_null[0])
                merged["nullable"] = True
                node.pop("anyOf")
                node.update(merged)
        # Recurse into properties
        props = node.get("properties")
        if isinstance(props, dict):
            for k, v in props.items():
                props[k] = _clean(v)
        # Recurse into items
        items = node.get("items")
        if isinstance(items, dict):
            node["items"] = _clean(items)
        # Ensure object type has properties
        if node.get("type") == "object" and "properties" not in node:
            node["properties"] = {}
        return node

    def _inline_refs(node: Any, defs: dict[str, Any]) -> None:
        if isinstance(node, dict):
            ref = node.pop("$ref", None)
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                ref_name = ref.split("/")[-1]
                resolved = defs.get(ref_name, {})
                node.update(resolved)
            for v in node.values():
                _inline_refs(v, defs)
        elif isinstance(node, list):
            for item in node:
                _inline_refs(item, defs)

    _clean(schema)
    return schema
