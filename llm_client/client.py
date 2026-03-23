"""LLM client wrapping litellm + agent SDKs.

Fourteen functions (7 sync + 7 async), no class, no mutable state:
- call_llm / acall_llm: basic completion (+ agent SDK routing)
- call_llm_structured / acall_llm_structured: Pydantic extraction (instructor or Responses API)
- call_llm_with_tools / acall_llm_with_tools: tool/function calling
- call_llm_batch / acall_llm_batch: concurrent batch calls
- call_llm_structured_batch / acall_llm_structured_batch: concurrent structured batch
- stream_llm / astream_llm: streaming with retry/fallback
- stream_llm_with_tools / astream_llm_with_tools: streaming with tools

Features:
- Three-tier routing: Agent SDK → Responses API → Chat Completions
- Smart retry with jittered exponential backoff on transient errors,
  empty responses, and JSON parse failures
- Automatic Responses API routing for bare GPT-5 models
  (litellm.responses; when OpenRouter auto-routing is off)
- Agent SDK routing for "claude-code" and "codex" models
- Thinking model detection (Gemini 3/4 → budget_tokens: 0)
- Fallback models — automatic failover to secondary models
- Observability hooks (before_call, after_call, on_error)
- Response caching with sync and async cache protocols
- Fence stripping utility for manual JSON parsing
- Cost tracking via litellm.completion_cost
- finish_reason + raw_response on every result

Supported providers (just change the model string):
    call_llm("gpt-4o", messages)                     # OpenAI
    call_llm("gpt-5-mini", messages)                 # OpenAI (Responses API)
    call_llm("anthropic/claude-sonnet-4-5-20250929", messages)  # Anthropic
    call_llm("gemini/gemini-2.0-flash", messages)     # Google
    call_llm("mistral/mistral-large", messages)       # Mistral
    call_llm("ollama/llama3", messages)               # Local Ollama
    call_llm("bedrock/anthropic.claude-v2", messages)  # AWS Bedrock
    call_llm("claude-code", messages)                 # Claude Agent SDK
    call_llm("claude-code/opus", messages)            # Claude Agent SDK (specific model)
    call_llm("codex", messages)                       # Codex SDK
    call_llm("codex/gpt-5", messages)                 # Codex SDK (specific model)

Full provider list: https://docs.litellm.ai/docs/providers
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import os
import re
import time
import urllib.parse
from typing import Any, Callable, Iterable, TypeVar

import litellm
from pydantic import BaseModel

from llm_client.config import ClientConfig
from llm_client import io_log as _io_log
from llm_client import rate_limit as _rate_limit
from llm_client.call_contracts import (
    AGENT_RETRY_SAFE_ENV,
    ExecutionMode,
    _AGENT_ONLY_KWARGS,
    _CODEX_AGENT_ALIASES,
    _DEPRECATED_MODEL_EXCEPTIONS,
    _DEPRECATED_MODELS,
    _GPT5_ALWAYS_STRIP_SAMPLING,
    _GPT5_REASONING_GATED_SAMPLING,
    _GPT5_SAMPLING_PARAMS,
    _LONG_THINKING_MODELS,
    _LONG_THINKING_REASONING_EFFORTS,
    _NativeSchemaFallback,
    _SCHEMA_ERROR_PATTERNS,
    _UNSUPPORTED_PARAM_POLICIES,
    _UNSUPPORTED_PARAM_POLICY_ALIASES,
    _UNSUPPORTED_PARAM_POLICY_ENV,
    _VALID_EXECUTION_MODES,
    _WARNED_MODELS,
    _apply_max_tokens,
    _check_model_deprecation,
    _coerce_model_incompatible_params,
    _coerce_model_kwargs_for_execution,
    _compact_diagnostics,
    _is_agent_model,
    _is_schema_error,
    _raise_empty_response,
    _resolve_unsupported_param_policy,
    _strip_incompatible_sampling_params,
    _strip_llm_internal_kwargs,
    _validate_execution_contract,
    agent_retry_safe_enabled as _agent_retry_safe_enabled,
    check_budget as _check_budget,
    normalize_prompt_ref as _normalize_prompt_ref,
    require_tags as _require_tags,
)
from llm_client.result_metadata import (
    build_routing_trace as _build_routing_trace_base,
    warning_record as _warning_record,
)
from llm_client.result_finalization import (
    finalize_result as _finalize_result_base,
)
from llm_client.timeout_policy import (
    normalize_timeout as _normalize_timeout,
)
from llm_client.routing import (
    CallRequest,
    ResolvedCallPlan,
    resolve_api_base_for_model,
    resolve_call,
)
from llm_client.execution_kernel import (
    run_async_with_fallback,
    run_async_with_retry,
    run_sync_with_fallback,
    run_sync_with_retry,
)

from llm_client.errors import (
    LLMError,
    LLMCapabilityError,
    LLMConfigurationError,
    LLMEmptyResponseError,
    LLMModelNotFoundError,
    wrap_error,
)

# Re-export data types for backward compatibility — all downstream modules
# that import these from llm_client.client continue to work unchanged.
from llm_client.data_types import (  # noqa: F401
    AsyncCachePolicy,
    CachePolicy,
    EmbeddingResult,
    LLMCallResult,
    LRUCache,
    _async_cache_get,
    _async_cache_set,
    _cache_key,
)

# Re-export retry infrastructure for backward compatibility.
from llm_client.retry import (  # noqa: F401
    Hooks,
    RetryPolicy,
    _NON_RETRYABLE_PATTERNS,
    _RETRYABLE_PATTERNS,
    _EMPTY_POLICY_FINISH_REASONS,
    _EMPTY_TOOL_PROTOCOL_FINISH_REASONS,
    _calculate_backoff,
    _check_retryable,
    _coerce_retry_delay_seconds,
    _compute_retry_delay,
    _effective_retry,
    _error_status_code,
    _error_text,
    _is_retryable,
    _retry_delay_hint,
    _retry_delay_hint_seconds,
    exponential_backoff,
    fixed_backoff,
    linear_backoff,
)

# Re-export OpenRouter utilities for backward compatibility.
from llm_client.openrouter import (  # noqa: F401
    _is_openrouter_call,
    _is_openrouter_key_limit_error,
    _mask_api_key,
    _maybe_retry_with_openrouter_key_rotation,
    _normalize_api_key_value,
    _openrouter_key_candidates_from_env,
    _openrouter_routing_enabled,
    _reset_openrouter_key_rotation_state,
    _rotate_openrouter_api_key,
    _split_api_keys,
)

# Re-export streaming classes for backward compatibility.
from llm_client.streaming import (  # noqa: F401
    AsyncLLMStream,
    LLMStream,
)

# Re-export model detection utilities for backward compatibility.
from llm_client.model_detection import (  # noqa: F401
    _base_model_name,
    _is_claude_model,
    _is_gemini_model,
    _is_image_generation_model,
    _is_responses_api_model,
    _is_thinking_model,
    _normalize_model_for_routing,
    _resolve_api_base_for_model,
)

# Re-export cost/usage utilities for backward compatibility.
from llm_client.cost_utils import (  # noqa: F401
    FALLBACK_COST_FLOOR_USD_PER_TOKEN,  # noqa: F811
    _compute_cost,
    _extract_tool_calls,
    _extract_usage,
    _parse_cost_result,
)
from llm_client.call_lifecycle import (  # noqa: F401
    _AsyncLLMCallHeartbeatMonitor,
    _LLMCallProgressReporter,
    _LLMCallProgressSnapshot,
    _SyncLLMCallHeartbeatMonitor,
    _emit_llm_call_lifecycle_event,
    _new_llm_call_lifecycle_id,
    _provider_timeout_for_lifecycle,
    _resolve_lifecycle_monitoring_settings,
)
from llm_client.call_wrappers import (
    _prepare_public_call_envelope,
    _run_async_public_call,
    _run_sync_public_call,
)
from llm_client.background_runtime import (
    _BACKGROUND_DEFAULT_TIMEOUT,
    _BACKGROUND_POLL_INTERVAL,
    _background_mode_for_model,
    _background_polling_config,
    _maybe_apoll_background_response_impl,
    _maybe_poll_background_response_impl,
    _needs_background_mode,
    _poll_background_response_impl,
    _retrieve_background_response_impl,
    _apoll_background_response_impl,
    _aretrieve_background_response_impl,
)
from llm_client.responses_runtime import (  # noqa: F401
    _build_result_from_responses,
    _compute_responses_cost,
    _convert_messages_to_input,
    _convert_response_format_for_responses,
    _convert_tools_for_responses_api,
    _extract_responses_usage,
    _prepare_responses_kwargs,
    _strict_json_schema,
)
from llm_client.completion_runtime import (  # noqa: F401
    _build_result_from_response,
    _first_choice_or_empty_error,
    _prepare_call_kwargs,
    _provider_hint_from_response,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Silence litellm's noisy default logging
litellm.suppress_debug_info = True

# Activate Langfuse callbacks if LITELLM_CALLBACKS=langfuse is set
from llm_client.langfuse_callbacks import configure_langfuse_callbacks, inject_metadata as _inject_langfuse_metadata

configure_langfuse_callbacks()

# Re-export OpenRouter constants from their canonical home.
from llm_client.openrouter import (  # noqa: F811
    OPENROUTER_API_BASE_ENV,
    OPENROUTER_API_KEY_ENV,
    OPENROUTER_API_KEYS_ENV,
    OPENROUTER_DEFAULT_API_BASE,
    OPENROUTER_ROUTING_ENV,
)
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


def _model_warning_record(requested_model: str) -> dict[str, Any] | None:
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
# Helpers
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
# Background polling for long-thinking models (gpt-5.2-pro xhigh etc.)
# ---------------------------------------------------------------------------


def _maybe_poll_background_response(
    response: Any,
    *,
    api_base: str | None,
    request_timeout: int | None,
    model_kwargs: dict[str, Any],
) -> Any:
    """Poll a non-terminal background response to completion when possible."""
    return _maybe_poll_background_response_impl(
        response,
        api_base=api_base,
        request_timeout=request_timeout,
        model_kwargs=model_kwargs,
        poll_response=_poll_background_response,
    )


async def _maybe_apoll_background_response(
    response: Any,
    *,
    api_base: str | None,
    request_timeout: int | None,
    model_kwargs: dict[str, Any],
) -> Any:
    """Async variant of background polling helper."""
    return await _maybe_apoll_background_response_impl(
        response,
        api_base=api_base,
        request_timeout=request_timeout,
        model_kwargs=model_kwargs,
        poll_response=_apoll_background_response,
    )


def _poll_background_response(
    response_id: str,
    *,
    api_base: str | None = None,
    request_timeout: int | None = None,
    poll_interval: int = _BACKGROUND_POLL_INTERVAL,
    timeout: int = _BACKGROUND_DEFAULT_TIMEOUT,
) -> Any:
    """Poll for a background Responses API response until completed.

    Synchronous polling loop. Calls litellm.responses.retrieve()
    at regular intervals until status is 'completed' or timeout.

    Args:
        response_id: The response ID returned from the initial background request.
        poll_interval: Seconds between poll attempts.
        timeout: Max total wait time in seconds.

    Returns:
        The completed response object.

    Raises:
        TimeoutError: If the response doesn't complete within timeout.
        RuntimeError: If the response fails or is cancelled.
    """
    return _poll_background_response_impl(
        response_id,
        api_base=api_base,
        request_timeout=request_timeout,
        poll_interval=poll_interval,
        timeout=timeout,
        retrieve_response=_retrieve_background_response,
    )


async def _apoll_background_response(
    response_id: str,
    *,
    api_base: str | None = None,
    request_timeout: int | None = None,
    poll_interval: int = _BACKGROUND_POLL_INTERVAL,
    timeout: int = _BACKGROUND_DEFAULT_TIMEOUT,
) -> Any:
    """Async version of _poll_background_response.

    Uses asyncio.sleep between polls to avoid blocking the event loop.
    """
    return await _apoll_background_response_impl(
        response_id,
        api_base=api_base,
        request_timeout=request_timeout,
        poll_interval=poll_interval,
        timeout=timeout,
        retrieve_response=_aretrieve_background_response,
    )


def _retrieve_background_response(
    *,
    response_id: str,
    api_base: str | None,
    request_timeout: int | None,
) -> Any:
    """Retrieve a background response by ID.

    LiteLLM currently exposes `responses()` as a function (no `.retrieve` attr) in
    this environment, so retrieval uses OpenAI SDK clients directly.
    """
    return _retrieve_background_response_impl(
        response_id=response_id,
        api_base=api_base,
        request_timeout=request_timeout,
    )


async def _aretrieve_background_response(
    *,
    response_id: str,
    api_base: str | None,
    request_timeout: int | None,
) -> Any:
    """Async retrieve for background responses by ID."""
    return await _aretrieve_background_response_impl(
        response_id=response_id,
        api_base=api_base,
        request_timeout=request_timeout,
    )


# ---------------------------------------------------------------------------
# Sync functions
# ---------------------------------------------------------------------------


def call_llm(
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout: int = 60,
    num_retries: int = 2,
    reasoning_effort: str | None = None,
    api_base: str | None = None,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_on: list[str] | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    cache: CachePolicy | None = None,
    retry: RetryPolicy | None = None,
    fallback_models: list[str] | None = None,
    on_fallback: Callable[[str, Exception, str], None] | None = None,
    hooks: Hooks | None = None,
    execution_mode: ExecutionMode = "text",
    config: ClientConfig | None = None,
    **kwargs: Any,
) -> LLMCallResult:
    """Call any LLM. Routes by model string: litellm, Responses API, or Agent SDK.

    Just change the model string to switch providers. Everything else
    stays the same. Three-tier routing:
    - "claude-code[/model]" → Claude Agent SDK
    - Bare "gpt-5*" → litellm.responses() (Responses API)
    - Everything else → litellm.completion()

    By default, OpenRouter normalization is enabled
    (``LLM_CLIENT_OPENROUTER_ROUTING=on``), so bare OpenAI/Anthropic model IDs
    are rewritten to ``openrouter/...`` and use completion routing.

    Retries up to num_retries times with jittered exponential backoff on
    transient errors (rate limits, timeouts, empty responses, JSON parse
    failures). Non-retryable errors raise immediately. Agent models
    default to 0 retries (side effects) unless explicit retry policy.

    If ``fallback_models`` is provided, when all retries are exhausted for
    one model the next model in the list is tried automatically.

    Args:
        model: Model name (e.g., "gpt-4o", "gpt-5-mini",
               "anthropic/claude-sonnet-4-5-20250929",
               "gemini/gemini-2.0-flash", "claude-code",
               "claude-code/opus")
        messages: Chat messages in OpenAI format
                  [{"role": "user", "content": "Hello"}]
        timeout: Request timeout in seconds
        num_retries: Number of retries on transient failure
        reasoning_effort: Reasoning effort level — only used for Claude models,
                         silently ignored for others
        api_base: Optional API base URL (e.g., for OpenRouter:
                  "https://openrouter.ai/api/v1")
        fallback_models: Models to try if the primary model fails all retries
        on_fallback: ``(failed_model, error, next_model)`` callback
        hooks: Observability hooks (before_call, after_call, on_error)
        execution_mode: Capability contract for this call:
            ``"text"`` (default), ``"structured"``, ``"workspace_agent"``,
            or ``"workspace_tools"``.
        **kwargs: Additional params passed to litellm.completion
                  (e.g., temperature, max_tokens, stream).
                  ``prompt_ref`` is reserved for llm_client observability.
                  ``lifecycle_heartbeat_interval_s`` and
                  ``lifecycle_stall_after_s`` are reserved for llm_client
                  liveness observability. None of these values are forwarded
                  to the provider. Heartbeats mean `llm_client` is still
                  waiting on the call; they do not imply token-level provider
                  progress.
                  For GPT-5 models, response_format is automatically
                  converted and max_tokens is stripped.
                  For agent models, agent-specific kwargs are extracted:
                  allowed_tools, cwd, max_turns, max_tool_calls, permission_mode,
                  max_budget_usd.

    Returns:
        LLMCallResult with content, usage, cost, model, tool_calls,
        finish_reason, and raw_response
    """
    from llm_client.text_runtime import _call_llm_impl

    envelope = _prepare_public_call_envelope(
        caller="call_llm",
        timeout=timeout,
        kwargs=kwargs,
    )
    return _run_sync_public_call(
        model=model,
        call_kind="text",
        caller="call_llm",
        timeout=timeout,
        envelope=envelope,
        invoke=lambda runtime_kwargs: _call_llm_impl(
            model,
            messages,
            timeout=timeout,
            num_retries=num_retries,
            reasoning_effort=reasoning_effort,
            api_base=api_base,
            base_delay=base_delay,
            max_delay=max_delay,
            retry_on=retry_on,
            on_retry=on_retry,
            cache=cache,
            retry=retry,
            fallback_models=fallback_models,
            on_fallback=on_fallback,
            hooks=hooks,
            execution_mode=execution_mode,
            config=config,
            **runtime_kwargs,
        ),
        resolve_model=lambda result: result.resolved_model or str(result.model or "") or None,
    )


def call_llm_structured(
    model: str,
    messages: list[dict[str, Any]],
    response_model: type[T],
    *,
    timeout: int = 60,
    num_retries: int = 2,
    reasoning_effort: str | None = None,
    api_base: str | None = None,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_on: list[str] | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    cache: CachePolicy | None = None,
    retry: RetryPolicy | None = None,
    fallback_models: list[str] | None = None,
    on_fallback: Callable[[str, Exception, str], None] | None = None,
    hooks: Hooks | None = None,
    config: ClientConfig | None = None,
    **kwargs: Any,
) -> tuple[T, LLMCallResult]:
    """Call LLM and get back a validated Pydantic model.

    Three-tier routing: GPT-5 uses Responses API, models supporting
    native JSON schema use response_format, others fall back to instructor.
    No manual JSON parsing needed.

    Args:
        model: Model name
        messages: Chat messages in OpenAI format
        response_model: Pydantic model class to extract
        timeout: Request timeout in seconds
        num_retries: Number of retries on failure
        reasoning_effort: Reasoning effort level (Claude models only)
        api_base: Optional API base URL (e.g., for OpenRouter)
        fallback_models: Models to try if the primary model fails all retries
        on_fallback: ``(failed_model, error, next_model)`` callback
        hooks: Observability hooks (before_call, after_call, on_error)
        **kwargs: Additional params passed to litellm.completion.
                  ``prompt_ref`` is reserved for llm_client observability.
                  ``lifecycle_heartbeat_interval_s`` and
                  ``lifecycle_stall_after_s`` are reserved for llm_client
                  liveness observability. None of these values are forwarded
                  to the provider. Heartbeats mean `llm_client` is still
                  waiting on the call; they do not imply token-level provider
                  progress.

    Returns:
        Tuple of (parsed Pydantic model instance, LLMCallResult)
    """
    from llm_client.structured_runtime import _call_llm_structured_impl

    envelope = _prepare_public_call_envelope(
        caller="call_llm_structured",
        timeout=timeout,
        kwargs=kwargs,
    )
    return _run_sync_public_call(
        model=model,
        call_kind="structured",
        caller="call_llm_structured",
        timeout=timeout,
        envelope=envelope,
        invoke=lambda runtime_kwargs: _call_llm_structured_impl(
            model,
            messages,
            response_model,
            timeout=timeout,
            num_retries=num_retries,
            reasoning_effort=reasoning_effort,
            api_base=api_base,
            base_delay=base_delay,
            max_delay=max_delay,
            retry_on=retry_on,
            on_retry=on_retry,
            cache=cache,
            retry=retry,
            fallback_models=fallback_models,
            on_fallback=on_fallback,
            hooks=hooks,
            config=config,
            **runtime_kwargs,
        ),
        resolve_model=lambda outcome: outcome[1].resolved_model or str(outcome[1].model or "") or None,
    )


def call_llm_with_tools(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    timeout: int = 60,
    num_retries: int = 2,
    reasoning_effort: str | None = None,
    api_base: str | None = None,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_on: list[str] | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    cache: CachePolicy | None = None,
    retry: RetryPolicy | None = None,
    fallback_models: list[str] | None = None,
    on_fallback: Callable[[str, Exception, str], None] | None = None,
    hooks: Hooks | None = None,
    execution_mode: ExecutionMode = "text",
    config: ClientConfig | None = None,
    **kwargs: Any,
) -> LLMCallResult:
    """Call LLM with tool/function calling support.

    Args:
        model: Model name
        messages: Chat messages in OpenAI format
        tools: Tool definitions in OpenAI format
        timeout: Request timeout in seconds
        num_retries: Number of retries on failure
        reasoning_effort: Reasoning effort level (Claude models only)
        api_base: Optional API base URL (e.g., for OpenRouter)
        fallback_models: Models to try if the primary model fails all retries
        on_fallback: ``(failed_model, error, next_model)`` callback
        hooks: Observability hooks (before_call, after_call, on_error)
        **kwargs: Additional params passed to litellm.completion

    Returns:
        LLMCallResult with tool_calls populated if model chose to use tools
    """
    if _is_agent_model(model):
        raise NotImplementedError(
            "Agent models have built-in tools. Use allowed_tools= to configure them."
        )
    return call_llm(
        model,
        messages,
        timeout=timeout,
        num_retries=num_retries,
        reasoning_effort=reasoning_effort,
        api_base=api_base,
        base_delay=base_delay,
        max_delay=max_delay,
        retry_on=retry_on,
        on_retry=on_retry,
        cache=cache,
        retry=retry,
        fallback_models=fallback_models,
        on_fallback=on_fallback,
        hooks=hooks,
        execution_mode=execution_mode,
        config=config,
        tools=tools,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Async variants
# ---------------------------------------------------------------------------


async def acall_llm(
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout: int = 60,
    num_retries: int = 2,
    reasoning_effort: str | None = None,
    api_base: str | None = None,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_on: list[str] | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    cache: CachePolicy | AsyncCachePolicy | None = None,
    retry: RetryPolicy | None = None,
    fallback_models: list[str] | None = None,
    on_fallback: Callable[[str, Exception, str], None] | None = None,
    hooks: Hooks | None = None,
    execution_mode: ExecutionMode = "text",
    config: ClientConfig | None = None,
    **kwargs: Any,
) -> LLMCallResult:
    """Async version of call_llm. Same three-tier routing (Agent SDK / Responses API / Completions).

    Accepts both sync ``CachePolicy`` and async ``AsyncCachePolicy`` caches.

    Args:
        model: Model name (e.g., "gpt-4o", "gpt-5-mini",
               "anthropic/claude-sonnet-4-5-20250929",
               "claude-code", "claude-code/opus")
        messages: Chat messages in OpenAI format
        timeout: Request timeout in seconds
        num_retries: Number of retries on transient failure
        reasoning_effort: Reasoning effort level (Claude models only)
        api_base: Optional API base URL (e.g., for OpenRouter)
        fallback_models: Models to try if the primary model fails all retries
        on_fallback: ``(failed_model, error, next_model)`` callback
        hooks: Observability hooks (before_call, after_call, on_error)
        execution_mode: Capability contract for this call:
            ``"text"`` (default), ``"structured"``, ``"workspace_agent"``,
            or ``"workspace_tools"``.
        **kwargs: Additional params passed to litellm.
                  ``prompt_ref`` is reserved for llm_client observability.
                  ``lifecycle_heartbeat_interval_s`` and
                  ``lifecycle_stall_after_s`` are reserved for llm_client
                  liveness observability. None of these values are forwarded
                  to the provider. Heartbeats mean `llm_client` is still
                  waiting on the call; they do not imply token-level provider
                  progress.

    Returns:
        LLMCallResult with content, usage, cost, model, tool_calls,
        finish_reason, and raw_response
    """
    from llm_client.text_runtime import _acall_llm_impl

    envelope = _prepare_public_call_envelope(
        caller="acall_llm",
        timeout=timeout,
        kwargs=kwargs,
    )
    return await _run_async_public_call(
        model=model,
        call_kind="text",
        caller="acall_llm",
        timeout=timeout,
        envelope=envelope,
        invoke=lambda runtime_kwargs: _acall_llm_impl(
            model,
            messages,
            timeout=timeout,
            num_retries=num_retries,
            reasoning_effort=reasoning_effort,
            api_base=api_base,
            base_delay=base_delay,
            max_delay=max_delay,
            retry_on=retry_on,
            on_retry=on_retry,
            cache=cache,
            retry=retry,
            fallback_models=fallback_models,
            on_fallback=on_fallback,
            hooks=hooks,
            execution_mode=execution_mode,
            config=config,
            **runtime_kwargs,
        ),
        resolve_model=lambda result: result.resolved_model or str(result.model or "") or None,
    )


async def acall_llm_structured(
    model: str,
    messages: list[dict[str, Any]],
    response_model: type[T],
    *,
    timeout: int = 60,
    num_retries: int = 2,
    reasoning_effort: str | None = None,
    api_base: str | None = None,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_on: list[str] | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    cache: CachePolicy | AsyncCachePolicy | None = None,
    retry: RetryPolicy | None = None,
    fallback_models: list[str] | None = None,
    on_fallback: Callable[[str, Exception, str], None] | None = None,
    hooks: Hooks | None = None,
    config: ClientConfig | None = None,
    **kwargs: Any,
) -> tuple[T, LLMCallResult]:
    """Async version of call_llm_structured.

    Accepts both sync ``CachePolicy`` and async ``AsyncCachePolicy`` caches.
    For GPT-5 models, bypasses instructor and uses the Responses API's
    native JSON schema support via ``litellm.aresponses()``.

    Args:
        model: Model name
        messages: Chat messages in OpenAI format
        response_model: Pydantic model class to extract
        timeout: Request timeout in seconds
        num_retries: Number of retries on failure
        reasoning_effort: Reasoning effort level (Claude models only)
        api_base: Optional API base URL (e.g., for OpenRouter)
        fallback_models: Models to try if the primary model fails all retries
        on_fallback: ``(failed_model, error, next_model)`` callback
        hooks: Observability hooks (before_call, after_call, on_error)
        **kwargs: Additional params passed to litellm.acompletion.
                  ``prompt_ref`` is reserved for llm_client observability.
                  ``lifecycle_heartbeat_interval_s`` and
                  ``lifecycle_stall_after_s`` are reserved for llm_client
                  liveness observability. None of these values are forwarded
                  to the provider. Heartbeats mean `llm_client` is still
                  waiting on the call; they do not imply token-level provider
                  progress.

    Returns:
        Tuple of (parsed Pydantic model instance, LLMCallResult)
    """
    from llm_client.structured_runtime import _acall_llm_structured_impl

    envelope = _prepare_public_call_envelope(
        caller="acall_llm_structured",
        timeout=timeout,
        kwargs=kwargs,
    )
    return await _run_async_public_call(
        model=model,
        call_kind="structured",
        caller="acall_llm_structured",
        timeout=timeout,
        envelope=envelope,
        invoke=lambda runtime_kwargs: _acall_llm_structured_impl(
            model,
            messages,
            response_model,
            timeout=timeout,
            num_retries=num_retries,
            reasoning_effort=reasoning_effort,
            api_base=api_base,
            base_delay=base_delay,
            max_delay=max_delay,
            retry_on=retry_on,
            on_retry=on_retry,
            cache=cache,
            retry=retry,
            fallback_models=fallback_models,
            on_fallback=on_fallback,
            hooks=hooks,
            config=config,
            **runtime_kwargs,
        ),
        resolve_model=lambda outcome: outcome[1].resolved_model or str(outcome[1].model or "") or None,
    )


async def acall_llm_with_tools(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    timeout: int = 60,
    num_retries: int = 2,
    reasoning_effort: str | None = None,
    api_base: str | None = None,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_on: list[str] | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    cache: CachePolicy | AsyncCachePolicy | None = None,
    retry: RetryPolicy | None = None,
    fallback_models: list[str] | None = None,
    on_fallback: Callable[[str, Exception, str], None] | None = None,
    hooks: Hooks | None = None,
    execution_mode: ExecutionMode = "text",
    config: ClientConfig | None = None,
    **kwargs: Any,
) -> LLMCallResult:
    """Async version of call_llm_with_tools.

    Args:
        model: Model name
        messages: Chat messages in OpenAI format
        tools: Tool definitions in OpenAI format
        timeout: Request timeout in seconds
        num_retries: Number of retries on failure
        reasoning_effort: Reasoning effort level (Claude models only)
        api_base: Optional API base URL (e.g., for OpenRouter)
        fallback_models: Models to try if the primary model fails all retries
        on_fallback: ``(failed_model, error, next_model)`` callback
        hooks: Observability hooks (before_call, after_call, on_error)
        **kwargs: Additional params passed to litellm.acompletion

    Returns:
        LLMCallResult with tool_calls populated if model chose to use tools
    """
    if _is_agent_model(model):
        raise NotImplementedError(
            "Agent models have built-in tools. Use allowed_tools= to configure them."
        )
    return await acall_llm(
        model,
        messages,
        timeout=timeout,
        num_retries=num_retries,
        reasoning_effort=reasoning_effort,
        api_base=api_base,
        base_delay=base_delay,
        max_delay=max_delay,
        retry_on=retry_on,
        on_retry=on_retry,
        cache=cache,
        retry=retry,
        fallback_models=fallback_models,
        on_fallback=on_fallback,
        hooks=hooks,
        execution_mode=execution_mode,
        config=config,
        tools=tools,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Batch/parallel functions
# ---------------------------------------------------------------------------


async def acall_llm_batch(
    model: str,
    messages_list: list[list[dict[str, Any]]],
    *,
    max_concurrent: int = 5,
    return_exceptions: bool = False,
    on_item_complete: Callable[[int, LLMCallResult], None] | None = None,
    on_item_error: Callable[[int, Exception], None] | None = None,
    timeout: int = 60,
    num_retries: int = 2,
    reasoning_effort: str | None = None,
    api_base: str | None = None,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_on: list[str] | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    cache: CachePolicy | AsyncCachePolicy | None = None,
    retry: RetryPolicy | None = None,
    fallback_models: list[str] | None = None,
    on_fallback: Callable[[str, Exception, str], None] | None = None,
    hooks: Hooks | None = None,
    config: ClientConfig | None = None,
    **kwargs: Any,
) -> list[LLMCallResult | Exception]:
    """Run multiple LLM calls concurrently with semaphore-based rate limiting.

    Each item delegates to ``acall_llm`` for full retry/fallback/cache/hooks.
    Results are returned in the same order as ``messages_list``.

    Args:
        model: Model name
        messages_list: List of message lists — one per call
        max_concurrent: Maximum concurrent requests (semaphore)
        return_exceptions: If True, exceptions are returned in the result list
            at the corresponding index instead of propagating
        on_item_complete: ``(index, result)`` callback per successful item
        on_item_error: ``(index, error)`` callback per failed item
        **kwargs: All standard params forwarded to ``acall_llm``

    Returns:
        List of LLMCallResult (or Exception if return_exceptions=True),
        in the same order as messages_list
    """
    from llm_client.batch_runtime import acall_llm_batch_impl

    return await acall_llm_batch_impl(
        model,
        messages_list,
        max_concurrent=max_concurrent,
        return_exceptions=return_exceptions,
        on_item_complete=on_item_complete,
        on_item_error=on_item_error,
        timeout=timeout,
        num_retries=num_retries,
        reasoning_effort=reasoning_effort,
        api_base=api_base,
        base_delay=base_delay,
        max_delay=max_delay,
        retry_on=retry_on,
        on_retry=on_retry,
        cache=cache,
        retry=retry,
        fallback_models=fallback_models,
        on_fallback=on_fallback,
        hooks=hooks,
        config=config,
        **kwargs,
    )


def call_llm_batch(
    model: str,
    messages_list: list[list[dict[str, Any]]],
    *,
    max_concurrent: int = 5,
    return_exceptions: bool = False,
    on_item_complete: Callable[[int, LLMCallResult], None] | None = None,
    on_item_error: Callable[[int, Exception], None] | None = None,
    timeout: int = 60,
    num_retries: int = 2,
    reasoning_effort: str | None = None,
    api_base: str | None = None,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_on: list[str] | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    cache: CachePolicy | None = None,
    retry: RetryPolicy | None = None,
    fallback_models: list[str] | None = None,
    on_fallback: Callable[[str, Exception, str], None] | None = None,
    hooks: Hooks | None = None,
    config: ClientConfig | None = None,
    **kwargs: Any,
) -> list[LLMCallResult | Exception]:
    """Sync wrapper for :func:`acall_llm_batch`.

    Runs the async batch in a new event loop. If called from within a
    running event loop (e.g., Jupyter), uses a thread to avoid nested
    event loop errors.

    See :func:`acall_llm_batch` for full parameter documentation.
    """
    from llm_client.batch_runtime import call_llm_batch_impl

    return call_llm_batch_impl(
        model,
        messages_list,
        max_concurrent=max_concurrent,
        return_exceptions=return_exceptions,
        on_item_complete=on_item_complete,
        on_item_error=on_item_error,
        timeout=timeout,
        num_retries=num_retries,
        reasoning_effort=reasoning_effort,
        api_base=api_base,
        base_delay=base_delay,
        max_delay=max_delay,
        retry_on=retry_on,
        on_retry=on_retry,
        cache=cache,
        retry=retry,
        fallback_models=fallback_models,
        on_fallback=on_fallback,
        hooks=hooks,
        config=config,
        **kwargs,
    )


async def acall_llm_structured_batch(
    model: str,
    messages_list: list[list[dict[str, Any]]],
    response_model: type[T],
    *,
    max_concurrent: int = 5,
    return_exceptions: bool = False,
    on_item_complete: Callable[[int, tuple[T, LLMCallResult]], None] | None = None,
    on_item_error: Callable[[int, Exception], None] | None = None,
    timeout: int = 60,
    num_retries: int = 2,
    reasoning_effort: str | None = None,
    api_base: str | None = None,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_on: list[str] | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    cache: CachePolicy | AsyncCachePolicy | None = None,
    retry: RetryPolicy | None = None,
    fallback_models: list[str] | None = None,
    on_fallback: Callable[[str, Exception, str], None] | None = None,
    hooks: Hooks | None = None,
    config: ClientConfig | None = None,
    **kwargs: Any,
) -> list[tuple[T, LLMCallResult] | Exception]:
    """Run multiple structured LLM calls concurrently.

    Each item delegates to ``acall_llm_structured``. See
    :func:`acall_llm_batch` for concurrency/callback semantics.

    Returns:
        List of (parsed_model, LLMCallResult) tuples (or Exception if
        return_exceptions=True), in input order.
    """
    from llm_client.batch_runtime import acall_llm_structured_batch_impl

    return await acall_llm_structured_batch_impl(
        model,
        messages_list,
        response_model,
        max_concurrent=max_concurrent,
        return_exceptions=return_exceptions,
        on_item_complete=on_item_complete,
        on_item_error=on_item_error,
        timeout=timeout,
        num_retries=num_retries,
        reasoning_effort=reasoning_effort,
        api_base=api_base,
        base_delay=base_delay,
        max_delay=max_delay,
        retry_on=retry_on,
        on_retry=on_retry,
        cache=cache,
        retry=retry,
        fallback_models=fallback_models,
        on_fallback=on_fallback,
        hooks=hooks,
        config=config,
        **kwargs,
    )


def call_llm_structured_batch(
    model: str,
    messages_list: list[list[dict[str, Any]]],
    response_model: type[T],
    *,
    max_concurrent: int = 5,
    return_exceptions: bool = False,
    on_item_complete: Callable[[int, tuple[T, LLMCallResult]], None] | None = None,
    on_item_error: Callable[[int, Exception], None] | None = None,
    timeout: int = 60,
    num_retries: int = 2,
    reasoning_effort: str | None = None,
    api_base: str | None = None,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_on: list[str] | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    cache: CachePolicy | None = None,
    retry: RetryPolicy | None = None,
    fallback_models: list[str] | None = None,
    on_fallback: Callable[[str, Exception, str], None] | None = None,
    hooks: Hooks | None = None,
    config: ClientConfig | None = None,
    **kwargs: Any,
) -> list[tuple[T, LLMCallResult] | Exception]:
    """Sync wrapper for :func:`acall_llm_structured_batch`.

    See :func:`acall_llm_batch` for concurrency semantics.
    """
    from llm_client.batch_runtime import call_llm_structured_batch_impl

    return call_llm_structured_batch_impl(
        model,
        messages_list,
        response_model,
        max_concurrent=max_concurrent,
        return_exceptions=return_exceptions,
        on_item_complete=on_item_complete,
        on_item_error=on_item_error,
        timeout=timeout,
        num_retries=num_retries,
        reasoning_effort=reasoning_effort,
        api_base=api_base,
        base_delay=base_delay,
        max_delay=max_delay,
        retry_on=retry_on,
        on_retry=on_retry,
        cache=cache,
        retry=retry,
        fallback_models=fallback_models,
        on_fallback=on_fallback,
        hooks=hooks,
        config=config,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Streaming functions
# ---------------------------------------------------------------------------


def stream_llm(
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout: int = 60,
    num_retries: int = 2,
    reasoning_effort: str | None = None,
    api_base: str | None = None,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_on: list[str] | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    retry: RetryPolicy | None = None,
    fallback_models: list[str] | None = None,
    on_fallback: Callable[[str, Exception, str], None] | None = None,
    hooks: Hooks | None = None,
    config: ClientConfig | None = None,
    **kwargs: Any,
) -> LLMStream:
    """Stream an LLM response, yielding text chunks as they arrive.

    Retries on **pre-stream** errors (rate limits, connection errors) with
    the same backoff logic as :func:`call_llm`. If the stream creation
    succeeds, errors during chunk consumption are not retried (that would
    require buffering, defeating streaming's purpose).

    Supports ``fallback_models`` — if the primary model exhausts retries,
    the next model in the list is tried.

    Example::

        stream = stream_llm("gpt-4o", messages)
        for chunk in stream:
            print(chunk, end="", flush=True)
        print()
        print(stream.result.usage)

    Args:
        model: Model name
        messages: Chat messages in OpenAI format
        timeout: Request timeout in seconds
        num_retries: Number of retries on pre-stream failure
        reasoning_effort: Reasoning effort level (Claude models only)
        api_base: Optional API base URL
        retry: Reusable RetryPolicy (overrides individual retry params)
        fallback_models: Models to try if the primary model fails all retries
        on_fallback: ``(failed_model, error, next_model)`` callback
        hooks: Observability hooks (before_call, after_call, on_error)
        **kwargs: Additional params passed to litellm.completion.
                  ``prompt_ref`` is reserved for llm_client observability and
                  is not forwarded to the provider.

    Returns:
        LLMStream that yields text chunks and exposes ``.result``
    """
    from llm_client.stream_runtime import stream_llm_impl

    return stream_llm_impl(
        model,
        messages,
        timeout=timeout,
        num_retries=num_retries,
        reasoning_effort=reasoning_effort,
        api_base=api_base,
        base_delay=base_delay,
        max_delay=max_delay,
        retry_on=retry_on,
        on_retry=on_retry,
        retry=retry,
        fallback_models=fallback_models,
        on_fallback=on_fallback,
        hooks=hooks,
        config=config,
        **kwargs,
    )


async def astream_llm(
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout: int = 60,
    num_retries: int = 2,
    reasoning_effort: str | None = None,
    api_base: str | None = None,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_on: list[str] | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    retry: RetryPolicy | None = None,
    fallback_models: list[str] | None = None,
    on_fallback: Callable[[str, Exception, str], None] | None = None,
    hooks: Hooks | None = None,
    config: ClientConfig | None = None,
    **kwargs: Any,
) -> AsyncLLMStream:
    """Async version of :func:`stream_llm` with retry/fallback support.

    Retries on pre-stream errors only. See :func:`stream_llm` for details.

    Returns:
        AsyncLLMStream that yields text chunks and exposes ``.result``
    """
    from llm_client.stream_runtime import astream_llm_impl

    return await astream_llm_impl(
        model,
        messages,
        timeout=timeout,
        num_retries=num_retries,
        reasoning_effort=reasoning_effort,
        api_base=api_base,
        base_delay=base_delay,
        max_delay=max_delay,
        retry_on=retry_on,
        on_retry=on_retry,
        retry=retry,
        fallback_models=fallback_models,
        on_fallback=on_fallback,
        hooks=hooks,
        config=config,
        **kwargs,
    )


def stream_llm_with_tools(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    timeout: int = 60,
    num_retries: int = 2,
    reasoning_effort: str | None = None,
    api_base: str | None = None,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_on: list[str] | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    retry: RetryPolicy | None = None,
    fallback_models: list[str] | None = None,
    on_fallback: Callable[[str, Exception, str], None] | None = None,
    hooks: Hooks | None = None,
    config: ClientConfig | None = None,
    **kwargs: Any,
) -> LLMStream:
    """Stream an LLM response with tool/function calling support.

    Same as :func:`stream_llm` but passes ``tools`` to the model. After
    consuming the stream, ``stream.result.tool_calls`` contains any tool
    calls the model made.

    Args:
        model: Model name
        messages: Chat messages in OpenAI format
        tools: Tool definitions in OpenAI format
        **kwargs: All other params forwarded to :func:`stream_llm`

    Returns:
        LLMStream with tool_calls available on ``.result`` after consumption
    """
    if _is_agent_model(model):
        raise NotImplementedError(
            "Agent models have built-in tools. Use allowed_tools= to configure them."
        )
    return stream_llm(
        model, messages,
        timeout=timeout,
        num_retries=num_retries,
        reasoning_effort=reasoning_effort,
        api_base=api_base,
        base_delay=base_delay,
        max_delay=max_delay,
        retry_on=retry_on,
        on_retry=on_retry,
        retry=retry,
        fallback_models=fallback_models,
        on_fallback=on_fallback,
        hooks=hooks,
        config=config,
        tools=tools,
        **kwargs,
    )


async def astream_llm_with_tools(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    timeout: int = 60,
    num_retries: int = 2,
    reasoning_effort: str | None = None,
    api_base: str | None = None,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_on: list[str] | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    retry: RetryPolicy | None = None,
    fallback_models: list[str] | None = None,
    on_fallback: Callable[[str, Exception, str], None] | None = None,
    hooks: Hooks | None = None,
    config: ClientConfig | None = None,
    **kwargs: Any,
) -> AsyncLLMStream:
    """Async version of :func:`stream_llm_with_tools`.

    Returns:
        AsyncLLMStream with tool_calls available on ``.result`` after consumption
    """
    if _is_agent_model(model):
        raise NotImplementedError(
            "Agent models have built-in tools. Use allowed_tools= to configure them."
        )
    return await astream_llm(
        model, messages,
        timeout=timeout,
        num_retries=num_retries,
        reasoning_effort=reasoning_effort,
        api_base=api_base,
        base_delay=base_delay,
        max_delay=max_delay,
        retry_on=retry_on,
        on_retry=on_retry,
        retry=retry,
        fallback_models=fallback_models,
        on_fallback=on_fallback,
        hooks=hooks,
        config=config,
        tools=tools,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def embed(
    model: str,
    input: str | list[str],
    *,
    dimensions: int | None = None,
    timeout: int = 60,
    api_base: str | None = None,
    api_key: str | None = None,
    task: str | None = None,
    trace_id: str | None = None,
    **kwargs: Any,
) -> EmbeddingResult:
    """Generate embeddings for text input(s).

    Wraps litellm.embedding() for provider-agnostic embedding generation.
    Swap models by changing the model string — same interface for OpenAI,
    Cohere, Bedrock, etc.

    Args:
        model: Embedding model (e.g., "text-embedding-3-small",
               "text-embedding-3-large", "cohere/embed-english-v3.0")
        input: Single string or list of strings to embed
        dimensions: Optional output dimensions (for models that support it,
                    e.g., text-embedding-3-small supports 256/512/1536)
        timeout: Request timeout in seconds
        api_base: Optional API base URL
        api_key: Optional API key override
        task: Optional task tag for io_log tracking
        **kwargs: Additional params passed to litellm.embedding

    Returns:
        EmbeddingResult with embeddings list, usage, and cost
    """
    from llm_client.embedding_runtime import embed_impl

    return embed_impl(
        model,
        input,
        dimensions=dimensions,
        timeout=timeout,
        api_base=api_base,
        api_key=api_key,
        task=task,
        trace_id=trace_id,
        **kwargs,
    )


async def aembed(
    model: str,
    input: str | list[str],
    *,
    dimensions: int | None = None,
    timeout: int = 60,
    api_base: str | None = None,
    api_key: str | None = None,
    task: str | None = None,
    trace_id: str | None = None,
    **kwargs: Any,
) -> EmbeddingResult:
    """Async version of embed(). See embed() for full docs."""
    from llm_client.embedding_runtime import aembed_impl

    return await aembed_impl(
        model,
        input,
        dimensions=dimensions,
        timeout=timeout,
        api_base=api_base,
        api_key=api_key,
        task=task,
        trace_id=trace_id,
        **kwargs,
    )
