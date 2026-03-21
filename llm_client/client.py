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
import socket
import threading
import time
import urllib.parse
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, NoReturn, Protocol, TypeVar

import litellm
from pydantic import BaseModel

from llm_client.config import ClientConfig
from llm_client import io_log as _io_log
from llm_client import rate_limit as _rate_limit
from llm_client.call_contracts import (
    AGENT_RETRY_SAFE_ENV,
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
    timeout_policy_label as _timeout_policy_label,
)
from llm_client.foundation import (
    coerce_run_id as _coerce_foundation_run_id,
    new_event_id as _new_foundation_event_id,
    now_iso as _foundation_now_iso,
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

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class _LLMCallProgressSnapshot:
    """Capture truthful progress-observability state for one in-flight call."""

    progress_observable: bool
    progress_source: str | None
    progress_event_count: int
    last_progress_at_monotonic: float | None


class _LLMCallProgressReporter(Protocol):
    """Minimal contract for lifecycles that can observe real call progress."""

    def enable_progress_tracking(self, *, default_source: str | None = None) -> None:
        """Declare that the call path exposes truthful progress signals."""

    def mark_progress(self, *, source: str) -> None:
        """Record one observed unit of forward progress for the in-flight call."""

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
_CODEX_AGENT_ALIASES: frozenset[str] = frozenset({"codex-mini-latest"})
_LLM_CALL_RUNTIME_ACTOR_ID = "service:llm_client:call_runtime:1"
_LIFECYCLE_HEARTBEAT_INTERVAL_ENV = "LLM_CLIENT_LIFECYCLE_HEARTBEAT_INTERVAL_S"
_LIFECYCLE_STALL_AFTER_ENV = "LLM_CLIENT_LIFECYCLE_STALL_AFTER_S"
_DEFAULT_LIFECYCLE_HEARTBEAT_INTERVAL_S = 15.0
_DEFAULT_LIFECYCLE_STALL_AFTER_S = 300.0


def _process_host_name() -> str | None:
    """Return the current host name for same-host lifecycle correlation."""

    try:
        hostname = socket.gethostname().strip()
    except Exception:
        return None
    return hostname or None


def _linux_process_start_token(pid: int) -> str | None:
    """Return a Linux procfs start token for one process when available."""

    if pid <= 0:
        return None
    try:
        stat_text = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
    except OSError:
        return None
    _, _, remainder = stat_text.partition(") ")
    if not remainder:
        return None
    fields = remainder.split()
    if len(fields) <= 19:
        return None
    start_ticks = fields[19].strip()
    return f"linux-proc-start:{start_ticks}" if start_ticks else None


_PROCESS_HOST_NAME = _process_host_name()
_PROCESS_ID = os.getpid()
_PROCESS_START_TOKEN = _linux_process_start_token(_PROCESS_ID)


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


class _NativeSchemaFallback(Exception):
    """Signal native-schema rejection and trigger instructor fallback."""


def _compact_diagnostics(diagnostics: dict[str, Any], *, max_len: int = 600) -> str:
    """Render diagnostics dict into a bounded JSON string for errors/logging."""
    try:
        rendered = _json.dumps(diagnostics, sort_keys=True, ensure_ascii=True, default=str)
    except Exception:
        rendered = str(diagnostics)
    if len(rendered) <= max_len:
        return rendered
    return rendered[:max_len] + "...(truncated)"


def _raise_empty_response(
    *,
    provider: str,
    classification: str,
    retryable: bool,
    diagnostics: dict[str, Any],
) -> NoReturn:
    """Raise typed empty-response error with structured diagnostics."""
    payload = dict(diagnostics)
    payload["provider"] = provider
    payload["classification"] = classification
    payload["retryable"] = retryable
    message = (
        f"Empty content from LLM [{provider}:{classification} retryable={retryable}] "
        f"diagnostics={_compact_diagnostics(payload)}"
    )
    raise LLMEmptyResponseError(
        message,
        retryable=retryable,
        classification=classification,
        diagnostics=payload,
    )


# Patterns indicating the provider rejected the JSON schema itself (not a
# transient error).  When detected in the native JSON-schema path, the call
# falls back to the instructor path which prompts for JSON instead of
# enforcing via API-level schema constraints.
_SCHEMA_ERROR_PATTERNS: list[str] = [
    "nesting depth",
    "schema is invalid",
    "schema exceeds",
    "invalid schema",
    "unsupported schema",
    "schema too complex",
    "schema validation",
    "not a valid json schema",
    "response_format",
]


def _is_schema_error(error: Exception) -> bool:
    """Check if an error indicates the provider rejected the response schema."""
    error_str = str(error).lower()
    # Must be a 400-class error (BadRequest), not a transient/server error
    error_type = type(error).__name__.lower()
    is_bad_request = "badrequest" in error_type or "invalid_argument" in error_str or "400" in error_str
    if not is_bad_request:
        return False
    return any(p in error_str for p in _SCHEMA_ERROR_PATTERNS)





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


# ---------------------------------------------------------------------------
# Responses API helpers (GPT-5 models)
# ---------------------------------------------------------------------------
_GPT5_ALWAYS_STRIP_SAMPLING = {"gpt-5", "gpt-5-mini", "gpt-5-nano"}
_GPT5_REASONING_GATED_SAMPLING = {
    "gpt-5.1",
    "gpt-5.2",
    "gpt-5.2-pro",
    "gpt-5.1-chat-latest",
    "gpt-5.2-chat-latest",
}
# Models that support long-thinking (5-10 min) and need background polling
_LONG_THINKING_MODELS = {"gpt-5.2-pro"}
_LONG_THINKING_REASONING_EFFORTS = {"high", "xhigh"}
_BACKGROUND_POLL_INTERVAL = 15  # seconds between polls
_BACKGROUND_DEFAULT_TIMEOUT = 900  # 15 minutes
_GPT5_SAMPLING_PARAMS = ("temperature", "top_p", "logprobs", "top_logprobs")
_UNSUPPORTED_PARAM_POLICY_ENV = "LLM_CLIENT_UNSUPPORTED_PARAM_POLICY"
_UNSUPPORTED_PARAM_POLICIES = frozenset({"coerce_and_warn", "coerce_silent", "error"})
_UNSUPPORTED_PARAM_POLICY_ALIASES = {
    "warn": "coerce_and_warn",
    "coerce": "coerce_and_warn",
    "silent": "coerce_silent",
    "strict": "error",
    "raise": "error",
    "error_only": "error",
}

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
) -> None:
    """Write one observability record for an LLM call."""
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
    )


def _new_llm_call_lifecycle_id() -> str:
    """Return a stable correlation id shared across lifecycle events for one call."""

    return f"llmcall_{uuid.uuid4().hex}"


def _llm_call_lifecycle_session_id(call_id: str) -> str:
    """Derive a deterministic Foundation session id for one public call lifecycle."""

    suffix = call_id.removeprefix("llmcall_")
    return f"sess_{suffix}"


def _llm_lifecycle_error_message(error: Exception) -> str:
    """Return a non-empty error message for lifecycle failure records."""

    if isinstance(error, LLMError) and error.original is not None:
        error = error.original
    message = str(error).strip()
    if message:
        return message
    return error.__class__.__name__


def _llm_lifecycle_error_type(error: Exception) -> str:
    """Return the most informative error type for lifecycle failure records."""

    if isinstance(error, LLMError) and error.original is not None:
        return error.original.__class__.__name__
    return error.__class__.__name__


def _provider_timeout_for_lifecycle(timeout: Any) -> int:
    """Compute the effective provider-timeout value for lifecycle observability."""

    if _timeout_policy_label() == "ban":
        return 0
    try:
        parsed = int(timeout)
    except (TypeError, ValueError):
        return 0
    return max(parsed, 0)


def _normalize_lifecycle_seconds(value: Any, *, default: float) -> float:
    """Normalize lifecycle heartbeat/stall thresholds from caller or env values."""

    if value is None:
        return max(default, 0.0)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if parsed <= 0:
        return 0.0
    return parsed


def _resolve_lifecycle_monitoring_settings(
    *,
    heartbeat_interval: Any,
    stall_after: Any,
) -> tuple[float, float]:
    """Resolve public liveness settings without forwarding them to providers."""

    heartbeat_default = _normalize_lifecycle_seconds(
        os.environ.get(
            _LIFECYCLE_HEARTBEAT_INTERVAL_ENV,
            _DEFAULT_LIFECYCLE_HEARTBEAT_INTERVAL_S,
        ),
        default=_DEFAULT_LIFECYCLE_HEARTBEAT_INTERVAL_S,
    )
    stall_default = _normalize_lifecycle_seconds(
        os.environ.get(_LIFECYCLE_STALL_AFTER_ENV, _DEFAULT_LIFECYCLE_STALL_AFTER_S),
        default=_DEFAULT_LIFECYCLE_STALL_AFTER_S,
    )
    resolved_heartbeat = _normalize_lifecycle_seconds(
        heartbeat_interval,
        default=heartbeat_default,
    )
    resolved_stall = _normalize_lifecycle_seconds(
        stall_after,
        default=stall_default,
    )
    return resolved_heartbeat, resolved_stall


def _emit_llm_call_lifecycle_event(
    *,
    call_id: str,
    phase: Literal["started", "heartbeat", "progress", "stalled", "completed", "failed"],
    call_kind: Literal["text", "structured"],
    caller: str,
    task: str,
    trace_id: str,
    requested_model: str,
    provider_timeout_s: int,
    prompt_ref: str | None,
    resolved_model: str | None = None,
    latency_s: float | None = None,
    elapsed_s: float | None = None,
    heartbeat_interval_s: float | None = None,
    stall_after_s: float | None = None,
    progress_observable: bool | None = None,
    progress_source: str | None = None,
    progress_event_count: int | None = None,
    error: Exception | None = None,
) -> None:
    """Emit a Foundation-backed lifecycle event for one public LLM call."""

    params: dict[str, Any] = {
        "task": task,
        "trace_id": trace_id,
        "call_kind": call_kind,
    }
    if prompt_ref is not None:
        params["prompt_ref"] = prompt_ref
    if resolved_model is not None:
        params["resolved_model"] = resolved_model

    _io_log.log_foundation_event(
        event={
            "event_id": _new_foundation_event_id(),
            "event_type": "LLMCallLifecycle",
            "timestamp": _foundation_now_iso(),
            "run_id": _coerce_foundation_run_id(None, trace_id),
            "session_id": _llm_call_lifecycle_session_id(call_id),
            "actor_id": _LLM_CALL_RUNTIME_ACTOR_ID,
            "operation": {"name": caller, "version": None},
            "inputs": {
                "artifact_ids": [],
                "params": params,
                "bindings": {},
            },
            "outputs": {
                "artifact_ids": [],
                "payload_hashes": [],
            },
            "llm_call_lifecycle": {
                "call_id": call_id,
                "phase": phase,
                "call_kind": call_kind,
                "requested_model_id": requested_model,
                "resolved_model_id": resolved_model,
                "provider_timeout_s": provider_timeout_s if provider_timeout_s > 0 else None,
                "timeout_policy": _timeout_policy_label(),
                "prompt_ref": prompt_ref,
                "host_name": _PROCESS_HOST_NAME,
                "process_id": _PROCESS_ID if _PROCESS_ID > 0 else None,
                "process_start_token": _PROCESS_START_TOKEN,
                "progress_observable": progress_observable,
                "progress_source": progress_source,
                "progress_event_count": progress_event_count,
                "elapsed_s": elapsed_s,
                "latency_s": latency_s,
                "heartbeat_interval_s": (
                    heartbeat_interval_s
                    if heartbeat_interval_s and heartbeat_interval_s > 0
                    else None
                ),
                "stall_after_s": stall_after_s if stall_after_s and stall_after_s > 0 else None,
                "error_type": _llm_lifecycle_error_type(error) if error is not None else None,
                "error_message": _llm_lifecycle_error_message(error) if error is not None else None,
            },
        },
        caller=caller,
        task=task,
        trace_id=trace_id,
    )


class _SyncLLMCallHeartbeatMonitor:
    """Emit lifecycle updates for one sync call, including real progress when available."""

    def __init__(
        self,
        *,
        call_id: str,
        call_kind: Literal["text", "structured"],
        caller: str,
        task: str,
        trace_id: str,
        requested_model: str,
        provider_timeout_s: int,
        prompt_ref: str | None,
        heartbeat_interval_s: float,
        stall_after_s: float,
        started_at: float,
        progress_observable: bool = False,
    ) -> None:
        self.call_id = call_id
        self.call_kind = call_kind
        self.caller = caller
        self.task = task
        self.trace_id = trace_id
        self.requested_model = requested_model
        self.provider_timeout_s = provider_timeout_s
        self.prompt_ref = prompt_ref
        self.heartbeat_interval_s = heartbeat_interval_s
        self.stall_after_s = stall_after_s
        self.started_at = started_at
        self._state_lock = threading.Lock()
        self._progress_observable = progress_observable
        self._progress_source: str | None = None
        self._progress_event_count = 0
        self._last_progress_at_monotonic: float | None = None
        self._stalled_emitted = False
        self._last_progress_event_emitted_at: float | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background liveness monitor when thresholds are enabled."""

        if self.heartbeat_interval_s <= 0 and self.stall_after_s <= 0:
            return
        self._thread = threading.Thread(
            target=self._run,
            name=f"llm-call-heartbeat-{self.call_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the monitor and wait briefly for clean exit."""

        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=1.0)

    def enable_progress_tracking(self, *, default_source: str | None = None) -> None:
        """Declare that this call path exposes truthful observable progress."""

        with self._state_lock:
            self._progress_observable = True
            if default_source:
                self._progress_source = default_source

    def mark_progress(self, *, source: str) -> None:
        """Record one unit of observed progress and rate-limit event emission."""

        now = time.monotonic()
        with self._state_lock:
            self._progress_observable = True
            self._progress_source = source
            self._progress_event_count += 1
            self._last_progress_at_monotonic = now
            self._stalled_emitted = False
            should_emit = (
                self._last_progress_event_emitted_at is None
                or self.heartbeat_interval_s <= 0
                or (now - self._last_progress_event_emitted_at) >= self.heartbeat_interval_s
            )
            if should_emit:
                self._last_progress_event_emitted_at = now
            snapshot = self._snapshot_locked()
        if should_emit:
            self._emit_phase("progress", elapsed_s=now - self.started_at, snapshot=snapshot)

    def snapshot(self) -> _LLMCallProgressSnapshot:
        """Return the current progress-observability state for this call."""

        with self._state_lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> _LLMCallProgressSnapshot:
        """Build a progress snapshot while the internal state lock is held."""

        return _LLMCallProgressSnapshot(
            progress_observable=self._progress_observable,
            progress_source=self._progress_source,
            progress_event_count=self._progress_event_count,
            last_progress_at_monotonic=self._last_progress_at_monotonic,
        )

    def _emit_phase(
        self,
        phase: Literal["heartbeat", "progress", "stalled"],
        *,
        elapsed_s: float,
        snapshot: _LLMCallProgressSnapshot | None = None,
    ) -> None:
        """Emit one non-terminal lifecycle phase with the latest progress metadata."""

        effective_snapshot = snapshot or self.snapshot()
        _emit_llm_call_lifecycle_event(
            call_id=self.call_id,
            phase=phase,
            call_kind=self.call_kind,
            caller=self.caller,
            task=self.task,
            trace_id=self.trace_id,
            requested_model=self.requested_model,
            provider_timeout_s=self.provider_timeout_s,
            prompt_ref=self.prompt_ref,
            elapsed_s=elapsed_s,
            heartbeat_interval_s=self.heartbeat_interval_s,
            stall_after_s=self.stall_after_s,
            progress_observable=effective_snapshot.progress_observable,
            progress_source=effective_snapshot.progress_source,
            progress_event_count=effective_snapshot.progress_event_count,
        )

    def _next_stall_wait(
        self,
        *,
        now: float,
        snapshot: _LLMCallProgressSnapshot,
    ) -> float | None:
        """Return seconds until the next eligible stall event, if any."""

        if self.stall_after_s <= 0:
            return None
        with self._state_lock:
            stalled_emitted = self._stalled_emitted
        if stalled_emitted:
            return None
        if snapshot.progress_observable:
            last_progress = snapshot.last_progress_at_monotonic
            if last_progress is None:
                return None
            idle_for_s = now - last_progress
            return max(self.stall_after_s - idle_for_s, 0.001)
        elapsed_s = now - self.started_at
        return max(self.stall_after_s - elapsed_s, 0.001)

    def _should_emit_stalled(
        self,
        *,
        now: float,
        snapshot: _LLMCallProgressSnapshot,
    ) -> bool:
        """Return whether the current call state has crossed the stall threshold."""

        if self.stall_after_s <= 0:
            return False
        with self._state_lock:
            if self._stalled_emitted:
                return False
        if snapshot.progress_observable:
            last_progress = snapshot.last_progress_at_monotonic
            if last_progress is None:
                return False
            return (now - last_progress) >= self.stall_after_s
        return (now - self.started_at) >= self.stall_after_s

    def _mark_stalled_emitted(self) -> None:
        """Remember that the current idle period already emitted a stall marker."""

        with self._state_lock:
            self._stalled_emitted = True

    def _run(self) -> None:
        """Emit in-flight lifecycle markers until the wrapped call terminates."""

        next_heartbeat = self.heartbeat_interval_s if self.heartbeat_interval_s > 0 else None
        while True:
            now = time.monotonic()
            elapsed = now - self.started_at
            snapshot = self.snapshot()
            waits: list[float] = []
            if next_heartbeat is not None:
                waits.append(max(next_heartbeat - elapsed, 0.001))
            stall_wait = self._next_stall_wait(now=now, snapshot=snapshot)
            if stall_wait is not None:
                waits.append(stall_wait)
            if not waits:
                return
            if self._stop_event.wait(min(waits)):
                return
            now = time.monotonic()
            elapsed = now - self.started_at
            snapshot = self.snapshot()
            if next_heartbeat is not None and elapsed >= next_heartbeat:
                self._emit_phase("heartbeat", elapsed_s=elapsed, snapshot=snapshot)
                while next_heartbeat is not None and elapsed >= next_heartbeat:
                    next_heartbeat += self.heartbeat_interval_s
            if self._should_emit_stalled(now=now, snapshot=snapshot):
                self._emit_phase("stalled", elapsed_s=elapsed, snapshot=snapshot)
                self._mark_stalled_emitted()


class _AsyncLLMCallHeartbeatMonitor:
    """Emit lifecycle updates for one async call, including real progress when available."""

    def __init__(
        self,
        *,
        call_id: str,
        call_kind: Literal["text", "structured"],
        caller: str,
        task: str,
        trace_id: str,
        requested_model: str,
        provider_timeout_s: int,
        prompt_ref: str | None,
        heartbeat_interval_s: float,
        stall_after_s: float,
        started_at: float,
        progress_observable: bool = False,
    ) -> None:
        self.call_id = call_id
        self.call_kind = call_kind
        self.caller = caller
        self.task = task
        self.trace_id = trace_id
        self.requested_model = requested_model
        self.provider_timeout_s = provider_timeout_s
        self.prompt_ref = prompt_ref
        self.heartbeat_interval_s = heartbeat_interval_s
        self.stall_after_s = stall_after_s
        self.started_at = started_at
        self._state_lock = threading.Lock()
        self._progress_observable = progress_observable
        self._progress_source: str | None = None
        self._progress_event_count = 0
        self._last_progress_at_monotonic: float | None = None
        self._stalled_emitted = False
        self._last_progress_event_emitted_at: float | None = None
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Start the async heartbeat task when thresholds are enabled."""

        if self.heartbeat_interval_s <= 0 and self.stall_after_s <= 0:
            return
        self._task = asyncio.create_task(
            self._run(),
            name=f"llm-call-heartbeat-{self.call_id}",
        )

    async def stop(self) -> None:
        """Stop the heartbeat task and await clean exit."""

        if self._task is None:
            return
        self._stop_event.set()
        await self._task

    def enable_progress_tracking(self, *, default_source: str | None = None) -> None:
        """Declare that this async call path exposes truthful observable progress."""

        with self._state_lock:
            self._progress_observable = True
            if default_source:
                self._progress_source = default_source

    def mark_progress(self, *, source: str) -> None:
        """Record one unit of observed progress and rate-limit event emission."""

        now = time.monotonic()
        with self._state_lock:
            self._progress_observable = True
            self._progress_source = source
            self._progress_event_count += 1
            self._last_progress_at_monotonic = now
            self._stalled_emitted = False
            should_emit = (
                self._last_progress_event_emitted_at is None
                or self.heartbeat_interval_s <= 0
                or (now - self._last_progress_event_emitted_at) >= self.heartbeat_interval_s
            )
            if should_emit:
                self._last_progress_event_emitted_at = now
            snapshot = self._snapshot_locked()
        if should_emit:
            self._emit_phase("progress", elapsed_s=now - self.started_at, snapshot=snapshot)

    def snapshot(self) -> _LLMCallProgressSnapshot:
        """Return the current progress-observability state for this call."""

        with self._state_lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> _LLMCallProgressSnapshot:
        """Build a progress snapshot while the internal state lock is held."""

        return _LLMCallProgressSnapshot(
            progress_observable=self._progress_observable,
            progress_source=self._progress_source,
            progress_event_count=self._progress_event_count,
            last_progress_at_monotonic=self._last_progress_at_monotonic,
        )

    def _emit_phase(
        self,
        phase: Literal["heartbeat", "progress", "stalled"],
        *,
        elapsed_s: float,
        snapshot: _LLMCallProgressSnapshot | None = None,
    ) -> None:
        """Emit one non-terminal lifecycle phase with the latest progress metadata."""

        effective_snapshot = snapshot or self.snapshot()
        _emit_llm_call_lifecycle_event(
            call_id=self.call_id,
            phase=phase,
            call_kind=self.call_kind,
            caller=self.caller,
            task=self.task,
            trace_id=self.trace_id,
            requested_model=self.requested_model,
            provider_timeout_s=self.provider_timeout_s,
            prompt_ref=self.prompt_ref,
            elapsed_s=elapsed_s,
            heartbeat_interval_s=self.heartbeat_interval_s,
            stall_after_s=self.stall_after_s,
            progress_observable=effective_snapshot.progress_observable,
            progress_source=effective_snapshot.progress_source,
            progress_event_count=effective_snapshot.progress_event_count,
        )

    def _next_stall_wait(
        self,
        *,
        now: float,
        snapshot: _LLMCallProgressSnapshot,
    ) -> float | None:
        """Return seconds until the next eligible stall event, if any."""

        if self.stall_after_s <= 0:
            return None
        with self._state_lock:
            stalled_emitted = self._stalled_emitted
        if stalled_emitted:
            return None
        if snapshot.progress_observable:
            last_progress = snapshot.last_progress_at_monotonic
            if last_progress is None:
                return None
            idle_for_s = now - last_progress
            return max(self.stall_after_s - idle_for_s, 0.001)
        elapsed_s = now - self.started_at
        return max(self.stall_after_s - elapsed_s, 0.001)

    def _should_emit_stalled(
        self,
        *,
        now: float,
        snapshot: _LLMCallProgressSnapshot,
    ) -> bool:
        """Return whether the current call state has crossed the stall threshold."""

        if self.stall_after_s <= 0:
            return False
        with self._state_lock:
            if self._stalled_emitted:
                return False
        if snapshot.progress_observable:
            last_progress = snapshot.last_progress_at_monotonic
            if last_progress is None:
                return False
            return (now - last_progress) >= self.stall_after_s
        return (now - self.started_at) >= self.stall_after_s

    def _mark_stalled_emitted(self) -> None:
        """Remember that the current idle period already emitted a stall marker."""

        with self._state_lock:
            self._stalled_emitted = True

    async def _run(self) -> None:
        """Emit in-flight lifecycle markers until the wrapped async call terminates."""

        next_heartbeat = self.heartbeat_interval_s if self.heartbeat_interval_s > 0 else None
        while True:
            now = time.monotonic()
            elapsed = now - self.started_at
            snapshot = self.snapshot()
            waits: list[float] = []
            if next_heartbeat is not None:
                waits.append(max(next_heartbeat - elapsed, 0.001))
            stall_wait = self._next_stall_wait(now=now, snapshot=snapshot)
            if stall_wait is not None:
                waits.append(stall_wait)
            if not waits:
                return
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=min(waits))
                return
            except TimeoutError:
                pass
            now = time.monotonic()
            elapsed = now - self.started_at
            snapshot = self.snapshot()
            if next_heartbeat is not None and elapsed >= next_heartbeat:
                self._emit_phase("heartbeat", elapsed_s=elapsed, snapshot=snapshot)
                while next_heartbeat is not None and elapsed >= next_heartbeat:
                    next_heartbeat += self.heartbeat_interval_s
            if self._should_emit_stalled(now=now, snapshot=snapshot):
                self._emit_phase("stalled", elapsed_s=elapsed, snapshot=snapshot)
                self._mark_stalled_emitted()


def _strip_incompatible_sampling_params(model: str, call_kwargs: dict[str, Any]) -> list[str]:
    """Drop sampling params that are unsupported for GPT-5 family variants.

    GPT-5 legacy models reject sampling controls entirely in many reasoning
    configurations. Keeping this normalization at the client layer avoids
    provider-specific 400s and silent retries when callers pass generic kwargs.
    """
    base = _base_model_name(model)
    reasoning_effort = str(call_kwargs.get("reasoning_effort", "")).strip().lower()

    should_strip = False
    if base in _GPT5_ALWAYS_STRIP_SAMPLING:
        should_strip = True
    elif base in _GPT5_REASONING_GATED_SAMPLING and reasoning_effort and reasoning_effort != "none":
        should_strip = True

    if not should_strip:
        return []

    removed: list[str] = []
    for key in _GPT5_SAMPLING_PARAMS:
        if key in call_kwargs:
            call_kwargs.pop(key, None)
            removed.append(key)
    return removed


def _resolve_unsupported_param_policy(explicit_policy: Any) -> str:
    raw = explicit_policy
    if raw is None:
        raw = os.environ.get(_UNSUPPORTED_PARAM_POLICY_ENV, "coerce_and_warn")
    policy = str(raw).strip().lower()
    policy = _UNSUPPORTED_PARAM_POLICY_ALIASES.get(policy, policy)
    if policy not in _UNSUPPORTED_PARAM_POLICIES:
        allowed = ", ".join(sorted(_UNSUPPORTED_PARAM_POLICIES))
        raise ValueError(
            f"Invalid unsupported_param_policy={raw!r}. "
            f"Allowed: {allowed} (or aliases: {', '.join(sorted(_UNSUPPORTED_PARAM_POLICY_ALIASES))})"
        )
    return policy


def _coerce_model_incompatible_params(
    *,
    model: str,
    kwargs: dict[str, Any],
    policy: str,
    warning_sink: list[str] | None = None,
) -> list[str]:
    """Normalize unsupported params and emit loud diagnostics."""
    removed: list[str] = []

    # Bare GPT-5 models route via responses API and reject temperature.
    if _is_responses_api_model(model) and "temperature" in kwargs:
        kwargs.pop("temperature", None)
        removed.append("temperature")

    # GPT-5 family sampling incompatibilities across providers/completions.
    removed.extend(_strip_incompatible_sampling_params(model, kwargs))

    if not removed:
        return []

    removed_unique = sorted(set(removed))
    detail = (
        f"COERCE_PARAMS model={model} policy={policy} "
        f"removed={','.join(removed_unique)} "
        f"rule=gpt5_sampling_compatibility"
    )
    if policy == "error":
        raise LLMCapabilityError(
            f"Unsupported params for model {model}: {', '.join(removed_unique)}. "
            "Use unsupported_param_policy='coerce_and_warn' to auto-coerce."
        )
    if policy == "coerce_and_warn":
        logger.warning(detail)
        if warning_sink is not None:
            warning_sink.append(detail)
    else:
        logger.info(detail)
    return removed_unique


def _is_agent_model(model: str) -> bool:
    """Check if model routes to an agent SDK instead of litellm.

    Agent models like "claude-code" or "claude-code/opus" use the Claude
    Agent SDK. "openai-agents/*" is reserved for future OpenAI Agents SDK.
    """
    lower = model.lower()
    for prefix in ("claude-code", "codex", "openai-agents"):
        if lower == prefix or lower.startswith(prefix + "/"):
            return True
    # Support selected Codex aliases that map to Codex agent SDK models.
    if lower in _CODEX_AGENT_ALIASES:
        return True
    return False


ExecutionMode = Literal["text", "structured", "workspace_agent", "workspace_tools"]
_VALID_EXECUTION_MODES: frozenset[str] = frozenset(
    {"text", "structured", "workspace_agent", "workspace_tools"}
)
_AGENT_ONLY_KWARGS: frozenset[str] = frozenset(
    {
        "allowed_tools",
        "cwd",
        "max_turns",
        "max_tool_calls",
        "permission_mode",
        "max_budget_usd",
        "sandbox_mode",
        "working_directory",
        "approval_policy",
        "model_reasoning_effort",
        "network_access_enabled",
        "web_search_enabled",
        "additional_directories",
        "skip_git_repo_check",
        "yolo_mode",
        "codex_home",
    }
)


def _validate_execution_contract(
    *,
    models: list[str],
    execution_mode: str,
    kwargs: dict[str, Any],
    caller: str,
) -> None:
    """Validate model/kwargs capability compatibility before dispatch."""
    if execution_mode not in _VALID_EXECUTION_MODES:
        valid = ", ".join(sorted(_VALID_EXECUTION_MODES))
        raise ValueError(f"Invalid execution_mode={execution_mode!r}. Valid values: {valid}")

    if execution_mode == "workspace_agent":
        non_agent = [m for m in models if not _is_agent_model(m)]
        if non_agent:
            raise LLMCapabilityError(
                f"{caller}: execution_mode='workspace_agent' requires agent models "
                f"(codex/claude-code/openai-agents). Incompatible models: {non_agent}"
            )

    if execution_mode == "workspace_tools":
        agent_models = [m for m in models if _is_agent_model(m)]
        if agent_models:
            raise LLMCapabilityError(
                f"{caller}: execution_mode='workspace_tools' requires non-agent models. "
                f"Incompatible models: {agent_models}"
            )
        if not any(k in kwargs for k in ("python_tools", "mcp_servers", "mcp_sessions")):
            raise LLMCapabilityError(
                f"{caller}: execution_mode='workspace_tools' requires python_tools "
                "or mcp_servers/mcp_sessions."
            )

    # max_turns/max_tool_calls are valid for non-agent models when using MCP/python_tools
    has_tool_loop = any(k in kwargs for k in ("mcp_servers", "mcp_sessions", "python_tools"))
    check_set = _AGENT_ONLY_KWARGS - {"max_turns", "max_tool_calls"} if has_tool_loop else _AGENT_ONLY_KWARGS
    agent_only = sorted(k for k in kwargs if k in check_set)
    if agent_only:
        non_agent = [m for m in models if not _is_agent_model(m)]
        agent_models = [m for m in models if _is_agent_model(m)]
        if non_agent and not agent_models:
            raise LLMCapabilityError(
                f"{caller}: agent-only kwargs {agent_only} are incompatible with "
                f"non-agent model(s) {non_agent}. Use codex/claude-code or remove "
                "agent-only kwargs."
            )
        if non_agent and agent_models:
            logger.warning(
                "%s: mixed agent/non-agent fallback chain detected; agent-only kwargs %s "
                "will be ignored on non-agent fallback legs.",
                caller,
                agent_only,
            )


def _coerce_model_kwargs_for_execution(
    *,
    current_model: str,
    kwargs: dict[str, Any],
    warning_sink: list[str] | None,
) -> dict[str, Any]:
    """Strip kwargs unsupported for the current execution leg.

    This enables mixed agent/non-agent fallback chains by removing agent-only
    kwargs when executing non-agent models.
    """
    if _is_agent_model(current_model):
        return kwargs

    removed = sorted(k for k in kwargs if k in _AGENT_ONLY_KWARGS)
    if not removed:
        return kwargs

    model_kwargs = dict(kwargs)
    for key in removed:
        model_kwargs.pop(key, None)

    detail = (
        f"COERCE_PARAMS model={current_model} policy=coerce_and_warn "
        f"removed={','.join(removed)} "
        "rule=agent_fallback_compatibility"
    )
    logger.warning(detail)
    if warning_sink is not None:
        warning_sink.append(detail)
    return model_kwargs


# ---------------------------------------------------------------------------
# Model deprecation warnings
# ---------------------------------------------------------------------------

# Models that are outclassed on both price and quality by newer alternatives.
# Key: model substring (matched case-insensitively against the model string).
# Value: (replacement suggestion, reason).
# Checked at every call_llm / stream_llm entry point.
_DEPRECATED_MODELS: dict[str, tuple[str, str]] = {
    "gpt-4o-mini": (
        "deepseek/deepseek-chat OR gemini/gemini-2.5-flash",
        "GPT-4o-mini (intel 30, $0.15/$0.60) is outclassed by DeepSeek V3.2 "
        "(intel 42, $0.28/$0.42) and MiMo-V2-Flash (intel 41, $0.15 blended). "
        "Both are smarter AND cheaper.",
    ),
    # gpt-4o moved to _WARNED_MODELS (warn-only, never banned)
    "o1-mini": (
        "o3-mini",
        "o1-mini is deprecated. Use o3-mini for reasoning tasks.",
    ),
    "o4-mini": (
        "o3-mini",
        "o4-mini was retired by OpenAI on Feb 16, 2026. Use o3-mini "
        "for reasoning tasks or gpt-5-mini for general tasks.",
    ),
    "o1-pro": (
        "o3",
        "o1-pro ($150/$600) is superseded by o3 ($2/$8) which is better at "
        "reasoning at a fraction of the cost.",
    ),
    "gemini-1.5": (
        "gemini/gemini-2.5-flash OR gemini/gemini-2.5-pro",
        "All Gemini 1.5 models are superseded by 2.5+ equivalents at the "
        "same price with better quality. Use gemini-2.5-flash or gemini-2.5-pro.",
    ),
    "gemini-2.0-flash": (
        "gemini/gemini-2.5-flash",
        "Gemini 2.0 Flash is superseded by 2.5 Flash at the same price with "
        "significantly better quality.",
    ),
    "claude-3-5": (
        "anthropic/claude-sonnet-4-5-20250929 OR anthropic/claude-haiku-4-5-20251001",
        "Claude 3.5 models are superseded by 4.5 equivalents at the same price "
        "with better quality.",
    ),
    "claude-3-opus": (
        "anthropic/claude-opus-4-6",
        "Claude 3 Opus is superseded by Opus 4.5/4.6 at a lower price with "
        "dramatically better quality.",
    ),
    "claude-3-sonnet": (
        "anthropic/claude-sonnet-4-5-20250929",
        "Claude 3 Sonnet is superseded by Sonnet 4.5 at the same price with "
        "much better quality.",
    ),
    "claude-3-haiku": (
        "anthropic/claude-haiku-4-5-20251001",
        "Claude 3 Haiku is superseded by Haiku 4.5 at the same price with "
        "much better quality.",
    ),
    "mistral-large": (
        "deepseek/deepseek-chat OR gemini/gemini-2.5-flash",
        "Mistral Large (intel ~27, $2.75 blended) is dramatically overpriced "
        "for its quality. DeepSeek V3.2 (intel 42, $0.32) is 8x cheaper and smarter.",
    ),
}

# Models that are outclassed but still usable — warn loudly, never ban.
# Same format as _DEPRECATED_MODELS. Useful for benchmarking against baselines.
_WARNED_MODELS: dict[str, tuple[str, str]] = {
    "gpt-4o": (
        "gpt-5",
        "GPT-4o ($2.50/$10) is outclassed by GPT-5 ($1.25/$10) — "
        "GPT-5 is cheaper and smarter. Consider switching.",
    ),
}

# Models that match a deprecated pattern but should NOT be flagged
_DEPRECATED_MODEL_EXCEPTIONS: set[str] = {
    "gpt-4o-mini",  # has its own entry — prevent double-match from gpt-4o
    "gemini-2.0-flash-lite",  # NOT deprecated — cheapest Google model, no 2.5 equivalent
}


def _check_model_deprecation(model: str) -> None:
    """Emit a loud warning (or raise) if the model is outclassed.

    The warning text is intentionally aggressive because it may be read by
    an LLM agent that needs to STOP and ask the user before proceeding.

    Set ``LLM_CLIENT_STRICT_MODELS=1`` to raise ``LLMModelNotFoundError``
    instead of warning. This is recommended for benchmarks and CI.
    """
    lower = model.lower()
    strict = os.environ.get("LLM_CLIENT_STRICT_MODELS", "").strip() == "1"
    for pattern, (replacement, reason) in _DEPRECATED_MODELS.items():
        if pattern in lower:
            # Check exceptions (e.g., don't flag gpt-4o-mini under gpt-4o)
            if any(exc in lower and exc != pattern for exc in _DEPRECATED_MODEL_EXCEPTIONS):
                continue
            if strict:
                raise LLMModelNotFoundError(
                    f"BANNED MODEL: {model}. {reason} Use instead: {replacement}. "
                    f"(LLM_CLIENT_STRICT_MODELS=1 is set — deprecated models are blocked.)"
                )
            warning_msg = (
                f"\n{'=' * 72}\n"
                f"WARNING: DEPRECATED MODEL DETECTED: {model}\n"
                f"{'=' * 72}\n"
                f"\n"
                f"STOP. DO NOT USE THIS MODEL UNLESS THE USER HAS SPECIFICALLY "
                f"TOLD YOU TO AND HAS CONFIRMED THEY UNDERSTAND WHY NOT TO USE IT. "
                f"YOU MUST GET USER PERMISSION. STOP NOW AND ASK IF YOU DON'T HAVE IT.\n"
                f"\n"
                f"Reason: {reason}\n"
                f"\n"
                f"Use instead: {replacement}\n"
                f"{'=' * 72}\n"
            )
            logger.warning(warning_msg)
            import warnings
            warnings.warn(warning_msg, DeprecationWarning, stacklevel=3)
            return
    # Warned models: loud warning but never banned, even in strict mode
    for pattern, (replacement, reason) in _WARNED_MODELS.items():
        if pattern in lower:
            if any(exc in lower and exc != pattern for exc in _DEPRECATED_MODEL_EXCEPTIONS):
                continue
            warning_msg = (
                f"\n{'=' * 72}\n"
                f"WARNING: OUTCLASSED MODEL: {model}\n"
                f"{'=' * 72}\n"
                f"Reason: {reason}\n"
                f"Use instead: {replacement}\n"
                f"{'=' * 72}\n"
            )
            logger.warning(warning_msg)
            import warnings
            warnings.warn(warning_msg, UserWarning, stacklevel=3)
            return


def _strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Add additionalProperties: false to all objects for OpenAI strict mode.

    OpenAI's structured output requires every object in the schema to have
    additionalProperties: false. Pydantic's model_json_schema() doesn't
    include this by default. Recursively processes all combinators (anyOf,
    allOf, oneOf) and nested structures.
    """
    if schema.get("type") == "object":
        if "properties" in schema:
            # Structured model — lock down with strict mode
            schema["additionalProperties"] = False
            # OpenAI strict mode requires ALL properties in required
            schema["required"] = list(schema["properties"].keys())
            for prop in schema["properties"].values():
                _strict_json_schema(prop)
        elif isinstance(schema.get("additionalProperties"), dict):
            # Freeform dict (e.g. dict[str, str]) — preserve the value schema,
            # don't overwrite with false which would make it always-empty
            _strict_json_schema(schema["additionalProperties"])
        else:
            schema["additionalProperties"] = False
    if "items" in schema:
        _strict_json_schema(schema["items"])
    # Handle combinators (Optional, Union, discriminated unions)
    for combinator in ("anyOf", "allOf", "oneOf"):
        for sub_schema in schema.get(combinator, []):
            _strict_json_schema(sub_schema)
    # Handle $defs for nested models
    for defn in schema.get("$defs", {}).values():
        _strict_json_schema(defn)
    return schema


def _convert_messages_to_input(messages: list[dict[str, Any]]) -> str:
    """Convert chat messages to a single input string for responses() API.

    The Responses API accepts either a string or a message list as input.
    We convert to string to handle all message formats uniformly.
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"User: {content}")
    return "\n\n".join(parts)


def _convert_response_format_for_responses(
    response_format: dict[str, Any] | None,
) -> dict[str, Any]:
    """Convert completion() response_format to responses() text parameter.

    The Responses API uses a 'text' parameter with a 'format' key instead of
    the Chat Completions API's 'response_format' parameter.
    """
    if not response_format:
        return {"format": {"type": "text"}}

    if response_format.get("type") == "json_object":
        return {"format": {"type": "text"}}

    if response_format.get("type") == "json_schema":
        json_schema = response_format.get("json_schema", {})
        return {
            "format": {
                "type": "json_schema",
                "name": json_schema.get("name", "response_schema"),
                "schema": json_schema.get("schema", {}),
                "strict": json_schema.get("strict", True),
            }
        }

    return {"format": {"type": "text"}}


def _convert_tools_for_responses_api(
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert tool schemas from ChatCompletions to Responses API format.

    ChatCompletions: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
    Responses API:   {"type": "function", "name": ..., "description": ..., "parameters": ...}

    Idempotent — already-flat schemas pass through unchanged.
    """
    converted = []
    for tool in tools:
        if "function" in tool and isinstance(tool["function"], dict):
            flat = {"type": tool.get("type", "function")}
            flat.update(tool["function"])
            converted.append(flat)
        else:
            converted.append(tool)
    return converted


def _prepare_responses_kwargs(
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout: int,
    reasoning_effort: str | None,
    api_base: str | None,
    kwargs: dict[str, Any],
    warning_sink: list[str] | None = None,
) -> dict[str, Any]:
    """Build kwargs for litellm.responses() / aresponses().

    Converts messages to input string, response_format to text parameter,
    and strips max_tokens/max_output_tokens (GPT-5 uses reasoning tokens
    before output tokens — setting limits can exhaust them on reasoning
    and return empty output while still billing you).
    """
    kwargs = dict(kwargs)  # Don't mutate caller's dict
    policy = _resolve_unsupported_param_policy(kwargs.pop("unsupported_param_policy", None))
    _coerce_model_incompatible_params(
        model=model,
        kwargs=kwargs,
        policy=policy,
        warning_sink=warning_sink,
    )

    input_text = _convert_messages_to_input(messages)

    resp_kwargs: dict[str, Any] = {
        "model": model,
        "input": input_text,
    }
    if timeout > 0:
        resp_kwargs["timeout"] = timeout

    if api_base is not None:
        resp_kwargs["api_base"] = api_base

    # Convert response_format → text parameter
    response_format = kwargs.pop("response_format", None)
    if response_format:
        resp_kwargs["text"] = _convert_response_format_for_responses(
            response_format
        )

    # Pass through reasoning_effort for models that support it (gpt-5.2-pro etc.).
    # Named arg wins; kwargs fallback supports legacy/internal call paths.
    effort = reasoning_effort
    if effort is None:
        effort = kwargs.pop("reasoning_effort", None)
    else:
        kwargs.pop("reasoning_effort", None)
    if effort and _base_model_name(model) in _GPT5_REASONING_GATED_SAMPLING:
        resp_kwargs["reasoning"] = {"effort": effort}

    # Enable background mode for long-thinking models
    if _needs_background_mode(model, effort):
        resp_kwargs["background"] = True

    # Strip parameters that break GPT-5 or don't apply to responses API
    for key in ("max_tokens", "max_output_tokens", "messages",
                "thinking", "temperature", "unsupported_param_policy",
                "background_timeout", "background_poll_interval"):
        kwargs.pop(key, None)

    # Convert tools from ChatCompletions format to Responses API format.
    # ChatCompletions: {"type": "function", "function": {"name": ..., ...}}
    # Responses API:   {"type": "function", "name": ..., ...}
    if "tools" in kwargs:
        kwargs["tools"] = _convert_tools_for_responses_api(kwargs["tools"])

    resp_kwargs.update(kwargs)
    return resp_kwargs


def _extract_responses_usage(response: Any) -> dict[str, Any]:
    """Extract token usage from responses() API response.

    Responses API uses input_tokens/output_tokens and input_tokens_details
    (vs prompt_tokens/completion_tokens and prompt_tokens_details in Chat Completions).
    """
    usage = getattr(response, "usage", None)
    if usage is not None:
        result = {
            "prompt_tokens": getattr(usage, "input_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "output_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }
        # Responses API: input_tokens_details.cached_tokens
        itd = getattr(usage, "input_tokens_details", None)
        if itd is not None:
            cached = getattr(itd, "cached_tokens", None) or 0
            result["cached_tokens"] = cached  # Always include, even if 0
        return result
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _compute_responses_cost(response: Any, usage: dict[str, Any]) -> tuple[float, str]:
    """Compute cost for a responses() API call."""
    # Try litellm's built-in cost calculation
    try:
        cost = float(litellm.completion_cost(completion_response=response))
        if cost > 0:
            return cost, "computed"
    except Exception:
        pass

    # Try the usage.cost field (responses API sometimes includes this)
    raw_usage = getattr(response, "usage", None)
    if raw_usage and hasattr(raw_usage, "cost") and raw_usage.cost:
        return float(raw_usage.cost), "provider_reported"

    # Fallback estimate
    total = usage["total_tokens"]
    fallback = total * FALLBACK_COST_FLOOR_USD_PER_TOKEN
    if total > 0:
        logger.warning(
            "completion_cost failed for responses API, "
            "using fallback: $%.6f for %d tokens",
            fallback,
            total,
        )
    return fallback, "fallback_estimate"


def _build_result_from_responses(
    response: Any,
    model: str,
    warnings: list[str] | None = None,
) -> LLMCallResult:
    """Build LLMCallResult from a responses() API response."""
    def _item_get(item: Any, key: str, default: Any = None) -> Any:
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)

    def _extract_responses_tool_calls(resp: Any) -> list[dict[str, Any]]:
        output_items = getattr(resp, "output", None) or []
        tool_calls: list[dict[str, Any]] = []
        for idx, item in enumerate(output_items):
            item_type = _item_get(item, "type")
            if item_type not in {"function_call", "tool_call", "function"}:
                continue

            fn_name = _item_get(item, "name")
            fn_args = _item_get(item, "arguments")

            # Some providers nest function payloads under "function".
            if not fn_name:
                fn_obj = _item_get(item, "function")
                if fn_obj is not None:
                    if isinstance(fn_obj, dict):
                        fn_name = fn_obj.get("name")
                        fn_args = fn_args if fn_args is not None else fn_obj.get("arguments")
                    else:
                        fn_name = getattr(fn_obj, "name", fn_name)
                        fn_args = fn_args if fn_args is not None else getattr(fn_obj, "arguments", None)

            if not fn_name:
                continue

            if fn_args is None:
                args_raw = "{}"
            elif isinstance(fn_args, str):
                args_raw = fn_args
            else:
                try:
                    args_raw = _json.dumps(fn_args)
                except Exception:
                    args_raw = str(fn_args)

            call_id = _item_get(item, "call_id") or _item_get(item, "id") or f"call_{idx}"
            tool_calls.append({
                "id": str(call_id),
                "type": "function",
                "function": {
                    "name": str(fn_name),
                    "arguments": args_raw,
                },
            })

        return tool_calls

    # Use litellm's output_text convenience property
    content = getattr(response, "output_text", None) or ""
    tool_calls = _extract_responses_tool_calls(response)

    usage = _extract_responses_usage(response)
    cost, cost_source = _parse_cost_result(_compute_responses_cost(response, usage), default_source="computed")

    # Map responses API status to finish_reason
    status = getattr(response, "status", "completed")
    if status == "incomplete":
        details = getattr(response, "incomplete_details", None)
        reason = str(getattr(details, "reason", "")) if details else ""
        if "max_output_tokens" in reason and not tool_calls:
            raise RuntimeError(
                f"LLM response truncated ({len(content)} chars). "
                "Responses API hit max_output_tokens limit."
            )
        finish_reason = "length"
    else:
        finish_reason = "stop"

    if tool_calls:
        finish_reason = "tool_calls"

    # Empty content is retryable only when no tool calls were emitted.
    if not content.strip() and not tool_calls:
        detail_reason = ""
        if status == "incomplete":
            details = getattr(response, "incomplete_details", None)
            detail_reason = str(getattr(details, "reason", "")).strip().lower() if details else ""
        diagnostics = {
            "model": model,
            "status": status,
            "finish_reason": finish_reason,
            "incomplete_reason": detail_reason or None,
            "output_items": len(getattr(response, "output", None) or []),
        }
        if detail_reason in _EMPTY_POLICY_FINISH_REASONS or finish_reason in _EMPTY_POLICY_FINISH_REASONS:
            _raise_empty_response(
                provider="responses_api",
                classification="provider_policy_block",
                retryable=False,
                diagnostics=diagnostics,
            )
        if detail_reason in _EMPTY_TOOL_PROTOCOL_FINISH_REASONS:
            _raise_empty_response(
                provider="responses_api",
                classification="provider_tool_protocol",
                retryable=False,
                diagnostics=diagnostics,
            )
        _raise_empty_response(
            provider="responses_api",
            classification="provider_empty_unknown",
            retryable=True,
            diagnostics=diagnostics,
        )

    logger.debug(
        "LLM call (responses API): model=%s tokens=%d cost=$%.6f status=%s tool_calls=%d",
        model,
        usage["total_tokens"],
        cost,
        status,
        len(tool_calls),
    )

    return LLMCallResult(
        content=content,
        usage=usage,
        cost=cost,
        model=model,
        resolved_model=model,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        raw_response=response,
        warnings=warnings or [],
        cost_source=cost_source,
    )


# ---------------------------------------------------------------------------
# Background polling for long-thinking models (gpt-5.2-pro xhigh etc.)
# ---------------------------------------------------------------------------


_BACKGROUND_ERR_ENDPOINT_UNSUPPORTED = "LLMC_ERR_BACKGROUND_ENDPOINT_UNSUPPORTED"
_BACKGROUND_ERR_MISSING_OPENAI_KEY = "LLMC_ERR_BACKGROUND_OPENAI_KEY_REQUIRED"
_BACKGROUND_ERR_MISSING_OPENROUTER_KEY = "LLMC_ERR_BACKGROUND_OPENROUTER_KEY_REQUIRED"

def _validate_background_retrieval_api_base(api_base: str | None) -> str:
    """Return endpoint kind for background retrieval ("openai" or "openrouter")."""
    if api_base is None:
        return "openai"
    base = str(api_base).strip()
    if not base:
        return "openai"

    parsed = urllib.parse.urlparse(base)
    hostname = (parsed.hostname or "").strip().lower()
    if hostname == "api.openai.com" or hostname.endswith(".api.openai.com"):
        return "openai"
    if "openai.com" in hostname and "openrouter" not in hostname:
        return "openai"
    if hostname == "openrouter.ai" or hostname.endswith(".openrouter.ai"):
        return "openrouter"

    raise LLMConfigurationError(
        "Background response retrieval for long-thinking models currently supports "
        f"OpenAI/OpenRouter endpoints only. Received api_base={base!r}. "
        "Use https://api.openai.com/v1, https://openrouter.ai/api/v1, or default.",
        error_code=_BACKGROUND_ERR_ENDPOINT_UNSUPPORTED,
        details={"api_base": base},
    )


def _needs_background_mode(model: str, reasoning_effort: str | None) -> bool:
    """Check if a model+reasoning_effort combination needs background polling.

    Long-thinking models like gpt-5.2-pro with high/xhigh reasoning can
    think for 5-10 minutes, exceeding normal HTTP timeouts.
    """
    base = _base_model_name(model)
    return (
        base in _LONG_THINKING_MODELS
        and reasoning_effort is not None
        and reasoning_effort.lower() in _LONG_THINKING_REASONING_EFFORTS
    )


def _coerce_positive_int(value: Any, default: int) -> int:
    """Best-effort int coercion with positive guard and fallback."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _background_mode_for_model(
    *,
    model: str,
    use_responses: bool,
    reasoning_effort: str | None,
) -> bool | None:
    """Return background mode flag for routing trace / execution policy."""
    if not use_responses:
        return None
    return _needs_background_mode(model, reasoning_effort)


def _background_polling_config(model_kwargs: dict[str, Any]) -> tuple[int, int]:
    """Return validated (timeout, poll_interval) for background polling."""
    timeout = _coerce_positive_int(
        model_kwargs.get("background_timeout"),
        _BACKGROUND_DEFAULT_TIMEOUT,
    )
    poll_interval = _coerce_positive_int(
        model_kwargs.get("background_poll_interval"),
        _BACKGROUND_POLL_INTERVAL,
    )
    return timeout, poll_interval


def _maybe_poll_background_response(
    response: Any,
    *,
    api_base: str | None,
    request_timeout: int | None,
    model_kwargs: dict[str, Any],
) -> Any:
    """Poll a non-terminal background response to completion when possible."""
    bg_status = getattr(response, "status", None)
    if not bg_status or bg_status == "completed":
        return response

    response_id = getattr(response, "id", None)
    if not response_id:
        return response

    bg_timeout, bg_poll_interval = _background_polling_config(model_kwargs)
    return _poll_background_response(
        response_id,
        api_base=api_base,
        request_timeout=request_timeout,
        timeout=bg_timeout,
        poll_interval=bg_poll_interval,
    )


async def _maybe_apoll_background_response(
    response: Any,
    *,
    api_base: str | None,
    request_timeout: int | None,
    model_kwargs: dict[str, Any],
) -> Any:
    """Async variant of background polling helper."""
    bg_status = getattr(response, "status", None)
    if not bg_status or bg_status == "completed":
        return response

    response_id = getattr(response, "id", None)
    if not response_id:
        return response

    bg_timeout, bg_poll_interval = _background_polling_config(model_kwargs)
    return await _apoll_background_response(
        response_id,
        api_base=api_base,
        request_timeout=request_timeout,
        timeout=bg_timeout,
        poll_interval=bg_poll_interval,
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
    import time as _time

    deadline = _time.monotonic() + timeout
    attempt = 0
    while _time.monotonic() < deadline:
        try:
            response = _retrieve_background_response(
                response_id=response_id,
                api_base=api_base,
                request_timeout=request_timeout,
            )
        except LLMConfigurationError:
            # Endpoint/auth misconfiguration is deterministic; fail immediately.
            raise
        except Exception as e:
            logger.warning("Background poll attempt %d failed: %s", attempt, e)
            _time.sleep(poll_interval)
            attempt += 1
            continue

        status = getattr(response, "status", None)
        if status == "completed":
            logger.info(
                "Background response %s completed after %d polls",
                response_id, attempt + 1,
            )
            return response
        if status in ("failed", "cancelled"):
            error = getattr(response, "error", None)
            raise RuntimeError(
                f"Background response {response_id} {status}: {error}"
            )

        logger.debug(
            "Background response %s status=%s, poll %d",
            response_id, status, attempt + 1,
        )
        _time.sleep(poll_interval)
        attempt += 1

    raise TimeoutError(
        f"Background response {response_id} did not complete within {timeout}s"
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
    import asyncio

    deadline = time.monotonic() + timeout
    attempt = 0
    while time.monotonic() < deadline:
        try:
            response = await _aretrieve_background_response(
                response_id=response_id,
                api_base=api_base,
                request_timeout=request_timeout,
            )
        except LLMConfigurationError:
            raise
        except Exception as e:
            logger.warning("Background poll attempt %d failed: %s", attempt, e)
            await asyncio.sleep(poll_interval)
            attempt += 1
            continue

        status = getattr(response, "status", None)
        if status == "completed":
            logger.info(
                "Background response %s completed after %d polls",
                response_id, attempt + 1,
            )
            return response
        if status in ("failed", "cancelled"):
            error = getattr(response, "error", None)
            raise RuntimeError(
                f"Background response {response_id} {status}: {error}"
            )

        logger.debug(
            "Background response %s status=%s, poll %d",
            response_id, status, attempt + 1,
        )
        await asyncio.sleep(poll_interval)
        attempt += 1

    raise TimeoutError(
        f"Background response {response_id} did not complete within {timeout}s"
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
    from openai import OpenAI

    endpoint_kind = _validate_background_retrieval_api_base(api_base)
    if endpoint_kind == "openrouter":
        api_key = _normalize_api_key_value(os.environ.get(OPENROUTER_API_KEY_ENV))
        if not api_key:
            ring = _openrouter_key_candidates_from_env()
            api_key = ring[0] if ring else ""
        if not api_key:
            raise LLMConfigurationError(
                "OPENROUTER_API_KEY is required to retrieve background responses "
                "for long-thinking models via OpenRouter",
                error_code=_BACKGROUND_ERR_MISSING_OPENROUTER_KEY,
                details={"api_base": api_base},
            )
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise LLMConfigurationError(
                "OPENAI_API_KEY is required to retrieve background responses for long-thinking models",
                error_code=_BACKGROUND_ERR_MISSING_OPENAI_KEY,
                details={"api_base": api_base},
            )

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if api_base:
        client_kwargs["base_url"] = api_base
    if request_timeout is not None:
        client_kwargs["timeout"] = request_timeout
    client = OpenAI(**client_kwargs)
    return client.responses.retrieve(response_id)


async def _aretrieve_background_response(
    *,
    response_id: str,
    api_base: str | None,
    request_timeout: int | None,
) -> Any:
    """Async retrieve for background responses by ID."""
    from openai import AsyncOpenAI

    endpoint_kind = _validate_background_retrieval_api_base(api_base)
    if endpoint_kind == "openrouter":
        api_key = _normalize_api_key_value(os.environ.get(OPENROUTER_API_KEY_ENV))
        if not api_key:
            ring = _openrouter_key_candidates_from_env()
            api_key = ring[0] if ring else ""
        if not api_key:
            raise LLMConfigurationError(
                "OPENROUTER_API_KEY is required to retrieve background responses "
                "for long-thinking models via OpenRouter",
                error_code=_BACKGROUND_ERR_MISSING_OPENROUTER_KEY,
                details={"api_base": api_base},
            )
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise LLMConfigurationError(
                "OPENAI_API_KEY is required to retrieve background responses for long-thinking models",
                error_code=_BACKGROUND_ERR_MISSING_OPENAI_KEY,
                details={"api_base": api_base},
            )

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if api_base:
        client_kwargs["base_url"] = api_base
    if request_timeout is not None:
        client_kwargs["timeout"] = request_timeout
    client = AsyncOpenAI(**client_kwargs)
    return await client.responses.retrieve(response_id)


# ---------------------------------------------------------------------------
# Completion API helpers
# ---------------------------------------------------------------------------


def _apply_max_tokens(model: str, call_kwargs: dict[str, Any]) -> None:
    """Clamp explicit output-token caps to the model maximum when present.

    The client does not invent output-token ceilings when callers omit them.
    Defaulting to the provider maximum turns routine calls into accidental
    high-cost requests, especially for structured-output workloads where a
    large generated cap is not the same as a useful response. When callers do
    supply an explicit cap, this helper only prevents provider-side
    ``max_tokens > model_max`` validation errors.

    Silently skips if model info lookup fails (unknown/custom models).
    """
    try:
        info = litellm.get_model_info(model)
    except Exception:
        return  # Unknown model — pass through unchanged

    model_max = info.get("max_output_tokens")
    if not model_max:
        return

    # Determine which key the caller used (if any)
    token_key = None
    for key in ("max_completion_tokens", "max_tokens"):
        if key in call_kwargs:
            token_key = key
            break

    if token_key:
        # Clamp to model's max
        if call_kwargs[token_key] > model_max:
            logger.debug(
                "Clamping %s from %d to %d for %s",
                token_key, call_kwargs[token_key], model_max, model,
            )
            call_kwargs[token_key] = model_max
    else:
        return


def _prepare_call_kwargs(
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout: int,
    num_retries: int,
    reasoning_effort: str | None,
    api_base: str | None,
    kwargs: dict[str, Any],
    warning_sink: list[str] | None = None,
) -> dict[str, Any]:
    """Build kwargs dict shared by call_llm and acall_llm."""
    raw_kwargs = dict(kwargs)
    policy = _resolve_unsupported_param_policy(raw_kwargs.pop("unsupported_param_policy", None))
    call_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        # Don't pass num_retries to litellm — our own retry loop handles
        # all retries with jittered backoff. Passing it to litellm causes
        # double retry (litellm retries HTTP errors internally, then our
        # loop retries the same errors again).
        **raw_kwargs,
    }
    if timeout > 0:
        call_kwargs["timeout"] = timeout

    if api_base is not None:
        call_kwargs["api_base"] = api_base

    # Only pass reasoning_effort for Claude models
    if reasoning_effort and _is_claude_model(model):
        call_kwargs["reasoning_effort"] = reasoning_effort
    elif reasoning_effort:
        logger.debug(
            "reasoning_effort=%s ignored for non-Claude model %s",
            reasoning_effort,
            model,
        )

    # Thinking model detection: suppress thinking tokens for Gemini 2.5+
    # so all output budget goes to the actual response.
    # - Native Gemini path handles this internally via thinkingConfig.
    # - litellm path: only inject `thinking` when litellm says the specific
    #   model supports it (litellm's model-level param list is authoritative).
    # - OpenRouter: skip — doesn't support the `thinking` parameter.
    if _is_thinking_model(model) and "thinking" not in raw_kwargs:
        _is_openrouter = model.lower().startswith("openrouter/")
        if not _is_openrouter:
            try:
                from litellm import get_supported_openai_params
                _provider = model.split("/")[0] if "/" in model else ""
                _supported = get_supported_openai_params(
                    model=model, custom_llm_provider=_provider,
                ) or []
                if "thinking" in _supported:
                    call_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 0}
            except Exception:
                pass  # If we can't check, don't inject — fail-safe

    # Guard against GPT-5-family sampling param incompatibilities across
    # providers (e.g., provider-prefixed GPT-5 models on completion path).
    _coerce_model_incompatible_params(
        model=model,
        kwargs=call_kwargs,
        policy=policy,
        warning_sink=warning_sink,
    )

    # Never invent output-token ceilings; only clamp explicit caller values.
    if not _is_responses_api_model(model):
        _apply_max_tokens(model, call_kwargs)

    return call_kwargs


def _provider_hint_from_response(response: Any) -> str | None:
    """Best-effort provider hint from litellm response metadata."""
    hidden = getattr(response, "_hidden_params", None)
    if isinstance(hidden, dict):
        for key in ("custom_llm_provider", "provider", "litellm_provider"):
            value = hidden.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for attr in ("provider", "llm_provider"):
        value = getattr(response, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _first_choice_or_empty_error(
    response: Any,
    *,
    model: str,
    provider: str,
) -> Any:
    """Return first completion choice or raise a typed empty-response error."""
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        _raise_empty_response(
            provider=provider,
            classification="provider_empty_candidates",
            retryable=True,
            diagnostics={
                "model": model,
                "provider_hint": _provider_hint_from_response(response),
                "has_choices": isinstance(choices, list),
                "choice_count": len(choices) if isinstance(choices, list) else 0,
            },
        )
    return choices[0]


def _build_result_from_response(
    response: Any,
    model: str,
    warnings: list[str] | None = None,
) -> LLMCallResult:
    """Extract all fields from a litellm response into LLMCallResult."""
    first_choice = _first_choice_or_empty_error(
        response, model=model, provider="litellm_completion"
    )
    content: str = first_choice.message.content or ""
    finish_reason: str = first_choice.finish_reason or ""
    tool_calls = _extract_tool_calls(first_choice.message)
    usage = _extract_usage(response)
    cost, cost_source = _parse_cost_result(_compute_cost(response))

    # Raise on truncation (non-retryable) — retrying won't help, token limit is fixed
    if finish_reason == "length":
        raise RuntimeError(
            f"LLM response truncated ({len(content)} chars). "
            "Increase max_tokens or simplify the prompt."
        )

    # Raise on empty content (retryable) — unless model made tool calls.
    # Note: finish_reason="tool_calls" with no actual tool_calls is a model bug
    # that should be retried, so we only check for actual tool_calls presence.
    if not content.strip() and not tool_calls:
        finish_norm = str(finish_reason).strip().lower()
        diagnostics = {
            "model": model,
            "provider_hint": _provider_hint_from_response(response),
            "finish_reason": finish_reason or None,
            "has_tool_calls": bool(tool_calls),
        }
        if finish_norm in _EMPTY_POLICY_FINISH_REASONS:
            _raise_empty_response(
                provider="litellm_completion",
                classification="provider_policy_block",
                retryable=False,
                diagnostics=diagnostics,
            )
        if finish_norm in _EMPTY_TOOL_PROTOCOL_FINISH_REASONS:
            _raise_empty_response(
                provider="litellm_completion",
                classification="provider_tool_protocol",
                retryable=False,
                diagnostics=diagnostics,
            )
        _raise_empty_response(
            provider="litellm_completion",
            classification="provider_empty_unknown",
            retryable=True,
            diagnostics=diagnostics,
        )

    logger.debug(
        "LLM call: model=%s tokens=%d cost=$%.6f finish=%s",
        model,
        usage["total_tokens"],
        cost,
        finish_reason,
    )

    return LLMCallResult(
        content=content,
        usage=usage,
        cost=cost,
        model=model,
        resolved_model=model,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        raw_response=response,
        warnings=warnings or [],
        cost_source=cost_source,
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

    normalized_prompt_ref = _normalize_prompt_ref(kwargs.get("prompt_ref"))
    resolved_task, resolved_trace_id, resolved_max_budget, _ = _require_tags(
        kwargs.get("task"),
        kwargs.get("trace_id"),
        kwargs.get("max_budget"),
        caller="call_llm",
    )
    _check_budget(resolved_trace_id, resolved_max_budget)
    effective_provider_timeout = _provider_timeout_for_lifecycle(timeout)

    runtime_kwargs = dict(kwargs)
    heartbeat_interval_s, stall_after_s = _resolve_lifecycle_monitoring_settings(
        heartbeat_interval=runtime_kwargs.pop("lifecycle_heartbeat_interval_s", None),
        stall_after=runtime_kwargs.pop("lifecycle_stall_after_s", None),
    )
    runtime_kwargs["task"] = resolved_task
    runtime_kwargs["trace_id"] = resolved_trace_id
    runtime_kwargs["max_budget"] = resolved_max_budget
    runtime_kwargs["prompt_ref"] = normalized_prompt_ref

    call_id = _new_llm_call_lifecycle_id()
    started_at = time.monotonic()
    monitor = _SyncLLMCallHeartbeatMonitor(
        call_id=call_id,
        call_kind="text",
        caller="call_llm",
        task=resolved_task,
        trace_id=resolved_trace_id,
        requested_model=model,
        provider_timeout_s=effective_provider_timeout,
        prompt_ref=normalized_prompt_ref,
        heartbeat_interval_s=heartbeat_interval_s,
        stall_after_s=stall_after_s,
        started_at=started_at,
    )
    started_snapshot = monitor.snapshot()
    _emit_llm_call_lifecycle_event(
        call_id=call_id,
        phase="started",
        call_kind="text",
        caller="call_llm",
        task=resolved_task,
        trace_id=resolved_trace_id,
        requested_model=model,
        provider_timeout_s=effective_provider_timeout,
        prompt_ref=normalized_prompt_ref,
        heartbeat_interval_s=heartbeat_interval_s,
        stall_after_s=stall_after_s,
        progress_observable=started_snapshot.progress_observable,
        progress_source=started_snapshot.progress_source,
        progress_event_count=started_snapshot.progress_event_count,
    )
    runtime_kwargs["_lifecycle_monitor"] = monitor
    monitor.start()
    try:
        result = _call_llm_impl(
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
        )
    except Exception as exc:
        monitor.stop()
        snapshot = monitor.snapshot()
        _emit_llm_call_lifecycle_event(
            call_id=call_id,
            phase="failed",
            call_kind="text",
            caller="call_llm",
            task=resolved_task,
            trace_id=resolved_trace_id,
            requested_model=model,
            provider_timeout_s=effective_provider_timeout,
            prompt_ref=normalized_prompt_ref,
            latency_s=time.monotonic() - started_at,
            error=exc,
            heartbeat_interval_s=heartbeat_interval_s,
            stall_after_s=stall_after_s,
            progress_observable=snapshot.progress_observable,
            progress_source=snapshot.progress_source,
            progress_event_count=snapshot.progress_event_count,
        )
        raise
    monitor.stop()
    elapsed_s = time.monotonic() - started_at
    completed_snapshot = monitor.snapshot()
    _emit_llm_call_lifecycle_event(
        call_id=call_id,
        phase="completed",
        call_kind="text",
        caller="call_llm",
        task=resolved_task,
        trace_id=resolved_trace_id,
        requested_model=model,
        provider_timeout_s=effective_provider_timeout,
        prompt_ref=normalized_prompt_ref,
        resolved_model=result.resolved_model or str(result.model or "") or None,
        elapsed_s=elapsed_s,
        latency_s=elapsed_s,
        heartbeat_interval_s=heartbeat_interval_s,
        stall_after_s=stall_after_s,
        progress_observable=completed_snapshot.progress_observable,
        progress_source=completed_snapshot.progress_source,
        progress_event_count=completed_snapshot.progress_event_count,
    )
    return result


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

    normalized_prompt_ref = _normalize_prompt_ref(kwargs.get("prompt_ref"))
    resolved_task, resolved_trace_id, resolved_max_budget, _ = _require_tags(
        kwargs.get("task"),
        kwargs.get("trace_id"),
        kwargs.get("max_budget"),
        caller="call_llm_structured",
    )
    _check_budget(resolved_trace_id, resolved_max_budget)
    effective_provider_timeout = _provider_timeout_for_lifecycle(timeout)

    runtime_kwargs = dict(kwargs)
    heartbeat_interval_s, stall_after_s = _resolve_lifecycle_monitoring_settings(
        heartbeat_interval=runtime_kwargs.pop("lifecycle_heartbeat_interval_s", None),
        stall_after=runtime_kwargs.pop("lifecycle_stall_after_s", None),
    )
    runtime_kwargs["task"] = resolved_task
    runtime_kwargs["trace_id"] = resolved_trace_id
    runtime_kwargs["max_budget"] = resolved_max_budget
    runtime_kwargs["prompt_ref"] = normalized_prompt_ref

    call_id = _new_llm_call_lifecycle_id()
    started_at = time.monotonic()
    monitor = _SyncLLMCallHeartbeatMonitor(
        call_id=call_id,
        call_kind="structured",
        caller="call_llm_structured",
        task=resolved_task,
        trace_id=resolved_trace_id,
        requested_model=model,
        provider_timeout_s=effective_provider_timeout,
        prompt_ref=normalized_prompt_ref,
        heartbeat_interval_s=heartbeat_interval_s,
        stall_after_s=stall_after_s,
        started_at=started_at,
    )
    started_snapshot = monitor.snapshot()
    _emit_llm_call_lifecycle_event(
        call_id=call_id,
        phase="started",
        call_kind="structured",
        caller="call_llm_structured",
        task=resolved_task,
        trace_id=resolved_trace_id,
        requested_model=model,
        provider_timeout_s=effective_provider_timeout,
        prompt_ref=normalized_prompt_ref,
        heartbeat_interval_s=heartbeat_interval_s,
        stall_after_s=stall_after_s,
        progress_observable=started_snapshot.progress_observable,
        progress_source=started_snapshot.progress_source,
        progress_event_count=started_snapshot.progress_event_count,
    )
    runtime_kwargs["_lifecycle_monitor"] = monitor
    monitor.start()
    try:
        parsed, result = _call_llm_structured_impl(
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
        )
    except Exception as exc:
        monitor.stop()
        snapshot = monitor.snapshot()
        _emit_llm_call_lifecycle_event(
            call_id=call_id,
            phase="failed",
            call_kind="structured",
            caller="call_llm_structured",
            task=resolved_task,
            trace_id=resolved_trace_id,
            requested_model=model,
            provider_timeout_s=effective_provider_timeout,
            prompt_ref=normalized_prompt_ref,
            latency_s=time.monotonic() - started_at,
            error=exc,
            heartbeat_interval_s=heartbeat_interval_s,
            stall_after_s=stall_after_s,
            progress_observable=snapshot.progress_observable,
            progress_source=snapshot.progress_source,
            progress_event_count=snapshot.progress_event_count,
        )
        raise
    monitor.stop()
    elapsed_s = time.monotonic() - started_at
    completed_snapshot = monitor.snapshot()
    _emit_llm_call_lifecycle_event(
        call_id=call_id,
        phase="completed",
        call_kind="structured",
        caller="call_llm_structured",
        task=resolved_task,
        trace_id=resolved_trace_id,
        requested_model=model,
        provider_timeout_s=effective_provider_timeout,
        prompt_ref=normalized_prompt_ref,
        resolved_model=result.resolved_model or str(result.model or "") or None,
        elapsed_s=elapsed_s,
        latency_s=elapsed_s,
        heartbeat_interval_s=heartbeat_interval_s,
        stall_after_s=stall_after_s,
        progress_observable=completed_snapshot.progress_observable,
        progress_source=completed_snapshot.progress_source,
        progress_event_count=completed_snapshot.progress_event_count,
    )
    return parsed, result


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

    normalized_prompt_ref = _normalize_prompt_ref(kwargs.get("prompt_ref"))
    resolved_task, resolved_trace_id, resolved_max_budget, _ = _require_tags(
        kwargs.get("task"),
        kwargs.get("trace_id"),
        kwargs.get("max_budget"),
        caller="acall_llm",
    )
    _check_budget(resolved_trace_id, resolved_max_budget)
    effective_provider_timeout = _provider_timeout_for_lifecycle(timeout)

    runtime_kwargs = dict(kwargs)
    heartbeat_interval_s, stall_after_s = _resolve_lifecycle_monitoring_settings(
        heartbeat_interval=runtime_kwargs.pop("lifecycle_heartbeat_interval_s", None),
        stall_after=runtime_kwargs.pop("lifecycle_stall_after_s", None),
    )
    runtime_kwargs["task"] = resolved_task
    runtime_kwargs["trace_id"] = resolved_trace_id
    runtime_kwargs["max_budget"] = resolved_max_budget
    runtime_kwargs["prompt_ref"] = normalized_prompt_ref

    call_id = _new_llm_call_lifecycle_id()
    started_at = time.monotonic()
    monitor = _AsyncLLMCallHeartbeatMonitor(
        call_id=call_id,
        call_kind="text",
        caller="acall_llm",
        task=resolved_task,
        trace_id=resolved_trace_id,
        requested_model=model,
        provider_timeout_s=effective_provider_timeout,
        prompt_ref=normalized_prompt_ref,
        heartbeat_interval_s=heartbeat_interval_s,
        stall_after_s=stall_after_s,
        started_at=started_at,
    )
    started_snapshot = monitor.snapshot()
    _emit_llm_call_lifecycle_event(
        call_id=call_id,
        phase="started",
        call_kind="text",
        caller="acall_llm",
        task=resolved_task,
        trace_id=resolved_trace_id,
        requested_model=model,
        provider_timeout_s=effective_provider_timeout,
        prompt_ref=normalized_prompt_ref,
        heartbeat_interval_s=heartbeat_interval_s,
        stall_after_s=stall_after_s,
        progress_observable=started_snapshot.progress_observable,
        progress_source=started_snapshot.progress_source,
        progress_event_count=started_snapshot.progress_event_count,
    )
    runtime_kwargs["_lifecycle_monitor"] = monitor
    monitor.start()
    try:
        result = await _acall_llm_impl(
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
        )
    except Exception as exc:
        await monitor.stop()
        snapshot = monitor.snapshot()
        _emit_llm_call_lifecycle_event(
            call_id=call_id,
            phase="failed",
            call_kind="text",
            caller="acall_llm",
            task=resolved_task,
            trace_id=resolved_trace_id,
            requested_model=model,
            provider_timeout_s=effective_provider_timeout,
            prompt_ref=normalized_prompt_ref,
            latency_s=time.monotonic() - started_at,
            error=exc,
            heartbeat_interval_s=heartbeat_interval_s,
            stall_after_s=stall_after_s,
            progress_observable=snapshot.progress_observable,
            progress_source=snapshot.progress_source,
            progress_event_count=snapshot.progress_event_count,
        )
        raise
    await monitor.stop()
    elapsed_s = time.monotonic() - started_at
    completed_snapshot = monitor.snapshot()
    _emit_llm_call_lifecycle_event(
        call_id=call_id,
        phase="completed",
        call_kind="text",
        caller="acall_llm",
        task=resolved_task,
        trace_id=resolved_trace_id,
        requested_model=model,
        provider_timeout_s=effective_provider_timeout,
        prompt_ref=normalized_prompt_ref,
        resolved_model=result.resolved_model or str(result.model or "") or None,
        elapsed_s=elapsed_s,
        latency_s=elapsed_s,
        heartbeat_interval_s=heartbeat_interval_s,
        stall_after_s=stall_after_s,
        progress_observable=completed_snapshot.progress_observable,
        progress_source=completed_snapshot.progress_source,
        progress_event_count=completed_snapshot.progress_event_count,
    )
    return result


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

    normalized_prompt_ref = _normalize_prompt_ref(kwargs.get("prompt_ref"))
    resolved_task, resolved_trace_id, resolved_max_budget, _ = _require_tags(
        kwargs.get("task"),
        kwargs.get("trace_id"),
        kwargs.get("max_budget"),
        caller="acall_llm_structured",
    )
    _check_budget(resolved_trace_id, resolved_max_budget)
    effective_provider_timeout = _provider_timeout_for_lifecycle(timeout)

    runtime_kwargs = dict(kwargs)
    heartbeat_interval_s, stall_after_s = _resolve_lifecycle_monitoring_settings(
        heartbeat_interval=runtime_kwargs.pop("lifecycle_heartbeat_interval_s", None),
        stall_after=runtime_kwargs.pop("lifecycle_stall_after_s", None),
    )
    runtime_kwargs["task"] = resolved_task
    runtime_kwargs["trace_id"] = resolved_trace_id
    runtime_kwargs["max_budget"] = resolved_max_budget
    runtime_kwargs["prompt_ref"] = normalized_prompt_ref

    call_id = _new_llm_call_lifecycle_id()
    started_at = time.monotonic()
    monitor = _AsyncLLMCallHeartbeatMonitor(
        call_id=call_id,
        call_kind="structured",
        caller="acall_llm_structured",
        task=resolved_task,
        trace_id=resolved_trace_id,
        requested_model=model,
        provider_timeout_s=effective_provider_timeout,
        prompt_ref=normalized_prompt_ref,
        heartbeat_interval_s=heartbeat_interval_s,
        stall_after_s=stall_after_s,
        started_at=started_at,
    )
    started_snapshot = monitor.snapshot()
    _emit_llm_call_lifecycle_event(
        call_id=call_id,
        phase="started",
        call_kind="structured",
        caller="acall_llm_structured",
        task=resolved_task,
        trace_id=resolved_trace_id,
        requested_model=model,
        provider_timeout_s=effective_provider_timeout,
        prompt_ref=normalized_prompt_ref,
        heartbeat_interval_s=heartbeat_interval_s,
        stall_after_s=stall_after_s,
        progress_observable=started_snapshot.progress_observable,
        progress_source=started_snapshot.progress_source,
        progress_event_count=started_snapshot.progress_event_count,
    )
    runtime_kwargs["_lifecycle_monitor"] = monitor
    monitor.start()
    try:
        parsed, result = await _acall_llm_structured_impl(
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
        )
    except Exception as exc:
        await monitor.stop()
        snapshot = monitor.snapshot()
        _emit_llm_call_lifecycle_event(
            call_id=call_id,
            phase="failed",
            call_kind="structured",
            caller="acall_llm_structured",
            task=resolved_task,
            trace_id=resolved_trace_id,
            requested_model=model,
            provider_timeout_s=effective_provider_timeout,
            prompt_ref=normalized_prompt_ref,
            latency_s=time.monotonic() - started_at,
            error=exc,
            heartbeat_interval_s=heartbeat_interval_s,
            stall_after_s=stall_after_s,
            progress_observable=snapshot.progress_observable,
            progress_source=snapshot.progress_source,
            progress_event_count=snapshot.progress_event_count,
        )
        raise
    await monitor.stop()
    elapsed_s = time.monotonic() - started_at
    completed_snapshot = monitor.snapshot()
    _emit_llm_call_lifecycle_event(
        call_id=call_id,
        phase="completed",
        call_kind="structured",
        caller="acall_llm_structured",
        task=resolved_task,
        trace_id=resolved_trace_id,
        requested_model=model,
        provider_timeout_s=effective_provider_timeout,
        prompt_ref=normalized_prompt_ref,
        resolved_model=result.resolved_model or str(result.model or "") or None,
        elapsed_s=elapsed_s,
        latency_s=elapsed_s,
        heartbeat_interval_s=heartbeat_interval_s,
        stall_after_s=stall_after_s,
        progress_observable=completed_snapshot.progress_observable,
        progress_source=completed_snapshot.progress_source,
        progress_event_count=completed_snapshot.progress_event_count,
    )
    return parsed, result


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
