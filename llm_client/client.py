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
import concurrent.futures
import hashlib
import inspect
import json as _json
import logging
import os
import random
import re
import threading
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, OrderedDict
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal, Protocol, TypeVar, runtime_checkable

import litellm
from pydantic import BaseModel

from llm_client.config import ClientConfig, ResultModelSemantics
from llm_client import io_log as _io_log
from llm_client import rate_limit as _rate_limit
from llm_client.routing import CallRequest, resolve_api_base_for_model, resolve_call

from llm_client.errors import (
    LLMBudgetExceededError,
    LLMCapabilityError,
    LLMEmptyResponseError,
    LLMModelNotFoundError,
    wrap_error,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Silence litellm's noisy default logging
litellm.suppress_debug_info = True

# Accounting constants (documented in agent_ecology3/docs/ACCOUNTING_CONSTANTS.md)
FALLBACK_COST_FLOOR_USD_PER_TOKEN = 0.000001
GEMINI_NATIVE_MODE_ENV = "LLM_CLIENT_GEMINI_NATIVE_MODE"
GEMINI_NATIVE_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_NATIVE_SUPPORTED_KWARGS: frozenset[str] = frozenset({
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "stop",
    "tools",
    "tool_choice",
    "thinking",
})
OPENROUTER_ROUTING_ENV = "LLM_CLIENT_OPENROUTER_ROUTING"
OPENROUTER_DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_API_BASE_ENV = "OPENROUTER_API_BASE"
REQUIRE_TAGS_ENV = "LLM_CLIENT_REQUIRE_TAGS"
AGENT_RETRY_SAFE_ENV = "LLM_CLIENT_AGENT_RETRY_SAFE"
SEMANTICS_TELEMETRY_ENV = "LLM_CLIENT_SEMANTICS_TELEMETRY"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class LLMCallResult:
    """Result from an LLM call. Returned by all call_llm* functions.

    Attributes:
        content: The text response from the model
        usage: Token counts (prompt_tokens, completion_tokens, total_tokens)
        cost: Cost in USD for this call
        model: The model string that was used
        tool_calls: List of tool calls if the model invoked tools, else empty
        finish_reason: Why the model stopped: "stop", "length", "tool_calls",
                       "content_filter", etc. Empty string if unavailable.
        raw_response: The full litellm response object for edge cases
                      (e.g., accessing provider-specific data like Claude
                      thinking blocks). Excluded from repr to keep logs clean.
    """

    content: str
    usage: dict[str, Any]
    cost: float
    model: str
    requested_model: str | None = None
    """Raw model string provided at the public API boundary."""
    resolved_model: str | None = None
    """Best-effort model string used for the successful terminal attempt."""
    execution_model: str | None = None
    """Alias for resolved terminal model, kept additive for migration clarity."""
    routing_trace: dict[str, Any] | None = field(default=None, repr=False)
    """Optional routing/fallback trace for contract characterization and debugging."""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = ""
    raw_response: Any = field(default=None, repr=False)
    warnings: list[str] = field(default_factory=list)
    """Diagnostic warnings accumulated during retry/fallback/routing.
    Empty list on clean calls. Populated with RETRY/FALLBACK/STICKY_FALLBACK
    messages when non-obvious decisions occurred."""
    warning_records: list[dict[str, Any]] = field(default_factory=list, repr=False)
    """Machine-readable warning records (code/category/message/remediation)."""
    full_text: str | None = field(default=None, repr=False)
    """For agent SDKs: full conversation text (all assistant messages).
    ``content`` holds only the final assistant message.
    None for non-agent calls."""
    cost_source: str = "unspecified"
    """How cost was determined: provider_reported, computed, fallback_estimate, cache_hit, etc."""
    billing_mode: str = "api_metered"
    """Billing mode: api_metered, subscription_included, or unknown."""
    marginal_cost: float | None = None
    """Incremental cost attributed to this call; defaults to ``cost`` when omitted."""
    cache_hit: bool = False
    """Whether this result came from cache instead of a model call."""

    def __post_init__(self) -> None:
        if self.marginal_cost is None:
            self.marginal_cost = 0.0 if self.cache_hit else float(self.cost)
        if self.execution_model is None and self.resolved_model is not None:
            self.execution_model = self.resolved_model


@dataclass
class EmbeddingResult:
    """Result from an embedding call.

    Attributes:
        embeddings: List of embedding vectors (one per input text)
        usage: Token counts (prompt_tokens, total_tokens)
        cost: Cost in USD for this call
        model: The model string that was used
    """

    embeddings: list[list[float]]
    usage: dict[str, Any]
    cost: float
    model: str


# ---------------------------------------------------------------------------
# Cache infrastructure
# ---------------------------------------------------------------------------


@runtime_checkable
class CachePolicy(Protocol):
    """Protocol for LLM response caches. Implement get/set for custom backends."""

    def get(self, key: str) -> LLMCallResult | None: ...
    def set(self, key: str, value: LLMCallResult) -> None: ...


@runtime_checkable
class AsyncCachePolicy(Protocol):
    """Protocol for async LLM response caches (Redis, etc.).

    Async functions accept either ``CachePolicy`` or ``AsyncCachePolicy``.
    When an ``AsyncCachePolicy`` is detected, ``await`` is used for get/set
    so the event loop is never blocked.
    """

    async def get(self, key: str) -> LLMCallResult | None: ...
    async def set(self, key: str, value: LLMCallResult) -> None: ...


class LRUCache:
    """Thread-safe in-memory LRU cache for LLM responses.

    Args:
        maxsize: Maximum number of entries. Oldest evicted on overflow.
        ttl: Time-to-live in seconds. ``None`` means entries never expire.
    """

    def __init__(self, maxsize: int = 128, ttl: float | None = None) -> None:
        self._cache: OrderedDict[str, tuple[LLMCallResult, float]] = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl
        self._lock = threading.Lock()

    def get(self, key: str) -> LLMCallResult | None:
        with self._lock:
            if key not in self._cache:
                return None
            value, ts = self._cache[key]
            if self._ttl is not None and time.monotonic() - ts > self._ttl:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return value

    def set(self, key: str, value: LLMCallResult) -> None:
        with self._lock:
            self._cache[key] = (value, time.monotonic())
            self._cache.move_to_end(key)
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


def _cache_key(model: str, messages: list[dict[str, Any]], **kwargs: Any) -> str:
    """Build a deterministic cache key from call parameters."""
    key_data = _json.dumps({"model": model, "messages": messages, **kwargs}, sort_keys=True)
    return hashlib.sha256(key_data.encode()).hexdigest()


def _mark_cache_hit(result: LLMCallResult) -> LLMCallResult:
    """Return a cache-hit view of a cached result without mutating cache state."""
    return replace(
        result,
        cache_hit=True,
        marginal_cost=0.0,
        cost_source="cache_hit",
    )


_SEMANTICS_ADOPTION_LOCK = threading.Lock()
_SEMANTICS_ADOPTION_COUNTS: Counter[tuple[str, str, str]] = Counter()


def _semantics_telemetry_enabled() -> bool:
    raw = os.environ.get(SEMANTICS_TELEMETRY_ENV, "1").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on", ""}:
        return True
    logger.warning(
        "Invalid %s=%r; expected on/off boolean. Defaulting to on.",
        SEMANTICS_TELEMETRY_ENV,
        raw,
    )
    return True


def _record_semantics_adoption(
    *,
    cfg: ClientConfig,
    explicit_config: bool,
    caller: str,
    task: str | None,
    trace_id: str | None,
) -> None:
    """Emit lightweight, metadata-only telemetry for semantics adoption."""
    mode = str(cfg.result_model_semantics)
    source = "explicit_config" if explicit_config else "env_or_default"

    with _SEMANTICS_ADOPTION_LOCK:
        key = (caller, source, mode)
        _SEMANTICS_ADOPTION_COUNTS[key] += 1
        observed_count = int(_SEMANTICS_ADOPTION_COUNTS[key])

    if not _semantics_telemetry_enabled():
        return

    try:
        from llm_client.foundation import coerce_run_id, new_event_id, new_session_id, now_iso

        event = {
            "event_id": new_event_id(),
            "event_type": "ConfigChanged",
            "timestamp": now_iso(),
            "run_id": coerce_run_id(None, trace_id),
            "session_id": new_session_id(),
            "actor_id": "llm_client:semantics_telemetry",
            "operation": {
                "name": "result_model_semantics_adoption",
                "version": "0.6.1",
            },
            "inputs": {
                "artifact_ids": [],
                "params": {
                    "caller": caller,
                    "config_source": source,
                    "result_model_semantics": mode,
                    "observed_count": observed_count,
                },
                "bindings": {},
            },
            "outputs": {
                "artifact_ids": [],
                "payload_hashes": [],
            },
        }
        _io_log.log_foundation_event(
            event=event,
            caller=caller,
            task=task,
            trace_id=trace_id,
        )
    except Exception:
        logger.debug("semantics-adoption telemetry emit failed", exc_info=True)


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
    routing_policy: str | None = None,
) -> dict[str, Any]:
    """Build a minimal routing trace for week-1 contract characterization."""
    trace: dict[str, Any] = {"routing_policy": routing_policy or _routing_policy_label()}
    attempts = [m for m in (attempted_models or []) if isinstance(m, str) and m.strip()]
    if attempts:
        trace["attempted_models"] = attempts
        if requested_model != attempts[0]:
            trace["normalized_from"] = requested_model
            trace["normalized_to"] = attempts[0]
    if selected_model:
        trace["selected_model"] = selected_model
    if sticky_fallback is not None:
        trace["sticky_fallback"] = bool(sticky_fallback)
    if (
        requested_api_base is None
        and effective_api_base is not None
    ):
        trace["api_base_injected"] = True
    elif requested_api_base is not None:
        trace["api_base_injected"] = False
    return trace


def _warning_record(
    *,
    code: str,
    category: str,
    message: str,
    field_path: str | None = None,
    remediation: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "code": code,
        "category": category,
        "message": message,
    }
    if field_path:
        record["field_path"] = field_path
    if remediation:
        record["remediation"] = remediation
    return record


def _warning_record_from_message(message: str) -> dict[str, Any] | None:
    text = str(message or "")
    if text.startswith("RETRY "):
        return _warning_record(
            code="LLMC_WARN_RETRY",
            category="RuntimeWarning",
            message=text,
            remediation="Inspect transient provider/network conditions.",
        )
    if text.startswith("FALLBACK:"):
        return _warning_record(
            code="LLMC_WARN_FALLBACK",
            category="UserWarning",
            message=text,
            remediation="Check primary model/provider health and fallback policy.",
        )
    if text.startswith("STICKY_FALLBACK:"):
        return _warning_record(
            code="LLMC_WARN_STICKY_FALLBACK",
            category="UserWarning",
            message=text,
            remediation="Investigate persistent failures on the requested primary model.",
        )
    if text.startswith("AUTO_TAG:"):
        return _warning_record(
            code="LLMC_WARN_AUTO_TAG",
            category="UserWarning",
            message=text,
            remediation="Pass explicit task/trace_id/max_budget for deterministic observability.",
        )
    if text.startswith("AGENT_RETRY_DISABLED:"):
        return _warning_record(
            code="LLMC_WARN_AGENT_RETRY_DISABLED",
            category="UserWarning",
            message=text,
            remediation="Enable agent_retry_safe only for read-only/idempotent agent runs.",
        )
    if text.startswith("GEMINI_NATIVE_SKIP:"):
        return _warning_record(
            code="LLMC_WARN_GEMINI_NATIVE_SKIP",
            category="UserWarning",
            message=text,
            remediation="Adjust kwargs or api_base to use native Gemini path.",
        )
    if text.startswith("TOOL_DISCLOSURE:"):
        return _warning_record(
            code="LLMC_WARN_TOOL_DISCLOSURE",
            category="UserWarning",
            message=text,
        )
    return None


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


def _merge_warning_records(
    *,
    existing: list[dict[str, Any]] | None,
    warnings: list[str] | None,
    requested_model: str,
    extra_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = [dict(r) for r in (existing or []) if isinstance(r, dict)]
    seen: set[tuple[str, str]] = set()
    for rec in merged:
        seen.add((str(rec.get("code", "")), str(rec.get("message", ""))))

    for msg in warnings or []:
        rec = _warning_record_from_message(msg)
        if rec is None:
            continue
        key = (str(rec.get("code", "")), str(rec.get("message", "")))
        if key not in seen:
            merged.append(rec)
            seen.add(key)

    model_rec = _model_warning_record(requested_model)
    if model_rec is not None:
        key = (str(model_rec.get("code", "")), str(model_rec.get("message", "")))
        if key not in seen:
            merged.append(model_rec)
            seen.add(key)

    for rec in extra_records or []:
        if not isinstance(rec, dict):
            continue
        key = (str(rec.get("code", "")), str(rec.get("message", "")))
        if key not in seen:
            merged.append(dict(rec))
            seen.add(key)
    return merged


def _annotate_result_identity(
    result: LLMCallResult,
    *,
    requested_model: str,
    resolved_model: str | None = None,
    routing_trace: dict[str, Any] | None = None,
    warning_records: list[dict[str, Any]] | None = None,
    result_model_semantics: ResultModelSemantics = "legacy",
) -> LLMCallResult:
    """Attach additive model-identity fields without changing legacy model semantics."""
    if result.requested_model is None:
        result.requested_model = requested_model
    if resolved_model is not None and result.resolved_model is None:
        result.resolved_model = resolved_model
    if result.execution_model is None and result.resolved_model is not None:
        result.execution_model = result.resolved_model

    if routing_trace:
        existing = result.routing_trace if isinstance(result.routing_trace, dict) else {}
        merged = dict(existing)
        merged.update(routing_trace)
        result.routing_trace = merged

    if result_model_semantics == "requested":
        result.model = result.requested_model or requested_model
    elif result_model_semantics == "resolved":
        resolved_identity = result.resolved_model or resolved_model
        if resolved_identity:
            result.model = resolved_identity

    result.warning_records = _merge_warning_records(
        existing=result.warning_records,
        warnings=result.warnings,
        requested_model=result.requested_model or requested_model,
        extra_records=warning_records,
    )
    return result


# ---------------------------------------------------------------------------
# Retry infrastructure
# ---------------------------------------------------------------------------

_RETRYABLE_PATTERNS = [
    "rate limit",
    "rate_limit",
    "timeout",
    "timed out",
    "connection reset",
    "connection error",
    "network error",
    "service unavailable",
    "unavailable",
    "high demand",
    "internal server error",
    "server error",
    "overloaded",
    "http 500",
    "http 502",
    "http 503",
    "http 529",
    "empty content",
    "empty response",
    "no json found",
    "json parse error",
    "invalid json",
    "malformed json",
    "unterminated string",
    "expecting",
    "delimiter",
    "temporary failure",
]

_EMPTY_POLICY_FINISH_REASONS = frozenset({
    "blocked",
    "content_filter",
    "recitation",
    "safety",
})

_EMPTY_TOOL_PROTOCOL_FINISH_REASONS = frozenset({
    "malformed_function_call",
    "unexpected_tool_call",
    "too_many_tool_calls",
})

# Patterns in error messages that indicate permanent failure (never retry).
# Checked before _RETRYABLE_PATTERNS so they take precedence.
_NON_RETRYABLE_PATTERNS = [
    "quota",
    "billing",
    "insufficient",
    "exceeded your current",
    "plan and billing",
    "account deactivated",
    "account suspended",
]


def _coerce_retry_delay_seconds(raw_value: Any) -> float | None:
    """Normalize a retry-delay value into seconds with sanity bounds."""
    if raw_value is None:
        return None

    value: float | None = None
    unit = "s"
    if isinstance(raw_value, (int, float)):
        value = float(raw_value)
    else:
        text = str(raw_value).strip()
        if not text:
            return None
        match = re.fullmatch(
            r"([0-9]+(?:\.[0-9]+)?)\s*(ms|s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hour|hours)?",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        try:
            value = float(match.group(1))
        except Exception:
            return None
        unit = (match.group(2) or "s").strip().lower()

    if value is None:
        return None
    if unit in {"ms"}:
        value /= 1000.0
    elif unit in {"m", "min", "mins", "minute", "minutes"}:
        value *= 60.0
    elif unit in {"h", "hour", "hours"}:
        value *= 3600.0
    if value <= 0:
        return None
    # Bound absurd delays while still honoring provider windows.
    return min(value, 600.0)


def _retry_delay_hint_seconds(error: Exception) -> float | None:
    """Parse provider retry-after hints from an error message (seconds)."""
    text = str(error)
    if not text:
        return None

    patterns = [
        r'retry(?:ing)?\s+(?:in|after)\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hour|hours)?',
        r'"retryDelay"\s*:\s*"([0-9]+(?:\.[0-9]+)?)(ms|s|m|h)?"',
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if not m:
            continue
        hint = _coerce_retry_delay_seconds(f"{m.group(1)}{m.group(2) or 's'}")
        if hint is not None:
            return hint
    return None


def _retry_delay_hint(error: Exception) -> tuple[float | None, str]:
    """Return retry hint delay with source classification."""
    for attr in ("retry_after", "retry_after_seconds", "retry_after_s", "retry_delay"):
        hint = _coerce_retry_delay_seconds(getattr(error, attr, None))
        if hint is not None:
            return hint, "structured"

    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        header_value: Any = None
        if hasattr(headers, "get"):
            header_value = headers.get("retry-after") or headers.get("Retry-After")
        hint = _coerce_retry_delay_seconds(header_value)
        if hint is not None:
            return hint, "structured"

    hint = _retry_delay_hint_seconds(error)
    if hint is not None:
        return hint, "parsed"
    return None, "none"


def _is_retryable(error: Exception, extra_patterns: list[str] | None = None) -> bool:
    """Check if an error is transient and worth retrying.

    Uses litellm exception types for reliable classification, with string
    pattern matching as fallback for generic exceptions.
    """
    if isinstance(error, LLMEmptyResponseError):
        return bool(error.retryable)

    # RuntimeError is used for non-retryable conditions (e.g., truncation)
    if isinstance(error, RuntimeError):
        return False

    # -- Check litellm exception types (preferred over string matching) -------
    try:
        import litellm as _lt

        # Permanent failures — never retry
        if isinstance(error, (
            _lt.AuthenticationError,      # 401: bad API key
            _lt.PermissionDeniedError,    # 403: forbidden
            _lt.BudgetExceededError,      # litellm budget limit
            _lt.ContentPolicyViolationError,  # content filter
            _lt.NotFoundError,            # 404: model doesn't exist
        )):
            return False

        # RateLimitError (429) is ambiguous — could be transient rate limit
        # or permanent quota exhaustion. Check the message.
        if isinstance(error, _lt.RateLimitError):
            error_str = str(error).lower()
            # Provider-specified retry windows are considered retryable even
            # when the message includes "quota" phrasing.
            hint_delay, _hint_source = _retry_delay_hint(error)
            if hint_delay is not None:
                return True
            if any(p in error_str for p in _NON_RETRYABLE_PATTERNS):
                return False
            return True  # transient rate limit — retry

        # Transient server errors — always retry
        if isinstance(error, (
            _lt.InternalServerError,   # 500
            _lt.ServiceUnavailableError,  # 503
            _lt.APIConnectionError,    # network issues
            _lt.BadGatewayError,       # 502
        )):
            return True
    except ImportError:
        pass  # litellm not available, fall through to string matching

    # -- Fallback: string pattern matching for generic exceptions --------------
    error_str = str(error).lower()

    # Check non-retryable patterns first
    if any(p in error_str for p in _NON_RETRYABLE_PATTERNS):
        return False

    patterns = _RETRYABLE_PATTERNS
    if extra_patterns:
        patterns = list(patterns) + [p.lower() for p in extra_patterns]
    return any(p in error_str for p in patterns)


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
) -> None:
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


# -- Backoff strategies ----------------------------------------------------


def exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 30.0) -> float:
    """Exponential backoff with jitter, capped at *max_delay*."""
    delay = base_delay * (2 ** attempt)
    jitter = random.uniform(0.5, 1.5)
    return min(delay * jitter, max_delay)


def linear_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 30.0) -> float:
    """Linear backoff with jitter, capped at *max_delay*."""
    delay = base_delay * (attempt + 1)
    jitter = random.uniform(0.8, 1.2)
    return min(delay * jitter, max_delay)


def fixed_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 30.0) -> float:
    """Fixed delay (no escalation), capped at *max_delay*."""
    return min(base_delay, max_delay)


# Backward-compat alias (used by existing tests)
_calculate_backoff = exponential_backoff


# -- RetryPolicy -----------------------------------------------------------


@dataclass
class RetryPolicy:
    """Reusable retry configuration.

    Create once and pass to multiple calls for consistent behaviour::

        policy = RetryPolicy(max_retries=5, base_delay=0.5, on_retry=my_logger)
        call_llm("gpt-4o", msgs, retry=policy)
        call_llm("gpt-4o", msgs2, retry=policy)

    When ``retry`` is provided it **overrides** the individual retry params
    (``num_retries``, ``base_delay``, ``max_delay``, ``retry_on``,
    ``on_retry``).

    Attributes:
        max_retries: How many times to retry on transient failure.
        base_delay: Starting delay for backoff (seconds).
        max_delay: Cap on backoff delay (seconds).
        retry_on: Extra retryable patterns (added to built-in defaults).
        on_retry: ``(attempt, error, delay)`` callback fired before each sleep.
        backoff: Backoff function ``(attempt, base_delay, max_delay) → delay``.
            Defaults to :func:`exponential_backoff`. Also available:
            :func:`linear_backoff`, :func:`fixed_backoff`, or any custom
            callable.
        should_retry: Fully custom retryability check ``(error) → bool``.
            When set, **replaces** the built-in pattern matching entirely.
    """

    max_retries: int = 2
    base_delay: float = 1.0
    max_delay: float = 30.0
    retry_on: list[str] | None = None
    on_retry: Callable[[int, Exception, float], None] | None = None
    backoff: Callable[[int, float, float], float] | None = None
    should_retry: Callable[[Exception], bool] | None = None


@dataclass
class Hooks:
    """Observability hooks fired during LLM calls.

    Attach callbacks for logging, metrics, tracing, or OpenTelemetry
    integration. All fields are optional — set only the ones you need.

    Example::

        hooks = Hooks(
            before_call=lambda model, msgs, kw: print(f"Calling {model}"),
            after_call=lambda result: print(f"Got {len(result.content)} chars"),
            on_error=lambda err, attempt: print(f"Attempt {attempt} failed: {err}"),
        )
        result = call_llm("gpt-4o", messages, hooks=hooks)

    Attributes:
        before_call: ``(model, messages, kwargs) → None``. Fired before each
            LLM API call (including retries and fallbacks).
        after_call: ``(LLMCallResult) → None``. Fired after a successful call.
        on_error: ``(error, attempt) → None``. Fired on each failed attempt.
    """

    before_call: Callable[[str, list[dict[str, Any]], dict[str, Any]], None] | None = None
    after_call: Callable[[LLMCallResult], None] | None = None
    on_error: Callable[[Exception, int], None] | None = None


def _effective_retry(
    retry: RetryPolicy | None,
    num_retries: int,
    base_delay: float,
    max_delay: float,
    retry_on: list[str] | None,
    on_retry: Callable[[int, Exception, float], None] | None,
) -> RetryPolicy:
    """Resolve a RetryPolicy — use the explicit object or build one from individual params."""
    if retry is not None:
        return retry
    return RetryPolicy(
        max_retries=num_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        retry_on=retry_on,
        on_retry=on_retry,
    )


def _check_retryable(error: Exception, policy: RetryPolicy) -> bool:
    """Decide if *error* is retryable according to *policy*."""
    if policy.should_retry is not None:
        return policy.should_retry(error)
    return _is_retryable(error, extra_patterns=policy.retry_on)


def _compute_retry_delay(
    *,
    attempt: int,
    error: Exception,
    policy: RetryPolicy,
    backoff_fn: Callable[[int, float, float], float],
) -> tuple[float, str]:
    """Compute retry delay and source, honoring provider hints when present."""
    delay = backoff_fn(attempt, policy.base_delay, policy.max_delay)
    hint, hint_source = _retry_delay_hint(error)
    if hint is None:
        return delay, "none"
    # Use the larger delay to avoid hammering providers before their window.
    return max(delay, hint), hint_source


# ---------------------------------------------------------------------------
# Async cache helpers
# ---------------------------------------------------------------------------


async def _async_cache_get(cache: Any, key: str) -> LLMCallResult | None:
    """Get from cache, awaiting if the cache is async."""
    result = cache.get(key)
    if inspect.isawaitable(result):
        return await result
    return result  # type: ignore[return-value]


async def _async_cache_set(cache: Any, key: str, value: LLMCallResult) -> None:
    """Set into cache, awaiting if the cache is async."""
    result = cache.set(key, value)
    if inspect.isawaitable(result):
        await result


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class LLMStream:
    """Sync streaming wrapper. Yields text chunks, then exposes ``.result``.

    Example::

        stream = stream_llm("gpt-4o", messages)
        for chunk in stream:
            print(chunk, end="", flush=True)
        print()
        print(stream.result.usage)
    """

    def __init__(
        self, response_iter: Any, model: str, hooks: Hooks | None = None,
        messages: list[dict[str, Any]] | None = None,
        task: str | None = None,
        trace_id: str | None = None,
        warnings: list[str] | None = None,
        requested_model: str | None = None,
        resolved_model: str | None = None,
        routing_trace: dict[str, Any] | None = None,
        result_model_semantics: ResultModelSemantics = "legacy",
    ) -> None:
        self._iter = response_iter
        self._model = model
        self._hooks = hooks
        self._messages = messages
        self._task = task
        self._trace_id = trace_id
        self._warnings = warnings or []
        self._requested_model = requested_model
        self._resolved_model = resolved_model
        self._routing_trace = routing_trace
        self._result_model_semantics = result_model_semantics
        self._t0 = time.monotonic()
        self._chunks_text: list[str] = []
        self._raw_chunks: list[Any] = []
        self._result: LLMCallResult | None = None

    def __iter__(self) -> LLMStream:
        return self

    def __next__(self) -> str:
        try:
            chunk = next(self._iter)
        except StopIteration:
            self._finalize()
            raise
        self._raw_chunks.append(chunk)
        text = ""
        if chunk.choices:
            delta = chunk.choices[0].delta
            text = (delta.content if delta and delta.content else "") or ""
        self._chunks_text.append(text)
        return text

    def _finalize(self) -> None:
        content = "".join(self._chunks_text)
        usage: dict[str, Any] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        cost = 0.0
        finish_reason = "stop"
        tool_calls: list[dict[str, Any]] = []
        try:
            complete = litellm.stream_chunk_builder(self._raw_chunks)
            if complete:
                usage = _extract_usage(complete)
                cost, cost_source = _parse_cost_result(_compute_cost(complete))
                finish_reason = complete.choices[0].finish_reason or "stop"
                if complete.choices[0].message.tool_calls:
                    tool_calls = _extract_tool_calls(complete.choices[0].message)
            else:
                cost_source = "unavailable"
        except Exception:
            cost_source = "unavailable"
        self._result = LLMCallResult(
            content=content,
            usage=usage,
            cost=cost,
            model=self._model,
            requested_model=self._requested_model,
            resolved_model=self._resolved_model or self._model,
            routing_trace=self._routing_trace,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            raw_response=self._raw_chunks[-1] if self._raw_chunks else None,
            warnings=self._warnings,
            cost_source=cost_source,
        )
        self._result = _annotate_result_identity(
            self._result,
            requested_model=self._requested_model or self._model,
            resolved_model=self._resolved_model or self._model,
            routing_trace=self._routing_trace,
            result_model_semantics=self._result_model_semantics,
        )
        if self._hooks and self._hooks.after_call:
            self._hooks.after_call(self._result)
        _io_log.log_call(model=self._model, messages=self._messages, result=self._result, latency_s=time.monotonic() - self._t0, caller="stream_llm", task=self._task, trace_id=self._trace_id)

    @property
    def result(self) -> LLMCallResult:
        """The accumulated result. Available after the stream is fully consumed."""
        if self._result is None:
            raise RuntimeError("Stream not yet consumed. Iterate first.")
        return self._result


class AsyncLLMStream:
    """Async streaming wrapper. Yields text chunks, then exposes ``.result``.

    Example::

        stream = await astream_llm("gpt-4o", messages)
        async for chunk in stream:
            print(chunk, end="", flush=True)
        print()
        print(stream.result.usage)
    """

    def __init__(
        self, response_iter: Any, model: str, hooks: Hooks | None = None,
        messages: list[dict[str, Any]] | None = None,
        task: str | None = None,
        trace_id: str | None = None,
        warnings: list[str] | None = None,
        requested_model: str | None = None,
        resolved_model: str | None = None,
        routing_trace: dict[str, Any] | None = None,
        result_model_semantics: ResultModelSemantics = "legacy",
    ) -> None:
        self._iter = response_iter
        self._model = model
        self._hooks = hooks
        self._messages = messages
        self._task = task
        self._trace_id = trace_id
        self._warnings = warnings or []
        self._requested_model = requested_model
        self._resolved_model = resolved_model
        self._routing_trace = routing_trace
        self._result_model_semantics = result_model_semantics
        self._t0 = time.monotonic()
        self._chunks_text: list[str] = []
        self._raw_chunks: list[Any] = []
        self._result: LLMCallResult | None = None

    def __aiter__(self) -> AsyncLLMStream:
        return self

    async def __anext__(self) -> str:
        try:
            chunk = await self._iter.__anext__()
        except StopAsyncIteration:
            self._finalize()
            raise
        self._raw_chunks.append(chunk)
        text = ""
        if chunk.choices:
            delta = chunk.choices[0].delta
            text = (delta.content if delta and delta.content else "") or ""
        self._chunks_text.append(text)
        return text

    def _finalize(self) -> None:
        content = "".join(self._chunks_text)
        usage: dict[str, Any] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        cost = 0.0
        finish_reason = "stop"
        tool_calls: list[dict[str, Any]] = []
        try:
            complete = litellm.stream_chunk_builder(self._raw_chunks)
            if complete:
                usage = _extract_usage(complete)
                cost, cost_source = _parse_cost_result(_compute_cost(complete))
                finish_reason = complete.choices[0].finish_reason or "stop"
                if complete.choices[0].message.tool_calls:
                    tool_calls = _extract_tool_calls(complete.choices[0].message)
            else:
                cost_source = "unavailable"
        except Exception:
            cost_source = "unavailable"
        self._result = LLMCallResult(
            content=content,
            usage=usage,
            cost=cost,
            model=self._model,
            requested_model=self._requested_model,
            resolved_model=self._resolved_model or self._model,
            routing_trace=self._routing_trace,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            raw_response=self._raw_chunks[-1] if self._raw_chunks else None,
            warnings=self._warnings,
            cost_source=cost_source,
        )
        self._result = _annotate_result_identity(
            self._result,
            requested_model=self._requested_model or self._model,
            resolved_model=self._resolved_model or self._model,
            routing_trace=self._routing_trace,
            result_model_semantics=self._result_model_semantics,
        )
        if self._hooks and self._hooks.after_call:
            self._hooks.after_call(self._result)
        _io_log.log_call(model=self._model, messages=self._messages, result=self._result, latency_s=time.monotonic() - self._t0, caller="astream_llm", task=self._task, trace_id=self._trace_id)

    @property
    def result(self) -> LLMCallResult:
        """The accumulated result. Available after the stream is fully consumed."""
        if self._result is None:
            raise RuntimeError("Stream not yet consumed. Iterate first.")
        return self._result


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


def _is_claude_model(model: str) -> bool:
    """Check if model string refers to a Claude model."""
    return "claude" in model.lower() or "anthropic" in model.lower()


def _is_thinking_model(model: str) -> bool:
    """Check if model needs thinking budget configuration.

    Gemini 2.5+ thinking models allocate reasoning tokens by default,
    consuming output token budget. Setting budget_tokens=0 disables
    this so all tokens go to the actual response.
    """
    lower = model.lower()
    # Gemini 2.5-flash, 2.5-pro, 2.5-flash-lite, 3.x, 4.x are all thinking models
    return "gemini-2.5" in lower or "gemini-3" in lower or "gemini-4" in lower


def _extract_usage(response: Any) -> dict[str, Any]:
    """Extract token usage dict from litellm response.

    Includes provider-level prompt caching details when available
    (OpenAI cached_tokens, DeepSeek prompt_cache_hit_tokens,
    Anthropic cache_read_input_tokens).
    """
    usage = response.usage
    result = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }
    # Extract prompt caching details (litellm normalizes all providers
    # into prompt_tokens_details.cached_tokens)
    ptd = getattr(usage, "prompt_tokens_details", None)
    if ptd is not None:
        cached = getattr(ptd, "cached_tokens", None) or 0
        cache_creation = getattr(ptd, "cache_creation_tokens", None) or 0
        result["cached_tokens"] = cached
        result["cache_creation_tokens"] = cache_creation
    return result


def _compute_cost(response: Any) -> tuple[float, str]:
    """Compute cost via litellm.completion_cost, with explicit source tagging."""
    try:
        cost = float(litellm.completion_cost(completion_response=response))
        return cost, "computed"
    except Exception:
        # Fallback: rough estimate based on total tokens
        total: int = response.usage.total_tokens
        fallback = total * FALLBACK_COST_FLOOR_USD_PER_TOKEN
        logger.warning(
            "completion_cost failed, using fallback: $%.6f for %d tokens",
            fallback,
            total,
        )
        return fallback, "fallback_estimate"


def _parse_cost_result(value: float | tuple[float, str], default_source: str = "computed") -> tuple[float, str]:
    """Normalize cost helper return values.

    Supports both new tuple return and legacy float return to keep monkeypatch
    compatibility in tests and downstream callers.
    """
    if isinstance(value, tuple) and len(value) == 2:
        return float(value[0]), str(value[1])
    return float(value), default_source


def _extract_tool_calls(message: Any) -> list[dict[str, Any]]:
    """Extract tool calls from response message into plain dicts."""
    if not message.tool_calls:
        return []
    result: list[dict[str, Any]] = []
    for tc in message.tool_calls:
        result.append({
            "id": tc.id,
            "type": tc.type,
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            },
        })
    return result


# ---------------------------------------------------------------------------
# Responses API helpers (GPT-5 models)
# ---------------------------------------------------------------------------


_RESPONSES_API_MODELS = {"gpt-5", "gpt-5-mini", "gpt-5-nano"}
_GPT5_ALWAYS_STRIP_SAMPLING = {"gpt-5", "gpt-5-mini", "gpt-5-nano"}
_GPT5_REASONING_GATED_SAMPLING = {
    "gpt-5.1",
    "gpt-5.2",
    "gpt-5.1-chat-latest",
    "gpt-5.2-chat-latest",
}
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

def _is_responses_api_model(model: str) -> bool:
    """Check if model requires litellm.responses() instead of completion().

    GPT-5 models use OpenAI's Responses API which has different parameters
    and response format than the Chat Completions API. This function
    detects them so call_llm/acall_llm can route automatically.

    Only bare OpenAI model names match. Provider-prefixed models
    (openrouter/openai/gpt-5, azure/gpt-5, etc.) use Chat Completions API.
    """
    lower = model.lower()
    # Any provider prefix means proxied → Chat Completions, not Responses API
    if "/" in lower:
        return False
    return lower in _RESPONSES_API_MODELS


def _base_model_name(model: str) -> str:
    """Return the provider-agnostic lowercase model name."""
    return model.lower().rsplit("/", 1)[-1]


def _is_image_generation_model(model: str) -> bool:
    """Best-effort detection for image-generation model families."""
    base = _base_model_name(model)
    hints = (
        "gpt-image",
        "dall-e",
        "imagen",
        "stable-diffusion",
        "sdxl",
        "flux",
    )
    return any(h in base for h in hints)


def _openrouter_routing_enabled() -> bool:
    """Whether automatic OpenRouter model normalization is enabled."""
    raw = os.environ.get(OPENROUTER_ROUTING_ENV, "on").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on", ""}:
        return True
    logger.warning(
        "Invalid %s=%r; expected on/off boolean. Defaulting to on.",
        OPENROUTER_ROUTING_ENV,
        raw,
    )
    return True


def _normalize_model_for_routing(model: str) -> str:
    """Route non-Gemini, non-image model IDs through OpenRouter by default.

    Note: with default routing enabled, bare ``gpt-5*`` model IDs are normalized
    to ``openrouter/openai/gpt-5*`` and therefore use completion routing instead
    of OpenAI Responses API. Disable routing via
    ``LLM_CLIENT_OPENROUTER_ROUTING=off`` to keep bare OpenAI IDs.
    """
    cfg = ClientConfig.from_env()
    req = CallRequest(model=model)
    return resolve_call(req, cfg).primary_model


def _resolve_api_base_for_model(
    model: str,
    api_base: str | None,
    config: ClientConfig | None = None,
) -> str | None:
    """Resolve provider API base after model normalization."""
    cfg = config or ClientConfig.from_env()
    return resolve_api_base_for_model(model, api_base, cfg)


def _is_gemini_model(model: str) -> bool:
    """Check if model targets Google's Gemini API namespace."""
    return model.lower().startswith("gemini/")


def _gemini_native_mode_enabled() -> bool:
    """Whether native Gemini REST path is enabled via env flag."""
    raw = os.environ.get(GEMINI_NATIVE_MODE_ENV, "off").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off", ""}:
        return False
    logger.warning(
        "Invalid %s=%r; expected on/off boolean. Defaulting to off.",
        GEMINI_NATIVE_MODE_ENV,
        raw,
    )
    return False


def _gemini_model_name(model: str) -> str:
    """Return raw Gemini model id without provider prefix."""
    if not _is_gemini_model(model):
        return model
    return model.split("/", 1)[1]


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


def _safe_json_object(value: Any) -> dict[str, Any]:
    """Parse JSON-ish values into dicts for Gemini function args/response."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = _json.loads(value)
            if isinstance(parsed, dict):
                return parsed
            return {"value": parsed}
        except Exception:
            return {"value": value}
    if value is None:
        return {}
    return {"value": value}


def _convert_tools_for_gemini_native(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI tool schema to Gemini functionDeclarations format."""
    declarations: list[dict[str, Any]] = []
    for tool in tools:
        fn = tool.get("function")
        if not isinstance(fn, dict):
            continue
        name = str(fn.get("name", "")).strip()
        if not name:
            continue
        decl: dict[str, Any] = {"name": name}
        description = fn.get("description")
        if isinstance(description, str) and description.strip():
            decl["description"] = description
        parameters = fn.get("parameters")
        if isinstance(parameters, dict):
            # Gemini REST accepts JSON schema under "parameters".
            decl["parameters"] = parameters
        declarations.append(decl)
    if not declarations:
        return []
    return [{"functionDeclarations": declarations}]


def _convert_tool_choice_for_gemini_native(tool_choice: Any) -> dict[str, Any] | None:
    """Convert OpenAI tool_choice to Gemini toolConfig when possible."""
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        choice = tool_choice.strip().lower()
        if choice in {"auto", ""}:
            return None
        if choice == "none":
            return {"functionCallingConfig": {"mode": "NONE"}}
        if choice in {"required", "any"}:
            return {"functionCallingConfig": {"mode": "ANY"}}
        return None
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function")
        if isinstance(fn, dict):
            name = str(fn.get("name", "")).strip()
            if name:
                return {
                    "functionCallingConfig": {
                        "mode": "ANY",
                        "allowedFunctionNames": [name],
                    },
                }
    return None


def _build_gemini_native_payload(
    model: str,
    messages: list[dict[str, Any]],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Translate OpenAI-style messages/tools kwargs into Gemini REST payload."""
    system_lines: list[str] = []
    contents: list[dict[str, Any]] = []
    tool_call_name_by_id: dict[str, str] = {}

    for msg in messages:
        role = str(msg.get("role", "")).strip().lower()
        if role == "system":
            text = _as_text_content(msg.get("content"))
            if text.strip():
                system_lines.append(text.strip())
            continue

        if role == "assistant":
            parts: list[dict[str, Any]] = []
            text = _as_text_content(msg.get("content"))
            if text.strip():
                parts.append({"text": text})

            raw_tool_calls = msg.get("tool_calls")
            if isinstance(raw_tool_calls, list):
                for idx, tc in enumerate(raw_tool_calls):
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function")
                    if not isinstance(fn, dict):
                        continue
                    name = str(fn.get("name", "")).strip()
                    if not name:
                        continue
                    args = _safe_json_object(fn.get("arguments"))
                    tc_id = str(tc.get("id", "")).strip() or f"call_{len(tool_call_name_by_id) + idx + 1}"
                    tool_call_name_by_id[tc_id] = name
                    parts.append({"functionCall": {"name": name, "args": args}})

            if parts:
                contents.append({"role": "model", "parts": parts})
            continue

        if role == "tool":
            tool_call_id = str(msg.get("tool_call_id", "")).strip()
            tool_name = str(msg.get("name", "")).strip() or tool_call_name_by_id.get(tool_call_id, "tool_response")
            response_obj = _safe_json_object(msg.get("content"))
            contents.append({
                "role": "user",
                "parts": [{
                    "functionResponse": {
                        "name": tool_name,
                        "response": response_obj,
                    },
                }],
            })
            continue

        # Default: user/unknown roles become user text turns.
        text = _as_text_content(msg.get("content"))
        if text.strip():
            contents.append({"role": "user", "parts": [{"text": text}]})

    if not contents:
        # Gemini requires at least one content turn.
        contents = [{"role": "user", "parts": [{"text": ""}]}]

    payload: dict[str, Any] = {"contents": contents}
    if system_lines:
        payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_lines)}]}

    tools = kwargs.get("tools")
    if isinstance(tools, list):
        gemini_tools = _convert_tools_for_gemini_native(tools)
        if gemini_tools:
            payload["tools"] = gemini_tools

    tool_config = _convert_tool_choice_for_gemini_native(kwargs.get("tool_choice"))
    if tool_config:
        payload["toolConfig"] = tool_config

    generation_config: dict[str, Any] = {}
    if "temperature" in kwargs:
        generation_config["temperature"] = kwargs["temperature"]
    if "top_p" in kwargs:
        generation_config["topP"] = kwargs["top_p"]
    max_tokens = kwargs.get("max_completion_tokens", kwargs.get("max_tokens"))
    if max_tokens is not None:
        generation_config["maxOutputTokens"] = int(max_tokens)
    stop = kwargs.get("stop")
    if stop is not None:
        if isinstance(stop, str):
            generation_config["stopSequences"] = [stop]
        elif isinstance(stop, list):
            generation_config["stopSequences"] = [s for s in stop if isinstance(s, str)]

    thinking = kwargs.get("thinking")
    if isinstance(thinking, dict):
        budget = thinking.get("budget_tokens", thinking.get("thinkingBudget"))
        if budget is not None:
            generation_config["thinkingConfig"] = {"thinkingBudget": int(budget)}
    elif _is_thinking_model(model):
        # Match default llm_client behavior: disable thinking budget unless explicitly set.
        generation_config["thinkingConfig"] = {"thinkingBudget": 0}

    if generation_config:
        payload["generationConfig"] = generation_config

    return payload


def _gemini_native_unsupported_kwargs(kwargs: dict[str, Any]) -> list[str]:
    """Return sorted kwargs that native Gemini path does not currently support."""
    return sorted(k for k in kwargs.keys() if k not in GEMINI_NATIVE_SUPPORTED_KWARGS)


def _should_use_gemini_native(
    model: str,
    *,
    api_base: str | None,
    kwargs: dict[str, Any],
    warning_sink: list[str] | None = None,
) -> bool:
    """Determine if call should route through native Gemini REST path."""
    if not _is_gemini_model(model):
        return False
    if not _gemini_native_mode_enabled():
        return False
    if api_base is not None:
        msg = (
            f"GEMINI_NATIVE_SKIP: api_base provided for {model}; "
            "using litellm completion route"
        )
        logger.info(msg)
        if warning_sink is not None:
            warning_sink.append(msg)
        return False
    unsupported = _gemini_native_unsupported_kwargs(kwargs)
    if unsupported:
        msg = (
            f"GEMINI_NATIVE_SKIP: unsupported kwargs for {model}: "
            + ", ".join(unsupported)
        )
        logger.info(msg)
        if warning_sink is not None:
            warning_sink.append(msg)
        return False
    return True


def _gemini_native_usage(response: dict[str, Any]) -> dict[str, Any]:
    """Normalize Gemini usageMetadata into llm_client usage schema."""
    raw = response.get("usageMetadata", {}) if isinstance(response, dict) else {}
    if not isinstance(raw, dict):
        raw = {}

    prompt_tokens = int(raw.get("promptTokenCount", raw.get("inputTokenCount", 0)) or 0)
    completion_tokens = int(raw.get("candidatesTokenCount", raw.get("outputTokenCount", 0)) or 0)
    total_tokens = int(raw.get("totalTokenCount", prompt_tokens + completion_tokens) or 0)
    cached_tokens = int(raw.get("cachedContentTokenCount", 0) or 0)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "prompt_tokens_details": {"cached_tokens": cached_tokens},
        "cache_creation_input_tokens": 0,
    }


def _build_result_from_gemini_native(
    response: dict[str, Any],
    model: str,
    warnings: list[str] | None = None,
) -> LLMCallResult:
    """Build ``LLMCallResult`` from native Gemini REST response payload."""
    candidates = response.get("candidates") if isinstance(response, dict) else None
    if not isinstance(candidates, list) or not candidates:
        _raise_empty_response(
            provider="gemini_native",
            classification="provider_empty_candidates",
            retryable=True,
            diagnostics={
                "model": model,
                "has_candidates": isinstance(candidates, list),
                "candidate_count": len(candidates) if isinstance(candidates, list) else 0,
                "has_prompt_feedback": isinstance(response.get("promptFeedback"), dict),
            },
        )

    first = candidates[0] if isinstance(candidates[0], dict) else {}
    finish_raw = str(first.get("finishReason", "")).strip()
    finish_reason = finish_raw.lower()
    content_obj = first.get("content")
    parts = content_obj.get("parts", []) if isinstance(content_obj, dict) else []
    prompt_feedback = response.get("promptFeedback", {}) if isinstance(response, dict) else {}
    block_reason = ""
    block_reason_message = ""
    if isinstance(prompt_feedback, dict):
        block_reason = str(prompt_feedback.get("blockReason", "")).strip()
        block_reason_message = str(prompt_feedback.get("blockReasonMessage", "")).strip()

    text_segments: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    blocked_safety_categories: list[str] = []
    for idx, part in enumerate(parts):
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str) and text:
            text_segments.append(text)
        fn_call = part.get("functionCall")
        if isinstance(fn_call, dict):
            name = str(fn_call.get("name", "")).strip()
            if not name:
                continue
            args = fn_call.get("args", {})
            args_json = args if isinstance(args, str) else _json.dumps(args)
            tool_calls.append({
                "id": f"gemini_fn_{idx}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": args_json,
                },
            })
    safety_ratings = first.get("safetyRatings")
    if isinstance(safety_ratings, list):
        for sr in safety_ratings:
            if not isinstance(sr, dict):
                continue
            if sr.get("blocked") is True:
                category = str(sr.get("category", "unknown")).strip() or "unknown"
                blocked_safety_categories.append(category)

    content = "".join(text_segments).strip()
    if finish_reason in {"max_tokens", "max_output_tokens"} and not tool_calls:
        raise RuntimeError(
            f"LLM response truncated ({len(content)} chars). "
            "Increase max_tokens or simplify the prompt."
        )

    if not content and not tool_calls:
        diagnostics = {
            "model": model,
            "finish_reason": finish_raw or "unknown",
            "prompt_block_reason": block_reason or None,
            "prompt_block_reason_message": block_reason_message or None,
            "blocked_safety_categories": blocked_safety_categories,
            "candidate_count": len(candidates),
            "parts_count": len(parts),
        }
        if (
            block_reason
            or blocked_safety_categories
            or finish_reason in _EMPTY_POLICY_FINISH_REASONS
        ):
            _raise_empty_response(
                provider="gemini_native",
                classification="provider_policy_block",
                retryable=False,
                diagnostics=diagnostics,
            )
        if finish_reason in _EMPTY_TOOL_PROTOCOL_FINISH_REASONS:
            _raise_empty_response(
                provider="gemini_native",
                classification="provider_tool_protocol",
                retryable=False,
                diagnostics=diagnostics,
            )
        _raise_empty_response(
            provider="gemini_native",
            classification="provider_empty_unknown",
            retryable=True,
            diagnostics=diagnostics,
        )

    usage = _gemini_native_usage(response)
    total = int(usage.get("total_tokens", 0) or 0)
    cost = total * FALLBACK_COST_FLOOR_USD_PER_TOKEN
    result_finish = "tool_calls" if tool_calls else (finish_reason or "stop")

    return LLMCallResult(
        content=content,
        usage=usage,
        cost=cost,
        model=model,
        resolved_model=model,
        tool_calls=tool_calls,
        finish_reason=result_finish,
        raw_response=response,
        warnings=warnings or [],
        cost_source="fallback_estimate",
    )


def _call_gemini_native(
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout: int,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Direct Gemini REST call (bypasses OpenAI-compat translation layers)."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required for native Gemini mode")

    gemini_model = _gemini_model_name(model)
    url = GEMINI_NATIVE_ENDPOINT.format(model=urllib.parse.quote(gemini_model, safe=".-_"))
    payload = _build_gemini_native_payload(model, messages, kwargs)
    body = _json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        message = f"Gemini native HTTP {e.code}: {err_body[:500]}"
        if e.code in {429, 500, 502, 503, 504}:
            raise ValueError(message)
        raise RuntimeError(message)
    except urllib.error.URLError as e:
        raise ValueError(f"Gemini native network error: {e}") from e

    try:
        parsed = _json.loads(raw)
    except Exception as e:
        raise ValueError("Gemini native returned invalid JSON") from e

    if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
        err = parsed["error"]
        code = int(err.get("code", 0) or 0)
        status = str(err.get("status", "")).strip()
        msg = str(err.get("message", "")).strip()
        rendered = f"Gemini native API error {code} {status}: {msg}".strip()
        if code in {429, 500, 502, 503, 504} or ("rate" in msg.lower() and "limit" in msg.lower()):
            raise ValueError(rendered)
        raise RuntimeError(rendered)

    if not isinstance(parsed, dict):
        raise ValueError("Gemini native returned non-object JSON payload")
    return parsed


async def _acall_gemini_native(
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout: int,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Async wrapper around native Gemini REST call."""
    return await asyncio.to_thread(
        _call_gemini_native,
        model,
        messages,
        timeout=timeout,
        kwargs=kwargs,
    )


def _build_model_chain(
    model: str,
    fallback_models: list[str] | None,
    config: ClientConfig | None = None,
) -> list[str]:
    """Build primary+fallback model chain with stable de-duplication."""
    cfg = config or ClientConfig.from_env()
    plan = resolve_call(
        CallRequest(model=model, fallback_models=fallback_models),
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
    return plan.models


def _truthy_env(value: Any) -> bool:
    """Parse common truthy env-style values."""
    if isinstance(value, bool):
        return value
    raw = str(value or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _tags_strict_mode(task: str | None) -> bool:
    """Whether missing task/trace/budget tags should raise instead of defaulting."""
    if _truthy_env(os.environ.get(REQUIRE_TAGS_ENV)):
        return True
    if _truthy_env(os.environ.get("CI")):
        return True
    t = str(task or "").strip().lower()
    return t.startswith(("benchmark", "bench", "eval", "ci"))


def _agent_retry_safe_enabled(explicit: Any | None) -> bool:
    """Whether retries on agent SDK calls are allowed."""
    if explicit is not None:
        return _truthy_env(explicit)
    return _truthy_env(os.environ.get(AGENT_RETRY_SAFE_ENV))


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
        if non_agent:
            raise LLMCapabilityError(
                f"{caller}: agent-only kwargs {agent_only} are incompatible with "
                f"non-agent model(s) {non_agent}. Use codex/claude-code or remove "
                "agent-only kwargs."
            )


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


def _require_tags(
    task: str | None,
    trace_id: str | None,
    max_budget: float | None,
    *,
    caller: str,
) -> tuple[str, str, float, list[str]]:
    """Resolve observability tags.

    In strict mode (CI, benchmark/eval tasks, or ``LLM_CLIENT_REQUIRE_TAGS=1``),
    missing tags raise. Otherwise they are auto-populated.
    """
    missing = []
    if not task:
        missing.append("task")
    if not trace_id:
        missing.append("trace_id")
    if max_budget is None:
        missing.append("max_budget")
    strict_mode = _tags_strict_mode(task)
    if strict_mode and missing:
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
    # Optional org-level guardrail: benchmark/eval tasks should run inside an
    # active experiment context so observability is complete and comparable.
    _io_log.enforce_experiment_context(resolved_task, caller="llm_client.client")
    return resolved_task, resolved_trace_id, resolved_max_budget, auto_warnings


def _check_budget(trace_id: str, max_budget: float) -> None:
    """Check if trace has exceeded its budget. Raises LLMBudgetExceededError."""
    if max_budget <= 0:
        return
    spent = _io_log.get_cost(trace_id=trace_id)
    if spent >= max_budget:
        raise LLMBudgetExceededError(
            f"Budget exceeded for trace {trace_id}: "
            f"${spent:.4f} spent >= ${max_budget:.4f} limit"
        )


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
        "timeout": timeout,
    }

    if api_base is not None:
        resp_kwargs["api_base"] = api_base

    # Convert response_format → text parameter
    response_format = kwargs.pop("response_format", None)
    if response_format:
        resp_kwargs["text"] = _convert_response_format_for_responses(
            response_format
        )

    # Strip parameters that break GPT-5 or don't apply to responses API
    for key in ("max_tokens", "max_output_tokens", "messages",
                "reasoning_effort", "thinking", "temperature", "unsupported_param_policy"):
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
# Completion API helpers
# ---------------------------------------------------------------------------


def _apply_max_tokens(model: str, call_kwargs: dict[str, Any]) -> None:
    """Auto-set max output tokens to model's max, or clamp caller's value.

    If no max_tokens/max_completion_tokens is set, defaults to the model's
    maximum output tokens. If one is set, clamps it to the model's max to
    prevent "value X but max is X-1" errors across providers.
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
        # Default to model's max
        call_kwargs["max_completion_tokens"] = model_max


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
        "timeout": timeout,
        # Don't pass num_retries to litellm — our own retry loop handles
        # all retries with jittered backoff. Passing it to litellm causes
        # double retry (litellm retries HTTP errors internally, then our
        # loop retries the same errors again).
        **raw_kwargs,
    }

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

    # Thinking model detection: set budget_tokens=0 to disable reasoning tokens.
    # For Gemini 2.5+, thinkingBudget=0 disables thinking. An empty
    # thinkingConfig {} means "use default" which enables thinking.
    if _is_thinking_model(model) and "thinking" not in raw_kwargs:
        call_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 0}

    # Guard against GPT-5-family sampling param incompatibilities across
    # providers (e.g., provider-prefixed GPT-5 models on completion path).
    _coerce_model_incompatible_params(
        model=model,
        kwargs=call_kwargs,
        policy=policy,
        warning_sink=warning_sink,
    )

    # Auto-set max_tokens to model's max if not specified, or clamp if too high.
    # Prevents "65536 but max is 65535" errors across different models.
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
                  For GPT-5 models, response_format is automatically
                  converted and max_tokens is stripped.
                  For agent models, agent-specific kwargs are extracted:
                  allowed_tools, cwd, max_turns, max_tool_calls, permission_mode,
                  max_budget_usd.

    Returns:
        LLMCallResult with content, usage, cost, model, tool_calls,
        finish_reason, and raw_response
    """
    _check_model_deprecation(model)
    explicit_config = config is not None
    cfg = config or ClientConfig.from_env()
    _log_t0 = time.monotonic()
    task = kwargs.pop("task", None)
    trace_id = kwargs.pop("trace_id", None)
    max_budget: float | None = kwargs.pop("max_budget", None)
    agent_retry_safe = kwargs.pop("agent_retry_safe", None)
    task, trace_id, max_budget, _entry_warnings = _require_tags(
        task, trace_id, max_budget, caller="call_llm",
    )
    _check_budget(trace_id, max_budget)
    _record_semantics_adoption(
        cfg=cfg,
        explicit_config=explicit_config,
        caller="call_llm",
        task=task,
        trace_id=trace_id,
    )

    # Named params that must flow through to per-turn _inner_acall_llm calls
    # inside the agent loop (retry, fallback, hooks, reasoning, api_base).
    _inner_named: dict[str, Any] = {}
    if num_retries != 2:
        _inner_named["num_retries"] = num_retries
    if base_delay != 1.0:
        _inner_named["base_delay"] = base_delay
    if max_delay != 30.0:
        _inner_named["max_delay"] = max_delay
    if retry_on is not None:
        _inner_named["retry_on"] = retry_on
    if on_retry is not None:
        _inner_named["on_retry"] = on_retry
    if retry is not None:
        _inner_named["retry"] = retry
    if fallback_models is not None:
        _inner_named["fallback_models"] = fallback_models
    if on_fallback is not None:
        _inner_named["on_fallback"] = on_fallback
    if reasoning_effort is not None:
        _inner_named["reasoning_effort"] = reasoning_effort
    if api_base is not None:
        _inner_named["api_base"] = api_base
    if hooks is not None:
        _inner_named["hooks"] = hooks
    if execution_mode != "text":
        _inner_named["execution_mode"] = execution_mode
    _inner_named["config"] = cfg

    plan = resolve_call(
        CallRequest(model=model, fallback_models=fallback_models, api_base=api_base),
        cfg,
    )
    models = plan.models
    primary_model = plan.primary_model
    fallback_chain = plan.fallback_models or None
    routing_policy = str(plan.routing_trace.get("routing_policy", _routing_policy_label(cfg)))
    if fallback_chain is not None:
        _inner_named["fallback_models"] = fallback_chain
    else:
        _inner_named.pop("fallback_models", None)
    _validate_execution_contract(
        models=models,
        execution_mode=execution_mode,
        kwargs=kwargs,
        caller="call_llm",
    )

    # MCP agent loop: non-agent model + (mcp_servers or mcp_sessions) → tool-calling loop
    if ("mcp_servers" in kwargs or "mcp_sessions" in kwargs) and not _is_agent_model(model):
        from llm_client.mcp_agent import MCP_LOOP_KWARGS, _acall_with_mcp
        from llm_client.agents import _run_sync
        mcp_kw: dict[str, Any] = {}
        remaining = dict(kwargs)
        remaining["task"] = task
        remaining["trace_id"] = trace_id
        remaining["max_budget"] = max_budget
        for k in MCP_LOOP_KWARGS:
            if k in remaining:
                mcp_kw[k] = remaining.pop(k)
        result = _run_sync(_acall_with_mcp(
            primary_model, messages, timeout=timeout, **_inner_named, **mcp_kw, **remaining,
        ))
        result = _annotate_result_identity(
            result,
            requested_model=model,
            resolved_model=result.resolved_model,
            routing_trace=_build_routing_trace(
                requested_model=model,
                attempted_models=[primary_model],
                selected_model=result.resolved_model,
                requested_api_base=api_base,
                effective_api_base=_resolve_api_base_for_model(primary_model, api_base, cfg),
                sticky_fallback=any("STICKY_FALLBACK" in w for w in (result.warnings or [])),
                routing_policy=routing_policy,
            ),
            result_model_semantics=cfg.result_model_semantics,
        )
        _io_log.log_call(model=primary_model, messages=messages, result=result, latency_s=time.monotonic() - _log_t0, caller="call_llm", task=task, trace_id=trace_id)
        return result

    # Direct Python tool loop: non-agent model + python_tools → in-process tool-calling loop
    if "python_tools" in kwargs and not _is_agent_model(model):
        if "mcp_servers" in kwargs or "mcp_sessions" in kwargs:
            raise ValueError("python_tools and mcp_servers/mcp_sessions are mutually exclusive.")
        from llm_client.mcp_agent import TOOL_LOOP_KWARGS, _acall_with_tools
        from llm_client.agents import _run_sync
        from llm_client.models import supports_tool_calling
        tool_kw: dict[str, Any] = {}
        remaining = dict(kwargs)
        remaining["task"] = task
        remaining["trace_id"] = trace_id
        remaining["max_budget"] = max_budget
        for k in TOOL_LOOP_KWARGS:
            if k in remaining:
                tool_kw[k] = remaining.pop(k)
        if not supports_tool_calling(model):
            from llm_client.tool_shim import _acall_with_tool_shim
            result = _run_sync(_acall_with_tool_shim(
                primary_model, messages, timeout=timeout, **_inner_named, **tool_kw, **remaining,
            ))
        else:
            result = _run_sync(_acall_with_tools(
                primary_model, messages, timeout=timeout, **_inner_named, **tool_kw, **remaining,
            ))
        result = _annotate_result_identity(
            result,
            requested_model=model,
            resolved_model=result.resolved_model,
            routing_trace=_build_routing_trace(
                requested_model=model,
                attempted_models=[primary_model],
                selected_model=result.resolved_model,
                requested_api_base=api_base,
                effective_api_base=_resolve_api_base_for_model(primary_model, api_base, cfg),
                sticky_fallback=any("STICKY_FALLBACK" in w for w in (result.warnings or [])),
                routing_policy=routing_policy,
            ),
            result_model_semantics=cfg.result_model_semantics,
        )
        _io_log.log_call(model=primary_model, messages=messages, result=result, latency_s=time.monotonic() - _log_t0, caller="call_llm", task=task, trace_id=trace_id)
        return result

    r = _effective_retry(retry, num_retries, base_delay, max_delay, retry_on, on_retry)
    if cache is not None and _is_agent_model(model):
        raise ValueError("Caching not supported for agent models — they have side effects.")
    last_error: Exception | None = None
    _warnings: list[str] = list(_entry_warnings)
    agent_retry_safe_enabled = _agent_retry_safe_enabled(agent_retry_safe)

    for model_idx, current_model in enumerate(models):
        is_agent = _is_agent_model(current_model)
        use_responses = not is_agent and _is_responses_api_model(current_model)
        current_api_base = _resolve_api_base_for_model(current_model, api_base, cfg)
        use_gemini_native = (
            not is_agent
            and not use_responses
            and _should_use_gemini_native(
                current_model,
                api_base=current_api_base,
                kwargs=kwargs,
                warning_sink=_warnings,
            )
        )

        if is_agent:
            pass  # No kwargs preparation needed for agent models
        elif use_responses:
            call_kwargs = _prepare_responses_kwargs(
                current_model, messages,
                timeout=timeout,
                api_base=current_api_base,
                kwargs=kwargs,
                warning_sink=_warnings,
            )
        elif use_gemini_native:
            pass  # Native Gemini route builds payload per-attempt.
        else:
            call_kwargs = _prepare_call_kwargs(
                current_model, messages,
                timeout=timeout,
                num_retries=r.max_retries,
                reasoning_effort=reasoning_effort,
                api_base=current_api_base,
                kwargs=kwargs,
                warning_sink=_warnings,
            )

        if cache is not None:
            key = _cache_key(current_model, messages, **kwargs)
            cached = cache.get(key)
            if cached is not None:
                cached_result = _mark_cache_hit(cached)
                cached_result = _annotate_result_identity(
                    cached_result,
                    requested_model=model,
                    resolved_model=current_model,
                    routing_trace=_build_routing_trace(
                        requested_model=model,
                        attempted_models=models[:model_idx + 1],
                        selected_model=current_model,
                        requested_api_base=api_base,
                        effective_api_base=current_api_base,
                        routing_policy=routing_policy,
                    ),
                    result_model_semantics=cfg.result_model_semantics,
                )
                _io_log.log_call(
                    model=current_model,
                    messages=messages,
                    result=cached_result,
                    latency_s=time.monotonic() - _log_t0,
                    caller="call_llm",
                    task=task,
                    trace_id=trace_id,
                )
                return cached_result

        if hooks and hooks.before_call:
            hooks.before_call(current_model, messages, kwargs)

        backoff_fn = r.backoff or exponential_backoff
        if is_agent and not agent_retry_safe_enabled:
            effective_retries = 0
            if r.max_retries > 0:
                msg = (
                    "AGENT_RETRY_DISABLED: retries for agent models are disabled by default "
                    "to avoid duplicate side effects. Set agent_retry_safe=True (or "
                    f"{AGENT_RETRY_SAFE_ENV}=1) only for explicitly safe/read-only runs."
                )
                if msg not in _warnings:
                    _warnings.append(msg)
                    logger.warning(msg)
        else:
            effective_retries = r.max_retries
        try:
            for attempt in range(effective_retries + 1):
                try:
                    if is_agent:
                        from llm_client.agents import _route_call
                        result = _route_call(
                            current_model, messages,
                            timeout=timeout, **kwargs,
                        )
                    elif use_responses:
                        with _rate_limit.acquire(current_model):
                            response = litellm.responses(**call_kwargs)
                        result = _build_result_from_responses(response, current_model, warnings=_warnings)
                    elif use_gemini_native:
                        with _rate_limit.acquire(current_model):
                            response = _call_gemini_native(
                                current_model,
                                messages,
                                timeout=timeout,
                                kwargs=kwargs,
                            )
                        result = _build_result_from_gemini_native(response, current_model, warnings=_warnings)
                    else:
                        with _rate_limit.acquire(current_model):
                            response = litellm.completion(**call_kwargs)
                        result = _build_result_from_response(response, current_model, warnings=_warnings)
                    if attempt > 0:
                        logger.info("call_llm succeeded after %d retries", attempt)
                    if is_agent:
                        resolved_model = result.resolved_model
                    else:
                        resolved_model = current_model
                    result = _annotate_result_identity(
                        result,
                        requested_model=model,
                        resolved_model=resolved_model,
                        routing_trace=_build_routing_trace(
                            requested_model=model,
                            attempted_models=models[:model_idx + 1],
                            selected_model=resolved_model,
                            requested_api_base=api_base,
                            effective_api_base=current_api_base,
                            sticky_fallback=any("STICKY_FALLBACK" in w for w in (result.warnings or [])),
                            routing_policy=routing_policy,
                        ),
                        result_model_semantics=cfg.result_model_semantics,
                    )
                    if hooks and hooks.after_call:
                        hooks.after_call(result)
                    if cache is not None:
                        cache.set(key, result)
                    _io_log.log_call(model=current_model, messages=messages, result=result, latency_s=time.monotonic() - _log_t0, caller="call_llm", task=task, trace_id=trace_id)
                    return result
                except Exception as e:
                    last_error = e
                    if hooks and hooks.on_error:
                        hooks.on_error(e, attempt)
                    if not _check_retryable(e, r) or attempt >= effective_retries:
                        raise
                    delay, retry_delay_source = _compute_retry_delay(
                        attempt=attempt,
                        error=e,
                        policy=r,
                        backoff_fn=backoff_fn,
                    )
                    if r.on_retry is not None:
                        r.on_retry(attempt, e, delay)
                    _warnings.append(
                        f"RETRY {attempt + 1}/{effective_retries + 1}: "
                        f"{current_model} ({type(e).__name__}: {e}) "
                        f"[retry_delay_source={retry_delay_source}]"
                    )
                    logger.warning(
                        "call_llm attempt %d/%d failed (retrying in %.1fs, source=%s): %s",
                        attempt + 1,
                        effective_retries + 1,
                        delay,
                        retry_delay_source,
                        e,
                    )
                    time.sleep(delay)
            raise wrap_error(last_error) from last_error  # type: ignore[misc]  # unreachable
        except Exception as e:
            last_error = e
            if model_idx < len(models) - 1:
                next_model = models[model_idx + 1]
                if on_fallback is not None:
                    on_fallback(current_model, e, next_model)
                _warnings.append(
                    f"FALLBACK: {current_model} -> {next_model} "
                    f"({type(e).__name__}: {e})"
                )
                logger.warning(
                    "Falling back from %s to %s: %s", current_model, next_model, e,
                )
                continue
            _io_log.log_call(model=current_model, messages=messages, error=e, latency_s=time.monotonic() - _log_t0, caller="call_llm", task=task, trace_id=trace_id)
            raise wrap_error(e) from e

    raise wrap_error(last_error) from last_error  # type: ignore[misc]  # unreachable


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
        **kwargs: Additional params passed to litellm.completion

    Returns:
        Tuple of (parsed Pydantic model instance, LLMCallResult)
    """
    _check_model_deprecation(model)
    explicit_config = config is not None
    cfg = config or ClientConfig.from_env()
    _log_t0 = time.monotonic()
    task = kwargs.pop("task", None)
    trace_id = kwargs.pop("trace_id", None)
    max_budget: float | None = kwargs.pop("max_budget", None)
    task, trace_id, max_budget, _entry_warnings = _require_tags(
        task, trace_id, max_budget, caller="call_llm_structured",
    )
    _check_budget(trace_id, max_budget)
    _record_semantics_adoption(
        cfg=cfg,
        explicit_config=explicit_config,
        caller="call_llm_structured",
        task=task,
        trace_id=trace_id,
    )
    plan = resolve_call(
        CallRequest(model=model, fallback_models=fallback_models, api_base=api_base),
        cfg,
    )
    models = plan.models
    routing_policy = str(plan.routing_trace.get("routing_policy", _routing_policy_label(cfg)))

    if _is_agent_model(model):
        from llm_client.agents import _route_call_structured
        if hooks and hooks.before_call:
            hooks.before_call(model, messages, kwargs)
        parsed, llm_result = _route_call_structured(
            model, messages, response_model, timeout=timeout, **kwargs,
        )
        llm_result = _annotate_result_identity(
            llm_result,
            requested_model=model,
            resolved_model=llm_result.resolved_model,
            routing_trace=_build_routing_trace(
                requested_model=model,
                attempted_models=[plan.primary_model],
                selected_model=llm_result.resolved_model,
                requested_api_base=api_base,
                effective_api_base=api_base,
                routing_policy=routing_policy,
            ),
            result_model_semantics=cfg.result_model_semantics,
        )
        if hooks and hooks.after_call:
            hooks.after_call(llm_result)
        _io_log.log_call(model=model, messages=messages, result=llm_result, latency_s=time.monotonic() - _log_t0, caller="call_llm_structured", task=task, trace_id=trace_id)
        return parsed, llm_result
    r = _effective_retry(retry, num_retries, base_delay, max_delay, retry_on, on_retry)
    last_error: Exception | None = None
    _warnings: list[str] = list(_entry_warnings)

    _model_fqn = f"{response_model.__module__}.{response_model.__qualname__}"

    for model_idx, current_model in enumerate(models):
        current_api_base = _resolve_api_base_for_model(current_model, api_base, cfg)
        if cache is not None:
            key = _cache_key(current_model, messages, response_model=_model_fqn, **kwargs)
            cached = cache.get(key)
            if cached is not None:
                reparsed = response_model.model_validate_json(cached.content)
                cached_result = _mark_cache_hit(cached)
                cached_result = _annotate_result_identity(
                    cached_result,
                    requested_model=model,
                    resolved_model=current_model,
                    routing_trace=_build_routing_trace(
                        requested_model=model,
                        attempted_models=models[:model_idx + 1],
                        selected_model=current_model,
                        requested_api_base=api_base,
                        effective_api_base=current_api_base,
                        routing_policy=routing_policy,
                    ),
                    result_model_semantics=cfg.result_model_semantics,
                )
                _io_log.log_call(
                    model=current_model,
                    messages=messages,
                    result=cached_result,
                    latency_s=time.monotonic() - _log_t0,
                    caller="call_llm_structured",
                    task=task,
                    trace_id=trace_id,
                )
                return reparsed, cached_result

        if hooks and hooks.before_call:
            hooks.before_call(current_model, messages, kwargs)

        backoff_fn = r.backoff or exponential_backoff
        try:
            if _is_responses_api_model(current_model):
                # GPT-5 path: Responses API with native JSON schema
                schema = _strict_json_schema(response_model.model_json_schema())
                resp_kwargs = _prepare_responses_kwargs(
                    current_model, messages,
                    timeout=timeout, api_base=current_api_base, kwargs=kwargs,
                    warning_sink=_warnings,
                )
                resp_kwargs["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": response_model.__name__,
                        "schema": schema,
                        "strict": True,
                    }
                }

                for attempt in range(r.max_retries + 1):
                    try:
                        with _rate_limit.acquire(current_model):
                            response = litellm.responses(**resp_kwargs)
                        raw_content = getattr(response, "output_text", None) or ""
                        if not raw_content.strip():
                            raise ValueError("Empty content from LLM (responses API structured)")
                        parsed = response_model.model_validate_json(raw_content)
                        content = str(parsed.model_dump_json())
                        usage = _extract_responses_usage(response)
                        cost, cost_source = _parse_cost_result(_compute_responses_cost(response, usage), default_source="computed")

                        if attempt > 0:
                            logger.info("call_llm_structured (responses) succeeded after %d retries", attempt)

                        llm_result = LLMCallResult(
                            content=content, usage=usage, cost=cost,
                            model=current_model, finish_reason="stop",
                            raw_response=response, warnings=_warnings,
                            cost_source=cost_source,
                        )
                        llm_result = _annotate_result_identity(
                            llm_result,
                            requested_model=model,
                            resolved_model=current_model,
                            routing_trace=_build_routing_trace(
                                requested_model=model,
                                attempted_models=models[:model_idx + 1],
                                selected_model=current_model,
                                requested_api_base=api_base,
                                effective_api_base=current_api_base,
                                routing_policy=routing_policy,
                            ),
                            result_model_semantics=cfg.result_model_semantics,
                        )
                        if hooks and hooks.after_call:
                            hooks.after_call(llm_result)
                        if cache is not None:
                            cache.set(key, llm_result)
                        _io_log.log_call(model=current_model, messages=messages, result=llm_result, latency_s=time.monotonic() - _log_t0, caller="call_llm_structured", task=task, trace_id=trace_id)
                        return parsed, llm_result
                    except Exception as e:
                        last_error = e
                        if hooks and hooks.on_error:
                            hooks.on_error(e, attempt)
                        if not _check_retryable(e, r) or attempt >= r.max_retries:
                            raise
                        delay, retry_delay_source = _compute_retry_delay(
                            attempt=attempt,
                            error=e,
                            policy=r,
                            backoff_fn=backoff_fn,
                        )
                        if r.on_retry is not None:
                            r.on_retry(attempt, e, delay)
                        _warnings.append(
                            f"RETRY {attempt + 1}/{r.max_retries + 1}: "
                            f"{current_model} ({type(e).__name__}: {e}) "
                            f"[retry_delay_source={retry_delay_source}]"
                        )
                        logger.warning(
                            "call_llm_structured attempt %d/%d failed (retrying in %.1fs, source=%s): %s",
                            attempt + 1, r.max_retries + 1, delay, retry_delay_source, e,
                        )
                        time.sleep(delay)
                raise wrap_error(last_error) from last_error  # type: ignore[misc]  # unreachable
            elif litellm.supports_response_schema(model=current_model):
                # Native JSON schema path: litellm.completion + response_format
                # If the provider rejects the schema (e.g. Gemini nesting depth
                # limit), fall through to the instructor path automatically.
                _native_schema_failed = False
                schema = _strict_json_schema(response_model.model_json_schema())
                base_kwargs = _prepare_call_kwargs(
                    current_model, messages,
                    timeout=timeout,
                    num_retries=r.max_retries,
                    reasoning_effort=reasoning_effort,
                    api_base=current_api_base,
                    kwargs=kwargs,
                    warning_sink=_warnings,
                )
                base_kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "schema": schema,
                        "strict": True,
                    },
                }

                for attempt in range(r.max_retries + 1):
                    try:
                        with _rate_limit.acquire(current_model):
                            response = litellm.completion(**base_kwargs)
                        first_choice = _first_choice_or_empty_error(
                            response,
                            model=current_model,
                            provider="litellm_completion_structured",
                        )
                        raw_content = first_choice.message.content or ""
                        if not raw_content.strip():
                            raise ValueError("Empty content from LLM (native JSON schema structured)")
                        parsed = response_model.model_validate_json(raw_content)
                        content = str(parsed.model_dump_json())
                        usage = _extract_usage(response)
                        cost, cost_source = _parse_cost_result(_compute_cost(response))
                        finish_reason: str = first_choice.finish_reason or "stop"

                        if attempt > 0:
                            logger.info("call_llm_structured (native schema) succeeded after %d retries", attempt)

                        llm_result = LLMCallResult(
                            content=content, usage=usage, cost=cost,
                            model=current_model, finish_reason=finish_reason,
                            raw_response=response, warnings=_warnings,
                            cost_source=cost_source,
                        )
                        llm_result = _annotate_result_identity(
                            llm_result,
                            requested_model=model,
                            resolved_model=current_model,
                            routing_trace=_build_routing_trace(
                                requested_model=model,
                                attempted_models=models[:model_idx + 1],
                                selected_model=current_model,
                                requested_api_base=api_base,
                                effective_api_base=current_api_base,
                                routing_policy=routing_policy,
                            ),
                            result_model_semantics=cfg.result_model_semantics,
                        )
                        if hooks and hooks.after_call:
                            hooks.after_call(llm_result)
                        if cache is not None:
                            cache.set(key, llm_result)
                        _io_log.log_call(model=current_model, messages=messages, result=llm_result, latency_s=time.monotonic() - _log_t0, caller="call_llm_structured", task=task, trace_id=trace_id)
                        return parsed, llm_result
                    except Exception as e:
                        if _is_schema_error(e):
                            logger.warning(
                                "Native JSON schema rejected by provider (%s), "
                                "falling back to instructor: %s",
                                current_model, e,
                            )
                            _native_schema_failed = True
                            break
                        last_error = e
                        if hooks and hooks.on_error:
                            hooks.on_error(e, attempt)
                        if not _check_retryable(e, r) or attempt >= r.max_retries:
                            raise
                        delay, retry_delay_source = _compute_retry_delay(
                            attempt=attempt,
                            error=e,
                            policy=r,
                            backoff_fn=backoff_fn,
                        )
                        if r.on_retry is not None:
                            r.on_retry(attempt, e, delay)
                        _warnings.append(
                            f"RETRY {attempt + 1}/{r.max_retries + 1}: "
                            f"{current_model} ({type(e).__name__}: {e}) "
                            f"[retry_delay_source={retry_delay_source}]"
                        )
                        logger.warning(
                            "call_llm_structured attempt %d/%d failed (retrying in %.1fs, source=%s): %s",
                            attempt + 1, r.max_retries + 1, delay, retry_delay_source, e,
                        )
                        time.sleep(delay)
                else:
                    raise wrap_error(last_error) from last_error  # type: ignore[misc]  # unreachable

            if not litellm.supports_response_schema(model=current_model) or _native_schema_failed:
                # Fallback path: instructor + litellm.completion
                import instructor

                client = instructor.from_litellm(litellm.completion)
                base_kwargs = _prepare_call_kwargs(
                    current_model, messages,
                    timeout=timeout,
                    num_retries=r.max_retries,
                    reasoning_effort=reasoning_effort,
                    api_base=current_api_base,
                    kwargs=kwargs,
                    warning_sink=_warnings,
                )
                call_kwargs = {**base_kwargs, "response_model": response_model, "max_retries": 0}

                for attempt in range(r.max_retries + 1):
                    try:
                        parsed, completion_response = client.chat.completions.create_with_completion(
                            **call_kwargs,
                        )

                        usage = _extract_usage(completion_response)
                        cost, cost_source = _parse_cost_result(_compute_cost(completion_response))
                        content = str(parsed.model_dump_json())
                        completion_choice = _first_choice_or_empty_error(
                            completion_response,
                            model=current_model,
                            provider="instructor_completion_structured",
                        )
                        finish_reason = completion_choice.finish_reason or ""

                        if attempt > 0:
                            logger.info("call_llm_structured succeeded after %d retries", attempt)

                        llm_result = LLMCallResult(
                            content=content,
                            usage=usage,
                            cost=cost,
                            model=current_model,
                            finish_reason=finish_reason,
                            raw_response=completion_response,
                            warnings=_warnings,
                            cost_source=cost_source,
                        )
                        llm_result = _annotate_result_identity(
                            llm_result,
                            requested_model=model,
                            resolved_model=current_model,
                            routing_trace=_build_routing_trace(
                                requested_model=model,
                                attempted_models=models[:model_idx + 1],
                                selected_model=current_model,
                                requested_api_base=api_base,
                                effective_api_base=current_api_base,
                                routing_policy=routing_policy,
                            ),
                            result_model_semantics=cfg.result_model_semantics,
                        )

                        if hooks and hooks.after_call:
                            hooks.after_call(llm_result)
                        if cache is not None:
                            cache.set(key, llm_result)
                        _io_log.log_call(model=current_model, messages=messages, result=llm_result, latency_s=time.monotonic() - _log_t0, caller="call_llm_structured", task=task, trace_id=trace_id)
                        return parsed, llm_result
                    except Exception as e:
                        last_error = e
                        if hooks and hooks.on_error:
                            hooks.on_error(e, attempt)
                        if not _check_retryable(e, r) or attempt >= r.max_retries:
                            raise
                        delay, retry_delay_source = _compute_retry_delay(
                            attempt=attempt,
                            error=e,
                            policy=r,
                            backoff_fn=backoff_fn,
                        )
                        if r.on_retry is not None:
                            r.on_retry(attempt, e, delay)
                        _warnings.append(
                            f"RETRY {attempt + 1}/{r.max_retries + 1}: "
                            f"{current_model} ({type(e).__name__}: {e}) "
                            f"[retry_delay_source={retry_delay_source}]"
                        )
                        logger.warning(
                            "call_llm_structured attempt %d/%d failed (retrying in %.1fs, source=%s): %s",
                            attempt + 1,
                            r.max_retries + 1,
                            delay,
                            retry_delay_source,
                            e,
                        )
                        time.sleep(delay)
                raise wrap_error(last_error) from last_error  # type: ignore[misc]  # unreachable
        except Exception as e:
            last_error = e
            if model_idx < len(models) - 1:
                next_model = models[model_idx + 1]
                if on_fallback is not None:
                    on_fallback(current_model, e, next_model)
                _warnings.append(
                    f"FALLBACK: {current_model} -> {next_model} "
                    f"({type(e).__name__}: {e})"
                )
                logger.warning(
                    "Falling back from %s to %s: %s", current_model, next_model, e,
                )
                continue
            _io_log.log_call(model=current_model, messages=messages, error=e, latency_s=time.monotonic() - _log_t0, caller="call_llm_structured", task=task, trace_id=trace_id)
            raise wrap_error(e) from e

    raise wrap_error(last_error) from last_error  # type: ignore[misc]  # unreachable


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
        **kwargs: Additional params passed to litellm

    Returns:
        LLMCallResult with content, usage, cost, model, tool_calls,
        finish_reason, and raw_response
    """
    _check_model_deprecation(model)
    explicit_config = config is not None
    cfg = config or ClientConfig.from_env()
    _log_t0 = time.monotonic()
    task = kwargs.pop("task", None)
    trace_id = kwargs.pop("trace_id", None)
    max_budget: float | None = kwargs.pop("max_budget", None)
    agent_retry_safe = kwargs.pop("agent_retry_safe", None)
    task, trace_id, max_budget, _entry_warnings = _require_tags(
        task, trace_id, max_budget, caller="acall_llm",
    )
    _check_budget(trace_id, max_budget)
    _record_semantics_adoption(
        cfg=cfg,
        explicit_config=explicit_config,
        caller="acall_llm",
        task=task,
        trace_id=trace_id,
    )

    # Named params that must flow through to per-turn _inner_acall_llm calls
    # inside the agent loop (retry, fallback, hooks, reasoning, api_base).
    _inner_named: dict[str, Any] = {}
    if num_retries != 2:
        _inner_named["num_retries"] = num_retries
    if base_delay != 1.0:
        _inner_named["base_delay"] = base_delay
    if max_delay != 30.0:
        _inner_named["max_delay"] = max_delay
    if retry_on is not None:
        _inner_named["retry_on"] = retry_on
    if on_retry is not None:
        _inner_named["on_retry"] = on_retry
    if retry is not None:
        _inner_named["retry"] = retry
    if fallback_models is not None:
        _inner_named["fallback_models"] = fallback_models
    if on_fallback is not None:
        _inner_named["on_fallback"] = on_fallback
    if reasoning_effort is not None:
        _inner_named["reasoning_effort"] = reasoning_effort
    if api_base is not None:
        _inner_named["api_base"] = api_base
    if hooks is not None:
        _inner_named["hooks"] = hooks
    if execution_mode != "text":
        _inner_named["execution_mode"] = execution_mode
    _inner_named["config"] = cfg

    plan = resolve_call(
        CallRequest(model=model, fallback_models=fallback_models, api_base=api_base),
        cfg,
    )
    models = plan.models
    primary_model = plan.primary_model
    fallback_chain = plan.fallback_models or None
    routing_policy = str(plan.routing_trace.get("routing_policy", _routing_policy_label(cfg)))
    if fallback_chain is not None:
        _inner_named["fallback_models"] = fallback_chain
    else:
        _inner_named.pop("fallback_models", None)
    _validate_execution_contract(
        models=models,
        execution_mode=execution_mode,
        kwargs=kwargs,
        caller="acall_llm",
    )

    # MCP agent loop: non-agent model + (mcp_servers or mcp_sessions) → tool-calling loop
    if ("mcp_servers" in kwargs or "mcp_sessions" in kwargs) and not _is_agent_model(model):
        from llm_client.mcp_agent import MCP_LOOP_KWARGS, _acall_with_mcp
        mcp_kw: dict[str, Any] = {}
        remaining = dict(kwargs)
        remaining["task"] = task
        remaining["trace_id"] = trace_id
        remaining["max_budget"] = max_budget
        for k in MCP_LOOP_KWARGS:
            if k in remaining:
                mcp_kw[k] = remaining.pop(k)
        result = await _acall_with_mcp(
            primary_model, messages, timeout=timeout, **_inner_named, **mcp_kw, **remaining,
        )
        result = _annotate_result_identity(
            result,
            requested_model=model,
            resolved_model=result.resolved_model,
            routing_trace=_build_routing_trace(
                requested_model=model,
                attempted_models=[primary_model],
                selected_model=result.resolved_model,
                requested_api_base=api_base,
                effective_api_base=_resolve_api_base_for_model(primary_model, api_base, cfg),
                sticky_fallback=any("STICKY_FALLBACK" in w for w in (result.warnings or [])),
                routing_policy=routing_policy,
            ),
            result_model_semantics=cfg.result_model_semantics,
        )
        _io_log.log_call(model=primary_model, messages=messages, result=result, latency_s=time.monotonic() - _log_t0, caller="acall_llm", task=task, trace_id=trace_id)
        return result

    # Direct Python tool loop: non-agent model + python_tools → in-process tool-calling loop
    if "python_tools" in kwargs and not _is_agent_model(model):
        if "mcp_servers" in kwargs or "mcp_sessions" in kwargs:
            raise ValueError("python_tools and mcp_servers/mcp_sessions are mutually exclusive.")
        from llm_client.mcp_agent import TOOL_LOOP_KWARGS, _acall_with_tools
        from llm_client.models import supports_tool_calling
        tool_kw: dict[str, Any] = {}
        remaining = dict(kwargs)
        remaining["task"] = task
        remaining["trace_id"] = trace_id
        remaining["max_budget"] = max_budget
        for k in TOOL_LOOP_KWARGS:
            if k in remaining:
                tool_kw[k] = remaining.pop(k)
        if not supports_tool_calling(model):
            from llm_client.tool_shim import _acall_with_tool_shim
            result = await _acall_with_tool_shim(
                primary_model, messages, timeout=timeout, **_inner_named, **tool_kw, **remaining,
            )
        else:
            result = await _acall_with_tools(
                primary_model, messages, timeout=timeout, **_inner_named, **tool_kw, **remaining,
            )
        result = _annotate_result_identity(
            result,
            requested_model=model,
            resolved_model=result.resolved_model,
            routing_trace=_build_routing_trace(
                requested_model=model,
                attempted_models=[primary_model],
                selected_model=result.resolved_model,
                requested_api_base=api_base,
                effective_api_base=_resolve_api_base_for_model(primary_model, api_base, cfg),
                sticky_fallback=any("STICKY_FALLBACK" in w for w in (result.warnings or [])),
                routing_policy=routing_policy,
            ),
            result_model_semantics=cfg.result_model_semantics,
        )
        _io_log.log_call(model=primary_model, messages=messages, result=result, latency_s=time.monotonic() - _log_t0, caller="acall_llm", task=task, trace_id=trace_id)
        return result

    r = _effective_retry(retry, num_retries, base_delay, max_delay, retry_on, on_retry)
    if cache is not None and _is_agent_model(model):
        raise ValueError("Caching not supported for agent models — they have side effects.")
    last_error: Exception | None = None
    _warnings: list[str] = list(_entry_warnings)
    agent_retry_safe_enabled = _agent_retry_safe_enabled(agent_retry_safe)

    for model_idx, current_model in enumerate(models):
        is_agent = _is_agent_model(current_model)
        use_responses = not is_agent and _is_responses_api_model(current_model)
        current_api_base = _resolve_api_base_for_model(current_model, api_base, cfg)
        use_gemini_native = (
            not is_agent
            and not use_responses
            and _should_use_gemini_native(
                current_model,
                api_base=current_api_base,
                kwargs=kwargs,
                warning_sink=_warnings,
            )
        )

        if is_agent:
            pass  # No kwargs preparation needed for agent models
        elif use_responses:
            call_kwargs = _prepare_responses_kwargs(
                current_model, messages,
                timeout=timeout,
                api_base=current_api_base,
                kwargs=kwargs,
                warning_sink=_warnings,
            )
        elif use_gemini_native:
            pass  # Native Gemini route builds payload per-attempt.
        else:
            call_kwargs = _prepare_call_kwargs(
                current_model, messages,
                timeout=timeout,
                num_retries=r.max_retries,
                reasoning_effort=reasoning_effort,
                api_base=current_api_base,
                kwargs=kwargs,
                warning_sink=_warnings,
            )

        if cache is not None:
            key = _cache_key(current_model, messages, **kwargs)
            cached = await _async_cache_get(cache, key)
            if cached is not None:
                cached_result = _mark_cache_hit(cached)
                cached_result = _annotate_result_identity(
                    cached_result,
                    requested_model=model,
                    resolved_model=current_model,
                    routing_trace=_build_routing_trace(
                        requested_model=model,
                        attempted_models=models[:model_idx + 1],
                        selected_model=current_model,
                        requested_api_base=api_base,
                        effective_api_base=current_api_base,
                        routing_policy=routing_policy,
                    ),
                    result_model_semantics=cfg.result_model_semantics,
                )
                _io_log.log_call(
                    model=current_model,
                    messages=messages,
                    result=cached_result,
                    latency_s=time.monotonic() - _log_t0,
                    caller="acall_llm",
                    task=task,
                    trace_id=trace_id,
                )
                return cached_result

        if hooks and hooks.before_call:
            hooks.before_call(current_model, messages, kwargs)

        backoff_fn = r.backoff or exponential_backoff
        if is_agent and not agent_retry_safe_enabled:
            effective_retries = 0
            if r.max_retries > 0:
                msg = (
                    "AGENT_RETRY_DISABLED: retries for agent models are disabled by default "
                    "to avoid duplicate side effects. Set agent_retry_safe=True (or "
                    f"{AGENT_RETRY_SAFE_ENV}=1) only for explicitly safe/read-only runs."
                )
                if msg not in _warnings:
                    _warnings.append(msg)
                    logger.warning(msg)
        else:
            effective_retries = r.max_retries
        try:
            for attempt in range(effective_retries + 1):
                try:
                    if is_agent:
                        from llm_client.agents import _route_acall
                        result = await _route_acall(
                            current_model, messages,
                            timeout=timeout, **kwargs,
                        )
                    elif use_responses:
                        async with _rate_limit.aacquire(current_model):
                            response = await litellm.aresponses(**call_kwargs)
                        result = _build_result_from_responses(response, current_model, warnings=_warnings)
                    elif use_gemini_native:
                        async with _rate_limit.aacquire(current_model):
                            response = await _acall_gemini_native(
                                current_model,
                                messages,
                                timeout=timeout,
                                kwargs=kwargs,
                            )
                        result = _build_result_from_gemini_native(response, current_model, warnings=_warnings)
                    else:
                        async with _rate_limit.aacquire(current_model):
                            response = await litellm.acompletion(**call_kwargs)
                        result = _build_result_from_response(response, current_model, warnings=_warnings)
                    if attempt > 0:
                        logger.info("acall_llm succeeded after %d retries", attempt)
                    if is_agent:
                        resolved_model = result.resolved_model
                    else:
                        resolved_model = current_model
                    result = _annotate_result_identity(
                        result,
                        requested_model=model,
                        resolved_model=resolved_model,
                        routing_trace=_build_routing_trace(
                            requested_model=model,
                            attempted_models=models[:model_idx + 1],
                            selected_model=resolved_model,
                            requested_api_base=api_base,
                            effective_api_base=current_api_base,
                            sticky_fallback=any("STICKY_FALLBACK" in w for w in (result.warnings or [])),
                            routing_policy=routing_policy,
                        ),
                        result_model_semantics=cfg.result_model_semantics,
                    )
                    if hooks and hooks.after_call:
                        hooks.after_call(result)
                    if cache is not None:
                        await _async_cache_set(cache, key, result)
                    _io_log.log_call(model=current_model, messages=messages, result=result, latency_s=time.monotonic() - _log_t0, caller="acall_llm", task=task, trace_id=trace_id)
                    return result
                except Exception as e:
                    last_error = e
                    if hooks and hooks.on_error:
                        hooks.on_error(e, attempt)
                    if not _check_retryable(e, r) or attempt >= effective_retries:
                        raise
                    delay, retry_delay_source = _compute_retry_delay(
                        attempt=attempt,
                        error=e,
                        policy=r,
                        backoff_fn=backoff_fn,
                    )
                    if r.on_retry is not None:
                        r.on_retry(attempt, e, delay)
                    _warnings.append(
                        f"RETRY {attempt + 1}/{effective_retries + 1}: "
                        f"{current_model} ({type(e).__name__}: {e}) "
                        f"[retry_delay_source={retry_delay_source}]"
                    )
                    logger.warning(
                        "acall_llm attempt %d/%d failed (retrying in %.1fs, source=%s): %s",
                        attempt + 1,
                        effective_retries + 1,
                        delay,
                        retry_delay_source,
                        e,
                    )
                    await asyncio.sleep(delay)
            raise wrap_error(last_error) from last_error  # type: ignore[misc]  # unreachable
        except Exception as e:
            last_error = e
            if model_idx < len(models) - 1:
                next_model = models[model_idx + 1]
                if on_fallback is not None:
                    on_fallback(current_model, e, next_model)
                _warnings.append(
                    f"FALLBACK: {current_model} -> {next_model} "
                    f"({type(e).__name__}: {e})"
                )
                logger.warning(
                    "Falling back from %s to %s: %s", current_model, next_model, e,
                )
                continue
            _io_log.log_call(model=current_model, messages=messages, error=e, latency_s=time.monotonic() - _log_t0, caller="acall_llm", task=task, trace_id=trace_id)
            raise wrap_error(e) from e

    raise wrap_error(last_error) from last_error  # type: ignore[misc]  # unreachable


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
        **kwargs: Additional params passed to litellm.acompletion

    Returns:
        Tuple of (parsed Pydantic model instance, LLMCallResult)
    """
    _check_model_deprecation(model)
    explicit_config = config is not None
    cfg = config or ClientConfig.from_env()
    _log_t0 = time.monotonic()
    task = kwargs.pop("task", None)
    trace_id = kwargs.pop("trace_id", None)
    max_budget: float | None = kwargs.pop("max_budget", None)
    task, trace_id, max_budget, _entry_warnings = _require_tags(
        task, trace_id, max_budget, caller="acall_llm_structured",
    )
    _check_budget(trace_id, max_budget)
    _record_semantics_adoption(
        cfg=cfg,
        explicit_config=explicit_config,
        caller="acall_llm_structured",
        task=task,
        trace_id=trace_id,
    )
    plan = resolve_call(
        CallRequest(model=model, fallback_models=fallback_models, api_base=api_base),
        cfg,
    )
    models = plan.models
    routing_policy = str(plan.routing_trace.get("routing_policy", _routing_policy_label(cfg)))

    if _is_agent_model(model):
        from llm_client.agents import _route_acall_structured
        if hooks and hooks.before_call:
            hooks.before_call(model, messages, kwargs)
        parsed, llm_result = await _route_acall_structured(
            model, messages, response_model, timeout=timeout, **kwargs,
        )
        llm_result = _annotate_result_identity(
            llm_result,
            requested_model=model,
            resolved_model=llm_result.resolved_model,
            routing_trace=_build_routing_trace(
                requested_model=model,
                attempted_models=[plan.primary_model],
                selected_model=llm_result.resolved_model,
                requested_api_base=api_base,
                effective_api_base=api_base,
                routing_policy=routing_policy,
            ),
            result_model_semantics=cfg.result_model_semantics,
        )
        if hooks and hooks.after_call:
            hooks.after_call(llm_result)
        _io_log.log_call(model=model, messages=messages, result=llm_result, latency_s=time.monotonic() - _log_t0, caller="acall_llm_structured", task=task, trace_id=trace_id)
        return parsed, llm_result
    r = _effective_retry(retry, num_retries, base_delay, max_delay, retry_on, on_retry)
    last_error: Exception | None = None
    _warnings: list[str] = list(_entry_warnings)

    _model_fqn = f"{response_model.__module__}.{response_model.__qualname__}"

    for model_idx, current_model in enumerate(models):
        current_api_base = _resolve_api_base_for_model(current_model, api_base, cfg)
        if cache is not None:
            key = _cache_key(current_model, messages, response_model=_model_fqn, **kwargs)
            cached = await _async_cache_get(cache, key)
            if cached is not None:
                reparsed = response_model.model_validate_json(cached.content)
                cached_result = _mark_cache_hit(cached)
                cached_result = _annotate_result_identity(
                    cached_result,
                    requested_model=model,
                    resolved_model=current_model,
                    routing_trace=_build_routing_trace(
                        requested_model=model,
                        attempted_models=models[:model_idx + 1],
                        selected_model=current_model,
                        requested_api_base=api_base,
                        effective_api_base=current_api_base,
                        routing_policy=routing_policy,
                    ),
                    result_model_semantics=cfg.result_model_semantics,
                )
                _io_log.log_call(
                    model=current_model,
                    messages=messages,
                    result=cached_result,
                    latency_s=time.monotonic() - _log_t0,
                    caller="acall_llm_structured",
                    task=task,
                    trace_id=trace_id,
                )
                return reparsed, cached_result

        if hooks and hooks.before_call:
            hooks.before_call(current_model, messages, kwargs)

        backoff_fn = r.backoff or exponential_backoff
        try:
            if _is_responses_api_model(current_model):
                # GPT-5 path: Responses API with native JSON schema
                schema = _strict_json_schema(response_model.model_json_schema())
                resp_kwargs = _prepare_responses_kwargs(
                    current_model, messages,
                    timeout=timeout, api_base=current_api_base, kwargs=kwargs,
                    warning_sink=_warnings,
                )
                resp_kwargs["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": response_model.__name__,
                        "schema": schema,
                        "strict": True,
                    }
                }

                for attempt in range(r.max_retries + 1):
                    try:
                        async with _rate_limit.aacquire(current_model):
                            response = await litellm.aresponses(**resp_kwargs)
                        raw_content = getattr(response, "output_text", None) or ""
                        if not raw_content.strip():
                            raise ValueError("Empty content from LLM (responses API structured)")
                        parsed = response_model.model_validate_json(raw_content)
                        content = str(parsed.model_dump_json())
                        usage = _extract_responses_usage(response)
                        cost, cost_source = _parse_cost_result(_compute_responses_cost(response, usage), default_source="computed")

                        if attempt > 0:
                            logger.info("acall_llm_structured (responses) succeeded after %d retries", attempt)

                        llm_result = LLMCallResult(
                            content=content, usage=usage, cost=cost,
                            model=current_model, finish_reason="stop",
                            raw_response=response, warnings=_warnings,
                            cost_source=cost_source,
                        )
                        llm_result = _annotate_result_identity(
                            llm_result,
                            requested_model=model,
                            resolved_model=current_model,
                            routing_trace=_build_routing_trace(
                                requested_model=model,
                                attempted_models=models[:model_idx + 1],
                                selected_model=current_model,
                                requested_api_base=api_base,
                                effective_api_base=current_api_base,
                                routing_policy=routing_policy,
                            ),
                            result_model_semantics=cfg.result_model_semantics,
                        )
                        if hooks and hooks.after_call:
                            hooks.after_call(llm_result)
                        if cache is not None:
                            await _async_cache_set(cache, key, llm_result)
                        _io_log.log_call(model=current_model, messages=messages, result=llm_result, latency_s=time.monotonic() - _log_t0, caller="acall_llm_structured", task=task, trace_id=trace_id)
                        return parsed, llm_result
                    except Exception as e:
                        last_error = e
                        if hooks and hooks.on_error:
                            hooks.on_error(e, attempt)
                        if not _check_retryable(e, r) or attempt >= r.max_retries:
                            raise
                        delay, retry_delay_source = _compute_retry_delay(
                            attempt=attempt,
                            error=e,
                            policy=r,
                            backoff_fn=backoff_fn,
                        )
                        if r.on_retry is not None:
                            r.on_retry(attempt, e, delay)
                        _warnings.append(
                            f"RETRY {attempt + 1}/{r.max_retries + 1}: "
                            f"{current_model} ({type(e).__name__}: {e}) "
                            f"[retry_delay_source={retry_delay_source}]"
                        )
                        logger.warning(
                            "acall_llm_structured attempt %d/%d failed (retrying in %.1fs, source=%s): %s",
                            attempt + 1, r.max_retries + 1, delay, retry_delay_source, e,
                        )
                        await asyncio.sleep(delay)
                raise wrap_error(last_error) from last_error  # type: ignore[misc]  # unreachable
            elif litellm.supports_response_schema(model=current_model):
                # Native JSON schema path: litellm.acompletion + response_format
                # If the provider rejects the schema (e.g. Gemini nesting depth
                # limit), fall through to the instructor path automatically.
                _native_schema_failed = False
                schema = _strict_json_schema(response_model.model_json_schema())
                base_kwargs = _prepare_call_kwargs(
                    current_model, messages,
                    timeout=timeout,
                    num_retries=r.max_retries,
                    reasoning_effort=reasoning_effort,
                    api_base=current_api_base,
                    kwargs=kwargs,
                    warning_sink=_warnings,
                )
                base_kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "schema": schema,
                        "strict": True,
                    },
                }

                for attempt in range(r.max_retries + 1):
                    try:
                        async with _rate_limit.aacquire(current_model):
                            response = await litellm.acompletion(**base_kwargs)
                        first_choice = _first_choice_or_empty_error(
                            response,
                            model=current_model,
                            provider="litellm_completion_structured",
                        )
                        raw_content = first_choice.message.content or ""
                        if not raw_content.strip():
                            raise ValueError("Empty content from LLM (native JSON schema structured)")
                        parsed = response_model.model_validate_json(raw_content)
                        content = str(parsed.model_dump_json())
                        usage = _extract_usage(response)
                        cost, cost_source = _parse_cost_result(_compute_cost(response))
                        finish_reason: str = first_choice.finish_reason or "stop"

                        if attempt > 0:
                            logger.info("acall_llm_structured (native schema) succeeded after %d retries", attempt)

                        llm_result = LLMCallResult(
                            content=content, usage=usage, cost=cost,
                            model=current_model, finish_reason=finish_reason,
                            raw_response=response, warnings=_warnings,
                            cost_source=cost_source,
                        )
                        llm_result = _annotate_result_identity(
                            llm_result,
                            requested_model=model,
                            resolved_model=current_model,
                            routing_trace=_build_routing_trace(
                                requested_model=model,
                                attempted_models=models[:model_idx + 1],
                                selected_model=current_model,
                                requested_api_base=api_base,
                                effective_api_base=current_api_base,
                                routing_policy=routing_policy,
                            ),
                            result_model_semantics=cfg.result_model_semantics,
                        )
                        if hooks and hooks.after_call:
                            hooks.after_call(llm_result)
                        if cache is not None:
                            await _async_cache_set(cache, key, llm_result)
                        _io_log.log_call(model=current_model, messages=messages, result=llm_result, latency_s=time.monotonic() - _log_t0, caller="acall_llm_structured", task=task, trace_id=trace_id)
                        return parsed, llm_result
                    except Exception as e:
                        if _is_schema_error(e):
                            logger.warning(
                                "Native JSON schema rejected by provider (%s), "
                                "falling back to instructor: %s",
                                current_model, e,
                            )
                            _native_schema_failed = True
                            break
                        last_error = e
                        if hooks and hooks.on_error:
                            hooks.on_error(e, attempt)
                        if not _check_retryable(e, r) or attempt >= r.max_retries:
                            raise
                        delay, retry_delay_source = _compute_retry_delay(
                            attempt=attempt,
                            error=e,
                            policy=r,
                            backoff_fn=backoff_fn,
                        )
                        if r.on_retry is not None:
                            r.on_retry(attempt, e, delay)
                        _warnings.append(
                            f"RETRY {attempt + 1}/{r.max_retries + 1}: "
                            f"{current_model} ({type(e).__name__}: {e}) "
                            f"[retry_delay_source={retry_delay_source}]"
                        )
                        logger.warning(
                            "acall_llm_structured attempt %d/%d failed (retrying in %.1fs, source=%s): %s",
                            attempt + 1, r.max_retries + 1, delay, retry_delay_source, e,
                        )
                        await asyncio.sleep(delay)
                else:
                    raise wrap_error(last_error) from last_error  # type: ignore[misc]  # unreachable

            if not litellm.supports_response_schema(model=current_model) or _native_schema_failed:
                # Fallback path: instructor + litellm.acompletion
                import instructor

                client = instructor.from_litellm(litellm.acompletion)
                base_kwargs = _prepare_call_kwargs(
                    current_model, messages,
                    timeout=timeout,
                    num_retries=r.max_retries,
                    reasoning_effort=reasoning_effort,
                    api_base=current_api_base,
                    kwargs=kwargs,
                    warning_sink=_warnings,
                )
                call_kwargs = {**base_kwargs, "response_model": response_model, "max_retries": 0}

                for attempt in range(r.max_retries + 1):
                    try:
                        parsed, completion_response = await client.chat.completions.create_with_completion(
                            **call_kwargs,
                        )

                        usage = _extract_usage(completion_response)
                        cost, cost_source = _parse_cost_result(_compute_cost(completion_response))
                        content = str(parsed.model_dump_json())
                        completion_choice = _first_choice_or_empty_error(
                            completion_response,
                            model=current_model,
                            provider="instructor_completion_structured",
                        )
                        finish_reason = completion_choice.finish_reason or ""

                        if attempt > 0:
                            logger.info("acall_llm_structured succeeded after %d retries", attempt)

                        llm_result = LLMCallResult(
                            content=content,
                            usage=usage,
                            cost=cost,
                            model=current_model,
                            finish_reason=finish_reason,
                            raw_response=completion_response,
                            warnings=_warnings,
                            cost_source=cost_source,
                        )
                        llm_result = _annotate_result_identity(
                            llm_result,
                            requested_model=model,
                            resolved_model=current_model,
                            routing_trace=_build_routing_trace(
                                requested_model=model,
                                attempted_models=models[:model_idx + 1],
                                selected_model=current_model,
                                requested_api_base=api_base,
                                effective_api_base=current_api_base,
                                routing_policy=routing_policy,
                            ),
                            result_model_semantics=cfg.result_model_semantics,
                        )

                        if hooks and hooks.after_call:
                            hooks.after_call(llm_result)
                        if cache is not None:
                            await _async_cache_set(cache, key, llm_result)
                        _io_log.log_call(model=current_model, messages=messages, result=llm_result, latency_s=time.monotonic() - _log_t0, caller="acall_llm_structured", task=task, trace_id=trace_id)
                        return parsed, llm_result
                    except Exception as e:
                        last_error = e
                        if hooks and hooks.on_error:
                            hooks.on_error(e, attempt)
                        if not _check_retryable(e, r) or attempt >= r.max_retries:
                            raise
                        delay, retry_delay_source = _compute_retry_delay(
                            attempt=attempt,
                            error=e,
                            policy=r,
                            backoff_fn=backoff_fn,
                        )
                        if r.on_retry is not None:
                            r.on_retry(attempt, e, delay)
                        _warnings.append(
                            f"RETRY {attempt + 1}/{r.max_retries + 1}: "
                            f"{current_model} ({type(e).__name__}: {e}) "
                            f"[retry_delay_source={retry_delay_source}]"
                        )
                        logger.warning(
                            "acall_llm_structured attempt %d/%d failed (retrying in %.1fs, source=%s): %s",
                            attempt + 1,
                            r.max_retries + 1,
                            delay,
                            retry_delay_source,
                            e,
                        )
                        await asyncio.sleep(delay)
                raise wrap_error(last_error) from last_error  # type: ignore[misc]  # unreachable
        except Exception as e:
            last_error = e
            if model_idx < len(models) - 1:
                next_model = models[model_idx + 1]
                if on_fallback is not None:
                    on_fallback(current_model, e, next_model)
                _warnings.append(
                    f"FALLBACK: {current_model} -> {next_model} "
                    f"({type(e).__name__}: {e})"
                )
                logger.warning(
                    "Falling back from %s to %s: %s", current_model, next_model, e,
                )
                continue
            _io_log.log_call(model=current_model, messages=messages, error=e, latency_s=time.monotonic() - _log_t0, caller="acall_llm_structured", task=task, trace_id=trace_id)
            raise wrap_error(e) from e

    raise wrap_error(last_error) from last_error  # type: ignore[misc]  # unreachable


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
    if not messages_list:
        return []

    sem = asyncio.Semaphore(max_concurrent)

    async def _call_one(idx: int, messages: list[dict[str, Any]]) -> LLMCallResult:
        async with sem:
            try:
                result = await acall_llm(
                    model, messages,
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
                if on_item_complete is not None:
                    on_item_complete(idx, result)
                return result
            except Exception as e:
                if on_item_error is not None:
                    on_item_error(idx, e)
                raise

    tasks = [_call_one(i, msgs) for i, msgs in enumerate(messages_list)]
    return await asyncio.gather(*tasks, return_exceptions=return_exceptions)  # type: ignore[return-value]


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
    coro = acall_llm_batch(
        model, messages_list,
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
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


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
    if not messages_list:
        return []

    sem = asyncio.Semaphore(max_concurrent)

    async def _call_one(idx: int, messages: list[dict[str, Any]]) -> tuple[T, LLMCallResult]:
        async with sem:
            try:
                result = await acall_llm_structured(
                    model, messages, response_model,
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
                if on_item_complete is not None:
                    on_item_complete(idx, result)
                return result
            except Exception as e:
                if on_item_error is not None:
                    on_item_error(idx, e)
                raise

    tasks = [_call_one(i, msgs) for i, msgs in enumerate(messages_list)]
    return await asyncio.gather(*tasks, return_exceptions=return_exceptions)  # type: ignore[return-value]


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
    coro = acall_llm_structured_batch(
        model, messages_list, response_model,
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
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


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
        **kwargs: Additional params passed to litellm.completion

    Returns:
        LLMStream that yields text chunks and exposes ``.result``
    """
    _check_model_deprecation(model)
    explicit_config = config is not None
    cfg = config or ClientConfig.from_env()
    task = kwargs.pop("task", None)
    trace_id = kwargs.pop("trace_id", None)
    max_budget: float | None = kwargs.pop("max_budget", None)
    task, trace_id, max_budget, _entry_warnings = _require_tags(
        task, trace_id, max_budget, caller="stream_llm",
    )
    _check_budget(trace_id, max_budget)
    _record_semantics_adoption(
        cfg=cfg,
        explicit_config=explicit_config,
        caller="stream_llm",
        task=task,
        trace_id=trace_id,
    )
    if _is_agent_model(model):
        from llm_client.agents import _route_stream
        return _route_stream(model, messages, hooks=hooks, timeout=timeout, **kwargs)
    r = _effective_retry(retry, num_retries, base_delay, max_delay, retry_on, on_retry)
    models = _build_model_chain(model, fallback_models, cfg)
    routing_policy = _routing_policy_label(cfg)
    last_error: Exception | None = None
    _warnings: list[str] = list(_entry_warnings)
    backoff_fn = r.backoff or exponential_backoff

    for model_idx, current_model in enumerate(models):
        current_api_base = _resolve_api_base_for_model(current_model, api_base, cfg)
        call_kwargs = _prepare_call_kwargs(
            current_model, messages,
            timeout=timeout,
            num_retries=0,
            reasoning_effort=reasoning_effort,
            api_base=current_api_base,
            kwargs=kwargs,
            warning_sink=_warnings,
        )
        call_kwargs["stream"] = True

        if hooks and hooks.before_call:
            hooks.before_call(current_model, messages, kwargs)

        try:
            for attempt in range(r.max_retries + 1):
                try:
                    with _rate_limit.acquire(current_model):
                        response = litellm.completion(**call_kwargs)
                    if attempt > 0:
                        logger.info("stream_llm succeeded after %d retries", attempt)
                    return LLMStream(
                        response,
                        current_model,
                        hooks=hooks,
                        messages=messages,
                        task=task,
                        trace_id=trace_id,
                        warnings=_warnings,
                        requested_model=model,
                        resolved_model=current_model,
                        routing_trace=_build_routing_trace(
                            requested_model=model,
                            attempted_models=models[:model_idx + 1],
                            selected_model=current_model,
                            requested_api_base=api_base,
                            effective_api_base=current_api_base,
                            routing_policy=routing_policy,
                        ),
                        result_model_semantics=cfg.result_model_semantics,
                    )
                except Exception as e:
                    last_error = e
                    if hooks and hooks.on_error:
                        hooks.on_error(e, attempt)
                    if not _check_retryable(e, r) or attempt >= r.max_retries:
                        raise
                    delay, retry_delay_source = _compute_retry_delay(
                        attempt=attempt,
                        error=e,
                        policy=r,
                        backoff_fn=backoff_fn,
                    )
                    if r.on_retry is not None:
                        r.on_retry(attempt, e, delay)
                    _warnings.append(
                        f"RETRY {attempt + 1}/{r.max_retries + 1}: "
                        f"{current_model} ({type(e).__name__}: {e}) "
                        f"[retry_delay_source={retry_delay_source}]"
                    )
                    logger.warning(
                        "stream_llm attempt %d/%d failed (retrying in %.1fs, source=%s): %s",
                        attempt + 1, r.max_retries + 1, delay, retry_delay_source, e,
                    )
                    time.sleep(delay)
            raise wrap_error(last_error) from last_error  # type: ignore[misc]  # unreachable
        except Exception as e:
            last_error = e
            if model_idx < len(models) - 1:
                next_model = models[model_idx + 1]
                if on_fallback is not None:
                    on_fallback(current_model, e, next_model)
                _warnings.append(
                    f"FALLBACK: {current_model} -> {next_model} "
                    f"({type(e).__name__}: {e})"
                )
                logger.warning(
                    "Falling back from %s to %s: %s", current_model, next_model, e,
                )
                continue
            raise wrap_error(e) from e

    raise wrap_error(last_error) from last_error  # type: ignore[misc]  # unreachable


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
    _check_model_deprecation(model)
    explicit_config = config is not None
    cfg = config or ClientConfig.from_env()
    task = kwargs.pop("task", None)
    trace_id = kwargs.pop("trace_id", None)
    max_budget: float | None = kwargs.pop("max_budget", None)
    task, trace_id, max_budget, _entry_warnings = _require_tags(
        task, trace_id, max_budget, caller="astream_llm",
    )
    _check_budget(trace_id, max_budget)
    _record_semantics_adoption(
        cfg=cfg,
        explicit_config=explicit_config,
        caller="astream_llm",
        task=task,
        trace_id=trace_id,
    )
    if _is_agent_model(model):
        from llm_client.agents import _route_astream
        return await _route_astream(model, messages, hooks=hooks, timeout=timeout, **kwargs)
    r = _effective_retry(retry, num_retries, base_delay, max_delay, retry_on, on_retry)
    models = _build_model_chain(model, fallback_models, cfg)
    routing_policy = _routing_policy_label(cfg)
    last_error: Exception | None = None
    _warnings: list[str] = list(_entry_warnings)
    backoff_fn = r.backoff or exponential_backoff

    for model_idx, current_model in enumerate(models):
        current_api_base = _resolve_api_base_for_model(current_model, api_base, cfg)
        call_kwargs = _prepare_call_kwargs(
            current_model, messages,
            timeout=timeout,
            num_retries=0,
            reasoning_effort=reasoning_effort,
            api_base=current_api_base,
            kwargs=kwargs,
            warning_sink=_warnings,
        )
        call_kwargs["stream"] = True

        if hooks and hooks.before_call:
            hooks.before_call(current_model, messages, kwargs)

        try:
            for attempt in range(r.max_retries + 1):
                try:
                    async with _rate_limit.aacquire(current_model):
                        response = await litellm.acompletion(**call_kwargs)
                    if attempt > 0:
                        logger.info("astream_llm succeeded after %d retries", attempt)
                    return AsyncLLMStream(
                        response,
                        current_model,
                        hooks=hooks,
                        messages=messages,
                        task=task,
                        trace_id=trace_id,
                        warnings=_warnings,
                        requested_model=model,
                        resolved_model=current_model,
                        routing_trace=_build_routing_trace(
                            requested_model=model,
                            attempted_models=models[:model_idx + 1],
                            selected_model=current_model,
                            requested_api_base=api_base,
                            effective_api_base=current_api_base,
                            routing_policy=routing_policy,
                        ),
                        result_model_semantics=cfg.result_model_semantics,
                    )
                except Exception as e:
                    last_error = e
                    if hooks and hooks.on_error:
                        hooks.on_error(e, attempt)
                    if not _check_retryable(e, r) or attempt >= r.max_retries:
                        raise
                    delay, retry_delay_source = _compute_retry_delay(
                        attempt=attempt,
                        error=e,
                        policy=r,
                        backoff_fn=backoff_fn,
                    )
                    if r.on_retry is not None:
                        r.on_retry(attempt, e, delay)
                    _warnings.append(
                        f"RETRY {attempt + 1}/{r.max_retries + 1}: "
                        f"{current_model} ({type(e).__name__}: {e}) "
                        f"[retry_delay_source={retry_delay_source}]"
                    )
                    logger.warning(
                        "astream_llm attempt %d/%d failed (retrying in %.1fs, source=%s): %s",
                        attempt + 1, r.max_retries + 1, delay, retry_delay_source, e,
                    )
                    await asyncio.sleep(delay)
            raise wrap_error(last_error) from last_error  # type: ignore[misc]  # unreachable
        except Exception as e:
            last_error = e
            if model_idx < len(models) - 1:
                next_model = models[model_idx + 1]
                if on_fallback is not None:
                    on_fallback(current_model, e, next_model)
                _warnings.append(
                    f"FALLBACK: {current_model} -> {next_model} "
                    f"({type(e).__name__}: {e})"
                )
                logger.warning(
                    "Falling back from %s to %s: %s", current_model, next_model, e,
                )
                continue
            raise wrap_error(e) from e

    raise wrap_error(last_error) from last_error  # type: ignore[misc]  # unreachable


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
    texts = [input] if isinstance(input, str) else list(input)
    _log_t0 = time.monotonic()

    call_kwargs: dict[str, Any] = {"model": model, "input": texts, "timeout": timeout}
    if dimensions is not None:
        call_kwargs["dimensions"] = dimensions
    if api_base is not None:
        call_kwargs["api_base"] = api_base
    if api_key is not None:
        call_kwargs["api_key"] = api_key
    call_kwargs.update(kwargs)

    _check_model_deprecation(model)
    try:
        with _rate_limit.acquire(model):
            response = litellm.embedding(**call_kwargs)

        embeddings = [item["embedding"] for item in response.data]
        usage = dict(response.usage) if hasattr(response, "usage") and response.usage else {}
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            cost = 0.0

        result = EmbeddingResult(embeddings=embeddings, usage=usage, cost=cost, model=model)
        _io_log.log_embedding(
            model=model, input_count=len(texts),
            input_chars=sum(len(t) for t in texts),
            dimensions=len(result.embeddings[0]) if result.embeddings else None,
            usage=result.usage, cost=result.cost,
            latency_s=time.monotonic() - _log_t0, caller="embed", task=task, trace_id=trace_id,
        )
        return result
    except Exception as e:
        _io_log.log_embedding(
            model=model, input_count=len(texts),
            input_chars=sum(len(t) for t in texts), dimensions=None,
            usage=None, cost=None,
            latency_s=time.monotonic() - _log_t0, error=e, caller="embed", task=task, trace_id=trace_id,
        )
        raise


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
    texts = [input] if isinstance(input, str) else list(input)
    _log_t0 = time.monotonic()

    call_kwargs: dict[str, Any] = {"model": model, "input": texts, "timeout": timeout}
    if dimensions is not None:
        call_kwargs["dimensions"] = dimensions
    if api_base is not None:
        call_kwargs["api_base"] = api_base
    if api_key is not None:
        call_kwargs["api_key"] = api_key
    call_kwargs.update(kwargs)

    _check_model_deprecation(model)
    try:
        async with _rate_limit.aacquire(model):
            response = await litellm.aembedding(**call_kwargs)

        embeddings = [item["embedding"] for item in response.data]
        usage = dict(response.usage) if hasattr(response, "usage") and response.usage else {}
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            cost = 0.0

        result = EmbeddingResult(embeddings=embeddings, usage=usage, cost=cost, model=model)
        _io_log.log_embedding(
            model=model, input_count=len(texts),
            input_chars=sum(len(t) for t in texts),
            dimensions=len(result.embeddings[0]) if result.embeddings else None,
            usage=result.usage, cost=result.cost,
            latency_s=time.monotonic() - _log_t0, caller="aembed", task=task, trace_id=trace_id,
        )
        return result
    except Exception as e:
        _io_log.log_embedding(
            model=model, input_count=len(texts),
            input_chars=sum(len(t) for t in texts), dimensions=None,
            usage=None, cost=None,
            latency_s=time.monotonic() - _log_t0, error=e, caller="aembed", task=task, trace_id=trace_id,
        )
        raise
