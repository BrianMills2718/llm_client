"""Retry infrastructure for LLM client.

Owns the RetryPolicy dataclass, Hooks dataclass, backoff strategies,
retryability classification, and retry-delay computation. All retry-related
logic for the call paths lives here so it can be tested in isolation.

This module depends on data_types (for LLMCallResult type in Hooks) and
errors (for LLMEmptyResponseError). It must not import from client.py.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Callable

from llm_client.core.data_types import LLMCallResult
from llm_client.core.errors import LLMEmptyResponseError, _QUOTA_PATTERNS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retryable pattern lists
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
    *_QUOTA_PATTERNS,
]


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------


def _error_text(error: Exception) -> str:
    """Return a stable non-empty error string for classification/logging."""
    text = str(error).strip()
    if text:
        return text
    return type(error).__name__


def _error_status_code(error: Exception) -> int | None:
    """Extract HTTP status code from litellm/generic errors."""
    for attr in ("status_code", "status", "http_status", "code"):
        value = getattr(error, attr, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)

    response = getattr(error, "response", None)
    if response is not None:
        for attr in ("status_code", "status", "code"):
            value = getattr(response, attr, None)
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)

    text = str(error)
    match = re.search(r"\b(401|403|404|409|429|500|502|503)\b", text)
    if match:
        return int(match.group(1))
    return None


# ---------------------------------------------------------------------------
# Retry delay helpers
# ---------------------------------------------------------------------------


def _coerce_retry_delay_seconds(
    raw_value: Any,
    *,
    max_seconds: float | None = 600.0,
) -> float | None:
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
    if max_seconds is not None:
        return min(value, max_seconds)
    return value


def _retry_delay_hint_seconds(
    error: Exception,
    *,
    max_seconds: float | None = 600.0,
) -> float | None:
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
        hint = _coerce_retry_delay_seconds(
            f"{m.group(1)}{m.group(2) or 's'}",
            max_seconds=max_seconds,
        )
        if hint is not None:
            return hint
    return None


def _retry_delay_hint(
    error: Exception,
    *,
    max_seconds: float | None = 600.0,
) -> tuple[float | None, str]:
    """Return retry hint delay with source classification."""
    for attr in ("retry_after", "retry_after_seconds", "retry_after_s", "retry_delay"):
        hint = _coerce_retry_delay_seconds(
            getattr(error, attr, None),
            max_seconds=max_seconds,
        )
        if hint is not None:
            return hint, "structured"

    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        header_value: Any = None
        if hasattr(headers, "get"):
            header_value = headers.get("retry-after") or headers.get("Retry-After")
        hint = _coerce_retry_delay_seconds(header_value, max_seconds=max_seconds)
        if hint is not None:
            return hint, "structured"

    hint = _retry_delay_hint_seconds(error, max_seconds=max_seconds)
    if hint is not None:
        return hint, "parsed"
    return None, "none"


def _max_in_call_retry_delay_seconds() -> float:
    """Return the max provider hint to honor inside a single call attempt chain."""
    raw = os.environ.get("LLM_CLIENT_MAX_IN_CALL_RETRY_DELAY_S")
    if raw is None or not raw.strip():
        return 120.0
    try:
        value = float(raw)
    except ValueError:
        return 120.0
    return 120.0 if value <= 0 else value


# ---------------------------------------------------------------------------
# Retryability classification
# ---------------------------------------------------------------------------


def _is_retryable(error: Exception, extra_patterns: list[str] | None = None) -> bool:
    """Check if an error is transient and worth retrying.

    Uses litellm exception types for reliable classification, with string
    pattern matching as fallback for generic exceptions.
    """
    if isinstance(error, LLMEmptyResponseError):
        return bool(error.retryable)

    # Timeout exceptions are transient even when message is empty.
    if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
        return True

    # RuntimeError is used for non-retryable conditions (e.g., truncation)
    if isinstance(error, RuntimeError):
        return False

    # -- Check litellm exception types (preferred over string matching) -------
    try:
        import litellm as _lt

        def _litellm_error_types(*names: str) -> tuple[type[BaseException], ...]:
            out: list[type[BaseException]] = []
            for name in names:
                candidate = getattr(_lt, name, None)
                if isinstance(candidate, type) and issubclass(candidate, BaseException):
                    out.append(candidate)
            return tuple(out)

        # Permanent failures — never retry
        permanent_types = _litellm_error_types(
            "AuthenticationError",      # 401: bad API key
            "PermissionDeniedError",    # 403: forbidden
            "BudgetExceededError",      # litellm budget limit
            "ContentPolicyViolationError",  # content filter
            "NotFoundError",            # 404: model doesn't exist
        )
        if permanent_types and isinstance(error, permanent_types):
            return False

        # RateLimitError (429) is ambiguous — could be transient rate limit
        # or permanent quota exhaustion. Check the message.
        rate_limit_types = _litellm_error_types("RateLimitError")
        if rate_limit_types and isinstance(error, rate_limit_types):
            error_str = str(error).lower()
            # Provider-specified retry windows are considered retryable even
            # when the message includes "quota" phrasing, but only when the
            # requested wait is short enough to make in-call retry sensible.
            hint_delay, _hint_source = _retry_delay_hint(error, max_seconds=None)
            if hint_delay is not None and any(p in error_str for p in _NON_RETRYABLE_PATTERNS):
                return hint_delay <= _max_in_call_retry_delay_seconds()
            if hint_delay is not None:
                return True
            if any(p in error_str for p in _NON_RETRYABLE_PATTERNS):
                return False
            return True  # transient rate limit — retry

        # JSON Schema validation failures — retryable because the structured
        # runtime appends a repair prompt with the specific validation errors,
        # giving the model a chance to self-correct on retry.
        json_schema_types = _litellm_error_types("JSONSchemaValidationError")
        if json_schema_types and isinstance(error, json_schema_types):
            return True

        # Transient server errors — always retry
        transient_types = _litellm_error_types(
            "InternalServerError",   # 500
            "ServiceUnavailableError",  # 503
            "APIConnectionError",    # network issues
            "BadGatewayError",       # 502
        )
        if transient_types and isinstance(error, transient_types):
            return True
    except ImportError:
        pass  # litellm not available, fall through to string matching

    # -- Fallback: string pattern matching for generic exceptions --------------
    error_str = _error_text(error).lower()

    # Check non-retryable patterns first
    if any(p in error_str for p in _NON_RETRYABLE_PATTERNS):
        return False

    patterns = _RETRYABLE_PATTERNS
    if extra_patterns:
        patterns = list(patterns) + [p.lower() for p in extra_patterns]
    return any(p in error_str for p in patterns)


# ---------------------------------------------------------------------------
# Backoff strategies
# ---------------------------------------------------------------------------


def exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 30.0) -> float:
    """Exponential backoff with jitter, capped at *max_delay*."""
    delay = base_delay * (2 ** attempt)
    jitter = random.uniform(0.5, 1.5)
    return float(min(delay * jitter, max_delay))


def linear_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 30.0) -> float:
    """Linear backoff with jitter, capped at *max_delay*."""
    delay = base_delay * (attempt + 1)
    jitter = random.uniform(0.8, 1.2)
    return float(min(delay * jitter, max_delay))


def fixed_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 30.0) -> float:
    """Fixed delay (no escalation), capped at *max_delay*."""
    return float(min(base_delay, max_delay))


# Backward-compat alias (used by existing tests)
_calculate_backoff = exponential_backoff


# ---------------------------------------------------------------------------
# RetryPolicy and Hooks
# ---------------------------------------------------------------------------


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
        backoff: Backoff function ``(attempt, base_delay, max_delay) -> delay``.
            Defaults to :func:`exponential_backoff`. Also available:
            :func:`linear_backoff`, :func:`fixed_backoff`, or any custom
            callable.
        should_retry: Fully custom retryability check ``(error) -> bool``.
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
    integration. All fields are optional -- set only the ones you need.

    Example::

        hooks = Hooks(
            before_call=lambda model, msgs, kw: print(f"Calling {model}"),
            after_call=lambda result: print(f"Got {len(result.content)} chars"),
            on_error=lambda err, attempt: print(f"Attempt {attempt} failed: {err}"),
        )
        result = call_llm("gpt-4o", messages, hooks=hooks)

    Attributes:
        before_call: ``(model, messages, kwargs) -> None``. Fired before each
            LLM API call (including retries and fallbacks).
        after_call: ``(LLMCallResult) -> None``. Fired after a successful call.
        on_error: ``(error, attempt) -> None``. Fired on each failed attempt.
    """

    before_call: Callable[[str, list[dict[str, Any]], dict[str, Any]], None] | None = None
    after_call: Callable[[LLMCallResult], None] | None = None
    on_error: Callable[[Exception, int], None] | None = None


# ---------------------------------------------------------------------------
# Retry policy resolution helpers
# ---------------------------------------------------------------------------


def _effective_retry(
    retry: RetryPolicy | None,
    num_retries: int,
    base_delay: float,
    max_delay: float,
    retry_on: list[str] | None,
    on_retry: Callable[[int, Exception, float], None] | None,
) -> RetryPolicy:
    """Resolve a RetryPolicy -- use the explicit object or build one from individual params."""
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
