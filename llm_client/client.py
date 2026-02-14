"""LLM client wrapping litellm.

Six functions (3 sync + 3 async), no class, no mutable state:
- call_llm / acall_llm: basic completion
- call_llm_structured / acall_llm_structured: instructor-based Pydantic extraction
- call_llm_with_tools / acall_llm_with_tools: tool/function calling

Features:
- Smart retry with jittered exponential backoff on transient errors,
  empty responses, and JSON parse failures
- Automatic Responses API routing for GPT-5 models (litellm.responses)
- Thinking model detection (Gemini 3/4 → budget_tokens: 0)
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

Full provider list: https://docs.litellm.ai/docs/providers
"""

from __future__ import annotations

import asyncio
import hashlib
import json as _json
import logging
import random
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, TypeVar, runtime_checkable

import litellm
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Silence litellm's noisy default logging
litellm.suppress_debug_info = True


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
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = ""
    raw_response: Any = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Cache infrastructure
# ---------------------------------------------------------------------------


@runtime_checkable
class CachePolicy(Protocol):
    """Protocol for LLM response caches. Implement get/set for custom backends."""

    def get(self, key: str) -> LLMCallResult | None: ...
    def set(self, key: str, value: LLMCallResult) -> None: ...


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


def _is_retryable(error: Exception, extra_patterns: list[str] | None = None) -> bool:
    """Check if an error is transient and worth retrying."""
    # RuntimeError is used for non-retryable conditions (e.g., truncation)
    if isinstance(error, RuntimeError):
        return False
    error_str = str(error).lower()
    patterns = _RETRYABLE_PATTERNS
    if extra_patterns:
        patterns = list(patterns) + [p.lower() for p in extra_patterns]
    return any(p in error_str for p in patterns)


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

    Gemini 3/4 thinking models allocate reasoning tokens by default,
    consuming output token budget. Setting budget_tokens=0 disables
    this so all tokens go to the actual response.
    """
    lower = model.lower()
    return "gemini-3" in lower or "gemini-4" in lower


def _extract_usage(response: Any) -> dict[str, Any]:
    """Extract token usage dict from litellm response."""
    usage = response.usage
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }


def _compute_cost(response: Any) -> float:
    """Compute cost via litellm.completion_cost, with fallback."""
    try:
        cost = float(litellm.completion_cost(completion_response=response))
        return cost
    except Exception:
        # Fallback: rough estimate based on total tokens
        total: int = response.usage.total_tokens
        fallback = total * 0.000001  # $1 per million tokens as rough floor
        logger.warning(
            "completion_cost failed, using fallback: $%.6f for %d tokens",
            fallback,
            total,
        )
        return fallback


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


def _is_responses_api_model(model: str) -> bool:
    """Check if model requires litellm.responses() instead of completion().

    GPT-5 models use OpenAI's Responses API which has different parameters
    and response format than the Chat Completions API. This function
    detects them so call_llm/acall_llm can route automatically.
    """
    return "gpt-5" in model.lower()


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


def _prepare_responses_kwargs(
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout: int,
    api_base: str | None,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Build kwargs for litellm.responses() / aresponses().

    Converts messages to input string, response_format to text parameter,
    and strips max_tokens/max_output_tokens (GPT-5 uses reasoning tokens
    before output tokens — setting limits can exhaust them on reasoning
    and return empty output while still billing you).
    """
    kwargs = dict(kwargs)  # Don't mutate caller's dict

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
                "reasoning_effort", "thinking"):
        kwargs.pop(key, None)

    resp_kwargs.update(kwargs)
    return resp_kwargs


def _extract_responses_usage(response: Any) -> dict[str, Any]:
    """Extract token usage from responses() API response."""
    usage = getattr(response, "usage", None)
    if usage is not None:
        return {
            "prompt_tokens": getattr(usage, "input_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "output_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _compute_responses_cost(response: Any, usage: dict[str, Any]) -> float:
    """Compute cost for a responses() API call."""
    # Try litellm's built-in cost calculation
    try:
        cost = float(litellm.completion_cost(completion_response=response))
        if cost > 0:
            return cost
    except Exception:
        pass

    # Try the usage.cost field (responses API sometimes includes this)
    raw_usage = getattr(response, "usage", None)
    if raw_usage and hasattr(raw_usage, "cost") and raw_usage.cost:
        return float(raw_usage.cost)

    # Fallback estimate
    total = usage["total_tokens"]
    fallback = total * 0.000001
    if total > 0:
        logger.warning(
            "completion_cost failed for responses API, "
            "using fallback: $%.6f for %d tokens",
            fallback,
            total,
        )
    return fallback


def _build_result_from_responses(
    response: Any,
    model: str,
) -> LLMCallResult:
    """Build LLMCallResult from a responses() API response."""
    # Use litellm's output_text convenience property
    content = getattr(response, "output_text", None) or ""

    usage = _extract_responses_usage(response)
    cost = _compute_responses_cost(response, usage)

    # Map responses API status to finish_reason
    status = getattr(response, "status", "completed")
    if status == "incomplete":
        details = getattr(response, "incomplete_details", None)
        reason = str(getattr(details, "reason", "")) if details else ""
        if "max_output_tokens" in reason:
            raise RuntimeError(
                f"LLM response truncated ({len(content)} chars). "
                "Responses API hit max_output_tokens limit."
            )
        finish_reason = "length"
    else:
        finish_reason = "stop"

    # Empty content (retryable)
    if not content.strip():
        raise ValueError("Empty content from LLM (responses API)")

    logger.debug(
        "LLM call (responses API): model=%s tokens=%d cost=$%.6f status=%s",
        model,
        usage["total_tokens"],
        cost,
        status,
    )

    return LLMCallResult(
        content=content,
        usage=usage,
        cost=cost,
        model=model,
        tool_calls=[],
        finish_reason=finish_reason,
        raw_response=response,
    )


# ---------------------------------------------------------------------------
# Completion API helpers
# ---------------------------------------------------------------------------


def _prepare_call_kwargs(
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout: int,
    num_retries: int,
    reasoning_effort: str | None,
    api_base: str | None,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Build kwargs dict shared by call_llm and acall_llm."""
    call_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "timeout": timeout,
        # Don't pass num_retries to litellm — our own retry loop handles
        # all retries with jittered backoff. Passing it to litellm causes
        # double retry (litellm retries HTTP errors internally, then our
        # loop retries the same errors again).
        **kwargs,
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

    # Thinking model detection: disable reasoning token budget
    if _is_thinking_model(model) and "thinking" not in kwargs:
        call_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 0}

    # GPT-5 models don't support the temperature parameter
    # (structured calls go through instructor + litellm.completion, not responses API)
    if _is_responses_api_model(model):
        call_kwargs.pop("temperature", None)

    return call_kwargs


def _build_result_from_response(
    response: Any,
    model: str,
) -> LLMCallResult:
    """Extract all fields from a litellm response into LLMCallResult."""
    content: str = response.choices[0].message.content or ""
    finish_reason: str = response.choices[0].finish_reason or ""
    tool_calls = _extract_tool_calls(response.choices[0].message)
    usage = _extract_usage(response)
    cost = _compute_cost(response)

    # Raise on truncation (non-retryable) — retrying won't help, token limit is fixed
    if finish_reason == "length":
        raise RuntimeError(
            f"LLM response truncated ({len(content)} chars). "
            "Increase max_tokens or simplify the prompt."
        )

    # Raise on empty content (retryable) — unless model made tool calls
    if not content.strip() and finish_reason != "tool_calls" and not tool_calls:
        raise ValueError("Empty content from LLM")

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
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        raw_response=response,
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
    **kwargs: Any,
) -> LLMCallResult:
    """Call any LLM via litellm. Automatically routes GPT-5 to responses API.

    Just change the model string to switch providers. Everything else
    stays the same. GPT-5 models are automatically routed through
    litellm.responses() instead of litellm.completion().

    Retries up to num_retries times with jittered exponential backoff on
    transient errors (rate limits, timeouts, empty responses, JSON parse
    failures). Non-retryable errors raise immediately.

    Args:
        model: Model name (e.g., "gpt-4o", "gpt-5-mini",
               "anthropic/claude-sonnet-4-5-20250929",
               "gemini/gemini-2.0-flash", "ollama/llama3")
        messages: Chat messages in OpenAI format
                  [{"role": "user", "content": "Hello"}]
        timeout: Request timeout in seconds
        num_retries: Number of retries on transient failure
        reasoning_effort: Reasoning effort level — only used for Claude models,
                         silently ignored for others
        api_base: Optional API base URL (e.g., for OpenRouter:
                  "https://openrouter.ai/api/v1")
        **kwargs: Additional params passed to litellm.completion
                  (e.g., temperature, max_tokens, stream).
                  For GPT-5 models, response_format is automatically
                  converted and max_tokens is stripped.

    Returns:
        LLMCallResult with content, usage, cost, model, tool_calls,
        finish_reason, and raw_response
    """
    r = _effective_retry(retry, num_retries, base_delay, max_delay, retry_on, on_retry)
    use_responses = _is_responses_api_model(model)

    if use_responses:
        call_kwargs = _prepare_responses_kwargs(
            model, messages,
            timeout=timeout,
            api_base=api_base,
            kwargs=kwargs,
        )
    else:
        call_kwargs = _prepare_call_kwargs(
            model, messages,
            timeout=timeout,
            num_retries=r.max_retries,
            reasoning_effort=reasoning_effort,
            api_base=api_base,
            kwargs=kwargs,
        )

    if cache is not None:
        key = _cache_key(model, messages, **kwargs)
        cached = cache.get(key)
        if cached is not None:
            return cached

    backoff_fn = r.backoff or exponential_backoff
    last_error: Exception | None = None
    for attempt in range(r.max_retries + 1):
        try:
            if use_responses:
                response = litellm.responses(**call_kwargs)
                result = _build_result_from_responses(response, model)
            else:
                response = litellm.completion(**call_kwargs)
                result = _build_result_from_response(response, model)
            if attempt > 0:
                logger.info("call_llm succeeded after %d retries", attempt)
            if cache is not None:
                cache.set(key, result)
            return result
        except Exception as e:
            last_error = e
            if not _check_retryable(e, r) or attempt >= r.max_retries:
                raise
            delay = backoff_fn(attempt, r.base_delay, r.max_delay)
            if r.on_retry is not None:
                r.on_retry(attempt, e, delay)
            logger.warning(
                "call_llm attempt %d/%d failed (retrying in %.1fs): %s",
                attempt + 1,
                r.max_retries + 1,
                delay,
                e,
            )
            time.sleep(delay)

    raise last_error  # type: ignore[misc]  # unreachable


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
    **kwargs: Any,
) -> tuple[T, LLMCallResult]:
    """Call LLM and get back a validated Pydantic model.

    Uses instructor + litellm for reliable structured extraction
    across all providers. No manual JSON parsing needed.

    Instructor handles validation retries internally. On top of that,
    this function retries on transient errors (rate limits, timeouts,
    empty responses) with jittered exponential backoff.

    Example:
        class Sentiment(BaseModel):
            label: str
            score: float

        result, meta = call_llm_structured(
            "gpt-4o",
            [{"role": "user", "content": "I love this!"}],
            response_model=Sentiment,
        )
        print(result.label, result.score)  # "positive", 0.95
        print(meta.cost)                    # 0.0003

    Args:
        model: Model name
        messages: Chat messages in OpenAI format
        response_model: Pydantic model class to extract
        timeout: Request timeout in seconds
        num_retries: Number of retries on failure
        reasoning_effort: Reasoning effort level (Claude models only)
        api_base: Optional API base URL (e.g., for OpenRouter)
        **kwargs: Additional params passed to litellm.completion

    Returns:
        Tuple of (parsed Pydantic model instance, LLMCallResult)
    """
    r = _effective_retry(retry, num_retries, base_delay, max_delay, retry_on, on_retry)

    import instructor

    client = instructor.from_litellm(litellm.completion)

    # Build kwargs using shared helper for thinking/reasoning detection
    base_kwargs = _prepare_call_kwargs(
        model, messages,
        timeout=timeout,
        num_retries=r.max_retries,
        reasoning_effort=reasoning_effort,
        api_base=api_base,
        kwargs=kwargs,
    )

    # Adapt for instructor: add response_model, disable instructor's internal
    # retry (our outer loop handles all retries to avoid double-retry)
    call_kwargs = {**base_kwargs, "response_model": response_model, "max_retries": 0}

    _model_fqn = f"{response_model.__module__}.{response_model.__qualname__}"
    if cache is not None:
        key = _cache_key(model, messages, response_model=_model_fqn, **kwargs)
        cached = cache.get(key)
        if cached is not None:
            # Re-parse the cached content back into the Pydantic model
            reparsed = response_model.model_validate_json(cached.content)
            return reparsed, cached

    backoff_fn = r.backoff or exponential_backoff
    last_error: Exception | None = None
    for attempt in range(r.max_retries + 1):
        try:
            parsed, completion_response = client.chat.completions.create_with_completion(
                **call_kwargs,
            )

            usage = _extract_usage(completion_response)
            cost = _compute_cost(completion_response)
            content = str(parsed.model_dump_json())
            finish_reason: str = completion_response.choices[0].finish_reason or ""

            if attempt > 0:
                logger.info("call_llm_structured succeeded after %d retries", attempt)

            llm_result = LLMCallResult(
                content=content,
                usage=usage,
                cost=cost,
                model=model,
                finish_reason=finish_reason,
                raw_response=completion_response,
            )

            if cache is not None:
                cache.set(key, llm_result)
            return parsed, llm_result
        except Exception as e:
            last_error = e
            if not _check_retryable(e, r) or attempt >= r.max_retries:
                raise
            delay = backoff_fn(attempt, r.base_delay, r.max_delay)
            if r.on_retry is not None:
                r.on_retry(attempt, e, delay)
            logger.warning(
                "call_llm_structured attempt %d/%d failed (retrying in %.1fs): %s",
                attempt + 1,
                r.max_retries + 1,
                delay,
                e,
            )
            time.sleep(delay)

    raise last_error  # type: ignore[misc]  # unreachable


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
    **kwargs: Any,
) -> LLMCallResult:
    """Call LLM with tool/function calling support.

    Example:
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        }]

        result = call_llm_with_tools("gpt-4o", messages, tools)
        if result.tool_calls:
            print(result.tool_calls[0]["function"]["name"])  # "get_weather"

    Args:
        model: Model name
        messages: Chat messages in OpenAI format
        tools: Tool definitions in OpenAI format
        timeout: Request timeout in seconds
        num_retries: Number of retries on failure
        reasoning_effort: Reasoning effort level (Claude models only)
        api_base: Optional API base URL (e.g., for OpenRouter)
        **kwargs: Additional params passed to litellm.completion

    Returns:
        LLMCallResult with tool_calls populated if model chose to use tools
    """
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
    cache: CachePolicy | None = None,
    retry: RetryPolicy | None = None,
    **kwargs: Any,
) -> LLMCallResult:
    """Async version of call_llm. Routes GPT-5 to aresponses(), others to acompletion().

    Retries up to num_retries times with jittered exponential backoff on
    transient errors.

    Args:
        model: Model name (e.g., "gpt-4o", "gpt-5-mini",
               "anthropic/claude-sonnet-4-5-20250929")
        messages: Chat messages in OpenAI format
        timeout: Request timeout in seconds
        num_retries: Number of retries on transient failure
        reasoning_effort: Reasoning effort level (Claude models only)
        api_base: Optional API base URL (e.g., for OpenRouter)
        **kwargs: Additional params passed to litellm

    Returns:
        LLMCallResult with content, usage, cost, model, tool_calls,
        finish_reason, and raw_response
    """
    r = _effective_retry(retry, num_retries, base_delay, max_delay, retry_on, on_retry)
    use_responses = _is_responses_api_model(model)

    if use_responses:
        call_kwargs = _prepare_responses_kwargs(
            model, messages,
            timeout=timeout,
            api_base=api_base,
            kwargs=kwargs,
        )
    else:
        call_kwargs = _prepare_call_kwargs(
            model, messages,
            timeout=timeout,
            num_retries=r.max_retries,
            reasoning_effort=reasoning_effort,
            api_base=api_base,
            kwargs=kwargs,
        )

    if cache is not None:
        key = _cache_key(model, messages, **kwargs)
        cached = cache.get(key)
        if cached is not None:
            return cached

    backoff_fn = r.backoff or exponential_backoff
    last_error: Exception | None = None
    for attempt in range(r.max_retries + 1):
        try:
            if use_responses:
                response = await litellm.aresponses(**call_kwargs)
                result = _build_result_from_responses(response, model)
            else:
                response = await litellm.acompletion(**call_kwargs)
                result = _build_result_from_response(response, model)
            if attempt > 0:
                logger.info("acall_llm succeeded after %d retries", attempt)
            if cache is not None:
                cache.set(key, result)
            return result
        except Exception as e:
            last_error = e
            if not _check_retryable(e, r) or attempt >= r.max_retries:
                raise
            delay = backoff_fn(attempt, r.base_delay, r.max_delay)
            if r.on_retry is not None:
                r.on_retry(attempt, e, delay)
            logger.warning(
                "acall_llm attempt %d/%d failed (retrying in %.1fs): %s",
                attempt + 1,
                r.max_retries + 1,
                delay,
                e,
            )
            await asyncio.sleep(delay)

    raise last_error  # type: ignore[misc]  # unreachable


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
    cache: CachePolicy | None = None,
    retry: RetryPolicy | None = None,
    **kwargs: Any,
) -> tuple[T, LLMCallResult]:
    """Async version of call_llm_structured.

    Uses instructor + litellm.acompletion for async structured extraction.
    Retries on transient errors with jittered exponential backoff.

    Args:
        model: Model name
        messages: Chat messages in OpenAI format
        response_model: Pydantic model class to extract
        timeout: Request timeout in seconds
        num_retries: Number of retries on failure
        reasoning_effort: Reasoning effort level (Claude models only)
        api_base: Optional API base URL (e.g., for OpenRouter)
        **kwargs: Additional params passed to litellm.acompletion

    Returns:
        Tuple of (parsed Pydantic model instance, LLMCallResult)
    """
    r = _effective_retry(retry, num_retries, base_delay, max_delay, retry_on, on_retry)

    import instructor

    client = instructor.from_litellm(litellm.acompletion)

    # Build kwargs using shared helper for thinking/reasoning detection
    base_kwargs = _prepare_call_kwargs(
        model, messages,
        timeout=timeout,
        num_retries=r.max_retries,
        reasoning_effort=reasoning_effort,
        api_base=api_base,
        kwargs=kwargs,
    )

    # Adapt for instructor: add response_model, disable instructor's internal
    # retry (our outer loop handles all retries to avoid double-retry)
    call_kwargs = {**base_kwargs, "response_model": response_model, "max_retries": 0}

    _model_fqn = f"{response_model.__module__}.{response_model.__qualname__}"
    if cache is not None:
        key = _cache_key(model, messages, response_model=_model_fqn, **kwargs)
        cached = cache.get(key)
        if cached is not None:
            reparsed = response_model.model_validate_json(cached.content)
            return reparsed, cached

    backoff_fn = r.backoff or exponential_backoff
    last_error: Exception | None = None
    for attempt in range(r.max_retries + 1):
        try:
            parsed, completion_response = await client.chat.completions.create_with_completion(
                **call_kwargs,
            )

            usage = _extract_usage(completion_response)
            cost = _compute_cost(completion_response)
            content = str(parsed.model_dump_json())
            finish_reason: str = completion_response.choices[0].finish_reason or ""

            if attempt > 0:
                logger.info("acall_llm_structured succeeded after %d retries", attempt)

            llm_result = LLMCallResult(
                content=content,
                usage=usage,
                cost=cost,
                model=model,
                finish_reason=finish_reason,
                raw_response=completion_response,
            )

            if cache is not None:
                cache.set(key, llm_result)
            return parsed, llm_result
        except Exception as e:
            last_error = e
            if not _check_retryable(e, r) or attempt >= r.max_retries:
                raise
            delay = backoff_fn(attempt, r.base_delay, r.max_delay)
            if r.on_retry is not None:
                r.on_retry(attempt, e, delay)
            logger.warning(
                "acall_llm_structured attempt %d/%d failed (retrying in %.1fs): %s",
                attempt + 1,
                r.max_retries + 1,
                delay,
                e,
            )
            await asyncio.sleep(delay)

    raise last_error  # type: ignore[misc]  # unreachable


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
    cache: CachePolicy | None = None,
    retry: RetryPolicy | None = None,
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
        **kwargs: Additional params passed to litellm.acompletion

    Returns:
        LLMCallResult with tool_calls populated if model chose to use tools
    """
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
        tools=tools,
        **kwargs,
    )
