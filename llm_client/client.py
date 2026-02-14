"""LLM client wrapping litellm.

Six functions (3 sync + 3 async), no class, no mutable state:
- call_llm / acall_llm: basic completion
- call_llm_structured / acall_llm_structured: instructor-based Pydantic extraction
- call_llm_with_tools / acall_llm_with_tools: tool/function calling

Features:
- Smart retry with jittered exponential backoff on transient errors,
  empty responses, and JSON parse failures
- Thinking model detection (Gemini 3/4 → budget_tokens: 0)
- Fence stripping utility for manual JSON parsing
- Cost tracking via litellm.completion_cost
- finish_reason + raw_response on every result

Supported providers (just change the model string):
    call_llm("gpt-4o", messages)                     # OpenAI
    call_llm("anthropic/claude-sonnet-4-5-20250929", messages)  # Anthropic
    call_llm("gemini/gemini-2.0-flash", messages)     # Google
    call_llm("mistral/mistral-large", messages)       # Mistral
    call_llm("ollama/llama3", messages)               # Local Ollama
    call_llm("bedrock/anthropic.claude-v2", messages)  # AWS Bedrock

Full provider list: https://docs.litellm.ai/docs/providers
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

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


def _is_retryable(error: Exception) -> bool:
    """Check if an error is transient and worth retrying."""
    # RuntimeError is used for non-retryable conditions (e.g., truncation)
    if isinstance(error, RuntimeError):
        return False
    error_str = str(error).lower()
    return any(p in error_str for p in _RETRYABLE_PATTERNS)


def _calculate_backoff(attempt: int, base_delay: float = 1.0) -> float:
    """Exponential backoff with jitter, capped at 30s."""
    delay = base_delay * (2 ** attempt)
    jitter = random.uniform(0.5, 1.5)
    return min(delay * jitter, 30.0)


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
    **kwargs: Any,
) -> LLMCallResult:
    """Call any LLM via litellm.completion.

    Just change the model string to switch providers. Everything else
    stays the same.

    Retries up to num_retries times with jittered exponential backoff on
    transient errors (rate limits, timeouts, empty responses, JSON parse
    failures). Non-retryable errors raise immediately.

    Args:
        model: Model name (e.g., "gpt-4o", "anthropic/claude-sonnet-4-5-20250929",
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
                  (e.g., temperature, max_tokens, stream)

    Returns:
        LLMCallResult with content, usage, cost, model, tool_calls,
        finish_reason, and raw_response
    """
    call_kwargs = _prepare_call_kwargs(
        model, messages,
        timeout=timeout,
        num_retries=num_retries,
        reasoning_effort=reasoning_effort,
        api_base=api_base,
        kwargs=kwargs,
    )

    last_error: Exception | None = None
    for attempt in range(num_retries + 1):
        try:
            response = litellm.completion(**call_kwargs)
            result = _build_result_from_response(response, model)
            if attempt > 0:
                logger.info("call_llm succeeded after %d retries", attempt)
            return result
        except Exception as e:
            last_error = e
            if not _is_retryable(e) or attempt >= num_retries:
                raise
            delay = _calculate_backoff(attempt)
            logger.warning(
                "call_llm attempt %d/%d failed (retrying in %.1fs): %s",
                attempt + 1,
                num_retries + 1,
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
    import instructor

    client = instructor.from_litellm(litellm.completion)

    # Build kwargs using shared helper for thinking/reasoning detection
    base_kwargs = _prepare_call_kwargs(
        model, messages,
        timeout=timeout,
        num_retries=num_retries,
        reasoning_effort=reasoning_effort,
        api_base=api_base,
        kwargs=kwargs,
    )

    # Adapt for instructor: add response_model, disable instructor's internal
    # retry (our outer loop handles all retries to avoid double-retry)
    call_kwargs = {**base_kwargs, "response_model": response_model, "max_retries": 0}

    last_error: Exception | None = None
    for attempt in range(num_retries + 1):
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

            return parsed, llm_result
        except Exception as e:
            last_error = e
            if not _is_retryable(e) or attempt >= num_retries:
                raise
            delay = _calculate_backoff(attempt)
            logger.warning(
                "call_llm_structured attempt %d/%d failed (retrying in %.1fs): %s",
                attempt + 1,
                num_retries + 1,
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
    **kwargs: Any,
) -> LLMCallResult:
    """Async version of call_llm. Uses litellm.acompletion.

    Retries up to num_retries times with jittered exponential backoff on
    transient errors.

    Args:
        model: Model name (e.g., "gpt-4o", "anthropic/claude-sonnet-4-5-20250929")
        messages: Chat messages in OpenAI format
        timeout: Request timeout in seconds
        num_retries: Number of retries on transient failure
        reasoning_effort: Reasoning effort level (Claude models only)
        api_base: Optional API base URL (e.g., for OpenRouter)
        **kwargs: Additional params passed to litellm.acompletion

    Returns:
        LLMCallResult with content, usage, cost, model, tool_calls,
        finish_reason, and raw_response
    """
    call_kwargs = _prepare_call_kwargs(
        model, messages,
        timeout=timeout,
        num_retries=num_retries,
        reasoning_effort=reasoning_effort,
        api_base=api_base,
        kwargs=kwargs,
    )

    last_error: Exception | None = None
    for attempt in range(num_retries + 1):
        try:
            response = await litellm.acompletion(**call_kwargs)
            result = _build_result_from_response(response, model)
            if attempt > 0:
                logger.info("acall_llm succeeded after %d retries", attempt)
            return result
        except Exception as e:
            last_error = e
            if not _is_retryable(e) or attempt >= num_retries:
                raise
            delay = _calculate_backoff(attempt)
            logger.warning(
                "acall_llm attempt %d/%d failed (retrying in %.1fs): %s",
                attempt + 1,
                num_retries + 1,
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
    import instructor

    client = instructor.from_litellm(litellm.acompletion)

    # Build kwargs using shared helper for thinking/reasoning detection
    base_kwargs = _prepare_call_kwargs(
        model, messages,
        timeout=timeout,
        num_retries=num_retries,
        reasoning_effort=reasoning_effort,
        api_base=api_base,
        kwargs=kwargs,
    )

    # Adapt for instructor: add response_model, disable instructor's internal
    # retry (our outer loop handles all retries to avoid double-retry)
    call_kwargs = {**base_kwargs, "response_model": response_model, "max_retries": 0}

    last_error: Exception | None = None
    for attempt in range(num_retries + 1):
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

            return parsed, llm_result
        except Exception as e:
            last_error = e
            if not _is_retryable(e) or attempt >= num_retries:
                raise
            delay = _calculate_backoff(attempt)
            logger.warning(
                "acall_llm_structured attempt %d/%d failed (retrying in %.1fs): %s",
                attempt + 1,
                num_retries + 1,
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
        tools=tools,
        **kwargs,
    )
