"""Thin LLM client wrapping litellm.

Three functions, no class, no mutable state:
- call_llm: basic completion
- call_llm_structured: instructor-based Pydantic extraction
- call_llm_with_tools: tool/function calling

Cost returned per-call as a value, not stored as mutable state.
Retry delegated to litellm's num_retries.

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

import logging
from dataclasses import dataclass, field
from typing import Any, TypeVar

import litellm
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Silence litellm's noisy default logging
litellm.suppress_debug_info = True


@dataclass
class LLMCallResult:
    """Result from an LLM call. Returned by all call_llm* functions.

    Attributes:
        content: The text response from the model
        usage: Token counts (prompt_tokens, completion_tokens, total_tokens)
        cost: Cost in USD for this call
        model: The model string that was used
        tool_calls: List of tool calls if the model invoked tools, else empty
    """

    content: str
    usage: dict[str, Any]
    cost: float
    model: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


def _is_claude_model(model: str) -> bool:
    """Check if model string refers to a Claude model."""
    return "claude" in model.lower() or "anthropic" in model.lower()


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

    Args:
        model: Model name (e.g., "gpt-4o", "anthropic/claude-sonnet-4-5-20250929",
               "gemini/gemini-2.0-flash", "ollama/llama3")
        messages: Chat messages in OpenAI format
                  [{"role": "user", "content": "Hello"}]
        timeout: Request timeout in seconds
        num_retries: Number of retries on failure (litellm built-in)
        reasoning_effort: Reasoning effort level — only used for Claude models,
                         silently ignored for others
        api_base: Optional API base URL (e.g., for OpenRouter:
                  "https://openrouter.ai/api/v1")
        **kwargs: Additional params passed to litellm.completion
                  (e.g., temperature, max_tokens, stream)

    Returns:
        LLMCallResult with content, usage, cost, model, and tool_calls
    """
    call_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "timeout": timeout,
        "num_retries": num_retries,
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

    response = litellm.completion(**call_kwargs)

    content: str = response.choices[0].message.content or ""
    usage = _extract_usage(response)
    cost = _compute_cost(response)
    tool_calls = _extract_tool_calls(response.choices[0].message)

    logger.debug(
        "LLM call: model=%s tokens=%d cost=$%.6f",
        model,
        usage["total_tokens"],
        cost,
    )

    return LLMCallResult(
        content=content,
        usage=usage,
        cost=cost,
        model=model,
        tool_calls=tool_calls,
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
    **kwargs: Any,
) -> tuple[T, LLMCallResult]:
    """Call LLM and get back a validated Pydantic model.

    Uses instructor + litellm for reliable structured extraction
    across all providers. No manual JSON parsing needed.

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

    call_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_model": response_model,
        "timeout": timeout,
        "max_retries": num_retries,
        **kwargs,
    }

    if api_base is not None:
        call_kwargs["api_base"] = api_base

    # Only pass reasoning_effort for Claude models
    if reasoning_effort and _is_claude_model(model):
        call_kwargs["reasoning_effort"] = reasoning_effort

    result, raw_response = client.chat.completions.create_with_completion(
        **call_kwargs,
    )

    usage = _extract_usage(raw_response)
    cost = _compute_cost(raw_response)
    content = str(result.model_dump_json())

    llm_result = LLMCallResult(
        content=content,
        usage=usage,
        cost=cost,
        model=model,
    )

    return result, llm_result


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

    Args:
        model: Model name (e.g., "gpt-4o", "anthropic/claude-sonnet-4-5-20250929")
        messages: Chat messages in OpenAI format
        timeout: Request timeout in seconds
        num_retries: Number of retries on failure (litellm built-in)
        reasoning_effort: Reasoning effort level (Claude models only)
        api_base: Optional API base URL (e.g., for OpenRouter)
        **kwargs: Additional params passed to litellm.acompletion

    Returns:
        LLMCallResult with content, usage, cost, model, and tool_calls
    """
    call_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "timeout": timeout,
        "num_retries": num_retries,
        **kwargs,
    }

    if api_base is not None:
        call_kwargs["api_base"] = api_base

    if reasoning_effort and _is_claude_model(model):
        call_kwargs["reasoning_effort"] = reasoning_effort
    elif reasoning_effort:
        logger.debug(
            "reasoning_effort=%s ignored for non-Claude model %s",
            reasoning_effort,
            model,
        )

    response = await litellm.acompletion(**call_kwargs)

    content: str = response.choices[0].message.content or ""
    usage = _extract_usage(response)
    cost = _compute_cost(response)
    tool_calls = _extract_tool_calls(response.choices[0].message)

    logger.debug(
        "LLM call: model=%s tokens=%d cost=$%.6f",
        model,
        usage["total_tokens"],
        cost,
    )

    return LLMCallResult(
        content=content,
        usage=usage,
        cost=cost,
        model=model,
        tool_calls=tool_calls,
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
    **kwargs: Any,
) -> tuple[T, LLMCallResult]:
    """Async version of call_llm_structured.

    Uses instructor + litellm.acompletion for async structured extraction.

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

    call_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_model": response_model,
        "timeout": timeout,
        "max_retries": num_retries,
        **kwargs,
    }

    if api_base is not None:
        call_kwargs["api_base"] = api_base

    if reasoning_effort and _is_claude_model(model):
        call_kwargs["reasoning_effort"] = reasoning_effort

    result, raw_response = await client.chat.completions.create_with_completion(
        **call_kwargs,
    )

    usage = _extract_usage(raw_response)
    cost = _compute_cost(raw_response)
    content = str(result.model_dump_json())

    llm_result = LLMCallResult(
        content=content,
        usage=usage,
        cost=cost,
        model=model,
    )

    return result, llm_result


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
