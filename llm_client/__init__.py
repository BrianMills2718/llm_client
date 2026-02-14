"""Thin LLM client wrapping litellm.

Swap any model by changing the model string. Everything else stays the same.

Usage:
    from llm_client import call_llm, call_llm_structured, call_llm_with_tools

    # Sync
    result = call_llm("gpt-4", [{"role": "user", "content": "Hello"}])
    print(result.content, result.cost)

    # Async
    from llm_client import acall_llm, acall_llm_structured, acall_llm_with_tools

    result = await acall_llm("gpt-4", [{"role": "user", "content": "Hello"}])
"""

from llm_client.client import (
    LLMCallResult,
    acall_llm,
    acall_llm_structured,
    acall_llm_with_tools,
    call_llm,
    call_llm_structured,
    call_llm_with_tools,
    strip_fences,
)

__all__ = [
    "LLMCallResult",
    "acall_llm",
    "acall_llm_structured",
    "acall_llm_with_tools",
    "call_llm",
    "call_llm_structured",
    "call_llm_with_tools",
    "strip_fences",
]
