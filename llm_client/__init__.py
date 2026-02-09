"""Thin LLM client wrapping litellm.

Swap any model by changing the model string. Everything else stays the same.

Usage:
    from llm_client import call_llm, call_llm_structured, call_llm_with_tools

    result = call_llm("gpt-4", [{"role": "user", "content": "Hello"}])
    print(result.content, result.cost)
"""

from llm_client.client import (
    LLMCallResult,
    call_llm,
    call_llm_structured,
    call_llm_with_tools,
)

__all__ = [
    "LLMCallResult",
    "call_llm",
    "call_llm_structured",
    "call_llm_with_tools",
]
