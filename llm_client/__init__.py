"""LLM client wrapping litellm.

Swap any model by changing the model string. Everything else stays the same.

Usage:
    from llm_client import call_llm, call_llm_structured, call_llm_with_tools, stream_llm

    # Sync
    result = call_llm("gpt-4o", [{"role": "user", "content": "Hello"}])
    print(result.content, result.cost)

    # Batch (concurrent)
    results = call_llm_batch("gpt-4o", [msgs1, msgs2, msgs3], max_concurrent=5)

    # Streaming
    for chunk in stream_llm("gpt-4o", [{"role": "user", "content": "Hello"}]):
        print(chunk, end="")

    # Async
    from llm_client import acall_llm, astream_llm

    result = await acall_llm("gpt-4o", [{"role": "user", "content": "Hello"}])
"""

from llm_client.client import (
    AsyncCachePolicy,
    AsyncLLMStream,
    CachePolicy,
    Hooks,
    LLMCallResult,
    LLMStream,
    LRUCache,
    RetryPolicy,
    acall_llm,
    acall_llm_batch,
    acall_llm_structured,
    acall_llm_structured_batch,
    acall_llm_with_tools,
    astream_llm,
    astream_llm_with_tools,
    call_llm,
    call_llm_batch,
    call_llm_structured,
    call_llm_structured_batch,
    call_llm_with_tools,
    exponential_backoff,
    fixed_backoff,
    linear_backoff,
    stream_llm,
    stream_llm_with_tools,
    strip_fences,
)

__all__ = [
    "AsyncCachePolicy",
    "AsyncLLMStream",
    "CachePolicy",
    "Hooks",
    "LLMCallResult",
    "LLMStream",
    "LRUCache",
    "RetryPolicy",
    "acall_llm",
    "acall_llm_batch",
    "acall_llm_structured",
    "acall_llm_structured_batch",
    "acall_llm_with_tools",
    "astream_llm",
    "astream_llm_with_tools",
    "call_llm",
    "call_llm_batch",
    "call_llm_structured",
    "call_llm_structured_batch",
    "call_llm_with_tools",
    "exponential_backoff",
    "fixed_backoff",
    "linear_backoff",
    "stream_llm",
    "stream_llm_with_tools",
    "strip_fences",
]
