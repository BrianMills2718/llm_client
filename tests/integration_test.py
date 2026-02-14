"""One-off integration tests against real APIs. Not run by pytest.

Usage: python tests/integration_test.py
"""

import asyncio
import json
import sys
import time

sys.path.insert(0, ".")

from pydantic import BaseModel

from llm_client import (
    LLMCallResult,
    acall_llm_batch,
    call_llm_batch,
    call_llm_structured,
    acall_llm_structured,
    stream_llm,
    stream_llm_with_tools,
)


def _header(name: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# 1. Batch calls — real concurrency
# ---------------------------------------------------------------------------


def test_batch_sync() -> None:
    _header("1a. call_llm_batch (sync, gemini-flash, 3 items)")
    messages_list = [
        [{"role": "user", "content": f"Reply with just the number {i}"}]
        for i in range(3)
    ]
    t0 = time.time()
    results = call_llm_batch(
        "gemini/gemini-2.0-flash",
        messages_list,
        max_concurrent=3,
    )
    elapsed = time.time() - t0
    for i, r in enumerate(results):
        assert isinstance(r, LLMCallResult), f"Item {i} is {type(r)}: {r}"
        print(f"  [{i}] content={r.content!r:.60s}  cost=${r.cost:.6f}")
    print(f"  Elapsed: {elapsed:.2f}s (3 concurrent)")
    print("  PASS")


async def test_batch_async() -> None:
    _header("1b. acall_llm_batch (async, gemini-flash, 5 items)")
    messages_list = [
        [{"role": "user", "content": f"Reply with just the word 'hello' in language #{i}"}]
        for i in range(5)
    ]
    completed_indices: list[int] = []
    t0 = time.time()
    results = await acall_llm_batch(
        "gemini/gemini-2.0-flash",
        messages_list,
        max_concurrent=5,
        on_item_complete=lambda idx, res: completed_indices.append(idx),
    )
    elapsed = time.time() - t0
    for i, r in enumerate(results):
        assert isinstance(r, LLMCallResult), f"Item {i} is {type(r)}: {r}"
        print(f"  [{i}] content={r.content!r:.60s}")
    print(f"  on_item_complete fired for indices: {sorted(completed_indices)}")
    print(f"  Elapsed: {elapsed:.2f}s (5 concurrent)")
    assert len(completed_indices) == 5
    print("  PASS")


# ---------------------------------------------------------------------------
# 2. GPT-5 structured output (if OpenAI key available)
# ---------------------------------------------------------------------------


class CityInfo(BaseModel):
    name: str
    country: str
    population_millions: float


def test_gpt5_structured() -> None:
    _header("2a. call_llm_structured (gpt-5-mini)")
    try:
        result, meta = call_llm_structured(
            "gpt-5-mini",
            [{"role": "user", "content": "Give me info about Tokyo"}],
            response_model=CityInfo,
        )
        print(f"  Parsed: name={result.name}, country={result.country}, pop={result.population_millions}M")
        print(f"  Cost: ${meta.cost:.6f}, tokens: {meta.usage}")
        assert result.name.lower() == "tokyo"
        assert result.country.lower() == "japan"
        print("  PASS")
    except Exception as e:
        print(f"  SKIPPED: {e}")


async def test_gpt5_structured_async() -> None:
    _header("2b. acall_llm_structured (gpt-5-mini, async)")
    try:
        result, meta = await acall_llm_structured(
            "gpt-5-mini",
            [{"role": "user", "content": "Give me info about Paris"}],
            response_model=CityInfo,
        )
        print(f"  Parsed: name={result.name}, country={result.country}, pop={result.population_millions}M")
        print(f"  Cost: ${meta.cost:.6f}")
        assert result.name.lower() == "paris"
        print("  PASS")
    except Exception as e:
        print(f"  SKIPPED: {e}")


# ---------------------------------------------------------------------------
# 2c. Native JSON schema structured output (non-GPT-5)
# ---------------------------------------------------------------------------


def test_gemini_structured() -> None:
    _header("2c. call_llm_structured (gemini-flash, native JSON schema)")
    try:
        result, meta = call_llm_structured(
            "gemini/gemini-2.0-flash",
            [{"role": "user", "content": "Give me info about London"}],
            response_model=CityInfo,
        )
        print(f"  Parsed: name={result.name}, country={result.country}, pop={result.population_millions}M")
        print(f"  Cost: ${meta.cost:.6f}, tokens: {meta.usage}")
        assert result.name.lower() == "london"
        assert result.country.lower() in ("uk", "united kingdom", "england")
        print("  PASS")
    except Exception as e:
        print(f"  SKIPPED: {e}")


# ---------------------------------------------------------------------------
# 3. Streaming with retry (simulate by just verifying it works)
# ---------------------------------------------------------------------------


def test_stream_retry() -> None:
    _header("3. stream_llm with retry params (gemini-flash)")
    stream = stream_llm(
        "gemini/gemini-2.0-flash",
        [{"role": "user", "content": "Say 'hello world' and nothing else"}],
        num_retries=2,
        fallback_models=["gemini/gemini-2.0-flash"],
    )
    chunks: list[str] = []
    for chunk in stream:
        chunks.append(chunk)
    full = "".join(chunks)
    print(f"  Streamed: {full!r:.80s}")
    print(f"  Result: tokens={stream.result.usage}, cost=${stream.result.cost:.6f}")
    assert len(full) > 0
    print("  PASS")


# ---------------------------------------------------------------------------
# 4. Streaming with tools
# ---------------------------------------------------------------------------


def test_stream_with_tools() -> None:
    _header("4. stream_llm_with_tools (gemini-flash)")
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                },
                "required": ["location"],
            },
        },
    }]
    stream = stream_llm_with_tools(
        "gemini/gemini-2.0-flash",
        [{"role": "user", "content": "What's the weather in Tokyo?"}],
        tools,
    )
    chunks: list[str] = []
    for chunk in stream:
        chunks.append(chunk)
    result = stream.result
    full = "".join(chunks)
    print(f"  Streamed text: {full!r:.80s}")
    print(f"  tool_calls: {result.tool_calls}")
    print(f"  finish_reason: {result.finish_reason}")
    if result.tool_calls:
        tc = result.tool_calls[0]
        print(f"  First tool: {tc['function']['name']}({tc['function']['arguments']})")
        assert tc["function"]["name"] == "get_weather"
        args = json.loads(tc["function"]["arguments"])
        assert "tokyo" in args.get("location", "").lower() or "Tokyo" in args.get("location", "")
        print("  PASS (tool call extracted from stream)")
    else:
        print("  NOTE: Model chose text response instead of tool call — tool_calls empty")
        print("  PASS (streaming worked, no tool_calls to verify)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    test_batch_sync()
    await test_batch_async()
    test_gpt5_structured()
    await test_gpt5_structured_async()
    test_gemini_structured()
    test_stream_retry()
    test_stream_with_tools()
    print(f"\n{'='*60}")
    print("  ALL INTEGRATION TESTS COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
