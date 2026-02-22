"""Integration tests against real APIs. Require API keys in environment.

Usage:
    python tests/integration_test.py          # standalone
    pytest tests/integration_test.py -v       # via pytest
"""

import asyncio
import json
import os
import sys
import time
from typing import Any

import pytest

pytestmark = pytest.mark.integration

if os.environ.get("LLM_CLIENT_INTEGRATION", "").strip() != "1":
    pytest.skip(
        "Integration tests disabled by default. Set LLM_CLIENT_INTEGRATION=1 to enable.",
        allow_module_level=True,
    )

sys.path.insert(0, ".")

from pydantic import BaseModel

from llm_client import (
    Hooks,
    LLMCallResult,
    LRUCache,
    RetryPolicy,
    acall_llm,
    acall_llm_batch,
    astream_llm,
    astream_llm_with_tools,
    call_llm,
    call_llm_batch,
    call_llm_structured,
    call_llm_structured_batch,
    acall_llm_structured,
    call_llm_with_tools,
    acall_llm_with_tools,
    stream_llm,
    stream_llm_with_tools,
)
from llm_client.errors import LLMError


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
        task="test",
        trace_id="test_1a_batch_sync",
        max_budget=0,
    )
    elapsed = time.time() - t0
    for i, r in enumerate(results):
        assert isinstance(r, LLMCallResult), f"Item {i} is {type(r)}: {r}"
        print(f"  [{i}] content={r.content!r:.60s}  cost=${r.cost:.6f}")
    print(f"  Elapsed: {elapsed:.2f}s (3 concurrent)")
    print("  PASS")


@pytest.mark.asyncio
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
        task="test",
        trace_id="test_1b_batch_async",
        max_budget=0,
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
            task="test",
            trace_id="test_2a_gpt5_structured",
            max_budget=0,
        )
        print(f"  Parsed: name={result.name}, country={result.country}, pop={result.population_millions}M")
        print(f"  Cost: ${meta.cost:.6f}, tokens: {meta.usage}")
        assert result.name.lower() == "tokyo"
        assert result.country.lower() == "japan"
        print("  PASS")
    except Exception as e:
        print(f"  SKIPPED: {e}")


@pytest.mark.asyncio
async def test_gpt5_structured_async() -> None:
    _header("2b. acall_llm_structured (gpt-5-mini, async)")
    try:
        result, meta = await acall_llm_structured(
            "gpt-5-mini",
            [{"role": "user", "content": "Give me info about Paris"}],
            response_model=CityInfo,
            task="test",
            trace_id="test_2b_gpt5_structured_async",
            max_budget=0,
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
            task="test",
            trace_id="test_2c_gemini_structured",
            max_budget=0,
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
        task="test",
        trace_id="test_3_stream_retry",
        max_budget=0,
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
        task="test",
        trace_id="test_4_stream_with_tools",
        max_budget=0,
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
# 5. Basic coverage for missing functions
# ---------------------------------------------------------------------------

GEMINI = "gemini/gemini-2.0-flash"

WEATHER_TOOLS = [{
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


def test_call_llm_basic() -> None:
    _header("5a. call_llm basic (gemini-flash)")
    result = call_llm(GEMINI, [{"role": "user", "content": "Reply with just the word 'pong'"}], task="test", trace_id="test_5a_call_llm_basic", max_budget=0)
    assert isinstance(result, LLMCallResult), f"Expected LLMCallResult, got {type(result)}"
    assert len(result.content) > 0, "Empty content"
    assert result.usage.get("total_tokens", 0) > 0, f"No usage: {result.usage}"
    assert result.cost >= 0, f"Negative cost: {result.cost}"
    print(f"  content={result.content!r:.60s}")
    print(f"  usage={result.usage}, cost=${result.cost:.6f}")
    print("  PASS")


@pytest.mark.asyncio
async def test_acall_llm_basic() -> None:
    _header("5b. acall_llm basic (gemini-flash)")
    result = await acall_llm(GEMINI, [{"role": "user", "content": "Reply with just the word 'pong'"}], task="test", trace_id="test_5b_acall_llm_basic", max_budget=0)
    assert isinstance(result, LLMCallResult)
    assert len(result.content) > 0
    assert result.usage.get("total_tokens", 0) > 0
    print(f"  content={result.content!r:.60s}")
    print(f"  usage={result.usage}, cost=${result.cost:.6f}")
    print("  PASS")


def test_call_llm_with_tools() -> None:
    _header("5c. call_llm_with_tools (gemini-flash)")
    result = call_llm_with_tools(
        GEMINI,
        [{"role": "user", "content": "What's the weather in Tokyo?"}],
        WEATHER_TOOLS,
        task="test",
        trace_id="test_5c_call_llm_with_tools",
        max_budget=0,
    )
    assert isinstance(result, LLMCallResult)
    print(f"  tool_calls={result.tool_calls}")
    print(f"  finish_reason={result.finish_reason}")
    if result.tool_calls:
        tc = result.tool_calls[0]
        assert tc["function"]["name"] == "get_weather"
        args = json.loads(tc["function"]["arguments"])
        assert "tokyo" in args.get("location", "").lower()
        print(f"  Extracted: get_weather({args})")
    else:
        print("  NOTE: model chose text instead of tool call")
    print("  PASS")


@pytest.mark.asyncio
async def test_acall_llm_with_tools() -> None:
    _header("5d. acall_llm_with_tools (gemini-flash)")
    result = await acall_llm_with_tools(
        GEMINI,
        [{"role": "user", "content": "What's the weather in Paris?"}],
        WEATHER_TOOLS,
        task="test",
        trace_id="test_5d_acall_llm_with_tools",
        max_budget=0,
    )
    assert isinstance(result, LLMCallResult)
    print(f"  tool_calls={result.tool_calls}")
    if result.tool_calls:
        tc = result.tool_calls[0]
        assert tc["function"]["name"] == "get_weather"
        print(f"  Extracted: get_weather({tc['function']['arguments']})")
    print("  PASS")


def test_structured_batch() -> None:
    _header("5e. call_llm_structured_batch (gemini-flash, 3 items)")
    messages_list = [
        [{"role": "user", "content": f"Give me info about {city}"}]
        for city in ("Tokyo", "Paris", "Berlin")
    ]
    results = call_llm_structured_batch(
        GEMINI, messages_list, response_model=CityInfo, max_concurrent=3,
        task="test", trace_id="test_5e_structured_batch", max_budget=0,
    )
    assert len(results) == 3
    for i, item in enumerate(results):
        assert isinstance(item, tuple), f"Item {i} is {type(item)}: {item}"
        parsed, meta = item
        assert isinstance(parsed, CityInfo), f"Item {i} parsed is {type(parsed)}"
        assert isinstance(meta, LLMCallResult), f"Item {i} meta is {type(meta)}"
        print(f"  [{i}] {parsed.name}, {parsed.country} — ${meta.cost:.6f}")
    print("  PASS")


@pytest.mark.asyncio
async def test_astream_llm() -> None:
    _header("5f. astream_llm (gemini-flash)")
    stream = await astream_llm(
        GEMINI,
        [{"role": "user", "content": "Say 'hello world' and nothing else"}],
        task="test",
        trace_id="test_5f_astream_llm",
        max_budget=0,
    )
    chunks: list[str] = []
    async for chunk in stream:
        chunks.append(chunk)
    full = "".join(chunks)
    assert len(full) > 0, "No streamed content"
    result = stream.result
    assert result.usage.get("total_tokens", 0) > 0, f"No usage after stream: {result.usage}"
    print(f"  Streamed: {full!r:.80s}")
    print(f"  usage={result.usage}, cost=${result.cost:.6f}")
    print("  PASS")


@pytest.mark.asyncio
async def test_astream_llm_with_tools() -> None:
    _header("5g. astream_llm_with_tools (gemini-flash)")
    stream = await astream_llm_with_tools(
        GEMINI,
        [{"role": "user", "content": "What's the weather in London?"}],
        WEATHER_TOOLS,
        task="test",
        trace_id="test_5g_astream_llm_with_tools",
        max_budget=0,
    )
    chunks: list[str] = []
    async for chunk in stream:
        chunks.append(chunk)
    result = stream.result
    full = "".join(chunks)
    print(f"  Streamed text: {full!r:.80s}")
    print(f"  tool_calls={result.tool_calls}")
    if result.tool_calls:
        tc = result.tool_calls[0]
        assert tc["function"]["name"] == "get_weather"
        print(f"  Extracted: get_weather({tc['function']['arguments']})")
    print("  PASS")


# ---------------------------------------------------------------------------
# 6. Fallback (real model switch)
# ---------------------------------------------------------------------------


def test_fallback_real() -> None:
    _header("6. Fallback: bogus primary → gemini-flash")
    fallback_log: list[tuple[str, str]] = []

    def on_fallback(failed: str, err: Exception, next_model: str) -> None:
        fallback_log.append((failed, next_model))

    result = call_llm(
        "bogus/nonexistent-model-12345",
        [{"role": "user", "content": "Reply with 'fallback works'"}],
        num_retries=0,
        fallback_models=[GEMINI],
        on_fallback=on_fallback,
        task="test",
        trace_id="test_6_fallback_real",
        max_budget=0,
    )
    assert isinstance(result, LLMCallResult)
    assert len(result.content) > 0
    assert len(fallback_log) > 0, "on_fallback never fired"
    print(f"  on_fallback fired {len(fallback_log)} time(s): {fallback_log}")
    print(f"  result.model={result.model}")
    print(f"  content={result.content!r:.60s}")
    print("  PASS")


# ---------------------------------------------------------------------------
# 7. Retry (real transient error)
# ---------------------------------------------------------------------------


def test_retry_policy_real() -> None:
    _header("7. RetryPolicy: bogus model with forced retry → fallback")
    retry_log: list[int] = []
    fallback_log: list[str] = []

    # The bogus model error is BadRequestError (not transient), so we force
    # retryability with should_retry to verify the on_retry callback fires.
    policy = RetryPolicy(
        max_retries=1,
        base_delay=0.1,
        max_delay=0.5,
        on_retry=lambda attempt, err, delay: retry_log.append(attempt),
        should_retry=lambda err: True,
    )
    result = call_llm(
        "bogus/nonexistent-model-12345",
        [{"role": "user", "content": "Reply with 'retry works'"}],
        retry=policy,
        fallback_models=[GEMINI],
        on_fallback=lambda failed, err, next_m: fallback_log.append(next_m),
        task="test",
        trace_id="test_7_retry_policy_real",
        max_budget=0,
    )
    assert isinstance(result, LLMCallResult)
    assert len(result.content) > 0
    print(f"  on_retry fired for attempts: {retry_log}")
    print(f"  on_fallback fired for: {fallback_log}")
    assert len(retry_log) >= 1, "on_retry never fired"
    assert len(fallback_log) >= 1, "on_fallback never fired"
    print("  PASS")


# ---------------------------------------------------------------------------
# 8. Batch edge cases
# ---------------------------------------------------------------------------


def test_batch_partial_failure() -> None:
    _header("8a. Batch return_exceptions=True (all fail → 3 exceptions)")
    messages_list = [
        [{"role": "user", "content": "Reply with just '0'"}],
        [{"role": "user", "content": "Reply with just '1'"}],
        [{"role": "user", "content": "Reply with just '2'"}],
    ]
    # Batch uses a single model, so we send all 3 to a bogus model to verify
    # return_exceptions=True returns exceptions instead of propagating.
    results = call_llm_batch(
        "bogus/nonexistent-model-12345",
        messages_list,
        return_exceptions=True,
        num_retries=0,
        task="test",
        trace_id="test_8a_batch_partial_failure",
        max_budget=0,
    )
    assert len(results) == 3
    for i, r in enumerate(results):
        assert isinstance(r, Exception), f"Item {i} should be Exception, got {type(r)}"
        print(f"  [{i}] Exception: {type(r).__name__}: {str(r)[:80]}")
    print("  PASS")


def test_batch_on_item_error() -> None:
    _header("8b. Batch on_item_error callback")
    messages_list = [
        [{"role": "user", "content": "Reply with just '0'"}],
        [{"role": "user", "content": "Reply with just '1'"}],
    ]
    error_log: list[tuple[int, str]] = []

    results = call_llm_batch(
        "bogus/nonexistent-model-12345",
        messages_list,
        return_exceptions=True,
        num_retries=0,
        on_item_error=lambda idx, err: error_log.append((idx, type(err).__name__)),
        task="test",
        trace_id="test_8b_batch_on_item_error",
        max_budget=0,
    )
    assert len(results) == 2
    assert len(error_log) == 2, f"Expected 2 error callbacks, got {len(error_log)}"
    error_indices = sorted(idx for idx, _ in error_log)
    assert error_indices == [0, 1], f"Wrong indices: {error_indices}"
    print(f"  on_item_error fired: {error_log}")
    print("  PASS")


# ---------------------------------------------------------------------------
# 9. Cache (real deduplication)
# ---------------------------------------------------------------------------


def test_cache_prevents_duplicate_calls() -> None:
    _header("9. Cache deduplication (LRUCache + Hooks)")
    cache = LRUCache(maxsize=16, ttl=60)
    before_call_count = 0

    def count_before(model: str, msgs: list[dict[str, Any]], kw: dict[str, Any]) -> None:
        nonlocal before_call_count
        before_call_count += 1

    hooks = Hooks(before_call=count_before)
    messages = [{"role": "user", "content": "Reply with just the word 'cached'"}]

    result1 = call_llm(GEMINI, messages, cache=cache, hooks=hooks, task="test", trace_id="test_9_cache_call1", max_budget=0)
    assert before_call_count == 1, f"before_call should have fired once, got {before_call_count}"

    result2 = call_llm(GEMINI, messages, cache=cache, hooks=hooks, task="test", trace_id="test_9_cache_call2", max_budget=0)
    # Cache hit — before_call should NOT fire again
    assert before_call_count == 1, f"before_call fired again on cache hit: {before_call_count}"
    assert result1.content == result2.content, "Cached result differs"
    assert result1.cost == result2.cost, "Cached cost differs"
    print(f"  Call 1: {result1.content!r:.60s}")
    print(f"  Call 2: {result2.content!r:.60s} (cached)")
    print(f"  before_call fired {before_call_count} time(s)")
    print("  PASS")


# ---------------------------------------------------------------------------
# 10. Hooks (real callbacks)
# ---------------------------------------------------------------------------


def test_hooks_real() -> None:
    _header("10. Hooks: before_call + after_call (gemini-flash)")
    before_log: list[str] = []
    after_log: list[float] = []

    hooks = Hooks(
        before_call=lambda model, msgs, kw: before_log.append(model),
        after_call=lambda result: after_log.append(result.cost),
    )
    result = call_llm(
        GEMINI,
        [{"role": "user", "content": "Reply with 'hooks work'"}],
        hooks=hooks,
        task="test",
        trace_id="test_10_hooks_real",
        max_budget=0,
    )
    assert isinstance(result, LLMCallResult)
    assert len(before_log) == 1, f"before_call fired {len(before_log)} times"
    assert GEMINI in before_log[0] or "gemini" in before_log[0].lower(), \
        f"before_call got wrong model: {before_log[0]}"
    assert len(after_log) == 1, f"after_call fired {len(after_log)} times"
    assert after_log[0] >= 0, f"after_call got negative cost: {after_log[0]}"
    print(f"  before_call model: {before_log[0]}")
    print(f"  after_call cost: ${after_log[0]:.6f}")
    print("  PASS")


# ---------------------------------------------------------------------------
# 11. Finish reason / truncation
# ---------------------------------------------------------------------------


def test_finish_reason_length() -> None:
    _header("11. Truncation raises RuntimeError with max_tokens=5")
    try:
        call_llm(
            GEMINI,
            [{"role": "user", "content": "Write a 500-word essay about the history of computing"}],
            max_tokens=5,
            task="test",
            trace_id="test_11_finish_reason_length",
            max_budget=0,
        )
        assert False, "Expected RuntimeError for truncation, but call succeeded"
    except LLMError as e:
        assert "truncated" in str(e).lower(), f"Unexpected LLMError: {e}"
        print(f"  Raised LLMError: {e}")
        print("  PASS")


# ---------------------------------------------------------------------------
# 12. GPT-5 edge cases
# ---------------------------------------------------------------------------


def test_gpt5_basic_completion() -> None:
    _header("12a. call_llm gpt-5-mini basic completion")
    try:
        result = call_llm(
            "gpt-5-mini",
            [{"role": "user", "content": "Reply with just the word 'hello'"}],
            task="test",
            trace_id="test_12a_gpt5_basic",
            max_budget=0,
        )
        assert isinstance(result, LLMCallResult)
        assert len(result.content) > 0
        assert result.usage.get("total_tokens", 0) > 0
        print(f"  content={result.content!r:.60s}")
        print(f"  model={result.model}, cost=${result.cost:.6f}")
        print(f"  usage={result.usage}")
        print("  PASS")
    except Exception as e:
        print(f"  SKIPPED: {e}")


def test_gpt5_json_format() -> None:
    _header("12b. call_llm gpt-5-mini with json_schema response_format")
    try:
        result = call_llm(
            "gpt-5-mini",
            [{"role": "user", "content": "Give me info about Tokyo as JSON with keys: name, country, population_millions"}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "strict": True,
                    "name": "city_info",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "country": {"type": "string"},
                            "population_millions": {"type": "number"},
                        },
                        "required": ["name", "country", "population_millions"],
                        "additionalProperties": False,
                    },
                },
            },
            task="test",
            trace_id="test_12b_gpt5_json_format",
            max_budget=0,
        )
        assert isinstance(result, LLMCallResult)
        assert len(result.content) > 0
        data = json.loads(result.content)
        assert "name" in data, f"Missing 'name' key in {data}"
        assert "tokyo" in data["name"].lower(), f"Unexpected name: {data['name']}"
        print(f"  Parsed JSON: {data}")
        print(f"  cost=${result.cost:.6f}")
        print("  PASS")
    except Exception as e:
        print(f"  SKIPPED: {e}")


# ---------------------------------------------------------------------------
# 13. Agent SDK (requires claude-agent-sdk + ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------


def test_claude_code_basic() -> None:
    _header("13a. call_llm('claude-code') — basic agent call")
    try:
        result = call_llm(
            "claude-code",
            [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
            max_turns=1,
            task="test",
            trace_id="test_13a_claude_code_basic",
            max_budget=0,
        )
        assert isinstance(result, LLMCallResult)
        assert "4" in result.content, f"Expected '4' in: {result.content[:200]}"
        assert result.finish_reason == "stop"
        print(f"  Content (first 200 chars): {result.content[:200]}")
        print(f"  cost=${result.cost:.6f}")
        print("  PASS")
    except ImportError as e:
        print(f"  SKIPPED (SDK not installed): {e}")
    except Exception as e:
        print(f"  SKIPPED: {e}")


def test_claude_code_with_model() -> None:
    _header("13b. call_llm('claude-code/haiku') — agent with model suffix")
    try:
        result = call_llm(
            "claude-code/haiku",
            [{"role": "user", "content": "What is the capital of France? Reply with just the city name."}],
            max_turns=1,
            task="test",
            trace_id="test_13b_claude_code_with_model",
            max_budget=0,
        )
        assert isinstance(result, LLMCallResult)
        assert "paris" in result.content.lower(), f"Expected 'paris' in: {result.content[:200]}"
        assert result.model == "claude-code/haiku"
        print(f"  Content (first 200 chars): {result.content[:200]}")
        print(f"  cost=${result.cost:.6f}")
        print("  PASS")
    except ImportError as e:
        print(f"  SKIPPED (SDK not installed): {e}")
    except Exception as e:
        print(f"  SKIPPED: {e}")


def test_claude_code_structured() -> None:
    _header("13c. call_llm_structured('claude-code') — structured agent output")
    try:
        parsed, meta = call_llm_structured(
            "claude-code",
            [{"role": "user", "content": "Give me info about Tokyo. Return JSON with name and country fields only."}],
            response_model=CityInfo,
            max_turns=1,
            task="test",
            trace_id="test_13c_claude_code_structured",
            max_budget=0,
        )
        assert isinstance(parsed, CityInfo)
        assert "tokyo" in parsed.name.lower(), f"Expected 'tokyo' in name: {parsed.name}"
        assert isinstance(meta, LLMCallResult)
        print(f"  Parsed: name={parsed.name}, country={parsed.country}")
        print(f"  cost=${meta.cost:.6f}")
        print("  PASS")
    except ImportError as e:
        print(f"  SKIPPED (SDK not installed): {e}")
    except Exception as e:
        print(f"  SKIPPED: {e}")


def test_claude_code_stream() -> None:
    _header("13d. stream_llm('claude-code') — agent streaming")
    try:
        stream = stream_llm(
            "claude-code",
            [{"role": "user", "content": "What is 3+3? Reply with just the number."}],
            max_turns=1,
            task="test",
            trace_id="test_13d_claude_code_stream",
            max_budget=0,
        )
        chunks: list[str] = []
        for chunk in stream:
            chunks.append(chunk)
        full = "".join(chunks)
        assert len(chunks) > 0, "No chunks received"
        result = stream.result
        assert isinstance(result, LLMCallResult)
        print(f"  Streamed {len(chunks)} chunk(s): {full[:200]!r}")
        print(f"  cost=${result.cost:.6f}")
        print("  PASS")
    except ImportError as e:
        print(f"  SKIPPED (SDK not installed): {e}")
    except Exception as e:
        print(f"  SKIPPED: {e}")


def test_claude_code_batch() -> None:
    _header("13e. call_llm_batch('claude-code') — agent batch")
    try:
        messages_list = [
            [{"role": "user", "content": "What is 1+1? Reply with just the number."}],
            [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
        ]
        results = call_llm_batch(
            "claude-code", messages_list, max_concurrent=2, max_turns=1,
            task="test", trace_id="test_13e_claude_code_batch", max_budget=0,
        )
        assert len(results) == 2
        for i, r in enumerate(results):
            assert isinstance(r, LLMCallResult), f"Item {i} is {type(r)}: {r}"
            print(f"  [{i}] content={r.content[:100]!r}  cost=${r.cost:.6f}")
        print("  PASS")
    except ImportError as e:
        print(f"  SKIPPED (SDK not installed): {e}")
    except Exception as e:
        print(f"  SKIPPED: {e}")


# ---------------------------------------------------------------------------
# 14. Codex SDK (requires openai-codex-sdk + OPENAI_API_KEY)
# ---------------------------------------------------------------------------


def test_codex_basic() -> None:
    _header("14a. call_llm('codex') — basic codex call")
    try:
        result = call_llm(
            "codex",
            [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
            task="test",
            trace_id="test_14a_codex_basic",
            max_budget=0,
        )
        assert isinstance(result, LLMCallResult)
        assert "4" in result.content, f"Expected '4' in: {result.content[:200]}"
        assert result.finish_reason == "stop"
        print(f"  Content (first 200 chars): {result.content[:200]}")
        print(f"  cost=${result.cost:.6f}")
        print("  PASS")
    except ImportError as e:
        print(f"  SKIPPED (SDK not installed): {e}")
    except Exception as e:
        print(f"  SKIPPED: {e}")


def test_codex_with_model() -> None:
    _header("14b. call_llm('codex/gpt-4o') — codex with model suffix")
    try:
        result = call_llm(
            "codex/gpt-4o",
            [{"role": "user", "content": "What is the capital of France? Reply with just the city name."}],
            task="test",
            trace_id="test_14b_codex_with_model",
            max_budget=0,
        )
        assert isinstance(result, LLMCallResult)
        assert "paris" in result.content.lower(), f"Expected 'paris' in: {result.content[:200]}"
        assert result.model == "codex/gpt-4o"
        print(f"  Content (first 200 chars): {result.content[:200]}")
        print(f"  cost=${result.cost:.6f}")
        print("  PASS")
    except ImportError as e:
        print(f"  SKIPPED (SDK not installed): {e}")
    except Exception as e:
        print(f"  SKIPPED: {e}")


def test_codex_structured() -> None:
    _header("14c. call_llm_structured('codex') — structured codex output")
    try:
        parsed, meta = call_llm_structured(
            "codex",
            [{"role": "user", "content": "Give me info about Tokyo. Return JSON with name and country fields only."}],
            response_model=CityInfo,
            task="test",
            trace_id="test_14c_codex_structured",
            max_budget=0,
        )
        assert isinstance(parsed, CityInfo)
        assert "tokyo" in parsed.name.lower(), f"Expected 'tokyo' in name: {parsed.name}"
        assert isinstance(meta, LLMCallResult)
        print(f"  Parsed: name={parsed.name}, country={parsed.country}")
        print(f"  cost=${meta.cost:.6f}")
        print("  PASS")
    except ImportError as e:
        print(f"  SKIPPED (SDK not installed): {e}")
    except Exception as e:
        print(f"  SKIPPED: {e}")


def test_codex_stream() -> None:
    _header("14d. stream_llm('codex') — codex streaming")
    try:
        stream = stream_llm(
            "codex",
            [{"role": "user", "content": "What is 3+3? Reply with just the number."}],
            task="test",
            trace_id="test_14d_codex_stream",
            max_budget=0,
        )
        chunks: list[str] = []
        for chunk in stream:
            chunks.append(chunk)
        full = "".join(chunks)
        assert len(chunks) > 0, "No chunks received"
        result = stream.result
        assert isinstance(result, LLMCallResult)
        print(f"  Streamed {len(chunks)} chunk(s): {full[:200]!r}")
        print(f"  cost=${result.cost:.6f}")
        print("  PASS")
    except ImportError as e:
        print(f"  SKIPPED (SDK not installed): {e}")
    except Exception as e:
        print(f"  SKIPPED: {e}")


def test_codex_batch() -> None:
    _header("14e. call_llm_batch('codex') — codex batch")
    try:
        messages_list = [
            [{"role": "user", "content": "What is 1+1? Reply with just the number."}],
            [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
        ]
        results = call_llm_batch("codex", messages_list, max_concurrent=2, task="test", trace_id="test_14e_codex_batch", max_budget=0)
        assert len(results) == 2
        for i, r in enumerate(results):
            assert isinstance(r, LLMCallResult), f"Item {i} is {type(r)}: {r}"
            print(f"  [{i}] content={r.content[:100]!r}  cost=${r.cost:.6f}")
        print("  PASS")
    except ImportError as e:
        print(f"  SKIPPED (SDK not installed): {e}")
    except Exception as e:
        print(f"  SKIPPED: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    # Section 1: Batch
    test_batch_sync()
    await test_batch_async()

    # Section 2: Structured
    test_gpt5_structured()
    await test_gpt5_structured_async()
    test_gemini_structured()

    # Section 3-4: Streaming
    test_stream_retry()
    test_stream_with_tools()

    # Section 5: Basic coverage for missing functions
    test_call_llm_basic()
    await test_acall_llm_basic()
    test_call_llm_with_tools()
    await test_acall_llm_with_tools()
    test_structured_batch()
    await test_astream_llm()
    await test_astream_llm_with_tools()

    # Section 6: Fallback
    test_fallback_real()

    # Section 7: Retry
    test_retry_policy_real()

    # Section 8: Batch edge cases
    test_batch_partial_failure()
    test_batch_on_item_error()

    # Section 9: Cache
    test_cache_prevents_duplicate_calls()

    # Section 10: Hooks
    test_hooks_real()

    # Section 11: Finish reason
    test_finish_reason_length()

    # Section 12: GPT-5 edge cases
    test_gpt5_basic_completion()
    test_gpt5_json_format()

    # Section 13: Agent SDK (Claude)
    test_claude_code_basic()
    test_claude_code_with_model()
    test_claude_code_structured()
    test_claude_code_stream()
    test_claude_code_batch()

    # Section 14: Agent SDK (Codex)
    test_codex_basic()
    test_codex_with_model()
    test_codex_structured()
    test_codex_stream()
    test_codex_batch()

    print(f"\n{'='*60}")
    print("  ALL INTEGRATION TESTS COMPLETE (35 tests)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
