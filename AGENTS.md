# LLM Client

Wrapper around litellm. Swap any model by changing one string — everything else stays the same. Smart retry, fallback models, streaming, observability hooks, response caching, Responses API routing for GPT-5, thinking model detection, and cost tracking built in.

## Quick Reference

```python
from llm_client import call_llm, call_llm_structured, call_llm_with_tools

# Basic completion (works with any provider)
result = call_llm("gpt-4o", messages)
result = call_llm("gpt-5-mini", messages)                          # Auto-routes to Responses API
result = call_llm("anthropic/claude-sonnet-4-5-20250929", messages)
result = call_llm("gemini/gemini-2.5-flash", messages)

# Structured extraction (requires instructor)
from pydantic import BaseModel
class Entity(BaseModel):
    name: str
    type: str
data, meta = call_llm_structured("gpt-4o", messages, response_model=Entity)

# Tool calling
result = call_llm_with_tools("gpt-4o", messages, tools=[...])
result.tool_calls  # normalized across providers

# Batch/parallel calls
from llm_client import call_llm_batch
results = call_llm_batch("gpt-4o", [msgs1, msgs2, msgs3], max_concurrent=5)

# Streaming (with retry/fallback)
from llm_client import stream_llm
stream = stream_llm("gpt-4o", messages, num_retries=2)
for chunk in stream:
    print(chunk, end="")
print(stream.result.usage)

# Streaming with tools
from llm_client import stream_llm_with_tools
stream = stream_llm_with_tools("gpt-4o", messages, tools=[...])
for chunk in stream:
    print(chunk, end="")
print(stream.result.tool_calls)
```

### Async

```python
from llm_client import acall_llm, acall_llm_structured, acall_llm_with_tools, astream_llm, acall_llm_batch

result = await acall_llm("gpt-4o", messages)
data, meta = await acall_llm_structured("gpt-4o", messages, response_model=Entity)
result = await acall_llm_with_tools("gpt-4o", messages, tools=[...])
results = await acall_llm_batch("gpt-4o", messages_list, max_concurrent=10)

stream = await astream_llm("gpt-4o", messages)
async for chunk in stream:
    print(chunk, end="")
```

## API

Fourteen functions (7 sync + 7 async):

| Function | Async Variant | Purpose |
|----------|---------------|---------|
| `call_llm(model, messages, **kwargs)` | `acall_llm(...)` | Basic chat completion |
| `call_llm_structured(model, messages, response_model, **kwargs)` | `acall_llm_structured(...)` | Pydantic extraction (native JSON schema, Responses API, or instructor) |
| `call_llm_with_tools(model, messages, tools, **kwargs)` | `acall_llm_with_tools(...)` | Tool/function calling |
| `call_llm_batch(model, messages_list, **kwargs)` | `acall_llm_batch(...)` | Concurrent batch calls |
| `call_llm_structured_batch(model, messages_list, response_model, **kwargs)` | `acall_llm_structured_batch(...)` | Concurrent structured batch |
| `stream_llm(model, messages, **kwargs)` | `astream_llm(...)` | Streaming with retry/fallback |
| `stream_llm_with_tools(model, messages, tools, **kwargs)` | `astream_llm_with_tools(...)` | Streaming with tools |

`LLMCallResult` fields: `.content`, `.usage`, `.cost`, `.model`, `.tool_calls`, `.finish_reason`, `.raw_response`

`call_llm`, `call_llm_structured`, `call_llm_with_tools` (and async variants) accept: `timeout` (60s), `num_retries` (2), `reasoning_effort` (Claude only), `api_base` (optional), `retry_on`, `on_retry`, `cache`, `retry` (RetryPolicy), `fallback_models`, `on_fallback`, `hooks` (Hooks), plus any litellm kwargs.

`stream_llm` / `astream_llm` (and `*_with_tools` variants) accept: `timeout`, `num_retries`, `reasoning_effort`, `api_base`, `retry`, `fallback_models`, `on_fallback`, `hooks`, plus litellm kwargs.

`*_batch` functions additionally accept: `max_concurrent` (5), `return_exceptions`, `on_item_complete`, `on_item_error`.

### Response Caching

```python
from llm_client import LRUCache, call_llm

cache = LRUCache(maxsize=128, ttl=3600)  # thread-safe, 1h TTL
result = call_llm("gpt-4o", messages, cache=cache)
```

Implement `CachePolicy` protocol for custom backends (Redis, disk, etc.). Async functions also accept `AsyncCachePolicy` for non-blocking cache access.

### RetryPolicy

```python
from llm_client import RetryPolicy, linear_backoff

policy = RetryPolicy(
    max_retries=5, backoff=linear_backoff,
    retry_on=["custom error"], on_retry=lambda a, e, d: ...,
    should_retry=lambda e: True,  # fully custom retryability
)
result = call_llm("gpt-4o", messages, retry=policy)
```

Backoff strategies: `exponential_backoff` (default), `linear_backoff`, `fixed_backoff`, or any `(attempt, base_delay, max_delay) -> delay` callable.

### Fallback Models

```python
result = call_llm("gpt-4o", messages,
    fallback_models=["gpt-3.5-turbo", "ollama/llama3"],
    on_fallback=lambda failed, err, next_: print(f"Trying {next_}"),
)
```

### Observability Hooks

```python
from llm_client import Hooks

hooks = Hooks(
    before_call=lambda model, msgs, kw: ...,
    after_call=lambda result: ...,
    on_error=lambda err, attempt: ...,
)
result = call_llm("gpt-4o", messages, hooks=hooks)
```

## Installation

```bash
pip install -e .                    # Basic
pip install -e ".[structured]"      # With instructor for structured output
```

## Environment

API keys via env vars (litellm convention):
```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GEMINI_API_KEY=...
```

## Tests

```bash
pytest tests/ -v   # All mocked (no real API calls)
```

## Dependencies

- `litellm>=1.81.3` — Multi-provider abstraction
- `pydantic>=2.0` — Data validation
- `instructor>=1.14.0` — Structured output (optional)
- `pytest-asyncio>=0.23` — Async test support (dev only)
