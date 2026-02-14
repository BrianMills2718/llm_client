# LLM Client

Wrapper around litellm. Swap any model by changing one string — everything else stays the same. Smart retry, automatic Responses API routing for GPT-5, thinking model detection, and cost tracking built in.

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
```

### Async

```python
from llm_client import acall_llm, acall_llm_structured, acall_llm_with_tools

# Same signatures, just async
result = await acall_llm("gpt-4o", messages)
data, meta = await acall_llm_structured("gpt-4o", messages, response_model=Entity)
result = await acall_llm_with_tools("gpt-4o", messages, tools=[...])
```

## API

Six functions (3 sync + 3 async), all return `LLMCallResult`:

| Function | Async Variant | Purpose |
|----------|---------------|---------|
| `call_llm(model, messages, **kwargs)` | `acall_llm(...)` | Basic chat completion |
| `call_llm_structured(model, messages, response_model, **kwargs)` | `acall_llm_structured(...)` | Pydantic extraction via instructor |
| `call_llm_with_tools(model, messages, tools, **kwargs)` | `acall_llm_with_tools(...)` | Tool/function calling |

`LLMCallResult` fields: `.content`, `.usage`, `.cost`, `.model`, `.tool_calls`, `.finish_reason`, `.raw_response`

All accept: `timeout` (60s), `num_retries` (2), `reasoning_effort` (Claude only), `api_base` (optional), `retry_on`, `on_retry`, `cache`, `retry` (RetryPolicy), plus any litellm kwargs.

### RetryPolicy (reusable retry configuration)

Create once, reuse across calls. Overrides individual retry params when provided:

```python
from llm_client import RetryPolicy, call_llm, linear_backoff

policy = RetryPolicy(
    max_retries=5,
    base_delay=0.5,
    max_delay=10.0,
    retry_on=["custom error"],                   # extend built-in patterns
    on_retry=lambda attempt, err, delay: ...,     # callback per retry
    backoff=linear_backoff,                       # or exponential_backoff, fixed_backoff, custom fn
    should_retry=lambda err: "fatal" not in str(err),  # fully custom retryability
)

result = call_llm("gpt-4o", messages, retry=policy)
result = call_llm("gpt-4o", messages2, retry=policy)  # same policy reused
```

Individual params still work for quick one-offs:
```python
result = call_llm("gpt-4o", messages, num_retries=5, retry_on=["custom"])
```

### Backoff strategies

Three built-in, or pass any `(attempt, base_delay, max_delay) -> delay` callable:
- `exponential_backoff` — default, `base * 2^attempt` with jitter
- `linear_backoff` — `base * (attempt+1)` with jitter
- `fixed_backoff` — constant `base_delay`, no escalation

### Response Caching (`cache`)

Thread-safe, TTL-capable LRU cache built in. Implement `CachePolicy` protocol for custom backends:

```python
from llm_client import LRUCache, call_llm

cache = LRUCache(maxsize=128, ttl=3600)  # 1 hour TTL, thread-safe
result = call_llm("gpt-4o", messages, cache=cache)  # calls LLM
result = call_llm("gpt-4o", messages, cache=cache)  # returns cached
cache.clear()  # manual invalidation
```

Custom backend (Redis, disk, etc.):
```python
from llm_client import CachePolicy, LLMCallResult

class RedisCache:
    def get(self, key: str) -> LLMCallResult | None: ...
    def set(self, key: str, value: LLMCallResult) -> None: ...
```

### Truncation Detection

Check `finish_reason` to detect truncated responses:
```python
result = call_llm("gpt-4o", messages)
if result.finish_reason == "length":
    # Response was truncated — retry with higher max_tokens or split the task
    ...
```

Common values: `"stop"` (normal), `"length"` (truncated), `"tool_calls"` (model invoked tools), `"content_filter"` (blocked).

### Raw Response Access

Use `raw_response` for provider-specific data (e.g., Claude thinking blocks):
```python
result = call_llm("anthropic/claude-sonnet-4-5-20250929", messages)
result.raw_response  # full litellm response object
```

## Smart Retry

All functions retry on transient failures with jittered exponential backoff (capped at 30s). Retryable errors include:

- **Transport**: rate limits, timeouts, connection resets, 500/502/503 errors
- **Application**: empty responses, JSON parse errors, malformed JSON

Non-retryable errors (e.g., invalid API key, content filter) raise immediately. `num_retries` controls the count (default: 2). Empty responses are automatically retried unless the model made tool calls.

## GPT-5 / Responses API

GPT-5 models use OpenAI's Responses API instead of Chat Completions. llm_client detects this automatically and routes through `litellm.responses()` / `litellm.aresponses()`. No code changes needed — just pass the model string:

```python
# These all work identically from the caller's perspective
result = call_llm("gpt-5-mini", messages)
result = call_llm("gpt-5", messages)
result = await acall_llm("gpt-5-mini", messages)
```

What happens automatically for GPT-5 models:
- Messages converted to Responses API input format
- `response_format` converted to Responses API `text` parameter
- `max_tokens` / `max_output_tokens` stripped (GPT-5 uses reasoning tokens before output — setting limits can exhaust them on reasoning and return empty output)
- Response normalized into the same `LLMCallResult` you get from any other model
- Smart retry still works
- `response_format` with `json_schema` works for structured JSON output

```python
# Structured JSON with GPT-5 (use response_format, not call_llm_structured)
result = call_llm("gpt-5-mini", messages, response_format={
    "type": "json_schema",
    "json_schema": {
        "strict": True,
        "name": "my_schema",
        "schema": {"type": "object", "properties": {"key": {"type": "string"}}},
    }
})
data = json.loads(result.content)
```

## Thinking Model Detection

Gemini 3/4 thinking models allocate reasoning tokens by default, consuming output budget. llm_client automatically injects `thinking: {type: "enabled", budget_tokens: 0}` for these models so all tokens go to the actual response.

Override with your own config if you want thinking tokens:
```python
result = call_llm("gemini/gemini-3-flash", messages, thinking={"type": "enabled", "budget_tokens": 1000})
```

## Fence Stripping

`strip_fences()` removes markdown code fences from LLM output. Useful when calling `call_llm()` and parsing JSON manually:

```python
from llm_client import call_llm, strip_fences
import json

result = call_llm("gpt-4o", messages)
clean = strip_fences(result.content)  # removes ```json ... ``` wrapping
data = json.loads(clean)
```

Not needed with `call_llm_structured` — instructor handles JSON parsing automatically.

## Caching (litellm built-in)

litellm also has its own built-in caching (separate from the `cache` parameter above). Set up globally, then pass `caching=True` per-call:

```python
import litellm
from litellm.caching.caching import Cache

litellm.cache = Cache(type="local")  # or "redis", "s3", "disk"

result = call_llm("gpt-4o", messages, caching=True)
```

## OpenRouter

OpenRouter works two ways:

**1. Model prefix (recommended)** — uses the `openrouter/` prefix, requires `OPENROUTER_API_KEY` env var:
```python
result = call_llm("openrouter/anthropic/claude-sonnet-4-5-20250929", messages)
result = await acall_llm("openrouter/google/gemini-2.5-flash", messages)
```

**2. Explicit api_base** — pass the base URL directly:
```python
result = call_llm(
    "anthropic/claude-sonnet-4-5-20250929",
    messages,
    api_base="https://openrouter.ai/api/v1",
)
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
export OPENROUTER_API_KEY=sk-or-...
```

## Tests

```bash
pytest tests/ -v   # All mocked (no real API calls)
```

## Dependencies

- `litellm>=1.81.3` — Multi-provider abstraction (bumped from 1.40.0 to fix Gemini nullable type bug)
- `pydantic>=2.0` — Data validation
- `instructor>=1.14.0` — Structured output (optional)
- `pytest-asyncio>=0.23` — Async test support (dev only)
