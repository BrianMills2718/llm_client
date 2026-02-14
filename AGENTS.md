# LLM Client

Thin wrapper around litellm. Swap any model by changing one string — everything else stays the same.

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

All accept: `timeout` (60s), `num_retries` (2), `reasoning_effort` (Claude only), `api_base` (optional), `retry_on`, `on_retry`, `cache`, plus any litellm kwargs.

### Response Caching

```python
from llm_client import LRUCache, call_llm

cache = LRUCache(maxsize=128)
result = call_llm("gpt-4o", messages, cache=cache)
```

Implement `CachePolicy` protocol for custom backends (Redis, disk, etc.).

### Custom Retry Patterns

```python
result = call_llm("gpt-4o", messages, retry_on=["custom error pattern"])
```

### Retry Callback

```python
result = call_llm("gpt-4o", messages, on_retry=lambda attempt, err, delay: print(f"Retry {attempt}"))
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
