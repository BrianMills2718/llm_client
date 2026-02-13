# LLM Client

Thin wrapper around litellm. Swap any model by changing one string — everything else stays the same.

## Quick Reference

```python
from llm_client import call_llm, call_llm_structured, call_llm_with_tools

# Basic completion (works with any provider)
result = call_llm("gpt-4o", messages)
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

## API

Three functions, all return `LLMCallResult`:

| Function | Purpose |
|----------|---------|
| `call_llm(model, messages, **kwargs)` | Basic chat completion |
| `call_llm_structured(model, messages, response_model, **kwargs)` | Pydantic extraction via instructor |
| `call_llm_with_tools(model, messages, tools, **kwargs)` | Tool/function calling |

`LLMCallResult` fields: `.content`, `.usage`, `.cost`, `.model`, `.tool_calls`

All accept: `timeout` (60s), `num_retries` (2), `reasoning_effort` (Claude only), plus any litellm kwargs.

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
pytest tests/   # 8 tests, all mocked (no real API calls)
```

## Dependencies

- `litellm>=1.40.0` — Multi-provider abstraction
- `pydantic>=2.0` — Data validation
- `instructor>=1.14.0` — Structured output (optional)
