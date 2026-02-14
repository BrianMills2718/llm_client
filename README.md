# llm-client

Thin wrapper around [litellm](https://github.com/BerriAI/litellm). Swap models by changing one string.

## Install

```bash
pip install -e /path/to/llm_client            # basic
pip install -e "/path/to/llm_client[structured]"  # + instructor for Pydantic extraction
```

## Usage

```python
from llm_client import call_llm

# OpenAI
result = call_llm("gpt-4o", [{"role": "user", "content": "Hello"}])

# Anthropic — just change the string
result = call_llm("anthropic/claude-sonnet-4-5-20250929", [{"role": "user", "content": "Hello"}])

# Google
result = call_llm("gemini/gemini-2.0-flash", [{"role": "user", "content": "Hello"}])

# Local Ollama
result = call_llm("ollama/llama3", [{"role": "user", "content": "Hello"}])

print(result.content)  # "Hi there!"
print(result.cost)     # 0.0003
print(result.usage)    # {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
```

### Structured output (Pydantic)

```python
from pydantic import BaseModel
from llm_client import call_llm_structured

class Sentiment(BaseModel):
    label: str
    score: float

sentiment, meta = call_llm_structured(
    "gpt-4o",
    [{"role": "user", "content": "I love this product!"}],
    response_model=Sentiment,
)
print(sentiment.label)  # "positive"
print(sentiment.score)  # 0.95
print(meta.cost)        # 0.0004
```

### Tool calling

```python
from llm_client import call_llm_with_tools

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }
    }
}]

result = call_llm_with_tools("gpt-4o", messages, tools)
if result.tool_calls:
    print(result.tool_calls[0]["function"]["name"])  # "get_weather"
```

### Async

```python
from llm_client import acall_llm, acall_llm_structured, acall_llm_with_tools

result = await acall_llm("gpt-4o", messages)
data, meta = await acall_llm_structured("gpt-4o", messages, response_model=Entity)
result = await acall_llm_with_tools("gpt-4o", messages, tools=[...])
```

### Response caching

```python
from llm_client import LRUCache, call_llm

cache = LRUCache(maxsize=128, ttl=3600)  # thread-safe, 1h TTL
result = call_llm("gpt-4o", messages, cache=cache)  # calls LLM
result = call_llm("gpt-4o", messages, cache=cache)  # returns cached
```

Implement `CachePolicy` protocol for custom backends (Redis, disk, etc.).

### Retry policy

```python
from llm_client import RetryPolicy, call_llm, linear_backoff

# Reusable policy object — overrides individual retry params
policy = RetryPolicy(
    max_retries=5,
    base_delay=0.5,
    backoff=linear_backoff,                     # or exponential_backoff, fixed_backoff
    retry_on=["custom error"],                  # extend built-in patterns
    on_retry=lambda a, err, d: print(f"Retry {a}"),
    should_retry=lambda err: True,              # fully custom retryability check
)
result = call_llm("gpt-4o", messages, retry=policy)

# Or quick one-offs with individual params
result = call_llm("gpt-4o", messages, num_retries=5, retry_on=["custom"])
```

## API

Six functions (3 sync + 3 async), all return `LLMCallResult`:

| Function | Async Variant | Returns | Description |
|----------|---------------|---------|-------------|
| `call_llm(model, messages, **kw)` | `acall_llm(...)` | `LLMCallResult` | Basic completion |
| `call_llm_structured(model, messages, response_model, **kw)` | `acall_llm_structured(...)` | `(T, LLMCallResult)` | Pydantic extraction via instructor |
| `call_llm_with_tools(model, messages, tools, **kw)` | `acall_llm_with_tools(...)` | `LLMCallResult` | Tool/function calling |

`LLMCallResult` fields: `.content`, `.usage`, `.cost`, `.model`, `.tool_calls`, `.finish_reason`, `.raw_response`

All accept: `timeout`, `num_retries`, `reasoning_effort` (Claude only), `api_base`, `retry_on`, `on_retry`, `cache`, `retry` (RetryPolicy), plus any `**kwargs` passed through to `litellm.completion`.

## API keys

Set via environment variables (litellm convention):

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GEMINI_API_KEY=...
```

## Using from another project

```bash
# From your other project's directory:
pip install -e ~/brian_projects/llm_client

# Then in code:
from llm_client import call_llm
```
