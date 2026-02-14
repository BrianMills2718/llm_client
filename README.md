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

### Batch/parallel calls

```python
from llm_client import call_llm_batch, acall_llm_batch

# Run multiple prompts concurrently (semaphore-based rate limiting)
results = call_llm_batch("gpt-4o", [msgs1, msgs2, msgs3], max_concurrent=5)

# Async version
results = await acall_llm_batch("gpt-4o", messages_list, max_concurrent=10)

# Structured batch
from llm_client import call_llm_structured_batch
results = call_llm_structured_batch("gpt-4o", messages_list, response_model=Entity)
```

### Async

```python
from llm_client import acall_llm, acall_llm_structured, acall_llm_with_tools

result = await acall_llm("gpt-4o", messages)
data, meta = await acall_llm_structured("gpt-4o", messages, response_model=Entity)
result = await acall_llm_with_tools("gpt-4o", messages, tools=[...])
```

### Streaming

```python
from llm_client import stream_llm, astream_llm

# Streaming with retry/fallback support
stream = stream_llm("gpt-4o", messages, num_retries=2, fallback_models=["gpt-3.5-turbo"])
for chunk in stream:
    print(chunk, end="", flush=True)
print(stream.result.usage)  # usage available after stream ends

# Streaming with tools
from llm_client import stream_llm_with_tools
stream = stream_llm_with_tools("gpt-4o", messages, tools=[...])
for chunk in stream:
    print(chunk, end="", flush=True)
print(stream.result.tool_calls)

# Async
stream = await astream_llm("gpt-4o", messages)
async for chunk in stream:
    print(chunk, end="", flush=True)
```

### Fallback models

```python
result = call_llm(
    "gpt-4o", messages,
    fallback_models=["gpt-3.5-turbo", "ollama/llama3"],
    on_fallback=lambda failed, err, next_: print(f"{failed} failed, trying {next_}"),
)
```

### Observability hooks

```python
from llm_client import Hooks, call_llm

hooks = Hooks(
    before_call=lambda model, msgs, kw: print(f"Calling {model}"),
    after_call=lambda result: print(f"${result.cost:.4f}"),
    on_error=lambda err, attempt: print(f"Attempt {attempt} failed"),
)
result = call_llm("gpt-4o", messages, hooks=hooks)
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

Fourteen functions (7 sync + 7 async):

| Function | Async Variant | Returns | Description |
|----------|---------------|---------|-------------|
| `call_llm(model, messages, **kw)` | `acall_llm(...)` | `LLMCallResult` | Basic completion |
| `call_llm_structured(model, messages, response_model, **kw)` | `acall_llm_structured(...)` | `(T, LLMCallResult)` | Pydantic extraction (native JSON schema, Responses API, or instructor) |
| `call_llm_with_tools(model, messages, tools, **kw)` | `acall_llm_with_tools(...)` | `LLMCallResult` | Tool/function calling |
| `call_llm_batch(model, messages_list, **kw)` | `acall_llm_batch(...)` | `list[LLMCallResult]` | Concurrent batch calls |
| `call_llm_structured_batch(model, messages_list, response_model, **kw)` | `acall_llm_structured_batch(...)` | `list[(T, LLMCallResult)]` | Concurrent structured batch |
| `stream_llm(model, messages, **kw)` | `astream_llm(...)` | `LLMStream` | Streaming with retry/fallback |
| `stream_llm_with_tools(model, messages, tools, **kw)` | `astream_llm_with_tools(...)` | `LLMStream` | Streaming with tools |

`LLMCallResult` fields: `.content`, `.usage`, `.cost`, `.model`, `.tool_calls`, `.finish_reason`, `.raw_response`

`call_llm`, `call_llm_structured`, `call_llm_with_tools` (and async variants) accept: `timeout`, `num_retries`, `reasoning_effort` (Claude only), `api_base`, `retry_on`, `on_retry`, `cache`, `retry` (RetryPolicy), `fallback_models`, `on_fallback`, `hooks` (Hooks), plus any `**kwargs` passed through to `litellm.completion`.

`stream_llm` / `astream_llm` (and `*_with_tools` variants) accept: `timeout`, `num_retries`, `reasoning_effort`, `api_base`, `retry`, `fallback_models`, `on_fallback`, `hooks`, plus `**kwargs`. No `cache` param (caching streams doesn't make sense).

`*_batch` functions additionally accept: `max_concurrent` (5), `return_exceptions`, `on_item_complete`, `on_item_error`.

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
pip install -e ~/projects/llm_client

# Then in code:
from llm_client import call_llm
```
