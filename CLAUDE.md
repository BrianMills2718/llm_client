# LLM Client

Wrapper around litellm. Swap any model by changing one string — everything else stays the same. Smart retry, fallback models, streaming, observability hooks, response caching, Responses API routing for GPT-5, thinking model detection, and cost tracking built in.

## Quick Reference

```python
from llm_client import call_llm, call_llm_structured, call_llm_with_tools

# Basic completion (works with any provider)
result = call_llm("gpt-5-mini", messages)
result = call_llm("gpt-5-mini", messages)                          # Auto-routes to Responses API
result = call_llm("anthropic/claude-sonnet-4-5-20250929", messages)
result = call_llm("gemini/gemini-2.5-flash", messages)

# Structured extraction (native JSON schema or instructor fallback)
from pydantic import BaseModel
class Entity(BaseModel):
    name: str
    type: str
data, meta = call_llm_structured("gpt-5-mini", messages, response_model=Entity)

# Tool calling
result = call_llm_with_tools("gpt-5-mini", messages, tools=[...])
result.tool_calls  # normalized across providers
```

### Batch/Parallel Calls

```python
from llm_client import call_llm_batch, acall_llm_batch

# Run multiple prompts concurrently with rate limiting
messages_list = [
    [{"role": "user", "content": "Summarize doc 1"}],
    [{"role": "user", "content": "Summarize doc 2"}],
    [{"role": "user", "content": "Summarize doc 3"}],
]
results = call_llm_batch("gpt-5-mini", messages_list, max_concurrent=5)

# Async version (core implementation — call_llm_batch wraps this)
results = await acall_llm_batch("gpt-5-mini", messages_list, max_concurrent=10)

# With progress callbacks
results = await acall_llm_batch(
    "gpt-5-mini", messages_list,
    on_item_complete=lambda idx, res: print(f"Done {idx}: {len(res.content)} chars"),
    on_item_error=lambda idx, err: print(f"Failed {idx}: {err}"),
    return_exceptions=True,  # exceptions in results instead of propagating
)

# Structured batch
from llm_client import call_llm_structured_batch
results = call_llm_structured_batch("gpt-5-mini", messages_list, response_model=Entity)
for parsed, meta in results:
    print(parsed.name, meta.cost)
```

Each item gets full retry/fallback/cache/hooks via `acall_llm` or `acall_llm_structured`. Semaphore-based concurrency control.

### Async

```python
from llm_client import acall_llm, acall_llm_structured, acall_llm_with_tools

# Same signatures, just async
result = await acall_llm("gpt-5-mini", messages)
data, meta = await acall_llm_structured("gpt-5-mini", messages, response_model=Entity)
result = await acall_llm_with_tools("gpt-5-mini", messages, tools=[...])
```

### Streaming

```python
from llm_client import stream_llm, astream_llm

# Sync streaming (with retry/fallback support)
stream = stream_llm("gpt-5-mini", messages, num_retries=2, fallback_models=["deepseek/deepseek-chat"])
for chunk in stream:
    print(chunk, end="", flush=True)
print()
print(stream.result.usage)  # usage/cost available after stream ends

# Async streaming
stream = await astream_llm("gpt-5-mini", messages)
async for chunk in stream:
    print(chunk, end="", flush=True)
print(stream.result.cost)

# Streaming with tools
from llm_client import stream_llm_with_tools
stream = stream_llm_with_tools("gpt-5-mini", messages, tools=[...])
for chunk in stream:
    print(chunk, end="", flush=True)
print(stream.result.tool_calls)  # tool calls available after stream ends
```

Streaming retries on **pre-stream** errors (rate limits, connection errors) and wraps final failures in `LLMError` subclasses. Mid-stream errors are not retried (would require buffering, defeating streaming's purpose).

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

Also exported: `LLMError`, `LLMRateLimitError`, `LLMQuotaExhaustedError`, `LLMAuthError`, `LLMContentFilterError`, `LLMTransientError`, `LLMModelNotFoundError`, `classify_error`, `wrap_error` (see [Structured Error Types](#structured-error-types)).

`LLMCallResult` fields: `.content`, `.usage`, `.cost`, `.model`, `.tool_calls`, `.finish_reason`, `.raw_response`

`call_llm`, `call_llm_structured`, `call_llm_with_tools` (and async variants) accept: `timeout` (60s), `num_retries` (2), `reasoning_effort` (Claude only), `api_base` (optional), `retry_on`, `on_retry`, `cache`, `retry` (RetryPolicy), `fallback_models`, `on_fallback`, `hooks` (Hooks), plus any litellm kwargs.

`stream_llm` / `astream_llm` (and `*_with_tools` variants) accept: `timeout`, `num_retries`, `reasoning_effort`, `api_base`, `retry` (RetryPolicy), `fallback_models`, `on_fallback`, `hooks`, plus litellm kwargs. No `cache` param (caching streams doesn't make sense).

`*_batch` functions additionally accept: `max_concurrent` (5), `return_exceptions` (False), `on_item_complete`, `on_item_error`.

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

result = call_llm("gpt-5-mini", messages, retry=policy)
result = call_llm("gpt-5-mini", messages2, retry=policy)  # same policy reused
```

Individual params still work for quick one-offs:
```python
result = call_llm("gpt-5-mini", messages, num_retries=5, retry_on=["custom"])
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
result = call_llm("gpt-5-mini", messages, cache=cache)  # calls LLM
result = call_llm("gpt-5-mini", messages, cache=cache)  # returns cached
cache.clear()  # manual invalidation
```

Custom sync backend (Redis, disk, etc.):
```python
from llm_client import CachePolicy, LLMCallResult

class RedisCache:
    def get(self, key: str) -> LLMCallResult | None: ...
    def set(self, key: str, value: LLMCallResult) -> None: ...
```

Async cache backend (non-blocking in async code):
```python
from llm_client import AsyncCachePolicy, LLMCallResult

class AsyncRedisCache:
    async def get(self, key: str) -> LLMCallResult | None: ...
    async def set(self, key: str, value: LLMCallResult) -> None: ...
```

Async functions (`acall_llm`, etc.) accept either `CachePolicy` or `AsyncCachePolicy` — sync caches work everywhere, async caches avoid blocking the event loop.

### Fallback Models

Automatic fallback to secondary models when the primary fails all retries:

```python
result = call_llm(
    "gpt-5-mini", messages,
    fallback_models=["deepseek/deepseek-chat", "ollama/llama3"],
    on_fallback=lambda failed, err, next_model: print(f"{failed} failed, trying {next_model}"),
)
```

Each model gets its own full retry cycle. Non-retryable errors on one model still trigger fallback to the next.

### Observability Hooks

```python
from llm_client import Hooks, call_llm

hooks = Hooks(
    before_call=lambda model, msgs, kw: print(f"Calling {model}"),
    after_call=lambda result: print(f"Got {len(result.content)} chars, ${result.cost:.4f}"),
    on_error=lambda err, attempt: print(f"Attempt {attempt} failed: {err}"),
)

result = call_llm("gpt-5-mini", messages, hooks=hooks)
```

All three callbacks are optional. Hooks fire for each attempt (including retries and fallbacks). Works with all functions including streaming.

### Truncation Detection

Check `finish_reason` to detect truncated responses:
```python
result = call_llm("gpt-5-mini", messages)
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

**Non-retryable errors skip the retry loop** for the current model, but still trigger fallback to the next model in `fallback_models` (if configured). If all models fail, the error is wrapped in an `LLMError` subclass and raised:
- `AuthenticationError` (401): invalid API key → `LLMAuthError`
- `PermissionDeniedError` (403): forbidden → `LLMAuthError`
- `NotFoundError` (404): model doesn't exist → `LLMModelNotFoundError`
- `BudgetExceededError`: litellm budget limit hit → `LLMQuotaExhaustedError`
- `ContentPolicyViolationError`: content filter triggered → `LLMContentFilterError`
- `RateLimitError` with quota keywords (e.g., "exceeded your current quota"): billing/credits exhausted → `LLMQuotaExhaustedError`

Note: `RateLimitError` (429) without quota keywords is treated as transient and retried (`LLMRateLimitError`). Error classification uses litellm exception types first, with string pattern fallback for generic exceptions.

`num_retries` controls the count (default: 2). Empty responses are automatically retried unless the model made tool calls.

## Structured Error Types

After all retries and fallbacks are exhausted, the final error is wrapped in a structured `LLMError` subclass. Callers can catch specific failure modes:

```python
from llm_client import LLMQuotaExhaustedError, LLMRateLimitError, LLMAuthError, LLMError

try:
    result = await acall_llm("gpt-5-mini", messages, fallback_models=["gemini/gemini-2.5-flash"])
except LLMQuotaExhaustedError:
    # All models exhausted their quota — switch provider or abort
    ...
except LLMAuthError:
    # API key invalid — fix credentials
    ...
except LLMRateLimitError:
    # Transient rate limit survived all retries — wait longer
    ...
except LLMError as e:
    # Any other LLM failure (truncation, empty response, etc.)
    print(e.original)  # access the underlying exception
```

Error hierarchy (all subclass `LLMError` which subclasses `Exception`):

| Error Type | Meaning | Retryable? |
|---|---|---|
| `LLMRateLimitError` | Transient 429 | Yes |
| `LLMQuotaExhaustedError` | Permanent quota/billing exhaustion | No |
| `LLMAuthError` | 401/403 — bad API key or forbidden | No |
| `LLMContentFilterError` | Content policy violation | No |
| `LLMModelNotFoundError` | 404 — model doesn't exist | No |
| `LLMTransientError` | 500/502/503, timeout, connection | Yes |
| `LLMError` | Base class / uncategorized | Varies |

The `original` attribute on any `LLMError` holds the underlying exception (e.g., the raw litellm error). The `classify_error()` and `wrap_error()` functions are also exported for manual use.

**Agent models note**: Agent SDK calls (`claude-code`, `codex`) do NOT wrap exceptions in `LLMError` — they propagate SDK-native exceptions directly. Only litellm-routed calls get structured error wrapping.

## Structured Output Routing

`call_llm_structured` uses three-tier routing — no code changes needed:

1. **GPT-5** → Responses API with native `text.format` JSON schema
2. **Models supporting `response_schema`** (GPT-5-mini, Claude, Gemini, etc.) → `litellm.completion()` with `response_format` JSON schema
3. **Older models** → instructor fallback

If the native JSON schema path fails due to a provider schema limitation (e.g., Gemini's nesting depth limit for deeply nested Pydantic models), the call automatically falls back to the instructor path. No code changes needed on the consumer side.

```python
# All use the same call — routing is automatic
data, meta = call_llm_structured("gpt-5", messages, response_model=Entity)              # Responses API
data, meta = call_llm_structured("gpt-5-mini", messages, response_model=Entity)          # native JSON schema
data, meta = call_llm_structured("gemini/gemini-2.5-flash", messages, response_model=Entity)  # native JSON schema
```

Pydantic schemas are automatically made strict-mode-compatible (`additionalProperties: false` added recursively, including through `anyOf`/`allOf`/`oneOf` combinators for Optional/Union fields).

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
- `response_format` with `json_schema` works for structured JSON output via `call_llm`

```python
# Or manual JSON schema via call_llm
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

Gemini 2.5+ thinking models allocate reasoning tokens by default, consuming output budget. llm_client automatically injects `thinking: {type: "enabled", budget_tokens: 0}` for these models so all tokens go to the actual response.

Override with your own config if you want thinking tokens:
```python
result = call_llm("gemini/gemini-3-flash", messages, thinking={"type": "enabled", "budget_tokens": 1000})
```

## Deprecated Model Warnings

llm_client emits loud `DeprecationWarning` when a deprecated/outclassed model is used. The warning text is intentionally aggressive — it's designed to make LLM agents STOP and ask the user before proceeding.

Deprecated models (as of Feb 2026):
- `gpt-4o` → use `gpt-5` (cheaper and smarter)
- `gpt-4o-mini` → use `deepseek/deepseek-chat` or `gemini/gemini-2.5-flash`
- `o1-mini` → use `o4-mini`
- `o1-pro` → use `o3`
- `gemini-1.5-*` → use `gemini-2.5-flash` or `gemini-2.5-pro`
- `gemini-2.0-flash` → use `gemini-2.5-flash`
- `claude-3-*` → use `claude-4.5-*` equivalents
- `mistral-large` → use `deepseek/deepseek-chat`

The warning fires on all entry points (`call_llm`, `acall_llm`, `call_llm_structured`, etc.). Batch and `*_with_tools` functions inherit the check from the core functions they delegate to.

To add new deprecated models, update `_DEPRECATED_MODELS` in `client.py`.

See `~/projects/LLM_MODELS.md` for the full model comparison guide.

## Fence Stripping

`strip_fences()` removes markdown code fences from LLM output. Useful when calling `call_llm()` and parsing JSON manually:

```python
from llm_client import call_llm, strip_fences
import json

result = call_llm("gpt-5-mini", messages)
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

result = call_llm("gpt-5-mini", messages, caching=True)
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

## Agent SDK Models

Agent models route through the Claude Agent SDK instead of litellm. Same `call_llm()` interface — the model string determines routing:

```python
# Agent SDK (Claude Agent SDK)
result = call_llm("claude-code", messages)              # Default model
result = call_llm("claude-code/opus", messages)          # Specify underlying model
result = call_llm("claude-code/haiku", messages)         # Cheaper model

# Regular API (existing litellm routing)
result = call_llm("anthropic/claude-opus-4", messages)   # Raw Anthropic API
```

### Model Naming

| Model String | SDK | Underlying Model | Notes |
|---|---|---|---|
| `claude-code` | Claude Agent SDK | SDK default | |
| `claude-code/opus` | Claude Agent SDK | opus | |
| `claude-code/sonnet` | Claude Agent SDK | sonnet | |
| `claude-code/haiku` | Claude Agent SDK | haiku | |
| `codex` | Codex SDK | SDK default (gpt-5.3-codex) | |
| `codex/gpt-5.3-codex` | Codex SDK | gpt-5.3-codex | Best capability. No API access — CLI/SDK only. |
| `codex/gpt-5.3-codex-spark` | Codex SDK | gpt-5.3-codex-spark | Near-instant speed. ChatGPT Pro only. |
| `codex/gpt-5.2-codex` | Codex SDK | gpt-5.2-codex | Previous gen. Has API access. |
| `codex/gpt-5.1-codex` | Codex SDK | gpt-5.1-codex | Older. |
| `codex/gpt-5` | Codex SDK | gpt-5 | General-purpose, not coding-optimized. |
| `openai-agents/*` | Reserved | NotImplementedError | |

### Agent-specific kwargs

Pass these through `call_llm` `**kwargs`:

```python
# Claude Agent SDK kwargs
result = call_llm("claude-code", messages,
    allowed_tools=["Read", "Edit", "Glob"],
    permission_mode="acceptEdits",
    cwd="/path/to/project",
    max_turns=10,
    max_budget_usd=1.0,
    mcp_servers={                          # dict of name -> McpServerConfig
        "my-server": {
            "command": "/path/to/python",
            "args": ["-u", "server.py"],
            "env": {"SOME_VAR": "value"},
        },
    },
)

# Codex SDK kwargs
result = call_llm("codex", messages,
    sandbox_mode="workspace-write",       # "read-only" | "workspace-write" | "danger-full-access"
    approval_policy="never",              # "never" | "on-request" | "on-failure" | "untrusted"
    working_directory="/path/to/project",
    model_reasoning_effort="medium",      # "minimal" | "low" | "medium" | "high"
    network_access_enabled=True,
    web_search_enabled=False,
    additional_directories=["/other/path"],
    skip_git_repo_check=False,
    api_key="sk-...",                     # optional override
    base_url="https://...",               # optional override
    # MCP server control (load only specified servers instead of global config)
    mcp_servers={                          # dict of name -> {command, args?, cwd?, env?}
        "my-server": {
            "command": "/path/to/python",
            "args": ["-u", "server.py"],
            "env": {"SOME_VAR": "value"},
        },
    },
    codex_home="/path/to/custom/home",    # low-level: custom HOME dir (reads .codex/config.toml)
)
```

**Codex cost estimation**: Codex SDK only provides token counts (no USD). Cost is estimated via `litellm.completion_cost()` using the underlying model name. Documented as approximate.

### MCP Server Control

Both agent SDKs support `mcp_servers` — dict of `{name: {command, args?, env?}}`:

```python
# Claude Code — passes mcp_servers natively to ClaudeAgentOptions
result = await acall_llm("claude-code", messages,
    mcp_servers={
        "digimon-kgrag": {
            "command": "/path/to/python",
            "args": ["-u", "digimon_mcp_stdio_server.py"],
            "env": {"SOME_KEY": "value"},
        },
    },
    cwd="/path/to/project",
    permission_mode="bypassPermissions",
)

# Codex — creates temp config.toml with only specified servers
result = await acall_llm("codex", messages,
    mcp_servers={
        "digimon-kgrag": {
            "command": "/path/to/python",
            "args": ["-u", "digimon_mcp_stdio_server.py"],
            "env": {"CLAUDECODE": ""},
        },
    },
    working_directory="/path/to/project",
    approval_policy="never",
)
```

For Codex, this creates a temporary config with only the specified servers, dramatically reducing context window usage (17 servers = 253k tokens overhead vs 1 server = ~15k). For Claude Code, `mcp_servers` is passed directly to the SDK's native `ClaudeAgentOptions.mcp_servers`.

**`codex_home`** (low-level, Codex only): Path to a directory used as `HOME` — Codex reads `$HOME/.codex/config.toml`. Mutually exclusive with `mcp_servers`.

### Agent Capabilities

- **Streaming**: `stream_llm("claude-code", ...)` / `astream_llm(...)` — message-level granularity (each `TextBlock`), not token-level
- **Structured output**: `call_llm_structured("claude-code", ..., response_model=MyModel)` — uses SDK `output_format` with JSON schema
- **Batch**: `call_llm_batch("claude-code", ...)` — concurrent `call_llm` calls via semaphore
- **Fallback works**: `call_llm("claude-code", ..., fallback_models=["gpt-5-mini"])` works
- **Default 0 retries**: Agent calls have side effects; retries default to 0 unless explicit `retry=RetryPolicy(...)` is passed
- **Clean subprocess env**: Auto-loaded API keys (from `~/.secrets/api_keys.env`) and `CLAUDECODE` are stripped from the agent subprocess env. The bundled Claude CLI uses OAuth, not API keys — inheriting `ANTHROPIC_API_KEY` causes it to crash. Keys already in `os.environ` before `import llm_client` are preserved.

### Agent Limitations

| Feature | Status | Reason |
|---------|--------|--------|
| Tool calling | Won't implement | Agents run tools autonomously. Use `allowed_tools=` kwarg to configure agent's built-in tools. |
| Caching | Won't implement | Agents have side effects (file writes, bash commands). Caching is unsafe. |
| Token-level streaming | Deferred | Message-level streaming works. Token-level requires parsing raw `StreamEvent` dicts (fragile). |
| OpenAI Agents SDK | Deferred | `openai-agents/*` prefix reserved. Architecture supports it. |
| Gemini CLI SDK | Deferred | v0.1.0, 19 commits, 5mo stale. Uses gpt-4o-mini to parse CLI stdout (extra cost/latency, misparses stderr). No cost/usage/streaming/structured. Gemini CLI has `--output-format json` but SDK doesn't use it yet. Use `gemini/gemini-2.5-flash` via litellm instead. Revisit when SDK ships native JSON parsing. |

## MCP Agent Loop

Any litellm model can drive MCP tool-calling via `mcp_servers` or `mcp_sessions` kwargs:

```python
# Per-call: starts/stops MCP servers each call
result = await acall_llm("gemini/gemini-3-flash-preview", messages,
    mcp_servers={"my-server": {"command": "python", "args": ["server.py"]}},
    max_turns=20,
)

# Persistent pool: start servers once, reuse across calls
from llm_client import MCPSessionPool

async with MCPSessionPool(mcp_servers) as pool:
    for question in questions:
        result = await acall_llm(model, msgs, mcp_sessions=pool, max_turns=25)
```

`MCPSessionPool` avoids per-call server startup overhead for batch workloads. Pass `mcp_sessions=pool` instead of `mcp_servers=...`. Requires `pip install mcp`.

MCP loop kwargs: `mcp_servers`, `mcp_sessions`, `max_turns` (20), `mcp_init_timeout` (30s), `tool_result_max_length` (50k chars).

## Installation

```bash
pip install -e .                    # Basic (includes error types, retry, fallback, caching, streaming)
pip install -e ".[structured]"      # With instructor for structured output
pip install -e ".[agents]"          # With Claude Agent SDK
pip install -e ".[codex]"           # With Codex SDK
pip install -e ".[all-agents]"      # Both agent SDKs
```

## Environment

### API Key Auto-Loading

On import, llm_client automatically loads API keys from `~/.secrets/api_keys.env` (or the path in `LLM_CLIENT_KEYS_FILE` env var). Standard `.env` format — `KEY=VALUE` lines, comments with `#`, `export` prefix optional. Existing env vars are never overwritten.

This means any project that `import llm_client` gets API keys automatically — no per-project `.env` files or `load_dotenv()` calls needed.

Manual override still works (litellm convention):
```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GEMINI_API_KEY=...
export OPENROUTER_API_KEY=sk-or-...
```

## I/O Logging

Every `call_llm` / `acall_llm` call is logged to JSONL. Enabled by default.

**Output**: `{DATA_ROOT}/{PROJECT}/{PROJECT}_llm_client_data/calls.jsonl`

**Env vars**:
- `LLM_CLIENT_LOG_ENABLED` — `"1"` (default) or `"0"` to disable
- `LLM_CLIENT_DATA_ROOT` — base dir (default: `~/projects/data`)
- `LLM_CLIENT_PROJECT` — project name (default: `basename(os.getcwd())`)

**Runtime config**:
```python
from llm_client import configure_logging
configure_logging(enabled=False)                    # disable
configure_logging(project="my_project")             # override project name
configure_logging(data_root="/tmp/llm_logs")        # override data root
```

Each JSONL record contains: `timestamp`, `model`, `messages` (truncated), `response` (truncated), `usage`, `cost`, `finish_reason`, `latency_s`, `error`, `caller`, `task`.

Pass `task="extraction"` (or any task name) to `call_llm` / `acall_llm` / `call_llm_structured` / etc. to tag log records for performance tracking.

Logging never raises — failures are silently dropped to avoid breaking LLM calls.

## Model Registry + Task-Based Selection

Centralized model matrix — no more hardcoded model strings. Each model has intelligence/speed/cost attributes. Task profiles define requirements and sort preferences.

```python
from llm_client import get_model, list_models, query_performance

# Task-based selection
model = get_model("extraction")      # → "gemini/gemini-3-flash" (highest intelligence w/ structured output)
model = get_model("bulk_cheap")      # → "gpt-5-nano" (cheapest)
model = get_model("graph_building")  # → "xai/grok-4.1-fast" (cheapest w/ structured + min intel)

# Use with call_llm — task kwarg tags the log for performance tracking
result = call_llm(get_model("synthesis"), messages, task="synthesis")

# Introspection
for m in list_models():
    print(f'{m["name"]:25s} intel={m["intelligence"]}  cost=${m["cost"]:.2f}')

# Performance analytics from logged calls
perf = query_performance(task="extraction", days=7)
# → [{"task": "extraction", "model": "...", "call_count": 142, "total_cost": 0.83, ...}]
```

### Default models (8)

| Name | litellm_id | Intel | Speed | Cost ($/1M) | Notes |
|------|-----------|-------|-------|-------------|-------|
| deepseek-chat | deepseek/deepseek-chat | 42 | 36 | $0.32 | Bulk default |
| gemini-3-flash | gemini/gemini-3-flash | 46 | 207 | $1.13 | Best mid-tier |
| gemini-2.5-flash | gemini/gemini-2.5-flash | 34 | 152 | $0.68 | Free tier |
| gemini-2.5-flash-lite | gemini/gemini-2.5-flash-lite | 28 | 250 | $0.175 | Cheapest Google |
| gpt-5-mini | gpt-5-mini | 41 | 127 | $0.69 | Reliable structured |
| gpt-5 | gpt-5 | 45 | 98 | $3.44 | Frontier |
| gpt-5-nano | gpt-5-nano | 27 | 141 | $0.14 | Cheapest OpenAI |
| grok-4.1-fast | xai/grok-4.1-fast | 39 | 179 | $0.28 | 2M context |

### Default tasks (6)

| Task | Min Intel | Requires | Prefer |
|------|-----------|----------|--------|
| extraction | 35 | structured_output | intelligence, -cost |
| bulk_cheap | 25 | — | -cost, speed |
| synthesis | 40 | — | intelligence, -cost |
| graph_building | 30 | structured_output | -cost, speed |
| agent_reasoning | 42 | — | intelligence |
| code_generation | 38 | — | intelligence, speed |

### Config override

Config loading chain: `LLM_CLIENT_MODELS_CONFIG` env var → `~/.config/llm_client/models.yaml` → built-in defaults.

`available_only=True` (default) filters to models whose `api_key_env` is set in `os.environ`.

## Task Graph Runner

Parse YAML DAGs, dispatch to agents, validate outputs, checkpoint via git. See `docs/TASK_GRAPH_DESIGN.md` for full specification.

```python
from llm_client import load_graph, run_graph

graph = load_graph("path/to/graph.yaml")
report = await run_graph(graph)
for tr in report.task_results:
    print(f"{tr.task_id}: {tr.status} (${tr.cost_usd:.3f})")
```

### YAML Format

```yaml
graph:
  id: my_pipeline
  description: "Example pipeline"
  timeout_minutes: 60
  checkpoint: git_tag  # git_tag | git_commit | none

tasks:
  collect:
    difficulty: 2
    agent: codex
    prompt: "Collect data and write to results/data.json"
    mcp_servers:
      - sam-gov-government
    validate:
      - type: file_exists
        path: results/data.json
    outputs:
      data_file: results/data.json

  process:
    difficulty: 3
    depends_on: [collect]
    prompt: "Process {collect.outputs.data_file}"
```

### Difficulty Tiers

| Tier | What | Default Model |
|------|------|--------------|
| 0 | Scripted — no LLM | None |
| 1 | Simple: formatting, extraction | deepseek-chat or ollama |
| 2 | Moderate: classification, analysis | gemini-2.5-flash |
| 3 | Complex: multi-hop reasoning, synthesis | claude-sonnet-4.5 |
| 4 | Agent: multi-step autonomous tool use | codex or claude-code SDK |

```python
from llm_client import get_model_for_difficulty
model = get_model_for_difficulty(2)  # cheapest available at tier 2
```

### Validators

| Type | What |
|------|------|
| `file_exists` | File was created |
| `file_not_empty` | File exists and has content |
| `json_schema` | JSON matches schema |
| `pytest` | pytest exits 0 |
| `sql_count` | SQL count matches check expression |
| `command` | Shell command exits 0 |
| `mcp_call` | MCP tool result matches check |

```python
from llm_client import run_validators, register_validator

results = run_validators([
    {"type": "file_exists", "path": "output.json"},
    {"type": "sql_count", "db": "data.db", "query": "SELECT count(*) FROM items", "check": "> 0"},
])
```

## Tests

```bash
pytest tests/ -v   # All mocked (no real API calls)
```

## Dependencies

- `litellm>=1.81.3` — Multi-provider abstraction (bumped from 1.40.0 to fix Gemini nullable type bug)
- `pydantic>=2.0` — Data validation
- `instructor>=1.14.0` — Structured output fallback for older models (optional; modern models use native JSON schema)
- `claude-agent-sdk>=0.1.30` — Claude Agent SDK for agent models (optional; install with `pip install llm_client[agents]`)
- `openai-codex-sdk>=0.1.11` — Codex SDK for codex agent models (optional; install with `pip install llm_client[codex]`)
- `pyyaml>=6.0` — YAML parsing for task graphs
- `jsonschema>=4.0` — JSON schema validation (optional; `json_schema` validator degrades gracefully without it)
- `pytest-asyncio>=0.23` — Async test support (dev only)
