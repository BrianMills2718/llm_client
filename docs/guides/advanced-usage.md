# Advanced Usage

## Long-thinking mode

`gpt-5.2-pro` supports long-thinking runs. Set `reasoning_effort="high"` or
`"xhigh"` to enable Responses background mode with automatic polling:

```python
result = call_llm(
    "gpt-5.2-pro",
    messages,
    reasoning_effort="xhigh",
    task="long_thinking",
    trace_id="long_thinking",
    max_budget=5.00,
    background_timeout=900,
    background_poll_interval=15,
)
```

Requires `OPENROUTER_API_KEY` (OpenRouter endpoint) or `OPENAI_API_KEY` (direct).

## Execution modes

`execution_mode` enforces model capabilities before dispatch:

- `text` (default): regular completion calls.
- `structured`: structured extraction intent.
- `workspace_agent`: requires agent models (`codex`, `claude-code`, `openai-agents/*`).
- `workspace_tools`: requires non-agent models with `python_tools`, `mcp_servers`, or `mcp_sessions`.

## Retry policy

```python
from llm_client import RetryPolicy, call_llm, linear_backoff

policy = RetryPolicy(
    max_retries=5,
    base_delay=0.5,
    backoff=linear_backoff,
    retry_on=["custom error"],
    on_retry=lambda a, err, d: print(f"Retry {a}"),
)
result = call_llm("gpt-5-mini", messages, retry=policy, task="...", trace_id="...", max_budget=1.00)
```

## Fallback models

```python
result = call_llm(
    "gpt-5-mini", messages,
    fallback_models=["gemini/gemini-2.5-flash", "ollama/llama3"],
    task="fallbacks",
    trace_id="fallbacks",
    max_budget=1.00,
    on_fallback=lambda failed, err, next_: print(f"{failed} failed, trying {next_}"),
)
```

## Observability hooks

```python
from llm_client import Hooks, call_llm

hooks = Hooks(
    before_call=lambda model, msgs, kw: print(f"Calling {model}"),
    after_call=lambda result: print(f"${result.cost:.4f}"),
    on_error=lambda err, attempt: print(f"Attempt {attempt} failed"),
)
result = call_llm("gpt-5-mini", messages, hooks=hooks, task="...", trace_id="...", max_budget=1.00)
```

## Response caching

```python
from llm_client import LRUCache, call_llm

cache = LRUCache(maxsize=128, ttl=3600)
result = call_llm("gpt-5-mini", messages, cache=cache, task="...", trace_id="...", max_budget=1.00)
# Second call with same args returns cached (cache_hit=True, marginal_cost=0.0)
```

Implement `CachePolicy` protocol for custom backends (Redis, disk, etc.).

## Routing configuration

OpenRouter-first routing is on by default. To use direct provider routing:

```python
from llm_client import ClientConfig, call_llm

cfg = ClientConfig(routing_policy="direct")
result = call_llm("gpt-5-mini", messages, config=cfg, task="...", trace_id="...", max_budget=1.00)
```

Or via environment: `LLM_CLIENT_OPENROUTER_ROUTING=off`

Provider-governance rules are applied before the final routing decision:

- exact `gpt-5.4` requests canonicalize to `codex/gpt-5.4`
- bare Gemini ids canonicalize to `gemini/<model>`
- `result.routing_trace["provider_governance_events"]` records these decisions

### Shared provider coordination

Gemini and other providers can use shared cooldown and lease state across
processes/worktrees to prevent first-attempt stampedes against the same quota
surface.

- `LLM_CLIENT_RATE_LIMIT_SHARED_ENABLED=1` enables shared leases
- `LLM_CLIENT_RATE_LIMIT_SHARED_LIMITS='{"google": 4}'` overrides provider caps
- `LLM_CLIENT_RATE_LIMIT_COOLDOWN_FLOORS='{"google": 15}'` overrides provider cooldown floors
- `LLM_CLIENT_RATE_LIMIT_STATE_PATH=/path/to/llm_rate_limit_state.sqlite3` changes the SQLite state location

### OpenRouter key rotation

If OpenRouter returns key exhaustion (402/403), retry loops auto-rotate to backup keys.
Configure a key pool with:
- `OPENROUTER_API_KEYS` (comma/semicolon/newline-delimited), or
- `OPENROUTER_API_KEY` plus numbered vars (`OPENROUTER_API_KEY_2`, `_3`, ...).

## Timeout policy

- `LLM_CLIENT_TIMEOUT_POLICY=ban` — disable all per-call request timeouts globally.
- `LLM_CLIENT_TIMEOUT_POLICY=allow` (default) — permit explicit `timeout` values.

## Foundation event strict mode

MCP/tool loops validate emitted foundation events. Enable strict failure mode:

```bash
export FOUNDATION_SCHEMA_STRICT=1
```

## Inspect active calls

```python
from llm_client import get_active_llm_calls
active = get_active_llm_calls(project="my-project", limit=20)
```

## Model identity fields

- `result.model` — legacy compatibility; use `resolved_model` instead
- `result.requested_model` — caller input
- `result.resolved_model` / `result.execution_model` — terminal executed model
- `result.routing_trace` — routing/fallback metadata
- `result.warning_records` — machine-readable `LLMC_WARN_*` warnings
