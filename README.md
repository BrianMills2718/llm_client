# llm-client

Thin wrapper around [litellm](https://github.com/BerriAI/litellm). Swap models by changing one string.

## Release notes

- Current version: `0.6.1`
- Latest changes are tracked in `CHANGELOG.md`.

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

# Agent SDK models default to subscription accounting (no per-call API USD)
agent_result = call_llm("claude-code", [{"role": "user", "content": "Refactor this"}], task="dev", trace_id="demo/agent", max_budget=0)
print(agent_result.cost)         # 0.0
print(agent_result.billing_mode) # "subscription_included"
```

### Routing defaults

- OpenRouter-first routing is enabled by default (`LLM_CLIENT_OPENROUTER_ROUTING=on`).
- Bare model IDs like `gpt-5-mini` are normalized to `openrouter/openai/gpt-5-mini`.
- If you want bare OpenAI IDs to stay bare (including Responses API routing for `gpt-5*`), set:

```bash
export LLM_CLIENT_OPENROUTER_ROUTING=off
```

OpenRouter key rotation on key/quota exhaustion:
- If OpenRouter returns key-level exhaustion (for example `Key limit exceeded`
  or `Insufficient credits` with `402`), retry loops can auto-rotate to a
  backup key and retry immediately.
- Configure a key pool with either:
  - `OPENROUTER_API_KEYS` (comma/semicolon/newline-delimited), or
  - `OPENROUTER_API_KEY` plus numbered vars (`OPENROUTER_API_KEY_2`, `_3`, ...).
- If an explicit `api_key=...` is passed at call time, automatic rotation is
  disabled for that call.

### Typed routing/config (preferred)

For deterministic behavior across environments, pass explicit `ClientConfig`:

```python
from llm_client import ClientConfig, call_llm

cfg = ClientConfig(
    routing_policy="direct",              # or "openrouter"
)

result = call_llm("gpt-5-mini", messages, config=cfg)
```

### Long-thinking mode (`gpt-5.2-pro`)

`gpt-5.2-pro` supports long-thinking runs. When you set
`reasoning_effort="high"` or `"xhigh"`, the client automatically enables
Responses background mode and polls to completion.

```python
result = call_llm(
    "gpt-5.2-pro",
    messages,
    reasoning_effort="xhigh",
    background_timeout=900,        # optional, seconds (default 900)
    background_poll_interval=15,   # optional, seconds (default 15)
)
```

Notes:
- Requires `OPENAI_API_KEY`.
- Background retrieval currently supports OpenAI API bases only (`api.openai.com`).
- Unsupported endpoint/key failures raise `LLMConfigurationError` with stable
  `error_code` values:
  - `LLMC_ERR_BACKGROUND_ENDPOINT_UNSUPPORTED`
  - `LLMC_ERR_BACKGROUND_OPENAI_KEY_REQUIRED`
- `background_timeout` caps total polling time.
- `background_poll_interval` controls polling cadence.
- `result.routing_trace["background_mode"]` indicates background-mode routing.

Model identity fields on results:
- `result.model` (legacy compatibility field; do not rely on it for routing provenance)
- `result.requested_model` (caller input)
- `result.resolved_model` / `result.execution_model` (terminal executed model)
- `result.routing_trace` (routing/fallback metadata)
- `result.warning_records` (machine-readable `LLMC_WARN_*` warnings)

Foundation event schema strict mode:
- MCP/tool loops validate emitted foundation events.
- By default, invalid events are recorded as warnings in run metadata.
- Enable strict failure mode to fail fast on any invalid event payload:

```bash
export FOUNDATION_SCHEMA_STRICT=1
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

Structured-output provider notes:
- Some Gemini models can reject deeply nested JSON schemas with
  `INVALID_ARGUMENT` (nesting depth), and tool-mode may return empty choices
  for certain requests.
- `call_llm_structured` automatically falls back from native schema routing to
  instructor-based structured parsing when this happens.
- Empty `choices=[]` payloads are classified as typed empty-response events
  (`provider_empty_candidates`) and retried according to policy instead of
  surfacing raw `IndexError` failures.
- For 429 responses that include provider retry hints (for example Gemini
  `retryDelay`), retry loops honor the hinted window before the next attempt.
- For high-reliability extraction, set a fallback model that handles deep
  schemas well (for example `openrouter/openai/gpt-5-mini`) via
  `fallback_models=[...]` at call sites.
- For long agentic tool loops, prefer an explicit primary+fallback chain rather
  than Gemini-only retries:
  `openrouter/deepseek/deepseek-chat` ->
  `openrouter/openai/gpt-5-mini` ->
  `gemini/gemini-2.5-flash` (optional).
- Optional A/B switch: set `LLM_CLIENT_GEMINI_NATIVE_MODE=on` to route
  `gemini/*` calls through a direct Gemini REST path (bypassing the
  OpenAI-chat compatibility layer). Unsupported kwargs automatically fall back
  to the standard litellm route.

### Execution mode contract

`execution_mode` enforces model capabilities before dispatch, so incompatible
model/kwargs combinations fail fast with `LLMCapabilityError`.

- `text` (default): regular completion calls.
- `structured`: structured extraction intent.
- `workspace_agent`: requires agent models (`codex`, `claude-code`,
  `openai-agents/*`) and supports agent kwargs like `working_directory`,
  `approval_policy`, `cwd`, `permission_mode`.
- `workspace_tools`: requires non-agent models and one of `python_tools`,
  `mcp_servers`, or `mcp_sessions`.

For code-generation/editing workflows that depend on workspace side effects,
always set `execution_mode="workspace_agent"` to prevent accidental routing to
chat-only models.

### Observability tags

- `task`, `trace_id`, and `max_budget` are optional in normal local usage.
- Missing values are auto-filled as `task="adhoc"`, `trace_id="auto/<caller>/<id>"`, and `max_budget=0` (unlimited).
- Strict enforcement is enabled automatically in CI and benchmark/eval-style tasks, or explicitly with:

```bash
export LLM_CLIENT_REQUIRE_TAGS=1
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

### MCP Agent Composability Contracts

When using MCP agent loops (`mcp_servers=...` or `mcp_sessions=...`), you can enforce
tool-chain legality and expose only currently legal tools:

```python
result = await acall_llm(
    "openrouter/deepseek/deepseek-chat",
    messages,
    mcp_servers={...},
    enforce_tool_contracts=True,
    progressive_tool_disclosure=True,
    initial_artifacts=("QUERY_TEXT",),
    tool_contracts={
        "entity_onehop": {
            "requires_all": [{"kind": "ENTITY_SET", "ref_type": "id"}],
            "produces": [{"kind": "ENTITY_SET", "ref_type": "id"}],
        },
        "chunk_get_text_by_entity_ids": {
            "requires_all": [{"kind": "ENTITY_SET", "ref_type": "id"}],
            "produces": [{"kind": "CHUNK_SET", "ref_type": "fulltext"}],
        },
        "chunk_get_text_by_chunk_ids": {
            "artifact_prereqs": "none",
            "produces": [{"kind": "CHUNK_SET", "ref_type": "fulltext"}],
        },
        "todo_list": {"is_control": True},
    },
    # Optional reliability controls:
    forced_final_max_attempts=3,
    forced_final_circuit_breaker_threshold=2,
    finalization_fallback_models=["openrouter/openai/gpt-5-mini"],
    retrieval_stagnation_turns=4,
)

raw = result.raw_response  # MCPAgentResult
print(raw.metadata["primary_failure_class"])
print(raw.metadata["first_terminal_failure_event_code"])
print(raw.metadata["failure_event_code_counts"])
print(raw.metadata["available_capabilities_final"])
print(raw.metadata["hard_bindings_hash"], raw.metadata["full_bindings_hash"])
print(raw.metadata["lane_closure_analysis"]["lane_closed"])
print(raw.metadata["tool_disclosure_repair_suggestions"])
```

Notes:
- `artifact_prereqs: "none"` declares a self-contained tool for artifact-state checks only.
- Binding checks and schema checks still run pre-execution.
- Metadata includes both enforcement and attribution binding fingerprints:
  - `hard_bindings_hash` (authoritative binding scope)
  - `full_bindings_hash` (full normalized binding state)
- Runtime also emits `run_config_hash` for model/lane/policy comparability.
- Forced-final fallback is finalization-only (no tool calls): configure with
  `finalization_fallback_models=[...]`.
- Forced-final attempts are bounded by `forced_final_max_attempts` and
  `forced_final_circuit_breaker_threshold`.
- Retrieval stagnation fuse (`retrieval_stagnation_turns`) terminates long
  evidence loops that produce no new evidence digest.

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

Cache hits are returned with:
- `cache_hit=True`
- `marginal_cost=0.0`
- `cost_source="cache_hit"`

The original `cost` field remains as the attributed/original call cost for analytics.

Implement `CachePolicy` protocol for custom backends (Redis, disk, etc.).

### Agent billing mode

By default, agent SDK models (`claude-code`, `codex`) use subscription accounting:
- `LLM_CLIENT_AGENT_BILLING_MODE=subscription` (default)
- Per-call `cost=0.0`, `billing_mode="subscription_included"`

If you run agent SDK calls in API-metered mode, set:

```bash
export LLM_CLIENT_AGENT_BILLING_MODE=api
```

### Agent retry safety

- Agent SDK retries are disabled by default to avoid duplicate side effects.
- To enable retries for explicitly safe/read-only agent runs:

```bash
export LLM_CLIENT_AGENT_RETRY_SAFE=1
```

- Or per-call:

```python
result = call_llm(
    "claude-code",
    messages,
    agent_retry_safe=True,
    num_retries=2,
)
```

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

`LLMCallResult` fields: `.content`, `.usage`, `.cost`, `.marginal_cost`, `.cost_source`, `.billing_mode`, `.cache_hit`, `.model`, `.tool_calls`, `.finish_reason`, `.raw_response`

`call_llm`, `call_llm_structured`, `call_llm_with_tools` (and async variants) accept: `timeout`, `num_retries`, `reasoning_effort` (Claude models and `gpt-5.2-pro` long-thinking), `api_base`, `retry_on`, `on_retry`, `cache`, `retry` (RetryPolicy), `fallback_models`, `on_fallback`, `hooks` (Hooks), `execution_mode` (`text`/`structured`/`workspace_agent`/`workspace_tools`), plus any `**kwargs` passed through to `litellm.completion`.

`stream_llm` / `astream_llm` (and `*_with_tools` variants) accept: `timeout`, `num_retries`, `reasoning_effort`, `api_base`, `retry`, `fallback_models`, `on_fallback`, `hooks`, plus `**kwargs`. No `cache` param (caching streams doesn't make sense).

`*_batch` functions additionally accept: `max_concurrent` (5), `return_exceptions`, `on_item_complete`, `on_item_error`.

## Experiment Observability

Use the built-in CLI to inspect and compare benchmark/eval runs recorded via
`start_run` / `log_item` / `finish_run`.

```bash
python -m llm_client experiments
python -m llm_client experiments --condition-id forced_off --scenario-id phase1_falsification --phase phase1
python -m llm_client experiments --compare RUN_BASE RUN_CANDIDATE
python -m llm_client experiments --compare-cohorts baseline forced_reduced forced_off --baseline-condition-id baseline --scenario-id phase1_falsification --phase phase1
python -m llm_client experiments --compare-diff RUN_BASE RUN_CANDIDATE
python -m llm_client experiments --detail RUN_ID
python -m llm_client experiments --detail RUN_ID --det-checks default
python -m llm_client experiments --detail RUN_ID --review-rubric extraction_quality
python -m llm_client experiments --detail RUN_ID --gate-policy '{"pass_if":{"avg_llm_em_gte":80}}' --gate-fail-exit-code
```

`--detail` now supports:
- automatic triage over item-level error classes,
- deterministic checks (`--det-checks`),
- rubric-based LLM review (`--review-rubric` / `--review-model`),
- policy gates (`--gate-policy`) with optional non-zero exit on failure.

Run-level cohort metadata is also supported through `start_run(..., condition_id=..., seed=..., replicate=..., scenario_id=..., phase=...)`, with cohort-level aggregate comparisons available via `compare_cohorts(...)` and `experiments --compare-cohorts`.

For lightweight long-thinking adoption telemetry from task-graph JSONL:

```python
from llm_client import get_background_mode_adoption

summary = get_background_mode_adoption(
    experiments_path="~/projects/data/task_graph/experiments.jsonl",
    run_id_prefix="nightly_",
)
print(summary["background_mode_rate_among_reasoning"])
```

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
