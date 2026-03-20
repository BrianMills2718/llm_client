# llm-client

`llm_client` is a runtime substrate and control plane for multi-agent LLM
work. It uses [LiteLLM](https://github.com/BerriAI/litellm) for commodity
provider transport, but its main value is the shared execution contract on top:
mandatory call metadata, shared observability, prompt assets, structured
output, agent SDK routing, retries/fallbacks, and cross-project provenance.

## What it is

- one public API for text, structured output, tools, streaming, batch, and embeddings
- one shared observability layer for cost, traces, runs, items, and aggregates
- one place to enforce `task=`, `trace_id=`, and `max_budget=` on project code
- one place to render prompt assets and attach prompt provenance to runs
- one place to route both provider APIs and agent SDK models

## What it is not

- not just a thin transport shim
- not a requirement to use every optional subsystem
- not a reason to hand-roll direct LiteLLM calls in project repos

## Release notes

- Package metadata, including the current version, is declared in
  `pyproject.toml`.
- Latest changes are tracked in `CHANGELOG.md`.

## Install

```bash
pip install -e ~/projects/llm_client
pip install -e "~/projects/llm_client[structured]"  # + instructor for Pydantic extraction
pip install -e "~/projects/llm_client[workflow]"    # + LangGraph for optional checkpoint/resume workflows
```

## Usage

Every real project call should pass:

- `task=`
- `trace_id=`
- `max_budget=`

Prefer task-based model selection for normal project code. Use raw model IDs as
explicit overrides or transport demos, not as the default integration pattern.

```python
from llm_client import call_llm, get_model

model = get_model("graph_building")
result = call_llm(
    model,
    [{"role": "user", "content": "Summarize this note"}],
    task="graph_building",
    trace_id="readme/basic",
    max_budget=1.00,
)

# Explicit raw-model override when you want to pin transport/provider behavior
override_result = call_llm(
    "anthropic/claude-sonnet-4-5-20250929",
    [{"role": "user", "content": "Summarize this note"}],
    task="readme/provider_override",
    trace_id="readme/provider_override",
    max_budget=1.00,
)

# Agent SDK models default to subscription accounting (no per-call API USD)
agent_result = call_llm(
    "claude-code",
    [{"role": "user", "content": "Refactor this"}],
    task="dev",
    trace_id="readme/agent",
    max_budget=0,
)

print(result.content)
print(result.cost)
print(result.usage)
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

### Task-based model selection (preferred)

For most projects, do not hardcode a raw model ID in source or default config.
Resolve from the shared task registry instead:

```python
from llm_client import (
    call_llm_structured,
    resolve_model_chain,
    resolve_model_selection,
    strict_model_policy,
)

selection = resolve_model_chain(
    "extraction",
    fallback_tasks=["budget_extraction"],
    strict_models=True,
)

with strict_model_policy(selection.primary.strict_models):
    payload, meta = call_llm_structured(
        selection.primary.model,
        messages,
        response_model=MySchema,
        fallback_models=selection.fallback_models,
        task="extraction",
        trace_id="readme/extraction",
        max_budget=2.00,
    )
```

This gives you:
- shared task-based defaults from `get_model(task)`
- optional task-based fallback chains
- one explicit escape hatch for benchmark overrides
- deprecated-model blocking in strict lanes

Task guidance:
- `extraction`: highest-quality structured extraction, cost secondary
- `budget_extraction`: cheaper structured extraction with a minimum quality floor
- `graph_building`: lowest-cost structured graph-building default

To audit a repo for raw model literals and direct-call bypasses:

```bash
python ~/projects/llm_client/scripts/check_model_policy.py /path/to/repo --strict
```

### Typed routing/config

For deterministic behavior across environments, pass explicit `ClientConfig`:

```python
from llm_client import ClientConfig, call_llm

cfg = ClientConfig(
    routing_policy="direct",              # or "openrouter"
)

result = call_llm(
    "gpt-5-mini",
    messages,
    config=cfg,
    task="readme/typed_config",
    trace_id="readme/typed_config",
    max_budget=1.00,
)
```

Even with explicit `ClientConfig`, project code should still pass explicit
`task`, `trace_id`, and `max_budget` at call sites.

### Long-thinking mode (`gpt-5.2-pro`)

`gpt-5.2-pro` supports long-thinking runs. When you set
`reasoning_effort="high"` or `"xhigh"`, the client automatically enables
Responses background mode and polls to completion.

```python
result = call_llm(
    "gpt-5.2-pro",
    messages,
    reasoning_effort="xhigh",
    task="readme/long_thinking",
    trace_id="readme/long_thinking",
    max_budget=5.00,
    background_timeout=900,        # optional, seconds (default 900)
    background_poll_interval=15,   # optional, seconds (default 15)
)
```

Notes:
- Requires provider key for the active endpoint:
  - OpenRouter endpoint (`https://openrouter.ai/api/v1`): `OPENROUTER_API_KEY`
  - OpenAI endpoint (`https://api.openai.com/v1`): `OPENAI_API_KEY`
- Background retrieval supports OpenAI and OpenRouter API bases.
- Unsupported endpoint/key failures raise `LLMConfigurationError` with stable
  `error_code` values:
  - `LLMC_ERR_BACKGROUND_ENDPOINT_UNSUPPORTED`
  - `LLMC_ERR_BACKGROUND_OPENAI_KEY_REQUIRED`
  - `LLMC_ERR_BACKGROUND_OPENROUTER_KEY_REQUIRED`
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
    task="readme/sentiment",
    trace_id="readme/sentiment",
    max_budget=1.00,
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

For fully headless agent runs, `yolo_mode=True` is a convenience flag:

- for `codex`, it defaults to `approval_policy="never"` and
  `skip_git_repo_check=True`
- for `claude-code`, it defaults to `permission_mode="bypassPermissions"`

Explicit kwargs still override these defaults when you need a more specific
setup.

### Codex process isolation (hang containment)

For `codex` calls that occasionally become cancellation-unresponsive under long
tool loops, you can run non-streaming turns in a dedicated worker process:

```python
result = call_llm(
    "codex/gpt-5",
    messages,
    execution_mode="workspace_agent",
    task="readme/codex_isolation",
    trace_id="readme/codex_isolation",
    max_budget=5.00,
    codex_process_isolation=True,
    codex_process_start_method="fork",   # optional
    codex_process_grace_s=3.0,           # optional
)
```

Environment defaults are also supported:

```bash
export LLM_CLIENT_CODEX_PROCESS_ISOLATION=1
export LLM_CLIENT_CODEX_PROCESS_START_METHOD=fork
export LLM_CLIENT_CODEX_PROCESS_GRACE_S=3.0
```

When enabled, `llm_client` can hard-terminate the worker process if the SDK
turn does not honor cancellation within the timeout window. Timeout/error
diagnostics remain surfaced in `CODEX_TIMEOUT[...]` messages.

### Codex transport fallback

Codex calls support three transport modes:

- `codex_transport="sdk"`: use the SDK only.
- `codex_transport="cli"`: use `codex exec` directly.
- `codex_transport="auto"`: prefer SDK, but fall back to CLI on SDK failure.

If provider request timeouts are globally disabled with
`LLM_CLIENT_TIMEOUT_POLICY=ban`, pair auto transport with
`agent_hard_timeout`:

```python
result = call_llm(
    "codex",
    messages,
    execution_mode="workspace_agent",
    task="readme/codex_transport",
    trace_id="readme/codex_transport",
    max_budget=5.00,
    codex_transport="auto",
    agent_hard_timeout=300,
    model_reasoning_effort="medium",
)
```

That lets `llm_client` choose the CLI lane immediately instead of waiting on an
SDK timeout that cannot fire in the current policy mode. When the CLI lane is
used, `model_reasoning_effort` is forwarded to `codex exec` via config
overrides so code-generation tasks can trade speed for depth explicitly.

Codex reasoning effort note:
- `model_reasoning_effort=minimal` is often rejected on ChatGPT-account Codex
  lanes due platform tool constraints. `llm_client` coerces `minimal -> low`
  by default to avoid deterministic failures.
- Set `LLM_CLIENT_CODEX_ALLOW_MINIMAL_EFFORT=1` to force minimal unchanged.

### Observability tags

- Pass `task`, `trace_id`, and `max_budget` on every project call.
- Local non-strict adhoc use can still auto-fill these values when enforcement
  is disabled, but project code, tests, and benchmarks should not rely on that.
- Strict enforcement is enabled automatically in CI and benchmark/eval-style
  tasks, or explicitly with:

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

result = call_llm_with_tools(
    "gpt-4o",
    messages,
    tools,
    task="readme/tool_calling",
    trace_id="readme/tool_calling",
    max_budget=1.00,
)
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
    task="readme/mcp_contracts",
    trace_id="readme/mcp_contracts",
    max_budget=5.00,
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
            "artifact_prereqs": "none",
            "requires_all": [{"kind": "ENTITY_SET", "ref_type": "id"}],
            "produces": [{"kind": "CHUNK_SET", "ref_type": "fulltext"}],
        },
        "chunk_get_text_by_chunk_ids": {
            "artifact_prereqs": "none",
            "produces": [{"kind": "CHUNK_SET", "ref_type": "fulltext"}],
        },
        "todo_write": {"is_control": True},
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
- There is no hidden tool-name fallback for artifact prereq bypass; if a tool should
  be callable from explicit IDs alone, declare that behavior in its contract.
- For multi-mode tools, use declarative `call_modes` instead of runtime special cases.
  Example:

```python
"chunk_get_text": {
    "requires_all": ["CHUNK_SET"],
    "call_modes": [
        {
            "name": "by_chunk_id",
            "when_arg_equals": {"mode": "by_chunk_id"},
            "artifact_prereqs": "none",
            "produces": [{"kind": "CHUNK_SET", "ref_type": "fulltext"}],
        },
        {
            "name": "by_entity_ids",
            "when_args_present_any": ["entity_ids", "entity_id"],
            "artifact_prereqs": "none",
            "produces": [{"kind": "CHUNK_SET", "ref_type": "fulltext"}],
        },
    ],
}
```

- `call_modes` are resolved per call from arguments. For best disclosure/closure behavior,
  prefer explicit split tools at the LLM boundary and reserve `call_modes` for internal or
  discriminated-union style contracts.
- Tools can also consume previously emitted typed artifact handles declaratively with
  `handle_inputs`. Example:

```python
"extract_date_mentions_from_artifacts": {
    "artifact_prereqs": "none",
    "handle_inputs": [
        {
            "arg": "chunk_artifact_ids",
            "inject_arg": "chunk_artifacts",
            "representation": "payload",
            "accepts": [{"kind": "CHUNK_SET", "ref_type": "fulltext"}],
        }
    ],
    "produces": [{"kind": "CHUNK_SET", "ref_type": "fulltext"}],
}
```

- With this contract shape, the model only passes stable `artifact_id` values in
  `chunk_artifact_ids`. The runtime resolves those handles from the active artifact
  registry and injects the matching typed artifact payloads into `chunk_artifacts`
  before executor dispatch.
- Optional eval helpers can normalize these outcome fields across runs/items:

```python
from llm_client.experiment_eval import (
    build_gate_signals,
    extract_agent_outcome,
    summarize_agent_outcomes,
)

outcome = extract_agent_outcome(item_result)
summary = summarize_agent_outcomes(run_items)
signals = build_gate_signals(run_info=run_info, items=run_items)

print(outcome["submit_completion_mode"])
print(summary["grounded_completed_rate"])
print(signals["forced_terminal_accepted_rate"])
```

Experiment run finalization also auto-merges these outcome counts/rates into stored
`summary_metrics`, so compare/cohort/reporting surfaces can use them without project-specific glue.

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
- Rolling artifact-context summaries are enabled by default for agent loops:
  - `active_artifact_context_enabled`
  - `active_artifact_context_max_handles`
  - `active_artifact_context_max_chars`
  These inject a bounded compact summary of recent typed artifact handles and
  current artifact/capability state into active context. When typed artifact
  handles exist, the runtime also exposes an explicit built-in tool,
  `runtime_artifact_read`, so the model can reopen prior typed artifacts by
  `artifact_id` after older tool payloads were cleared from active context.
- MCP loop default completion cap is `8192` tokens; override with
  `LLM_CLIENT_MCP_MAX_COMPLETION_TOKENS` (minimum applied: `1024`).
- Forced-final accepted submit answers are normalized to a short factual span
  (e.g., explicit answer/date/year) before final metadata output.

### Direct tool registry lint

For direct Python tools, `llm_client` can lint registry quality before runtime:

```python
from llm_client import lint_tool_callable, lint_tool_registry

report = lint_tool_registry(
    [search_entities, chunk_get_text_by_chunk_ids],
    tool_contracts={
        "chunk_get_text_by_chunk_ids": {
            "artifact_prereqs": "none",
            "produces": [{"kind": "CHUNK_SET", "ref_type": "fulltext"}],
        }
    },
)
print(report["n_errors"], report["n_warnings"])

findings = lint_tool_callable(search_entities, contract={"produces": ["ENTITY_SET"]})
for finding in findings:
    print(finding["severity"], finding["code"], finding["message"])
```

Lint is designed for adoption gates:
- warn on missing one-line descriptions
- warn when nontrivial tools lack input examples
- warn when nontrivial tools lack declarative contracts
- error when declared `call_modes` are structurally invalid
- error when declared `handle_inputs` reference missing args/injected args on direct tools

In strict adoption profiles, agent loops also surface these tool-quality issues as
structured adoption-profile violations during setup.

Shared CLI/CI entrypoints:

```bash
python -m llm_client tool-lint \
  --module /path/to/tools.py \
  --tool-list-var DIRECT_TOOLS \
  --contracts-var TOOL_CONTRACTS \
  --fail-on-warning

python -m llm_client experiments --detail RUN_ID \
  --require-adoption-profile strict \
  --require-adoption-satisfied \
  --adoption-gate-fail-exit-code 4
```

Stored experiment runs now auto-aggregate adoption-profile summary metrics, and
`experiments --compare`/`--detail` surfaces render those summaries alongside the
existing grounded/forced outcome metrics.

When old tool payloads are cleared from active context, the compact stub now preserves
typed artifact handles (`artifact_id`, `artifact_type`, and leading capability metadata) when available.
The runtime-provided `runtime_artifact_read` tool can consume those preserved
`artifact_id` handles directly and return the stored typed artifact envelopes.

### Batch/parallel calls

```python
from llm_client import call_llm_batch, acall_llm_batch

# Run multiple prompts concurrently (semaphore-based rate limiting)
results = call_llm_batch(
    "gpt-4o",
    [msgs1, msgs2, msgs3],
    max_concurrent=5,
    task="readme/batch_sync",
    trace_id="readme/batch_sync",
    max_budget=2.00,
)

# Async version
results = await acall_llm_batch(
    "gpt-4o",
    messages_list,
    max_concurrent=10,
    task="readme/batch_async",
    trace_id="readme/batch_async",
    max_budget=2.00,
)

# Structured batch
from llm_client import call_llm_structured_batch
results = call_llm_structured_batch(
    "gpt-4o",
    messages_list,
    response_model=Entity,
    task="readme/batch_structured",
    trace_id="readme/batch_structured",
    max_budget=2.00,
)
```

### Async

```python
from llm_client import acall_llm, acall_llm_structured, acall_llm_with_tools

result = await acall_llm(
    "gpt-4o",
    messages,
    task="readme/async_text",
    trace_id="readme/async_text",
    max_budget=1.00,
)
data, meta = await acall_llm_structured(
    "gpt-4o",
    messages,
    response_model=Entity,
    task="readme/async_structured",
    trace_id="readme/async_structured",
    max_budget=1.00,
)
result = await acall_llm_with_tools(
    "gpt-4o",
    messages,
    tools=[...],
    task="readme/async_tools",
    trace_id="readme/async_tools",
    max_budget=1.00,
)
```

### Streaming

```python
from llm_client import stream_llm, astream_llm

# Streaming with retry/fallback support
stream = stream_llm(
    "gpt-4o",
    messages,
    num_retries=2,
    fallback_models=["gpt-3.5-turbo"],
    task="readme/stream_sync",
    trace_id="readme/stream_sync",
    max_budget=1.00,
)
for chunk in stream:
    print(chunk, end="", flush=True)
print(stream.result.usage)  # usage available after stream ends

# Streaming with tools
from llm_client import stream_llm_with_tools
stream = stream_llm_with_tools(
    "gpt-4o",
    messages,
    tools=[...],
    task="readme/stream_tools",
    trace_id="readme/stream_tools",
    max_budget=1.00,
)
for chunk in stream:
    print(chunk, end="", flush=True)
print(stream.result.tool_calls)

# Async
stream = await astream_llm(
    "gpt-4o",
    messages,
    task="readme/stream_async",
    trace_id="readme/stream_async",
    max_budget=1.00,
)
async for chunk in stream:
    print(chunk, end="", flush=True)
```

### Fallback models

```python
result = call_llm(
    "gpt-4o", messages,
    fallback_models=["gpt-3.5-turbo", "ollama/llama3"],
    task="readme/fallbacks",
    trace_id="readme/fallbacks",
    max_budget=1.00,
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
result = call_llm(
    "gpt-4o",
    messages,
    hooks=hooks,
    task="readme/hooks",
    trace_id="readme/hooks",
    max_budget=1.00,
)
```

### Response caching

```python
from llm_client import LRUCache, call_llm

cache = LRUCache(maxsize=128, ttl=3600)  # thread-safe, 1h TTL
result = call_llm(
    "gpt-4o",
    messages,
    cache=cache,
    task="readme/cache",
    trace_id="readme/cache",
    max_budget=1.00,
)  # calls LLM
result = call_llm(
    "gpt-4o",
    messages,
    cache=cache,
    task="readme/cache",
    trace_id="readme/cache",
    max_budget=1.00,
)  # returns cached
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
    task="readme/agent_retry_safe",
    trace_id="readme/agent_retry_safe",
    max_budget=2.00,
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
result = call_llm(
    "gpt-4o",
    messages,
    retry=policy,
    task="readme/retry_policy",
    trace_id="readme/retry_policy",
    max_budget=1.00,
)

# Or quick one-offs with individual params
result = call_llm(
    "gpt-4o",
    messages,
    num_retries=5,
    retry_on=["custom"],
    task="readme/retry_one_off",
    trace_id="readme/retry_one_off",
    max_budget=1.00,
)
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

`call_llm`, `call_llm_structured`, `call_llm_with_tools` (and async variants) accept: `task`, `trace_id`, `max_budget`, `timeout`, `num_retries`, `reasoning_effort` (Claude models and `gpt-5.2-pro` long-thinking), `api_base`, `retry_on`, `on_retry`, `cache`, `retry` (RetryPolicy), `fallback_models`, `on_fallback`, `hooks` (Hooks), `execution_mode` (`text`/`structured`/`workspace_agent`/`workspace_tools`), plus any `**kwargs` passed through to `litellm.completion`.

`stream_llm` / `astream_llm` (and `*_with_tools` variants) accept: `task`, `trace_id`, `max_budget`, `timeout`, `num_retries`, `reasoning_effort`, `api_base`, `retry`, `fallback_models`, `on_fallback`, `hooks`, plus `**kwargs`. No `cache` param (caching streams doesn't make sense).

`*_batch` functions additionally accept: `max_concurrent` (5), `return_exceptions`, `on_item_complete`, `on_item_error`.

Timeout policy:
- Set `LLM_CLIENT_TIMEOUT_POLICY=ban` to disable all per-call request timeouts globally (timeouts are normalized to disabled and omitted from provider calls).
- Set `LLM_CLIENT_TIMEOUT_POLICY=allow` (default) to permit explicit `timeout` values.
- For long-running work, use `get_active_llm_calls()` to distinguish waiting, progressing, and idle calls before treating them as hung.

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
python -m llm_client adoption --run-id-prefix nightly_ --format table
python -m llm_client adoption --run-id-prefix nightly_ --since 2026-02-20 --min-rate 0.95 --metric among_reasoning --min-samples 20
python -m llm_client experiments --detail RUN_ID --det-checks default
python -m llm_client experiments --detail RUN_ID --review-rubric extraction_quality
python -m llm_client experiments --detail RUN_ID --gate-policy '{"pass_if":{"avg_llm_em_gte":80}}' --gate-fail-exit-code
python -m llm_client experiments --detail RUN_ID --require-adoption-profile strict --require-adoption-satisfied
python -m llm_client tool-lint --module /path/to/tools.py --tool-list-var DIRECT_TOOLS --contracts-var TOOL_CONTRACTS --fail-on-warning
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

`adoption` can enforce a local gate without GitHub Actions: when `--min-rate` is set, the command exits non-zero on failure (unless `--warn-only`). This is scheduler-friendly for cron/Jenkins/Buildkite.

You can also use the wrapper script (env-configurable defaults):

```bash
./scripts/adoption_gate.sh
```

If your gate reports `missing_reasoning_effort_dimension` (legacy records), run one
live probe to append a fresh long-thinking record:

```bash
./scripts/adoption_probe.sh
```

That probe uses `gpt-5.2-pro` + `reasoning_effort=high` by default, routes via
OpenRouter, and writes one task-graph experiment row with
routing/effort/background fields.

Cron example (daily at 06:15 UTC, log to file, no GitHub Actions):

```cron
15 6 * * * cd /home/brian/projects/llm_client && ./scripts/adoption_gate.sh >> /home/brian/projects/llm_client/adoption_gate.log 2>&1
```

## API keys

Set via environment variables (litellm convention):

```bash
export OPENROUTER_API_KEY=sk-or-...
export ANTHROPIC_API_KEY=sk-ant-...
export GEMINI_API_KEY=...
```

Use `OPENROUTER_API_KEY` by default for non-Gemini traffic. Use `GEMINI_API_KEY`
for direct Gemini models. `OPENAI_API_KEY` is only needed if you intentionally
run direct OpenAI routing (`LLM_CLIENT_OPENROUTER_ROUTING=off`).

## Using from another project

```bash
# From your other project's directory:
pip install -e ~/projects/llm_client

# Then in code:
from llm_client import call_llm
```
