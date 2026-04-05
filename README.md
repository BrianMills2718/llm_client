# llm-client

Unified LLM client with mandatory observability, cost tracking, and policy
enforcement. Built on [LiteLLM](https://github.com/BerriAI/litellm) for
multi-provider transport.

**Why this exists:** every LLM call across every project gets logged with cost,
tokens, latency, and trace context — automatically. No call escapes without
declaring what it's for (`task`), where it fits (`trace_id`), and how much it
can spend (`max_budget`).

Repo-local ownership and boundary posture now live in
[docs/ops/CAPABILITY_DECOMPOSITION.md](docs/ops/CAPABILITY_DECOMPOSITION.md).
Use that doc when you need the current source of record for what `llm_client`
owns versus what it intentionally consumes from `prompt_eval`, `project-meta`,
or project repos.

## Governed workflow

`llm_client` now exposes sanctioned worktree coordination through its Makefile:

```bash
make worktree BRANCH=plan-22-example TASK="Describe the task" PLAN=22
make worktree-list
make worktree-remove BRANCH=plan-22-example
```

Those entrypoints should be preferred over ad hoc local worktree commands when
doing bounded implementation work in this repo.

## Install

```bash
pip install -e ~/projects/llm_client
pip install -e "~/projects/llm_client[structured]"  # + instructor for Pydantic extraction
```

## Quick start

```python
from llm_client import call_llm, get_model

# Task-based model selection (preferred)
model = get_model("extraction")
result = call_llm(
    model,
    [{"role": "user", "content": "Summarize this note"}],
    task="extraction",
    trace_id="demo/basic",
    max_budget=1.00,
)
print(result.content)
print(f"${result.cost:.4f} | {result.usage['total_tokens']} tokens")
```

### Structured output

```python
from pydantic import BaseModel
from llm_client import call_llm_structured

class Sentiment(BaseModel):
    label: str
    score: float

sentiment, meta = call_llm_structured(
    "openrouter/openai/gpt-5-mini",
    [{"role": "user", "content": "I love this product!"}],
    response_model=Sentiment,
    task="sentiment",
    trace_id="demo/sentiment",
    max_budget=1.00,
)
print(sentiment.label, sentiment.score, f"${meta.cost:.4f}")
```

### Async

```python
from llm_client import acall_llm, acall_llm_structured, acall_llm_batch

result = await acall_llm("openrouter/openai/gpt-5-mini", messages,
    task="async_demo", trace_id="demo/async", max_budget=1.00)

# Concurrent batch
results = await acall_llm_batch("openrouter/openai/gpt-5-mini", [msgs1, msgs2, msgs3],
    max_concurrent=5, task="batch_demo", trace_id="demo/batch", max_budget=2.00)
```

## Core API

Sixteen functions (8 sync + 8 async):

| Function | Async | Returns | Purpose |
|----------|-------|---------|---------|
| `call_llm` | `acall_llm` | `LLMCallResult` | Text completion |
| `call_llm_structured` | `acall_llm_structured` | `(T, LLMCallResult)` | Pydantic extraction |
| `call_llm_with_tools` | `acall_llm_with_tools` | `LLMCallResult` | Tool/function calling |
| `call_llm_batch` | `acall_llm_batch` | `list[LLMCallResult]` | Concurrent batch |
| `call_llm_structured_batch` | `acall_llm_structured_batch` | `list[(T, LLMCallResult)]` | Structured batch |
| `stream_llm` | `astream_llm` | `LLMStream` | Streaming |
| `stream_llm_with_tools` | `astream_llm_with_tools` | `LLMStream` | Streaming + tools |
| `embed` | `aembed` | `EmbeddingResult` | Embeddings |

**Required on every call:** `task=`, `trace_id=`, `max_budget=`

**Result fields:** `.content`, `.usage`, `.cost`, `.marginal_cost`, `.model`,
`.tool_calls`, `.finish_reason`, `.routing_trace`, `.cache_hit`

## Model selection

Prefer task-based selection over hardcoded model IDs:

```python
from llm_client import get_model, list_models

model = get_model("extraction")       # Best model for extraction tasks
models = list_models("extraction")    # All candidates, ranked
```

Task profiles: `extraction`, `budget_extraction`, `graph_building`,
`fast_extraction`, `bulk_cheap`, `synthesis`, `deep_review`,
`code_generation`, `judging`, `agent_reasoning`. Use `make models` or
`list_models(task)` to see candidates per task.
`graph_building` (lowest cost).

## Observability

Every call is logged to JSONL + SQLite automatically.

### Cost CLI

```bash
python -m llm_client cost                          # total spend
python -m llm_client cost --group-by project       # spend per project
python -m llm_client cost --group-by model --days 7  # spend per model, last week
python -m llm_client cost --project myproject --format json
```

### Query from code

```python
from llm_client import get_cost, get_runs

cost = get_cost(project="myproject", days=7)
runs = get_runs(project="myproject")
```

### Traces

```bash
python -m llm_client traces --project myproject --days 3
```

## Configuration

### API keys

```bash
export OPENROUTER_API_KEY=sk-or-...    # Default for most models
export GEMINI_API_KEY=...              # Direct Gemini models
export ANTHROPIC_API_KEY=sk-ant-...    # Direct Anthropic
```

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_CLIENT_OPENROUTER_ROUTING` | `on` | Route through OpenRouter by default |
| `LLM_CLIENT_DATA_ROOT` | `~/projects/data` | Observability data directory |
| `LLM_CLIENT_PROJECT` | `basename(cwd)` | Project name for logging |
| `LLM_CLIENT_REQUIRE_TAGS` | off | Strict enforcement of task/trace_id/max_budget |
| `LLM_CLIENT_TIMEOUT_POLICY` | `allow` | `ban` to disable all per-call timeouts |
| `LLM_CLIENT_LOG_ENABLED` | `1` | Disable logging with `0` |
| `LLM_CLIENT_RATE_LIMIT_SHARED_ENABLED` | `1` | Enable cross-process shared provider leases |
| `LLM_CLIENT_RATE_LIMIT_SHARED_LIMITS` | provider defaults | Override cross-process provider caps as JSON |
| `LLM_CLIENT_RATE_LIMIT_COOLDOWN_FLOORS` | provider defaults | Override provider cooldown floors as JSON |
| `LLM_CLIENT_RATE_LIMIT_STATE_PATH` | `~/projects/data/llm_rate_limit_state.sqlite3` | Shared SQLite state for cooldowns and leases |

### Provider governance

- Exact `gpt-5.4` requests canonicalize to `codex/gpt-5.4` before routing policy is applied.
- Bare Gemini ids canonicalize to `gemini/<model>` before provider selection.
- Gemini shared-cap and cooldown defaults come from the typed provider-governance policy, not scattered literals.
- `result.routing_trace["provider_governance_events"]` records canonicalization decisions for click-through debugging and downstream operator tooling.

### Tool-call observability

Non-LLM tool calls (retrieval, fetch, extraction) can be logged alongside LLM calls:

```python
from llm_client import log_tool_call

log_tool_call(
    tool_name="search_entities",
    operation="retrieval",
    result_count=15,
    task="graph_building",
    trace_id="demo/tools",
)
```

```bash
python -m llm_client tools --group-by tool_name --days 7
```

### Rubric scoring

```python
from llm_client import load_rubric, score_categorical

rubric = load_rubric("extraction_quality")
score = score_categorical(rubric, {"completeness": "good", "accuracy": "excellent"})
```

### Log maintenance

```bash
python scripts/log_maintenance.py stats             # Show log sizes and date ranges
python scripts/log_maintenance.py rotate --days 7    # Compress logs older than 7 days
python scripts/log_maintenance.py cleanup --days 90  # Delete logs older than 90 days
```

## Using from another project

```bash
pip install -e ~/projects/llm_client

# Then in code:
from llm_client import call_llm
```

## Detailed guides

- [Advanced usage](docs/guides/advanced-usage.md) — retry policies, fallback models, caching, streaming, hooks, routing config
- [MCP agent contracts](docs/guides/mcp-agent-contracts.md) — tool-chain enforcement, progressive disclosure, artifact handles
- [Codex integration](docs/guides/codex-integration.md) — process isolation, transport fallback, billing modes
- [Experiment observability](docs/guides/experiment-observability.md) — runs, items, CLI, adoption gates
- [Capability decomposition](docs/ops/CAPABILITY_DECOMPOSITION.md) — repo-local ownership and boundary source of record
- [API reference](docs/API_REFERENCE.md) — full generated reference
- [Architecture decisions](docs/adr/) — ADRs for routing, identity, observability boundaries
