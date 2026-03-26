# llm_client Requirements

## What It Is

Runtime substrate for LLM execution. Every LLM call across every project flows
through `llm_client`, which enforces observability, cost tracking, and policy
compliance automatically.

Built on [LiteLLM](https://github.com/BerriAI/litellm) for multi-provider
transport. Routing defaults to OpenRouter.

## What It Is NOT

- **Not a workflow engine** — no DAGs, no pipelines, no orchestration.
- **Not an agent framework** — no planning, no memory, no tool selection.
- **Not a prompt library** — prompts live in consumer projects as YAML/Jinja2
  templates, loaded via `render_prompt()`.

## Core API Contract

Sixteen functions: 8 sync + 8 async.

| Function | Async Variant | Returns | Purpose |
|----------|---------------|---------|---------|
| `call_llm` | `acall_llm` | `LLMCallResult` | Text completion |
| `call_llm_structured` | `acall_llm_structured` | `(T, LLMCallResult)` | Pydantic extraction |
| `call_llm_with_tools` | `acall_llm_with_tools` | `LLMCallResult` | Tool/function calling |
| `call_llm_batch` | `acall_llm_batch` | `list[LLMCallResult]` | Concurrent batch |
| `call_llm_structured_batch` | `acall_llm_structured_batch` | `list[(T, LLMCallResult)]` | Structured batch |
| `stream_llm` | `astream_llm` | `LLMStream` | Streaming |
| `stream_llm_with_tools` | `astream_llm_with_tools` | `LLMStream` | Streaming + tools |
| `embed` | `aembed` | `EmbeddingResult` | Embeddings |

### Required kwargs on every call

```python
call_llm(model, messages,
    task="what_this_call_does",       # required — logged for cost attribution
    trace_id="project/operation",     # required — groups related calls
    max_budget=1.00,                  # required — per-call spend cap in USD
)
```

No call may omit these. Enforcement is configurable via `LLM_CLIENT_REQUIRE_TAGS`.

### Result contract

Every call returns `LLMCallResult` with: `.content`, `.usage`, `.cost`,
`.marginal_cost`, `.model`, `.tool_calls`, `.finish_reason`, `.routing_trace`,
`.cache_hit`.

### Structured output

Always use `json_schema` response format (never `json_object`). Use Pydantic
`Field(description=...)` on every field. Parse permissively with a separate
model (`extra="ignore"`, defaults).

## Model Registry

Task-based model selection via `get_model(task)` and `list_models(task)`.

Task profiles: `extraction`, `budget_extraction`, `graph_building`,
`fast_extraction`, `bulk_cheap`, `synthesis`, `deep_review`,
`code_generation`, `judging`, `agent_reasoning`.

Default for bulk/fast work: `gemini/gemini-2.5-flash-lite`.

## Observability

- **JSONL logs** — every call appended to per-project log files.
- **SQLite** — `~/projects/data/llm_observability.db` with cost, tokens,
  latency, errors, traces.
- **Cost CLI** — `python -m llm_client cost --group-by project --days 7`.
- **Trace rollup** — `python -m llm_client traces` groups calls by trace_id.
- **Tool call logging** — non-LLM tool calls via `log_tool_call()`.

Query the observability DB for real costs. Never estimate.

## Consumers

Projects that depend on `llm_client` (via `pip install -e ~/projects/llm_client`):

- **research_v3** — KG-driven OSINT platform
- **grounded-research** — adjudication layer
- **prompt_eval** — prompt evaluation and optimization
- **agentic_scaffolding** — safety patterns library
- **onto-canon** — ontology canonicalization
- **Digimon_for_KG_application** — composable operator GraphRAG
