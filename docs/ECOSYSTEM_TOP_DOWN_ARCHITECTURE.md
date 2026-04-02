# Ecosystem Top-Down Architecture

doc_role: architecture
mutable_facts_allowed: yes
enforcement_status: authoritative

**Purpose**: Define the layer model for Brian's project ecosystem. Maps which
capabilities belong to which layer, and where the boundaries lie. This is the
primary boundary contract for `llm_client` consumers.

**Updated**: 2026-04-02

---

## Layer Model

The ecosystem has four explicit layers:

```
┌──────────────────────────────────────────────────────────────────┐
│  Projects (research_v3, onto-canon6, Digimon, grounded-research) │
│  Task-specific application logic. Consume all layers below.      │
└──────────────────────────────────────┬───────────────────────────┘
                                       │ uses
┌──────────────────────────────────────▼───────────────────────────┐
│  Shared Capability Libraries                                      │
│  open_web_retrieval, osint_tools, data_contracts, mcp-servers    │
│  Domain-specific tools (web, government APIs, social, code).     │
└──────────────────────────────────────┬───────────────────────────┘
                                       │ uses
┌──────────────────────────────────────▼───────────────────────────┐
│  Eval Layer (prompt_eval)                                         │
│  Experiment models, evaluators, rubric scoring, benchmarks.      │
│  Consumes llm_client for LLM calls. Does NOT own LLM dispatch.   │
└──────────────────────────────────────┬───────────────────────────┘
                                       │ uses
┌──────────────────────────────────────▼───────────────────────────┐
│  Substrate Layer (llm_client)                                     │
│  LLM dispatch, structured output, embeddings, prompt rendering,  │
│  observability (JSONL + SQLite), model registry, experiment log. │
│  Every LLM call in the ecosystem goes through this layer.        │
└──────────────────────────────────────────────────────────────────┘
```

---

## Substrate Layer (`llm_client`)

**Owns**: Provider dispatch, SDK routing, structured output (json_schema),
tool calling, embeddings, streaming, batch, retry/fallback, cost tracking,
observability (JSONL + SQLite), model registry, experiment logging, prompt
asset loading (YAML/Jinja2 templates), MCP loop integration.

**Does NOT own**: Project-specific orchestration semantics, evaluation rubrics,
web retrieval, government API clients, social media fetching.

**Required kwargs on every call**: `task=`, `trace_id=`, `max_budget=`

**Boundary contract**: Consumers call `llm_client`; they do not import
provider SDKs directly. All LLM spend flows through the observability DB
at `~/projects/data/llm_observability.db`.

---

## Eval Layer (`prompt_eval`)

**Owns**: Prompt evaluation lifecycle — experiment models, prompt variants,
dataset management, evaluator dispatch, rubric scoring, statistical comparison
(bootstrap CI, Welch's test), optimization loops.

**Consumes**: `llm_client` for LLM calls (not direct provider imports).

**Does NOT own**: LLM dispatch, observability DB schema, provider SDK wrappers.

**Boundary contract**: `prompt_eval` receives structured outputs from
`llm_client`. Evaluation results feed back into experiment logs via
`llm_client` observability, not a separate DB.

---

## Workflow Layer (planned — not yet built)

**Target boundary** (see `docs/plans/04_workflow-layer-boundary.md`):

Lines 170-190 of this document define the coexistence rules:

The workflow layer will own **durable multi-step orchestration** — e.g.,
LangGraph-backed pipelines where steps persist state, can be retried, and
have explicit entry/exit contracts. It is NOT the same as `task_graph.py`
(synchronous YAML DAG runner for single-process execution).

**Coexistence rules**:
- `task_graph.py` (in project-meta) is synchronous, single-process. Use it
  for simple linear plans where each step completes before the next begins.
- A future workflow layer handles async, durable, re-entrant workflows.
- The two systems are not competing. `task_graph.py` may become a lane
  runner within OpenClaw.
- The workflow layer MUST consume `llm_client` for all LLM calls. It does
  not get its own logging or prompt-resolution stack.
- No durable workflow implementation should import provider SDKs directly.

**When a workflow layer is built**, its plan doc must specify:
1. The entry/exit contract (typed Pydantic models at boundaries)
2. What calls `llm_client` and what it expects back
3. How observability events flow to the substrate DB
4. The coexistence contract with `task_graph.py`

---

## Shared Capability Libraries

These live between the substrate and project layers:

| Library | Domain | What it owns |
|---------|--------|--------------|
| `open_web_retrieval` | Web | Search (Brave, SearxNG, Exa, Tavily), fetch, Trafilatura extraction |
| `osint_tools` | Government | USAspending, FEC APIs |
| `data_contracts` | Infrastructure | `@boundary` decorator, ContractRegistry, schema compatibility |
| `mcp-servers` | Agent interfaces | Social media (Twitter, Reddit), YouTube transcript |
| `agentic_scaffolding` | Safety | Validators, circuit breakers, governance patterns |

**Rule**: Shared capabilities are Python libraries (`pip install -e`), not MCP
servers. MCP is the optional presentation layer for interactive agent discovery
only. Structured pipelines import libraries directly.

---

## Substrate vs. Eval Layer: Key Distinctions

| Concern | Substrate (`llm_client`) | Eval Layer (`prompt_eval`) |
|---------|--------------------------|---------------------------|
| LLM dispatch | ✓ owns | Delegates to llm_client |
| Observability DB | ✓ owns schema | Reads results, does NOT own |
| Rubric scoring | Provides call infra | ✓ owns scoring logic |
| Experiment logging | ✓ stores raw call data | ✓ owns experiment aggregation |
| Prompt templates | ✓ owns YAML/Jinja2 loading | Uses prompt_eval for comparison |
| Model registry | ✓ owns | Reads, does NOT extend |

---

## Pipeline Flow

The canonical data pipeline across the ecosystem:

```
research_v3 (investigate)
    → grounded-research (adjudicate)
        → onto-canon6 (canonicalize + govern + infer)
            → Digimon (retrieve + compose + answer)
```

Each stage produces typed outputs (Pydantic models) consumed by the next.
`llm_client` is the substrate for LLM work at every stage.
`prompt_eval` benchmarks prompts and models at any stage.
`open_web_retrieval` and `osint_tools` feed `research_v3` and `grounded-research`.

---

## Boundary Violations (Anti-Patterns)

These are signals that a boundary is being crossed incorrectly:

1. **Direct provider SDK import** outside `llm_client` → move to llm_client
2. **Evaluation rubric logic** in `llm_client` → move to prompt_eval
3. **Web fetch/search code** duplicated in a project → move to open_web_retrieval
4. **Observability DB queries** in multiple projects → use `db.py` from ecosystem-ops
5. **Prompt f-strings** in project code → move to YAML templates in `llm_client/prompts/`

---

*This document is authoritative. Changes require updating the plan docs that
reference it: `docs/plans/02_client-boundary-hardening.md`,
`docs/plans/04_workflow-layer-boundary.md`, and
`docs/plans/05_eval-boundary-cleanup.md`.*
