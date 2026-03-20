# ADR 0010: Cross-Project Runtime and Observability Substrate

Status: Accepted  
Date: 2026-03-17
Last verified: 2026-03-19 (governed-repo friction telemetry now imports into the shared substrate)

## Context

The intended direction for the project is that any application, coding agent, or
research workflow can point at `llm_client` for LLM and embedding work and get
standardized execution, packaging, observability, and experiment recording by
default.

At the same time, adjacent repos such as `prompt_eval` already provide
higher-level capabilities like prompt comparison, evaluators, and optimization.
That created recurring confusion about which package owns:

1. the shared execution substrate,
2. the shared observability and experiment store,
3. prompt-specific evaluation semantics,
4. workflow orchestration.

We also want to avoid recreating commodity routing and workflow machinery that
existing libraries already solve well.

## Decision

1. `llm_client` is the mandatory cross-project substrate for LLM and embedding
   execution.
2. `llm_client` owns the generic runtime surfaces that many projects share:
   - provider and SDK dispatch,
   - structured output,
   - tool calling and agent runtime integration,
   - embeddings,
   - prompt rendering,
   - cost, latency, and trace capture,
   - shared run and event persistence.
3. `llm_client` owns the authoritative shared observability backend for
   cross-project work. JSONL and SQLite are the current sinks; the storage
   backend may evolve later without changing this ownership boundary.
   Imported governed-repo hook telemetry is part of this shared observability
   backend rather than a separate repo-local analytics stack.
4. The shared experiment envelope belongs to `llm_client`, including fields such
   as `project`, `dataset`, `condition_id`, `scenario_id`, `phase`, `seed`,
   `replicate`, `metrics_schema`, `config`, `provenance`, and per-item
   `metrics`/`extra`.
5. Higher-level packages such as `prompt_eval` consume this substrate rather
   than creating separate primary execution or observability stacks.
6. Commodity routing and normalization should be wrapped instead of recreated.
   Current preference:
   - use LiteLLM for provider normalization and routing where practical,
   - use LangGraph or an equivalent workflow runtime if durable orchestration
     requirements outgrow the simple local DAG layer.
7. Workflow orchestration is above the core client boundary. `task_graph` may
   remain as a simple orchestrator, but `llm_client` should not turn into a
   bespoke general-purpose workflow engine.

## Consequences

Positive:
1. A single place to standardize execution, cost tracking, and run metadata
   across projects.
2. Cleaner separation between generic runtime infrastructure and prompt-specific
   evaluation logic.
3. Lower risk of duplicating provider routing, retry, and observability code in
   every project.
4. Clearer strategy for reusing existing libraries instead of rebuilding them.

Negative:
1. `llm_client` remains a broad dependency and needs tighter module boundaries.
2. Some current behavior is transitional, especially where other packages still
   keep their own local result stores.
3. Future contributors must distinguish shared substrate features from
   higher-level product features instead of adding everything into one layer.

## Testing Contract

1. Core execution tests must continue to prove that shared runtime surfaces work
   across multiple task types, not just prompt-eval use cases.
2. Observability tests must continue to prove that run/event storage remains a
   shared facility rather than a prompt-specific one.
3. Integration work in higher-level packages should verify that they can depend
   on `llm_client` without recreating primary execution or analytics backends.
