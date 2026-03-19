# Ecosystem Top-Down Architecture

This document expresses the target cross-project architecture using the planning
method:

1. define requirements,
2. define boundaries,
3. define the domain model,
4. define contracts and failure semantics,
5. derive schema and APIs,
6. prove the design on the smallest real slice.

It is not an ADR. ADRs record stable decisions. This document is the compact
top-down specification that explains how those decisions fit together.

## 1. Requirements

### Core Requirement

Any project or coding agent should be able to declare:

- which LLM or embedding capability it needs,
- which prompts and evaluators it wants to use,
- which outcomes it wants to track,

and then rely on shared infrastructure for execution, observability,
provenance, and comparison across projects.

### Functional Requirements

1. One shared execution substrate for text generation, structured output, tool
   calling, agent runtime integration, and embeddings.
2. One shared observability backend for calls, runs, items, scores, costs,
   traces, and provenance.
3. Prompt assets and rubrics are treated as data, not embedded source strings.
4. Prompt evaluation and optimization are available as first-class workflows.
5. Reusable datasets, artifacts, and schemas can be referenced consistently
   across projects.
6. The system supports cross-project comparisons, not just per-project logs.
7. Existing libraries should be wrapped where they already solve the hard part.

### Quality Requirements

1. Fail loud.
   Missing required metadata, invalid prompt references, and inconsistent
   lineage should raise errors rather than degrade silently.
2. Maximum observability.
   Every execution path that matters should emit enough metadata to reconstruct
   what happened and why.
3. Explicit provenance.
   Every meaningful run should answer:
   - what was executed,
   - on which data,
   - under which configuration,
   - with which prompt assets,
   - producing which outputs and scores.
4. Agent readability.
   The source of truth for prompts, runs, and artifacts must be easy for coding
   agents to inspect without relying on hidden overrides or local convention.
5. Simplest thing that works.
   Shared infrastructure should remove plumbing, not create a bespoke platform
   where commodity libraries would suffice.

### Success Criteria

The architecture is succeeding when:

1. a project can switch models, providers, or SDK transports without rewriting
   business logic,
2. `prompt_eval` experiments show up in the same observability backend as other
   project runs,
3. agents can compare runs across projects using shared identifiers and
   provenance,
4. prompt variants, datasets, and artifacts are explicit enough to audit and
   reuse,
5. orchestration and policy concerns do not bloat the execution core.

### Failure Criteria

The architecture is failing when:

1. multiple packages recreate their own primary observability stores,
2. prompt identity depends on implicit override order,
3. datasets or artifacts cannot be referenced consistently across projects,
4. orchestration logic leaks into the runtime core until boundaries disappear,
5. policy and routing logic are hand-maintained beyond the point where wrapped
   libraries would be simpler and more robust.

## 2. Boundaries

### A. Runtime Substrate: `llm_client`

Owns:

- provider and SDK dispatch,
- structured output,
- tool calling,
- agent runtime integration,
- embeddings,
- prompt rendering,
- observability,
- shared experiment/run persistence,
- cost and trace capture.

Does not own:

- prompt-specific optimization semantics,
- the canonical shared prompt registry,
- the entire shared data plane,
- a bespoke durable workflow runtime unless a real requirement proves it.

### B. Prompt Evaluation Layer: `prompt_eval`

Owns:

- `Experiment`,
- `PromptVariant`,
- evaluators and judges,
- prompt-search and optimization strategies,
- prompt-eval-specific MCP tooling,
- prompt-eval aggregate artifacts such as per-variant summaries.

Consumes:

- `llm_client` execution,
- `llm_client` observability.

Does not own:

- the primary execution backend,
- the primary cross-project observability backend,
- the canonical prompt registry.

### C. Prompt Asset Layer

Owns:

- reusable prompt assets,
- prompt versions,
- prompt lineage,
- prompt metadata and tags,
- prompt schema references.

Design rule:

- one prompt asset has one explicit identity,
- customization creates a new asset with lineage,
- hidden overrides are not the default architecture.

### D. Shared Data Plane

Owns:

- dataset identity,
- artifact identity,
- schema registry,
- lineage metadata,
- storage adapters for canonical shared data.

Does not own:

- LLM execution,
- prompt optimization logic,
- project-specific transient caches unless promoted into shared canonical data.

### E. Workflow and Orchestration Layer

Owns:

- multi-step execution plans,
- checkpoints and resumes if needed,
- long-running graph semantics,
- optional human-in-the-loop workflow state.

Initial rule:

- keep the local DAG layer simple,
- wrap a workflow runtime such as LangGraph if durability needs become real,
- do not grow `llm_client` into a custom workflow engine by default.

### F. Project Repos

Own:

- project logic,
- domain-specific adapters,
- project config,
- project-specific derived views and temporary materializations,
- project-specific evaluators where needed.

Do not own:

- their own ad hoc LLM calling stack,
- their own primary observability stack if the run belongs in the shared
  ecosystem.

## 3. Domain Model

### Primary Objects

1. `PromptAsset`
   - explicit prompt identity,
   - versioned,
   - optionally derived from another prompt asset.

2. `RubricAsset`
   - explicit evaluation rubric or scoring definition,
   - reusable across projects.

3. `Dataset`
   - a named, versioned collection of evaluation or retrieval inputs.

4. `Artifact`
   - a named output or intermediate object,
   - linked to datasets, runs, and parent artifacts by lineage.

5. `ExperimentFamily`
   - a conceptual evaluation campaign that may emit multiple runs.

6. `Run`
   - one execution condition over a dataset slice under one model/config.

7. `Condition`
   - the variant or arm being tested.

8. `Replicate`
   - one repeated execution of the same condition.

9. `Item`
   - one input case inside a run.

10. `Score`
    - one item-level or aggregate metric output.

11. `ProvenanceRecord`
    - the metadata that ties prompts, configs, datasets, artifacts, code state,
      and outputs together.

### Key Relationships

1. A `PromptAsset` may derive from another `PromptAsset`.
2. A `Dataset` contains many logical items.
3. An `ExperimentFamily` emits many `Run`s.
4. A `Run` belongs to one `Condition` and one `Replicate`.
5. A `Run` contains many `Item` results.
6. An `Artifact` may be produced by a `Run` and derived from another
   `Artifact`.
7. A `Score` may attach to an item or to an aggregate family-level summary.
8. A `ProvenanceRecord` binds runs to prompts, datasets, artifacts, config, and
   code state.

### Important Specialization

For `prompt_eval`:

- `Experiment` = `ExperimentFamily`
- `PromptVariant` = `Condition`
- `run_idx` = `Replicate`
- `ExperimentInput` = `Item`
- `Trial` = item result inside a run family
- `VariantSummary` = aggregate view derived across multiple runs

### Policies

1. Prompt identity is explicit.
2. Shared run identity is explicit.
3. Dataset and artifact identity are explicit.
4. Hidden runtime override behavior is discouraged.
5. Compatibility shims are transitional, not the target model.

## 4. Contracts and Failure Semantics

### Execution Contract

Callers provide:

- task intent,
- trace identity,
- budget,
- prompt or prompt reference,
- model selection intent,
- optional structured output contract.

The runtime guarantees:

- execution through the shared substrate,
- cost and trace capture,
- deterministic metadata emission for observability,
- no silent fallback on failure-critical paths.

Failure semantics:

- invalid prompt asset reference: error,
- invalid structured output: error,
- exhausted budget: error,
- unsupported provider or routing path: error,
- missing required task metadata: error.

### Observability Contract

Every meaningful run should emit:

- run start metadata,
- item-level results,
- run finish summary,
- trace links back to low-level calls,
- provenance linking code state, prompt identity, and datasets.

Failure semantics:

- observability should never silently invent substitute metadata,
- required identifiers must be present or the call should fail,
- optional sinks may fail without breaking the workload only when the canonical
  sink still succeeds.

### Prompt Asset Contract

Prompt rendering accepts:

- explicit prompt reference, or
- explicit path for compatibility.

Prompt rendering guarantees:

- deterministic resolution,
- explicit asset provenance when available.

Failure semantics:

- ambiguous prompt identity: error,
- missing prompt reference: error,
- malformed prompt template: error,
- hidden override collision: not allowed by design.

### Run and Evaluation Contract

For prompt evaluation:

- one `ExperimentFamily` emits multiple shared runs,
- one `Condition x Replicate` maps to one shared run,
- item IDs must remain stable across conditions.

Failure semantics:

- inconsistent item IDs across comparable conditions: error,
- missing evaluator metadata when claimed: error,
- summary aggregates derived from incomparable runs: error.

### Data and Provenance Contract

Runs may reference datasets and artifacts by ID rather than inlining every raw
payload.

Failure semantics:

- missing referenced dataset or artifact when required for interpretation:
  error,
- lineage cycle or invalid parent reference: error,
- inconsistent schema version metadata: error.

## 5. Derived Schema and APIs

### Shared Run Schema

Run-level fields should support:

- `run_id`
- `project`
- `dataset`
- `model`
- `condition_id`
- `seed`
- `replicate`
- `scenario_id`
- `phase`
- `metrics_schema`
- `config`
- `provenance`
- `status`
- aggregate summary metrics and cost/timing fields

Item-level fields should support:

- `item_id`
- `metrics`
- `predicted`
- `gold`
- `latency_s`
- `cost`
- `error`
- `trace_id`
- `extra`

### Prompt Asset Schema

Prompt asset metadata should minimally support:

- `prompt_id`
- `version`
- `namespace`
- `derived_from`
- `template_uri`
- `input_schema`
- `output_schema`
- `tags`
- `status`

### Data Plane Schema

The shared data layer should minimally support:

- `dataset_id`
- `artifact_id`
- `kind`
- `uri`
- `content_hash`
- `schema_name`
- `schema_version`
- `parent_artifact_id`
- `run_id`
- `project`
- free-form metadata

### Package-Level API Shape

1. `llm_client`
   - call and embed APIs,
   - prompt rendering,
   - observability APIs,
   - query/compare APIs.
2. `prompt_eval`
   - experiment definition,
   - evaluators,
   - optimization strategies,
   - prompt-eval aggregate reporting,
   - adapters that emit shared runs into `llm_client`.
3. prompt asset layer
   - asset lookup,
   - version resolution,
   - lineage lookup,
   - render helpers.
4. shared data layer
   - dataset lookup,
   - artifact registration,
   - lineage query,
   - storage adapter selection.

### Library Reuse Policy

Derived technical strategy:

- use LiteLLM for commodity provider routing and normalization where practical,
- use `llm_client` as the integration and policy surface above it,
- use LangGraph or an equivalent runtime if workflow durability requirements
  exceed the simple local DAG layer.

## 6. Smallest Real Slice

### Proving Slice

Prove the architecture by integrating `prompt_eval` with shared
`llm_client` observability before building broader platform pieces.

### Concrete Implementation Slice

1. Keep current `prompt_eval` JSON artifacts for compatibility.
2. Add dual-write from `prompt_eval` into `llm_client` run APIs.
3. Emit one shared run per `PromptVariant x Replicate`.
4. Emit one shared item per `ExperimentInput.id`.
5. Record prompt identity metadata in provenance when available.

### Acceptance Criteria

1. A `prompt_eval` experiment with `V` variants and `R` repeats emits `V * R`
   shared runs.
2. Shared runs preserve:
   - `condition_id = PromptVariant.name`
   - `replicate = run_idx`
   - stable item IDs across conditions.
3. The resulting shared runs can be queried with `get_runs()`,
   `compare_runs()`, and `compare_cohorts()`.
4. Local JSON artifacts still exist so current workflows do not break.
5. No new hidden prompt-override behavior is introduced.

### What Not To Build First

Do not start with:

1. a full shared prompt registry implementation,
2. a full shared data platform,
3. a repo-topology merger,
4. a rewrite of `task_graph`,
5. a large custom routing or workflow engine.

### Reason

If this smallest slice works, the core boundaries are probably right. If it
does not, that failure should revise the architecture before more infrastructure
is built on top of it.
