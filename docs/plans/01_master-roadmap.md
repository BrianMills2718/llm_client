# Plan 01: LLM Client Master Roadmap

**Status:** Complete
**Type:** program
**Priority:** Highest
**Blocked By:** None
**Blocks:** all slice-level execution clarity in this repo

---

## Gap

**Current:** `llm_client` has multiple good plan documents, but they are split
by subsystem. That makes it too easy for work to devolve into a loop of
"complete one slice, report, ask what next" even when the next unblocked slice
is already obvious from the existing plans.

**Target:** one canonical roadmap states:

1. the long-term programs,
2. the success criteria for each program,
3. the current order of execution,
4. the rule that agents keep going until a program is done or a real blocker
   appears.

**Why:** the repo needs one control surface that answers "what next?" without
re-planning from scratch after every passing slice.

---

## Canonical Execution Rule

This roadmap is the default execution contract for work inside `llm_client`.

Agents working in this repo must:

1. anchor every implementation slice to this roadmap and one child plan,
2. define pass/fail criteria before editing code,
3. keep executing consecutive unblocked slices after each passing checkpoint,
4. stop only for:
   - a real blocker,
   - a user-requested reprioritization,
   - a high-leverage architecture decision that cannot be resolved from repo
     context,
5. update the roadmap and child plan when the next default slice changes.

Passing one thin slice is not, by itself, a reason to stop.

---

## Repo-Level Definition Of Done

The long-term `llm_client` program is done only when all of the following are
true:

1. `llm_client` is truthfully and consistently described as a runtime
   substrate/control plane rather than a thin wrapper.
2. Core substrate boundaries are explicit: call boundary, observability,
   budgets, prompt identity, and agent SDK routing.
3. Optional runtimes are isolated enough that core code does not depend on
   their private entrypoints or private helper internals.
4. Static model policy is auditable data and empirical model policy is a
   separable overlay.
5. Workflow ambition is held behind a separate LangGraph-backed layer rather
   than grown inside `task_graph`.
6. Eval/review helpers are either explicitly optional or moved behind a clearer
   boundary.

---

## Program Order

### Program A: Runtime Boundary Hardening

**Plan:** [02_client-boundary-hardening.md](./02_client-boundary-hardening.md)  
**Status:** Complete

**Success criteria:**

- `client.py` is materially smaller and no longer mixes unrelated concerns
- public substrate APIs stay stable
- optional agent/runtime code no longer leaks through private imports into core
  runtime paths
- package/docs/public surface reflect the real substrate boundary

**Completed to date:**

- pre-call, timeout, metadata, and result-finalization seams extracted
- text and structured runtimes split out of `client.py`
- public-surface audit and low-risk deprecation pilots completed
- first optional-runtime isolation slices completed

### Program B: Model Policy Modernization

**Plan:** [03_model-policy-modernization.md](./03_model-policy-modernization.md)  
**Status:** Complete

**Success criteria:**

- built-in registry/task policy is packaged data, not embedded literals
- static policy and empirical overlay are explicit, separable layers
- default behavior is parity-tested before any ranking changes
- the role of `difficulty.py` is made explicit

**Completed to date:**

- packaged default model registry extracted and parity-tested
- static candidate selection path made explicit before empirical demotion
- performance overlay made explicit and inspectable without changing current
  selection semantics
- `difficulty.py` status clarified as a frozen compatibility-guidance layer for
  `task_graph` and analyzer logic, not a second primary policy system

### Program C: Workflow Layer Boundary

**Plan:** [04_workflow-layer-boundary.md](./04_workflow-layer-boundary.md)  
**Status:** Complete

**Success criteria:**

- durable workflow requirements are proven in a LangGraph-backed layer
- `task_graph` does not absorb durable workflow features during substrate work

**Execution rule:** do not start this program until Program A has no active
boundary blockers.

### Program D: Eval Boundary Cleanup

**Plan:** [05_eval-boundary-cleanup.md](./05_eval-boundary-cleanup.md)  
**Status:** Complete

**Success criteria:**

- shared observability remains in `llm_client`
- eval helpers stop looking like equal peers of transport/runtime substrate

**Completed to date:**

- top-level eval-root re-exports were already deprecated in favor of module
  namespaces
- shared outcome/adoption summary bookkeeping now lives in
  `llm_client.experiment_summary`
- core observability no longer imports `llm_client.experiment_eval` just to
  compute run summaries

**Execution rule:** do not start this program until Programs A and B are
stable enough that package-boundary churn is low.

### Program E: Simplification and Observability Modernization

**Plan:** [06_simplification-and-observability.md](./06_simplification-and-observability.md)
**Status:** In Progress

**Success criteria:**

- no module exceeds ~1,200 lines (soft) / 1,500 lines (hard)
- each extracted module has a single clear responsibility
- Langfuse callback available when configured, invisible when not
- shared replay/divergence diagnosis exists for call-level operational mismatches
- JSONL logs rotate by date or size
- model registry inspectable via CLI

**Evidence:** strategic review conducted 2026-03-18, confirming:

- mega-file density is a maintainability risk (not redundancy with LiteLLM)
- retry/fallback, structured output routing, budget enforcement are genuinely
  additive — NOT redundant wrapper code
- observability JSONL+SQLite will hit scaling wall; Langfuse callback is
  complementary (not replacement)
- MCP agent loop capabilities are unique and cannot be replaced by PydanticAI
- live-vs-proxy debugging pressure on 2026-03-22 showed a missing shared
  observability capability: controlled replay and divergence diagnosis
- that replay/divergence capability is now implemented and proved on a real
  `onto-canon6` mismatch case
- the generated browser API reference is now code-derived and guarded by a
  `--check` pipeline in pre-commit

---

## Current Default Next Step

Program E (Simplification and Observability Modernization) is the active
program. The replay/divergence slice in
[09_replay-and-divergence-diagnosis.md](./09_replay-and-divergence-diagnosis.md)
and the API reference slice in
[10_api-reference-generation-pipeline.md](./10_api-reference-generation-pipeline.md)
are complete. The module-size reduction slice in
[11_program-e-module-size-reduction.md](./11_program-e-module-size-reduction.md)
has now completed three verified `io_log.py` extractions
(`interventions.py`, `context.py`, then compatibility-facade reduction),
reducing `io_log.py` from `2102` lines to `1222` and clearing that module from
the hard-threshold blocker set. The default next slice remains Plan 11, with
the next tranche having cleared `agents_codex.py` below the hard threshold via
`agents_codex_process.py` and `agents_codex_runtime.py`, reducing it from
`1931` to `1317`. `mcp_agent.py` was then reduced from `3335` lines to `1037`
by extracting the turn-execution implementation into
`mcp_turn_execution.py`, so the next default Plan 11 slice is the
`mcp_turn_execution.py` follow-on decomposition. The first cleanup pass has
already removed duplicate runtime facades there, the next verified slice split
end-of-run bookkeeping into `mcp_loop_summary.py`, and the latest verified
slice extracted per-turn tool processing into `mcp_turn_tools.py`, reducing
`mcp_turn_execution.py` from `3202` to `2105`, and the latest verified slice
extracted the post-tool outcome path into `mcp_turn_outcomes.py`, reducing it
again to `1877`; the next verified slice then extracted the post-loop
completion handoff into `mcp_turn_completion.py`, reducing it to `1800`. The
next tranche now targets the pre-call tool-surface/disclosure/LLM-dispatch
block before Program E turns to `client.py`. Stop only for a real blocker or
user redirect.
