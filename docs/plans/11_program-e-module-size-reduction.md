# Plan 11: Program E Module Size Reduction

**Status:** In Progress
**Type:** implementation
**Priority:** High
**Blocked By:** None
**Blocks:** completion of Program E in [06_simplification-and-observability.md](./06_simplification-and-observability.md)

---

## Gap

**Current:** Program E's feature slices are materially ahead of its closeout
status. Langfuse callback wiring, replay/diff tooling, JSONL rotation, and the
models CLI already exist, but the program still fails its primary
maintainability criterion because several modules remain far above the plan's
size targets.

**Target:** reduce the remaining oversized modules into clearer, narrower
submodules until Program E's size/composition criteria are honestly satisfied,
or explicitly re-scope the criteria if a module's boundary is proven sound and
the threshold is wrong.

**Why:** leaving Program E marked active without an explicit child plan would
hide the real blocker. The remaining work is not feature invention. It is
structural decomposition and truthful closeout.

---

## References Reviewed

- `docs/plans/01_master-roadmap.md`
- `docs/plans/06_simplification-and-observability.md`
- `llm_client/client.py`
- `llm_client/mcp_agent.py`
- `llm_client/io_log.py`
- `llm_client/agents_codex.py`
- `llm_client/observability/experiments.py`
- `llm_client/agent_contracts.py`

---

## Current Audit Baseline

Fresh line-count audit on 2026-03-22:

1. `llm_client/client.py`: `4184`
2. `llm_client/mcp_turn_execution.py`: `1339`
3. `llm_client/mcp_turn_model.py`: `944`
4. `llm_client/mcp_turn_tools.py`: `702`
5. `llm_client/mcp_loop_summary.py`: `522`
6. `llm_client/mcp_turn_outcomes.py`: `503`
7. `llm_client/mcp_turn_completion.py`: `369`
8. `llm_client/observability/experiments.py`: `1322`
9. `llm_client/agent_contracts.py`: `1228`
10. `llm_client/agents_codex.py`: `1317`
11. `llm_client/io_log.py`: `1222`
12. `llm_client/mcp_agent.py`: `1037`

These counts are the current audit baseline for the remaining oversize set.

---

## Files Affected

> This section declares the intended implementation surface for this plan. New
> slices must amend this section before touching additional files.

- `llm_client/client.py` (modify/extract)
- `llm_client/call_lifecycle.py` (new extracted module)
- `llm_client/call_wrappers.py` (new extracted module)
- `llm_client/background_runtime.py` (new extracted module)
- `llm_client/responses_runtime.py` (new extracted module)
- `llm_client/completion_runtime.py` (new extracted module)
- `llm_client/io_log.py` (modify/extract)
- `llm_client/observability/context.py` (new extracted module)
- `llm_client/observability/events.py` (compatibility facade wiring)
- `llm_client/observability/interventions.py` (new extracted module)
- `llm_client/mcp_turn_execution.py` (new extracted module)
- `llm_client/mcp_loop_summary.py` (new extracted module)
- `llm_client/mcp_turn_tools.py` (new extracted module)
- `llm_client/mcp_turn_outcomes.py` (new extracted module)
- `llm_client/mcp_turn_completion.py` (new extracted module)
- `llm_client/mcp_turn_model.py` (new extracted module)
- `llm_client/mcp_agent.py` (modify/extract, later slice)
- `llm_client/agents_codex.py` (modify/extract, later slice)
- `llm_client/agents_codex_process.py` (new extracted module)
- `llm_client/agents_codex_runtime.py` (new extracted module)
- `llm_client/observability/experiments.py` (modify/extract if still needed after earlier tranches)
- `llm_client/agent_contracts.py` (modify/extract if still needed after earlier tranches)
- `docs/plans/06_simplification-and-observability.md` (update evidence/status)
- `docs/plans/01_master-roadmap.md` (update default next step as slices complete)
- `docs/API_REFERENCE.md` (generated)
- `docs/API_REFERENCE.html` (generated)

---

## Definition Of Done

This plan is done only when all of the following are true:

1. each targeted module is below the Program E hard threshold or has an
   explicit documented exception;
2. each extraction creates a narrower module with one clear responsibility;
3. existing behavior stays covered by focused regression tests at each wiring
   boundary; and
4. Program E can be evaluated honestly against its size/composition criteria
   without hand-waving about "already mostly done."

---

## Plan

## Thin Slices

### Phase 1: Boundary Map And First Tranche Selection

**Purpose:** choose the first extraction tranche from the current oversized set.

**Input -> Output:** raw size audit -> one concrete first write scope with
pass/fail tests and ownership

**Passes if:**

- the first tranche is concrete enough to implement without re-planning
- the tranche names source and destination modules explicitly
- affected tests and invariants are named up front

**Fails if:**

- the tranche is just "make client.py smaller"
- multiple unrelated mega-modules are edited at once without a boundary map

**Selected first tranche (2026-03-22):**

Start with `llm_client/io_log.py`, not `client.py`.

Write scope for the first implementation slice:

1. extract the experiment-run context / enforcement surface now living in
   `io_log.py` lines `282-673` into a dedicated observability-local module
2. extract the intervention logging/query/update surface now living in
   `io_log.py` lines `1934-2065` into a dedicated observability-local module
3. keep `io_log.py` as the compatibility facade so downstream imports remain
   truthful during the refactor

Why this tranche goes first:

1. it is meaningfully smaller and less coupled than attacking `client.py`
   first
2. it reduces one of the oversized modules without destabilizing the core call
   hot path
3. it already has strong regression coverage through:
   - `tests/test_experiment_log.py`
   - `tests/test_io_log.py`
   - `tests/test_io_log_compat.py`
   - experiment CLI coverage in `tests/test_cli_experiments.py`

### Phase 2: `client.py` / `io_log.py` First Decomposition Slice

**Purpose:** attack the most central oversized runtime modules first.

**Input -> Output:** oversized mixed-responsibility modules -> extracted
submodules with clearer seams

**Passes if:**

- one coherent concern is extracted from `client.py` or `io_log.py`
- public imports and compatibility behavior remain truthful
- focused regression tests pass

**Fails if:**

- extraction only moves code without improving responsibility boundaries
- compatibility surfaces silently drift

**Verified checkpoint 1 (2026-03-22):**

The intervention storage surface was extracted from `io_log.py` into
`llm_client/observability/interventions.py`, with `io_log.py` kept as the
compatibility facade.

What this checkpoint proved:

1. the tranche is viable without touching the core call hot path
2. the extraction surfaced and fixed a real latent bug in the old
   implementation (`log_intervention()` referenced missing `_resolve_project()`)
3. focused regression coverage remained green:
   - `pytest -q tests/test_experiment_log.py tests/test_io_log_compat.py`
   - result: `68 passed`

Effect on module size:

1. `llm_client/io_log.py`: `2102 -> 2011`
2. new module: `llm_client/observability/interventions.py` (`180` lines)

**Verified checkpoint 2 (2026-03-22):**

The experiment-context / feature-profile / AgentSpec guardrail surface was
extracted from `io_log.py` into `llm_client/observability/context.py`, with
`io_log.py` and `llm_client.observability.events` kept as compatibility
facades.

What this checkpoint proved:

1. the second half of the first tranche can move without changing downstream
   imports
2. the extraction reduced `io_log.py` materially while preserving the existing
   guardrail semantics used by call-contract and experiment surfaces
3. a public-surface regression was caught during test collection
   (`ActiveFeatureProfile` still had to be exported from
   `llm_client.observability.events`) and fixed before the slice was accepted
4. focused regression coverage remained green:
   - `pytest -q tests/test_experiment_log.py tests/test_io_log_compat.py tests/test_call_contracts.py`
   - result: `71 passed`

Effect on module size:

1. `llm_client/io_log.py`: `2011 -> 1600`
2. new module: `llm_client/observability/context.py` (`429` lines)

**Selected next tranche (2026-03-22):**

Finish `io_log.py` by replacing the remaining handwritten compatibility
wrappers for query / replay / experiment / intervention surfaces with direct
re-export bindings where the import graph is cycle-safe.

Passes if:

1. `io_log.py` drops below the hard threshold without changing the public API
2. the re-exports remain honest about where concrete behavior lives
3. focused compatibility, public-surface, and `io_log` regression tests stay
   green

Fails if:

1. the aliasing introduces import-cycle regressions
2. `io_log.py` still depends on large blocks of redundant wrapper code after
   the slice

**Verified checkpoint 3 (2026-03-22):**

The remaining `io_log.py` compatibility facade was reduced by replacing most
handwritten wrappers with truthful direct re-exports, while keeping dynamic
delegation only for the monkeypatch-sensitive compatibility surfaces that the
existing tests prove are load-bearing (`start_run`, `get_cost`, and
`get_background_mode_adoption`).

What this checkpoint proved:

1. `io_log.py` no longer blocks Program E's hard-threshold criterion
2. the compatibility surface can stay truthful without preserving hundreds of
   lines of redundant wrapper code
3. dynamic delegation remains in place where historical patchability is part
   of the effective contract
4. focused regression coverage remained green:
   - `pytest -q tests/test_io_log.py tests/test_io_log_compat.py tests/test_experiment_log.py tests/test_public_surface.py tests/test_api_reference_generation.py`
   - result: `137 passed`

Effect on module size:

1. `llm_client/io_log.py`: `1600 -> 1222`

**Verified checkpoint 6 (2026-03-22):**

The `mcp_agent.py` turn-execution implementation was extracted into
`llm_client/mcp_turn_execution.py`, while `mcp_agent.py` kept the facade
imports and compatibility entry points that existing tests patch.

What this checkpoint proved:

1. the large per-turn tool-processing loop can move into a dedicated module
   without breaking the public `mcp_agent` entry points
2. monkeypatch-sensitive helpers still need to resolve through
   `llm_client.mcp_agent` so tests and downstream callers continue to intercept
   them at the facade boundary
3. the new turn-execution module is now the current oversize follow-on slice,
   so Program E remains evidence-driven rather than aspirational
4. focused regression coverage remained green:
   - `pytest -q tests/test_mcp_agent.py`
   - result: `106 passed`

Effect on module size:

1. `llm_client/mcp_agent.py`: `3335 -> 1037`
2. new module: `llm_client/mcp_turn_execution.py` (`3202` lines)

**Verified checkpoint 7 (2026-03-22):**

The accidental duplicate MCP runtime / transport facades were removed from
`llm_client/mcp_turn_execution.py`, leaving that module focused on the
extracted `_agent_loop` implementation rather than also shadowing
`mcp_agent.py`'s public runtime surface.

What this checkpoint proved:

1. the broad `_agent_loop` extraction can be made more truthful without
   changing the facade contract in `llm_client.mcp_agent`
2. duplicate runtime helpers inside the extracted module were structural debt,
   not load-bearing compatibility surface
3. the module still remains above threshold, so Program E must continue on the
   new follow-on decomposition rather than treating the checkpoint as closeout
4. focused regression coverage remained green:
   - `pytest -q tests/test_mcp_agent.py`
   - result: `106 passed`
   - `pytest -q tests/test_tool_runtime_common.py tests/test_agent_runtime_adapters.py tests/test_model_identity_contract.py`
   - result: `27 passed, 1 warning`

Effect on module size:

1. `llm_client/mcp_turn_execution.py`: `3202 -> 2711`

**Verified checkpoint 8 (2026-03-22):**

The end-of-run summary / metadata writeout was extracted from
`llm_client/mcp_turn_execution.py` into `llm_client/mcp_loop_summary.py`.

What this checkpoint proved:

1. the long metadata/failure-summary tail is a clean, test-covered boundary
   separate from turn orchestration
2. the extraction keeps `mcp_turn_execution.py` focused on control flow rather
   than final bookkeeping
3. the remaining oversize debt is still real, so the next slice must keep
   splitting `mcp_turn_execution.py` rather than moving on prematurely
4. focused regression coverage remained green:
   - `pytest -q tests/test_mcp_agent.py`
   - result: `106 passed`
   - `pytest -q tests/test_tool_runtime_common.py tests/test_agent_runtime_adapters.py tests/test_model_identity_contract.py`
   - result: `27 passed, 1 warning`

Effect on module size:

1. `llm_client/mcp_turn_execution.py`: `2711 -> 2581`
2. new module: `llm_client/mcp_loop_summary.py` (`522` lines)

**Verified checkpoint 9 (2026-03-22):**

The per-turn tool-processing path was extracted from
`llm_client/mcp_turn_execution.py` into `llm_client/mcp_turn_tools.py`.

What this checkpoint proved:

1. compliance-gate validation, tool-contract validation, control-loop
   suppression, runtime-artifact reads, external tool execution, and
   artifact/binding/capability reconciliation form a coherent boundary
   separate from turn orchestration
2. the turn orchestrator can delegate that dense mid-loop path without losing
   the existing MCP-agent behavior proved by the focused regression suites
3. the extracted helper needed one follow-on fix for the control-churn
   bookkeeping path, which confirms the value of keeping these slices small
   and verified before moving on
4. focused regression coverage remained green:
   - `pytest -q tests/test_mcp_agent.py`
   - result: `106 passed`
   - `pytest -q tests/test_tool_runtime_common.py tests/test_agent_runtime_adapters.py tests/test_model_identity_contract.py`
   - result: `27 passed, 1 warning`

Effect on module size:

1. `llm_client/mcp_turn_execution.py`: `2581 -> 2105`
2. new module: `llm_client/mcp_turn_tools.py` (`702` lines)

**Selected next tranche (2026-03-22, post-tool-processing extraction):**

Stay on `llm_client/mcp_turn_execution.py`.

Write scope for the next implementation slice:

1. extract the post-tool outcome handling block from `mcp_turn_execution.py`
   into a dedicated helper module
2. move the evidence-digest update, retrieval-stagnation policy,
   `submit_answer` bookkeeping, TODO-state injection, and control-churn
   threshold handling behind a narrower interface
3. keep `mcp_turn_execution.py` as the per-turn orchestrator that sequences
   prompt calls, tool execution, and finalization decisions

Why this tranche goes next:

1. `mcp_turn_execution.py` is still the clearest remaining hard-threshold
   blocker after the tool-processing extraction
2. the post-tool outcome block is the next largest coherent responsibility
   after the tool-processing path moved out
3. this keeps Program E reducing the active blocker instead of jumping
   prematurely to `client.py`

**Verified checkpoint 10 (2026-03-22):**

The post-tool outcome path was extracted from
`llm_client/mcp_turn_execution.py` into `llm_client/mcp_turn_outcomes.py`.

What this checkpoint proved:

1. argument-coercion bookkeeping, evidence-digest updates,
   retrieval-stagnation policy, `submit_answer` recovery, TODO-state
   injection, and control-churn threshold handling form a coherent boundary
   after tool execution completes
2. `mcp_turn_execution.py` can keep sequencing and loop nudges while
   delegating the heavier post-tool policy block to a dedicated helper
3. the active blocker is still `mcp_turn_execution.py`, but the remaining
   debt is now concentrated in turn-exit / forced-finalization handoff logic
   rather than the already-extracted mid-loop mechanics
4. focused regression coverage remained green:
   - `pytest -q tests/test_mcp_agent.py`
   - result: `106 passed`
   - `pytest -q tests/test_tool_runtime_common.py tests/test_agent_runtime_adapters.py tests/test_model_identity_contract.py`
   - result: `27 passed, 1 warning`

Effect on module size:

1. `llm_client/mcp_turn_execution.py`: `2105 -> 1877`
2. new module: `llm_client/mcp_turn_outcomes.py` (`503` lines)

**Selected next tranche (2026-03-22, post-outcome extraction):**

Stay on `llm_client/mcp_turn_execution.py`.

Write scope for the next implementation slice:

1. extract the turn-exit / forced-finalization handoff block from
   `mcp_turn_execution.py` into a dedicated helper module
2. move tool-result trace capture, submit-success early exit,
   forced-finalization result adoption, and required-submit exhaustion policy
   behind a narrower interface
3. keep `mcp_turn_execution.py` as the top-level turn orchestrator and final
   summary caller

Why this tranche goes next:

1. `mcp_turn_execution.py` remains the largest active Program E blocker after
   `client.py`, so continuing here still gives the best blocker reduction per
   slice
2. the remaining turn-exit / forced-finalization logic is the next clean
   boundary visible in the module after the post-tool outcomes moved out
3. this keeps the decomposition boundary-led instead of bouncing early to
   `client.py`

**Verified checkpoint 11 (2026-03-22):**

The post-loop forced-finalization handoff and required-submit completion path
were extracted from `llm_client/mcp_turn_execution.py` into
`llm_client/mcp_turn_completion.py`.

What this checkpoint proved:

1. async forced-finalization adoption, submit exhaustion policy, and required
   submit failure handling form a coherent post-loop completion boundary
2. `mcp_turn_execution.py` can keep the main loop while delegating the
   termination/completion bookkeeping that only runs once the turn loop exits
3. the active blocker is now concentrated in the large pre-call
   tool-surface/disclosure/LLM-dispatch block inside `mcp_turn_execution.py`
4. focused regression coverage remained green:
   - `pytest -q tests/test_mcp_agent.py`
   - result: `106 passed`
   - `pytest -q tests/test_tool_runtime_common.py tests/test_agent_runtime_adapters.py tests/test_model_identity_contract.py`
   - result: `27 passed, 1 warning`

Effect on module size:

1. `llm_client/mcp_turn_execution.py`: `1877 -> 1800`
2. new module: `llm_client/mcp_turn_completion.py` (`369` lines)

**Selected next tranche (2026-03-22, post-completion extraction):**

Stay on `llm_client/mcp_turn_execution.py`.

Write scope for the next implementation slice:

1. extract the pre-call tool-surface/disclosure/LLM-dispatch block from
   `mcp_turn_execution.py` into a dedicated helper module
2. move context compaction, progressive disclosure filtering, deficit nudges,
   the `_inner_acall_llm` call, and no-tool response handling behind a
   narrower interface
3. keep `mcp_turn_execution.py` as the top-level turn loop that sequences
   budget checks, pre-call preparation, tool execution, outcomes, and
   completion

Why this tranche goes next:

1. it is now the largest coherent boundary left inside `mcp_turn_execution.py`
   and is large enough to plausibly clear the hard threshold in one more slice
2. it is more leverageful than the smaller trace/nudge tail because it
   removes a dense block that mixes policy, prompt-shaping, and model-call
   handling
3. if that slice lands cleanly, `mcp_turn_execution.py` should stop blocking
   Program E and the plan can shift to `client.py`

**Verified checkpoint 12 (2026-03-22):**

The pre-call tool-surface/disclosure/LLM-dispatch block was extracted from
`llm_client/mcp_turn_execution.py` into `llm_client/mcp_turn_model.py`,
leaving `mcp_turn_execution.py` as the per-turn orchestrator that sequences
budget gates, the model stage, tool execution, outcome handling, and
completion.

What this checkpoint proved:

1. active-artifact context updates, context compaction, progressive
   disclosure, deficit nudges, model dispatch, and no-tool handling form a
   coherent boundary separate from the turn loop
2. `mcp_turn_execution.py` now clears the Program E hard-threshold criterion,
   so the active blocker can honestly shift from MCP-turn orchestration to
   `client.py`
3. the extracted model-stage helper can preserve the existing call semantics
   without weakening focused MCP-agent regression coverage
4. focused regression coverage remained green:
   - `pytest -q tests/test_mcp_agent.py`
   - result: `106 passed`
   - `pytest -q tests/test_tool_runtime_common.py tests/test_agent_runtime_adapters.py tests/test_model_identity_contract.py`
   - result: `27 passed, 1 warning`

Effect on module size:

1. `llm_client/mcp_turn_execution.py`: `1800 -> 1339`
2. new module: `llm_client/mcp_turn_model.py` (`944` lines)

**Selected next tranche (2026-03-22, post-model-stage extraction):**

Shift the active blocker to `llm_client/client.py`.

Write scope for the next implementation slice:

1. extract the lifecycle/heartbeat monitoring cluster from `client.py` into a
   dedicated runtime-local module
2. move `_LLMCallProgressSnapshot`, `_LLMCallProgressReporter`,
   lifecycle-event helpers, and the sync/async heartbeat monitors behind a
   narrower interface that `client.py` can call from its public entrypoints
3. keep `client.py` as the public facade that owns routing and public call
   signatures while delegating lifecycle observability internals to the new
   module

Why this tranche goes next:

1. `client.py` is now the clearest remaining hard-threshold blocker in
   Program E
2. the lifecycle/heartbeat surface is a large, recent, observability-local
   seam rather than an arbitrary helper grab-bag
3. this slice reduces shared infrastructure debt directly in the same area
   where replay/divergence diagnosis and call-lifecycle observability have
   already been improved

**Verified checkpoint 13 (2026-03-22):**

The call-lifecycle monitoring cluster was extracted from `llm_client/client.py`
into `llm_client/call_lifecycle.py`, while `client.py` kept the private helper
names that `stream_runtime.py` already imports.

What this checkpoint proved:

1. call-lifecycle state, event emission, and sync/async heartbeat monitors are
   a coherent observability-local boundary separate from the public call
   facade
2. the extraction can reduce `client.py` materially without forcing an
   unrelated `stream_runtime.py` rewrite in the same slice
3. lifecycle-focused wrapper coverage and broader client/public-surface
   coverage both remained green after the move
4. focused regression coverage remained green:
   - `pytest -q tests/test_client_lifecycle.py`
   - result: `6 passed`
   - `pytest -q tests/test_client.py tests/test_public_surface.py`
   - result: `243 passed`

Effect on module size:

1. `llm_client/client.py`: `4184 -> 3528`
2. new module: `llm_client/call_lifecycle.py` (`688` lines)

**Selected next tranche (2026-03-22, post-call-lifecycle extraction):**

Stay on `llm_client/client.py`.

Write scope for the next implementation slice:

1. extract the duplicated public text/structured wrapper envelope from
   `client.py` into a dedicated helper module
2. move the shared tag normalization, lifecycle setup/teardown, and
   terminal-event emission scaffolding behind narrower sync/async wrapper
   helpers while leaving the public signatures in `client.py`
3. keep `client.py` as the public facade that owns routing choices and the
   user-facing entrypoints

Why this tranche goes next:

1. after the lifecycle extraction, the clearest remaining concentration in
   `client.py` is the repeated wrapper envelope around
   `call_llm` / `call_llm_structured` / `acall_llm` / `acall_llm_structured`
2. this seam is large enough to reduce the active blocker materially without
   mixing in provider-specific background polling or Responses API conversion
   logic
3. it keeps Program E moving on a truthful `client.py` boundary instead of
   jumping sideways to softer-target modules

**Verified checkpoint 14 (2026-03-22):**

The duplicated public text/structured wrapper envelope was extracted from
`llm_client/client.py` into `llm_client/call_wrappers.py`, while `client.py`
kept the public signatures and runtime-specific dispatch closures.

What this checkpoint proved:

1. tag normalization, lifecycle setup/teardown, and terminal-event emission
   are a real shared wrapper concern separate from provider/runtime logic
2. the four public text/structured wrappers can shrink materially without
   changing their public signatures or the runtime modules they dispatch into
3. lifecycle-focused coverage and broader client/public-surface coverage both
   remained green after the move
4. focused regression coverage remained green:
   - `pytest -q tests/test_client_lifecycle.py`
   - result: `6 passed`
   - `pytest -q tests/test_client.py tests/test_public_surface.py`
   - result: `243 passed`

Effect on module size:

1. `llm_client/client.py`: `3528 -> 3185`
2. new module: `llm_client/call_wrappers.py` (`276` lines)

**Selected next tranche (2026-03-22, post-wrapper extraction):**

Stay on `llm_client/client.py`.

Write scope for the next implementation slice:

1. extract the long-thinking/background polling cluster from `client.py` into
   a dedicated runtime-local module
2. move background-mode gating, polling configuration, sync/async polling
   loops, and response-retrieval helpers behind a narrower interface while
   leaving `client.py` as the public facade
3. preserve the monkeypatch-sensitive helper names that existing
   `tests/test_client.py` background tests patch today

Why this tranche goes next:

1. after the wrapper extraction, the background polling block is the clearest
   remaining coherent runtime seam inside `client.py`
2. it already has focused regression coverage for mode enabling, poll-until-
   complete behavior, and retrieval/auth/config failure cases
3. it reduces `client.py` further without mixing in Responses API conversion
   helpers or changing public entrypoints

**Verified checkpoint 15 (2026-03-22):**

The long-thinking/background polling runtime was extracted from
`llm_client/client.py` into `llm_client/background_runtime.py`, while
`client.py` kept the monkeypatch-sensitive helper names that existing
background tests patch.

What this checkpoint proved:

1. background-mode gating, polling config, sync/async polling loops, and
   response-retrieval logic form a coherent runtime-local boundary
2. the client-level patch surface for `_poll_background_response`,
   `_apoll_background_response`, `_retrieve_background_response`, and
   `_aretrieve_background_response` can stay stable while the implementation
   moves out
3. focused background tests and broader client/public-surface coverage both
   remained green after the move
4. focused regression coverage remained green:
   - `pytest -q tests/test_client.py -k 'background or poll_background or retrieve_background or long-thinking'`
   - result: `11 passed, 222 deselected`
   - `pytest -q tests/test_client_lifecycle.py tests/test_client.py tests/test_public_surface.py`
   - result: `249 passed`

Effect on module size:

1. `llm_client/client.py`: `3185 -> 2981`
2. new module: `llm_client/background_runtime.py` (`357` lines)

**Selected next tranche (2026-03-22, post-background-runtime extraction):**

Stay on `llm_client/client.py`.

Write scope for the next implementation slice:

1. extract the Responses API helper cluster from `client.py` into a dedicated
   runtime-local module
2. move strict-schema preparation, message/tool/response-format conversion,
   Responses API kwargs building, usage/cost extraction, and result
   finalization behind a narrower interface while keeping current helper names
   available from `client.py`
3. preserve the direct helper imports currently exercised by
   `tests/test_client.py`

Why this tranche goes next:

1. after the background extraction, the Responses API helper block is the
   clearest remaining coherent runtime seam inside `client.py`
2. it already has focused direct-helper tests plus routing coverage for GPT-5
   Responses API paths
3. it reduces `client.py` further without touching public entrypoints or the
   call-wrapper scaffolding that just stabilized

**Verified checkpoint 16 (2026-03-22):**

The Responses API helper cluster was extracted from `llm_client/client.py`
into `llm_client/responses_runtime.py`, while `client.py` kept the helper
names that direct helper tests import.

What this checkpoint proved:

1. strict-schema preparation, message/tool/response-format conversion,
   Responses kwargs building, usage/cost extraction, and Responses result
   finalization form a coherent runtime-local boundary
2. client-local policy hooks such as incompatible-param coercion and empty-
   response classification can stay explicit while the bulk of the helper
   logic moves out
3. focused Responses helper coverage and broader client/public-surface
   coverage both remained green after the move
4. focused regression coverage remained green:
   - `pytest -q tests/test_client.py -k 'responses or response_format or strict_json_schema or convert_tools_for_responses_api or prepare_responses_kwargs or extract_responses_usage or build_result_from_responses'`
   - result: `18 passed, 215 deselected`
   - `pytest -q tests/test_client_lifecycle.py tests/test_client.py tests/test_public_surface.py`
   - result: `249 passed`

Effect on module size:

1. `llm_client/client.py`: `2981 -> 2675`
2. new module: `llm_client/responses_runtime.py` (`353` lines)

**Verified checkpoint 17 (2026-03-22):**

The completion-path helper cluster was extracted from `llm_client/client.py`
into `llm_client/completion_runtime.py`, while `client.py` kept the helper
names that the completion path and tests already call.

What this checkpoint proved:

1. provider-kwargs preparation, provider-hint extraction, first-choice
   normalization, and completion-result finalization form a coherent runtime-
   local boundary
2. the completion helper logic can move out without changing the client-level
   empty-response/error policy hooks
3. focused completion-path coverage and broader client/public-surface coverage
   both remained green after the move
4. focused regression coverage remained green:
   - `pytest -q tests/test_client.py -k 'temperature or truncated or empty or choices'`
   - result: `11 passed, 222 deselected`
   - `pytest -q tests/test_client_lifecycle.py tests/test_client.py tests/test_public_surface.py`
   - result: `249 passed`

Effect on module size:

1. `llm_client/client.py`: `2675 -> 2547`
2. new module: `llm_client/completion_runtime.py` (`209` lines)

**Next selection status (2026-03-22, post-completion-runtime extraction):**

`client.py` is materially smaller, but the remaining 2.5k lines are no longer
dominated by one equally obvious helper seam. The next slice should start with
a fresh boundary-selection pass across the remaining client-local policy and
routing helpers rather than pretending the next tranche is already as clear as
the lifecycle, wrapper, background, Responses, and completion slices were.

**Proposed next tranche (call-contract extraction):**

1. extract the remaining call-contract helpers—budget/tag normalization,
   `_strip_incompatible_sampling_params`, `_resolve_unsupported_param_policy`,
   `_coerce_model_incompatible_params`, `_validate_execution_contract`,
   `_coerce_model_kwargs_for_execution`, `_strip_llm_internal_kwargs`, and the
   related routing introspection helpers—into a new `call_contracts.py`-like
   module that lives behind the current public entrypoints
2. keep `client.py` as the public facade emitting the same observability,
   lifecycle, and routing behavior; clients continue to call the same helper
   names while the implementation moves out
3. focus verification on the existing `tests/test_client.py` helpers around
   `unsupported_param_policy`, GPT-5 sampling stripping, agent-only kwargs,
   and execution-mode validation, plus the general call/public-surface suite

Acceptance criteria:

- the new module owns the density around `_prepare_call_kwargs`, `_raise_empty_response`, `execution_mode` validation, and signing agent vs non-agent kwargs
- helper names exported from `client.py` stay intact so downstream tests keep working
- targeted helper tests and the full client/public-surface suite pass before committing

**Verified checkpoint 18 (2026-03-22):**

The call-contract helper cluster was extracted from `llm_client/client.py`
into the existing `llm_client/call_contracts.py`, expanding it from a
tag/budget/retry-safety module into the centralized pre-call contract surface.

What this checkpoint proved:

1. empty-response classification, schema-error detection, GPT-5 sampling
   constants, unsupported-param policy, param coercion, agent-model detection,
   execution-mode validation, agent-only kwargs filtering, max-tokens clamping,
   and model deprecation warnings form a coherent pre-call contract boundary
2. `client.py` can re-export all moved names so downstream imports
   (`text_runtime`, `structured_runtime`, `stream_runtime`, `test_agents`)
   remain stable without code changes in those modules
3. focused regression coverage remained green:
   - `pytest -q tests/test_call_contracts.py tests/test_client.py tests/test_public_surface.py tests/test_client_lifecycle.py`
   - result: `252 passed`

Effect on module size:

1. `llm_client/client.py`: `2547 -> 2056`
2. `llm_client/call_contracts.py`: `129 -> 679`

**Verified checkpoint 4 (2026-03-22):**

The first `agents_codex.py` slice extracted Codex process diagnostics,
best-effort process-tree termination, and timeout-message helpers into
`llm_client/agents_codex_process.py`, while keeping `agents_codex.py` as the
compatibility and orchestration surface.

What this checkpoint proved:

1. `agents_codex.py` has a real separable OS/process-management concern that
   can move without touching the CLI transport or streaming surfaces
2. `agents.py` re-export compatibility remains load-bearing and must be
   preserved explicitly when helpers move
3. focused Codex timeout and process-isolation dispatch coverage remained
   green:
   - `pytest -q tests/test_agents.py -k 'codex_process_isolation or codex_structured_isolation_dispatch or codex_timeout or codex_transport_auto'`
   - result: `6 passed, 100 deselected`

Effect on module size:

1. `llm_client/agents_codex.py`: `1931 -> 1747`
2. new module: `llm_client/agents_codex_process.py` (`213` lines)

**Selected next slice (2026-03-22, within agents_codex tranche):**

Extract the Codex SDK runtime path into a dedicated module:

1. move in-process SDK execution, isolated-process worker entrypoints, and
   structured-runtime helpers into a new runtime module
2. keep monkeypatch-sensitive wrappers in `agents_codex.py` for
   `_acall_codex_inproc`, `_call_codex_in_isolated_process`,
   `_acall_codex_structured_inproc`, and
   `_call_codex_structured_in_isolated_process`
3. leave CLI transport and streaming in `agents_codex.py` for a later slice

**Verified checkpoint 5 (2026-03-22):**

The Codex SDK runtime path was extracted into
`llm_client/agents_codex_runtime.py`, with `agents_codex.py` retaining only
the orchestration logic and monkeypatch-sensitive wrapper names used by the
existing tests.

What this checkpoint proved:

1. the in-process SDK path and isolated-process worker/runtime path are a
   coherent boundary that can move together
2. preserving wrapper names in `agents_codex.py` keeps the effective test and
   monkeypatch contract intact while shrinking the main module materially
3. `agents_codex.py` now clears the Program E hard-threshold criterion
4. focused Codex runtime coverage remained green:
   - `pytest -q tests/test_agents.py -k 'codex_process_isolation or codex_structured_isolation_dispatch or codex_timeout or codex_transport_auto'`
   - result: `6 passed, 100 deselected`
   - `pytest -q tests/test_agents.py -k 'codex and (structured_sync or structured_async or structured_with_fenced_json or codex_timeout_explicit or codex_process_isolation_dispatches_sync or codex_transport_cli_dispatches_async or codex_transport_auto_dispatches_sync or structured_process_isolation_dispatches_sync)'`
   - result: `6 passed, 100 deselected`

Effect on module size:

1. `llm_client/agents_codex.py`: `1747 -> 1317`
2. new module: `llm_client/agents_codex_runtime.py` (`605` lines)

**Selected next tranche (2026-03-22, post-agents_codex closeout):**

Move to `llm_client/mcp_agent.py`.

Write scope for the next implementation slice:

1. extract the per-turn tool-processing path from `_agent_loop` into a
   dedicated module
2. keep `mcp_agent.py` as the orchestration facade so public runtime entry
   points and event-code imports remain stable
3. preserve behavior for:
   - compliance-gate rejection bookkeeping
   - tool-contract rejection bookkeeping
   - control-loop suppression
   - runtime artifact reads, handle injection, and execution ordering
   - artifact/binding/capability reconciliation after successful tool calls

Why this tranche goes next:

1. `mcp_agent.py` is now the clearest remaining hard-threshold blocker after
   `agents_codex.py` cleared the threshold
2. the per-turn tool-processing path is a coherent responsibility inside
   `_agent_loop`, already backed by dense regression coverage in
   `tests/test_mcp_agent.py`
3. it should materially reduce `mcp_agent.py` without starting with the even
   more central `client.py` facade

### Phase 3: Remaining Oversized Modules Or Explicit Re-Scope

**Purpose:** continue through the remaining oversized modules or document a
truthful exception if the threshold itself is wrong.

**Input -> Output:** unresolved oversize debt -> completed decompositions or
explicitly justified exceptions

**Passes if:**

- every module above threshold is either reduced or explicitly justified
- Program E closeout status becomes evidence-based rather than aspirational

**Fails if:**

- oversized modules are left implicit in conversation only
- completion depends on vague future cleanup

---

## Required Tests

### Existing Tests (Must Pass)

| Test Pattern | Why |
|--------------|-----|
| `tests/test_client.py` | core runtime behavior must stay stable during `client.py` extractions |
| `tests/test_io_log.py` | observability persistence must stay stable during `io_log.py` extractions |
| `tests/test_client_lifecycle.py` | lifecycle observability must not regress |
| targeted CLI / public-surface tests | extracted modules must preserve user-facing surfaces |

### Additional Verification

1. fresh module line-count audit after each tranche
2. roadmap and Program E plan status updates after each verified slice

---

## Acceptance Criteria

- [x] the remaining oversized modules are inventoried with a durable audit baseline
- [x] the first decomposition tranche is explicitly selected with pass/fail tests
- [x] the roadmap and Program E umbrella plan point to this child plan as the next default slice

---

## Notes

Do not treat this as a blanket refactor. Keep slices thin and responsibility
driven. The point is not to hit arbitrary line counts by shuffling code. The
point is to reduce oversized mixed-responsibility modules until Program E's
maintainability claim is true.
