# Plan 02: Client Boundary Hardening Program

**Status:** Complete
**Type:** implementation
**Priority:** High
**Blocked By:** None
**Blocks:** future workflow-layer adoption, eval-boundary cleanup, public-surface stabilization

---

## Gap

**Current:** `llm_client` has real strategic value as a shared runtime
substrate, but its internal boundaries are blurred. `client.py` is monolithic,
`__init__.py` exports too much, optional agent-runtime and eval concerns bleed
into the core package identity, and the README still frames the package as a
thin wrapper.

**Target:** `llm_client` keeps one public runtime substrate while hardening
internal boundaries in thin, verifiable slices:

1. substrate/core concerns stay explicit and stable,
2. optional agent-runtime concerns are isolated,
3. eval/workflow concerns stop expanding inside the core boundary,
4. future durable workflow capabilities can be added as a LangGraph-backed
   layer without first untangling a broader internal mess.

**Why:** The goal is not to shrink the vision. The goal is to stop boundary
drift from turning one strategically important substrate into an accidental
bundle of unrelated frameworks.

---

## References Reviewed

- `llm_client/client.py:2400-2555` - current call-boundary helpers and timeout/tag policy
- `llm_client/client.py:3905-4060` - synchronous public call entrypoint
- `llm_client/client.py:4935-5125` - asynchronous public call entrypoint
- `llm_client/__init__.py:1-260` - current public surface and package identity
- `llm_client/routing.py:1-153` - existing pure routing resolver
- `llm_client/config.py:1-45` - current typed runtime config boundary
- `llm_client/mcp_agent.py:1-40` - current agent-runtime identity and loop scope
- `llm_client/task_graph.py:1-260` - current simple DAG runner contract
- `docs/ECOSYSTEM_TOP_DOWN_ARCHITECTURE.md:170-190` - workflow-layer guidance
- `docs/ECOSYSTEM_UNCERTAINTIES.md:1-110` - current architectural uncertainty surface
- `README.md:1-220` - current public positioning and transport-layer claims

---

## Files Affected

> This section declares the files touched by the planning pass and the current
> thin implementation slices. New slices must amend this section before
> touching additional files.

- docs/plans/02_client-boundary-hardening.md (create)
- docs/PUBLIC_SURFACE_AUDIT.md (create)
- docs/plans/CLAUDE.md (modify)
- docs/ECOSYSTEM_UNCERTAINTIES.md (modify)
- README.md (modify)
- llm_client/call_contracts.py (create)
- llm_client/result_metadata.py (create)
- llm_client/result_finalization.py (create)
- llm_client/structured_runtime.py (create)
- llm_client/text_runtime.py (create)
- llm_client/timeout_policy.py (create)
- llm_client/__init__.py (modify)
- llm_client/client.py (modify)
- llm_client/agents.py (modify)
- llm_client/mcp_agent.py (modify)
- llm_client/tool_runtime_common.py (create)
- llm_client/tool_shim.py (modify)
- llm_client/tool_utils.py (modify)
- tests/test_call_contracts.py (create)
- tests/test_result_metadata.py (create)
- tests/test_result_finalization.py (create)
- tests/test_structured_runtime.py (create)
- tests/test_text_runtime.py (create)
- tests/test_agent_runtime_adapters.py (create)
- tests/test_tool_runtime_common.py (create)
- tests/test_public_surface.py (create)
- tests/test_timeout_policy.py (create)
- tests/test_client.py (modify)
- tests/test_agents.py (modify)
- tests/test_observability_defaults.py (modify)

---

## Program Guardrails

Every phase in this plan follows the shared planning rules:

1. Acceptance criteria are defined before implementation.
2. Deterministic behavior is proven with unit tests before broader adoption.
3. Work lands in thin, independently verifiable slices.
4. Integration points are tested when wired, not deferred.
5. Assumptions and risks are written down explicitly.
6. No big-bang package reorg or repo split is allowed under this plan.
7. Quality-hardening priorities must be based on live repository evidence
   (existing tests, executed coverage artifacts, or targeted verification),
   not on stale external percentage estimates.

If a proposed slice violates any of those rules, that slice fails planning and
must be redesigned before code changes proceed.

## Overall Definition Of Done

This program is done only when all of the following are true:

1. `client.py` is no longer a monolithic mixed-responsibility file and is
   materially smaller than today.
2. Each extracted internal module has a one-sentence responsibility statement
   that does not rely on "and" to hide multiple concerns.
3. The public `llm_client` substrate API remains stable while optional
   agent-runtime and eval concerns are no longer confused with the core
   package identity.
4. `task_graph` has been held to the simple-runner boundary during this
   cleanup program.
5. Agent SDK routing remains part of the core substrate rather than being
   pushed out into individual consuming projects.
6. The README and package docs describe `llm_client` truthfully as a runtime
   substrate/control plane rather than as a thin wrapper.

The program is not done merely because files were moved or because line counts
went down.

### Phase 2 Reassessment Trigger

After two to three more clean Phase 2 helper seams, reassess what remains in
`client.py`.

If the remaining bulk is still mostly entrypoint-specific orchestration rather
than helper policy, Phase 2 should stop extracting ever-smaller helpers and
Phase 5+ should instead split entrypoints into explicit runtime modules.

### Reassessment Result: Trigger Reached

**Date:** 2026-03-18

After three clean helper seams (`call_contracts.py`, `result_metadata.py`,
`result_finalization.py`), `client.py` still contains:

- four dominant public entrypoints of roughly 450-500 lines each,
- large provider/runtime clusters for Gemini-native, Responses/background, and
  streaming behavior,
- more entrypoint orchestration than reusable helper logic.

The next meaningful reduction in complexity should therefore come from
splitting runtime implementations by workload family, not from extracting
smaller and smaller helpers.

---

## Long-Term Phases

### Phase 1: Extract the Pre-Call Contract From `client.py`

**Purpose:** Pull the call-boundary substrate contract into one focused module:
`task`, `trace_id`, `max_budget`, `prompt_ref`, and retry-safety policy.

**Input -> Output:** monolithic client helper block -> dedicated internal
module with no public API change

**Passes if:**

- call-boundary helpers move out of `client.py` with behavior preserved
- all current public call entrypoints still enforce the same tag/budget rules
- focused tests prove prompt-ref validation, tag defaulting/strictness, and
  budget enforcement still behave the same

**Fails if:**

- any provider routing or result semantics change
- call kwargs forwarded to providers change
- public imports or existing call signatures change

### Phase 2: Decompose `client.py` Along Existing Runtime Seams

**Purpose:** Continue extracting coherent runtime helpers without changing the
public `call_llm*` API.

**Input -> Output:** one monolithic runtime file -> smaller internal runtime
modules with stable behavior

**Passes if:**

- structured-output, streaming, and entrypoint orchestration each have clearer
  internal homes
- `client.py` shrinks materially without semantic drift
- routing, observability, and error-path tests continue to pass
- routing decisions themselves remain inside `client.py` until a later phase
  defines a cleaner boundary for them

**Fails if:**

- decomposition introduces duplicate logic across sync/async paths
- provider-specific quirks become harder to reason about than before
- a slice mixes call-envelope extraction with dispatch/routing extraction
- the remaining responsibility of `client.py` becomes less clear, not more

#### Phase 2A: Helper-Seam Extraction

**Status:** completed

This subphase covered the low-risk shared helper seams:

- pre-call contracts,
- timeout policy,
- result metadata,
- post-dispatch result finalization.

This subphase is not the default strategy anymore unless a new helper seam is
clearly larger and cleaner than an entrypoint/runtime split.

#### Phase 2B: Runtime-Module Split

**Status:** completed

**Purpose:** Move whole runtime implementations out of `client.py` while
keeping `client.py` as the stable public facade.

**Input -> Output:** 450-500 line public entrypoint implementations in one
monolithic file -> runtime modules grouped by workload family

**Target direction:**

- `client.py` keeps public signatures, shared dataclasses/protocols, and
  top-level policy glue
- text-call implementation moves behind an internal text runtime
- structured-call implementation moves behind an internal structured runtime
- existing `stream_runtime.py`, `batch_runtime.py`, and `embedding_runtime.py`
  remain part of the same runtime-level direction

**Result:** completed. `client.py` now delegates text and structured workload
families to `text_runtime.py` and `structured_runtime.py`, while keeping the
public facade stable and materially shrinking the monolithic call surface.

**Passes if:**

- `client.py` becomes materially smaller because whole entrypoint
  implementations move out, not just helpers
- no public `call_llm*` or `acall_llm*` signature changes occur
- runtime modules own coherent workload families rather than random helper
  scraps
- routing and provider-specific behavior remain understandable inside the
  moved runtime, even if they are not yet separately abstracted

**Fails if:**

- the split produces thin wrapper modules with duplicated sync/async logic
- the new runtime module needs half its policy from `client.py` and half from
  local helpers
- a big-bang filesystem reorg is mixed into the slice
- the split tries to solve text, structured, streaming, and public-surface
  cleanup in one pass

### Phase 3: Slim the Public Surface

**Purpose:** Make `llm_client`’s public identity match the substrate boundary.

**Input -> Output:** broad package-level re-export surface -> deliberate public
API with optional/internal subsystems clearly separated

**Passes if:**

- `__init__.py` exports only the core substrate and explicitly sanctioned
  surfaces
- optional agent/eval/workflow helpers are no longer indistinguishable from the
  core substrate
- compatibility impact is documented before any export removal

**Fails if:**

- callers lose access to stable substrate APIs without migration notes
- the package identity remains “everything is first-class”

### Phase 4: Isolate Optional Agent Runtime

**Purpose:** Keep custom MCP/tool-loop/runtime features available while making
them an explicit optional layer instead of the package identity.

**Input -> Output:** boundary-leaky agent runtime -> isolated agent-runtime
surface with fewer private cross-imports

**Passes if:**

- private helper imports across `mcp_agent.py`, `tool_utils.py`, and related
  modules are reduced or eliminated
- custom runtime features remain available to projects that need them
- growth of the custom runtime is gated by explicit value judgments, not habit

**Fails if:**

- the refactor breaks current MCP/tool-loop use cases
- the runtime keeps expanding while still coupled to unrelated substrate code

#### Phase 4A: Shared Tool-Runtime Helpers

**Status:** completed

**Purpose:** Move record and normalization helpers shared by MCP-backed and
direct-tool execution into one narrow internal module so lower-level tool code
and the structured-output tool shim do not import the full MCP loop runtime
for shared data contracts.

**Input -> Output:** direct-tool helpers coupled to `mcp_agent.py` internals
-> shared `tool_runtime_common.py` seam with stable compatibility re-exports

**Passes if:**

- `tool_utils.py` stops importing private helpers from `mcp_agent.py`
- `tool_shim.py` stops importing private MCP helper functions for agent result
  shape, usage extraction, and inner-call indirection
- `mcp_agent.py` still exposes `MCPToolCallRecord` and related names for
  compatibility
- focused tests prove direct tools, the tool shim, and MCP schema helpers all
  still behave the same

**Fails if:**

- MCP policy/loop control leaks into the new shared helper module
- callers lose the existing `llm_client.mcp_agent` compatibility names

#### Phase 4B: Explicit Optional-Runtime Adapters

**Status:** completed

**Purpose:** Stop core runtime modules from importing private optional-runtime
entrypoints directly.

**Input -> Output:** `text_runtime.py` importing private MCP/tool-loop
functions -> `text_runtime.py` importing explicit adapter functions

**Passes if:**

- core runtime code calls named optional-runtime adapters instead of private
  `_acall_with_*` functions
- existing tests that patch the private loop functions keep working because the
  adapters delegate at call time
- no call signatures or routing behavior change

**Fails if:**

- adapter work forces a public package-surface expansion
- routing tests need behavioral rewrites instead of simple seam verification

### Phase 5: Hold the Line on Workflow Scope

**Status:** completed

**Purpose:** Preserve the current simple DAG runner while preventing it from
becoming the accidental durable workflow engine.

**Input -> Output:** ambiguous workflow future -> explicit decision rule

**Passes if:**

- `task_graph` stays limited to simple DAG dispatch/validation/logging
- durable state, resume, HITL, and long-lived memory remain deferred until a
  real shared workflow slice is ready
- one future LangGraph-backed slice can consume `llm_client` calls without
  reopening substrate boundaries

**Fails if:**

- `task_graph` grows new workflow-engine capabilities during substrate cleanup

**Result:** completed. The workflow boundary is now explicit in the roadmap and
workflow child plan, `task_graph` stayed at the simple-runner boundary during
this cleanup program, and no durable-workflow features were added under this
plan.

### Phase 6: Re-home Eval Concerns Deliberately

**Status:** completed

**Purpose:** Stop mixing transport/routing substrate concerns with scoring,
review, and experiment-eval logic.

**Input -> Output:** eval helpers mixed into substrate package -> explicit eval
boundary using shared observability

**Passes if:**

- shared observability remains in `llm_client`
- scoring/review/experiment helpers either move out or become explicitly
  optional
- `prompt_eval` or a sibling eval package can consume the same shared backend

**Fails if:**

- eval concerns keep expanding in the core runtime boundary
- shared run/query semantics split into multiple competing infrastructures

**Result:** completed. Shared observability stayed in `llm_client`,
`experiment_summary` now carries observability-owned summary bookkeeping, and
the remaining eval-facing modules are explicitly optional rather than part of
the core substrate story.

#### Phase 6A: Shared Summary Bookkeeping Extraction

**Status:** completed

**Purpose:** Stop core observability from importing the broader eval-review
module just to compute run-summary outcome/adoption aggregates.

**Input -> Output:** `observability/experiments.py` importing
`llm_client.experiment_eval` for summary bookkeeping -> shared
`experiment_summary.py` seam consumed by observability, CLI detail views, and
`experiment_eval`

**Passes if:**

- core observability no longer imports `llm_client.experiment_eval` for
  summary bookkeeping
- `llm_client.experiment_eval` keeps exposing the same helper surface for
  compatibility
- CLI experiment detail/reporting paths keep the same behavior
- focused summary, observability, and CLI tests pass without behavioral
  rewrites

**Fails if:**

- the slice changes gate-policy, review, or scoring behavior
- shared observability grows a new dependency on broader eval-review logic

### Phase 7: Rewrite the Public Story

**Status:** completed

**Purpose:** Make the README and package docs describe the system truthfully.

**Input -> Output:** “thin wrapper” framing -> “runtime substrate/control
plane” framing

**Passes if:**

- the README describes the actual substrate: routing, budgets, observability,
  prompts, provenance, agent SDK dispatch
- optional layers are described honestly as such

**Fails if:**

- public docs keep underselling the core or hiding the optional/runtime split

**Result:** completed. `README.md`, package docstrings, and repo guidance now
describe `llm_client` as a runtime substrate/control plane and label eval,
analyzer, and task-graph surfaces as optional layers.

---

## Closeout

This plan is complete for the current substrate-hardening tranche.

The next work is no longer "keep extracting pieces from `client.py`." The only
remaining significant architecture work now belongs to separate programs:

1. benchmark-backed model-ranking changes, not covered by this plan,
2. LangGraph-backed workflow proof, covered by the workflow-layer plan,
3. any future optional-runtime or eval-package moves, but only when live
   evidence shows a new boundary leak rather than a speculative cleanup wish.

### Helper-Seam Checkpoint

This checkpoint is complete and now serves as the boundary between the helper
extraction subphase and the runtime-module split subphase.

### Runtime-Split Checkpoint

This checkpoint is complete. `client.py` now delegates both text and
structured workload families to internal runtime modules while keeping the
public facades stable.

**Observed result:**

- `client.py` is down to 4629 lines from the 6249-line pre-runtime-split
  reassessment point
- text-call control flow lives in `llm_client/text_runtime.py`
- structured-call control flow lives in `llm_client/structured_runtime.py`
- existing public tests and focused seam tests still pass without patch-target
  churn beyond the runtime boundary

### Next Decision Point

The next pass should not assume another runtime split by default. Reassess
whether the remaining `client.py` bulk is best handled by:

- public-surface cleanup in `__init__.py`,
- README/package-story correction,
- or one additional coherent runtime family move if a workload boundary is
  still obvious and large enough to justify it.

Current recommendation: do the README/package-story correction before any
export slimming. The workspace has broad direct `from llm_client import ...`
usage across many repos, so public-surface removal now would be a high-risk
integration change rather than a thin cleanup slice.

### Completed Slice: Public Story Correction

**Purpose:** Make the top-level docs and package docstring describe the
runtime substrate truthfully without changing exports yet.

**Input -> Output:** “thin wrapper” framing and under-specified examples ->
truthful runtime-substrate framing plus explicit call-contract examples

**Passes if:**

- `README.md` no longer describes `llm_client` as a thin wrapper
- the top-level usage examples show the real runtime contract, including
  observability/budget fields
- docs distinguish between preferred task-based selection and raw-model
  override examples
- `__init__.py` docstring no longer misstates the package identity
- no exports are removed in the same slice

**Fails if:**

- the docs still teach a wrapper-only mental model
- the slice silently changes the public import surface
- examples encourage project code to omit required runtime metadata

### Completed Slice: Explicit Export Audit

**Purpose:** Classify the current top-level package surface before any
`__init__.py` slimming.

**Input -> Output:** one broad undifferentiated export surface -> explicit
keep/hold/move categories with downstream-usage evidence

**Passes if:**

- the current `__all__` surface is categorized into core substrate,
  optional-runtime hold, and candidate move-off-top-level groups
- direct workspace usage is measured well enough to identify high-risk
  removals
- the plan records that export removal is deferred until after this audit
- no exports are removed in the same slice

**Fails if:**

- export categories are asserted without checking downstream usage
- the slice silently removes or renames top-level imports
- the audit ignores direct `import llm_client` side-effect consumers

### Next Decision Point After Export Audit

Do not remove top-level exports yet. The next thin slice should be either:

- symbol-level downstream auditing for one candidate move group,
- explicit grouping/annotation inside `__init__.py` without behavior change,
- or a separate migration plan for the import-time side-effect contract.

### Completed Slice: Explicit `__init__` Export Grouping

**Purpose:** Make the top-level package surface intentional in code without
changing which names are exported.

**Input -> Output:** one flat `__all__` list -> grouped export categories
backed by one top-level public surface

**Passes if:**

- `__init__.py` defines explicit export groups for core substrate,
  compatibility holds, and candidate-move names
- `__all__` is derived from those groups without symbol loss or duplication
- names declared in `__all__` actually resolve on the package object
- no downstream-facing export removals or renames occur

**Fails if:**

- the slice changes the effective public surface instead of only organizing it
- grouped exports drift from `__all__`
- `__all__` continues to advertise names that the package does not expose

### Completed Slice: Symbol-Level Audit For The Difficulty Group

**Purpose:** Prove whether one candidate move-off-top-level group is actually
used outside `llm_client` before planning any export slimming.

**Input -> Output:** grouped candidate category in the public-surface audit ->
symbol-level usage evidence and a concrete migration-risk judgment

**Passes if:**

- the difficulty group is measured by symbol, not only by category label
- internal versus external consumers are distinguished explicitly
- the audit records whether a future top-level deprecation would be low-risk,
  medium-risk, or high-risk
- no exports are removed or renamed in the same slice

**Fails if:**

- the group is labeled low-risk without checking downstream imports
- the slice silently changes package behavior while claiming to be an audit
- internal test/doc references are mistaken for cross-project consumers

**Result:** completed. The audit found no non-`llm_client` workspace imports
for the difficulty helpers. Current live coupling is internal to
`llm_client` (`task_graph`, `analyzer`, tests, and docs), so this group is a
credible low-risk pilot for a future top-level deprecation/migration slice.
The migration itself remains deferred.

### Completed Slice: Pilot Difficulty-Group Top-Level Deprecation

**Purpose:** Use the narrowest audited candidate-move group to prove the
top-level export-migration mechanics without touching broadly used substrate
APIs.

**Input -> Output:** low-risk candidate-move audit result -> one explicit
compatibility-preserving deprecation slice plan

**Planned scope:**

- keep `llm_client.difficulty` stable
- keep top-level names working during the deprecation window
- update internal docs/tests to prefer module imports
- add explicit migration notes before any eventual removal

**Passes if:**

- the pilot does not remove `llm_client.difficulty`
- the pilot does not change core substrate imports such as `call_llm`,
  `render_prompt`, or `embed`
- deprecation behavior, if introduced, is explicit and test-covered rather
  than silent
- internal references in docs/tests stop teaching top-level imports for the
  difficulty group

**Fails if:**

- the pilot couples deprecation work to unrelated surface cleanup
- the pilot changes behavior for broad substrate imports
- top-level names are removed before migration guidance and internal cleanup
  land together

**Result:** completed. The package root now keeps the difficulty helpers only
through an explicit deprecation shim, while `llm_client.difficulty` remains
the stable home. Internal docs now teach the module import, and focused public
surface tests prove that:

- top-level compatibility still works,
- deprecated access emits explicit warnings,
- de facto compatibility for `get_model_candidates_for_difficulty` remains
  intact during the migration window.

### Completed Slice: Pilot Git-Utils Top-Level Deprecation

**Purpose:** Prove a second candidate-move migration path on an internal
supporting utility surface rather than on a task-policy helper.

**Input -> Output:** audited low-risk git-utils group -> compatibility-preserving
top-level deprecation while keeping the module namespace stable

**Passes if:**

- `llm_client.git_utils` remains the stable home
- top-level compatibility still works during the deprecation window
- both function and constant exports emit explicit warnings when resolved from
  the package root
- no core runtime APIs or broad substrate imports are changed in the same
  slice

**Fails if:**

- the slice removes the git-utils module instead of only deprecating the
  top-level re-exports
- top-level imports stop working before a migration window exists
- the slice couples utility-surface cleanup to unrelated runtime refactors

**Result:** completed. The package root now resolves git-utils names through
an explicit deprecation shim, while `llm_client.git_utils` remains the stable
module namespace. Focused public-surface tests prove compatibility for both
callable and constant exports.

### Completed Slice: Pilot Scoring And Experiment-Eval Top-Level Deprecation

**Purpose:** Migrate optional evaluation helpers off the package root without
changing the underlying scoring or experiment-eval modules.

**Input -> Output:** audited low-risk eval-related export groups -> explicit
module-level teaching plus compatibility-preserving top-level deprecation

**Passes if:**

- `llm_client.scoring` and `llm_client.experiment_eval` remain the stable
  module homes
- top-level compatibility still works during the deprecation window
- current docs/examples stop teaching top-level imports for these groups
- no core runtime APIs are changed in the same slice

**Fails if:**

- the slice changes scoring or gate-evaluation behavior rather than only the
  package surface
- top-level compatibility is removed before the deprecation window exists
- README/CLAUDE/examples continue teaching the deprecated package-root imports

**Result:** completed. The package root now resolves scoring and
experiment-eval names through explicit deprecation shims, while the canonical
docs/examples teach `llm_client.scoring` and `llm_client.experiment_eval`
instead.

### Completed Slice: Pilot Task-Graph And Analyzer Top-Level Deprecation

**Purpose:** Stop teaching workflow/analyzer helpers as package-root APIs while
keeping their real module namespaces stable for existing consumers.

**Input -> Output:** audited task-graph/analyzer top-level exports -> explicit
module-level teaching plus compatibility-preserving top-level deprecation

**Passes if:**

- `llm_client.task_graph` and `llm_client.analyzer` remain the stable homes
- top-level compatibility still works during the deprecation window
- current docs/examples stop teaching `from llm_client import load_graph` and
  similar package-root forms
- no workflow/analyzer runtime behavior changes occur in the same slice

**Fails if:**

- the slice confuses top-level deprecation with module removal
- real module-level consumers lose compatibility
- the slice makes architectural claims about removing `task_graph` or analyzer
  entirely

**Result:** completed. The package root now resolves task-graph and analyzer
names through deprecation shims, while the canonical docs teach
`llm_client.task_graph` and `llm_client.analyzer` instead. This changes the
package surface without changing the module-level contract used by existing
consumers.

---

## Required Tests

### New Tests (TDD)

| Test File | Test Function | What It Verifies |
|-----------|---------------|------------------|
| `tests/test_call_contracts.py` | `test_require_tags_calls_observability_guardrails` | extracted tag-resolution layer still invokes the same guardrails |
| `tests/test_call_contracts.py` | `test_normalize_prompt_ref_rejects_blank_values` | prompt provenance validation still fails loudly |
| `tests/test_call_contracts.py` | `test_check_budget_raises_when_trace_is_spent` | budget enforcement remains a pre-call contract |
| `tests/test_result_metadata.py` | `test_build_routing_trace_tracks_normalization_and_api_base_injection` | extracted metadata layer preserves routing-trace semantics without owning routing decisions |
| `tests/test_result_metadata.py` | `test_merge_warning_records_deduplicates_warning_messages_and_extra_records` | warning normalization remains deterministic and duplicate-safe |
| `tests/test_result_metadata.py` | `test_annotate_result_identity_sets_identity_and_merges_warning_records` | result identity stamping remains stable after extraction |
| `tests/test_result_finalization.py` | `test_cache_hit_view_returns_copy_without_mutating_source` | cache-hit normalization remains pure bookkeeping that does not mutate cached state |
| `tests/test_result_finalization.py` | `test_finalize_result_applies_cache_hit_and_identity_metadata` | shared finalization seam combines cache-hit stamping and identity normalization correctly |
| `tests/test_structured_runtime.py` | `test_structured_runtime_sync_and_async_preserve_cache_and_identity_contracts` | second runtime split preserves the shared structured-call contract behind the new module |
| `tests/test_public_surface.py` | `test_grouped_exports_flatten_to_public_surface_without_duplicates` | grouped export constants stay aligned with `__all__` and remain duplicate-free |
| `tests/test_public_surface.py` | `test_top_level_declared_exports_resolve_for_star_import_compatibility` | names promised by `__all__` actually resolve on the package surface |
| `tests/test_public_surface.py` | `test_top_level_difficulty_export_warns_and_resolves` | deprecated top-level difficulty exports still work while steering callers to the module namespace |
| `tests/test_public_surface.py` | `test_explicit_import_of_non_all_difficulty_helper_warns_and_resolves` | de facto compatibility exports remain available during the deprecation window |
| `tests/test_public_surface.py` | `test_top_level_git_utils_export_warns_and_resolves` | deprecated top-level git-utils functions still resolve while steering callers to the module namespace |
| `tests/test_public_surface.py` | `test_explicit_import_of_git_utils_constant_warns_and_resolves` | deprecated top-level git-utils constants remain available during the deprecation window |
| `tests/test_public_surface.py` | `test_top_level_scoring_export_warns_and_resolves` | deprecated top-level scoring functions still resolve while steering callers to `llm_client.scoring` |
| `tests/test_public_surface.py` | `test_explicit_import_of_experiment_eval_export_warns_and_resolves` | deprecated top-level experiment-eval helpers remain available during the deprecation window |
| `tests/test_public_surface.py` | `test_top_level_task_graph_export_warns_and_resolves` | deprecated top-level task-graph exports still resolve while steering callers to `llm_client.task_graph` |
| `tests/test_public_surface.py` | `test_explicit_import_of_analyzer_export_warns_and_resolves` | deprecated top-level analyzer exports remain available during the deprecation window |
| `tests/test_timeout_policy.py` | `test_normalize_timeout_ban_appends_warning_and_zeroes_timeout` | shared timeout-policy module preserves the ban behavior |
| `tests/test_timeout_policy.py` | `test_normalize_timeout_negative_values_clamp_to_zero` | timeout normalization remains deterministic and transport-agnostic |
| `tests/test_text_runtime.py` | `test_text_runtime_sync_and_async_preserve_cache_and_identity_contracts` | first runtime split preserves the shared text-call contract behind the new module |

### Existing Tests (Must Pass)

| Test Pattern | Why |
|--------------|-----|
| `tests/test_observability_defaults.py` | existing helper defaults and retry-safety behavior remain unchanged |
| `tests/test_client.py -k "cache_hit_skips_llm_call"` | public cache-hit behavior still bypasses provider calls and returns normalized cached results |
| `tests/test_client.py -k "RequiredTags or prompt_ref or budget_exceeded_not_retried"` | public entrypoints still honor extracted call-contract behavior |
| `tests/test_client.py -k "cache_hit or prompt_ref or budget_exceeded_not_retried or timeout_policy_ban"` | text-call facade still preserves the same public behavior after runtime extraction |
| `tests/test_client.py -k "structured"` | structured-call facade still preserves the same public behavior after runtime extraction |
| `tests/test_model_identity_contract.py -k "warning_records_include_stable_model_code or explicit_routing"` | public identity/routing semantics still honor the extracted metadata layer |
| `tests/test_model_identity_contract.py -k "structured"` | structured-call identity/routing semantics still survive the moved runtime |
| `tests/test_agents.py -k "structured"` | structured agent-SDK routing still works after the runtime split |
| `tests/test_client.py -k "timeout_policy_ban"` | client entrypoints still honor shared timeout policy |
| `tests/test_agents.py -k "timeout_policy_ban"` | agent SDK routing still honors shared timeout policy |

---

## Acceptance Criteria

- [x] Long-term phased plan is committed in-repo with explicit pass/fail rules
- [x] New or sharpened architectural uncertainties are recorded explicitly
- [x] `client.py` no longer owns the extracted pre-call contract helpers
- [x] timeout-policy logic is no longer duplicated between `client.py` and `agents.py`
- [x] pure result-metadata helpers are no longer owned inline by `client.py`
- [x] post-dispatch result finalization no longer lives inline in duplicated sync/async cache-hit paths
- [x] text-call implementation no longer lives inline in `client.py`
- [x] `client.py` keeps the public text-call facade while delegating to the internal runtime
- [x] focused text-runtime tests pass for the moved implementation
- [x] structured-call implementation no longer lives inline in `client.py`
- [x] `client.py` keeps the public structured-call facade while delegating to the internal runtime
- [x] focused structured-runtime tests pass for the moved implementation
- [x] no streaming move was mixed into this slice
- [x] README and package docstring no longer describe `llm_client` as a thin wrapper
- [x] README examples now show the real runtime call contract
- [x] explicit export audit is recorded before any top-level export slimming
- [x] `__init__.py` public exports are grouped explicitly without changing the public surface
- [x] all names listed in `__all__` resolve on the package object
- [x] Focused deterministic tests pass for the extracted seam
- [x] Public `call_llm*` behavior is unchanged for this slice
- [x] No filesystem/package reorg occurs in this slice
- [x] No README/public-surface churn is mixed into this slice
- [x] No routing/dispatch extraction occurs in this slice

---

## Known Risks And Assumptions

### Assumptions

1. `llm_client` remains the public substrate API; it is not reduced to a
   LiteLLM callback layer.
2. Logical boundary hardening comes before physical package/file moves.
3. Stable call-boundary contracts are more important than immediate line-count
   reduction.
4. Result identity metadata can be extracted cleanly only if model-warning
   policy remains local to `client.py` for now.

### Risks

1. Internal helper extraction can break tests that patch module-level globals
   if patch targets are not updated carefully.
2. Public-surface slimming later may reveal hidden downstream coupling that is
   not visible from unit tests alone.
3. The workflow-layer decision can get reopened prematurely if `task_graph`
   accretes features during the cleanup program.
4. A metadata extraction can accidentally drag routing decisions with it if the
   routing-policy label stops being an explicit input.
5. A runtime split can fail if sync and async text paths are moved separately
   and start diverging more than they already do.
6. A runtime split can produce misleading modularity if `client.py` keeps too
   much call-flow logic and the new module becomes only a partial delegate.

### Open Questions

Tracked in:

- `docs/ECOSYSTEM_UNCERTAINTIES.md` U6
- `docs/ECOSYSTEM_UNCERTAINTIES.md` U7
- `docs/ECOSYSTEM_UNCERTAINTIES.md` U8
- `docs/ECOSYSTEM_UNCERTAINTIES.md` U9
- `docs/ECOSYSTEM_UNCERTAINTIES.md` U10
