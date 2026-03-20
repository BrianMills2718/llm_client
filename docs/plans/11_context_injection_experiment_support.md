# Plan 11: Context Injection Experiment Support

**Status:** Complete
**Type:** implementation
**Priority:** High
**Blocked By:** None
**Blocks:** evidence-based resolution of `additionalContext` rollout policy

---

## Gap

**Current:** governed repos can block on required reading and log hook outcomes,
but we still do not know whether blocked-gate `additionalContext` content helps
in practice, where size limits become harmful, or how different context
payloads affect repeated friction and downstream task completion. There is no
canonical shared experiment linkage between governed-repo hook variants and the
rest of the active stack's observability.

**Target:** `llm_client` provides the shared experiment-linking and analysis
support needed to compare governed-repo context-injection variants empirically.
That support should let a caller or external experiment harness group hook
events, downstream runs, and outcome summaries by explicit experiment id and
variant without inventing a second observability backend.

**Why:** `project-meta` Plan 08 has one remaining uncertainty with the highest
leverage: whether `additionalContext` can reliably deliver enough governing
context to matter. We should answer that with shared telemetry and controlled
comparison, not guesswork or one-off manual anecdotes.

---

## References Reviewed

- `docs/plans/10_governed_repo_friction_observability.md` - required shared telemetry prerequisite
- `docs/adr/0003-warning-taxonomy.md` - operator-facing warning taxonomy for emitted diagnostics
- `docs/adr/0005-reason-code-registry-governance.md` - machine-readable failure and reason-code discipline
- `docs/adr/0006-actor-id-issuance-policy.md` - stable actor identity for shared telemetry
- `docs/adr/0007-observability-contract-boundary.md` - observability ownership boundary
- `docs/adr/0010-cross-project-runtime-substrate.md` - cross-project telemetry authority
- `docs/adr/0012-shared-data-plane-boundary.md` - shared data ownership and metadata boundaries
- `llm_client/observability/query.py` - current comparative query/report surface
- `llm_client/observability/experiments.py` - shared experiment bookkeeping
- `llm_client/foundation.py` - event identity and metadata contract
- `~/projects/project-meta/docs/plans/08_read-gating-activation-and-context-enforcement.md` - unresolved gate/context experiment
- `~/projects/project-meta/docs/ops/STRATEGIC_REVIEW_ACTIVE_STACK_2026-03-18.md` - current uncertainty and rollout advice
- `~/projects/project-meta/docs/ops/CONTEXT_INJECTION_AND_REQUIRED_READING_MATRIX.md` - context injection capability matrix
- `~/projects/prompt_eval/README.md` - current experiment-layer boundary
- `~/projects/prompt_eval/docs/plans/02_shared-observability-boundary.md` - shared observability reuse by `prompt_eval`

---

## Files Affected

- `docs/plans/01_master-roadmap.md` (modify)
- `docs/plans/11_context_injection_experiment_support.md` (create)
- `docs/plans/CLAUDE.md` (modify)
- `llm_client/__init__.py` (modify)
- `llm_client/foundation.py` (modify)
- `llm_client/io_log.py` (modify)
- `llm_client/observability/__init__.py` (modify)
- `llm_client/observability/governed_repo.py` (modify)
- `llm_client/observability/query.py` (modify)
- `llm_client_mcp_server.py` (modify)
- `tests/test_foundation.py` (modify)
- `tests/test_governed_repo_observability.py` (modify)

---

## Plan

### Steps

1. Define one explicit experiment-linkage contract for governed-repo friction
   studies: experiment id, variant id, repo identity, and joinable downstream
   run linkage.
2. Extend shared governed-repo telemetry so those experiment fields are carried
   through import and query surfaces.
3. Add report helpers that compare variants on at least:
   - repeated block rate,
   - repeated missing-read rate,
   - downstream completion/terminal-run proxy,
   - hook error rate.
4. Document the handoff boundary: `llm_client` owns shared telemetry and query
   semantics; external runners such as `prompt_eval` or thin scripts may
   execute the actual experiments.
5. Verify the slice with deterministic synthetic experiments before any broad
   repo-wide `additionalContext` rollout decision.
6. Ensure the shared reports do not overstate recurrence by collapsing
   independent sessions into synthetic file-based sessions when the raw hook
   rows lack a stable session key.

---

## Required Tests

### New Tests (TDD)

| Test File | Test Function | What It Verifies |
|-----------|---------------|------------------|
| `tests/test_governed_repo_observability.py` | `test_import_hook_log_records_experiment_identity_and_variant` | Hook telemetry can be grouped by explicit experiment metadata |
| `tests/test_governed_repo_observability.py` | `test_query_governed_repo_variant_comparison_summarizes_friction_metrics` | Variant comparison report returns repeated-friction metrics |
| `tests/test_governed_repo_observability.py` | `test_query_governed_repo_variant_comparison_joins_downstream_run_outcomes` | Hook variants can be correlated with downstream observability results |
| `tests/test_foundation.py` | `test_validate_foundation_event_governed_repo_hook_experiment_shape` | Experiment-linked hook telemetry still satisfies the shared schema |

### Existing Tests (Must Pass)

| Test Pattern | Why |
|--------------|-----|
| `pytest -q tests/test_governed_repo_observability.py` | Base governed telemetry behavior must stay intact |
| `pytest -q tests/test_foundation.py` | Shared schema discipline must remain stable |

---

## Acceptance Criteria

- [x] Shared governed-repo telemetry can be grouped by explicit experiment id and variant
- [x] Comparison reports expose repeated-friction and hook-error rates by variant
- [x] Comparison reports can join hook telemetry to downstream run outcomes without a second backend
- [x] Raw hook rows can preserve experiment metadata at the source when an experiment runner supplies it
- [x] Shared reports distinguish stable versus degraded session identity and avoid file-path pseudo-sessions for recurrence metrics
- [x] The ownership split between `llm_client` and external experiment runners is documented explicitly
- [x] `pytest -q tests/test_foundation.py tests/test_governed_repo_observability.py` passes

---

## Notes

- This plan does not require `prompt_eval` to become the primary telemetry home.
  `prompt_eval` remains an optional experiment runner on top of shared
  `llm_client` observability.
- The first intended consumer is the unresolved `additionalContext` question in
  `project-meta` Plan 08, but the telemetry contract should be general enough
  for later governed-repo friction studies.
- Experiment runners may stamp `experiment_id`, `variant_id`, and
  `downstream_run_id` either directly into imported hook rows or through import
  overrides. `llm_client` owns the shared storage and comparison semantics, not
  the experiment execution loop itself.
- Follow-up hardening for this plan is allowed when it improves causal honesty:
  source-stamped experiment metadata is preferred over importer-only overrides,
  and repeated-friction metrics should use stable session identity whenever it
  exists.

**Post-completion hardening (2026-03-19):**

- raw hook rows now carry `context_emitted` / `context_bytes` and optional
  experiment metadata at the source
- shared summaries now expose context-emission metrics and mark degraded
  session identity explicitly instead of collapsing unrelated events into one
  file-based pseudo-session

## Verification

- `pytest -q tests/test_foundation.py tests/test_governed_repo_observability.py`
- `pytest -q tests/test_observability_defaults.py`
- `python -m py_compile llm_client/observability/governed_repo.py llm_client/observability/query.py llm_client/io_log.py llm_client/foundation.py llm_client_mcp_server.py`
