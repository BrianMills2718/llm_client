# Plan 10: Governed Repo Friction Observability

**Status:** Planned
**Type:** implementation
**Priority:** Highest
**Blocked By:** None
**Blocks:** Plan 11 and evidence-driven governed rollout decisions

---

## Gap

**Current:** governed repos now emit repo-local read-gate telemetry through
`.claude/hook_log.jsonl`, but there is no canonical shared ingestion path,
schema, or query surface for that data. Operators can prove one hook worked in
one repo, but cannot answer cross-repo questions about friction, repeated
blocks, or rollout quality without manual log scraping.

**Target:** `llm_client` owns a fail-loud shared telemetry contract for
governed-repo hook/read-gate activity. Repo-local logs become edge buffers that
can be imported into shared observability, queried across repos, and summarized
in operator-facing reports.

**Why:** we should use the governed rollout before expanding it. That requires
real observability, not anecdotes. The active stack already standardized on
`llm_client` as the shared runtime/observability substrate, so governed-repo
friction belongs there too.

---

## References Reviewed

- `README.md` - runtime and observability substrate boundary
- `docs/plans/01_master-roadmap.md` - current repo execution order
- `docs/plans/06_simplification-and-observability.md` - current observability modernization work
- `docs/plans/07_governed_repo_contract_alignment.md` - repo-local governed alignment slice
- `docs/plans/08_call_liveness_and_timeout_policy.md` - recent Foundation-backed lifecycle observability
- `docs/plans/09_lifecycle_heartbeat_and_active_call_query.md` - current in-flight lifecycle observability work
- `docs/adr/0003-warning-taxonomy.md` - operator-facing warning taxonomy for emitted diagnostics
- `docs/adr/0005-reason-code-registry-governance.md` - machine-readable failure and reason-code discipline
- `docs/adr/0006-actor-id-issuance-policy.md` - stable actor identity for shared telemetry
- `docs/adr/0007-observability-contract-boundary.md` - shared observability ownership
- `docs/adr/0010-cross-project-runtime-substrate.md` - cross-project substrate boundary
- `docs/adr/0012-shared-data-plane-boundary.md` - shared data ownership and metadata boundaries
- `docs/adr/0013-provider-timeouts-are-not-the-default-liveness-mechanism.md` - lifecycle observability precedent
- `docs/adr/0014-emit-heartbeats-and-non-destructive-stall-markers-for-long-running-calls.md` - active liveness slice immediately before this program
- `llm_client/foundation.py` - typed event/schema boundary
- `llm_client/io_log.py` - shared persistence path
- `llm_client/observability/query.py` - current query/reporting surface
- `llm_client_mcp_server.py` - current operator query surface
- `scripts/meta/hook_log.py` - repo-local hook log emitter
- `~/projects/project-meta/docs/plans/08_read-gating-activation-and-context-enforcement.md` - governed read-gate rollout plan
- `~/projects/project-meta/docs/ops/GOVERNED_REPO_CONTRACT.md` - governed repo contract
- `~/projects/project-meta/docs/ops/CONTEXT_INJECTION_AND_REQUIRED_READING_MATRIX.md` - current hook/context capability matrix
- `~/projects/project-meta/scripts/meta/hook_log.py` - canonical hook-log writer source

---

## Files Affected

- `docs/adr/0015-governed-repo-friction-telemetry-belongs-in-shared-observability.md` (create)
- `docs/adr/README.md` (modify)
- `docs/plans/01_master-roadmap.md` (modify)
- `docs/plans/10_governed_repo_friction_observability.md` (create)
- `docs/plans/CLAUDE.md` (modify)
- `llm_client/foundation.py` (modify)
- `llm_client/io_log.py` (modify)
- `llm_client/observability/governed_repo.py` (create)
- `llm_client/observability/query.py` (modify)
- `llm_client_mcp_server.py` (modify)
- `tests/test_foundation.py` (modify)
- `tests/test_governed_repo_observability.py` (create)

---

## Plan

### Steps

1. Lock the ownership boundary with ADR 0015 and define one canonical governed
   telemetry shape in shared observability.
2. Add a deterministic importer/normalizer for canonical repo hook logs so
   governed repos can feed shared observability without changing their local
   hook behavior.
3. Add query/report helpers for the first operator questions:
   - block/allow/error counts by repo and time window,
   - top missing reads,
   - top repeated-friction files,
   - per-session friction summaries.
4. Expose the first summary/query surface through the existing `llm_client`
   observability APIs and, if low-friction, the MCP server.
5. Verify the slice with synthetic canonical hook logs before using it for live
   rollout analysis.

---

## Required Tests

### New Tests (TDD)

| Test File | Test Function | What It Verifies |
|-----------|---------------|------------------|
| `tests/test_foundation.py` | `test_validate_foundation_event_governed_repo_hook_shape` | Shared schema accepts the canonical governed hook event payload |
| `tests/test_governed_repo_observability.py` | `test_import_hook_log_preserves_repo_file_and_missing_reads` | Importer preserves the operator-critical metadata |
| `tests/test_governed_repo_observability.py` | `test_import_hook_log_fails_loud_on_malformed_rows` | Malformed hook rows do not disappear silently |
| `tests/test_governed_repo_observability.py` | `test_query_governed_repo_friction_summary_reports_block_allow_error_counts` | Query/report helper returns the first cross-repo summary surface |
| `tests/test_governed_repo_observability.py` | `test_query_governed_repo_top_missing_reads_ranks_repeated_gaps` | Repeated documentation misses are queryable |

### Existing Tests (Must Pass)

| Test Pattern | Why |
|--------------|-----|
| `pytest -q tests/test_observability_defaults.py` | Default-safe observability behavior must remain intact |
| `pytest -q tests/test_foundation.py` | Foundation schema discipline must stay valid |

---

## Acceptance Criteria

- [ ] ADR 0015 documents `llm_client` as the canonical governed-repo telemetry owner
- [ ] Shared governed-repo telemetry schema exists and validates deterministically
- [ ] Repo-local canonical hook logs can be imported into shared observability
- [ ] Malformed hook rows fail loud instead of being silently skipped
- [ ] Query/report helpers can answer block/allow/error counts by repo
- [ ] Query/report helpers can rank top missing reads and repeated-friction files
- [ ] `pytest -q tests/test_foundation.py tests/test_governed_repo_observability.py tests/test_observability_defaults.py` passes

---

## Notes

- This plan is intentionally about shared observability, not changing repo-local
  hook behavior. Governed repos should keep writing the same local hook logs.
- This plan should land before broader rollout resumes. The point is to create
  evidence and diagnosis before scaling the pattern further.
- Comparative `additionalContext` experiments are intentionally split into Plan
  11 so the base telemetry contract lands first.
