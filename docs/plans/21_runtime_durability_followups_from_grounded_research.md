# Plan #21: Runtime Durability Follow-Ups From Grounded Research

**Status:** Completed
**Type:** implementation
**Priority:** High
**Blocked By:** None
**Blocks:** Cleaner downstream runtime defaults for benchmark-heavy consumers

---

## Gap

**Current:** `grounded-research` now completes its long benchmark runs, but it
needed project-local mitigations to get there:

- run-local `LLM_CLIENT_DB_PATH` to avoid shared SQLite lock failures
- explicit finite request timeouts at long structured call sites
- manual SQL/Python inspection to understand late-stage stalls across
  `llm_calls` and `tool_calls`

Those mitigations work, but they should not remain app-local policy forever.

**Target:** shared runtime defaults and diagnostics inside `llm_client` so
benchmark-oriented consumers do not need bespoke safety policy to:

1. survive concurrent observability workloads,
2. finish long structured calls reliably,
3. diagnose a trace without ad hoc SQL spelunking.

**Why:** these are runtime-substrate concerns, not application logic. If
`grounded-research` needed them, other long-running consumers will too.

---

## References Reviewed

- `CLAUDE.md` — repo status and maintenance-mode guidance
- `docs/plans/01_master-roadmap.md` — canonical roadmap and active-work rules
- `docs/plans/14_batch-progress-and-stagnation.md` — prior timeout/stagnation design
- `docs/plans/15_centralize-defaults.md` — cancelled refactor baseline to avoid reopening as pure cleanup
- `llm_client/io_log.py` — SQLite/JSONL observability backend, `tool_calls`, busy timeout handling
- `tests/test_io_log.py` — existing logger verification surface
- `~/projects/grounded-research/docs/TECH_DEBT.md` — observed runtime failures and local mitigations
- `~/projects/grounded-research/output/ubi_dense_dedup_eval/llm_observability.db` — real long-run evidence surface

---

## Pre-Made Decisions

1. This is an evidence-driven maintenance slice, not a reopening of cancelled
   Plan 15.
2. Keep `task=`, `trace_id=`, and `max_budget=` unchanged on all public call
   surfaces.
3. Preserve per-project overrides like `LLM_CLIENT_DB_PATH`, but make them
   optional mitigation rather than required safety.
4. Solve durability first, diagnostics second, default cleanup third.
5. Do not add a second observability backend; build on the existing SQLite +
   JSONL path.

---

## Files Affected

- `docs/notebooks/04_runtime_durability_followups.ipynb` (create)
- `docs/plans/21_runtime_durability_followups_from_grounded_research.md` (create)
- `docs/plans/CLAUDE.md` (modify)
- `docs/plans/01_master-roadmap.md` (modify)
- `llm_client/io_log.py` (modify)
- `llm_client/execution/timeout_policy.py` (modify)
- `llm_client/core/` runtime call surfaces (modify as needed)
- `scripts/` trace-summary helper surface (create or modify)
- `tests/test_io_log.py` (modify)
- new focused tests for trace query/runtime durability as needed

---

## Plan

### Step 1: Make SQLite durability robust under normal concurrent use

- reproduce the `database is locked` failure mode with focused tests
- harden connection behavior, busy timeout, and write-path policy in `io_log.py`
- prove that concurrent read/write pressure does not kill benchmark runs

**Completed 2026-03-26**
- enabled WAL mode on the shared observability DB connection
- serialized writes through one connection inside the process
- added bounded retry for transient `database is locked` failures
- verified with a real transient external writer-lock test and a concurrent
  mixed `log_call`/`log_tool_call` test

### Step 2: Promote long-call runtime defaults into shared policy

- define shared finite-timeout defaults for long structured calls
- keep explicit per-call overrides, but stop requiring app-local policy just to
  avoid indefinite stalls
- ensure retries/timeouts are logged with consistent semantics

**Completed 2026-03-26**
- added env-backed shared request-timeout defaults in
  `llm_client/execution/timeout_policy.py`
- set structured public call surfaces to inherit a longer finite default when
  the caller omits `timeout=`
- preserved explicit per-call timeout overrides unchanged
- verified with focused timeout-policy tests and public structured-wrapper tests

### Step 3: Add trace-level query helpers for `llm_calls` + `tool_calls`

- expose a compact way to inspect one `trace_id`
- answer “what was the last completed LLM call?”, “what tool calls ran?”, and
  “where did retries/errors happen?” without custom SQL each time

**Completed 2026-03-26**
- added `summarize_trace()` in `llm_client/observability/query.py`
- extended `python -m llm_client traces --trace-id ...` to render one compact
  cross-table summary
- taught the helper to roll up trace families like
  `pipeline/<root-trace>/...`, matching grounded-research’s real trace shape
- verified with isolated logger tests plus a real grounded-research benchmark DB

### Step 4: Verify against the grounded-research-triggered use case

- replay the diagnostic questions raised by the UBI benchmark runs
- confirm the shared helpers answer them directly from the DB/log surface

**Completed 2026-03-26**
- verified the helper against
  `grounded-research/output/ubi_dense_dedup_eval/llm_observability.db`
- confirmed it answers the downstream questions directly:
  call counts, last completed LLM call, last tool call, and trace-family rollup

---

## Required Tests

### New Tests (TDD)

| Test File | Test Function | What It Verifies |
|-----------|---------------|------------------|
| `tests/test_io_log.py` | `test_concurrent_tool_and_llm_logging_no_lock_failure` | Concurrent logging does not raise SQLite lock errors |
| `tests/test_io_log.py` | `test_trace_summary_includes_tool_and_llm_calls` | One trace can be summarized across both tables |
| `tests/test_timeout_policy.py` | `test_long_structured_default_timeout_is_finite` | Shared runtime policy provides a finite default |
| `tests/test_timeout_policy.py` | `test_explicit_timeout_override_preserved` | Caller override still wins |

### Existing Tests (Must Pass)

| Test Pattern | Why |
|--------------|-----|
| `python -m pytest tests/test_io_log.py -q` | observability backend unchanged except for intended durability improvements |
| `python -m pytest tests/test_client_lifecycle.py -q` | runtime logging changes do not break client lifecycle |

---

## Acceptance Criteria

- [x] Concurrent observability writes no longer kill normal long-running consumer workloads
- [x] Long structured calls have shared finite-timeout defaults or one equivalent shared runtime policy
- [x] One `trace_id` can be summarized across `llm_calls` and `tool_calls` without ad hoc SQL
- [x] Existing logger/runtime tests still pass
- [x] The grounded-research-triggered failure modes are documented as resolved or explicitly reclassified

---

## Failure Modes

| Failure Mode | Detection | Response |
|--------------|-----------|----------|
| SQLite changes reduce lock failures but create silent data loss | logger tests pass but row counts drift under concurrency | fail loud on dropped writes, add explicit integrity checks |
| Timeout defaults stop hangs but clip legitimate long calls | long structured-call tests regress or real benchmarks truncate | split long-call defaults by call class, keep explicit override path |
| Trace summary helper is too generic to be useful | still need manual SQL for common diagnosis questions | narrow helper scope to real questions from grounded-research traces |

---

## Notes

- This plan is justified by a real downstream benchmark program, not by
  substrate-aesthetic cleanup.
- If Step 1 alone removes the operational pain, keep Steps 2–3 minimal.
