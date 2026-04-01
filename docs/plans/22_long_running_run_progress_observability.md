# Plan #22: Long-Running Run Progress Observability

**Status:** Planned
**Type:** implementation
**Priority:** High
**Blocked By:** None
**Blocks:** DIGIMON graph-build progress visibility, any shared long-running job status view

---

## Gap

**Current:** `llm_client` persists experiment run start/finish records and has a
local `BatchProgressTracker`, but it does not expose a durable, generic
progress contract for long-running jobs. Consumers can register a run and
checkpoint work locally, but they cannot emit or query shared status like:

- current stage,
- total vs completed units,
- last real progress timestamp,
- stagnation reason,
- checkpoint metadata,
- or latest active-run summaries.

**Target:** Add a shared observability contract for long-running runs:

1. durable stage/progress/stagnation events tied to an existing `run_id`,
2. SQLite + JSONL persistence,
3. a producer API that any project can call,
4. a query helper that returns the latest known status for active runs.

**Why:** This is shared infrastructure, not DIGIMON-specific logic. Any
project with a long-running build/index/extract/batch job should use the same
substrate.

---

## References Reviewed

- `llm_client/observability/experiments.py`
- `llm_client/io_log.py`
- `llm_client/execution/batch_runtime.py`
- `docs/plans/14_batch-progress-and-stagnation.md`
- `docs/plans/07_stream_lifecycle_observability.md`
- `tests/test_batch_progress.py`

---

## Pre-made Decisions

1. **This is shared infra.** Ownership stays in `llm_client`.
2. **Reuse `run_id`.** Progress is a child of an existing run, not a new ID space.
3. **Persist through existing observability stores.** SQLite + JSONL only.
4. **Keep the contract generic.** It must work for build/index/extract/batch jobs.
5. **Stage-oriented first.** Explicit stages plus numeric snapshots are enough for the first slice.
6. **No polling daemon.** Producers emit progress; `llm_client` stores and queries it.

---

## Files Affected

- `docs/plans/22_long_running_run_progress_observability.md`
- `docs/plans/CLAUDE.md`
- `llm_client/io_log.py`
- `llm_client/observability/experiments.py`
- `llm_client/observability/` (new module(s) if cleaner)
- `tests/test_experiment_log.py`
- `tests/test_io_log.py`
- `tests/test_batch_progress.py`
- `README.md` (if public surface changes)

---

## Contract

### Producer-facing API

```python
log_run_stage(
    run_id: str,
    *,
    stage: str,
    message: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None

log_run_progress(
    run_id: str,
    *,
    stage: str | None = None,
    total: int | None = None,
    completed: int | None = None,
    failed: int | None = None,
    progress_unit: str | None = None,
    avg_latency_s: float | None = None,
    checkpoint_ref: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None

mark_run_stagnated(
    run_id: str,
    *,
    stage: str | None = None,
    reason: str,
    metadata: dict[str, Any] | None = None,
) -> None
```

### Query helper

```python
get_active_run_progress(project: str | None = None) -> list[dict[str, Any]]
```

Each row should include at least:
- `run_id`
- `project`
- `task`
- `dataset`
- `status`
- `stage`
- `total`
- `completed`
- `failed`
- `progress_unit`
- `last_progress_at`
- `stagnated`
- `checkpoint_ref`

---

## Plan

### Step 1: Define persistence model

Add a durable run-progress event/table shape under `io_log`.

**Acceptance:**
- schema is explicit,
- every row references a `run_id`,
- invalid required fields fail loud.

### Step 2: Add producer APIs

Implement shared logging helpers for stage/progress/stagnation.

**Acceptance:**
- callers do not need to write SQL,
- events land in SQLite and JSONL.

### Step 3: Integrate the existing tracker

Wire `BatchProgressTracker` into the new persistence path instead of leaving it
callback-only.

**Acceptance:**
- tracker snapshots can emit run progress,
- no duplicate progress logic remains.

### Step 4: Add query/read helpers

Implement a helper to return the latest active-run status.

**Acceptance:**
- active runs can be listed without reading raw JSONL,
- stagnated/stale runs are visible.

### Step 5: Prove on a non-DIGIMON path if possible

Use an llm_client-side test path to prove the generic contract before consumer
integration.

**Acceptance:**
- tests show stage/progress rows are queryable and accurate.

---

## Required Tests

| Test File | Test Function | What It Verifies |
|-----------|---------------|------------------|
| `tests/test_experiment_log.py` | `test_run_progress_events_persist_and_query_latest_status` | progress rows persist and are queryable |
| `tests/test_experiment_log.py` | `test_run_stage_and_stagnation_events_update_active_status` | stage/stagnation updates alter active status correctly |
| `tests/test_batch_progress.py` | `test_tracker_can_emit_run_progress_snapshots` | existing tracker integrates with the shared path |
| `tests/test_io_log.py` | `test_active_run_progress_summary_reports_latest_stage_and_last_progress` | operator-facing summary is stable |

Existing tests that must pass:

- `tests/test_experiment_log.py`
- `tests/test_io_log.py`
- `tests/test_batch_progress.py`

---

## Acceptance Criteria

- [ ] Long-running progress is a first-class shared observability surface
- [ ] Run progress is tied to existing `run_id`s
- [ ] Stage/progress/stagnation events persist to SQLite and JSONL
- [ ] A query helper can show active run progress without manual log inspection
- [ ] Existing batch progress logic is integrated rather than duplicated
- [ ] Tests prove persistence and latest-status query behavior

---

## Uncertainties

### Q1: New table or generic event table?
**Status:** Open
**Why it matters:** A dedicated table makes queries simpler; a generic event
table is more flexible.
**Plan handling:** Choose the smallest truthful schema that makes active-run
queries cheap and explicit.

### Q2: How much state belongs in `experiment_runs` vs child progress rows?
**Status:** Open
**Why it matters:** Duplicating current stage/progress in the parent run row
helps querying but increases write coupling.
**Plan handling:** Parent-row denormalization is allowed only if it clearly
improves query simplicity.

### Q3: Should the first CLI/query surface live in `llm_client`?
**Status:** Open
**Why it matters:** DIGIMON needs a usable status view, but the shared API
should not become DIGIMON-shaped.
**Plan handling:** Shared helper first; CLI only if real operator use demands it.

---

## Notes

- This plan promotes the earlier batch-progress idea into a broader shared
  long-running observability contract.
- DIGIMON integration is handled separately downstream.
