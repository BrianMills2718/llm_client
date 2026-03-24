# Plan #14: Batch Progress Observability & Stagnation Detection

**Status:** ✅ Complete
**Type:** implementation
**Priority:** High
**Blocked By:** None
**Blocks:** None

---

## Gap

**Current:** llm_client tracks per-call metrics (latency, cost, tokens) but has
no batch-level progress tracking. A DIGIMON graph rebuild stalled at
5200/11656 chunks for 2+ hours with no alert. The only way to detect stalls
is manual checkpoint inspection.

**Target:** Three new capabilities:
1. Batch progress tracking — "5200/11656 done, 3 errors, ~2.1s avg latency"
2. Batch stagnation detection — "last 5 items all failed with identical error"
3. Per-call hang detection — "this extraction call has been running for 30min, skip it"

**Why:** Per CLAUDE.md: "Maximum Observability — Log all state changes with
context." Batch workloads are invisible at the task level. This is a tier-1
observability gap.

---

## References Reviewed

- `llm_client/execution/batch_runtime.py` — already has `on_item_complete`/`on_item_error` hooks and semaphore concurrency
- `llm_client/execution/call_lifecycle.py` — has `_AsyncLLMCallHeartbeatMonitor` with stall detection (per-call, not per-batch)
- `llm_client/execution/timeout_policy.py` — global timeout disable switch, normalize_timeout()
- `agentic_scaffolding/safety/stagnation.py` — proven StagnationDetector (rolling-window error hash)
- `llm_client/io_log.py` — Foundation event logging, SQLite persistence
- `llm_client/BACKLOG.md` — original observation from DIGIMON stall

---

## Pre-made Decisions

1. **Batch progress uses existing hooks** — `on_item_complete`/`on_item_error` in batch_runtime.py already fire per-item. We wrap them with an aggregator, not replace them.
2. **Stagnation reuses agentic_scaffolding's StagnationDetector** — proven pattern, don't reinvent. Import it (agentic_scaffolding is already a dependency in some envs).
3. **Per-call hang timeout is distinct from provider timeout** — `timeout=60` is the provider request timeout. The new `item_timeout_s` wraps the entire item (including retries) in `asyncio.wait_for`.
4. **Events go to io_log** — batch progress events logged to SQLite via existing Foundation schema, not a new persistence layer.
5. **Configurable, not hardcoded** — all thresholds in function signatures with sensible defaults. No env vars for now (YAGNI until someone needs runtime tuning).

---

## Files Affected

- `llm_client/execution/batch_runtime.py` (modify — add progress aggregator, stagnation, item timeout)
- `llm_client/io_log.py` (modify — add `log_batch_progress()` event type)
- `tests/test_batch_progress.py` (create)

---

## Plan

### Step 1: BatchProgressTracker dataclass

```python
@dataclass
class BatchProgressTracker:
    """Aggregates per-item results into batch-level progress."""
    total: int
    completed: int = 0
    errored: int = 0
    total_latency_s: float = 0.0
    started_at: float = field(default_factory=time.monotonic)

    @property
    def pending(self) -> int: ...
    @property
    def elapsed_s(self) -> float: ...
    @property
    def avg_latency_s(self) -> float | None: ...
    @property
    def completion_rate(self) -> float: ...
```

Add to `batch_runtime.py`. Pure data, no dependencies.

### Step 2: Wire progress tracking into batch functions

In `acall_llm_batch_impl` and `acall_llm_structured_batch_impl`:
- Create `BatchProgressTracker(total=len(messages_list))`
- On each item complete: increment `completed`, accumulate latency
- On each item error: increment `errored`
- Every N completions (configurable `progress_interval`, default 100): log progress event via `io_log.log_batch_progress()`
- On batch complete: log final summary

New parameters:
- `progress_interval: int = 100` — log progress every N items
- `on_batch_progress: Callable[[BatchProgressTracker], None] | None = None` — optional callback

### Step 3: Batch stagnation detection

Import `StagnationDetector` from `agentic_scaffolding.validators.framework` — wait, that's the wrong module. Check if it should come from `agentic_scaffolding.safety.stagnation`.

In the error handler:
- Feed error text to `StagnationDetector.record_error()`
- If `StagnationDetected` raised: log a BATCH_STAGNATION foundation event, optionally abort batch

New parameters:
- `stagnation_window: int = 5` — consecutive identical errors before alert
- `abort_on_stagnation: bool = False` — whether to stop the batch

### Step 4: Per-item timeout wrapper

Wrap each item dispatch in `asyncio.wait_for(coro, timeout=item_timeout_s)`:
- If timeout fires: log item as errored with reason "item_timeout"
- Continue to next item (don't block the batch)

New parameter:
- `item_timeout_s: float | None = None` — per-item wall-clock timeout (includes retries). None = no limit.

### Step 5: io_log.log_batch_progress()

New function in io_log.py:
```python
def log_batch_progress(
    task: str,
    trace_id: str,
    total: int,
    completed: int,
    errored: int,
    elapsed_s: float,
    avg_latency_s: float | None,
    stagnation_detected: bool = False,
) -> None:
```

Writes to `batch_progress` table in SQLite + JSONL.

---

## Required Tests

### New Tests (TDD)

| Test File | Test Function | What It Verifies |
|-----------|---------------|------------------|
| `tests/test_batch_progress.py` | `test_tracker_counts` | BatchProgressTracker increments correctly |
| `tests/test_batch_progress.py` | `test_tracker_avg_latency` | Average latency computed correctly |
| `tests/test_batch_progress.py` | `test_progress_callback_fires` | on_batch_progress called every N items |
| `tests/test_batch_progress.py` | `test_stagnation_detected` | 5 identical errors trigger stagnation event |
| `tests/test_batch_progress.py` | `test_stagnation_abort` | abort_on_stagnation=True stops batch |
| `tests/test_batch_progress.py` | `test_item_timeout` | Slow item times out, batch continues |
| `tests/test_batch_progress.py` | `test_item_timeout_logged` | Timeout logged as error with reason |

### Existing Tests (Must Pass)

| Test Pattern | Why |
|--------------|-----|
| `tests/test_public_surface.py` | Public API unchanged |
| `tests/test_client.py` | Core dispatch works |

---

## Acceptance Criteria

- [ ] BatchProgressTracker tracks completed/errored/pending/avg_latency
- [ ] Progress events logged to io_log every N items
- [ ] StagnationDetector imported from agentic_scaffolding, wired into error handler
- [ ] Stagnation event logged when consecutive identical errors detected
- [ ] Per-item timeout wraps entire item dispatch (including retries)
- [ ] Timed-out items logged as errors, batch continues
- [ ] All new tests pass
- [ ] Existing tests pass
- [ ] No new silent failures

---

## Notes

- The call_lifecycle.py `HeartbeatMonitor` handles per-call stall detection.
  This plan handles per-BATCH stall detection. They're complementary, not
  overlapping.
- agentic_scaffolding's StagnationDetector only detects identical consecutive
  errors. A future enhancement could detect "progress stall" (no completions
  for N minutes) — but that's a different pattern. Start with error-hash
  stagnation which has a proven implementation.
- Per-turn agent timeout (MCP loop) is NOT in this plan — it's a separate,
  higher-friction change. This plan covers batch workloads only.
