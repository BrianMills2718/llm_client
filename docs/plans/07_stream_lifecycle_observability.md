# Plan 07: Stream Lifecycle Heartbeat and Stagnation Visibility

**Status:** Complete  
**Type:** implementation  
**Priority:** High  
**Blocked By:** None  
**Blocks:** None

---

## Gap

**Current:** Stream paths emit result fragments but do not always emit complete lifecycle
state transitions (`started/progress/completed/failed`) in a durable table.

**Target:** Sync/async streaming should emit truthful lifecycle rows for:

- stream creation failures,
- natural completion after iteration,
- iterator failures before completion,
- heartbeat/stall settings on stream calls.

**Why:** `get_active_llm_calls` should represent stream work accurately, and
users need a way to detect hung or stalled in-flight calls.

## References Reviewed

- `llm_client/stream_runtime.py`
- `llm_client/client.py` (monitor + lifecycle contracts)
- `llm_client/streaming.py`
- `tests/test_client_lifecycle.py`
- `docs/adr/0013-stream-lifecycle-heartbeat-observability.md`

## Files Affected

- `llm_client/stream_runtime.py`
- `tests/test_client_lifecycle.py`
- `docs/adr/0013-stream-lifecycle-heartbeat-observability.md`
- `docs/adr/README.md`
- `docs/plans/CLAUDE.md`
- `README.md` (stream usage note)

## Plan

1. Restore sync/async stream wrappers to emit terminal lifecycle rows from iterator
   adapters.
2. Ensure progress events fire on chunk-level consumption.
3. Validate stream constructor argument consistency (including async model forwarding).
4. Add lifecycle regression tests for sync/async stream success and iteration-failure.
5. Update documentation with stream observability note and ADR + plan index updates.

## Required Tests

| Test File | Test Function | What It Verifies |
|-----------|---------------|------------------|
| `tests/test_client_lifecycle.py` | `test_stream_llm_emits_started_progress_completed_lifecycle` | non-iterative completion lifecycle |
| `tests/test_client_lifecycle.py` | `test_stream_llm_emits_failed_lifecycle_on_iteration_error` | iterator failure emits failed row |
| `tests/test_client_lifecycle.py` | `test_astream_llm_emits_started_progress_completed_lifecycle` | async completion lifecycle |
| `tests/test_client_lifecycle.py` | `test_astream_llm_emits_failed_lifecycle_on_iteration_error` | async iterator failure emits failed row |

## Acceptance Criteria

- [x] Sync stream lifecycle rows match `started -> progress -> completed` on success.
- [x] Sync stream lifecycle rows match `started -> progress -> failed` on iteration error.
- [x] Async stream lifecycle rows match same patterns.
- [x] No regressions in existing stream lifecycle tests.
- [x] README and ADR references updated to point to heartbeat/stall behavior.
