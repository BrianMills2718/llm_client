# Plan 12: Progress-Aware Idle Detection

**Status:** Complete
**Type:** implementation
**Priority:** High
**Blocked By:** None
**Blocks:** More truthful stall detection for long-running streaming and background-polled calls

---

## Gap

**Current:** Plan 09 gives us lifecycle `heartbeat` and non-destructive
`stalled` markers plus an active-call query, but stall detection is still
elapsed-time based. That is useful visibility, not true idle detection.

**Target:** `llm_client` reports progress-aware idle state only where it has a
truthful progress signal, starting with streaming calls and background-polled
Responses calls.

**Why:** if a call is still progressing, we should say so and let it run. If a
call stops making observable progress on a path that should progress, that is
the right place to classify it as idle/stalled.

---

## References Reviewed

- `CLAUDE.md` - canonical timeout/liveness rule
- `docs/adr/0009-long-thinking-background-polling.md` - current progress-like polling path
- `docs/adr/0013-provider-timeouts-are-not-the-default-liveness-mechanism.md` - timeout baseline
- `docs/adr/0014-emit-heartbeats-and-non-destructive-stall-markers-for-long-running-calls.md` - lifecycle baseline
- `docs/adr/0016-prefer-progress-aware-idle-detection-over-wall-clock-stall-inference.md` - decision for this slice
- `llm_client/client.py` - lifecycle emission and background polling helpers
- `llm_client/stream_runtime.py` - streaming execution boundary
- `llm_client/streaming.py` - chunk accumulation wrappers
- `llm_client/structured_runtime.py` - current opaque non-streaming structured path
- `llm_client/observability/query.py` - active-call query surface

---

## Files Expected To Change

- `docs/plans/12_progress_aware_idle_detection.md` (modify)
- `docs/plans/CLAUDE.md` (modify)
- `llm_client/foundation.py` (modify)
- `llm_client/client.py` (modify)
- `llm_client/stream_runtime.py` (modify)
- `llm_client/streaming.py` (modify)
- `llm_client/observability/query.py` (modify)
- `llm_client/text_runtime.py` (modify)
- `tests/test_foundation.py` (modify)
- `tests/test_client.py` (modify)
- `tests/test_io_log.py` (modify)

---

## Plan

### Step 1: Extend the lifecycle contract truthfully

- Add a new lifecycle phase: `"progress"`.
- Add optional lifecycle payload fields:
  - `progress_observable: bool | None`
  - `progress_source: str | None`
  - `progress_event_count: int | None`
- Keep `heartbeat` as "still waiting".
- Let the active-call query derive:
  - `last_progress_at`
  - `idle_for_s`
  - `activity_state`

### Step 2: Instrument streaming progress

- Start streaming lifecycle tracking lazily on first stream consumption.
- Emit `"started"` with `progress_observable=True`.
- Emit `"progress"` on every observed stream chunk.
- Use `progress_source="stream_chunk"`.
- Only treat a stream as progress-aware stalled after at least one chunk has
  been observed.

### Step 3: Instrument background-polled Responses progress

- When a Responses call enters background polling, switch the monitor to
  `progress_observable=True`.
- Emit `"progress"` on each successful background poll response, including the
  terminal completion poll.
- Use `progress_source="background_poll"`.
- Let progress-aware stall detection for this path key off the time since the
  last successful poll.

### Step 4: Keep opaque calls honest

- Non-streaming opaque text/structured calls continue to emit liveness
  `"heartbeat"` events only.
- Do not emit `"progress"` for those paths.
- Do not classify them as progress-aware stalled in the active-call query.

### Step 5: Upgrade the query/report surface

- `get_active_llm_calls()` returns per-call metadata including:
  - `progress_observable`
  - `progress_source`
  - `progress_event_count`
  - `last_progress_at`
  - `idle_for_s`
  - `activity_state`
- Callers can distinguish:
  - `waiting`
  - `progressing`
  - `idle`

---

## Required Tests

### New Tests

| Test File | Test Function | What It Verifies |
|-----------|---------------|------------------|
| `tests/test_client.py` | `test_stream_llm_updates_progress_state_from_chunks` | Streaming calls emit progress-aware lifecycle updates when chunks arrive |
| `tests/test_client.py` | `test_astream_llm_updates_progress_state_from_chunks` | Async streaming calls emit progress-aware lifecycle updates when chunks arrive |
| `tests/test_client.py` | `test_background_polling_updates_progress_state_from_successful_polls` | Background-polled calls advance progress state on successful poll activity |
| `tests/test_io_log.py` | `test_get_active_llm_calls_reports_progress_metadata` | Query layer exposes progress-observable state, last progress, and idle duration |

### Existing Tests

| Test Pattern | Why |
|--------------|-----|
| `tests/test_foundation.py -k lifecycle` | Lifecycle schema changes must stay valid |
| `tests/test_client.py -k 'heartbeat or stalled or lifecycle'` | Existing liveness semantics must not regress |
| `tests/test_public_surface.py` | Public query surface must remain truthful if new helpers are exported |

---

## Acceptance Criteria

- [x] Lifecycle contract distinguishes liveness from progress
- [x] Streaming calls emit real progress-aware updates
- [x] Background-polled Responses calls emit real progress-aware updates
- [x] Active-call query surface reports progress-aware idle state
- [x] Opaque non-streaming structured calls are not mislabeled as
      progress-aware stalled
- [x] Focused lifecycle/progress tests pass

## Completion Evidence

- `pytest -q tests/test_foundation.py -k lifecycle`
- `pytest -q tests/test_client.py -k 'lifecycle or background_polling_updates_progress_state_from_successful_polls or stream_llm_updates_progress_state_from_chunks or astream_llm_updates_progress_state_from_chunks'`
- `pytest -q tests/test_io_log.py -k 'get_active_llm_calls'`
- `mypy --follow-imports=silent llm_client/foundation.py llm_client/client.py llm_client/text_runtime.py llm_client/stream_runtime.py llm_client/streaming.py llm_client/observability/query.py`

---

## Design Notes

- **New phase `"progress"`:** Distinct from `"heartbeat"` to preserve the
  semantic contract. Heartbeat means "still waiting". Progress means
  "demonstrably moving forward".
- **Background polling starts opaque:** the public wrapper does not know in
  advance whether a Responses call will require background polling, so
  `progress_observable` becomes true only once the runtime actually enters the
  polling path.

## Risks / Assumptions

- Current non-streaming structured transports remain fundamentally opaque; this
  plan does not pretend otherwise.
- Some provider SDK paths may need additive wrapper instrumentation rather than
  client-facade-only changes.
- If true progress-aware support is later required for opaque structured calls,
  that may require a transport redesign rather than another small observability
  patch.
