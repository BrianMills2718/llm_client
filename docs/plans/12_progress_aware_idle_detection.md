# Plan 12: Progress-Aware Idle Detection

**Status:** Planned
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

- `docs/adr/README.md` (modify)
- `docs/plans/12_progress_aware_idle_detection.md` (create)
- `docs/plans/CLAUDE.md` (modify)
- `llm_client/foundation.py` (modify)
- `llm_client/client.py` (modify)
- `llm_client/stream_runtime.py` (modify)
- `llm_client/streaming.py` (modify)
- `llm_client/observability/query.py` (modify)
- `llm_client/io_log.py` (modify if compatibility shim needs extension)
- `tests/test_foundation.py` (modify)
- `tests/test_client.py` (modify)
- `tests/test_io_log.py` (modify)
- `tests/test_gpt5_background_polling.py` or equivalent polling coverage file (modify if present)

---

## Plan

### Step 1: Extend the lifecycle contract truthfully

- add additive lifecycle progress metadata and/or events
- keep `heartbeat` as "still waiting"
- define how active-call queries compute:
  - `progress_observable`
  - `last_progress_at`
  - `idle_for_s`

### Step 2: Instrument streaming progress

- emit progress-aware lifecycle signals when stream chunks/tool deltas arrive
- reset idle tracking on each observed progress event

### Step 3: Instrument background-polled Responses progress

- emit progress-aware lifecycle signals for successful polls and meaningful
  status transitions
- allow progress-aware idle detection on this path

### Step 4: Keep opaque calls honest

- non-streaming opaque text/structured calls continue to emit liveness
  heartbeats
- do not classify them as progress-aware stalled unless a real progress signal
  is introduced

### Step 5: Upgrade the query/report surface

- active-call queries distinguish:
  - waiting
  - progressing
  - idle/stalled-on-progress-aware-path

---

## Required Tests

### New Tests

| Test File | Test Function | What It Verifies |
|-----------|---------------|------------------|
| `tests/test_client.py` | `test_stream_llm_updates_progress_state_from_chunks` | Streaming calls emit progress-aware lifecycle updates when chunks arrive |
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

- [ ] Lifecycle contract distinguishes liveness from progress
- [ ] Streaming calls emit real progress-aware updates
- [ ] Background-polled Responses calls emit real progress-aware updates
- [ ] Active-call query surface reports progress-aware idle state
- [ ] Opaque non-streaming structured calls are not mislabeled as
      progress-aware stalled
- [ ] Focused lifecycle/progress tests pass

---

## Risks / Assumptions

- Current non-streaming structured transports remain fundamentally opaque; this
  plan does not pretend otherwise.
- Some provider SDK paths may need additive wrapper instrumentation rather than
  client-facade-only changes.
- If true progress-aware support is later required for opaque structured calls,
  that may require a transport redesign rather than another small observability
  patch.
