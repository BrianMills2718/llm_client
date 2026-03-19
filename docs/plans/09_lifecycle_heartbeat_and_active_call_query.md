# Plan 09: Lifecycle Heartbeat and Active Call Query

**Status:** In Progress
**Type:** implementation
**Priority:** High
**Blocked By:** None
**Blocks:** Better operator tooling for long-running extraction/eval runs

---

## Gap

**Current:** public non-streaming calls emit `started/completed/failed`, but
they do not emit liveness heartbeats and there is no shared query helper for
"what is still running right now?".

**Target:** long-running public text and structured calls emit periodic
`heartbeat` events, optionally emit non-terminal `stalled` markers after a
threshold, and shared observability exposes a helper for current active calls.

**Why:** slow calls should remain visible without relying on provider timeouts
or manual log diffing.

---

## References Reviewed

- `CLAUDE.md` - canonical repo governance and observability expectations
- `docs/adr/0001-model-identity-v0.md` - result identity compatibility contract still applies to lifecycle metadata
- `docs/adr/0002-routing-config-precedence.md` - routing precedence remains unchanged by lifecycle instrumentation
- `docs/adr/0003-warning-taxonomy.md` - lifecycle work must not blur warning/error channels
- `docs/adr/0004-result-model-semantics-migration.md` - lifecycle emission must preserve current result semantics
- `docs/adr/0005-reason-code-registry-governance.md` - lifecycle failure typing must stay additive and auditable
- `docs/adr/0006-actor-id-issuance-policy.md` - lifecycle events reuse existing Foundation actor issuance policy
- `docs/adr/0007-observability-contract-boundary.md` - observability implementation boundary
- `docs/adr/0009-long-thinking-background-polling.md` - long-running call visibility must not regress background-call observability
- `docs/adr/0010-cross-project-runtime-substrate.md` - lifecycle semantics are part of the shared cross-project contract
- `docs/adr/0012-shared-data-plane-boundary.md` - new active-call query remains in the shared data-plane surface
- `docs/adr/0013-provider-timeouts-are-not-the-default-liveness-mechanism.md` - lifecycle baseline already accepted
- `docs/adr/0014-emit-heartbeats-and-non-destructive-stall-markers-for-long-running-calls.md` - heartbeat/stall decision for this slice
- `llm_client/client.py` - current public wrapper lifecycle emission
- `llm_client/foundation.py` - Foundation lifecycle schema
- `hooks/pre-commit` - governed staged doc-coupling gate behavior
- `llm_client/io_log.py` - compatibility facade and SQLite sinks
- `llm_client/observability/query.py` - canonical query boundary
- `tests/test_client.py` - public client lifecycle coverage
- `tests/test_foundation.py` - Foundation validation coverage
- `tests/test_io_log.py` - observability query coverage and isolated DB fixture patterns

---

## Files Affected

- `CLAUDE.md` (modify)
- `AGENTS.md` (regenerate)
- `docs/adr/0014-emit-heartbeats-and-non-destructive-stall-markers-for-long-running-calls.md` (create)
- `docs/adr/README.md` (modify)
- `docs/plans/09_lifecycle_heartbeat_and_active_call_query.md` (create)
- `docs/plans/CLAUDE.md` (modify)
- `llm_client/client.py` (modify)
- `llm_client/foundation.py` (modify)
- `hooks/pre-commit` (modify)
- `llm_client/io_log.py` (modify)
- `llm_client/observability/query.py` (modify)
- `llm_client/observability/__init__.py` (modify)
- `llm_client/__init__.py` (modify)
- `tests/test_client.py` (modify)
- `tests/test_foundation.py` (modify)
- `tests/test_io_log.py` (modify)
- `tests/test_public_surface.py` (modify)

---

## Plan

### Step 1: Extend lifecycle semantics

- allow lifecycle phases `heartbeat` and `stalled`
- keep `stalled` non-terminal and non-destructive

### Step 2: Add thin heartbeat emitters

- emit periodic `heartbeat` events while a public text/structured call is in
  flight
- support sync and async wrappers
- add optional stall threshold emission without cancelling the call

### Step 3: Add active-call query surface

- expose a canonical query helper that returns the latest known active
  non-terminal lifecycle state for public calls
- keep `io_log.py` as a compatibility facade

### Step 4: Verify the slice

- add validation tests for new lifecycle phases
- add wrapper tests for heartbeat/stall emission
- add query tests for active call discovery
- ensure governed pre-commit doc-coupling checks use staged state for this
  slice instead of committed-only branch diffs

---

## Required Tests

### New Tests

| Test File | Test Function | What It Verifies |
|-----------|---------------|------------------|
| `tests/test_foundation.py` | `test_validate_foundation_event_llm_call_lifecycle_accepts_heartbeat_phase` | Foundation schema accepts heartbeat lifecycle events |
| `tests/test_client.py` | `test_call_llm_emits_heartbeat_and_stalled_lifecycle_events` | Sync text calls emit heartbeat and stalled events before terminal completion |
| `tests/test_client.py` | `test_acall_llm_emits_heartbeat_lifecycle_events` | Async text calls emit heartbeat events while in flight |
| `tests/test_client.py` | `test_call_llm_structured_emits_heartbeat_lifecycle_events` | Sync structured calls emit heartbeat events while in flight |
| `tests/test_io_log.py` | `test_get_active_llm_calls_returns_latest_non_terminal_lifecycle_state` | Observability query returns active calls from Foundation lifecycle records |
| `tests/test_public_surface.py` | `test_grouped_exports_flatten_to_public_surface_without_duplicates` | New active-call helper is reflected truthfully in the top-level public surface |

### Existing Tests

| Test Pattern | Why |
|--------------|-----|
| `tests/test_foundation.py -k lifecycle` | Lifecycle schema changes must stay valid |
| `tests/test_structured_runtime.py` | Structured runtime behavior must remain unchanged |

---

## Acceptance Criteria

- [ ] ADR and plan indices are updated
- [ ] Lifecycle schema accepts `heartbeat` and `stalled`
- [ ] Public text/structured wrappers can emit heartbeats without changing provider behavior
- [ ] `stalled` is emitted as a non-terminal observability event only
- [ ] Shared observability can query currently active calls
- [ ] `pytest -q tests/test_foundation.py tests/test_client.py tests/test_io_log.py -k 'lifecycle or active_llm_calls'` passes

---

## Notes

- This slice intentionally does not add automatic cancellation.
- Heartbeats mean "runtime still waiting", not "provider sent token progress".
