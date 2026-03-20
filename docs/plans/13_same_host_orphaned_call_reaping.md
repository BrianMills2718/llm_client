# Plan 13: Same-Host Orphaned Call Reaping

**Status:** Complete
**Type:** implementation
**Priority:** High
**Blocked By:** None
**Blocks:** Trustworthy active-call reporting for interrupted local consumers

---

## Gap

**Current:** `llm_client` lifecycle events and `get_active_llm_calls()` show
live non-terminal calls well while the originating process remains alive.
However, if a local process is interrupted before it emits `completed` or
`failed`, the active-call query can keep showing that call as active because it
has no same-host process identity.

**Target:** lifecycle events carry additive same-host process identity, and the
active-call query excludes non-terminal rows when it can prove that the
originating same-host process is gone.

**Why:** a long-running consumer needs to tell the difference between:

1. still running,
2. still waiting,
3. explicitly orphaned by local process death.

Without that, the active-call query is helpful but not trustworthy after local
interrupts.

---

## References Reviewed

- `CLAUDE.md` - liveness / timeout rules
- `docs/adr/0013-provider-timeouts-are-not-the-default-liveness-mechanism.md`
- `docs/adr/0014-emit-heartbeats-and-non-destructive-stall-markers-for-long-running-calls.md`
- `docs/adr/0016-prefer-progress-aware-idle-detection-over-wall-clock-stall-inference.md`
- `docs/adr/0017-exclude-same-host-orphaned-calls-from-the-active-call-query.md`
- `llm_client/client.py` - lifecycle emission
- `llm_client/foundation.py` - lifecycle payload schema
- `llm_client/observability/query.py` - active-call query logic
- `tests/test_foundation.py`
- `tests/test_client.py`
- `tests/test_io_log.py`

---

## Files Expected To Change

- `docs/adr/0017-exclude-same-host-orphaned-calls-from-the-active-call-query.md` (new)
- `docs/plans/13_same_host_orphaned_call_reaping.md` (new)
- `docs/adr/README.md` (modify)
- `docs/plans/CLAUDE.md` (modify)
- `llm_client/foundation.py` (modify)
- `llm_client/client.py` (modify)
- `llm_client/observability/query.py` (modify)
- `tests/test_foundation.py` (modify)
- `tests/test_client.py` (modify)
- `tests/test_io_log.py` (modify)

---

## Plan

### Step 1: Extend the lifecycle payload

- Add additive process-identity fields:
  - `host_name`
  - `process_id`
  - `process_start_token`
- Keep the fields optional so older producers and cross-host readers remain
  compatible.

### Step 2: Emit process identity from the public call wrappers

- Populate the new fields on lifecycle events emitted by the public call
  wrappers.
- Use a Linux procfs start token when available so same-host queries can detect
  PID reuse more safely than PID-only checks.

### Step 3: Reap same-host orphaned calls in the query

- In `get_active_llm_calls()`, determine whether a same-host process is:
  - definitely alive,
  - definitely gone,
  - or unknown.
- Exclude the record only in the `definitely gone` case.
- Surface `process_alive` for retained records so callers can see whether the
  liveness check was conclusive.

### Step 4: Prove it with focused tests

- Validate lifecycle schema acceptance for process identity.
- Validate lifecycle emission includes process identity.
- Validate same-host orphan exclusion.
- Validate unknown-liveness records remain visible.

---

## Required Tests

### New Tests

| Test File | Test Function | What It Verifies |
|-----------|---------------|------------------|
| `tests/test_io_log.py` | `test_get_active_llm_calls_excludes_same_host_orphaned_processes` | Same-host orphaned non-terminal calls are excluded from active-call results |
| `tests/test_io_log.py` | `test_get_active_llm_calls_keeps_records_when_process_liveness_is_unknown` | Records stay visible when liveness cannot be determined honestly |

### Existing Tests

| Test Pattern | Why |
|--------------|-----|
| `tests/test_foundation.py -k lifecycle` | Lifecycle schema additions must remain valid |
| `tests/test_client.py -k lifecycle` | Public lifecycle emission must include the additive fields without regressing event order |
| `tests/test_io_log.py -k get_active_llm_calls` | Active-call query behavior must remain truthful |

---

## Acceptance Criteria

- [x] Lifecycle schema accepts additive process-identity fields
- [x] Public lifecycle emission includes same-host process identity
- [x] Same-host orphaned non-terminal calls are excluded from the active-call query
- [x] Unknown-liveness rows remain visible instead of being guessed away
- [x] Focused lifecycle/query tests pass

## Completion Evidence

- `pytest -q tests/test_foundation.py -k lifecycle`
- `pytest -q tests/test_client.py -k lifecycle`
- `pytest -q tests/test_io_log.py -k get_active_llm_calls`
- `mypy --follow-imports=silent llm_client/foundation.py llm_client/client.py llm_client/observability/query.py`

---

## Design Notes

- `process_start_token` is best-effort. On Linux, it uses procfs start ticks so
  a recycled PID can be distinguished from the original process.
- `process_alive` is tri-state:
  - `True` when the same-host process is definitely still alive,
  - `False` when the process is definitely gone or mismatched,
  - `None` when liveness cannot be determined honestly.

## Risks / Assumptions

- Older rows without process identity remain ambiguous and cannot be repaired
  retroactively.
- Same-host orphan detection is only as strong as the available platform
  signals.
- Remote worker/process liveness remains out of scope for this slice.
