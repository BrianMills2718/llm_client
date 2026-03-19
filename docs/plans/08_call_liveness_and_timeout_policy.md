# Plan 08: Call Liveness and Timeout Policy

**Status:** Complete
**Type:** implementation
**Priority:** High
**Blocked By:** None
**Blocks:** Better orchestration controls for long-running structured calls

---

## Gap

**Current:** `llm_client` has a shared provider-timeout policy, but normal
non-streaming text and structured calls do not emit a generic
`started -> completed/failed` lifecycle record. Operators can see completed
call logs after the fact, but not a clear in-flight lifecycle for slow calls.

**Target:** Provider timeouts are documented as transport controls rather than
the default liveness mechanism, and public non-streaming text/structured calls
emit Foundation-backed lifecycle events that make in-flight work observable.

**Why:** Slow calls should be visible before they finish. Long-running
structured extraction work should not depend on blunt request timeouts for basic
operability.

---

## References Reviewed

- `CLAUDE.md` - repo governance and observability expectations
- `docs/adr/0001-model-identity-v0.md` - stable model identity baseline
- `docs/adr/0002-routing-config-precedence.md` - routing and config authority rules
- `docs/adr/0003-warning-taxonomy.md` - warning contract and operator-facing warning semantics
- `docs/adr/0004-result-model-semantics-migration.md` - result-shape and backward-compatibility contract
- `docs/adr/0005-reason-code-registry-governance.md` - explicit machine-readable failure-code discipline
- `docs/adr/0006-actor-id-issuance-policy.md` - canonical Foundation actor identity policy
- `docs/adr/0007-observability-contract-boundary.md` - existing observability boundary
- `docs/adr/0009-long-thinking-background-polling.md` - prior timeout-related runtime decision
- `docs/adr/0010-cross-project-runtime-substrate.md` - shared control-plane boundary for runtime changes
- `llm_client/client.py` - public call wrappers and shared logging helpers
- `llm_client/text_runtime.py` - text-call control flow and timeout normalization
- `llm_client/structured_runtime.py` - structured-call control flow and timeout normalization
- `llm_client/foundation.py` - Foundation event schema boundary
- `llm_client/io_log.py` - persistent Foundation-event logging
- `llm_client/timeout_policy.py` - shared timeout policy helpers
- `tests/test_foundation.py` - Foundation validation coverage
- `tests/test_client.py` - public client contract coverage

---

## Files Affected

- `CLAUDE.md` (modify)
- `AGENTS.md` (regenerate)
- `docs/adr/0013-provider-timeouts-are-not-the-default-liveness-mechanism.md` (create)
- `docs/adr/README.md` (modify)
- `docs/plans/08_call_liveness_and_timeout_policy.md` (create)
- `docs/plans/CLAUDE.md` (modify)
- `llm_client/foundation.py` (modify)
- `llm_client/client.py` (modify)
- `tests/test_foundation.py` (modify)
- `tests/test_client.py` (modify)

---

## Plan

### Step 1: Lock the policy boundary

- add an ADR that distinguishes provider request timeouts from liveness
  management
- define the first implementation slice as lifecycle observability, not full
  orchestration

### Step 2: Add a Foundation lifecycle event

- extend the Foundation schema with a typed lifecycle event for normal LLM
  calls
- include a stable per-call lifecycle id plus task/trace/run linkage

### Step 3: Emit lifecycle events from public call wrappers

- emit `started` before dispatch for public text and structured calls after
  tags and timeout policy are normalized
- emit `completed` or `failed` on terminal exit

### Step 4: Verify the first slice

- add Foundation validation tests
- add public-wrapper tests for sync/async text and structured call lifecycle
  logging

---

## Required Tests

### New Tests

| Test File | Test Function | What It Verifies |
|-----------|---------------|------------------|
| `tests/test_foundation.py` | `test_validate_foundation_event_llm_call_lifecycle_shape` | Foundation validation accepts the new lifecycle event payload |
| `tests/test_client.py` | `test_call_llm_emits_lifecycle_foundation_events` | Sync text calls emit `started` and `completed` lifecycle events |
| `tests/test_client.py` | `test_call_llm_failure_emits_failed_lifecycle_event` | Sync text failures emit a terminal `failed` lifecycle event |
| `tests/test_client.py` | `test_call_llm_structured_emits_lifecycle_foundation_events` | Sync structured calls emit `started` and `completed` lifecycle events |

### Existing Tests

| Test Pattern | Why |
|--------------|-----|
| `tests/test_structured_runtime.py` | Structured runtime behavior must remain unchanged |
| `tests/test_models.py` | Recent task-selection changes must stay intact |

---

## Acceptance Criteria

- [x] ADR and plan indices are updated
- [x] Foundation validation accepts the new lifecycle event type
- [x] Public text and structured wrappers emit `started` and terminal lifecycle events
- [x] Failed calls emit explicit lifecycle failures with error metadata
- [x] `pytest -q tests/test_foundation.py tests/test_client.py -k "lifecycle or prompt_ref or timeout_policy"` passes

**Verified:** 2026-03-19
**Evidence:**

- `pytest -q tests/test_foundation.py tests/test_client.py -k "lifecycle or prompt_ref or timeout_policy"`
- `pytest -q tests/test_structured_runtime.py`
- `pytest -q tests/test_client.py -k "call_llm_failure_emits_failed_lifecycle_event"`
- `mypy --follow-imports=silent llm_client/foundation.py llm_client/client.py tests/test_foundation.py`

---

## Notes

- This is a first slice, not the full progress-aware orchestration system.
- Streaming, embeddings, and background polling are intentionally deferred until
  the base lifecycle signal is in place.
