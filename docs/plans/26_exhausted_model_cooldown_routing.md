# Plan #26: exhausted-model cooldown routing

**Status:** Complete
**Type:** implementation
**Priority:** High
**Blocked By:** None
**Blocks:** Long-running batch jobs that repeatedly probe exhausted primary models

---

## Gap

**Current:** `llm_client` now treats Gemini spend-cap and daily-cap `429 RESOURCE_EXHAUSTED` failures as non-retryable within a single call, but each new call still starts from the same primary model. Batch jobs can therefore hit the same exhausted model thousands of times across chunks before falling through to the same fallback chain.

**Target:** When a model fails with a quota/spend-cap exhaustion signal, `llm_client` should mark it temporarily unavailable in-process and skip it during routing until a cooldown expires. Routing traces and logs should make the suppression explicit.

**Why:** The current behavior still wastes throughput, log volume, and provider quota on repeated known-bad probes. A process-local circuit breaker is the missing layer between per-call retry logic and long-running workloads.

---

## References Reviewed

- `CLAUDE.md` - repo workflow rules and verification expectations
- `docs/plans/CLAUDE.md` - plan index
- `llm_client/core/client_dispatch.py` - call-plan resolution hook used by runtimes
- `llm_client/core/routing.py` - resolved call-plan structure
- `llm_client/core/errors.py` - shared quota/spend-cap patterns and classification
- `llm_client/execution/execution_kernel.py` - shared fallback loop
- `tests/test_routing.py` - routing plan assertions
- `tests/test_execution_kernel.py` - fallback kernel tests

---

## Files Affected

- `docs/plans/CLAUDE.md` (modify)
- `docs/plans/26_exhausted_model_cooldown_routing.md` (create)
- `llm_client/core/client_dispatch.py` (modify)
- `llm_client/core/model_availability.py` (create)
- `llm_client/execution/execution_kernel.py` (modify)
- `tests/test_execution_kernel.py` (modify)
- `tests/test_routing.py` (modify)

---

## Plan

### Steps

1. Add a small process-local model-availability registry that records temporary unavailability windows for quota/spend-cap exhaustion errors.
2. Wire fallback execution so an exhausted model is recorded when a failure is observed.
3. Filter resolved call plans through the availability registry before runtimes begin execution.
4. Log and trace which models were suppressed so the behavior is explicit in diagnostics.
5. Add focused tests covering registration, suppression, expiry, and fallback continuity.

---

## Required Tests

### New Tests (TDD)

| Test File | Test Function | What It Verifies |
|-----------|---------------|------------------|
| `tests/test_execution_kernel.py` | `test_run_async_with_fallback_records_exhausted_model_for_future_calls` | Exhausted-model failures are registered at fallback time |
| `tests/test_routing.py` | `test_resolve_call_plan_skips_temporarily_unavailable_models` | Routing suppresses cooled-down models and promotes the first available fallback |

### Existing Tests (Must Pass)

| Test Pattern | Why |
|--------------|-----|
| `tests/test_execution_kernel.py` | Shared retry/fallback helpers still behave normally |
| `tests/test_routing.py` | Base routing normalization and dedupe semantics remain intact |
| `tests/test_client.py -k 'gemini_monthly_spend_cap or gemini_daily_request_cap'` | Prior quota classification fix stays correct |

---

## Acceptance Criteria

- [ ] Exhausted primary models are skipped on subsequent calls within the same process
- [ ] Routing logs/traces reveal when a model was suppressed due to recent exhaustion
- [ ] Cooldown behavior expires automatically without manual cleanup
- [ ] Existing fallback behavior still works for non-exhaustion failures
- [ ] Targeted tests pass

---

## Notes

- This is intentionally process-local. Cross-process/global provider health belongs in a separate design if we need it.
- Cooldown durations should be short and configurable enough to avoid stale suppression after a user raises a spend cap, while still preventing high-frequency reprobes.
- Verification completed with:
  - `pytest -q tests/test_execution_kernel.py tests/test_routing.py`
  - `pytest -q tests/test_client.py -k 'gemini_monthly_spend_cap or gemini_daily_request_cap or quota_exceeded_not_retried or transient_rate_limit_is_retried or rate_limit_quota_with_retry_delay_is_retried'`
