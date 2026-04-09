# Plan #25: Gemini exhaustion fallback hardening

**Status:** Complete
**Type:** implementation
**Priority:** High
**Blocked By:** None
**Blocks:** Digimon GraphRAG rebuild recovery on April 9, 2026

---

## Gap

**Current:** `llm_client` retries some Gemini `429 RESOURCE_EXHAUSTED` failures that are effectively exhausted-for-this-window conditions, including monthly spend-cap errors. Digimon GraphRAG also still defaults its extraction fallback chain to non-Gemini providers before the Gemini alternatives that remain available.

**Target:** `llm_client` should classify Gemini spend-cap and daily-cap exhaustion as non-retryable so model fallback engages immediately, while keeping retryable minute-scale throttles unchanged. Digimon GraphRAG should prefer `gemini/gemini-3-flash` and `gemini/gemini-2.5-flash-lite` before cross-provider fallbacks in the reliability lane.

**Why:** Repeating same-model retries after a permanent-or-day-bounded Gemini exhaustion event wastes time, budget, and throughput during long-running GraphRAG builds. The fallback chain should first exploit still-available Gemini capacity before crossing to more expensive or behaviorally different providers.

---

## References Reviewed

- `CLAUDE.md` - repo workflow and verification rules
- `docs/plans/CLAUDE.md` - plan index and numbering
- `llm_client/core/errors.py` - shared error classification and quota patterns
- `llm_client/execution/retry.py` - retryability decision path for per-model retries
- `tests/test_client.py` - existing retry/quota coverage
- `/home/brian/projects/Digimon_for_KG_application/eval/prebuild_graph.py` - GraphRAG build fallback wiring
- `/home/brian/projects/Digimon_for_KG_application/Option/Config2.yaml` - current GraphRAG runtime fallback chain
- `/home/brian/projects/Digimon_for_KG_application/logs/plan37_clean_rebuild_tmux.log` - observed Gemini spend-cap retry/fallback behavior

---

## Files Affected

- `docs/plans/CLAUDE.md` (modify)
- `docs/plans/25_gemini_exhaustion_fallback_hardening.md` (create)
- `llm_client/core/errors.py` (modify)
- `llm_client/execution/retry.py` (modify)
- `tests/test_client.py` (modify)
- `/home/brian/projects/Digimon_for_KG_application/Option/Config2.yaml` (modify)
- `/home/brian/projects/Digimon_for_KG_application/tests/unit/test_prebuild_graph_cli.py` (modify)

---

## Plan

### Steps

1. Extend shared quota/exhaustion message patterns in `llm_client/core/errors.py` to include Gemini spend-cap and daily-cap wording.
2. Reuse the same exhaustion patterns in `llm_client/execution/retry.py` so retryability and wrapped error classification stay aligned.
3. Add focused tests proving monthly spend-cap and daily request-cap 429s fail immediately instead of retrying.
4. Update Digimon GraphRAG fallback defaults in `Option/Config2.yaml` to prefer `gemini/gemini-3-flash` and `gemini/gemini-2.5-flash-lite`.
5. Update the Digimon prebuild CLI test fixture to reflect the intended reliability-lane fallback order.

---

## Required Tests

### New Tests (TDD)

| Test File | Test Function | What It Verifies |
|-----------|---------------|------------------|
| `tests/test_client.py` | `test_gemini_monthly_spend_cap_not_retried` | Gemini spend-cap 429s are treated as immediate fallback/abort conditions |
| `tests/test_client.py` | `test_gemini_daily_request_cap_not_retried` | Gemini daily request-cap 429s are treated as immediate fallback/abort conditions |

### Existing Tests (Must Pass)

| Test Pattern | Why |
|--------------|-----|
| `tests/test_client.py -k "quota or rate_limit"` | Existing retry/quota behavior must remain correct |
| `/home/brian/projects/Digimon_for_KG_application/tests/unit/test_prebuild_graph_cli.py` | GraphRAG build wiring still honors lane-policy and configured fallbacks |

---

## Acceptance Criteria

- [ ] `llm_client` no longer retries Gemini monthly spend-cap failures on the same model
- [ ] `llm_client` no longer retries Gemini daily-cap failures on the same model
- [ ] Retryable rate-limit cases with explicit retry hints still retry
- [ ] Digimon GraphRAG reliability-lane fallback order prefers Gemini 3 Flash then Gemini 2.5 Flash Lite
- [ ] Targeted tests pass in both repos

---

## Notes

- This slice deliberately avoids changing `text_runtime.py` or `structured_runtime.py`; the fallback machinery already exists there and only needs better retry classification inputs.
- If later evidence shows provider-specific exhaustion states need richer handling than string patterns, follow up with a typed exhaustion classifier shared by retry and wrapping layers.
- Verification completed with:
  - `pytest -q tests/test_client.py -k 'gemini_monthly_spend_cap or gemini_daily_request_cap or quota_exceeded_not_retried or transient_rate_limit_is_retried or rate_limit_quota_with_retry_delay_is_retried'`
  - `PYTHONPATH=/home/brian/projects/data_contracts/src:/home/brian/projects/llm_client .venv/bin/pytest -q tests/unit/test_prebuild_graph_cli.py` in `Digimon_for_KG_application`
