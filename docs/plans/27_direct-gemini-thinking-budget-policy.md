# Plan #27: Direct Gemini Thinking Budget Policy

**Status:** Complete
**Type:** implementation
**Priority:** High
**Blocked By:** 26
**Blocks:** durable direct-Gemini structured/text reliability on thinking-capable models

---

## Gap

**Current:** `llm_client` injects `thinking={"type":"enabled","budget_tokens":0}`
for non-OpenRouter Gemini thinking models when LiteLLM reports support for the
parameter. Plan 26's live study showed that direct
`gemini/gemini-2.5-pro` rejects that default on the structured path before
schema evaluation even begins.

**Target:** direct `gemini/*` thinking defaults become an explicit shared policy:

1. configurable in `ClientConfig` / env,
2. applied only to direct Gemini models,
3. overridable by an explicit caller-supplied `thinking` payload,
4. covered by transport-level tests for text and structured calls.

**Why:** the current zero-budget default overfits one provider path and is now
proven to break another supported provider path. This is a shared runtime
policy problem, not a repo-local workaround.

---

## References Reviewed

- `docs/plans/26_gemini-strict-schema-behavior-study.md` - live evidence for direct Gemini rejection vs positive-budget success
- `llm_client/execution/completion_runtime.py` - shared thinking default injection
- `llm_client/core/config.py` - runtime policy surface
- `llm_client/core/model_detection.py` - Gemini/thinking model classification
- `tests/test_client.py` - current direct-Gemini thinking-default assertions
- `tests/test_provider_kwargs.py` - focused transport-kwargs seam

---

## Files Affected

- `docs/plans/27_direct-gemini-thinking-budget-policy.md` (create)
- `docs/notebooks/06_direct_gemini_thinking_budget_policy.ipynb` (create)
- `docs/plans/CLAUDE.md` (modify)
- `llm_client/core/config.py` (modify)
- `llm_client/core/model_detection.py` (modify)
- `llm_client/execution/completion_runtime.py` (modify)
- `tests/test_client.py` (modify)
- `tests/test_provider_kwargs.py` (modify)

---

## Plan

### Steps

1. Add a typed direct-Gemini thinking-budget default to `ClientConfig` and env parsing.
2. Replace the hardcoded `budget_tokens=0` injection with provider-aware policy resolution.
3. Preserve caller authority:
   - explicit `thinking` payload wins
   - OpenRouter Gemini behavior stays unchanged
4. Update transport-focused tests to assert the new default and config override path.
5. Re-run the live Gemini study without the ad hoc study-only override once the shared default lands.

---

## Required Tests

### New Tests (TDD)

| Test File | Test Function | What It Verifies |
|-----------|---------------|------------------|
| `tests/test_provider_kwargs.py` | `test_prepare_call_kwargs_uses_configured_direct_gemini_thinking_budget` | direct Gemini defaults come from shared config, not a hardcoded zero |
| `tests/test_provider_kwargs.py` | `test_prepare_call_kwargs_allows_disabling_direct_gemini_auto_thinking` | config can disable automatic injection when a consumer needs manual control |

### Existing Tests (Must Pass)

| Test Pattern | Why |
|--------------|-----|
| `tests/test_client.py -k thinking` | public text/structured thinking behavior remains consistent |
| `tests/test_structured_runtime.py` | structured runtime transport contract remains intact |
| `tests/test_provider_kwargs.py` | provider kwargs stay clean and JSON-serializable |

---

## Acceptance Criteria

- [x] hardcoded direct-Gemini `budget_tokens=0` injection is gone
- [x] direct-Gemini thinking default is configurable from shared runtime config
- [x] explicit caller `thinking` payload still wins
- [x] OpenRouter Gemini behavior is unchanged
- [x] tests pass
- [x] live study rerun confirms direct Gemini succeeds without the study-only override

---

## Notes

- This plan is intentionally narrow. It does not reopen the broader provider-governance architecture in Plan 25.
- If live rerun shows another direct-Gemini requirement beyond thinking budget, document it explicitly instead of widening this plan silently.

## Completion Evidence (2026-04-08)

### Verified Commands

```bash
python -m py_compile llm_client/core/config.py llm_client/core/model_detection.py llm_client/execution/completion_runtime.py scripts/study_gemini_schema_behavior.py tests/test_gemini_schema_behavior_study.py tests/test_provider_kwargs.py
python -m pytest tests/test_gemini_schema_behavior_study.py tests/test_provider_kwargs.py tests/test_client.py -k thinking tests/test_structured_runtime.py -q
PYTHONPATH=/home/brian/projects/_worktrees/llm_client-gemini-schema-study python scripts/study_gemini_schema_behavior.py --model gemini/gemini-2.5-pro --db-path tmp/gemini_schema_behavior_study_postfix_worktree/llm_observability.db --output-json tmp/gemini_schema_behavior_study_postfix_worktree/summary.json --max-budget 1.0
```

### Observed Results

- direct `gemini/gemini-2.5-pro` now succeeds on `5/5` Tyler-like schema cases via `native_schema` without the study-only override
- the shared default is now configurable via `LLM_CLIENT_DIRECT_GEMINI_THINKING_BUDGET`
- explicit caller `thinking` payloads still override the shared default

### Important Note

The study script now prepends the repository root to `sys.path` before
importing `llm_client`. Without that, `python scripts/...` can evaluate an
installed package instead of the current checkout and produce false evidence.
