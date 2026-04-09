# Plan #28: OpenRouter Gemini 3.1 Pro Registry And Tyler Validation

**Status:** Complete
**Type:** implementation
**Priority:** High
**Blocked By:** None
**Blocks:** exact Tyler Gemini model-version parity follow-through in `grounded-research`

---

## Gap

**Current:** Tyler requests Gemini 3.1 Pro for decomposition and the Stage 3
structured-decomposition analyst. `grounded-research` still uses
`openrouter/google/gemini-2.5-pro`, and `llm_client`'s packaged registry does
not yet expose OpenRouter's `google/gemini-3.1-pro-preview` model surface.

**Target:** `llm_client` exposes `openrouter/google/gemini-3.1-pro-preview` in
the packaged registry, documents the new shared surface, and validates it on
Tyler-like structured-output cases using the existing Gemini schema study
harness.

**Why:** the remaining exact Tyler model-version row is no longer a vague
availability question. The model exists on OpenRouter; the shared stack now
needs a sanctioned registry surface and evidence-backed structured-output
validation before `grounded-research` can switch to it honestly.

---

## References Reviewed

- `llm_client/data/default_model_registry.json`
- `llm_client/core/models.py`
- `scripts/study_gemini_schema_behavior.py`
- `tests/test_models.py`
- `tests/test_gemini_schema_behavior_study.py`
- `~/projects/grounded-research/docs/TYLER_SPEC_GAP_LEDGER.md`

---

## Files Affected

- `docs/plans/28_openrouter-gemini31-pro-registry-and-tyler-validation.md`
- `docs/notebooks/06_openrouter_gemini31_pro_validation.ipynb`
- `docs/plans/CLAUDE.md`
- `llm_client/data/default_model_registry.json`
- `tests/test_models.py`
- `tests/test_gemini_schema_behavior_study.py`

---

## Pre-Made Decisions

1. The target surface is `openrouter/google/gemini-3.1-pro-preview`, not the
   direct `gemini/*` lane.
2. This slice adds the model to the shared packaged registry first; it does not
   immediately change default task selection policy elsewhere.
3. Validation reuses the existing Gemini schema study harness instead of adding
   a second ad hoc script.
4. The slice passes only if the registry entry exists, focused tests pass, and
   the live study produces machine-readable evidence for the new model.

---

## Steps

1. Add `openrouter/google/gemini-3.1-pro-preview` to the packaged default
   model registry with conservative attributes that do not silently rewrite
   unrelated default task winners.
2. Add focused tests that prove:
   - the new model is present in the packaged registry,
   - the study harness continues to treat OpenRouter Gemini models as
     non-direct Gemini calls for thinking-budget kwargs.
3. Run the existing Tyler-like structured-output study harness live against
   `openrouter/google/gemini-3.1-pro-preview`.
4. Record the validation evidence in this plan and update the plan index.

---

## Required Tests

### New Tests

| Test File | Test Function | What It Verifies |
|-----------|---------------|------------------|
| `tests/test_models.py` | `test_packaged_registry_includes_openrouter_gemini31_pro_preview` | Shared packaged registry exposes the new model surface |
| `tests/test_gemini_schema_behavior_study.py` | `test_build_study_call_kwargs_does_not_add_thinking_for_openrouter_gemini` | OpenRouter Gemini models do not inherit direct-Gemini thinking kwargs |

### Existing Tests

| Test Pattern | Why |
|--------------|-----|
| `tests/test_models.py` | registry loading and task-selection behavior stay valid |
| `tests/test_gemini_schema_behavior_study.py` | study harness contract stays stable |
| `tests/test_structured_runtime.py` | structured-output runtime remains intact |

---

## Acceptance Criteria

- [x] packaged registry includes `openrouter/google/gemini-3.1-pro-preview`
- [x] new focused tests pass
- [x] existing registry/structured-runtime tests pass
- [x] live Tyler-like structured-output study for the new model completes and
      writes JSON evidence
- [x] plan index updated

---

## Completion Evidence

### Verified Commands

```bash
python -m py_compile scripts/study_gemini_schema_behavior.py tests/test_gemini_schema_behavior_study.py tests/test_models.py
python -m pytest tests/test_gemini_schema_behavior_study.py tests/test_models.py tests/test_structured_runtime.py -q
python -m llm_client models show openrouter/google/gemini-3.1-pro-preview
python scripts/study_gemini_schema_behavior.py --model openrouter/google/gemini-3.1-pro-preview --db-path tmp/gemini31_pro_validation/llm_observability.db --output-json tmp/gemini31_pro_validation/summary.json --max-budget 2.0
```

### Observed Results

- focused verification passed: `48 passed`
- the shared CLI now exposes:
  - `openrouter/google/gemini-3.1-pro-preview`
  - provider `openrouter`
  - context `1M`
  - structured output `yes`
- live Tyler-like structured-output study result:
  - `5/5` cases succeeded
  - all `5/5` completed via `native_schema`
  - all `5/5` used `response_format_type = json_schema`
  - no validation errors
  - tracked review artifact:
    - `docs/reviews/2026-04-09-openrouter-gemini31-pro-validation.json`

### Conclusion

`llm_client` now exposes the exact OpenRouter Gemini 3.1 Pro preview surface
needed for Tyler model-version follow-through, and the existing Tyler-like
structured-output study harness shows that this shared surface succeeds
cleanly on the representative structured cases used by the Gemini study lane.
