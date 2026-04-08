# Plan #26: Gemini Strict-Schema Behavior Study

**Status:** Complete
**Type:** implementation
**Priority:** High
**Blocked By:** None
**Blocks:** evidence-backed closure of the shared Gemini strict-schema quality row

---

## Gap

**Current:** `llm_client` records structured-call execution metadata such as
`execution_path`, `schema_hash`, `response_format_type`, and
`validation_errors`, but there is no sanctioned harness that uses those fields
to study how Gemini behaves on Tyler-like schemas.

**Target:** a replayable live-study harness runs a small matrix of Tyler-like
schemas against configured Gemini models, captures whether each call succeeds
via native schema or instructor fallback, and writes a compact JSON summary.

**Why:** `grounded-research` and other consumers cannot honestly claim Gemini
strict-schema quality from compatibility code alone. They need run-backed
evidence for native-schema success, fallback frequency, and validation failure
rates on representative schemas.

---

## References Reviewed

- `llm_client/execution/structured_runtime.py:471-693` - sync structured path,
  native schema vs instructor fallback, observability logging
- `llm_client/execution/structured_runtime.py:1072-1294` - async structured
  path used by most real consumers
- `llm_client/io_log.py:1152-1186` - persisted fields for execution path,
  schema hash, response-format type, and validation errors
- `docs/REQUIREMENTS.md` - `json_schema` requirement
- `~/projects/grounded-research/docs/TYLER_EXECUTION_STATUS.md` - remaining
  shared Gemini strict-schema row

---

## Files Affected

- `scripts/study_gemini_schema_behavior.py` (create)
- `tests/test_gemini_schema_behavior_study.py` (create)
- `docs/plans/26_gemini-strict-schema-behavior-study.md` (create)
- `docs/notebooks/05_gemini_schema_behavior_study.ipynb` (create)
- `docs/plans/CLAUDE.md` (modify)

---

## Plan

### Steps

1. Define a small fixed set of Tyler-like schema cases inside the study script:
   - flat required fields
   - nested object
   - nullable/optional field
   - list of objects
   - enum-heavy decision object
2. Add a CLI that runs the case matrix against one or more Gemini models using
   `acall_llm_structured`.
3. Query the run-local observability DB rows created by those calls and attach:
   - `execution_path`
   - `response_format_type`
   - `schema_hash`
   - `validation_errors`
   - `error_type`
4. Write a compact JSON summary per run and one aggregate summary.
5. Add tests for:
   - case registry shape
   - observability-row summarization
   - CLI summary writing on synthetic DB/input data

---

## Required Tests

### New Tests (TDD)

| Test File | Test Function | What It Verifies |
|-----------|---------------|------------------|
| `tests/test_gemini_schema_behavior_study.py` | `test_build_case_registry_returns_named_cases` | Study cases are stable and explicit |
| `tests/test_gemini_schema_behavior_study.py` | `test_summarize_observability_rows_classifies_execution_paths` | Native-schema vs instructor vs error classification is computed from DB rows |
| `tests/test_gemini_schema_behavior_study.py` | `test_write_summary_json_emits_expected_shape` | Script output is machine-readable and stable |

### Existing Tests (Must Pass)

| Test Pattern | Why |
|--------------|-----|
| `tests/test_structured_runtime.py` | Structured runtime behavior remains intact |
| `tests/test_client.py -k structured` | Public structured-call contract unchanged |

---

## Acceptance Criteria

- [x] Study script can run a Gemini model/case matrix and write JSON output
- [x] Summary includes execution-path evidence from observability DB
- [x] New tests pass
- [x] Existing structured-runtime tests pass
- [x] Docs/index updated

---

## Notes

- This plan intentionally does **not** change runtime behavior yet.
- The first slice is evidence collection, not a fallback-policy refactor.

## Completion Evidence (2026-04-08)

### Verified Commands

```bash
python -m py_compile scripts/study_gemini_schema_behavior.py tests/test_gemini_schema_behavior_study.py
python -m pytest tests/test_gemini_schema_behavior_study.py tests/test_structured_runtime.py -q
python scripts/study_gemini_schema_behavior.py --model openrouter/google/gemini-2.5-pro --model gemini/gemini-2.5-pro --db-path tmp/gemini_schema_behavior_study/llm_observability.db --output-json tmp/gemini_schema_behavior_study/summary.json --max-budget 1.0
python scripts/study_gemini_schema_behavior.py --model gemini/gemini-2.5-pro --db-path tmp/gemini_schema_behavior_study_direct/llm_observability.db --output-json tmp/gemini_schema_behavior_study_direct/summary.json --max-budget 1.0 --direct-gemini-thinking-budget 256
```

### Observed Results

- `openrouter/google/gemini-2.5-pro`: `5/5` Tyler-like cases succeeded via `native_schema`
- direct `gemini/gemini-2.5-pro` with the default shared thinking config: `5/5` failed before schema evaluation with provider-side `BadRequestError`
- direct `gemini/gemini-2.5-pro` with `--direct-gemini-thinking-budget 256`: `5/5` Tyler-like cases succeeded via `native_schema`

### Conclusion

The strict-schema gap is not a proven Gemini schema-quality failure. The live
evidence shows a provider/runtime transport precondition on the direct
`gemini/*` lane: the default `budget_tokens=0` thinking configuration is
rejected there, while both OpenRouter Gemini and direct Gemini with a positive
thinking budget succeed natively on the same schema set.
