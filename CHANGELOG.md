# Changelog

All notable changes to `llm-client` are documented in this file.

## 0.6.1 - 2026-02-22

### Added

- Typed runtime configuration via `ClientConfig`:
  - `routing_policy` (`openrouter` or `direct`)
  - `result_model_semantics` (`legacy`, `requested`, `resolved`)
- Pure routing resolver in `llm_client.routing`:
  - `CallRequest`
  - `ResolvedCallPlan`
  - `resolve_call(request, config)`
- Result identity fields:
  - `requested_model`
  - `resolved_model`
  - `execution_model` (alias of terminal execution identity)
  - `routing_trace`
- Machine-readable warning metadata:
  - `LLMCallResult.warning_records`
  - stable warning codes (`LLMC_WARN_*`)
- Deterministic tool-call compliance gate (`llm_client.compliance_gate`).

### Changed

- All major call paths now support explicit typed config (`config=ClientConfig(...)`),
  including tool, stream, and batch wrappers.
- Routing behavior is resolved through typed, testable plan objects instead of
  scattered ad-hoc normalization in call sites.
- Integration tests are gated by marker + env:
  - default `pytest` excludes `integration`
  - enable with `LLM_CLIENT_INTEGRATION=1`.

### Compatibility Notes

- `result.model` remains compatibility-first by default (`legacy` semantics).
- Consumers can opt into deterministic identity semantics now:
  - `ClientConfig(result_model_semantics="requested")`
  - `ClientConfig(result_model_semantics="resolved")`
  - or `LLM_CLIENT_RESULT_MODEL_SEMANTICS=...`.
- Canonical identity for new consumers:
  - caller identity: `requested_model`
  - execution identity: `resolved_model` / `execution_model`.

### Validation

- Full test suite passed at release cut:
  - `757 passed, 1 skipped`.
