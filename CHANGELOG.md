# Changelog

All notable changes to `llm-client` are documented in this file.

## Unreleased

### Added

- Shared execution-kernel primitives in `llm_client.execution_kernel`:
  - `run_sync_with_retry` / `run_async_with_retry`
  - `run_sync_with_fallback` / `run_async_with_fallback`
- Observability boundary modules in `llm_client.observability`:
  - `events.py`, `experiments.py`, `query.py`
- Modular CLI command package in `llm_client.cli`:
  - `cost.py`, `traces.py`, `scores.py`, `experiments.py`, `backfill.py`
- MCP agent runtime controls for finalization reliability:
  - `finalization_fallback_models`
  - `forced_final_max_attempts`
  - `forced_final_circuit_breaker_threshold`
- Retrieval stagnation fuse for evidence loops:
  - `retrieval_stagnation_turns`
  - terminal event code `RETRIEVAL_STAGNATION`
- Extended MCP agent metadata and diagnostics:
  - finalization fallback usage/success/event traces
  - forced-final attempt and circuit-breaker telemetry
  - retrieval stagnation trigger/streak/turn metadata
- OpenRouter key-pool rotation for key-limit 403 responses:
  - supports `OPENROUTER_API_KEYS` and numbered `OPENROUTER_API_KEY_<n>` vars
  - rotates `OPENROUTER_API_KEY` to the next key and retries immediately
  - emits `OPENROUTER_KEY_ROTATED` / `OPENROUTER_KEY_ROTATION_UNAVAILABLE` warnings
- Digimon benchmark lane controls wired through runner:
  - `--lane-policy {pure,reliability}`
  - `--finalization-fallback-models`
  - `--forced-final-max-attempts`
  - `--forced-final-circuit-breaker-threshold`
  - `--retrieval-stagnation-turns`

### Changed

- `call_llm` / `acall_llm` now use shared retry+fallback kernel paths instead
  of duplicated in-function retry/fallback loops.
- `_agent_loop` now stages tool/contract initialization through typed
  `AgentLoopToolState` construction (`_initialize_agent_tool_state`).
- `python -m llm_client` is now a thin command router; command logic moved out
  of `llm_client.__main__` into per-command modules.
- Top-level `llm_client` exports observability APIs via
  `llm_client.observability` boundaries.
- Provider-empty taxonomy is canonicalized as `PROVIDER_EMPTY_CANDIDATES`
  (legacy aliases retained for compatibility).
- Forced-final path now attempts bounded model chains and keeps run-level
  failure attribution clean when fallback finalization succeeds.
- Benchmark summaries now report completion-conditioned accuracy plus provider,
  fallback, and retrieval-stagnation rates.

## 0.7.0 - 2026-02-23

### Breaking Changes

- Removed model-semantics compatibility modes.
  - `ClientConfig.result_model_semantics` removed.
  - `LLM_CLIENT_RESULT_MODEL_SEMANTICS` removed.
- `LLMCallResult.model` now has one fixed meaning:
  - always the terminal executed model (same identity as `resolved_model`).
- Removed semantics-adoption telemetry controls:
  - `LLM_CLIENT_SEMANTICS_TELEMETRY` removed.
- Removed semantics reporting CLI commands:
  - `python -m llm_client semantics`
  - `python -m llm_client semantics-snapshot`

### Changed

- Identity contract is now single-path and explicit:
  - `requested_model` = caller input
  - `resolved_model` / `execution_model` = executed model
  - `routing_trace` = why routing/fallback changed the model
- Observability smoke workflow now validates logging-disabled mode only.

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
- Lightweight semantics-adoption telemetry for migration planning:
  - foundation events (`ConfigChanged`) with caller/source/mode metadata
  - env switch: `LLM_CLIENT_SEMANTICS_TELEMETRY=off`.
- Semantics snapshot CLI for daily migration tracking:
  - `python -m llm_client semantics-snapshot`
  - appends JSONL snapshot records with filters + aggregate adoption summary.
- GitHub Actions smoke workflow for observability toggles:
  - `.github/workflows/smoke-observability.yml`
  - validates telemetry/logging disabled mode.

### Changed

- All major call paths now support explicit typed config (`config=ClientConfig(...)`),
  including tool, stream, and batch wrappers.
- Routing behavior is resolved through typed, testable plan objects instead of
  scattered ad-hoc normalization in call sites.
- Integration tests are gated by marker + env:
  - default `pytest` excludes `integration`
  - enable with `LLM_CLIENT_INTEGRATION=1`.
- Added CLI adoption report:
  - `python -m llm_client semantics`
  - summarizes semantics mode/source usage from `foundation_events`.
- Added telemetry-off contract test:
  - `LLM_CLIENT_SEMANTICS_TELEMETRY=off` suppresses foundation event emission.

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
