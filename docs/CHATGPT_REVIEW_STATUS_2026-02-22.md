# llm_client Status Dossier for External ChatGPT Review

Date: 2026-02-22
Project: `/home/brian/projects/llm_client`
Audience: external reviewer with no local context

Superseded by: `docs/CHATGPT_REVIEW_STATUS_2026-02-23.md`

## 1) Scope completed in this pass
This pass completed the requested hardening and documentation follow-through for provider/finalization reliability and Foundation contract safety.

Implemented scope:
1. Forced-final fallback + breaker behavior hardening.
2. Explicit finalization policy rejection path for tool-call-shaped outputs.
3. Provider-empty taxonomy normalization.
4. Retry-delay source attribution (`structured | parsed | none`) across retry loops.
5. Foundation schema strict mode (`FOUNDATION_SCHEMA_STRICT`) and strict-lane tests.
6. Documentation updates (status dossier + README + CI smoke strict lane).

## 2) What is now implemented

### 2.1 Forced-final controls (`llm_client/mcp_agent.py`)
1. Runtime controls:
   - `forced_final_max_attempts` (default `1`)
   - `forced_final_circuit_breaker_threshold` (default `2`)
   - `finalization_fallback_models`
2. Same-class breaker semantics are active:
   - breaker opens only on repeated same-class forced-final failures.
3. Added explicit policy rejection code for forced-final tool-call outputs:
   - `FINALIZATION_TOOL_CALL_DISALLOWED`
   - no tool execution occurs in forced-final even if model returns tool calls.
4. Added config coherence observability:
   - `forced_final_breaker_effective` metadata
   - warning when threshold is greater than attempts (inert breaker config).
5. Forced-final metadata now captures attempts, fallback usage, breaker status, and per-attempt records.

### 2.2 Foundation schema hardening (`llm_client/mcp_agent.py`)
1. Removed non-schema fields from `ToolFailed.failure` payloads.
2. Provider diagnostics previously in forbidden `failure.*` fields are now carried in schema-safe `inputs.params` keys.
3. Added strict mode:
   - env flag: `FOUNDATION_SCHEMA_STRICT=1`
   - invalid foundation event validation now raises immediately (instead of warning-only).

### 2.3 Retry-delay source instrumentation (`llm_client/client.py`)
1. Added structured hint extraction and normalization helpers.
2. `_compute_retry_delay` now returns `(delay, source)`.
3. Retry warnings/logs now include `retry_delay_source=structured|parsed|none` across sync/async/structured/stream paths.
4. Structured hints are preferred over parsed text when both are available.

### 2.4 Provider-empty canonicalization
1. Canonical failure code remains `PROVIDER_EMPTY_CANDIDATES`.
2. Legacy alias constants remain for import compatibility, but emissions/metrics use canonicalized behavior.

## 3) Tests added/updated

### 3.1 MCP/agent tests (`tests/test_mcp_agent.py`)
1. Taxonomy tests updated for:
   - `FINALIZATION_CIRCUIT_BREAKER_OPEN`
   - `FINALIZATION_TOOL_CALL_DISALLOWED`
2. Added forced-final tests for:
   - fallback recovery path
   - same-class breaker-open behavior
   - mixed-class failures not opening breaker
   - tool-call disallowed behavior in forced-final (no tool execution)
3. Added strict-mode validation test:
   - `FOUNDATION_SCHEMA_STRICT=1` causes invalid foundation event to raise.
4. Added assertion that known forced-final failure path has:
   - `foundation_event_validation_errors == 0`.

### 3.2 Client retry tests (`tests/test_client.py`)
1. Added assertions for `retry_delay_source=parsed` and `retry_delay_source=none`.
2. Added structured retry hint test (`retry_after` field) asserting `retry_delay_source=structured`.

## 4) CI/workflow update
Updated `.github/workflows/smoke-observability.yml` with a Foundation strict-lane step:
1. sets `FOUNDATION_SCHEMA_STRICT=1`
2. runs targeted MCP forced-final tests to prevent schema-drift regressions.

## 5) Validation results
Executed locally in repo root (`/home/brian/projects/llm_client`):
1. `pytest -q tests/test_mcp_agent.py tests/test_client.py`
   - Result: `291 passed`
2. `pytest -q`
   - Result: `776 passed, 1 skipped`

## 6) Current contract posture

### 6.1 Forced-final contract
1. Forced-final does not execute tools.
2. Tool-call-shaped forced-final outputs are explicitly rejected and recorded (`FINALIZATION_TOOL_CALL_DISALLOWED`).
3. Recovered forced-final runs remain classed as completed (`primary_failure_class="none"`).

### 6.2 Foundation contract
1. Failure-path event payloads are schema-safe in tested paths.
2. Strict mode is available for fail-fast enforcement.
3. Strict-lane test coverage now exists in CI workflow.

## 7) Remaining recommended follow-ups
1. Add aggregate metrics rollups to run summaries:
   - provider-caused incompletions
   - finalization fallback usage rate
   - breaker-open rates by model/provider
2. Consider full CI lane with `FOUNDATION_SCHEMA_STRICT=1` over broader MCP test subsets.
3. Lock policy docs for `reason_code` registry + `actor_id` issuance as separate governance artifacts.

## 8) Reviewer asks for ChatGPT
Please review:
1. whether finalization policy (`FINALIZATION_TOOL_CALL_DISALLOWED`) is the right boundary contract,
2. whether breaker class mapping should be further formalized as a stable code→class table,
3. whether strict mode should be on-by-default in benchmark lanes now.
