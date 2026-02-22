# llm_client Status Dossier for External ChatGPT Review

Date: 2026-02-22
Project: `/home/brian/projects/llm_client`
Audience: external reviewer with no local context

## 1) Scope of this implementation pass
This pass focused on the Week-1 hardening items for provider-driven failures and forced-final robustness in the MCP/tool agent loop.

Primary goals implemented:
1. Finalization-only fallback lane behavior (without enabling tools in fallback).
2. Forced-final circuit breaker semantics (stop repeated same-class failures).
3. Unified provider-empty failure taxonomy.
4. Retry-delay source attribution (`structured | parsed | none`) across retry loops.
5. Tests for the above and regression validation.

## 2) What is now implemented

### 2.1 Forced-final fallback and breaker controls
Implemented in `llm_client/mcp_agent.py`:
1. Added runtime knobs:
   - `forced_final_max_attempts` (default `1`)
   - `forced_final_circuit_breaker_threshold` (default `2`)
   - `finalization_fallback_models` (finalization-only chain)
2. Forced-final attempt scheduling now supports repeated attempts even without fallback models.
3. Circuit breaker now opens on repeated **same-class** forced-final failures (instead of any mixed failures).
4. Breaker emits terminal failure code: `FINALIZATION_CIRCUIT_BREAKER_OPEN`.
5. Forced-final metadata is recorded in run metadata:
   - attempts, thresholds, breaker-open flag
   - fallback used/succeeded
   - per-attempt finalization records
   - finalization event sequence

### 2.2 Provider-empty taxonomy unification
1. Canonical code is now `PROVIDER_EMPTY_CANDIDATES`.
2. Legacy aliases remain for compatibility:
   - `EVENT_CODE_PROVIDER_EMPTY_FIRST_TURN`
   - `EVENT_CODE_PROVIDER_EMPTY_RESPONSE`
   both alias to canonical code.
3. Failure classification now maps `FINALIZATION_*` codes into provider failure class.

### 2.3 Retry-delay source instrumentation
Implemented in `llm_client/client.py`:
1. Added structured retry-delay extraction helper.
2. Retry delay computation now returns `(delay, source)` where source is:
   - `structured` (typed fields/headers)
   - `parsed` (message parsing)
   - `none` (backoff only)
3. Applied consistently to sync/async/structured/stream retry loops.
4. Retry warnings now include source marker:
   - `retry_delay_source=structured|parsed|none`

## 3) Tests added/updated

### 3.1 MCP/agent tests (`tests/test_mcp_agent.py`)
1. Failure taxonomy expectation updated for canonical provider code + new terminal codes.
2. Added forced-final fallback recovery test.
3. Added forced-final same-class breaker-open test.
4. Existing retrieval-stagnation and fallback diagnostics tests remain passing.

### 3.2 Client retry tests (`tests/test_client.py`)
1. Added assertion for parsed retry-delay source in quota-with-retryDelay scenario.
2. Added test for structured `retry_after` hints.
3. Added assertion for `retry_delay_source=none` on normal transient retry.

## 4) Validation results
Executed locally in repo root:
1. `pytest -q tests/test_mcp_agent.py tests/test_client.py`
   - Result: `284 passed`
2. `pytest -q`
   - Result: `767 passed, 1 skipped`

## 5) Behavioral contract summary (post-change)

### 5.1 Forced-final behavior
1. Forced-final runs with no tools passed.
2. Primary forced-final may fail and optionally recover via finalization-only fallback models.
3. Repeated same-class forced-final failures open a circuit breaker and terminate early.
4. Recovered finalization does not count as a terminal run failure class.

### 5.2 Retry behavior
1. Provider retry hints still influence wait duration.
2. Each retry now records whether delay came from structured hint, parsed text, or none.

## 6) Known caveats / follow-up work
1. Foundation event validation still logs warnings in some failure paths where extra diagnostic fields are included in `ToolFailed.failure` payload.
   - This does **not** fail test suite currently.
   - Suggested follow-up: move extra diagnostics to schema-safe fields or metadata envelope.
2. Existing dirty file in worktree:
   - `llm_client/__main__.py`
   - It was pre-existing and not modified by this pass.

## 7) Recommended external review asks (for ChatGPT)
Please review and critique:
1. Whether forced-final breaker semantics are scoped correctly for provider instability lanes.
2. Whether finalization-only fallback policy is tight enough to avoid silent masking.
3. Whether retry-delay source instrumentation is sufficient for benchmark lane attribution.
4. Whether Foundation event payload strategy should be tightened now (schema-strict) or deferred.
5. Any missing acceptance criteria before cutting a tagged baseline.

## 8) Suggested next implementation slice
1. Add explicit negative tests to ensure non-schema fields never leak into strict Foundation event schemas.
2. Add metrics rollups for:
   - provider-caused incompletions
   - forced-final fallback usage rate
   - breaker-open rate by provider/model
3. Add a strict mode flag to fail fast on any Foundation event validation warning in CI.
