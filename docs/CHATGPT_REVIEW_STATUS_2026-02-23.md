# llm_client Status Dossier for External ChatGPT Review

Date: 2026-02-23
Project: `/home/brian/projects/llm_client`
Audience: external reviewer with no local context

## 1) Executive status
This pass completed the contract-hardening slice for forced-final reliability and Foundation schema safety.

Current outcome:
1. Forced-final behavior is now policy-constrained (no tools in forced-final, explicit disallowed code path).
2. Provider-empty handling is canonicalized around `PROVIDER_EMPTY_CANDIDATES`.
3. Retry-delay attribution is instrumented (`structured | parsed | none`) across retry loops.
4. Foundation strict mode exists (`FOUNDATION_SCHEMA_STRICT=1`) and can fail fast.
5. Negative Foundation schema tests now exist for custom event types and forbidden payload fields.
6. Full suite is currently green.

## 2) What was implemented

### 2.1 `llm_client/mcp_agent.py`
1. Added strict mode env gate support (`FOUNDATION_SCHEMA_STRICT`) so invalid Foundation event payloads raise in strict mode.
2. Added explicit finalization policy code: `FINALIZATION_TOOL_CALL_DISALLOWED`.
3. Added forced-final policy rejection path:
   - if forced-final output contains tool calls, no tools execute,
   - emits classified provider/policy failure telemetry,
   - records forced-final attempt failure and continues breaker logic.
4. Added failure-class mapping helper (`_failure_class_for_event_code`) used by breaker/classification paths.
5. Added breaker coherence metadata:
   - `forced_final_breaker_effective`
   - warning when breaker threshold is inert vs attempt budget.
6. Made `ToolFailed.failure` schema-safe by moving non-schema diagnostics into `inputs.params`.
7. Added run-level rollups in metadata for:
   - provider-caused incompletions,
   - fallback usage,
   - finalization attempts/failures/success by model,
   - breaker-open rates,
   - provider-empty counts by model,
   - failure class/code counters.

### 2.2 `llm_client/client.py`
1. `_compute_retry_delay` now emits `(delay, retry_delay_source)`.
2. Retry hint extraction prefers structured hints over parsed message text.
3. Retry warnings/logs now include `retry_delay_source` in sync/async/stream/structured paths.

### 2.3 Tests
1. `tests/test_mcp_agent.py`:
   - added/updated tests for breaker same-class semantics,
   - mixed-class no-breaker behavior,
   - forced-final tool-call rejection path,
   - strict mode raising path,
   - metadata rollups for fallback/breaker/provider counters.
2. `tests/test_client.py`:
   - added assertions for `retry_delay_source=parsed|structured|none`.
3. `tests/test_foundation.py`:
   - added negative schema test rejecting custom `event_type="BeliefStatusChanged"`,
   - added negative schema test rejecting extra forbidden `ToolFailed.failure` fields.

### 2.4 CI and docs
1. `.github/workflows/smoke-observability.yml` includes strict-lane step with `FOUNDATION_SCHEMA_STRICT=1` on targeted MCP tests.
2. `README.md` includes strict mode usage notes.

## 3) Validation evidence
Commands run from `/home/brian/projects/llm_client`:
1. `pytest -q tests/test_foundation.py tests/test_mcp_agent.py::TestAcallWithMcp::test_foundation_schema_strict_raises_on_invalid_event`
   - Result: `6 passed`
2. `pytest -q`
   - Result: `771 passed, 1 skipped, 1 warning`

## 4) Contract posture after this pass

### 4.1 Forced-final contract
1. Forced-final attempts do not execute tools.
2. Tool-call-shaped finalization output is rejected with `FINALIZATION_TOOL_CALL_DISALLOWED`.
3. Breaker classification is explicit and test-covered.

### 4.2 Foundation contract
1. Boundary events validate against Foundation schema paths used in current MCP flows.
2. Strict mode can enforce fail-fast behavior.
3. Negative tests now prove strict rejection for:
   - non-enum custom event types,
   - extra forbidden payload fields.

### 4.3 Taxonomy and observability
1. Provider-empty canonical code is standardized (`PROVIDER_EMPTY_CANDIDATES`).
2. Retry-delay source attribution is consistently emitted and test-covered.
3. Run metadata now separates reliability outcomes from reasoning outcomes via rollup counters.

## 5) Remaining policy decisions (non-code)
These are still recommended to lock formally in docs:
1. `reason_code` registry governance:
   - additive-only,
   - non-reassigning semantics,
   - version-stamped source of truth.
2. `actor_id` issuance model:
   - server-issued namespace policy (`user:` / `agent:` / `service:`),
   - server as trust boundary,
   - required on decision and transition records.

## 6) Reviewer prompts for ChatGPT
Please critique:
1. Whether strict mode should be promoted from targeted smoke to a broader CI lane now.
2. Whether forced-final breaker defaults should become strict-validated (`threshold <= max_attempts`) vs currently warning + metadata.
3. Whether the current failure-class mapping table should be treated as versioned public contract.

## 7) Baseline readiness
Engineering status is baseline-ready for this slice:
1. full suite green,
2. strict mode available and test-covered,
3. negative schema safeguards added,
4. forced-final reliability controls implemented and observable.

Recommended release caveat:
- keep strict Foundation validation in CI (at least targeted lane) until broader strict lane adoption is complete.

## 8) Architecture follow-up pass (same day)
Completed additional debt-reduction refactor steps without changing public contracts:
1. Extracted shared retry/fallback helpers to `llm_client/execution_kernel.py` and wired `call_llm`/`acall_llm` plus stream paths to use them.
2. Split `_agent_loop` initialization into a typed stage:
   - `AgentLoopToolState`
   - `_initialize_agent_tool_state(...)`
3. Introduced observability boundary modules:
   - `llm_client/observability/events.py`
   - `llm_client/observability/experiments.py`
   - `llm_client/observability/query.py`
4. Split CLI command monolith into per-command modules under `llm_client/cli/` and reduced `llm_client/__main__.py` to parser/dispatch only.

Validation snapshot after this pass:
1. `pytest -q tests/test_client.py tests/test_mcp_agent.py tests/test_model_identity_contract.py tests/test_experiment_log.py tests/test_io_log.py`
   - Result: `381 passed`
2. `pytest -q`
   - Result: `771 passed, 1 skipped, 1 warning`

Additional follow-up in same branch:
1. Moved concrete experiment/query logic out of `io_log.py` into:
   - `llm_client/observability/experiments.py`
   - `llm_client/observability/query.py`
2. Converted `io_log` query/experiment APIs to compatibility delegates.
3. Added seam-lock tests:
   - `tests/test_execution_kernel.py`
   - `tests/test_cli_smoke.py`
   - `tests/test_io_log_compat.py`
4. Full-suite validation after this follow-up:
   - `783 passed, 1 skipped, 1 warning`
