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

## 5) Policy decisions now locked
Previously open governance items are now accepted via ADRs:
1. `reason_code` registry governance is formalized in:
   - `docs/adr/0005-reason-code-registry-governance.md`
   - additive-only, non-reassigning semantics, versioned registry.
2. Foundation `actor_id` issuance policy is formalized in:
   - `docs/adr/0006-actor-id-issuance-policy.md`
   - canonical namespaces (`user:` / `agent:` / `service:`),
   - server/runtime trust boundary for authoritative issuance.

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

## 9) Additional follow-up (routing + identity coverage)
Completed another incremental architecture pass focused on routing seam consistency
and identity contracts:
1. Added shared `_resolve_call_plan(...)` in `llm_client/client.py` and routed
   text/structured/stream entrypoints through it for consistent normalization-event logging.
2. Expanded `tests/test_model_identity_contract.py` to cover:
   - `call_llm_structured` / `acall_llm_structured`
   - `stream_llm` / `astream_llm`
   with explicit routing policy assertions for `requested_model`, `resolved_model`,
   and `routing_trace`.
3. Switched YAML loads in `models.py` and `task_graph.py` to dynamic
   `import_module("yaml")` usage to keep typecheck deterministic without external
   stub dependency drift.
4. Converged structured retry behavior onto shared execution-kernel primitives
   in both sync and async paths:
   - responses branches now use `run_sync_with_retry(...)` / `run_async_with_retry(...)`
   - native-schema branches now use execution-kernel retry with explicit schema-fallback signaling
   - instructor branches now use execution-kernel retry wrappers

Validation snapshot after this pass:
1. `pytest -q tests/test_model_identity_contract.py`
   - Result: `11 passed, 1 warning`
2. `pytest -q`
   - Result: `793 passed, 1 skipped, 1 warning`
3. `mypy llm_client`
   - Result: `Success: no issues found in 36 source files`

## 10) Additional follow-up (structured fallback convergence)
Completed the next architecture ratchet for structured flows:
1. Replaced manual model-fallback loops in:
   - `call_llm_structured`
   - `acall_llm_structured`
   with shared execution-kernel fallback primitives:
   - `run_sync_with_fallback(...)`
   - `run_async_with_fallback(...)`
2. Kept existing retry semantics but removed bespoke outer fallback control flow.
3. Added shared structured result builder in `client.py`:
   - `_build_structured_call_result(...)`
   to reduce duplicated result/identity assembly logic across responses,
   native-schema, and instructor branches.
4. Expanded contract tests for structured fallback identity:
   - sync/async tests now assert normalized attempted-model chains and final
     `resolved_model` after fallback.

Validation snapshot after this pass:
1. `pytest -q tests/test_model_identity_contract.py tests/test_client.py -k "structured or fallback"`
   - Result: `42 passed, 183 deselected`
2. `pytest -q`
   - Result: `798 passed, 1 skipped, 1 warning`
3. `mypy llm_client`
   - Result: `Success: no issues found in 36 source files`

## 11) Additional follow-up (stream runtime extraction)
Completed the next architecture split for stream paths:
1. Extracted stream internals out of `llm_client/client.py` into:
   - `llm_client/stream_runtime.py`
   - `stream_llm_impl(...)`
   - `astream_llm_impl(...)`
2. Converted `stream_llm(...)` and `astream_llm(...)` in `client.py` into thin
   facade delegates with lazy imports to avoid circular initialization.
3. Preserved existing retry/fallback behavior and identity trace shaping for:
   - `requested_model`
   - `resolved_model`
   - `routing_trace`

Validation snapshot after this pass:
1. `pytest -q tests/test_client.py tests/test_model_identity_contract.py -k "stream or identity"`
   - Result: `29 passed, 196 deselected, 1 warning`
2. `pytest -q`
   - Result: `798 passed, 1 skipped, 1 warning`
3. `mypy llm_client`
   - Result: `Success: no issues found in 37 source files`

## 12) Additional follow-up (batch runtime extraction)
Completed the next architecture split for batch paths:
1. Extracted batch internals from `llm_client/client.py` into:
   - `llm_client/batch_runtime.py`
   - `acall_llm_batch_impl(...)`
   - `call_llm_batch_impl(...)`
   - `acall_llm_structured_batch_impl(...)`
   - `call_llm_structured_batch_impl(...)`
2. Converted `client.py` batch entrypoints into thin facade delegates with lazy
   imports to reduce client-module surface area.
3. Preserved callback/concurrency contracts:
   - per-item success/error callbacks,
   - semaphore-based max concurrency,
   - sync wrappers still use thread handoff when already in a running loop.

Validation snapshot after this pass:
1. `pytest -q tests/test_client.py -k "batch"`
   - Result: `11 passed, 201 deselected`
2. `mypy llm_client`
   - Result: `Success: no issues found in 38 source files`
3. `pytest -q`
   - Result: `798 passed, 1 skipped, 1 warning`

## 13) Additional follow-up (embedding runtime extraction)
Completed another facade split for embedding paths:
1. Extracted embedding internals from `llm_client/client.py` into:
   - `llm_client/embedding_runtime.py`
   - `embed_impl(...)`
   - `aembed_impl(...)`
2. Converted `client.py` embedding entrypoints into thin lazy-import delegates.
3. Preserved existing behavior for:
   - model deprecation checks,
   - sync/async rate limit acquisition,
   - io-log embedding telemetry fields and error logging.

Validation snapshot after this pass:
1. `pytest -q`
   - Result: `798 passed, 1 skipped, 1 warning`
2. `mypy llm_client`
   - Result: `Success: no issues found in 39 source files`

## 14) Additional follow-up (agent-loop kwarg/finalization dedupe)
Completed a targeted code-smell reduction inside `call_llm` + `acall_llm`:
1. Added shared helper for inner call kwargs propagation:
   - `_build_inner_named_call_kwargs(...)`
2. Added shared helper for MCP/tool loop kwarg partitioning:
   - `_split_agent_loop_kwargs(...)`
3. Added shared helper for loop result identity + logging finalization:
   - `_finalize_agent_loop_result(...)`
4. Replaced duplicated sync/async MCP and Python-tool loop blocks with the
   shared helpers, preserving existing behavior and contracts.

Validation snapshot after this pass:
1. `mypy llm_client`
   - Result: `Success: no issues found in 39 source files`
2. `pytest -q`
   - Result: `798 passed, 1 skipped, 1 warning`

## 15) Additional follow-up (strict CI lane broadening)
Expanded Foundation strict-mode CI coverage in:
- `.github/workflows/smoke-observability.yml`

Changes:
1. The `FOUNDATION_SCHEMA_STRICT=1` lane now runs:
   - `tests/test_foundation.py`
   - `tests/test_mcp_agent.py::TestAcallWithMcp::test_foundation_schema_strict_raises_on_invalid_event`
   - `tests/test_mcp_agent.py::TestAcallWithMcp::test_forced_final_llm_exception_preserves_tool_history`
   - `tests/test_mcp_agent.py::TestAcallWithMcp::test_forced_final_tool_calls_are_disallowed_without_execution`
2. This broadens strict-mode regression signal beyond the prior two-test subset.

Validation snapshot after this pass:
1. `FOUNDATION_SCHEMA_STRICT=1 pytest -q tests/test_foundation.py tests/test_mcp_agent.py::TestAcallWithMcp::test_foundation_schema_strict_raises_on_invalid_event tests/test_mcp_agent.py::TestAcallWithMcp::test_forced_final_llm_exception_preserves_tool_history tests/test_mcp_agent.py::TestAcallWithMcp::test_forced_final_tool_calls_are_disallowed_without_execution`
   - Result: `8 passed`

## 16) Additional follow-up (required-reading gate enabled)
Enabled concrete read-gate enforcement for source edits:
1. Added `scripts/meta/check_required_reading.py` and wired it to consume:
   - target file path,
   - session reads file (`/tmp/.claude_session_reads`),
   - `scripts/relationships.yaml` couplings/defaults.
2. Added `scripts/relationships.yaml` with baseline required-reading defaults
   and strict coupling docs for:
   - core call path modules (`client.py` + runtime split files),
   - MCP/Foundation governance modules,
   - routing/config policy modules.
3. Updated `scripts/CLAUDE.md` so operator docs include the new gate script
   and relationships config.

Validation snapshot after this pass:
1. `python -m py_compile scripts/meta/check_required_reading.py`
   - Result: pass
2. `python scripts/meta/check_required_reading.py llm_client/client.py --reads-file <empty-file>`
   - Result: blocked with explicit missing-doc list
3. `python scripts/meta/check_required_reading.py llm_client/client.py --reads-file <file containing required docs>`
   - Result: pass with required-doc summary

## 17) Additional follow-up (configurable read-gate modes)
Added explicit configuration and runtime overrides for required-reading
enforcement:
1. `meta-process.yaml` now includes:
   - `meta_process.quality.required_reading.enabled`
   - `mode` (`strict` | `warn` | `off`)
   - `uncoupled_mode` (`strict` | `warn` | `off`)
   - `config_file`
   - `show_success`
2. `scripts/meta/check_required_reading.py` now supports:
   - strict blocking mode (default),
   - warn-only mode (non-blocking),
   - off mode,
   - env overrides (`LLM_CLIENT_READ_GATE_*`).
3. `.claude/hooks/gate-edit.sh` now:
   - gates `llm_client/` edits (not template `src/` paths),
   - resolves checker from `scripts/meta/check_required_reading.py`,
   - supports custom reads file via `LLM_CLIENT_READS_FILE`.

Validation snapshot after this pass:
1. strict mode with missing reads: blocks (exit 1)
2. warn mode with missing reads: warns and allows (exit 0)
3. off mode with missing reads: allows (exit 0)

## 18) Additional follow-up (strict-start tuning profile)
Applied the next tuning pass for adoption ergonomics while keeping strict
contract protection where it matters:
1. `meta-process.yaml` now sets:
   - `required_reading.mode: strict`
   - `required_reading.uncoupled_mode: strict`
2. `scripts/relationships.yaml` was tightened to use focused strict couplings:
   - core call path + runtime/routing/execution modules -> ADR 0001/0002/0003/0004
   - MCP/Foundation/agent loop governance modules -> ADR 0005/0006
3. Removed status-dossier doc from global defaults so every edit only requires
   baseline `CLAUDE.md` unless the target file is in a strict coupling set.

Validation snapshot after this pass:
1. Coupled file with missing reads: strict block (exit 1)
2. Uncoupled file with missing reads: strict block (exit 1)
3. Coupled file with required docs read: pass (exit 0)

## 19) Additional follow-up (Makefile normalization + gate helpers)
Completed a maintenance pass on project tooling entrypoints:
1. Replaced duplicated `Makefile` content with one canonical target set.
2. Removed stale `src/` references and aligned checks to `llm_client/`.
3. Added read-gate helper targets:
   - `make read-gate-check FILE=...`
   - `make read-gate-check-warn FILE=...`

Validation snapshot after this pass:
1. `make help-meta` lists new read-gate helper targets.
2. `make read-gate-check-warn FILE=llm_client/errors.py` -> warning mode allows.
3. `make read-gate-check FILE=llm_client/client.py` -> strict mode blocks when reads are missing.

## 20) Additional follow-up (coverage expansion + CI smoke)
Expanded required-reading coverage and added CI smoke checks for mode behavior:
1. `scripts/relationships.yaml` now includes additional high-risk module groups:
   - `llm_client/io_log.py`, `llm_client/observability/*.py`
   - `llm_client/task_graph.py`, `llm_client/experiment_eval.py`
2. Added smoke/integration test files:
   - `tests/test_required_reading_gate.py`
   - `tests/test_gate_edit_hook_integration.py`
   - `tests/test_relationships_validation.py`
   - verifies strict/warn/off behavior and uncoupled strict default behavior,
     gate-edit hook block/warn semantics, and relationships config validity.
3. Expanded `.github/workflows/smoke-observability.yml` with:
   - `Required-reading gate mode smoke`
   - `Validate relationships config`
   - runs `pytest -q tests/test_required_reading_gate.py tests/test_gate_edit_hook_integration.py`
4. Stabilized remaining architecture couplings with ADR-backed sources:
   - observability coupling -> ADR 0007
   - task-graph/evaluation coupling -> ADR 0008

Validation snapshot after this pass:
1. `pytest -q tests/test_required_reading_gate.py`
2. `pytest -q`

## 21) Additional follow-up (agent-loop routing trace preservation)
Hardened call-path identity finalization so agent-loop metadata is not collapsed
at the client boundary:
1. `llm_client/client.py` `_finalize_agent_loop_result` now preserves
   loop-provided routing metadata when present:
   - `attempted_models` from loop result trace (instead of forcing `[primary_model]`)
   - `sticky_fallback` from loop trace (fallback to warning-derived detection)
2. Effective `api_base` and `selected_model` attribution now resolve against the
   finalized/selected model identity when available.
3. Added contract tests in `tests/test_model_identity_contract.py` for:
   - sync `call_llm` + `python_tools` loop path
   - async `acall_llm` + `python_tools` loop path
   - sync `call_llm` + `mcp_servers` loop path
   - async `acall_llm` + `mcp_servers` loop path
   Each test locks that loop-provided `attempted_models` and
   `sticky_fallback` survive finalization.

Validation snapshot after this pass:
1. `pytest -q tests/test_model_identity_contract.py tests/test_mcp_agent.py tests/test_routing.py`
2. `pytest -q`

## 22) Additional follow-up (long-thinking reliability hardening)
Hardened `gpt-5.2-pro` long-thinking behavior and traceability:
1. `llm_client/client.py`
   - wired explicit `reasoning_effort` into `_prepare_responses_kwargs(...)`
     (named argument now wins over forwarded kwargs).
   - added safe parsing for per-call polling controls:
     - `background_timeout`
     - `background_poll_interval`
   - added robust retrieval helpers:
     - `_retrieve_background_response(...)`
     - `_aretrieve_background_response(...)`
     using OpenAI SDK clients (`OpenAI` / `AsyncOpenAI`) for `response_id`
     retrieval.
   - polling now passes explicit `api_base` and request timeout context.
   - routing trace now includes optional `background_mode` for adoption tracking.
2. `llm_client/task_graph.py`
   - added optional task-level `reasoning_effort` passthrough so graph tasks can
     request long-thinking effort levels.
3. Tests expanded in `tests/test_client.py`:
   - `gpt-5.2-pro` Responses detection.
   - background mode + reasoning payload emission.
   - sync/async pending-background polling handoff assertions.
   - retrieval helper behavior (API key requirement + OpenAI client call shape).
4. Read-gate coupling updated:
   - `scripts/relationships.yaml` now requires ADR 0009 for `client.py`.
   - `tests/test_required_reading_gate.py` updated accordingly.

Validation snapshot after this pass:
1. `pytest -q tests/test_client.py -k "gpt52 or background or LongThinkingBackgroundRetrieval or ResponsesAPIDetection"`
   - Result: `10 passed`
2. `pytest -q`
   - Result: `824 passed, 1 skipped, 1 warning`

## 23) 24-hour execution queue (autonomous run plan)
Goal: close remaining long-thinking contract gaps, then continue architecture
ratchet with small, reversible steps.

### 0-4h: Contract closure (high priority)
1. Add explicit tests for `routing_trace["background_mode"]` in sync/async
   `call_llm` paths.
2. Add `task_graph` coverage for `reasoning_effort` passthrough and nullable
   `agent` handling safety in experiment records.
3. Keep default tests offline-safe and deterministic.

### 4-12h: Telemetry + observability consistency
1. Ensure long-thinking traces are preserved through result finalization paths
   (text + structured + agent-loop return points).
2. Add lightweight metadata counters in graph/reporting surfaces for how often
   long-thinking effort/background mode is used.

### 12-24h: Cleanup + migration prep
1. Remove any duplicate long-thinking decision logic that can drift between
   sync/async paths.
2. Tighten docs/examples for `gpt-5.2-pro` behavior and polling controls.
3. Run full suite and mypy; prepare one consolidated commit with validation
   evidence.

### Known uncertainties
1. LiteLLM runtime here exposes `responses()` as a function without `.retrieve`;
   retrieval therefore uses OpenAI SDK directly for now.
2. Background polling semantics for non-OpenAI providers remain out-of-scope and
   should be explicitly rejected/guarded in future ADR work.

## 24) 24-hour queue progress update (current session)
Executed now from the queue:
1. Added/expanded contract tests for background-mode routing trace visibility in
   sync/async `gpt-5.2-pro` calls (`tests/test_client.py`).
2. Hardened task-graph telemetry:
   - `TaskResult.reasoning_effort`
   - `TaskResult.background_mode`
   - experiment-record dimensions now include both fields.
3. Added task-graph tests for:
   - reasoning-effort passthrough,
   - background-mode capture from routing trace,
   - nullable `agent` fallback safety in experiment records.
4. Fixed potential nullable-agent runtime hazard in experiment logging by
   normalizing to `codex` label when `agent` is unset.
5. Removed mypy environment fragility from `llm_client/scoring.py` by switching
   YAML import to dynamic `import_module("yaml")` (same pattern as other modules).

Validation snapshot after queue progress:
1. `pytest -q tests/test_client.py -k "gpt52 or background or LongThinkingBackgroundRetrieval or ResponsesAPIDetection"`
   - Result: `10 passed`
2. `pytest -q tests/test_task_graph.py -k "reasoning_effort_passthrough or background_mode_capture or null_agent or experiment_record_defaults"`
   - Result: `2 passed` (plus deselected)
3. `pytest -q tests/test_scoring.py`
   - Result: `23 passed`
4. `mypy llm_client`
   - Result: `Success: no issues found in 39 source files`
5. `pytest -q`
   - Result: `827 passed, 1 skipped, 1 warning`

## 25) Additional follow-up (next-step execution: guard + telemetry consistency)
Implemented the next requested hardening pass:
1. Added explicit non-OpenAI background retrieval guard in
   `llm_client/client.py`:
   - `_validate_background_retrieval_api_base(...)`
   - `_BackgroundRetrievalConfigurationError`
2. Background poll loops now fail fast on deterministic configuration errors
   (unsupported endpoint / missing key) instead of retrying until timeout:
   - `_poll_background_response(...)`
   - `_apoll_background_response(...)`
3. Agent-loop finalization now explicitly carries forward `background_mode` when
   present in loop-provided routing traces.
4. Added/expanded tests in `tests/test_client.py`:
   - non-OpenAI `api_base` rejection (sync + async retrieval helpers),
   - poll-loop fail-fast behavior on config errors.
5. Expanded identity-contract coverage in
   `tests/test_model_identity_contract.py` to assert
   `routing_trace["background_mode"]` survives sync/async tool-loop and MCP-loop
   finalization.
6. Expanded task-graph coverage in `tests/test_task_graph.py` with an
   experiment-log shape lock for long-thinking dimensions:
   - `dimensions.reasoning_effort`
   - `dimensions.background_mode`

Validation snapshot after this pass:
1. `pytest -q tests/test_client.py -k "LongThinkingBackgroundRetrieval or gpt52 or background"`
   - Result: `9 passed`
2. `pytest -q tests/test_model_identity_contract.py -k "tool_loop_preserves_agent_routing_trace or mcp_loop_preserves_agent_routing_trace"`
   - Result: `4 passed`
3. `pytest -q tests/test_task_graph.py -k "background_mode_capture or long_thinking_dimensions or reasoning_effort_passthrough"`
   - Result: `2 passed` (plus deselected)
4. `mypy llm_client`
   - Result: `Success: no issues found in 39 source files`
5. `pytest -q`
   - Result: `837 passed, 1 skipped, 1 warning`

## 26) Additional follow-up (all requested next steps completed)
Executed all previously queued follow-ups:
1. Machine-readable configuration errors for background retrieval:
   - Introduced `LLMConfigurationError` in `llm_client/errors.py` with:
     - `error_code`
     - `details`
   - Bound long-thinking background config failures to stable codes:
     - `LLMC_ERR_BACKGROUND_ENDPOINT_UNSUPPORTED`
     - `LLMC_ERR_BACKGROUND_OPENAI_KEY_REQUIRED`
2. Added lightweight adoption helper in observability queries:
   - `get_background_mode_adoption(...)` in
     `llm_client/observability/query.py`
   - compatibility shim in `llm_client/io_log.py`
   - top-level export via `llm_client.__init__`
   - returns summarized counts/rates from task-graph experiment JSONL for:
     - `dimensions.reasoning_effort`
     - `dimensions.background_mode`
3. Added opt-in long-thinking integration smoke:
   - `tests/integration_long_thinking_smoke_test.py`
   - requires:
     - `LLM_CLIENT_INTEGRATION=1`
     - `LLM_CLIENT_LONG_THINKING_SMOKE=1`
     - `OPENAI_API_KEY`
   - checks `gpt-5.2-pro` with `reasoning_effort="high"` and confirms
     `routing_trace["background_mode"] == True`.
4. Added/updated test coverage:
   - `tests/test_client.py` for machine-readable error codes + fail-fast poll
     behavior.
   - `tests/test_errors.py` for `LLMConfigurationError` contract.
   - `tests/test_io_log.py` for adoption-helper counting/filtering behavior.
   - `tests/test_io_log_compat.py` shim-delegation check.
   - `tests/test_model_identity_contract.py` loop-finalization preservation of
     `background_mode`.
5. Documentation updates:
   - `README.md` long-thinking error-code + adoption-helper usage notes.
   - `docs/adr/0009-long-thinking-background-polling.md` updated with stable
     error-code contract.
   - `tests/CLAUDE.md` integration smoke run command.
