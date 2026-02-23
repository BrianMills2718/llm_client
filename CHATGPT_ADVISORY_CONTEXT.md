# LLM Client Advisory Context (Paste Into ChatGPT)

Date: 2026-02-23  
Project path: `/home/brian/projects/llm_client`  
Audience: external advisor (ChatGPT) with zero prior context

## Update (2026-02-23 hard cutover)

The model-semantics migration plan in this document is now superseded.
Current architecture uses one fixed identity contract:
1. `result.model` = terminal executed model.
2. `requested_model` = caller input.
3. `resolved_model` / `execution_model` = terminal executed model.
4. `routing_trace` explains normalization/fallback decisions.

Removed surfaces:
1. `ClientConfig.result_model_semantics`
2. `LLM_CLIENT_RESULT_MODEL_SEMANTICS`
3. `LLM_CLIENT_SEMANTICS_TELEMETRY`
4. CLI commands `semantics` and `semantics-snapshot`.

## Status Update (2026-02-23, post-implementation)

Current architecture highlights:
1. Pure router extracted: `llm_client/routing.py` with:
   - `CallRequest`
   - `ResolvedCallPlan`
   - `resolve_call(request, config)`
   - `resolve_api_base_for_model(model, requested_api_base, config)`
2. Typed config introduced: `llm_client/config.py` with:
   - `ClientConfig.routing_policy` (`openrouter` or `direct`)
   - `ClientConfig.from_env()` env parsing for routing only.
3. Core call paths now accept explicit `config: ClientConfig | None` and route
   via typed resolver, including tool wrappers, batch wrappers, and stream APIs.
4. Stable warning metadata added on results via
   `LLMCallResult.warning_records` with `LLMC_WARN_*` codes.
5. Fixed identity semantics shipped in `0.7.0`:
   - `result.model` always means terminal executed model
   - `requested_model` always means caller input
   - `resolved_model` / `execution_model` always mean terminal executed model
   - `routing_trace` explains normalization/fallback
6. Foundation event hardening completed:
   - strict mode env gate: `FOUNDATION_SCHEMA_STRICT=1`
   - invalid foundation event payloads now raise in strict mode
   - failure-path payloads moved to schema-safe fields (no extra forbidden keys)
7. Forced-final policy hardening completed:
   - explicit `FINALIZATION_TOOL_CALL_DISALLOWED` classification
   - no tool execution in forced-final even if model returns tool-call-shaped output
8. Current test status: `pytest -q` => `771 passed, 1 skipped`.
9. Foundation negative schema tests added:
   - custom `event_type` rejection
   - forbidden extra `ToolFailed.failure` field rejection

## 1. Why We Need Advice

We are evaluating whether to keep refactoring this codebase or rewrite it from scratch.  
The project is functional and feature-rich, but we suspect significant tech debt and design drift.

We want advice on:
1. Whether a full rewrite is justified.
2. If not, the best staged refactor plan.
3. How to stabilize behavior contracts without freezing progress.

## 2. What This Project Is

This is a Python package (`llm-client`, version `0.7.0`) that wraps `litellm` and agent SDKs behind a unified interface.

Core capabilities:
1. Sync + async LLM calls.
2. Structured output with Pydantic.
3. Tool calling.
4. Batch and streaming APIs.
5. Fallback model chains.
6. Retry policies with hooks.
7. Cost + usage tracking.
8. Cache support.
9. Agent SDK routing (`claude-code`, `codex`).
10. MCP and direct Python tool loops for agentic workflows.
11. Local JSONL + SQLite observability logging.
12. Experiment/eval helpers.

## 3. Current Repository Shape

Top modules by size:
1. `llm_client/client.py`: 5740 LOC
2. `llm_client/mcp_agent.py`: 4365 LOC
3. `llm_client/io_log.py`: 2234 LOC
4. `llm_client/agents.py`: 1386 LOC
5. `llm_client/__main__.py`: 972 LOC

Other notable modules:
1. `llm_client/models.py`
2. `llm_client/tool_utils.py`
3. `llm_client/foundation.py`
4. `llm_client/task_graph.py`
5. `llm_client/experiment_eval.py`

## 4. High-Level Architecture

The main orchestration lives in `llm_client/client.py`.

Primary entry points:
1. `call_llm` / `acall_llm`
2. `call_llm_structured` / `acall_llm_structured`
3. `call_llm_with_tools` / `acall_llm_with_tools`
4. `stream_llm` / `astream_llm`
5. Batch wrappers (`call_llm_batch`, `acall_llm_batch`, structured variants)

Routing layers currently mixed into call paths:
1. Model normalization + provider routing.
2. Responses API routing for bare GPT-5 names.
3. Agent SDK routing for `claude-code` and `codex`.
4. Gemini native REST optional path.
5. MCP/direct-tool loops.

Observability:
1. JSONL + SQLite writes for every call.
2. Experiment run/item tracking.
3. Foundation event logging.

## 5. Concrete Technical Findings From Review

## 5.1 Contract Drift From Routing Defaults

Behavior changed so that bare model IDs are auto-normalized to OpenRouter IDs by default.

Where:
1. `llm_client/client.py`:
2. `_openrouter_routing_enabled` (default `on`)
3. `_normalize_model_for_routing`
4. `_build_model_chain`

Effects:
1. `result.model` often returns normalized values like `openrouter/openai/gpt-4`, not caller input.
2. Hooks and fallback callbacks see normalized model IDs.
3. Some expectations around omitted `api_base` break because OpenRouter base gets auto-injected.
4. Bare GPT-5 names can be normalized away from the Responses API path.

## 5.2 Test Failures Cluster Around Contract Changes

Observed test outcomes:
1. `pytest -q` produced many failures primarily around routing/contract expectations.
2. Running with `LLM_CLIENT_OPENROUTER_ROUTING=off` reduced failures dramatically.
3. Remaining failures under routing=off were mostly:
4. `_require_tags` helper signature/test mismatch.
5. Warning category mismatch for `gpt-4o` (`UserWarning` vs expected `DeprecationWarning`).

Interpretation:
1. Most breakage is not catastrophic runtime failure.
2. Most breakage is API/contract drift and test suite not updated to a stable explicit policy.

## 5.3 Monolithic Core Functions and Duplication

Large functions in `client.py`:
1. `call_llm` ~323 lines
2. `acall_llm` ~295 lines
3. `call_llm_structured` ~326 lines
4. `acall_llm_structured` ~326 lines

Large state-machine function:
1. `_agent_loop` in `mcp_agent.py` ~1210 lines

Problems:
1. Multiple responsibilities per function.
2. High branch complexity.
3. Sync/async duplication raises maintenance burden.
4. Harder to reason about invariants and side effects.

## 5.4 Logging and Data Sensitivity Risk

By default logging is enabled and stores full messages/responses.

Where:
1. `llm_client/io_log.py`
2. `_enabled` default from env fallback = on.
3. `log_call` writes full prompts and responses.
4. `_truncate_messages` does not redact.

Risk:
1. Potential accidental storage of sensitive data in local files/DB.
2. No explicit redaction policy enforcement in core path.

## 5.5 Global Monkey-Patch Risk in Codex Path

`agents.py` monkey-patches `asyncio.create_subprocess_exec` inside patched Codex run logic to increase buffer size.

Risk:
1. Process-wide async behavior change while patched.
2. Fragile under concurrency.
3. Hard to reason about side effects outside codex path.

## 5.6 Typing Debt Is High Despite Strict Mypy Config

`pyproject.toml` enables strict mypy, but current run reports large error count across many files.

Observed:
1. `mypy llm_client` reported 123 errors in 15 files.
2. Errors include `Any` leakage, return type mismatches, unused ignores, and attr-defined mismatches.

Interpretation:
1. Type discipline has drifted from policy.
2. Strict setting does not currently enforce quality in CI as a gate.

## 5.7 Input/Cache Robustness Edge Cases

Observed issues:
1. Empty model string can surface as opaque `LLMError None`.
2. Cache key serialization can raise if kwargs include non-JSON-serializable values (example: `set`).
3. Responses input conversion stringifies content lists/dicts, which can degrade multimodal semantics.

## 5.8 Integration Test Strategy Is Not Isolated

`tests/integration_test.py` uses real APIs and is under default pytest discovery (`testpaths = ["tests"]`).

Risk:
1. Costly/flaky runs in environments without stable API keys/network.
2. Mixed signal from unit and integration behaviors.
3. Increases noise during refactors.

## 6. Why We Are Considering Rewrite

Rewrite drivers:
1. Perceived accumulation of policy logic in central call paths.
2. Contract drift and behavior surprises.
3. Large orchestrator functions with repeated code.
4. Growing complexity in tool/agent loops.
5. Difficulty guaranteeing non-regression when changing routing logic.

Counterweights:
1. Broad feature surface already implemented.
2. Large test suite exists and mostly passes when routing policy is explicit.
3. A rewrite would likely reintroduce many edge-case bugs already solved here.

## 7. Preliminary Internal Recommendation

Current recommendation is not a full rewrite now.

Preferred direction:
1. Stage an architectural refactor around a stable compatibility contract.
2. Separate policy from transport and from orchestration.
3. Keep behavior behind feature flags while converging tests and docs.

## 8. Refactor Plan We Are Considering

Phase 1: Contract Freeze
1. Define canonical API contract for model identity:
2. Decide and document whether `result.model` is caller model, execution model, or both.
3. Decide warning taxonomy (`DeprecationWarning` vs `UserWarning`) and enforce consistently.
4. Decide default routing policy and make explicit in tests.

Phase 2: Router Extraction
1. Extract model normalization/routing into a pure module with typed outputs.
2. Make call paths consume a `ResolvedCallPlan` object.
3. Reduce implicit env-driven behavior in deep execution paths.

Phase 3: Execution Kernel Split
1. Factor shared retry/fallback loop into reusable sync/async core.
2. Reduce duplicated logic between `call_llm` and `acall_llm`.
3. Isolate responses/completions/agent transports behind strategy adapters.

Phase 4: Tool/Agent Loop Decomposition
1. Break `_agent_loop` into composable policy stages:
2. turn setup
3. tool call validation/coercion
4. tool execution
5. loop control decisions
6. metrics/event logging
7. Make each stage unit-testable independently.

Phase 5: Observability Hardening
1. Add configurable redaction policy for prompts/responses before persistence.
2. Add explicit PII-safe mode.
3. Make integration tests opt-in via marker/env gate.

Phase 6: Quality Gates
1. Establish CI lanes:
2. unit-fast
3. typecheck
4. integration-opt-in
5. Enforce mypy and core test baseline incrementally (ratchet policy).

## 9. Decisions We Need Help Making

Please advise on these tradeoffs:
1. Should default routing be explicit opt-in rather than implicit env default?
2. Should we preserve caller model ID separately from resolved execution model?
3. How should we version or deprecate behavior-changing defaults without breaking users?
4. What minimum architecture split gives the largest debt reduction quickly?
5. How to redesign warning/deprecation semantics for agent + human consumers?
6. How to harden logging without losing observability value?
7. What test strategy best protects a staged refactor in a fast-moving LLM wrapper?

## 10. Constraints and Requirements

Hard requirements:
1. Keep current user-facing feature set.
2. Avoid long freeze.
3. Preserve sync and async APIs.
4. Keep fallback/retry/tooling support.
5. Maintain agent SDK support (`claude-code`, `codex`) and MCP/direct tool loops.

Operational requirements:
1. Safe for production-like usage in multi-provider environments.
2. Predictable behavior under env configuration changes.
3. Stronger non-regression guarantees before major policy changes.

## 11. What We Want Back From ChatGPT

Please return:
1. A go/no-go recommendation on rewrite vs refactor.
2. A concrete 4-8 week technical plan with checkpoints.
3. A proposed module boundary map (what to split first).
4. A compatibility strategy for model normalization and warnings.
5. A test migration plan (unit/integration/typecheck gating).
6. A risk register with mitigation for each phase.
7. Suggested ADR templates for key policy decisions.

## 12. Useful Commands and Observations

Commands used during review:
1. `pytest -q`
2. `LLM_CLIENT_OPENROUTER_ROUTING=off pytest -q`
3. `mypy llm_client`
4. `wc -l llm_client/*.py llm_client_mcp_server.py | sort -nr`

Observed:
1. Most failures tie back to routing default and warning-type contract changes.
2. The codebase appears salvageable with disciplined decomposition.
3. The main risk is uncontrolled policy drift, not absence of capability.

## 13. Current Working Conclusion

We should not rewrite from scratch immediately.  
We should perform a staged refactor, starting with explicit contract decisions and router extraction, then split execution and agent loop orchestration into testable components.
