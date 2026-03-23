## Program E Handoff (Claude Code)

**Context**
 - Plan 11 is the active Program E effort. The latest verified slices extracted the call-contract cluster, collapsed the completion/responses callback-passing pattern, and moved routing/dispatch helpers into dedicated modules. `client.py` is now below the Program E hard threshold (1,494 lines < 1,500 limit).
 - The repo is on `digimon-stable`, worktree clean, and API reference + plan docs are in sync.
 - `client.py` is no longer a hard-threshold blocker. The remaining modules above the soft target (~1,200 lines) are: `mcp_turn_execution.py` (1,339), `observability/experiments.py` (1,322), `agents_codex.py` (1,317), `agent_contracts.py` (1,228), `io_log.py` (1,222). These are between the soft target and hard limit, so Program E can proceed to documented justification or further decomposition for each.

**Current verification**
 - `pytest -q tests/test_call_contracts.py tests/test_client.py tests/test_public_surface.py tests/test_client_lifecycle.py tests/test_observability_defaults.py` (256 passing).
 - Focused helper suites for Responses, completion, and background polling all pass.
 - `python scripts/meta/generate_api_reference.py --write` run after every tranche; API reference docs are in sync.

**Modules extracted during this session**
 - `call_contracts.py`: expanded from 129→679 lines (empty-response classification, schema-error detection, GPT-5 sampling, param coercion, agent-model detection, execution-mode validation, model deprecation)
 - `client_dispatch.py`: new 519-line module (routing plan resolution, result finalization, structured-call result building, agent-loop orchestration, call-event logging, text/schema utilities)
 - `completion_runtime.py`: refactored to import directly from `call_contracts`/`model_detection` instead of receiving callbacks
 - `responses_runtime.py`: refactored to import directly from `call_contracts`/`background_runtime` instead of receiving callbacks

**Next action**
1. Assess whether the 5 modules between soft target and hard limit need further decomposition or can be documented as justified exceptions.
2. If all are justified, Program E's closeout criteria can be honestly evaluated.
3. If any need decomposition, the same thin-slice approach applies.
