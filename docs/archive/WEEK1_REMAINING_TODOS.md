# Week-1 Remaining TODOs

Date: 2026-02-22
Owner: Codex + parallel agents

## Immediate Execution Queue

- [x] ADRs added for model identity, routing precedence, warning taxonomy.
- [x] Integration tests gated behind `integration` marker + `LLM_CLIENT_INTEGRATION=1`.
- [x] Additive result identity fields added:
  - `requested_model`
  - `resolved_model` (best-effort)
  - `routing_trace` (minimal trace)
- [x] Characterization tests added for identity fields and MCP loop behavior.
- [x] Drift fixes in targeted tests:
  - `_require_tags(..., caller=...)` callsite mismatch
  - `gpt-4o` warning category (`UserWarning` for outclassed model)

## Remaining Week-1 Items (In Progress)

- [x] Stabilize `tests/test_client.py` defaults so routing behavior is explicit and deterministic.
- [x] Eliminate remaining live-network paths from unit tests (keep all provider calls mocked).
- [x] Reconcile legacy `result.model` assertions with explicit routing-policy setup in tests.
- [x] Run and pass:
  - `tests/test_client.py`
  - `tests/test_mcp_agent.py`
  - `tests/test_tool_utils.py`
  - `tests/test_model_identity_contract.py`

Validation snapshot (2026-02-22):
- `pytest -q` -> `749 passed, 1 skipped`

## Follow-up (Week 2+)

- [x] Extract pure router (`resolve_call(request, config) -> ResolvedCallPlan`).
- [x] Move from env-driven routing to typed config entrypoint while preserving compatibility flags.
- [x] Add stable warning codes (`LLMC_WARN_*`) with machine-readable metadata.
- [x] Unify/clarify long-term `result.model` semantics via follow-up ADR and migration window.

Validation snapshot (2026-02-22, post follow-up implementation):
- `pytest -q` -> `757 passed, 1 skipped`
