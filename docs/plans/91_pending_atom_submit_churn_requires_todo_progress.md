# Plan #91: Pending-Atom Submit Churn Requires TODO Progress

**Status:** In Progress
**Type:** implementation
**Priority:** High
**Blocked By:** None
**Blocks:** DIGIMON Phase 2 controller anti-churn verification

## Gap

The shared MCP turn runtime currently treats repeated `submit_answer`
rejections with `reason_code=pending_atoms` plus
`requires_forced_terminal_path=true` as a signal to force final answer
acceptance. In DIGIMON this produces a misleading path:

- unresolved semantic-plan atoms remain pending,
- normal submit was never accepted,
- but the loop escalates into forced-final submission anyway.

That is the wrong shared policy for this failure family. Pending atoms indicate
the controller still owes semantic progress, not that the runtime should accept
an ungrounded final answer.

## Target

When `submit_answer` is rejected for pending atoms, the shared runtime should:

1. require new evidence before retry when requested,
2. require TODO-state progress before retry,
3. suppress repeated submit attempts until those conditions are satisfied,
4. avoid converting this state directly into `CONTROL_CHURN_THRESHOLD_EXCEEDED`
   forced-final acceptance.

## Acceptance

- Focused agent tests show repeated pending-atom submit retries are suppressed
  until TODO progress occurs.
- The runtime no longer emits `SUBMIT_FORCED_ACCEPT_FORCED_FINAL` for this
  specific rejection family.
- Existing submit-evidence gating tests still pass.

## Files

- `llm_client/agent/mcp_turn_tools.py`
- `llm_client/agent/mcp_turn_outcomes.py`
- `llm_client/agent/mcp_turn_execution.py`
- `tests/test_mcp_agent.py`
