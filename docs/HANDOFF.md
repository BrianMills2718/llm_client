# Handoff: llm_client runtime follow-up relevant to DIGIMON

Updated: 2026-04-14
Working branch: `fix/instructor-retry-unwrapping`

## Current Posture

- `llm_client` is in maintenance mode, not an open-ended refactor sprint.
- Plans `#25`, `#26`, and `#27` already landed the GraphRAG-critical provider
  exhaustion fixes:
  - monthly spend-cap exhaustion is treated as immediate failover,
  - exhausted models are cooled down across calls,
  - multi-hour provider retry hints fail over instead of sleeping inside the
    call.
- The branch is ahead of `origin/fix/instructor-retry-unwrapping` by two
  unpublished commits:
  - `2308465` `[Plan #178] Add goal metadata to tool decorator`
  - `3739578` `[Plan #179] Add tool complexity and routing metadata`

Those two commits are repo-local metadata improvements, not GraphRAG blockers,
but they were not yet pushed before this handoff.

## What Is Finished

- Provider exhaustion classification is materially better than it was during the
  failed DIGIMON rebuild attempts.
- Routing now suppresses recently exhausted models instead of re-probing them
  on every chunk.
- Long retry windows now fail over rather than sleeping for hours inside a
  single batch call.
- Tool decorator metadata now includes:
  - `goal`
  - `complexity`
  - `routing hints`

The new tool metadata is implemented and tested, but downstream consumers have
not yet been broadly updated to exploit it.

## Unfinished Work

### 1. Plan `#91` is still the live shared-runtime blocker for DIGIMON controller churn

Plan: `docs/plans/91_pending_atom_submit_churn_requires_todo_progress.md`

Problem:

- repeated `submit_answer` attempts rejected for `pending_atoms` can still
  escalate into forced-terminal behavior;
- that is the wrong shared policy for DIGIMON’s unresolved-hop failure family;
- the correct shared behavior is to require genuine TODO/evidence progress
  before another submit is attempted.

Why this still matters:

- DIGIMON Plan `#30` should not hand-roll an app-local patch for this;
- if the controller keeps churning around unresolved atoms, the shared runtime
  is still the right fix surface.

Files:

- `llm_client/agent/mcp_turn_tools.py`
- `llm_client/agent/mcp_turn_outcomes.py`
- `llm_client/agent/mcp_turn_execution.py`
- `tests/test_mcp_agent.py`

Acceptance already written in the plan:

- pending-atom submit retries are suppressed until TODO progress occurs;
- the runtime no longer emits forced-final acceptance for this family;
- existing submit-evidence gating remains green.

### 2. The new tool-routing metadata is landed but not yet widely consumed

Commits:

- `2308465` goal metadata
- `3739578` complexity/routing metadata

What is still unfinished:

- deciding which downstream planners/runtimes should use the new metadata first;
- validating that the metadata helps routing choices instead of just enriching
  the registry surface;
- documenting the intended consumer contract more explicitly if these fields are
  now considered stable substrate.

This is not a blocker for DIGIMON recovery, but it is open follow-through.

### 3. DIGIMON rebuild resilience is improved, not fully proved

The retry/failover code is in much better shape, but one thing still lacks a
full end-to-end proof:

- a successful complete long GraphRAG rebuild under real provider pressure,
  with quotas rotating and fallback legs degrading in different ways.

The code changes are real. The remaining uncertainty is operational proof, not
obvious missing implementation.

## Recommended Next Steps

1. Push this branch so the tool-metadata commits and current handoff are no
   longer local-only.
2. When DIGIMON restarts controller anti-churn work, implement Plan `#91`
   before app-local controller patches.
3. After the next real long-running DIGIMON rebuild or benchmark batch, review
   whether any remaining provider failure mode still belongs in shared runtime.
4. Decide whether `goal` / `complexity` / routing metadata should be treated as
   stable planner contract or as experimental substrate hints.

## Read First

1. `CLAUDE.md`
2. `docs/plans/01_master-roadmap.md`
3. `docs/plans/CLAUDE.md`
4. `docs/plans/91_pending_atom_submit_churn_requires_todo_progress.md`
5. `llm_client/tools/decorator.py`
6. `tests/test_tool_decorator.py`

## Verification State

The tool-metadata commits already include test coverage in
`tests/test_tool_decorator.py`. This handoff update itself is documentation-only
and does not change code behavior.

## Bottom Line

`llm_client` is not the main blocker anymore. The critical GraphRAG quota and
retry fixes are already landed. The two unfinished truths are:

1. Plan `#91` still needs to land if DIGIMON’s submit-churn family stays live.
2. The new tool-routing metadata exists, but its real consumer contract is not
   yet fully operationalized.
