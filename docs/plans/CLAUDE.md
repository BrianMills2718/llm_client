# Implementation Plans

Track all implementation work here.

> **Deprecation notice (2026-03-21):** `llm_client` is being superseded by
> `llm_client_v2`. No new plans will be added here. Existing plans are
> retained for historical reference.

## Gap Summary

| # | Name | Priority | Status | Blocks |
|---|------|----------|--------|--------|
| 1 | [LLM Client Master Roadmap](01_master-roadmap.md) | Highest | ✅ Complete | - |
| 2 | [Client Boundary Hardening Program](02_client-boundary-hardening.md) | High | ✅ Complete | - |
| 3 | [Model Policy Modernization](03_model-policy-modernization.md) | High | ✅ Complete | - |
| 4 | [Workflow Layer Boundary](04_workflow-layer-boundary.md) | Medium | ✅ Complete | - |
| 5 | [Eval Boundary Cleanup](05_eval-boundary-cleanup.md) | Medium | ✅ Complete | - |
| 6 | [Simplification & Observability](06_simplification-and-observability.md) | High | ❓ Deferred (2026-03-20) | - |
| 7 | [Call Liveness and Active-Call Visibility](07_call-liveness-and-active-call-visibility.md) | High | ✅ Complete | - |

### Plans 8-13 (unmerged — developed on `plan-07-governed-repo-contract-alignment` branch)

The following plans were implemented and completed on the
`plan-07-governed-repo-contract-alignment` branch but never merged to `main`.
Their plan files and implementation code live only on that branch.

| # | Name | Priority | Status | Blocks |
|---|------|----------|--------|--------|
| 8 | Call Liveness and Timeout Policy | High | ✅ Complete (branch only) | - |
| 9 | Lifecycle Heartbeat and Active Call Query | High | ✅ Complete (branch only) | - |
| 10 | Governed Repo Friction Observability | Highest | ✅ Complete (branch only) | - |
| 11 | Context Injection Experiment Support | High | ✅ Complete (branch only) | - |
| 12 | Progress-Aware Idle Detection | High | ✅ Complete (branch only) | - |
| 13 | Same-Host Orphaned Call Reaping | High | ✅ Complete (branch only) | - |

## Status Key

| Status | Meaning |
|--------|---------|
| Planned | Ready to implement |
| In Progress | Being worked on |
| Blocked | Waiting on dependency |
| Complete | Implemented and verified |

## Creating a New Plan

1. Copy `TEMPLATE.md` to `NN_name.md`
2. Fill in gap, steps, required tests
3. Add to this index
4. Commit with `[Plan #N]` prefix

## Trivial Changes

Not everything needs a plan. Use `[Trivial]` for:
- Less than 20 lines changed
- No changes to `llm_client/` (production code)
- No new files created

```bash
git commit -m "[Trivial] Fix typo in README"
```

## Completing Plans

```bash
python scripts/meta/complete_plan.py --plan N
```

This verifies tests pass and records completion evidence.
