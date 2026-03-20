# Implementation Plans

Track all implementation work here.

## Gap Summary

| # | Name | Priority | Status | Blocks |
|---|------|----------|--------|--------|
| 1 | [LLM Client Master Roadmap](01_master-roadmap.md) | Highest | ✅ Complete | - |
| 2 | [Client Boundary Hardening Program](02_client-boundary-hardening.md) | High | ✅ Complete | - |
| 3 | [Model Policy Modernization](03_model-policy-modernization.md) | High | ✅ Complete | - |
| 4 | [Workflow Layer Boundary](04_workflow-layer-boundary.md) | Medium | ✅ Complete | - |
| 5 | [Eval Boundary Cleanup](05_eval-boundary-cleanup.md) | Medium | ✅ Complete | - |
| 6 | [Simplification & Observability](06_simplification-and-observability.md) | High | 🚧 In Progress | - |
| 7 | [Governed Repo Contract Alignment](07_governed_repo_contract_alignment.md) | High | ✅ Complete | - |
| 8 | [Call Liveness and Timeout Policy](08_call_liveness_and_timeout_policy.md) | High | ✅ Complete | - |
| 9 | [Lifecycle Heartbeat and Active Call Query](09_lifecycle_heartbeat_and_active_call_query.md) | High | 🚧 In Progress | - |
| 10 | [Governed Repo Friction Observability](10_governed_repo_friction_observability.md) | Highest | ✅ Complete | - |
| 11 | [Context Injection Experiment Support](11_context_injection_experiment_support.md) | High | 📋 Planned | - |
| 12 | [Progress-Aware Idle Detection](12_progress_aware_idle_detection.md) | High | 📋 Planned | - |

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
