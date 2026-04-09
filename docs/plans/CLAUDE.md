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
| 6 | [Simplification & Observability](06_simplification-and-observability.md) | High | ✅ Complete | - |
| 7 | [Stream Lifecycle Heartbeat and Stagnation Visibility](07_stream_lifecycle_observability.md) | High | ✅ Complete | - |
| 8 | [llm_client Subtree Instruction Rollout](08_llm_client-subtree-instructions.md) | Medium | ✅ Complete | - |
| 9 | [Replay and Divergence Diagnosis](09_replay-and-divergence-diagnosis.md) | High | ✅ Complete | - |
| 10 | [API Reference Generation Pipeline](10_api-reference-generation-pipeline.md) | High | ✅ Complete | - |
| 11 | [Program E Module Size Reduction](11_program-e-module-size-reduction.md) | High | ✅ Complete | 6 |
| 12 | [Module Reorganization (Flat → Layered)](12_module-reorganization.md) | High | ✅ Complete | 11 |
| 13 | [SDK Adapter Simplification](13_sdk-adapter-simplification.md) | Medium | ✅ Complete | 12 |
| 14 | [Batch Progress & Stagnation Detection](14_batch-progress-and-stagnation.md) | High | ✅ Complete | - |
| 15 | [Centralize Hardcoded Defaults into ClientConfig](15_centralize-defaults.md) | Low | ❓ Cancelled — "pure refactoring, no consumer value" (2026-03-25, confirmed 2026-04-04) | - |
| 16 | [Remove Compatibility Stubs](16_remove-compatibility-stubs.md) | Medium | ✅ Complete | 12 |
| 17 | [text_runtime Sync/Async Deduplication](17_text-runtime-dedup.md) | High | ✅ Complete | - |
| 18 | [Agent Loop Error Budget and Retry Policy](18_agent_loop_error_budget.md) | High | ✅ Complete | - |
| 19 | [Agent Planning and Working Memory](19_agent_planning_and_working_memory.md) | High | ✅ Complete | - |
| 20 | [Makefile and Requirements](20_makefile_and_requirements.md) | Medium | ✅ Complete | - |
| 21 | [Runtime Durability Follow-Ups From Grounded Research](21_runtime_durability_followups_from_grounded_research.md) | High | ✅ Complete | - |
| 22 | [Capability Ownership And Sanctioned Worktree Alignment](22_capability-ownership-and-sanctioned-worktree-alignment.md) | High | ✅ Complete | 21 |
| 23 | [Authoritative coordination wave-1 rollout](23_authoritative-coordination-wave-1-rollout.md) | Critical | ✅ Complete | - |
| 24 | [Workflow Kit Manifest, Validator, and Runtime Adapter Proving Slice](24_workflow-kit-manifest-validator-and-runtime-adapter-proving-slice.md) | — | Cancelled — execution strategy layer reassigned to agentic_scaffolding; packaging scope deferred to PROJECTS_DEFERRED. See `project-meta/docs/ops/ADR-2026-04-04-workflow-portability-revised-execution-strategies.md` (2026-04-04). | — |
| 25 | [Gemini exhaustion fallback hardening](25_gemini_exhaustion_fallback_hardening.md) | High | ✅ Complete | Digimon GraphRAG rebuild recovery |
| 91 | [Pending-Atom Submit Churn Requires TODO Progress](91_pending_atom_submit_churn_requires_todo_progress.md) | High | 🚧 In Progress | DIGIMON controller anti-churn verification |


## Status Key

| Status | Meaning |
|--------|---------|
| Planned | Ready to implement |
| In Progress | Being worked on |
| Blocked | Waiting on dependency |
| Complete | Implemented and verified |
| Cancelled | Explicitly rejected; no planned work |

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
