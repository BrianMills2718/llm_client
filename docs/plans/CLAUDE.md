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
| 8 | [llm_client Subtree Instruction Rollout](08_llm_client-subtree-instructions.md) | Medium | 📋 Planned | - |
| 9 | [Replay and Divergence Diagnosis](09_replay-and-divergence-diagnosis.md) | High | ✅ Complete | - |
| 10 | [API Reference Generation Pipeline](10_api-reference-generation-pipeline.md) | High | ✅ Complete | - |
| 11 | [Program E Module Size Reduction](11_program-e-module-size-reduction.md) | High | ✅ Complete | 6 |
| 12 | [Module Reorganization (Flat → Layered)](12_module-reorganization.md) | High | ✅ Complete | 11 |
| 13 | [SDK Adapter Simplification](13_sdk-adapter-simplification.md) | Medium | ✅ Complete | 12 |
| 14 | [Batch Progress & Stagnation Detection](14_batch-progress-and-stagnation.md) | High | ✅ Complete | - |
| 15 | [Centralize Hardcoded Defaults into ClientConfig](15_centralize-defaults.md) | Medium | 📋 Planned | - |
| 16 | [Remove Compatibility Stubs](16_remove-compatibility-stubs.md) | Medium | 📋 Planned | 12 |
| 17 | [text_runtime Sync/Async Deduplication](17_text-runtime-dedup.md) | High | 📋 Planned | - |


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
