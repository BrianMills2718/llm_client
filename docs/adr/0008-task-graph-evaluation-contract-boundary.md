# ADR 0008: Task Graph and Evaluation Contract Boundary

Status: Accepted  
Date: 2026-02-23

## Context

`llm_client/task_graph.py` and `llm_client/experiment_eval.py` govern how DAG
tasks are executed, validated, and recorded for follow-up analysis. The prior
required-reading coupling referenced only `docs/TASK_GRAPH_DESIGN.md`, which is
useful but not a stable contract source.

## Decision

1. `llm_client/task_graph.py` is the canonical runtime boundary for:
   - graph loading and dependency ordering,
   - task dispatch and validation,
   - fail-loud execution behavior.
2. `llm_client/experiment_eval.py` is the canonical evaluation boundary for:
   - experiment record consumption,
   - outcome scoring and trend analysis.
3. Compatibility rule:
   - changes that alter task-graph execution semantics or experiment payload
     expectations must update this ADR before release.
4. The read-gate coupling for these modules will reference this ADR as the
   stable contract source.

## Consequences

Positive:
1. Stable contract for orchestration/evaluation behavior.
2. Lower ambiguity for reviewers and contributors touching these modules.
3. Cleaner read-gate coupling map without interim uncertainty notes.

Negative:
1. Design details still live in `docs/TASK_GRAPH_DESIGN.md`; both documents must
   be kept aligned.
2. Future schema evolution in experiment records now requires explicit policy
   updates here.

## Testing Contract

1. Task graph tests must continue to validate dependency ordering, validation
   enforcement, and fail-loud behavior.
2. Evaluation tests must verify backward-compatible handling of experiment
   records used by current analysis workflows.
