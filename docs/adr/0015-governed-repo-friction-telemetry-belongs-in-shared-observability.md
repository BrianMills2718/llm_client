# ADR 0015: Governed-Repo Friction Telemetry Belongs In Shared Observability

Status: Accepted  
Date: 2026-03-19

## Context

The active stack now has governed repos with required-reading hooks, read
tracking, and repo-local `.claude/hook_log.jsonl` files. That proves local
behavior, but it does not yet answer the operator questions that matter for
rollout quality:

1. which repos or files produce the most read-gate friction,
2. which required docs are repeatedly missed,
3. whether a gate or context-injection change reduced repeated blocks,
4. whether hook friction correlates with downstream task success, retries, or
   abandonment.

If we leave governed-repo telemetry in per-repo JSONL files, the architecture
degrades quickly:

1. every repo would need to invent its own importer/reporting path,
2. cross-repo diagnosis would stay manual,
3. `prompt_eval` could accidentally become a second primary observability
   backend just because it runs experiments,
4. the active stack would lack one authoritative answer to "is the governed
   rollout actually helping?"

`llm_client` already owns the shared runtime and observability substrate for the
ecosystem. Governed-repo friction is not a repo-local artifact; it is a
cross-project operational signal.

## Decision

1. `llm_client` owns the canonical shared telemetry boundary for governed-repo
   hook and read-gate activity.
2. Repo-local `.claude/hook_log.jsonl` files remain edge buffers and
   compatibility artifacts, not the primary analysis backend.
3. The shared governed-repo telemetry contract must preserve, at minimum:
   - repo identity,
   - session identity when available,
   - explicit indication when session identity is degraded or event-local rather
     than stable,
   - hook name,
   - decision (`block`, `allow`, `read`, `error`, or equivalent),
   - target file path,
   - required/missing reads,
   - bounded context metadata such as whether `additionalContext` was emitted
     and how many bytes it carried,
   - trace/run linkage when available.
4. When governed-repo experiments set explicit experiment metadata, raw hook rows
   should preserve `experiment_id`, `variant_id`, and `downstream_run_id` at the
   source instead of relying only on importer-time overrides.
5. The default shared telemetry path must be metadata-first. Full injected
   document contents are not persisted by default just because they passed
   through a hook.
6. Importers and validators must fail loud on malformed governed-repo telemetry
   rather than silently dropping rows.
7. `prompt_eval` may consume governed-repo telemetry to run controlled
   experiments and comparative analysis, but it does not own the primary sink,
   schema, or storage contract.
8. Query/report helpers for governed-repo friction belong in `llm_client`
   observability surfaces and any MCP/CLI interfaces built on top of them.

## Consequences

Positive:
1. Cross-repo rollout analysis gets one canonical home.
2. Hook friction can be correlated with the rest of the active stack's shared
   traces, costs, and outcomes.
3. `prompt_eval` can focus on experiments without becoming a second
   observability backend.
4. The governed rollout can be measured before it is expanded broadly.
5. Session-level friction reports become more trustworthy because degraded
   session identity is explicit instead of being silently fabricated.

Negative:
1. `llm_client` observability scope grows beyond pure LLM call telemetry.
2. We must define a stable schema for events that originate outside
   `llm_client` runtime calls.
3. Importing repo-local logs creates another ingestion path that must be
   validated and tested carefully.

## Testing Contract

1. The shared governed-repo telemetry schema must reject malformed event shapes
   deterministically.
2. Importing canonical hook-log rows into shared observability must preserve
   repo/session/file/decision metadata without silent loss.
3. Query helpers must support cross-repo friction summaries such as
   block/allow/error counts, top missing reads, and repeated-friction files.
4. Comparative experiment/report helpers must be able to group governed-repo
   telemetry by explicit experiment metadata without inventing a second
   observability store.
5. When stable session identity is unavailable, shared telemetry must mark the
   degradation explicitly and avoid cross-session recurrence metrics that depend
   on fake session grouping.
