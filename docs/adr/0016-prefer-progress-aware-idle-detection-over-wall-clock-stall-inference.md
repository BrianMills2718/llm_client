# ADR 0016: Prefer Progress-Aware Idle Detection Over Wall-Clock Stall Inference

Status: Accepted
Date: 2026-03-19

Related: Builds on ADR 0013 (lifecycle baseline) and ADR 0014
(heartbeats/stall markers). ADR 0009 (background polling) is one of the
progress-observable paths instrumented by this decision.

## Context

ADR 0013 established that provider request timeouts are transport controls,
not the primary liveness mechanism. ADR 0014 then added lifecycle
`started/heartbeat/stalled/completed/failed` events plus an active-call query.

That slice improved visibility, but it intentionally stopped short of claiming
that `stalled` proves a provider or runtime is hung. For plain non-streaming
calls, especially structured-output calls, `llm_client` often has no truthful
progress signal between:

1. request sent, and
2. final provider response returned.

In that case, elapsed wall-clock time is not enough to distinguish:

1. a healthy slow call,
2. a provider-side long-running job,
3. an actually hung execution path.

Some paths do expose real progress signals, however:

1. streaming calls expose chunk/tool-delta arrival,
2. background-polled Responses calls expose successful poll activity and status
   transitions.

We need a clearer contract so `llm_client` only claims progress-aware stall
detection where it can observe real progress.

## Decision

1. `llm_client` must distinguish liveness from progress.
2. Lifecycle `heartbeat` continues to mean "the client runtime is still
   waiting on this call." It is not a claim of provider-side progress.
3. Progress-aware idle/stall detection must only be used on execution paths
   that expose explicit observable progress signals.
4. The first progress-observable paths are:
   - streaming text/tool-call flows,
   - background-polled Responses calls.
5. Non-streaming opaque calls, including current non-streaming structured
   calls, may emit liveness heartbeats and elapsed-time warnings, but must not
   be treated as progress-aware stalled executions unless the transport shape
   changes to expose real progress.
6. The lifecycle contract should gain additive progress-aware metadata and/or
   events so query surfaces can report:
   - whether progress is observable for a call,
   - the latest known progress timestamp,
   - idle duration since last observed progress,
   - the source of observed progress when available.
7. Query/report surfaces should present "active and progressing" separately
   from merely "active and waiting."

## Consequences

Positive:
1. Operators get a more truthful distinction between slow and stuck.
2. `llm_client` avoids overclaiming on opaque provider paths.
3. Future orchestration controls can build on a real progress signal instead of
   elapsed-time guesses.

Negative:
1. Lifecycle/query logic becomes more complex because not every path supports
   the same observability depth.
2. Some callers will still only get waiting/heartbeat semantics until the
   underlying transport exposes richer progress.
3. True progress-aware support for non-streaming structured extraction may
   require a larger transport redesign rather than a small instrumentation
   patch.

## Non-Goals

1. This ADR does not require automatic cancellation.
2. This ADR does not claim that every provider path can support progress-aware
   idle detection immediately.
3. This ADR does not require `llm_client` to invent synthetic progress for
   opaque calls.

## Testing Contract

1. Streaming paths must prove that observed stream chunks advance progress
   state.
2. Background-polling paths must prove that successful polls/status changes
   advance progress state.
3. Query helpers must distinguish:
   - active waiting calls,
   - active progressing calls,
   - progress-aware stalled calls.
4. Opaque non-streaming structured calls must not be labeled progress-aware
   stalled unless a real progress signal is available.
