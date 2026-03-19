# ADR 0014: Emit Heartbeats and Non-Destructive Stall Markers for Long-Running Calls

Status: Accepted  
Date: 2026-03-19

## Context

ADR 0013 established that provider request timeouts are not the default
liveness mechanism and introduced `started -> completed/failed` lifecycle
events for public non-streaming calls.

That was necessary but not sufficient. A call that only emits `started` and
then goes silent for several minutes is still hard to reason about:

1. it may be healthy and merely slow,
2. it may be stalled in provider or transport space,
3. operators cannot answer "what is still running right now?" without manually
   diffing lifecycle logs.

We need one more thin slice that improves visibility without reintroducing
blunt cancellation behavior.

## Decision

1. Public non-streaming text and structured call wrappers should emit periodic
   lifecycle `heartbeat` events while a call remains in flight.
2. Heartbeats indicate that `llm_client` is still actively waiting on the call.
   They are not a claim of provider-side token progress.
3. `llm_client` may emit a non-terminal lifecycle `stalled` event when a call
   exceeds a configured orchestration stall threshold.
4. `stalled` is an observability classification, not an automatic cancellation
   action.
5. Heartbeat and stall settings must be controllable without provider-specific
   branching:
   - process defaults via environment,
   - per-call overrides via public kwargs.
6. `llm_client` should expose a query helper that returns the latest known
   active non-terminal lifecycle state for current calls.

## Consequences

Positive:
1. Operators can distinguish "still alive" from "silent since started".
2. Long-running calls can be monitored without forcing transport cancellation.
3. The next orchestration step, if needed later, can build on explicit
   heartbeat and stall state instead of inventing a second liveness channel.

Negative:
1. Lifecycle event volume increases for long-running calls.
2. `stalled` is only a threshold-based warning; it cannot prove a provider is
   truly hung.
3. Sync wrappers need a small background watcher thread to emit heartbeats.

## Testing Contract

1. Foundation validation must accept lifecycle `heartbeat` and `stalled`
   phases.
2. Sync and async public wrappers must emit heartbeats for deliberately slow
   calls when heartbeat intervals are configured low.
3. Sync and async public wrappers must emit `stalled` when a call exceeds the
   configured threshold, while still allowing the call to complete normally.
4. Observability query helpers must report active in-flight calls from
   Foundation lifecycle events.
