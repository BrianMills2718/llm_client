# ADR 0013: Stream Lifecycle Heartbeat and Stagnation Observability

Status: Accepted  
Last verified: 2026-04-01
Verification context: causal_parent_id decision provenance column added to io_log + text_runtime (Plan #54)
Date: 2026-03-22

## Context

`stream_llm` and `astream_llm` are visible entrypoints, but prior extraction
introduced regressions and missing terminal lifecycle rows:

1. Stream wrappers were not wrapped for lifecycle finalization when consumers
   stopped iterating at natural end or encountered an iterator exception.
2. Non-streaming `llm_client` monitor fields were emitted, while stream paths
   often stayed unobserved, producing blind spots in `get_active_llm_calls`.
3. The first implementation mixed model identifiers in stream adapters, including
   argument-level inconsistencies in async stream construction.
4. Stagnation is currently inference-based (time since last observed progress),
   not provider-progressive heartbeat proof because standard chat-stream providers
   do not emit reliable chunk-level progress metadata.

## Decision

1. Keep stream lifecycle observability in `llm_client`, with explicit terminal
   events emitted by stream adapters at iterator boundary:
   - `started` emitted before stream creation attempt
   - `progress` emitted from chunk iteration (`mark_progress` per chunk)
   - `completed` when `StopIteration` is reached
   - `failed` when iteration raises or stream setup fails
2. Use heartbeat/stall settings consistently across sync and async stream paths:
   - `lifecycle_heartbeat_interval_s`
   - `lifecycle_stall_after_s`
3. Drive monitor state from chunk callbacks only; heartbeat/stall markers remain
   inferred from elapsed time without assuming provider token-level progress.
4. Pass monitor objects only through private `_lifecycle_monitor` kwargs into provider
   payload builders, and never into public stream constructors.
5. Treat stream model constructor arguments as provider iterator + requested model
   (no duplicate positional overloads).

## Consequences

Positive:
1. Live stream calls now appear in `get_active_llm_calls` with a truthful state
   transition from `started` to `completed`/`failed`.
2. Operators can diagnose stuck streams by combining heartbeat and stalled metadata
   without conflating provider-side heuristics with client truth.

Negative:
1. Long-running, non-progressing streams can still only be inferred as stale from
   inactivity (not provably "stuck").
2. Streams that are created and then never iterated remain in-progress until process
   end; this is a broader consumption contract issue and must be handled
   operationally.

## Testing Contract

1. Unit tests in `tests/test_client_lifecycle.py` must verify:
   - sync success emits `started -> progress -> completed`
   - sync iteration error emits `started -> progress -> failed` and captures error metadata
   - async success emits `started -> progress -> completed`
   - async iteration error emits `started -> progress -> failed` and captures error metadata
2. Existing stream fixtures should continue passing for non-streaming and streaming
   behavior after monitor extraction.
