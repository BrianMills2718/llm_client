# ADR 0013: Provider Timeouts Are Not the Default Liveness Mechanism

Status: Accepted
Date: 2026-03-19

Related: Extended by ADR 0014 (heartbeats/stall markers) and ADR 0016
(progress-aware idle detection). See also ADR 0009 (background polling for
long-thinking models).

## Context

`llm_client` currently exposes provider-facing `timeout` parameters across text,
structured, streaming, embedding, and agent runtimes. That has value as a
compatibility control, but it is not the same thing as liveness management.

For non-streaming LLM calls, especially structured extraction calls, an elapsed
wall-clock timeout cannot distinguish:

1. a healthy slow call that is still progressing,
2. an opaque provider call that may still complete successfully,
3. a genuinely stalled call.

The current substrate also lacks a generic call-lifecycle signal for normal
text and structured calls. Operators can see finished call records, but not a
clear `started -> completed/failed` sequence for in-flight non-streaming work.

We need a clearer contract:

1. provider request timeout is a transport compatibility knob,
2. call liveness is an orchestration and observability concern,
3. long-running calls must be visible before they terminate.

## Decision

1. Provider request timeouts are not the default liveness mechanism for normal
   `llm_client` call flows.
2. The substrate should prefer visibility and explicit orchestration over
   killing a call merely because it is slow.
3. `LLM_CLIENT_TIMEOUT_POLICY` remains the shared policy switch for whether
   provider timeout arguments are honored or disabled, but that policy is now
   interpreted as a transport boundary rather than the primary liveness model.
4. As the first implementation slice, `llm_client` will emit Foundation-backed
   lifecycle events for public non-streaming text and structured calls:
   - `started`
   - `completed`
   - `failed`
5. Lifecycle events must carry enough metadata to answer:
   - which public call is still in flight,
   - which task/trace/run it belongs to,
   - which model was requested/resolved,
   - which provider-timeout policy was in effect.
6. This ADR does not yet introduce full progress-aware idle timeout handling.
   That remains a later orchestration slice once stable lifecycle observability
   exists.

## Consequences

Positive:
1. Operators can tell whether a normal non-streaming call is still running
   without relying on a timeout exception.
2. Slow calls and genuinely failed calls become distinguishable in a shared,
   queryable event stream.
3. Future orchestration work can build on an explicit lifecycle contract
   instead of implicit timeout behavior.

Negative:
1. Public call wrappers now emit more observability records.
2. Provider timeouts still exist for compatibility, so the first slice does not
   fully eliminate timeout-related behavior differences.
3. Streaming, embeddings, and some long-polling/background paths are out of
   scope for this first implementation slice and will need follow-up work.

## Testing Contract

1. Foundation validation must accept the new lifecycle event shape and reject
   malformed lifecycle payloads.
2. Public `call_llm` / `acall_llm` must emit `started` and terminal lifecycle
   events around normal text calls.
3. Public `call_llm_structured` / `acall_llm_structured` must emit `started`
   and terminal lifecycle events around normal structured calls.
4. Failed calls must emit `failed` lifecycle events with explicit error type and
   message instead of disappearing into logs without a terminal record.
