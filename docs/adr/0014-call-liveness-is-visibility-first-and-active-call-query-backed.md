# 0014 - Call liveness is visibility-first and active-call-query-backed

## Status

Accepted.

## Context

`llm_client` runs long-lived calls across multiple transport shapes:

1. opaque non-streaming text and structured calls;
2. streaming calls with real chunk progress;
3. background-polled Responses calls with explicit poll events.

Users and downstream repos need a truthful answer to questions like:

1. is this call still alive;
2. is it merely waiting or actually progressing;
3. did the local process die;
4. is the system inferring a hang from wall-clock time alone.

The repo already implements lifecycle events and an active-call query, but the
governing decision was not clearly indexed in the docs. That made the feature
discoverable in the README but weakly connected to the architectural record.

## Decision

`llm_client` treats call liveness as a **visibility-first** concern backed by
lifecycle events and the active-call query.

Concretely:

1. public wrapper calls emit lifecycle events with non-terminal phases such as
   `started`, `heartbeat`, `progress`, and `stalled`, followed by terminal
   `completed` or `failed`;
2. `get_active_llm_calls()` is the supported inspection surface for
   cross-project liveness checks;
3. progress-aware states are only claimed when the transport actually exposes
   real progress signals;
4. opaque non-streaming structured calls stay truthfully in a waiting-style
   state instead of being mislabeled as progressing;
5. same-host orphaned processes are excluded from the active-call view when
   local process liveness proves they are dead.

This decision complements the timeout policy decisions:

1. provider timeouts are not the primary liveness mechanism;
2. users should inspect active-call state before assuming a hang;
3. visibility comes first, intervention second.

## Consequences

Positive:

1. downstream repos can tell “still alive” from “definitely dead” without
   guessing from elapsed time alone;
2. opaque transports stay honest about what they can and cannot prove;
3. liveness behavior is shared across consumers instead of reimplemented
   project-by-project.

Costs:

1. lifecycle semantics are more explicit and therefore need documentation and
   tests;
2. non-streaming opaque paths still cannot prove true provider-side progress;
3. the active-call view depends on observability storage being available and
   queryable.

## Related Files

Primary implementation:

1. `llm_client/client.py`
2. `llm_client/observability/query.py`
3. `llm_client/io_log.py`
4. `llm_client/foundation.py`

Primary proof:

1. `tests/test_client_lifecycle.py`
2. `tests/test_io_log.py`
3. `tests/test_public_surface.py`
