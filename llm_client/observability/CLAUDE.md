# Observability

This subtree contains the concrete logging, run, and query adapters that back
`llm_client` observability.

## Purpose

Keep the observability layer explicit and inspectable. The code here is the
boundary between the compatibility shim in `io_log` and the newer run/query
surfaces that other modules should use.

## What Lives Here

- `events.py` for configuration, enforcement, and event/logging adapters
- `experiments.py` for run lifecycle and item logging
- `comparison.py` for run comparison and cohort analysis
- `query.py` for read-only lookup and reporting helpers
- `__init__.py` for the public observability facade

## Local Rules

1. Preserve fail-loud behavior. Silent degradation in logging/query code hides
   the state that agents need to inspect.
2. Keep compatibility wrappers thin when they exist.
3. Keep query helpers read-only; do not move mutation logic here unless it is
   part of the observability contract.
4. Update docs or plans when observability semantics change, especially when
   fields are added to the logged run/item payloads.
