# Plan 07: Call Liveness and Active-Call Visibility

**Status:** Complete
**Type:** implementation slice
**Priority:** High
**Program:** E - Simplification and Observability Modernization

---

## Purpose

Make long-running `llm_client` calls inspectable without forcing downstream
projects to guess from elapsed wall-clock time or add ad hoc watchdog logic.

## Acceptance Criteria

This slice is complete when:

1. public wrapper calls emit lifecycle events that distinguish started,
   waiting/heartbeat, real progress, stalled/idled progress-aware paths, and
   terminal completion/failure;
2. `get_active_llm_calls()` returns a truthful current-state view for
   cross-project inspection;
3. progress-aware transports surface `progressing` while opaque transports stay
   `waiting`;
4. same-host orphaned processes are excluded from the active-call view;
5. the capability is visible from the README and roadmap rather than living
   only in code.

## Implemented Shape

The implemented surface is:

1. lifecycle emission from the public call wrappers in `llm_client/client.py`;
2. foundation-backed lifecycle event payloads in `llm_client/foundation.py`;
3. active-call query and activity-state derivation in
   `llm_client/observability/query.py`;
4. compatibility export via `llm_client/io_log.py` and package-level public
   re-export;
5. same-host orphan filtering in the active-call query;
6. truthful distinction between:
   - `activity_state=\"progressing\"` for progress-observable calls
   - `activity_state=\"waiting\"` for opaque non-streaming calls
   - `activity_state=\"idle\"` for progress-aware paths that emitted a
     non-terminal stalled marker

## Evidence

Primary proof:

1. `tests/test_client_lifecycle.py`
2. `tests/test_io_log.py`
3. `tests/test_public_surface.py`

Operator-facing entry point:

1. `README.md` active-call section

## Notes

This plan is intentionally about visibility, not aggressive intervention.
Opaque non-streaming calls still cannot prove real provider-side progress. The
goal is to make that limit visible rather than to hide it behind optimistic
timeout assumptions.
