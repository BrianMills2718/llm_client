# ADR 0001: Model Identity Contract v0

Status: Accepted  
Date: 2026-02-22

## Context

`LLMCallResult.model` is currently used as a public field by callers, but its
effective meaning is not fully uniform across all entrypoints. In some paths it
reflects an executed/resolved model, while in others it may reflect the input
model for a higher-level loop call.

We need a week-1 contract stabilization step that avoids behavioral breakage.

## Decision

1. `LLMCallResult.model` is treated as a legacy field in week 1.
2. Week 1 does not redefine `model` semantics across all paths.
3. We add additive disambiguation fields:
4. `requested_model`: raw caller model input at API boundary.
5. `resolved_model`: best-effort executed model for terminal successful output.
6. `routing_trace`: structured trace of routing/fallback decisions.
7. `resolved_model` is nullable and must be set only when provable.
8. `resolved_model` must never be guessed.

## Rationale

This prevents accidental compatibility breaks while making behavior explicit and
testable. A wrong resolved value is worse than `None`.

## Consequences

Positive:
1. Stabilizes current behavior for downstream consumers.
2. Enables future unification with explicit migration.
3. Improves diagnostics and auditing.

Negative:
1. Temporary dual semantics (`model` + new fields) adds short-term complexity.
2. Some paths may expose `resolved_model=None` until router/kernel extraction.

## Testing Contract

1. Add characterization tests per entrypoint for current `result.model`.
2. Assert `requested_model` always equals caller input.
3. Assert `resolved_model` is correct when provable, else `None`.

## Migration Notes

After router extraction (`resolve_call -> ResolvedCallPlan`) and shared kernel
work, propose a follow-up ADR to unify or deprecate ambiguous `model` usage.

