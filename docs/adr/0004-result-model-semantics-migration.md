# ADR 0004: Fixed `result.model` Semantics

Status: Accepted  
Date: 2026-02-23

## Context

`LLMCallResult.model` had compatibility modes and migration toggles. This made
debugging harder and forced callers to know mode-specific behavior.

## Decision

1. Remove semantics-mode switching.
2. Remove semantics telemetry and related CLI reporting commands.
3. Use one identity contract everywhere:
   - `result.model`: terminal executed model.
   - `result.requested_model`: caller input model.
   - `result.resolved_model` / `result.execution_model`: terminal executed model.
   - `result.routing_trace`: routing/fallback explanation.

## Consequences

Positive:
1. No ambiguity in `result.model`.
2. No mode/env drift between environments.
3. Simpler client API and docs.

Negative:
1. Breaking change for clients that relied on legacy/model-mode behavior.
2. Removed mode-adoption telemetry and semantics report commands.

## Rollout

1. Version cut to `0.7.0`.
2. Keep additive identity fields and routing trace as the canonical debugging
   surface.

## Testing Contract

1. Identity tests assert `result.model == result.resolved_model` when resolved
   identity is known.
2. MCP/agent tests assert fallback cases still preserve:
   - `requested_model` as caller input.
   - `routing_trace` attempted model chain.
