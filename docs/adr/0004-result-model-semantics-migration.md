# ADR 0004: `result.model` Semantics Migration Window

Status: Accepted  
Date: 2026-02-22

## Context

`LLMCallResult.model` is legacy and historically path-dependent. Week-1
introduced additive identity fields (`requested_model`, `resolved_model`,
`routing_trace`) and characterization tests. We now need a controlled migration
path for consumers that want deterministic model semantics.

## Decision

1. Keep default semantics as `legacy` for compatibility.
2. Add typed config control:
   - `ClientConfig.result_model_semantics="legacy"` (default)
   - `ClientConfig.result_model_semantics="requested"`
   - `ClientConfig.result_model_semantics="resolved"`
3. Add env compatibility flag:
   - `LLM_CLIENT_RESULT_MODEL_SEMANTICS=legacy|requested|resolved`
4. Keep additive alias `execution_model` on `LLMCallResult` as a stable
   operational identity alias for `resolved_model`.

## Rationale

This allows explicit migration for clients without a global breaking flip. The
public contract can evolve behind explicit opt-in while preserving existing
behavior for current callers.

## Consequences

Positive:
1. Explicit per-call semantic control via typed config.
2. Backward compatibility by default.
3. Cleaner long-term path to deprecate ambiguous `model`.

Negative:
1. Temporary dual-surface complexity (`model` plus identity fields).
2. Callers must opt in to deterministic `model` semantics.

## Migration Plan

1. `0.6.x`: default remains `legacy`; document opt-in semantics.
2. Next minor: gather adoption telemetry and compatibility feedback.
3. Future major/minor decision: consider changing default only with
   compatibility flag retained for one release cycle.

## Testing Contract

1. Characterization tests for legacy behavior remain.
2. Add explicit tests for `requested` and `resolved` semantics via
   `ClientConfig`.
