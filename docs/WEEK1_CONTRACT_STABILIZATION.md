# Week-1 Contract Stabilization Plan

Date: 2026-02-22  
Scope: contract freeze and low-risk guardrails only

## Non-Negotiables

1. `LLMCallResult.model` is legacy for week 1 and is characterized per
   entrypoint.
2. `resolved_model` is best-effort and nullable; never guessed.
3. Routing-sensitive tests must set routing policy explicitly.
4. Integration tests are opt-in and excluded from default pytest runs.
5. No large refactors, no routing-default flips, no transport rewrites.

## Week-1 Deliverables

1. ADRs:
   - model identity v0
   - routing/config precedence
   - warning taxonomy
2. Integration test gating (`integration` marker + opt-in gate).
3. Characterization/contract tests for current behavior.
4. Additive identity fields:
   - `requested_model`
   - `resolved_model` (best-effort, nullable)
   - `routing_trace` (minimal structured trace)
5. Drift fixes only:
   - `_require_tags` mismatch
   - warning category expectation mismatch

## Merge Gates

1. Default `pytest` run is offline-safe.
2. No public behavior changes except explicitly approved drift fixes.
3. Routing behavior assertions use explicit policy setup in tests.
