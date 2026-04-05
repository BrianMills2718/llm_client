# ADR 0003: Warning Taxonomy

Status: Accepted
Date: 2026-02-22
Last verified: 2026-04-05
Verification context: exact gpt-5.4 requests now canonicalize through the typed provider-governance policy, while shared 429 cooldown warnings now emit stable `PROVIDER_GOVERNANCE_EVENT[cooldown_registered]` records through the provider-coordination backend without duplicate waits or cooldown busy-spins

## Context

Current warnings include both model deprecation and model advisories
(outclassed-but-allowed). Warning category drift has caused test and contract
mismatch.

## Decision

1. `DeprecationWarning` is reserved for true deprecation/blocking paths.
2. `UserWarning` is used for outclassed-but-allowed advisories.
3. Week 1 locks category semantics; code identifiers can be added later.
4. Week 1 applies only drift fixes needed to align behavior/tests with this
   taxonomy.

## Rationale

Category consistency improves automation and human interpretation.

## Consequences

Positive:
1. Clear operational meaning of warnings.
2. Stable tests and less ambiguity in tool/agent behavior.

Negative:
1. Existing tests expecting different categories must be updated intentionally.

## Follow-up

Add stable warning codes (`LLMC_WARN_*`) with structured metadata once
router/kernel contracts are stabilized.
