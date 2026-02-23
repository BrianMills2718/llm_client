# ADR 0005: `reason_code` Registry Governance

Status: Accepted  
Date: 2026-02-23

## Context

MCP submit-validation paths emit `reason_code` values that are counted in run
metadata and used for diagnostics (`submit_validation_reason_counts`).
Without registry governance, code names can drift, get reused with different
meaning, or fragment into one-off strings that break trend analysis.

## Decision

1. `reason_code` values are governed by an explicit registry in this ADR.
2. Registry changes are additive-only:
   - new codes may be added,
   - existing codes must not be removed or reassigned to new semantics.
3. Code format is lowercase `snake_case`.
4. Unknown/unregistered codes are still accepted at runtime but treated as
   unregistered telemetry until promoted through an ADR update.
5. Initial registry (version `2026-02-23`):
   - `unfinished_todos`: submit blocked because required TODO items are not complete.
   - `answer_not_grounded`: submit blocked because answer lacks required evidence grounding.

## Consequences

Positive:
1. Stable long-term metrics and reporting across releases.
2. Fewer ambiguous failure reasons in agent policy analysis.
3. Clear change-control path for adding new validation reasons.

Negative:
1. Slight process overhead for introducing new reason codes.
2. Temporary unregistered-code noise may appear before governance catches up.

## Testing Contract

1. Contract tests should assert known reason codes remain unchanged.
2. New reason codes require:
   - ADR update,
   - targeted test coverage,
   - changelog/release note entry.
