# ADR 0006: Foundation `actor_id` Issuance Policy

Status: Accepted  
Date: 2026-02-23

## Context

Foundation events require `actor_id`, but issuance semantics were not formally
documented. This created ambiguity around trust boundaries and naming
consistency for decision/transition records.

## Decision

1. `actor_id` is required on all Foundation events and must identify the
   principal responsible for the recorded action.
2. Namespace prefixes are restricted to:
   - `user:`
   - `agent:`
   - `service:`
3. Canonical shape:
   - `<prefix><component>[:<scope>[:<version>]]`
   - example: `agent:mcp_loop:default:1`
4. Trust boundary:
   - server/runtime issuance is authoritative,
   - untrusted external values must not be passed through unchanged for
     privileged event classes.
5. Decision and transition-style events must always include a server-issued
   `actor_id` in one of the canonical namespaces.

## Consequences

Positive:
1. Clear accountability provenance in Foundation logs.
2. Consistent principal identity for policy analysis and auditing.
3. Reduced spoofing risk from pass-through actor labels.

Negative:
1. Additional implementation discipline is required when adding new emitters.
2. Legacy free-form actor identifiers should be migrated to canonical form.

## Testing Contract

1. Foundation event tests should assert non-empty canonical `actor_id` values.
2. New emitters must include coverage that verifies canonical namespace usage.
