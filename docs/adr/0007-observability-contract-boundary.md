# ADR 0007: Observability Contract Boundary

Status: Accepted  
Date: 2026-02-23

## Context

Observability logic was historically mixed into core call paths, and `io_log.py`
served both as implementation and public API surface. As the codebase split into
`llm_client/observability/*` modules, we needed a stable contract for what is
persisted, how compatibility is preserved, and where behavior should evolve.

## Decision

1. Canonical observability implementation lives in `llm_client/observability/*`.
2. `llm_client/io_log.py` remains a compatibility facade for existing imports.
3. Default persistence behavior remains safe-by-default:
   - metadata-first logging,
   - no requirement to persist full raw content in default paths.
4. Warning and routing-related diagnostics emitted into observability surfaces
   must remain aligned with the warning taxonomy contract in ADR 0003.
5. Any breaking changes to observability payload shape or sink behavior require
   a dedicated ADR update.

## Consequences

Positive:
1. Clear boundary between core execution and observability concerns.
2. Preserved compatibility for existing `io_log` consumers.
3. Better maintainability for query/experiment/reporting evolution.

Negative:
1. Transitional complexity while both compatibility facade and canonical modules exist.
2. Requires discipline to keep facade behavior aligned with canonical modules.

## Testing Contract

1. Compatibility tests must cover `io_log` delegated behavior.
2. Observability tests must verify default-safe persistence behavior.
3. Warning/diagnostic emission must remain category-consistent with ADR 0003.
