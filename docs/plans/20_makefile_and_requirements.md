# Plan #20: Makefile Test Targets and REQUIREMENTS.md

**Status:** Complete
**Type:** maintenance
**Priority:** Medium
**Blocked By:** None

---

## Gap

Makefile has observability targets only (cost, errors, traces) but no `make test`,
`make lint`, `make typecheck`. No REQUIREMENTS.md documenting the 16-function API contract.

## Steps

1. Add `test`, `test-verbose`, `lint`, `typecheck` targets to Makefile
2. Write REQUIREMENTS.md documenting the core API contract (16 functions, required kwargs, model registry, observability)
3. Push

## Acceptance Criteria

- [x] `make test` runs pytest
- [x] `make lint` runs ruff
- [x] `make typecheck` runs mypy
- [x] REQUIREMENTS.md exists with API contract
