# LLM Client

<!-- GENERATED FILE: DO NOT EDIT DIRECTLY -->
<!-- generated_by: scripts/meta/render_agents_md.py -->
<!-- canonical_claude: CLAUDE.md -->
<!-- canonical_relationships: scripts/relationships.yaml -->
<!-- canonical_relationships_sha256: 5667b5f5fd75 -->
<!-- sync_check: python scripts/meta/check_agents_sync.py --check -->

This file is a generated Codex-oriented projection of repo governance.
Edit the canonical sources instead of editing this file directly.

Canonical governance sources:
- `CLAUDE.md` — human-readable project rules, workflow, and references
- `scripts/relationships.yaml` — machine-readable ADR, coupling, and required-reading graph

## Purpose

LLM Client uses `CLAUDE.md` as canonical repo governance and workflow policy.

## Commands

- `make lint` — run the project linters.
- `pytest` — run test suite.
- `python -m llm_client experiments --help` — inspect experiment CLI entry points.
- `python scripts/audit_llm_client.py` — project-local smoke checks, if present.

## Operating Rules

This projection keeps the highest-signal rules in always-on Codex context.
For full project structure, detailed terminology, and any rule omitted here,
read `CLAUDE.md` directly.

### Principles

- `llm_client` is runtime substrate and control plane, not a thin wrapper.
- Keep the API surface stable and avoid speculative abstractions.
- Use strict input/output contracts, especially around `task`, `trace_id`, and
  `max_budget`.
- Fail loudly on configuration mismatches and observability gaps.
- Keep telemetry and experiments as the default mechanism for model and prompt
  comparison.

### Workflow

- Primary references when entering this repo:
  1. [AGENTS.md](AGENTS.md)
  2. [README.md](README.md)
  3. [docs/plans/01_master-roadmap.md](docs/plans/01_master-roadmap.md)
  4. [docs/plans/CLAUDE.md](docs/plans/CLAUDE.md)
  5. [docs/API_REFERENCE.md](docs/API_REFERENCE.md)
- Treat roadmap and plan files as constraints for sequence and scope.
- Avoid ad-hoc changes to completed Program A-D without measurable evidence.

## Machine-Readable Governance

`scripts/relationships.yaml` is the source of truth for machine-readable governance in this repo: ADR coupling, required-reading edges, and doc-code linkage. This generated file does not inline that graph; it records the canonical path and sync marker, then points operators and validators back to the source graph. Prefer deterministic validators over prompt-only memory when those scripts are available.

## References

- [AGENTS.md](AGENTS.md) (generated mirror of this file)
- [meta-process.yaml](meta-process.yaml)
- [docs/CHANGELOG.md](CHANGELOG.md)
