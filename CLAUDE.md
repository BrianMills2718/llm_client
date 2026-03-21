# LLM Client

## Principles

- `llm_client` is runtime substrate and control plane, not a thin wrapper.
- Keep the API surface stable and avoid speculative abstractions.
- Use strict input/output contracts, especially around `task`, `trace_id`, and
  `max_budget`.
- Fail loudly on configuration mismatches and observability gaps.
- Keep telemetry and experiments as the default mechanism for model and prompt
  comparison.

## Commands

- `make lint` — run the project linters.
- `pytest` — run test suite.
- `python -m llm_client experiments --help` — inspect experiment CLI entry points.
- `python scripts/audit_llm_client.py` — project-local smoke checks, if present.

## Workflow

- Primary references when entering this repo:
  1. [AGENTS.md](AGENTS.md)
  2. [README.md](README.md)
  3. [docs/plans/01_master-roadmap.md](docs/plans/01_master-roadmap.md)
  4. [docs/plans/CLAUDE.md](docs/plans/CLAUDE.md)
  5. [docs/API_REFERENCE.md](docs/API_REFERENCE.md)
- Treat roadmap and plan files as constraints for sequence and scope.
- Avoid ad-hoc changes to completed Program A-D without measurable evidence.

## References

- [AGENTS.md](AGENTS.md) (generated mirror of this file)
- [meta-process.yaml](meta-process.yaml)
- [docs/CHANGELOG.md](CHANGELOG.md)
