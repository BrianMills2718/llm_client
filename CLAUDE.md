# LLM Client

This is the canonical instruction file. AGENTS.md is symlinked to this file for Codex compatibility.

When working in this repo, read these first:

1. [docs/plans/01_master-roadmap.md](docs/plans/01_master-roadmap.md)
2. [docs/plans/CLAUDE.md](docs/plans/CLAUDE.md)
3. [docs/API_REFERENCE.md](docs/API_REFERENCE.md)
4. [docs/API_REFERENCE.html](docs/API_REFERENCE.html) for the generated browser reference
5. [scripts/meta/generate_api_reference.py](scripts/meta/generate_api_reference.py) to regenerate the docs
6. [OpenClaw success-criteria contract](.openclaw/success-criteria.yaml)

Short version:

- `llm_client` is a runtime substrate/control plane, not a thin wrapper.
- Programs A-D in the roadmap are complete; do not invent new cleanup slices
  without fresh evidence.
- Required call kwargs remain `task=`, `trace_id=`, and `max_budget=`.
- Optional runtime/eval modules are stable but should not grow by default.
- The browser API reference is generated, not hand-edited. Run
  `python scripts/meta/generate_api_reference.py --write` after changing the
  public package surface or docstrings/signatures.
