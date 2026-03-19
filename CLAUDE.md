# LLM Client

`llm_client` is the shared runtime substrate and control plane for LLM and
coding-agent work across the stack. It wraps commodity transport libraries, but
its value is the contract on top: mandatory call metadata, shared
observability, task-based model selection, prompt provenance, structured
output, MCP loops, and agent-runtime routing.

This file is the canonical repo-governance source. `AGENTS.md` is a generated
projection for Codex-facing instructions and must not become a second,
hand-maintained policy authority.

Programs A-D in the roadmap are complete. Current work should extend the
existing runtime contract, not restart architecture cleanup from scratch.

## Commands

```bash
source .venv/bin/activate
pytest -q tests/test_required_reading_gate.py tests/test_read_gate_hooks.py tests/test_relationships_validation.py
python scripts/meta/validate_relationships.py --strict
python scripts/meta/validate_plan.py --plan-file docs/plans/07_governed_repo_contract_alignment.md
python scripts/meta/sync_plan_status.py --check
python scripts/check_doc_coupling.py --suggest
python scripts/check_markdown_links.py CLAUDE.md docs/plans/CLAUDE.md scripts/CLAUDE.md
python ~/projects/project-meta/scripts/audit_governed_repo.py --repo-root "$PWD" --json
```

## Principles

- Keep `CLAUDE.md` canonical and regenerate `AGENTS.md`; do not maintain two policy files by hand.
- Treat `scripts/relationships.yaml` as the machine-readable governance graph for required reading and doc coupling.
- Preserve the runtime-substrate boundary: project repos should use `llm_client`, not bypass it with direct provider calls.
- Keep the required call contract explicit: project code must pass `task=`, `trace_id=`, and `max_budget=`.
- Fail loud on governance drift or missing hook dependencies; the gate should not silently allow unsafe edits.
- Add observability when changing control-plane behavior so operator review can reconstruct what happened.

## Workflow

1. Read `CLAUDE.md` and the files surfaced by `scripts/meta/file_context.py` before editing governed runtime files.
2. Create or update a plan in `docs/plans/` for non-trivial work and keep the plan index aligned.
3. Keep `scripts/relationships.yaml` truthful when runtime or ADR coupling changes.
4. Regenerate `AGENTS.md` instead of editing it manually.
5. Run relationship validation, plan validation, plan-status sync, doc-coupling, and markdown-link checks before closing a slice.
6. Preserve the custom read gate unless there is clear evidence it is weaker than the canonical generated alternative.

## References

- `README.md` - repo purpose, public API framing, and routing guidance
- `docs/plans/01_master-roadmap.md` - strategic roadmap and completed programs
- `docs/plans/CLAUDE.md` - plan index and current plan status
- `docs/API_REFERENCE.md` - public API contract and result-shape expectations
- `scripts/relationships.yaml` - required-reading defaults, ADR coupling, and doc-code linkage
- `scripts/check_required_reading.py` - required-reading wrapper used by the Claude edit gate
- `scripts/meta/file_context.py` - deterministic file-to-context resolver
- `scripts/meta/validate_plan.py` - plan gate against the documentation graph
- `scripts/meta/check_doc_coupling.py` - doc-coupling enforcement against `scripts/relationships.yaml`
- `scripts/meta/validate_relationships.py` - integrity checker for the governance graph
