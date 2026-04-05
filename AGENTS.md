# LLM Client

<!-- GENERATED FILE: DO NOT EDIT DIRECTLY -->
<!-- generated_by: scripts/meta/render_agents_md.py -->
<!-- canonical_claude: CLAUDE.md -->
<!-- canonical_relationships: scripts/relationships.yaml -->
<!-- canonical_relationships_sha256: 2377f914f439 -->
<!-- sync_check: python scripts/meta/check_agents_sync.py --check -->

This file is a generated Codex-oriented projection of repo governance.
Edit the canonical sources instead of editing this file directly.

Canonical governance sources:
- `CLAUDE.md` — human-readable project rules, workflow, and references
- `scripts/relationships.yaml` — machine-readable ADR, coupling, and required-reading graph

## Purpose

LLM Client uses `CLAUDE.md` as canonical repo governance and workflow policy.

## Commands

```bash
# Tests
make test                    # Full test suite
make test-quick              # Tests, minimal output
make check                   # Tests + type check + lint

# Observability
make cost                    # Total LLM spend (DAYS=7)
make cost-by-task            # Spend per task
make cost-by-model           # Spend per model
make errors                  # Error breakdown by model
make summary                 # Project stats

# Development
make status                  # git status --short --branch
make help                    # List all targets

# API Reference
python scripts/meta/generate_api_reference.py --write  # Regenerate docs

# Plans
python scripts/meta/check_plan_tests.py --plan N       # Run tests for plan
python scripts/meta/complete_plan.py --plan N           # Mark plan complete

# Read-gating
python scripts/meta/check_required_reading.py <file>    # Check read-gate for a source file
python scripts/meta/validate_relationships.py --strict  # Validate coupling config
```

## Operating Rules

This projection keeps the highest-signal rules in always-on Codex context.
For full project structure, detailed terminology, and any rule omitted here,
read `CLAUDE.md` directly.

### Principles

1. **Runtime substrate, not thin wrapper** -- `llm_client` is a control plane providing routing, observability, retry/fallback, and structured output enforcement. It is not a convenience layer over LiteLLM.
2. **Required kwargs on every call** -- `task=`, `trace_id=`, `max_budget=`. No exceptions.
3. **Prompts as data** -- YAML/Jinja2 templates in `prompts/`, loaded via `render_prompt()`. No f-string prompts in calling code.
4. **Structured output via `json_schema`** -- Always use `json_schema` response_format, never `json_object`. Schema field descriptions are the primary mechanism for correct output.
5. **Observability first** -- All state changes logged with context. Query `~/projects/data/llm_observability.db` for real costs; never estimate.
6. **Fail loud** -- No silent fallbacks, no `except: pass`. Errors surface with context.
7. **Programs A-D complete** -- Do not invent new cleanup slices without fresh evidence. The roadmap tracks what remains.
8. **API reference is generated** -- Run `python scripts/meta/generate_api_reference.py --write` after changing the public surface or docstrings.

### Workflow

### Process Awareness
- All significant work follows meta-process plans in `docs/plans/`.
- Use `[Trivial]` only for <20 lines with no production code changes.
- Plan index: `docs/plans/CLAUDE.md`. Template: `docs/plans/TEMPLATE.md`.

### Read-Gating
- Source files coupled to ADRs in `scripts/relationships.yaml` require reading the linked docs before editing.
- `.claude/hooks/gate-edit.sh` enforces this at edit time.
- Override temporarily: `LLM_CLIENT_READ_GATE_MODE=warn`.

### Testing
- `make test` runs the full suite. `make check` adds type checking and lint.
- Plan-specific tests: `python scripts/meta/check_plan_tests.py --plan N`.

## Machine-Readable Governance

`scripts/relationships.yaml` is the source of truth for machine-readable governance in this repo: ADR coupling, required-reading edges, and doc-code linkage. This generated file does not inline that graph; it records the canonical path and sync marker, then points operators and validators back to the source graph. Prefer deterministic validators over prompt-only memory when those scripts are available.

## References

| Doc | Purpose |
|-----|---------|
| `docs/plans/01_master-roadmap.md` | Master roadmap (Programs A-E) |
| `docs/plans/CLAUDE.md` | Plan index |
| `docs/API_REFERENCE.md` | Generated API reference |
| `docs/adr/README.md` | Architecture decision records index |
| `scripts/relationships.yaml` | Source-doc coupling and read-gate config |
| `scripts/CLAUDE.md` | Scripts directory reference |
| `.openclaw/success-criteria.yaml` | OpenClaw success criteria contract |
