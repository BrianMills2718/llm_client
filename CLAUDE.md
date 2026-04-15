# LLM Client

---

## ⚡ AUTONOMOUS OPERATION DIRECTIVE

> ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
> ⚠️  NEVER STOP — NO EXCEPTIONS, NO EXCUSES  ⚠️
> ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
>
> Brian has explicitly authorized continuous autonomous operation.
> A completed phase, a green test suite, or a finished commit is
> NOT a stop condition. Update the sprint tracker, commit, move
> immediately to the next phase.

**ONLY TWO LEGITIMATE STOP CONDITIONS:**
1. An action is BOTH irreversible AND affects shared state (force push, drop prod tables)
2. A genuine architectural decision not pre-made in the plan that cannot be safely defaulted

**Everything else — uncertainty, tool failure, partial results — is NOT a stop condition.**
Log it in the sprint tracker and proceed with the safer option.

**Always work in worktrees for Plan-level changes.** Commit at every verified milestone.
**No active sprint.** Check `project-meta/docs/ops/CURRENT_AUTHORITY_STACK.md` for current execution context.

---

Shared LLM infrastructure for Brian's project ecosystem: completions, structured output, tool calling, embeddings, streaming, batch, agent SDK routing, MCP loops, retry/fallback, cost tracking, observability (JSONL + SQLite), model registry, and experiment logging. AGENTS.md is generated from this file via `render_agents_md.py`.

Dead-code review is governed locally through `meta-process.yaml` and `dead_code_audit.json`. Real repo-local findings should be deleted or integrated; retained findings need explicit reviewed dispositions, not silent suppression.

When working in this repo, read these first:

1. [docs/plans/01_master-roadmap.md](docs/plans/01_master-roadmap.md)
2. [docs/plans/CLAUDE.md](docs/plans/CLAUDE.md)
3. [docs/HANDOFF.md](docs/HANDOFF.md)
4. [docs/API_REFERENCE.md](docs/API_REFERENCE.md)
5. [docs/API_REFERENCE.html](docs/API_REFERENCE.html) for the generated browser reference
6. [scripts/meta/generate_api_reference.py](scripts/meta/generate_api_reference.py) to regenerate the docs
7. [OpenClaw success-criteria contract](.openclaw/success-criteria.yaml)

---

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
make dead-code               # Run current dead-code scan
make dead-code-audit         # Refresh reviewed dead-code inventory
make dead-code-validate      # Validate reviewed dead-code dispositions

# API Reference
python scripts/meta/generate_api_reference.py --write  # Regenerate docs

# Plans
python scripts/meta/check_plan_tests.py --plan N       # Run tests for plan
python scripts/meta/complete_plan.py --plan N           # Mark plan complete

# Read-gating
python scripts/meta/check_required_reading.py <file>    # Check read-gate for a source file
python scripts/meta/validate_relationships.py --strict  # Validate coupling config
```

---

## Principles

1. **Runtime substrate, not thin wrapper** -- `llm_client` is a control plane providing routing, observability, retry/fallback, and structured output enforcement. It is not a convenience layer over LiteLLM.
2. **Required kwargs on every call** -- `task=`, `trace_id=`, `max_budget=`. No exceptions.
3. **Prompts as data** -- YAML/Jinja2 templates in `prompts/`, loaded via `render_prompt()`. No f-string prompts in calling code.
4. **Structured output via `json_schema`** -- Always use `json_schema` response_format, never `json_object`. Schema field descriptions are the primary mechanism for correct output.
5. **Observability first** -- All state changes logged with context. Query `~/projects/data/llm_observability.db` for real costs; never estimate.
6. **Fail loud** -- No silent fallbacks, no `except: pass`. Errors surface with context.
7. **Programs A-D complete** -- Do not invent new cleanup slices without fresh evidence. The roadmap tracks what remains.
8. **API reference is generated** -- Run `python scripts/meta/generate_api_reference.py --write` after changing the public surface or docstrings.

---

## Workflow

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

---

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
