# Plan 10: API Reference Generation Pipeline

**Status:** In Progress
**Type:** implementation
**Priority:** High
**Blocked By:** None
**Blocks:** none

---

## Gap

**Current:** `docs/API_REFERENCE.html` was being hand-authored as a one-off browser page, which means the API docs can drift from the typed runtime surface and docstrings.

**Target:** the browser API reference is generated automatically from the live `llm_client` package surface, with docstrings and signatures as the source material.

**Why:** the repo already requires strong typing and docstrings. The browser docs should exploit that structure instead of creating a second manually maintained documentation surface.

---

## References Reviewed

- `docs/API_REFERENCE.md`
- `docs/API_REFERENCE.html`
- `llm_client/__init__.py`
- `llm_client/client.py`
- `llm_client/io_log.py`
- `hooks/pre-commit`
- `scripts/doc_coupling.yaml`
- `docs/plans/CLAUDE.md`

---

## Files Affected

- `scripts/meta/generate_api_reference.py` (create, includes `--check` mode)
- `docs/API_REFERENCE.html` (generated output)
- `docs/API_REFERENCE.md` (generated index)
- `hooks/pre-commit` (add enforcement)
- `scripts/doc_coupling.yaml` (add generated-doc coupling)
- `docs/plans/CLAUDE.md` (modify)
- `tests/test_api_reference_generation.py` (create)

---

## Plan

### Steps

1. Build a generator that walks the `llm_client` package and extracts public module docstrings, functions, classes, methods, constants, and signatures.
2. Emit a browser-openable HTML reference page plus a short markdown index that links to it.
3. Add a generator `--check` mode that fails if the checked-in docs differ from the generator output.
4. Wire the generator check mode into pre-commit and doc-code coupling so stale API docs block commits.
5. Add tests that prove the generator covers multiple modules and that the sync check detects drift.

---

## Required Tests

### New Tests (TDD)

| Test File | Test Function | What It Verifies |
|-----------|---------------|------------------|
| `tests/test_api_reference_generation.py` | `test_generator_emits_package_modules` | The generated docs include multiple discovered public modules, not a hand-curated slice |
| `tests/test_api_reference_generation.py` | `test_generator_includes_signatures_and_docstrings` | The output contains typed signatures and docstring text from real package objects |
| `tests/test_api_reference_generation.py` | `test_sync_check_detects_drift` | The sync checker fails when the generated docs do not match checked-in output |

### Existing Tests (Must Pass)

| Test Pattern | Why |
|--------------|-----|
| `tests/test_*.py` focused on docs, hooks, and plan validation | No regression in repo governance |

---

## Acceptance Criteria

- [ ] The API reference page is generated from package code, not hand-authored.
- [ ] The browser view includes the package surface plus module-by-module docs for the public `llm_client` package.
- [ ] The docs include typed signatures and docstrings from the live code.
- [ ] `docs/API_REFERENCE.md` stays as a generated index, not a separate manually maintained narrative.
- [ ] The sync check fails on drift and passes on regenerated output.
- [ ] Pre-commit blocks commits when the generated API docs are stale.

---

## Notes

Keep the first implementation simple and deterministic. Favor a single generated page with module sections and collapsible detail blocks over a multi-page site until the generator proves useful in practice.
