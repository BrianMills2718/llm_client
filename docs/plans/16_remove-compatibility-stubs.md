# Plan #16: Remove Compatibility Stubs

**Status:** Planned
**Type:** implementation
**Priority:** Medium
**Blocked By:** 12 (module reorganization — complete)
**Blocks:** None

---

## Gap

**Current:** Plan 12 moved modules into subdirectories (core/, execution/,
agent/, sdk/, tools/, utils/) but left ~52 sys.modules alias files at the
root. These 4-line stubs make `from llm_client.client import X` still work
by aliasing `llm_client.client` → `llm_client.core.client`. They're correct
but they're tech debt — the canonical paths are the subdirectory paths.

**Target:** All internal imports use canonical paths (e.g.,
`from llm_client.core.client import X`). Stubs deleted. External consumers
use `from llm_client import X` (the public API via `__init__.py`), which is
unaffected.

**Why:** Per CLAUDE.md: "Delete > Comment — Remove unused code." The stubs
are a temporary compatibility layer. Once internal imports are migrated, they
serve no purpose.

---

## References Reviewed

- `llm_client/*.py` (root-level stubs) — each is 4 lines: import + sys.modules alias
- `llm_client/__init__.py` — public re-exports (not affected)
- All `from llm_client.X import Y` in llm_client/ and tests/

---

## Pre-made Decisions

1. **Incremental:** Remove stubs one-at-a-time after confirming zero importers.
   Don't batch — each stub removal is independently committable.
2. **Check internal + test imports only:** The public API (`from llm_client import X`)
   is via `__init__.py` and never touches the stubs.
3. **3 downstream projects using private paths will break:** active-stack-core,
   Digimon, sam_gov already import from internal paths. They're already broken
   by design (Plan #17 Phase 3). Don't let them block stub removal.
4. **Two already deleted:** `client_dispatch.py` and `model_detection.py`
   were removed in the code smell pass.

---

## Files Affected

- ~52 root-level stub files (delete, one at a time)
- Internal imports that reference old paths (update to canonical)
- `tests/*.py` — update mock patch targets

---

## Plan

### For each stub file:
1. `grep -rn "from llm_client.MODULE_NAME" llm_client/ tests/ --include="*.py" | grep -v core/ | grep -v execution/ | grep -v agent/ | grep -v sdk/ | grep -v tools/ | grep -v utils/`
2. If zero matches (excluding the stub itself): delete the stub
3. If matches found: rewrite imports to canonical path, then delete stub
4. Run `python -c "import llm_client"` + key tests
5. Commit

### Priority order:
Start with stubs that have zero importers (free wins), then work through
stubs with few importers (1-3 sites), leaving heavily-imported stubs last.

---

## Required Tests

### Existing Tests (Must Pass After Each Stub Removal)

| Test Pattern | Why |
|--------------|-----|
| `tests/test_public_surface.py` | Public API unchanged |
| `tests/test_client.py` | Core dispatch works |

---

## Acceptance Criteria

- [ ] All root-level stub files deleted
- [ ] All internal imports use canonical subdirectory paths
- [ ] `from llm_client import X` still works for all 101 exports
- [ ] All tests pass
- [ ] No stubs remain in `llm_client/*.py` (only `__init__.py`, `__main__.py`,
      and the ~10 modules marked "STAYS AT ROOT" in Plan 12)
