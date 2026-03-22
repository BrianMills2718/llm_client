# Plan 08: llm_client Subtree Instruction Rollout

**Status:** Planned
**Type:** implementation
**Priority:** Medium
**Blocked By:** None
**Blocks:** none

---

## Gap

**Current:** `llm_client` has strong repo-root instructions, but the package
subtrees that carry the actual runtime surface do not yet have local
`CLAUDE.md` / `AGENTS.md` files. Agents working inside `llm_client/llm_client/`
still have to infer local boundaries from root docs and module names.

**Target:** meaningful package subtrees have local, delta-only instruction
files that route agents to the right local surface without repeating parent
policy.

**Why:** the repo already has distinct surfaces for CLI commands, observability
logic, prompt assets, and rubric data. Those surfaces should be loaded
automatically and stay specific to their subtrees.

---

## References Reviewed

- `CLAUDE.md`
- `AGENTS.md`
- `docs/API_REFERENCE.md`
- `docs/ECOSYSTEM_TOP_DOWN_ARCHITECTURE.md`
- `scripts/CLAUDE.md`
- `tests/CLAUDE.md`
- `docs/meta-patterns/02_claude-md-authoring.md`

---

## Files Affected

- `llm_client/CLAUDE.md` (create)
- `llm_client/AGENTS.md` (create symlink mirror)
- `llm_client/cli/CLAUDE.md` (create)
- `llm_client/cli/AGENTS.md` (create symlink mirror)
- `llm_client/observability/CLAUDE.md` (create)
- `llm_client/observability/AGENTS.md` (create symlink mirror)
- `llm_client/prompts/CLAUDE.md` (create)
- `llm_client/prompts/AGENTS.md` (create symlink mirror)
- `llm_client/rubrics/CLAUDE.md` (create)
- `llm_client/rubrics/AGENTS.md` (create symlink mirror)
- `docs/plans/CLAUDE.md` (modify)
- `tests/test_llm_client_subtree_instructions.py` (create)

---

## Plan

### Steps

1. Write leaf `CLAUDE.md` files for the meaningful package subdirs.
2. Add `AGENTS.md` symlink mirrors for each new subtree file.
3. Add a package-root `llm_client/CLAUDE.md` router that points to the leaf
   surfaces.
4. Add tests that verify the package subtree docs and mirrors exist and stay
   aligned.
5. Update the plan index.

---

## Required Tests

### New Tests (TDD)

| Test File | Test Function | What It Verifies |
|-----------|---------------|------------------|
| `tests/test_llm_client_subtree_instructions.py` | `test_package_subtree_docs_exist` | Leaf docs exist for the selected package subdirs |
| `tests/test_llm_client_subtree_instructions.py` | `test_agents_mirror_claude_symlinks` | `AGENTS.md` mirrors are symlinks to `CLAUDE.md` |
| `tests/test_llm_client_subtree_instructions.py` | `test_package_root_routes_to_local_surfaces` | Package-root instructions route to the leaf surfaces |

### Existing Tests (Must Pass)

| Test Pattern | Why |
|--------------|-----|
| `tests/test_*.py` focused on docs, hooks, and plan validation | No regressions in repo governance |

---

## Acceptance Criteria

- [ ] The new package and subtree instruction files exist.
- [ ] Each new `AGENTS.md` is a symlink mirror of its `CLAUDE.md`.
- [ ] The package-root `CLAUDE.md` routes agents to the local leaf surfaces.
- [ ] The subtree docs stay delta-only and do not repeat parent policy.
- [ ] Tests pass and markdown links remain valid.

---

## Notes

Start from the deepest meaningful subdirs and move upward. Keep the leaf docs
specific to their own local responsibilities:

- `cli/` for command entrypoints and read-only inspection commands
- `observability/` for event/run/query adapters
- `prompts/` for prompt assets
- `rubrics/` for rubric YAML data
