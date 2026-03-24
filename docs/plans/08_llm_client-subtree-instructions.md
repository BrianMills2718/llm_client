# Plan 08: llm_client Subtree Instruction Rollout

**Status:** Planned
**Type:** implementation
**Priority:** Medium
**Blocked By:** None
**Blocks:** None

---

## Gap

**Current:** After Plan 12 reorganization, the package has 6 new subdirectories
(core/, execution/, agent/, sdk/, tools/, utils/) plus existing ones (cli/,
observability/, prompts/). The package-root `llm_client/CLAUDE.md` was updated
with a surfaces table, but the new subdirectories lack local CLAUDE.md files.

**Target:** Each meaningful package subdirectory has a delta-only CLAUDE.md
that routes agents to the right local surface. AGENTS.md symlink mirrors
for each.

**Why:** Agents working inside `llm_client/execution/` should get
execution-specific context loaded automatically without reading the full
package-root doc.

---

## References Reviewed

- `llm_client/CLAUDE.md` — updated surfaces table (Plan 12)
- `llm_client/cli/CLAUDE.md` — existing example of leaf instruction file
- `llm_client/observability/CLAUDE.md` — existing example
- `docs/meta-patterns/02_claude-md-authoring.md` — authoring guidelines

---

## Files Affected

- `llm_client/core/CLAUDE.md` (create)
- `llm_client/core/AGENTS.md` (create symlink)
- `llm_client/execution/CLAUDE.md` (create)
- `llm_client/execution/AGENTS.md` (create symlink)
- `llm_client/agent/CLAUDE.md` (create)
- `llm_client/agent/AGENTS.md` (create symlink)
- `llm_client/sdk/CLAUDE.md` (create)
- `llm_client/sdk/AGENTS.md` (create symlink)
- `llm_client/tools/CLAUDE.md` (create)
- `llm_client/tools/AGENTS.md` (create symlink)
- `llm_client/utils/CLAUDE.md` (create)
- `llm_client/utils/AGENTS.md` (create symlink)

---

## Pre-made Decisions

1. **Delta-only:** Each leaf file describes ONLY what's specific to that
   subdirectory. Do not repeat parent policy.
2. **Short:** 10-20 lines max. A surfaces table + 2-3 working rules.
3. **AGENTS.md is symlink:** `ln -s CLAUDE.md AGENTS.md` per convention.
4. **cli/ and observability/ already done:** Don't recreate, just verify.

---

## Plan

### For each new subdirectory (core, execution, agent, sdk, tools, utils):
1. Write CLAUDE.md with: purpose, key modules, working rules
2. Create AGENTS.md symlink
3. Commit

---

## Acceptance Criteria

- [ ] 6 new CLAUDE.md files (one per new subdirectory)
- [ ] 6 new AGENTS.md symlinks
- [ ] Each file is delta-only, <20 lines
- [ ] Existing cli/ and observability/ CLAUDE.md files still exist
- [ ] All tests pass
