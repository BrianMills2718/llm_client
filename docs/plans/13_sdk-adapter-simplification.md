# Plan #13: SDK Adapter Simplification

**Status:** Planned
**Type:** implementation
**Priority:** Medium
**Blocked By:** 12 (module reorganization — adapters should be in sdk/ first)
**Blocks:** None

---

## Gap

**Current:** `agents_claude.py` (618 lines) and `agents_codex.py` (1317 lines)
+ `agents_codex_process.py` + `agents_codex_runtime.py` total ~2500 lines.
They include process isolation, CLI fallback logic, and workaround chains for
SDK instability.

**Target:** Thin fail-loud adapters (~400-500 lines total). If an SDK call
fails, surface the error instead of falling back to a CLI subprocess.

**Why:** Per CLAUDE.md: "Fail Loud — No silent fallbacks." The current adapters
silently fall back to subprocess invocation when the SDK fails. This hides
real errors and makes debugging harder. v2's thin adapters (178 + 195 lines)
proved the reduced surface is sufficient.

---

## References Reviewed

- `llm_client/agents_claude.py` — current 618 lines
- `llm_client/agents_codex.py` — current 1317 lines
- `llm_client/agents_codex_process.py` — process isolation logic
- `llm_client/agents_codex_runtime.py` — runtime adapter
- `~/projects/archive/llm_client_v2/llm_client/sdk/claude.py` — 178 lines (reference)
- `~/projects/archive/llm_client_v2/llm_client/sdk/codex.py` — 195 lines (reference)
- `~/projects/.claude/CLAUDE.md` — "Fail Loud", "Libraries vs. hand-rolling"

---

## Pre-investigation Required

Before executing, answer these questions:

1. **Is the subprocess fallback actually used?** Grep all downstream projects
   for evidence of the fallback being triggered in logs/observability. If it
   fires regularly, the SDK has a real bug and we should fix that, not remove
   the fallback.

2. **Which projects use the SDK adapters?** The downstream audit (Plan #17)
   identified active-stack-core as a heavy user. Check if it relies on the
   process isolation behavior.

3. **Are the Claude/Codex SDKs stable now?** The fallback logic was written
   when the SDKs were unstable. If they've stabilized, the fallback is dead
   code and should be deleted per "Delete > Comment."

---

## Files Affected

- `llm_client/sdk/agents.py` (modify — after Plan 12 moves it here)
- `llm_client/sdk/agents_claude.py` (simplify)
- `llm_client/sdk/agents_codex.py` (simplify)
- `llm_client/sdk/agents_codex_process.py` (likely delete)
- `llm_client/sdk/agents_codex_runtime.py` (likely delete or merge)
- `tests/test_agents.py` (update)

---

## Plan

### Phase 1: Investigate
Answer the 3 pre-investigation questions above. If the fallback is regularly
triggered, this plan changes to "fix the SDK" not "remove the fallback."

### Phase 2: Simplify
Using v2's thin adapters as reference (not copying — they may not handle v1's
observability correctly), reduce to fail-loud wrappers.

### Phase 3: Verify
Run downstream projects that use SDK adapters. Confirm no silent failures.

---

## Acceptance Criteria

- [ ] Pre-investigation questions answered with evidence
- [ ] No silent fallbacks remain (grep for subprocess.Popen, subprocess.run in adapters)
- [ ] SDK errors surface immediately with useful diagnostics
- [ ] Existing tests pass
- [ ] active-stack-core still works (or breakage is documented as intentional)

---

## Notes

- Blocked by Plan 12 because the files should be in `sdk/` before simplifying.
- If investigation reveals the fallback IS needed, this plan pivots to
  "fix the SDK" or "make the fallback loud" instead of removing it.
