# Operational Knowledge — llm_client

Shared findings from all agent sessions. Any agent brain can read and append.
Human-reviewed periodically.

## Findings

<!-- Append new findings below this line. Do not overwrite existing entries. -->
<!-- Format: ### YYYY-MM-DD — {agent} — {category}                          -->
<!-- Categories: bug-pattern, performance, schema-gotcha, integration-issue, -->
<!--             workaround, best-practice                                   -->
<!-- Agent names: claude-code, codex, openclaw                               -->

---

### 2026-04-01 — claude-code — best-practice

**Ecosystem audit findings (Phase 7 of infra sprint).**

Health: 1240 tests, 94% pass rate, no circular imports, clean architecture.

Issues found:
1. Plan #17 (text_runtime sync/async dedup) cancelled but 400 LOC duplication
   remains in execution/text_runtime.py. High maintenance risk — bug fixed in
   one path but missed in other. Consider un-cancelling.
2. KNOWLEDGE.md was empty until this entry. Agents should use /learned to
   record findings during debugging sessions.
3. 5 test_models.py failures due to stale model ranking expectations
   (gemini-2.5-flash vs gemini-3-flash-preview). Parameterize or update.
4. test_public_surface.py expects 118 exports, actual is 120. Trivial fix.
5. Gate-edit hook test overly strict on doc files. Gate should skip non-source.
