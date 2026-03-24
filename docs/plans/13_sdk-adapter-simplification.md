# Plan #13: SDK Adapter Simplification

**Status:** ✅ Complete
**Type:** implementation
**Priority:** Medium
**Blocked By:** 12 (module reorganization — complete)
**Blocks:** None

---

## Gap

**Current:** `agents_claude.py` (618 lines) and `agents_codex.py` (1317 lines)
+ `agents_codex_process.py` + `agents_codex_runtime.py` total ~2500 lines.
They include process isolation, CLI fallback logic, and workaround chains for
SDK instability.

**Target (original):** Thin fail-loud adapters (~400-500 lines total).

**Target (revised after investigation):** Keep as-is. The complexity is earned.

---

## Investigation Results (2026-03-23)

### Q1: Is the subprocess fallback actually used?
**YES.** Codex SDK falls back to CLI transport on TimeoutError,
ConnectionError, OSError, and CODEX_WORKER_* errors. This is an active
stability mechanism, not dead code.

### Q2: Which projects use the SDK adapters?
4 projects: Digimon_for_KG_application (1610 observed calls), research_v2,
ac10, Digimon_autoloop. All route through `call_llm()`/`acall_llm()`.

### Q3: Are the SDKs stable?
Codex SDK (0.1.11) is production-grade. Buffer limit patch in
`agents_codex.py` (lines 207-281) increases asyncio subprocess buffer from
64KB to 4MB for large MCP tool results. The fallback handles real failure
modes that the SDK itself doesn't.

---

## Decision: No Simplification

The plan originally assumed the fallback was "just in case" dead code. The
investigation proved it's load-bearing:
- Process isolation is used in benchmarks for determinism
- Buffer limit patch addresses a real asyncio limitation
- CLI fallback fires on real SDK failures (timeout, connection, EOF)
- 1610+ documented calls through these adapters

Per CLAUDE.md: "Simplest thing that works." The adapters ARE the simplest
thing that works for SDK stability. Removing the fallback would make Codex
calls brittle under load.

**The only change worth making:** ensure the fallback logs at WARNING level
and emits an observability event so it's visible when it fires. If it already
does this (the code logs `CODEX_TRANSPORT_FALLBACK[sdk->cli]`), no changes
needed.

---

## Notes

- This plan demonstrates the value of "search before building" — investigation
  prevented a destructive simplification that would have broken 4 projects.
- v2's thin adapters (178+195 lines) worked because v2 deleted the fallback.
  That was a regression for stability, not an improvement.
