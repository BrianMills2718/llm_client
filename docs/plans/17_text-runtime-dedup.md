# Plan #17: text_runtime Sync/Async Deduplication

**Status:** Planned
**Type:** implementation
**Priority:** High
**Blocked By:** None
**Blocks:** None

---

## Gap

**Current:** `execution/text_runtime.py` contains `_call_llm_impl` (~400 lines)
and `_acall_llm_impl` (~400 lines) that are near character-for-character
identical — the only differences are `await`, `async with`, and `async def`.
Additionally, both functions start with an 86-line block that rebinds every
attribute from `_client` as local variables via `import_module()`.

**Target:** Shared logic extracted into a common path. Sync wrapper calls async
via `asyncio.run()` or a thin adapter. The 86-line rebinding blocks deleted.

**Why:** This is the #1 maintenance risk in llm_client. A bug fixed in one
path but missed in the other is inevitable. 800 lines of duplication → ~450
lines of shared logic + thin sync/async wrappers.

---

## References Reviewed

- `llm_client/execution/text_runtime.py` — lines 21-505 (sync), 508-924 (async)
- `llm_client/execution/text_runtime.py:43-81` — rebinding anti-pattern
- `llm_client/core/client.py` — `call_llm` delegates to `_call_llm_impl`, `acall_llm` to `_acall_llm_impl`
- `~/projects/.claude/CLAUDE.md` — "Delete > Comment", "Simplest thing that works"

---

## Pre-investigation Required

Before executing, decide:

1. **Which dedup strategy?**
   - **Option A: Async-first, sync wraps async.** `_call_llm_impl` becomes
     `asyncio.run(_acall_llm_impl(...))`. Simplest, but sync callers pay
     event loop creation cost.
   - **Option B: Shared inner function + thin async/sync shells.** Extract
     the common logic (argument prep, model selection, finalization) into
     shared helpers. Sync and async each call the shared helpers + their
     own execution path. More code but no runtime overhead.
   - **Option C: Generator/coroutine adapter.** Use a generator that yields
     at async points, with sync/async runners consuming differently.
     Clever but hard to debug.

   **Recommendation:** Option A for most paths. `call_llm` already wraps
   `acall_llm` for agent models. Extend that to all models. Measure
   `asyncio.run()` overhead — if <1ms, it's free.

2. **What about the rebinding blocks?**
   These exist to avoid circular imports. After Plan 12 reorganization,
   the circular import risk is reduced. Try direct imports from leaf
   modules (`from llm_client.core.errors import ...`) instead of
   importing everything from the `client.py` re-export hub.

---

## Files Affected

- `llm_client/execution/text_runtime.py` (major refactor)
- `llm_client/core/client.py` (may simplify `call_llm` delegation)
- `tests/test_client.py` (update mock targets if internal structure changes)

---

## Plan

### Phase 1: Measure asyncio.run() overhead
Benchmark: is `asyncio.run(acall_llm(...))` measurably slower than the
current direct sync implementation? If <1ms overhead, Option A is free.

### Phase 2: Delete _call_llm_impl
Make `call_llm` delegate to `acall_llm` via `asyncio.run()` (same pattern
already used for agent models and batch calls). Delete the 400-line sync
implementation.

### Phase 3: Remove rebinding blocks
Replace the 86-line `_client = import_module(...)` + local rebinding with
direct imports from leaf modules. Test for circular import issues.

### Phase 4: Verify
Run full test suite. Verify sync and async paths produce identical results.
Benchmark to confirm no performance regression.

---

## Required Tests

### New Tests

| Test File | Test Function | What It Verifies |
|-----------|---------------|------------------|
| `tests/test_text_runtime_dedup.py` | `test_sync_async_parity` | call_llm and acall_llm produce same result |
| `tests/test_text_runtime_dedup.py` | `test_sync_performance` | asyncio.run overhead < 5ms |

### Existing Tests (Must Pass)

| Test Pattern | Why |
|--------------|-----|
| `tests/test_client.py` | Core dispatch works |
| `tests/test_public_surface.py` | Public API unchanged |

---

## Acceptance Criteria

- [ ] `_call_llm_impl` deleted (or reduced to 1-line delegation)
- [ ] Rebinding blocks deleted
- [ ] `call_llm` delegates to `acall_llm` for all models
- [ ] Sync/async parity verified by test
- [ ] No performance regression (< 5ms overhead per call)
- [ ] All existing tests pass
- [ ] ~400 lines of duplicated code removed

---

## Risks

- **Circular imports:** Removing the rebinding blocks may surface circular
  import chains that the `import_module()` pattern was hiding. Mitigation:
  test imports after each change; use lazy imports where needed.
- **Event loop conflicts:** `asyncio.run()` fails if already inside an
  event loop (e.g., Jupyter). The current `call_llm` already handles this
  for agent models with a ThreadPoolExecutor fallback. Reuse that pattern.
