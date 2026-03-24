# Plan #15: Centralize Hardcoded Defaults into ClientConfig

**Status:** Planned
**Type:** implementation
**Priority:** Medium
**Blocked By:** None
**Blocks:** None

---

## Gap

**Current:** `timeout=60`, `num_retries=2`, `base_delay=1.0`, `max_delay=30.0`,
`max_concurrent=5` are hardcoded as defaults in 16+ public function signatures
in `core/client.py`. `ClientConfig` exists but these defaults bypass it.

**Target:** Defaults come from `ClientConfig`. Function signatures use
`None` as sentinel, resolving to config at call time. Users tune behavior
centrally via `ClientConfig` instead of passing kwargs to every call.

**Why:** Per CLAUDE.md: "Maximum Configurability — If you're about to hardcode
a threshold, limit, model name, or policy value, put it in config instead."

---

## References Reviewed

- `llm_client/core/client.py` — 16 occurrences of `timeout=60`, 14 of `num_retries=2`
- `llm_client/core/config.py` — ClientConfig dataclass (existing)
- `~/projects/.claude/CLAUDE.md` — "Maximum Configurability"

---

## Pre-made Decisions

1. **Sentinel pattern:** Change `timeout: int = 60` → `timeout: int | None = None`.
   At call time: `timeout = timeout if timeout is not None else config.default_timeout`.
2. **ClientConfig gets new fields:** `default_timeout`, `default_num_retries`,
   `default_base_delay`, `default_max_delay`, `default_max_concurrent`. All with
   the current hardcoded values as defaults so behavior is unchanged.
3. **Backward compatible:** Explicit kwargs still override config. Passing
   `timeout=60` works exactly as before.
4. **Batch functions too:** `batch_runtime.py` has the same hardcoded defaults.

---

## Files Affected

- `llm_client/core/config.py` (modify — add default fields)
- `llm_client/core/client.py` (modify — 16 signatures + resolution logic)
- `llm_client/execution/batch_runtime.py` (modify — same pattern)
- `tests/test_centralize_defaults.py` (create)

---

## Plan

### Step 1: Add fields to ClientConfig
```python
@dataclass
class ClientConfig:
    ...
    default_timeout: int = 60
    default_num_retries: int = 2
    default_base_delay: float = 1.0
    default_max_delay: float = 30.0
    default_max_concurrent: int = 5
```

### Step 2: Update signatures in core/client.py
Change each `timeout: int = 60` to `timeout: int | None = None`.
Add resolution at the top of each function body.

### Step 3: Update batch_runtime.py signatures
Same pattern.

### Step 4: Tests
Verify: default behavior unchanged, config override works, explicit kwarg overrides config.

---

## Required Tests

| Test File | Test Function | What It Verifies |
|-----------|---------------|------------------|
| `tests/test_centralize_defaults.py` | `test_default_timeout_from_config` | Config default used when kwarg is None |
| `tests/test_centralize_defaults.py` | `test_explicit_kwarg_overrides_config` | Explicit timeout=30 beats config |
| `tests/test_centralize_defaults.py` | `test_backward_compat_no_config` | No config → same defaults as before |

### Existing Tests (Must Pass)

| Test Pattern | Why |
|--------------|-----|
| `tests/test_public_surface.py` | Public API unchanged |
| `tests/test_client.py` | Core dispatch works |

---

## Acceptance Criteria

- [ ] ClientConfig has 5 new default fields
- [ ] All 16+ signatures use None sentinel
- [ ] Defaults resolve to config values at call time
- [ ] Explicit kwargs still override
- [ ] All tests pass
