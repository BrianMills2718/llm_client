# Plan #12: Module Reorganization (Flat → Layered)

**Status:** Planned
**Type:** implementation
**Priority:** High
**Blocked By:** None
**Blocks:** None

---

## Gap

**Current:** 79 Python modules sit flat in `llm_client/`. Program E (Plan 11)
decomposed the mega-files but left everything at root. Finding a module means
scanning 79 files. Related modules (e.g., 13 MCP files, 8 runtime files) have
no grouping except naming conventions.

**Target:** Modules organized into subdirectories by architectural layer:
`core/`, `execution/`, `agent/`, `tools/`, `utils/`. Public API unchanged —
`from llm_client import call_llm` still works via `__init__.py` re-exports.

**Why:** 79 flat files is not navigable. The decomposition work is done (Plan
11) — the grouping hasn't been applied. This is the natural completion of
Program E's structural goals.

---

## References Reviewed

- `llm_client/__init__.py` — current 101 exports, grouped into `_CORE_SUBSTRATE_EXPORTS`, `_COMPAT_HOLD_EXPORTS`, `_CANDIDATE_MOVE_EXPORTS`
- `docs/plans/11_program-e-module-size-reduction.md` — decomposition already done
- `~/projects/archive/llm_client_v2/llm_client/` — v2's layered structure (reference, not copying)
- `~/projects/.claude/CLAUDE.md` — "simplest thing that works", "delete > comment"

---

## Target Directory Structure

```
llm_client/
├── __init__.py              # Public API re-exports (unchanged surface)
├── __main__.py              # CLI entrypoint
│
├── core/                    # Types, config, errors, models — the foundation
│   ├── __init__.py
│   ├── client.py            # ← client.py (dispatch hub)
│   ├── client_dispatch.py   # ← client_dispatch.py
│   ├── config.py            # ← config.py
│   ├── data_types.py        # ← data_types.py
│   ├── errors.py            # ← errors.py
│   ├── model_detection.py   # ← model_detection.py
│   ├── model_selection.py   # ← model_selection.py
│   ├── models.py            # ← models.py
│   └── routing.py           # ← routing.py
│
├── execution/               # Call lifecycle, runtimes, retry, streaming
│   ├── __init__.py
│   ├── background_runtime.py
│   ├── batch_runtime.py
│   ├── call_contracts.py
│   ├── call_lifecycle.py
│   ├── call_wrappers.py
│   ├── completion_runtime.py
│   ├── embedding_runtime.py
│   ├── execution_kernel.py
│   ├── responses_runtime.py
│   ├── retry.py
│   ├── stream_runtime.py
│   ├── streaming.py
│   ├── structured_runtime.py
│   ├── text_runtime.py
│   └── timeout_policy.py
│
├── agent/                   # MCP loop, contracts, tools, turn lifecycle
│   ├── __init__.py
│   ├── agent_adoption.py
│   ├── agent_artifacts.py
│   ├── agent_contracts.py
│   ├── agent_disclosure.py
│   ├── agent_outcomes.py
│   ├── compliance_gate.py
│   ├── context_budget.py    # (ported from v2)
│   ├── deferred_tools.py    # (ported from v2)
│   ├── mcp_agent.py
│   ├── mcp_context.py
│   ├── mcp_contracts.py
│   ├── mcp_evidence.py
│   ├── mcp_finalization.py
│   ├── mcp_loop_summary.py
│   ├── mcp_state.py
│   ├── mcp_tools.py
│   ├── mcp_turn_completion.py
│   ├── mcp_turn_execution.py
│   ├── mcp_turn_model.py
│   ├── mcp_turn_outcomes.py
│   └── mcp_turn_tools.py
│
├── sdk/                     # Agent SDK adapters
│   ├── __init__.py
│   ├── agents.py            # ← agents.py (routing)
│   ├── agents_claude.py
│   ├── agents_codex.py
│   ├── agents_codex_process.py
│   └── agents_codex_runtime.py
│
├── tools/                   # Tool utilities, registry, cleaning
│   ├── __init__.py
│   ├── tool_registry.py     # (ported from v2)
│   ├── tool_result_cleaning.py  # (ported from v2)
│   ├── tool_runtime_common.py
│   ├── tool_shim.py
│   └── tool_utils.py
│
├── observability/           # STAYS AS-IS (already a subdirectory)
│   ├── __init__.py
│   ├── comparison.py
│   ├── context.py
│   ├── events.py
│   ├── experiments.py
│   ├── interventions.py
│   ├── query.py
│   └── replay.py
│
├── utils/                   # Standalone utilities
│   ├── __init__.py
│   ├── cost_utils.py
│   ├── git_utils.py         # (stub — compatibility)
│   ├── openrouter.py
│   └── rate_limit.py
│
├── io_log.py                # STAYS AT ROOT (deeply imported everywhere)
├── experiment_summary.py    # STAYS AT ROOT (3 core modules depend on it)
├── difficulty.py            # STAYS AT ROOT (control plane, widely imported)
├── foundation.py            # STAYS AT ROOT (event taxonomy)
├── langfuse_callbacks.py    # STAYS AT ROOT (LiteLLM callback registration)
├── model_policy_audit.py    # STAYS AT ROOT (governance)
├── prompt_assets.py         # STAYS AT ROOT (asset resolution)
├── prompts.py               # STAYS AT ROOT (render_prompt)
├── workflow_langgraph.py    # STAYS AT ROOT (optional LangGraph PoC)
│
├── cli/                     # STAYS AS-IS (already a subdirectory)
├── data/                    # STAYS AS-IS
├── prompt_assets/           # STAYS AS-IS
├── prompts/                 # STAYS AS-IS (YAML templates)
└── rubrics/                 # STAYS AS-IS
```

## Pre-made Decisions

1. **`io_log.py` stays at root** — too many modules import it directly. Moving it would be a second pass.
2. **`difficulty.py` stays at root** — control plane, deferred review in PROJECTS_DEFERRED.
3. **`__init__.py` re-exports are the compatibility layer** — `from llm_client import call_llm` keeps working because `__init__.py` imports from the new subpackage paths.
4. **Internal imports use relative paths** — within a subdirectory, modules use `from . import X`. Cross-directory imports use `from llm_client.core import X`.
5. **No renaming** — files keep their current names. Only their directory changes. This makes `git mv` diffs readable.
6. **Observability and CLI stay as-is** — already organized.

---

## Plan

### Phase 1: core/ (9 files)
Move foundational modules. Update internal imports. Verify `import llm_client` works.

### Phase 2: execution/ (15 files)
Move runtime and call lifecycle modules. These heavily import from core/ — update paths.

### Phase 3: agent/ (21 files)
Move MCP loop and agent modules. These import from core/ and execution/.

### Phase 4: sdk/ (5 files)
Move SDK adapters. Import from agent/ and core/.

### Phase 5: tools/ (5 files) + utils/ (4 files)
Move utilities. Fewest internal dependencies.

### Each phase:
1. `git mv` files to target directory
2. Add `__init__.py` for subdirectory
3. Update all internal `from llm_client.X import Y` → `from llm_client.core.X import Y` (or appropriate subdir)
4. Update `__init__.py` re-exports to import from new paths
5. Run tests, verify `import llm_client` works
6. Commit

---

## Required Tests

### Existing Tests (Must Pass After Each Phase)

| Test Pattern | Why |
|--------------|-----|
| `tests/test_public_surface.py` | Public API unchanged |
| `tests/test_client.py` | Core dispatch works |
| `tests/test_tool_registry.py` | Ported features work |
| `tests/test_deferred_tools.py` | Ported features work |
| `tests/test_context_budget.py` | Ported features work |
| `tests/test_tool_result_cleaning.py` | Ported features work |

---

## Acceptance Criteria

- [ ] All 79 root modules organized into core/, execution/, agent/, sdk/, tools/, utils/ (or explicitly marked STAYS AT ROOT with reason)
- [ ] `from llm_client import call_llm` still works (public API unchanged)
- [ ] 101 exports unchanged
- [ ] All existing tests pass
- [ ] No downstream project import breakage for public API consumers
- [ ] `python scripts/meta/generate_api_reference.py --write` succeeds

---

## Risks

- **Internal import chains** — modules in execution/ import from core/, agent/ imports from both. Import order and circular dependency issues may surface. Mitigation: move in dependency order (core first, consumers last).
- **3 projects using private paths** — active-stack-core, Digimon, sam_gov already import private internals. This reorganization will break those paths. That's intentional — they were already wrong. But it makes their eventual fix harder (two path changes instead of one). Acceptable.

## Notes

- This is the structural completion of Program E. Plans 6 and 11 did the decomposition. This plan does the organization.
- v2's directory structure is reference material, not a template. v1 has modules v2 deleted (io_log, difficulty, CLI, prompt_assets). The target structure accommodates v1's actual module set.
- Each phase is independently committable and revertable.
