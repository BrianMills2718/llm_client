# Handoff: llm_client consolidation & architecture cleanup

**Date:** 2026-03-24
**Session:** Claude Code (Opus 4.6)
**Duration:** ~5 hours
**Repos touched:** llm_client, agentic_scaffolding, prompt_eval, project-meta

---

## What was done

### Plan #17: llm_client consolidation (COMPLETE)
- Relocated 7 governance modules to proper homes per ADR-2026-03-22:
  - validators → agentic_scaffolding/validators/framework.py (68 tests)
  - git_utils, agent_spec, task_graph, analyzer → project-meta/scripts/meta/
  - scoring + experiment_eval → prompt_eval
- Discovered v2's observability is a regression (deleted SQLite, stubs for experiment_run)
- Cancelled wholesale v2 swap, cherry-picked 4 context-engineering features instead
- Archived v2 to ~/projects/archive/llm_client_v2/
- Updated root CLAUDE.md: scoring → prompt_eval, Langfuse rejected per library guidelines

### Plan #12: Module reorganization (COMPLETE)
- Moved 79 flat modules into 6 subdirectories: core/, execution/, agent/, sdk/, tools/, utils/
- sys.modules aliasing pattern for transparent compatibility stubs
- 101 public exports unchanged, 247+ tests pass

### Plan #14: Batch progress & stagnation (COMPLETE)
- BatchProgressTracker, stagnation detection (rolling error hash), per-item timeout
- New params: progress_interval, on_batch_progress, stagnation_window, abort_on_stagnation, item_timeout_s

### Plan #13: SDK adapter investigation (COMPLETE — no change needed)
- Subprocess fallback is load-bearing (1610+ observed calls, handles real SDK failures)

### Other completed work
- Fixed agent loop fallback bypass bug (text_runtime.py early returns removed)
- Code smell fixes: 6 silent except:pass → warnings, dead imports, misleading function name
- Deleted 28 zero-importer compatibility stubs, fixed all internal canonical paths
- Enabled litellm.enable_json_schema_validation + retryable JSONSchemaValidationError
- Plan #8: subtree CLAUDE.md for 6 new subdirectories
- Plan #15 step 1: ClientConfig default fields added
- Root CLAUDE.md: refined "don't speculate" → "don't add code paths for hypothetical scenarios"
- BACKLOG items resolved/closed
- TODOs added to Digimon, sam_gov, active-stack-core ISSUES.md files

---

## Current state

| Repo | Branch | Pushed | Key state |
|------|--------|--------|-----------|
| llm_client | main + digimon-stable (synced) | Yes | 101 exports, 6 subdirs, 29 stubs remaining |
| agentic_scaffolding | trip-backup-* | Yes | validators/framework.py added |
| prompt_eval | main | Yes | scoring.py + experiment_eval.py added |
| project-meta | master | Yes | Plan #17 complete, handoff docs |

---

## Remaining llm_client work (all planned, none urgent)

### Plan #15 step 2: Wire ClientConfig defaults into signatures
**File:** `docs/plans/15_centralize-defaults.md`
**What:** Change `timeout: int = 60` → `timeout: int | None = None` in 16+ signatures.
Resolve to `config.default_timeout` at call time.
**Blocker:** The resolution needs to happen in the impl functions (text_runtime,
structured_runtime, batch_runtime), not in the client.py facade. The call chain
passes `timeout` through multiple layers. An earlier attempt was reverted because
tests broke — mock targets reference these defaults.
**Step 1 done:** ClientConfig has the fields and resolve_*() methods.

### Plan #16: Remove remaining 29 compatibility stubs
**File:** `docs/plans/16_remove-compatibility-stubs.md`
**What:** 28 stubs already deleted. 29 remain — they're imported by `__init__.py`
and tests. To delete them: update `__init__.py` imports to canonical paths, update
test mock targets, then delete stubs.
**Key insight:** The `__init__.py` still does `from llm_client.client import ...`
(old path via stub) instead of `from llm_client.core.client import ...`. Same for
`mcp_agent` (62 references), `errors` (7), etc.

### Plan #17: text_runtime sync/async deduplication
**File:** `docs/plans/17_text-runtime-dedup.md`
**What:** `_call_llm_impl` and `_acall_llm_impl` are ~400 lines each, near
identical. Also has 86-line rebinding blocks. Strategy: async-first, make
sync wrap async via asyncio.run(). Pre-investigation needed: measure
asyncio.run() overhead.

---

## Downstream projects needing llm_client migration

| Project | Issue | Files | Priority |
|---------|-------|-------|----------|
| active-stack-core | Private API imports (_route_acall, etc.) | 95 | High |
| Digimon_for_KG_application | LiteLLMProvider in 40 files | 51 | Medium |
| sam_gov | Deprecated imports (LiteLLMClient, etc.) | 62 | Low (frozen) |

TODOs added to each project's ISSUES.md.

---

## Gotchas for next agent

- **Pre-commit hooks:** llm_client has doc-coupling, API reference sync, plan
  status consistency, and branch protection (main requires `ALLOW_MAIN_COMMIT=1`
  / `ALLOW_MAIN_PUSH=1`). Regenerate API docs after any public surface change:
  `python scripts/meta/generate_api_reference.py --write`

- **sys.modules stubs:** The 29 remaining stubs at root use `sys.modules[__name__] = _canonical`
  to transparently alias old paths to new subdirectories. This makes ALL attributes
  (including privates and mock targets) work. Don't replace with `from X import *`
  — that breaks private names.

- **project-meta .doc-coupling-acks:** YAML list with `path` + `reason` keys.
  Post-commit hook auto-deletes the file. agentic_scaffolding's hook doesn't
  support ack files — update CLAUDE.md "Last verified" date instead.

- **prompt_eval scoring.py** still has `from llm_client import io_log` (late import
  inside ascore_output). This works because prompt_eval depends on llm_client. If
  io_log ever moves, this breaks.

- **experiment_summary stays in llm_client** — 3 core observability modules import
  it. Don't move it.

- **git_utils.py** exists as both a root stub AND in utils/. The root stub was
  recreated by another agent for import compatibility. The canonical location is
  `llm_client/utils/git_utils.py`.

- **Langfuse rejected:** Per root CLAUDE.md "observability is tier 1, libraries
  are tier 2." Don't recommend adding Langfuse.

- **All personal repos use SSH** (`git@github.com:BrianMills2718/...`).
  `gh auth` active = `brian-steno` (work account).

---

## Session: 2026-03-25/26 — Strategic Review & Cleanup

**Agent:** Claude Code (Opus 4.6)
**Duration:** ~8 hours across 2 sessions

### What was done

#### Strategic review
- Assessed value proposition, alternatives landscape, adoption (398 files across 26 projects)
- Identified core value (observability, policy enforcement) vs redundant layers (retry/fallback)
- Declared maintenance mode: stop internal cleanup, invest in consumer projects

#### Repo declutter (-15,226 lines from tracked files)
- Archived 8 stale docs + 34 meta-patterns + 6 worktree-coordination scripts
- Rewrote README from 1002 to 169 lines
- Created docs/guides/ (4 focused guides extracted from README)
- Rewrote Makefile as consumer observability interface

#### Infrastructure improvements
- Externalized prompt assets to ~/projects/prompts/ (LLM_CLIENT_PROMPT_ASSET_ROOT)
- Added LiteLLM observer callback for unmigrated projects (litellm_observer_callback.py)
- Enabled Instructor validation re-ask (max_retries=0 → 2)
- Added error_type, execution_path, retry_count to observability schema
- Fixed test noise polluting observability DB (conftest.py autouse fixture)
- Added explicit ImportError for 8 extracted functions (clear migration messages)

#### Plan execution
- Plan 16: Removed all 29 compatibility stubs, canonicalized imports
- Plan 08: Marked Complete (subtree instructions exist)
- Plans 15, 17: Cancelled (pure refactoring, no consumer value)

#### Documentation audit
- Superseded ADR-0008 (task_graph/experiment_eval extracted)
- Fixed relationships.yaml stale module references
- Updated master roadmap (replaced 80-line stale checkpoint narrative)
- Added embed/aembed to README API table (16 functions, not 14)
- Listed all 10 task profiles in README

#### Skills created
- /create-skill (global) — skill-making skill
- /setup-project-makefile (projects-level) — standardized Makefile template

### Known issues from this session
- Plan 16 stub removal broke DIGIMON benchmark runner (module-level imports)
- Plan 18 error budget wiring had syntax error in mcp_agent.py (rescued by other agent)
- Downstream consumers need migration: DIGIMON (6 functions), agent_ontology (templates)
- TEMPORARY migration notes added to DIGIMON and agent_ontology CLAUDE.md files

### Gotchas for next agent
- Other agents added: tool-call observability (Wave 0/1), rubric registry, workflow builder, log maintenance — these are on tool-observability-wave1 branch, not merged to main
- Main is stable at 55497fc but tool-observability-wave1 has 7 additional commits
- Don't re-add compatibility stubs — force consumers to use canonical paths
- The rescue branch (rescue/2026-03-26-llm-client-mcp-agent-wip) has the broken error budget wiring — don't use it
