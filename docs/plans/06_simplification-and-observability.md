# Plan 06: Simplification and Observability Modernization

**Status:** In Progress
**Type:** implementation
**Priority:** High
**Blocked By:** None (Programs A-D complete)
**Blocks:** None

---

## Gap

**Current:** Programs A-D hardened boundaries and extracted runtimes, but three
mega-files remain oversized and hard to reason about:

- `mcp_agent.py` (4,959 lines) — mixes loop orchestration, context management,
  evidence tracking, tool contracts, and finalization logic
- `client.py` (4,630 lines) — still contains data classes, retry/backoff,
  OpenRouter key rotation, error classification, model detection, routing
  helpers, cost extraction, streaming classes, and cache protocols alongside
  its dispatch facade role
- `agents.py` (2,818 lines) — mixes Claude Agent SDK, Codex SDK, Codex CLI
  subprocess management, process isolation, and shared utilities

Additionally, observability has known scaling gaps:

- JSONL logs grow unbounded (no rotation)
- No integration with external observability (Langfuse) despite LiteLLM having
  built-in callback support
- Model registry has no CLI for inspection/updates
- No shared request fingerprint, compact call diff, or controlled replay
  surface when operational paths disagree

**Target:** Each module has a single clear responsibility. No module exceeds
~1,000 lines. Observability scales with Langfuse as a complementary backend,
supports controlled replay/divergence diagnosis, and model registry is
CLI-inspectable.

**Why:** This is a strategic review finding (2026-03-18), not a cleanup wish.
The code is unique and valuable — the problem is density, not redundancy.
The audit confirmed retry/fallback, structured output routing, and budget
enforcement all add genuine value beyond LiteLLM. The refactor preserves all
capabilities while making the codebase easier to navigate, modify, and review.

**Evidence:** Strategic review conducted 2026-03-18 with full competitive
analysis against LiteLLM, PydanticAI, Langfuse, Portkey, and 12 other tools.
All three mega-files identified as maintainability risks. Observability gap
confirmed against Langfuse feature set. A 2026-03-22 live-vs-proxy divergence
review also showed that current observability is better at proving a mismatch
than at replaying or localizing it.

---

## References Reviewed

- `llm_client/client.py` — 4,630 LOC, 60+ functions and classes
- `llm_client/mcp_agent.py` — 4,959 LOC, MCP agent loop + all supporting logic
- `llm_client/agents.py` — 2,818 LOC, dual SDK adapter
- `llm_client/io_log.py` — 1,692 LOC, JSONL + SQLite logging (no rotation)
- `docs/plans/02_client-boundary-hardening.md` — prior decomposition work
- Strategic review competitive analysis (2026-03-18)

---

## Files Affected

> This section declares files touched by this plan. New slices must amend this
> section before touching additional files.

### Phase 1A: client.py decomposition
- llm_client/client.py (modify — reduce to dispatch facade)
- llm_client/data_types.py (create — LLMCallResult, EmbeddingResult, CachePolicy, LRUCache)
- llm_client/retry.py (create — RetryPolicy, backoff strategies, retry helpers)
- llm_client/openrouter.py (create — key rotation, OpenRouter detection)
- llm_client/streaming.py (create — LLMStream, AsyncLLMStream)
- llm_client/model_detection.py (create — model type detection helpers)
- llm_client/cost.py (create — usage extraction, cost computation)
- llm_client/__init__.py (modify — update imports)
- tests/ (modify — update imports where needed)

### Phase 1B: mcp_agent.py decomposition
- llm_client/mcp_agent.py (modify — reduce to loop orchestration)
- llm_client/mcp_context.py (create — context window management, compaction)
- llm_client/mcp_evidence.py (create — evidence tracking, stagnation detection)
- llm_client/mcp_finalization.py (create — forced-final, circuit breaker)
- llm_client/mcp_state.py (create — AgentLoopRuntimePolicy, AgentLoopToolState, init)
- llm_client/mcp_tools.py (create — tool helpers, budget tracking, normalization)
- llm_client/mcp_contracts.py (create — contract validation, capability state, bindings)
- tests/ (modify — update imports where needed)

### Phase 1C: agents.py decomposition
- llm_client/agents.py (modify — reduce to dispatch + shared types)
- llm_client/agents_claude.py (create — Claude Agent SDK adapter)
- llm_client/agents_codex.py (create — Codex SDK + CLI + process isolation)
- tests/ (modify — update imports where needed)

### Phase 2: Langfuse callback
- llm_client/io_log.py (modify — add Langfuse callback configuration)
- docs/API_REFERENCE.md (modify — document Langfuse setup)

### Phase 2A: Replay and divergence diagnosis
- docs/plans/09_replay-and-divergence-diagnosis.md (create — dedicated child plan)
- docs/adr/0014-call-replay-and-divergence-diagnosis-boundary.md (create — boundary decision)
- scripts/relationships.yaml (modify — govern new observability ADR linkage)

### Phase 3: JSONL log rotation
- llm_client/io_log.py (modify — add rotation to JSONL appender)

### Phase 4: Model registry CLI
- llm_client/cli/models.py (create — list/inspect/add commands)
- llm_client/__main__.py (modify — register models subcommand)

---

## Program Guardrails

Every phase in this plan follows the shared planning rules:

1. Acceptance criteria are defined before implementation.
2. Deterministic behavior is proven with unit tests before broader adoption.
3. Work lands in thin, independently verifiable slices.
4. Integration points are tested when wired, not deferred.
5. Assumptions and risks are written down explicitly.
6. No big-bang package reorg or repo split is allowed under this plan.
7. Public API signatures must not change.

If a proposed slice violates any of those rules, that slice fails planning and
must be redesigned before code changes proceed.

---

## Overall Definition Of Done

This program is done only when all of the following are true:

1. No module in `llm_client/` exceeds ~1,200 lines (soft target; hard limit
   1,500 for complex modules with documented justification).
2. Each extracted module has a single-sentence responsibility that does not
   rely on "and" to hide multiple concerns.
3. All existing tests pass without modification to test logic (import paths
   may change).
4. Public `llm_client` API (all 14 functions + data types) unchanged.
5. Langfuse callback is available when configured, invisible when not.
6. Shared replay/divergence diagnosis exists for call-level operational mismatches.
7. JSONL logs rotate by date or size.
8. `llm-client models list` works from CLI.

---

## Long-Term Phases

### Phase 1A: Decompose `client.py` Into Concern-Specific Modules

**Purpose:** Extract data types, retry logic, OpenRouter key management,
streaming classes, model detection, and cost computation out of `client.py`,
leaving it as a thin dispatch facade.

**Input -> Output:** 4,630-line mixed-responsibility file -> ~1,000-line
dispatch facade + 6 focused modules

**Target decomposition:**

| New module | Responsibility | Approx lines |
|-----------|----------------|--------------|
| `data_types.py` | LLMCallResult, EmbeddingResult, CachePolicy, AsyncCachePolicy, LRUCache | ~200 |
| `retry.py` | RetryPolicy, Hooks, backoff strategies, retry/delay helpers | ~250 |
| `openrouter.py` | Key rotation, key pool management, OpenRouter detection | ~200 |
| `streaming.py` | LLMStream, AsyncLLMStream | ~250 |
| `model_detection.py` | _is_claude_model, _is_thinking_model, _is_responses_api_model, _is_gemini_model, etc. | ~150 |
| `cost.py` | _extract_usage, _compute_cost, _parse_cost_result | ~100 |

**Passes if:**

- `client.py` is under 1,200 lines
- All 14 public functions work identically
- All existing tests pass
- No behavior changes — pure structural move
- Internal imports updated consistently

**Fails if:**

- Any public signature or return type changes
- Circular imports introduced
- Test logic must be rewritten (import path updates are fine)
- Extracted modules depend back on `client.py` internals

### Phase 1B: Decompose `mcp_agent.py` Into Concern-Specific Modules

**Purpose:** Break the MCP agent loop into focused modules by concern while
preserving the full capability set (artifact contracts, progressive disclosure,
stagnation detection, lane closure, context management).

**Input -> Output:** 4,959-line monolith -> ~800-line loop orchestrator + 5-6
focused modules

**Target decomposition:**

| New module | Responsibility | Approx lines |
|-----------|----------------|--------------|
| `mcp_state.py` | AgentLoopRuntimePolicy, AgentLoopToolState, policy resolution, tool state init | ~400 |
| `mcp_context.py` | Context compaction, tool result clearing, message char measurement | ~200 |
| `mcp_evidence.py` | Evidence pointer tracking, stagnation detection, digest computation | ~200 |
| `mcp_finalization.py` | Forced finalization, circuit breaker, finalization fallback | ~300 |
| `mcp_tools.py` | Tool normalization, budget tracking, budget-exempt detection, trim-to-budget | ~200 |
| `mcp_contracts.py` | Capability state, binding hashes, contract validation, disclosure filtering, repair suggestions | ~400 |

**Passes if:**

- `mcp_agent.py` is under 1,200 lines
- `acall_with_mcp_runtime` produces identical results
- All 50+ event codes still emitted correctly
- Artifact contract validation unchanged
- Existing tests pass

**Fails if:**

- Agent loop behavior changes
- Event codes lost or reclassified
- State management becomes split across modules without clear ownership
- Circular imports between mcp_* modules

### Phase 1C: Decompose `agents.py` Into SDK-Specific Modules

**Purpose:** Separate Claude Agent SDK and Codex SDK adapters into distinct
modules with shared utilities.

**Input -> Output:** 2,818-line dual-adapter -> ~300-line dispatch + 2 focused
SDK modules

**Target decomposition:**

| New module | Responsibility | Approx lines |
|-----------|----------------|--------------|
| `agents.py` | Dispatch (_route_call, _route_acall), shared types, model parsing, billing | ~300 |
| `agents_claude.py` | Claude Agent SDK: _import_sdk, _build_agent_options, _acall_agent, _call_agent, streaming | ~500 |
| `agents_codex.py` | Codex SDK + CLI: process isolation, _call_codex, _acall_codex, CLI subprocess, tree termination | ~1,200 |

**Passes if:**

- All agent SDK routing works identically
- Codex CLI fallback + process isolation preserved
- All existing tests pass
- Claude and Codex concerns no longer interleaved

**Fails if:**

- Agent routing behavior changes
- Process isolation logic broken
- Fallback chain (SDK -> CLI) disrupted

### Phase 2: Langfuse Callback Integration

**Purpose:** Enable Langfuse as a complementary observability backend via
LiteLLM's built-in callback mechanism, while preserving local JSONL+SQLite
as the zero-infrastructure default.

**Passes if:**

- Setting `LITELLM_CALLBACKS=langfuse` + Langfuse env vars enables Langfuse
- No Langfuse dependency when not configured
- Local JSONL+SQLite logging continues unchanged
- `task=` and `trace_id=` metadata flows to Langfuse spans
- Documented in API reference

**Fails if:**

- Langfuse becomes a required dependency
- Local logging is degraded
- New runtime dependency added to core package

### Phase 2A: Shared Replay And Divergence Diagnosis

**Purpose:** add the missing observability capability for stable request
fingerprints, compact call diffs, and call-level replay when operational paths
disagree.

**Plan:** [09_replay-and-divergence-diagnosis.md](./09_replay-and-divergence-diagnosis.md)

**Passes if:**

- `llm_client` can compare two captured calls with a compact diff
- `llm_client` can replay a captured call under a fresh trace/project tag
- full replayable payloads are preserved without truncation
- the shared surface is proven on one real mismatch case

**Fails if:**

- comparison logic remains project-local
- replay depends on hidden workflow state not present at the call boundary
- the implementation silently truncates replayable payloads

**Completion evidence (2026-03-22):**

- Plan: [09_replay-and-divergence-diagnosis.md](./09_replay-and-divergence-diagnosis.md)
- Proof note:
  [2026-03-22_replay_divergence_proof_slice.md](../reviews/2026-03-22_replay_divergence_proof_slice.md)
- real proof rows:
  - live `201474`
  - parity `201477`
  - replay `201480`

### Remaining Completion Gap (2026-03-22 audit)

Program E is not ready to close yet. A fresh module-size audit still shows
multiple files above the plan thresholds:

1. `llm_client/client.py`: `4184` lines
2. `llm_client/mcp_turn_execution.py`: `1339` lines
3. `llm_client/observability/experiments.py`: `1322` lines
4. `llm_client/agents_codex.py`: `1317` lines
5. `llm_client/agent_contracts.py`: `1228` lines

`llm_client/io_log.py` was reduced from `2102` to `1222` across three verified
Plan 11 slices, so it no longer blocks the hard-threshold criterion.
`llm_client/agents_codex.py` was reduced from `1931` to `1317` across two
verified Plan 11 slices, so it is below the hard threshold but still slightly
above the soft target. `llm_client/mcp_agent.py` was reduced from `3335` to
`1037` by the turn-execution extraction, so it likewise no longer blocks the
hard-threshold criterion. The extracted
`llm_client/mcp_turn_execution.py` follow-on slices then removed duplicate MCP
runtime facades, split final bookkeeping, extracted the per-turn
tool-processing path, post-tool outcomes, post-loop completion handoff, and
the pre-call model stage, reducing that module from `3202` to `1339`. That
means it no longer blocks the hard-threshold criterion either.

The next child slice for this program is
[11_program-e-module-size-reduction.md](./11_program-e-module-size-reduction.md),
with the immediate next tranche now explicitly selected as the `client.py`
lifecycle/heartbeat monitoring extraction.

### Phase 3: JSONL Log Rotation

**Purpose:** Prevent unbounded log growth with date-based file rotation.

**Passes if:**

- JSONL files rotate daily (new file per day)
- Configurable retention (default: 30 days, env var override)
- Old files compressed or deleted based on config
- No data loss during rotation
- Backward-compatible file naming

**Fails if:**

- Active writes lost during rotation
- Performance degraded on normal append path
- Requires external cron or daemon

### Phase 4: Model Registry CLI

**Purpose:** Add CLI commands for inspecting and querying the model registry.

**Passes if:**

- `llm-client models list` shows all registered models with task suitability
- `llm-client models show <name>` shows full model details
- `llm-client models tasks` shows available task profiles
- Output is compact, terminal-friendly

**Fails if:**

- CLI modifies registry state (read-only for v1)
- Adds runtime dependencies

---

## Assumptions

1. Plan 02's runtime split (text_runtime.py, structured_runtime.py, etc.)
   was correct and should be preserved — this plan extends that pattern.
2. No downstream project imports private helpers from client.py, mcp_agent.py,
   or agents.py directly (they use the public API).
3. Langfuse integration via LiteLLM callbacks is sufficient — no need for
   direct Langfuse SDK integration.
4. JSONL log rotation can be implemented with stdlib (no new dependencies).

## Risks

1. **Import cycle risk** — Extracting modules from client.py could create
   circular imports if extracted modules need client.py types and client.py
   needs extracted functions. Mitigation: data_types.py is the leaf — nothing
   it imports should import it back.
2. **MCP state coupling** — The ~60 state variables in the agent loop may
   resist clean decomposition. Mitigation: state lives in typed dataclasses
   passed by reference; modules operate on state, they don't own it.
3. **Test import brittleness** — Some tests may import private helpers by
   path. Mitigation: leave re-exports in original modules during transition,
   remove in a follow-up pass.
