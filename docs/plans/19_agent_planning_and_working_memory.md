# Plan #19: Agent Planning and Working Memory

**Status:** Complete
**Type:** implementation
**Priority:** High
**Blocked By:** None
**Blocks:** DIGIMON benchmark quality, any agent loop consumer needing multi-step reasoning

---

## Gap

**Current:** The llm_client agent loop has no built-in planning or progress tracking. The agent gets tools and a prompt, then improvises each turn with no persistent working memory. DIGIMON built `semantic_plan` and `todo_write` as project-specific MCP tools, but they require the agent to explicitly call them and aren't auto-injected into context. agentic_scaffolding has cross-session `AttemptHistory` but no per-task planning. The result: agents lose track of what they've tried, repeat themselves (observed: 318 LLM calls for one question), and can't coordinate multi-hop reasoning across turns.

**Target:** llm_client provides built-in planning and todo tools that any agent loop consumer gets automatically. Plan and progress state is auto-injected into every turn's context so the agent always sees where it is.

**Why:** Claude Code and Codex both have explicit plan/task mechanisms. Every serious agent runtime needs working memory. This is general infrastructure, not a DIGIMON-specific concern.

---

## References Reviewed

- DIGIMON `semantic_plan` and `todo_write` (`digimon_mcp_stdio_server.py:6010-6493`): typed atoms with dependencies, todo status tracking, compact status line injection
- llm_client artifact context (`agent/mcp_state.py`): auto-injected artifact handles per turn — proves the injection pattern works
- agentic_scaffolding `AttemptHistory` (`memory/history.py`): relevance-gated cross-session memory
- llm_client context budget (`agent/mcp_context.py`): message compaction and tool result clearing — must not conflict with plan injection
- Claude Code TaskCreate/TaskUpdate: in-memory task list with session persistence
- llm_client `AgentLoopRuntimePolicy`: existing knobs for context injection behavior

---

## Requirements

1. **Plan creation tool**: Agent calls it at task start. Produces structured steps with dependencies. Harness stores the plan.
2. **Todo update tool**: Agent updates step status (pending → in_progress → done) and records evidence/results per step. Harness tracks state.
3. **Auto-injected context**: On every agent turn, the harness prepends a compact plan+progress summary to the messages. The agent doesn't need to ask — it always sees current state.
4. **Configurable**: Consumers declare whether planning is enabled, provide custom plan tools if needed, control injection format and budget.
5. **No timeouts**: Per CLAUDE.md, no time-based cutoffs. Budget controls are via turn counts (Plan #18).

---

## Domain Model

```
AgentPlan
├── plan_id: str
├── question: str (the original task)
├── steps: list[PlanStep]
│   ├── step_id: str (s1, s2, ...)
│   ├── description: str
│   ├── depends_on: list[str] (step_ids)
│   ├── status: pending | in_progress | done | blocked
│   ├── result: str | None (evidence or answer when done)
│   └── attempts: int (how many times this step was attempted)
└── created_turn: int

PlanningConfig
├── enabled: bool = True
├── auto_inject_context: bool = True
├── context_format: "compact" | "full" = "compact"
├── max_context_chars: int = 500 (budget for injected plan summary)
├── custom_plan_tool: Callable | None = None (project-specific planner)
├── custom_todo_tool: Callable | None = None (project-specific updater)
└── require_plan_before_tools: bool = False (if True, first action must be create_plan)
```

---

## Contracts

### `create_plan` tool

**Input:**
```python
async def create_plan(task: str, max_steps: int = 8) -> str:
    """Create a structured plan for the task. Call FIRST before other tools.

    Args:
        task: The question or task to plan for.
        max_steps: Maximum number of plan steps (2-8).

    Returns:
        JSON with plan steps, each having id, description, depends_on.
    """
```

**Output:** JSON with `steps: [{step_id, description, depends_on}]`

**Implementation:** LLM call to decompose the task. Uses the agent loop's own model (or a configurable planner model). DIGIMON can replace this with `semantic_plan` via `PlanningConfig.custom_plan_tool`.

### `update_plan` tool

**Input:**
```python
async def update_plan(
    step_id: str,
    status: str,  # "in_progress" | "done" | "blocked"
    result: str = "",  # Evidence or finding for this step
) -> str:
    """Update a plan step's status and record results.

    Args:
        step_id: Which step to update (s1, s2, ...).
        status: New status.
        result: What was found/achieved (required when status=done).

    Returns:
        Updated plan summary showing all steps and their status.
    """
```

**Output:** Compact plan summary (same format as auto-injected context).

### Auto-injected context message

Prepended to messages on every turn after plan creation:

```
[PLAN PROGRESS: 2/5 steps done]
[x] s1: Find the publisher of Labyrinth → "Panther Books"
[>] s2: Find when Panther Books ended (searching...)
[ ] s3: Return the year
Depends: s2 needs s1 ✓, s3 needs s2
```

Format: compact, one line per step, ≤500 chars total. Truncates long results. Shows dependency status.

---

## Pre-made Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Where to put PlanningConfig | `llm_client/agent/agent_contracts.py` | Same file as AgentErrorBudget — agent-level config |
| Where to put plan state | `llm_client/agent/agent_planning.py` (new) | Isolate planning logic from the already-large turn execution |
| Where to put built-in tools | `llm_client/agent/agent_planning.py` | Tools and state in same module |
| How to inject context | Prepend a `{"role": "user", "content": "[PLAN PROGRESS...]"}` message before each LLM call | Same pattern as artifact context injection in `mcp_state.py` |
| Context budget | 500 chars default, configurable | Fits ~8 steps with short results without competing with tool results |
| Plan tool model | Use the agent's own model (no separate LLM call) | The agent reasons about the plan in its own response, the tool just stores the structured output |
| Interaction with DIGIMON | DIGIMON sets `PlanningConfig(custom_plan_tool=semantic_plan, custom_todo_tool=todo_write)` | Existing DIGIMON tools are richer than the generic ones; llm_client should defer to them |
| Interaction with ErrorBudget | Planning tools are budget-exempt (like submit_answer) | Creating/updating a plan shouldn't count against the tool call budget |
| Default behavior | `PlanningConfig(enabled=True)` — planning tools always available | Consumers can disable with `enabled=False` |
| Status aliases | Accept "started"→"in_progress", "complete"→"done", "waiting"→"blocked" | Match DIGIMON's existing alias handling |

---

## Files Affected

| File | Change |
|------|--------|
| `llm_client/agent/agent_planning.py` | **NEW** — PlanningConfig, AgentPlan, PlanStep, built-in tools, context formatter |
| `llm_client/agent/agent_contracts.py` | Add PlanningConfig export |
| `llm_client/agent/mcp_turn_execution.py` | Inject plan context before each LLM call; register plan tools |
| `llm_client/agent/mcp_agent.py` | Accept `planning_config` kwarg, pass to `_agent_loop` |
| `llm_client/__init__.py` | Export PlanningConfig, AgentPlan |
| `tests/test_agent_planning.py` | **NEW** — unit tests for plan creation, status updates, context injection, budget interaction |

---

## Plan

### Step 1: Data model and state (agent_planning.py)

1. Create `PlanStep` dataclass (step_id, description, depends_on, status, result, attempts)
2. Create `AgentPlan` dataclass (plan_id, question, steps, created_turn)
3. Create `PlanState` class with methods:
   - `create_plan(task, steps)` → stores plan
   - `update_step(step_id, status, result)` → updates step
   - `format_context(max_chars)` → compact summary string
   - `summary()` → dict for observability
4. Create `PlanningConfig` dataclass
5. Unit tests: plan creation, step updates, status aliases, context formatting, char budget

### Step 2: Built-in tools (agent_planning.py)

1. Implement `create_plan` tool function — takes task string, returns structured steps
   - Default implementation: LLM call via the agent's own model to decompose
   - Actually simpler: the agent itself produces the plan in its response, the tool just structures and stores it. No extra LLM call needed.
2. Implement `update_plan` tool function — takes step_id + status + result, updates state
3. Both tools operate on the shared `PlanState` instance
4. Unit tests: tool functions produce correct state changes

### Step 3: Context injection (mcp_turn_execution.py)

1. In `_agent_loop`, after `PlanState` is created:
   - Before each LLM call, if plan exists, call `plan_state.format_context()` and prepend to messages
   - Use same injection pattern as artifact context (`_upsert_active_artifact_context_message`)
   - Respect `PlanningConfig.max_context_chars` budget
2. Register plan tools alongside other tools in the loop
3. Make plan tools budget-exempt in the error budget
4. Integration test: verify context injection happens, verify plan tools don't count against budget

### Step 4: Wire into callers

1. `mcp_agent.py` `acall_with_python_tools_runtime`: accept `planning_config` kwarg, pass to `_agent_loop`
2. `__init__.py`: export `PlanningConfig`, `AgentPlan`
3. Update API reference
4. Integration test: end-to-end with a simple tool-calling model

### Step 5: DIGIMON integration

1. In DIGIMON's benchmark runner, set `planning_config=PlanningConfig(custom_plan_tool=dms.semantic_plan, custom_todo_tool=dms.todo_write)`
2. Remove the manual tool injection of `semantic_plan` and `todo_write` from `build_consolidated_tools`
3. Verify: 3q HotpotQAsmallest produces same or better results

---

## Error Taxonomy

| Error | Diagnosis | Fix |
|-------|-----------|-----|
| Plan context exceeds budget | Too many steps or long results | Truncate results to fit `max_context_chars` |
| Agent ignores plan tools | Prompt doesn't instruct planning | Add planning instruction to system prompt template |
| Plan injection conflicts with context compaction | Old plan messages get compacted | Plan context uses a fixed message index (like artifact context) — replaced in place, not appended |
| Custom plan tool has different signature | DIGIMON's `semantic_plan` takes `question` not `task` | Adapter wrapper in PlanningConfig handles signature translation |
| Agent creates plan but never updates it | No progress tracking | Log warning if N turns pass without `update_plan` call |
| Step dependency cycle | Agent creates circular depends_on | Validate at creation time, reject cycles |

---

## Acceptance Criteria

- [ ] `PlanningConfig` dataclass with `enabled`, `auto_inject_context`, `max_context_chars`, `custom_plan_tool`, `custom_todo_tool`
- [ ] `create_plan` and `update_plan` tools registered automatically when `PlanningConfig.enabled=True`
- [ ] Plan progress auto-injected into every turn's context when plan exists
- [ ] Context injection respects `max_context_chars` budget
- [ ] Plan tools are budget-exempt (don't count against AgentErrorBudget)
- [ ] Custom tools (DIGIMON's semantic_plan/todo_write) work via `PlanningConfig` without code changes
- [ ] Existing benchmark runs (HotpotQAsmallest 3q) produce same or better results
- [ ] Unit tests for: plan creation, step updates, status aliases, context formatting, budget interaction

## Required Tests

- `pytest tests/test_agent_planning.py -q`
- `pytest tests/test_agent_runtime_adapters.py -q`
- `pytest tests/test_mcp_agent.py -q`
- `pytest tests/test_public_surface.py -q`

---

## Open Questions (Pattern 29)

### Q1: Should create_plan make an LLM call or structure the agent's own output?
**Status:** ⏸️ Deferred
**Raised:** 2026-03-25
**Context:** Option A: `create_plan` makes a separate LLM call to decompose the task (like DIGIMON's `semantic_plan`). Option B: The agent produces the plan in its own response, `create_plan` just structures and stores it (cheaper, no extra call). Option C: The tool accepts a list of steps as input — the agent decomposes in its reasoning, then passes the steps to the tool for tracking.
**Decision:** Start with Option C (simplest). The agent already knows how to decompose. The tool just stores and tracks. If this doesn't work, upgrade to Option A.

### Q2: Should plan injection replace or supplement artifact context?
**Status:** ❓ Open
**Raised:** 2026-03-25
**Context:** The agent loop already injects artifact context. Adding plan context increases context budget pressure. Should they share a budget or have separate budgets?
**Action:** Separate budgets — plan context is 500 chars, artifact context is its own budget. Total overhead is small relative to 260K max context.

---

## Budget

- Step 1 (data model): ~1 hour code, 0 LLM cost
- Step 2 (tools): ~30 min code, 0 LLM cost
- Step 3 (injection): ~1 hour code, 0 LLM cost
- Step 4 (wiring): ~30 min code, 0 LLM cost
- Step 5 (DIGIMON integration): ~30 min code, ~$0.50 for 3q smoke test
- **Total: ~3.5 hours, ~$0.50**
