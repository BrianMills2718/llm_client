# Task Graph Runner — Design Document

**Date**: 2026-02-16
**Status**: Phase 1 implemented
**Location**: `llm_client/task_graph.py` (core), `~/.openclaw/bin/` (OpenClaw integration)

## Problem

OpenClaw coordinates multi-step autonomous work by dispatching to Codex and Claude Code agents. The current `run_task.py` handles single flat tasks — no dependencies, no structured validation, no context handoff, no learning from past runs.

Real work has structure: "collect data" before "build graph" before "extract beliefs." Tasks need validated outputs, not just "did a git commit happen." And the system should get cheaper and more reliable over time by learning which models and prompts work for which tasks.

## Design Principles

1. **Dumb runner, smart coordinator.** The task graph runner is mechanical — parse, sort, dispatch, validate, log. OpenClaw (the LLM) does all reasoning about what to run and when. No AI in the runner itself.
2. **External validation only.** Never trust agent self-reports. Validate via file checks, pytest, SQL queries, MCP tool calls.
3. **Every run is an experiment.** Structured records with hypothesis, dimensions, outcome, learnings. The self-improvement loop reads these to optimize.
4. **Cheapest model that works.** Difficulty tiers map tasks to models. Start conservative, auto-downgrade based on success data. Never auto-upgrade without human approval.
5. **Fail loud.** No silent fallbacks in the runner. If validation fails, the task fails. If the DAG has cycles, it crashes. Quiet failures poison the experiment data.
6. **Investigate before acting.** Agents must verify assumptions before executing. Task prompts can include investigation questions that must be answered first. (From meta-process Pattern #28: Question-Driven Planning.)
7. **Lock specs before execution.** Task definition is frozen at dispatch time. The agent cannot weaken requirements to match a weak implementation. (From meta-process Pattern #13: Acceptance Gates.)
8. **Persist uncertainties.** When a task discovers something unknown, record it in the execution log. Downstream tasks and future runs inherit this knowledge instead of re-investigating. (From meta-process Pattern #29: Uncertainty Tracking.)

## Architecture

```
┌──────────────────────────────────────────────┐
│  OpenClaw (coordinator brain, gpt-5.2)       │
│  Writes YAML task graphs                     │
│  Reads experiment logs, applies proposals     │
│  Decides what to run next                    │
└──────────────┬───────────────────────────────┘
               │ calls
┌──────────────▼───────────────────────────────┐
│  Task Graph Runner (llm_client.task_graph)   │
│  YAML → topo sort → waves → dispatch →       │
│  validate → experiment record → checkpoint   │
└──────────────┬───────────────────────────────┘
               │ dispatches via
┌──────────────▼───────────────────────────────┐
│  llm_client agent SDKs                       │
│  acall_llm("codex", ..., mcp_servers={...})  │
│  acall_llm("claude-code", ..., cwd=...)      │
│  acall_llm("gemini/gemini-3-flash", ...)     │
└──────────────┬───────────────────────────────┘
               │ uses
┌──────────────▼───────────────────────────────┐
│  MCP Tools (170+)                            │
│  onto-canon, DIGIMON, sam_gov, twitter, etc. │
└──────────────────────────────────────────────┘
```

Three layers in the runner, built bottom-up:

| Layer | What | Lines (est.) |
|-------|------|-------------|
| Task Graph Runner | Parse YAML, topo sort, dispatch, validate, checkpoint | ~400 |
| Difficulty Router | Task difficulty → model selection from registry | ~100 |
| Self-Improvement Analyzer | Read logs, classify issues, propose fixes | ~300 |

## Data Formats

### Task Graph (YAML)

```yaml
graph:
  id: nightly_research_2026-02-17
  description: "Nightly OSINT collection and belief extraction"
  timeout_minutes: 120
  checkpoint: git_tag  # git_tag | git_commit | none

tasks:
  collect_sources:
    difficulty: 2
    agent: codex
    prompt: |
      Search sam_gov for new contracts matching the watchlist in
      ~/.openclaw/workspace/watchlist.yaml. Write results to results/sources.json.
    mcp_servers:
      - sam-gov-government
      - sam-gov-social
    working_directory: ~/projects/sam_gov
    validate:
      - type: file_exists
        path: results/sources.json
      - type: json_schema
        path: results/sources.json
        schema:
          type: array
          items:
            required: [title, url, date]
    outputs:
      sources_file: results/sources.json

  build_graph:
    difficulty: 3
    agent: codex
    depends_on: [collect_sources]
    prompt: |
      Build an ER graph from {collect_sources.outputs.sources_file}
      using corpus_prepare then graph_build_er.
      Dataset name: nightly_{date}.
    mcp_servers:
      - digimon-kgrag
    working_directory: ~/projects/Digimon_for_KG_application
    validate:
      - type: mcp_call
        server: digimon-kgrag
        tool: list_available_resources
        check: "result.graphs | length > 0"
    outputs:
      graph_name: "nightly_{date}"

  extract_beliefs:
    difficulty: 2
    agent: codex
    depends_on: [build_graph]
    prompt: |
      Export the graph {build_graph.outputs.graph_name} to onto-canon.
      Use canon_import_digimon_graph. Then run canon_extract_beliefs
      on any new evidence.
    mcp_servers:
      - digimon-kgrag
      - onto-canon
    working_directory: ~/projects/onto-canon
    validate:
      - type: sql_count
        db: ~/projects/onto-canon/onto_canon.db
        query: "SELECT count(*) FROM beliefs WHERE date(created_at) = date('now')"
        check: "> 0"

  analyze:
    difficulty: 3
    agent: claude-code
    depends_on: [extract_beliefs]
    prompt: |
      Review today's new beliefs in onto-canon. Identify tensions
      with existing beliefs using canon_find_tensions. Write a
      summary to ~/.openclaw/workspace/briefs/nightly_{date}.md.
    mcp_servers:
      - onto-canon
    validate:
      - type: file_exists
        path: ~/.openclaw/workspace/briefs/nightly_{date}.md
    outputs:
      brief: ~/.openclaw/workspace/briefs/nightly_{date}.md
```

### Experiment Record (JSONL)

Written per-task to `~/projects/data/task_graph/experiments.jsonl`:

```json
{
  "run_id": "nightly_research_2026-02-17",
  "task_id": "extract_beliefs",
  "wave": 2,
  "timestamp": "2026-02-17T03:15:22Z",

  "hypothesis": "gemini-flash can handle belief extraction at tier 2",
  "difficulty": 2,
  "model_selected": "gemini/gemini-2.5-flash",
  "agent": "codex",

  "result": {
    "status": "success",
    "duration_s": 34.2,
    "cost_usd": 0.012,
    "tokens_in": 4200,
    "tokens_out": 1800,
    "validation_results": [
      {"type": "sql_count", "passed": true, "value": 23}
    ]
  },

  "dimensions": {
    "beliefs_created": 23,
    "cost_per_belief": 0.00052,
    "extraction_time_s": 34.2
  },

  "outcome": "confirmed",
  "prior_tier": null,
  "learning": null
}
```

Failed experiment with downgrade attempt:

```json
{
  "run_id": "nightly_research_2026-02-18",
  "task_id": "analyze",
  "hypothesis": "deepseek-chat can handle analysis at tier 2 (downgraded from tier 3)",
  "difficulty": 2,
  "model_selected": "deepseek/deepseek-chat",
  "result": {
    "status": "failure",
    "duration_s": 45.1,
    "cost_usd": 0.003,
    "validation_results": [
      {"type": "file_exists", "passed": true},
      {"type": "quality_check", "passed": false, "reason": "brief under 200 words"}
    ]
  },
  "outcome": "hypothesis_rejected",
  "prior_tier": 3,
  "learning": "analysis/synthesis tasks need tier 3 minimum — deepseek produces shallow output"
}
```

### Improvement Proposal (JSONL)

Written by the analyzer to `~/projects/data/task_graph/proposals.jsonl`:

```json
{
  "proposal_id": "prop_2026-02-18_001",
  "timestamp": "2026-02-18T04:00:00Z",
  "category": "MODEL_OVERKILL",
  "task_id": "collect_sources",
  "graph_id": "nightly_research",
  "evidence": {
    "runs_analyzed": 7,
    "success_rate_at_current_tier": 1.0,
    "current_tier": 2,
    "current_model": "gemini/gemini-2.5-flash",
    "proposed_tier": 1,
    "proposed_model": "deepseek/deepseek-chat",
    "estimated_savings_per_run": 0.008
  },
  "risk": "low",
  "action": "downgrade_model",
  "auto_apply": true,
  "applied": false,
  "result": null
}
```

## Difficulty Tiers

| Tier | What | Default Model | Approx Cost |
|------|------|--------------|-------------|
| 0 | Scripted — no LLM needed | None | $0.00 |
| 1 | Simple: formatting, template fill, structured extraction from clean data | `deepseek/deepseek-chat` or local (ollama) | $0.001/call |
| 2 | Moderate: entity extraction, classification, structured analysis | `gemini/gemini-2.5-flash` | $0.01/call |
| 3 | Complex: multi-hop reasoning, synthesis, novel analysis, multi-tool composition | `anthropic/claude-sonnet-4-5-20250929` or `o4-mini` | $0.10/call |
| 4 | Agent: multi-step autonomous tool use, MCP composition, investigation | `codex` or `claude-code` SDK | $1.00/task |

The router extends llm_client's existing model registry. `get_model_for_difficulty(tier)` returns the cheapest available model at that capability level.

**Local model support**: Tier 0-1 tasks route to `ollama/llama-3.1-8b` or similar when available (Mac Mini). The model registry's `available_only=True` filter handles this — if ollama isn't running, falls back to the cheapest cloud model.

**Routing rules**:
- Task graph specifies `difficulty` per task (required field)
- Router picks cheapest model at that tier
- Self-improvement loop can propose tier changes (down only — auto-applied; up — queued for human review)
- Override: task graph can specify `model` directly to bypass the router

## Validation Framework

Validators are external checks. The runner ships with these types:

| Type | What | Params |
|------|------|--------|
| `file_exists` | File was created | `path` |
| `file_not_empty` | File exists and has content | `path`, optional `min_bytes` |
| `json_schema` | File parses as JSON matching schema | `path`, `schema` (JSON Schema) |
| `pytest` | pytest exits 0 | `path` (test file or dir), optional `markers` |
| `sql_count` | SQL query returns count matching check | `db`, `query`, `check` (e.g., `"> 0"`) |
| `mcp_call` | MCP tool returns result matching check | `server`, `tool`, `args`, `check` (jmespath) |
| `command` | Shell command exits 0 | `command` |

Validators return `{type, passed: bool, value, reason?}`. All validation results are recorded in the experiment record.

**Custom validators**: Any callable `(task_result) -> ValidationResult` can be registered. This is how project-specific checks (e.g., "onto-canon belief count increased by at least N") get added without modifying the core runner.

**Dry-run mode**: `validate(task, dry_run=True)` shows what checks would run without executing them. Agents can preview completion requirements before attempting work.

## Task Execution Lifecycle

Derived from meta-process patterns (Plan Workflow #15, Verification Enforcement #17, Question-Driven Planning #28, Uncertainty Tracking #29, Acceptance Gates #13).

### Phases

```
PENDING → SPEC_LOCKED → DISPATCHED → RUNNING → VALIDATING → COMPLETED
                                        ↓                      ↑
                                      FAILED ──(retry)──────────┘
```

**1. PENDING** — Task is defined in the graph, waiting for dependencies.

**2. SPEC_LOCKED** — All dependencies satisfied. Task spec is frozen (SHA256 hash recorded). No changes to prompt, validation criteria, or outputs after this point. This prevents the agent from weakening requirements to match a weak implementation.

**3. DISPATCHED** — Agent has been invoked via `acall_llm()`. The runner records: model selected, difficulty tier, MCP servers loaded, prompt (with template variables resolved). Agent permission modes are set automatically for headless dispatch: `permission_mode="bypassPermissions"` for claude-code, `approval_policy="never"` for codex. Override with `setdefault` — task YAML can't change this, but callers of `_execute_task` can pre-set kwargs.

**4. RUNNING** — Agent is working. The runner doesn't interfere. Timeout applies.

**5. VALIDATING** — Agent finished. Runner executes all validators. Results recorded with evidence (not just pass/fail — actual values).

**6. COMPLETED** — All validators passed. Verification evidence recorded in experiment log. Outputs registered for downstream handoff.

**6a. FAILED** — Validator(s) failed, or agent timed out/errored. Full context recorded: which validators failed, why, agent output (truncated). Runner stops the graph (no automatic retry within the same run).

### Investigation Phase (Optional)

Tasks can declare investigation questions that must be answered before the agent proceeds to implementation:

```yaml
tasks:
  build_graph:
    investigate_first:
      - "Does the corpus directory contain .txt or .json files?"
      - "Is there an existing graph for this dataset?"
    prompt: |
      After answering the investigation questions above,
      build an ER graph from the corpus...
```

The runner prepends investigation questions to the prompt. The agent's answers become part of the execution log, available to downstream tasks and the self-improvement analyzer.

### Uncertainty Log

During execution, agents may discover unknowns. These are captured in the execution record:

```json
{
  "task_id": "extract_beliefs",
  "uncertainties": [
    {
      "question": "Some entities have duplicate names across sources",
      "status": "deferred",
      "resolution": "Dedup handled by onto-canon's concept_dedup — not a blocker",
      "raised_at": "2026-02-17T03:15:22Z"
    }
  ]
}
```

Uncertainties persist across runs. The analyzer reads them to detect recurring issues (same uncertainty raised in >3 runs → `TOOL_GAP` or `DATA_QUALITY` proposal).

### Dependency Staleness Check

After each wave completes, the runner checks for stale blockers:

```python
for task in graph.tasks:
    if task.status == "pending":
        stale = [dep for dep in task.depends_on if graph.tasks[dep].status == "completed"]
        if stale and len(stale) == len(task.depends_on):
            task.status = "ready"  # All blockers resolved, promote to next wave
```

This prevents downstream tasks from staying blocked when their dependencies have completed. (From meta-process Pattern #16.)

## Self-Improvement Loop

Runs as the last task in every task graph (or as a standalone post-run step). Reads experiment records, classifies issues, writes proposals.

### Failure Taxonomy (8 categories)

| Category | Signal | Auto-fix? | Action |
|----------|--------|-----------|--------|
| `MODEL_OVERKILL` | Task succeeded N times at current tier | Yes | Propose downgrade, run as experiment |
| `MODEL_UNDERKILL` | Task failed at current tier | Yes | Bump tier +1 for next run |
| `PROMPT_DRIFT` | Agent used wrong tools or produced wrong format | Medium | Propose prompt edit (human review) |
| `VALIDATION_NOISE` | Same task, same inputs, inconsistent validation | Yes | Flag validator for review, increase sample size |
| `TOOL_GAP` | Agent described needing a capability no tool provides | No | Write to PROJECTS_DEFERRED |
| `STUCK_LOOP` | Agent timed out or made no progress | Yes | Reduce max_turns, add explicit stop conditions |
| `DATA_QUALITY` | Upstream task produced bad output that broke downstream | No | Trace to source, tighten upstream validation |
| `MEASUREMENT_ERROR` | Scorer/validator inconsistent across re-runs | Yes | Recalibrate (scorer reliability check) |

### Scorer Reliability Check

Before the analyzer trusts validation results, it periodically checks measurement stability (pattern from steno finetuning):

1. Pick a recently-succeeded task
2. Re-run validation on the same outputs (don't re-run the agent)
3. Compare: if results differ → `MEASUREMENT_ERROR`, don't trust score comparisons until fixed

This prevents false model downgrades caused by flaky validators.

### Proposal Risk Classification

| Risk | Criteria | Auto-apply? |
|------|----------|-------------|
| Low | Model downgrade with >5 consecutive successes at current tier | Yes |
| Medium | Prompt change, validation change, task restructuring | Apply on next run, revert if fails |
| High | New tool needed, tier upgrade, structural changes to task graph | Queue for human review |

### Cumulative Learning

The analyzer maintains a `model_floors.json` — the minimum difficulty tier proven to work for each (task_type, task_id) pair:

```json
{
  "collect_sources": {"floor": 1, "ceiling": 2, "last_tested": "2026-02-20", "runs": 14},
  "build_graph": {"floor": 3, "ceiling": 3, "last_tested": "2026-02-19", "runs": 7},
  "extract_beliefs": {"floor": 2, "ceiling": 2, "last_tested": "2026-02-18", "runs": 10},
  "analyze": {"floor": 3, "ceiling": 4, "last_tested": "2026-02-20", "runs": 5}
}
```

This file is the system's learned knowledge about what works. Over time:
- Floors drop as cheaper models prove capable
- Ceilings drop as expensive models prove unnecessary
- Both converge to the optimal tier per task

## Context Handoff

Task outputs flow to downstream tasks via `outputs` declarations:

```yaml
tasks:
  task_a:
    outputs:
      result_file: results/output.json
  task_b:
    depends_on: [task_a]
    prompt: "Process {task_a.outputs.result_file}"
```

The runner resolves `{task_a.outputs.result_file}` before dispatching task_b. Outputs are validated (file must exist) before handoff.

For non-file outputs (e.g., a graph name, entity count), the runner writes a `_handoff.json` per task containing all outputs. Downstream tasks can reference any field.

## Git Checkpointing

After each successful wave:

```bash
git tag "taskgraph/{graph_id}/wave_{n}_{timestamp}"
```

On wave failure:
- Failed task is logged with full context
- Successful tasks in the same wave keep their results
- Runner stops (no automatic retry of the wave)
- OpenClaw decides: retry, skip, or abort on next heartbeat

Rollback (manual, via OpenClaw):
```bash
git reset --hard "taskgraph/{graph_id}/wave_{n-1}_{timestamp}"
```

## OpenClaw Integration

OpenClaw's `run_task.py` evolves to call the task graph runner:

```python
from llm_client.task_graph import run_graph, load_graph

# Load and run
graph = load_graph("~/.openclaw/tasks/active/nightly_research.yaml")
report = await run_graph(graph)

# Report is an ExecutionReport with per-task experiment records
for task_result in report.tasks:
    print(f"{task_result.task_id}: {task_result.status} ({task_result.cost_usd:.3f})")
```

OpenClaw writes task graphs to `~/.openclaw/tasks/pending/`. The runner picks them up. Results go to the experiment log. The improvement analyzer runs. OpenClaw reads proposals on next heartbeat.

**Cost budget**: The existing `~/.openclaw/cost_log.jsonl` + $20/day cap stays. The runner checks remaining budget before each task dispatch and aborts the graph if budget is exhausted.

## File Layout

```
llm_client/
  llm_client/
    task_graph.py          # Core: parse, sort, dispatch, validate, checkpoint
    difficulty.py          # Difficulty router: tier → model selection
    analyzer.py            # Self-improvement: classify issues, propose fixes
    validators.py          # Validation framework: file, json, sql, mcp, pytest, command
  tests/
    test_task_graph.py
    test_difficulty.py
    test_analyzer.py
    test_validators.py
  docs/
    TASK_GRAPH_DESIGN.md   # This document
```

## Build Phases

### Phase 1: Task Graph Runner (~500 lines)

- `TaskGraph` dataclass: parse YAML, validate DAG (no cycles, all deps exist)
- `toposort_waves()`: group tasks into parallel execution waves
- `run_graph()`: async, dispatches each wave via `acall_llm`, runs validators
- Task lifecycle: PENDING → SPEC_LOCKED (SHA256 frozen) → DISPATCHED → RUNNING → VALIDATING → COMPLETED/FAILED
- Spec locking: hash task definition before dispatch, reject post-hoc changes
- Dependency staleness check: promote ready tasks after each wave
- Investigation phase: prepend `investigate_first` questions to agent prompt
- Uncertainty log: capture and persist unknowns discovered during execution
- `ExperimentRecord` dataclass: structured per-task result with dimensions + evidence
- `ExecutionReport`: aggregate results for the full graph run
- Validators: `file_exists`, `file_not_empty`, `json_schema`, `sql_count`, `command` + dry-run mode
- Git checkpoint after each wave
- Context handoff via `_handoff.json` + template resolution in prompts

### Phase 2: Difficulty Router (~100 lines)

- `get_model_for_difficulty(tier: int) -> str`: extends model registry
- Tier 0-4 definitions with default model mappings
- `available_only` filtering (skip models without API keys)
- Local model support via ollama detection
- Override: task-level `model` field bypasses router

### Phase 3: Self-Improvement Analyzer (~300 lines)

- `analyze_run(report: ExecutionReport) -> list[Proposal]`
- Issue classification (8 categories)
- Scorer reliability check (re-validate same outputs)
- Proposal generation with risk classification
- `model_floors.json` maintenance (cumulative learning)
- Auto-apply low-risk proposals, queue high-risk

### Phase 4: OpenClaw Integration (~100 lines)

- Evolve `run_task.py` to detect YAML task graphs vs flat tasks
- Heartbeat reads proposals, applies low-risk
- Morning brief includes execution summary + pending proposals
- Cost budget integration with existing cost_log.jsonl

## Open Design Decisions

1. **MCP server lifecycle**: Should the runner start/stop MCP servers per task, or maintain a pool? Per-task is simpler but slower. Pool is faster but risks state leakage between tasks. **Recommendation**: Per-task for Phase 1. Pool as Phase 2 optimization if startup cost is a problem.

2. **Parallel wave execution**: Should tasks within a wave run in parallel? `acall_llm_batch` supports this. **Recommendation**: Yes, parallel by default. Each task gets its own agent instance (no shared state within a wave).

3. **Template language for prompts**: Task prompts reference outputs from upstream tasks via `{task_id.outputs.key}`. Is Python f-string-style substitution enough, or do we need Jinja2? **Recommendation**: Simple `str.format_map()` for Phase 1. Jinja2 if conditionals/loops prove necessary.

4. **Experiment log location**: Currently proposed as `~/projects/data/task_graph/experiments.jsonl`. Should it be per-graph or global? **Recommendation**: Global file, filterable by `run_id` and `graph_id`. Matches llm_client's existing `calls.jsonl` pattern.

## Provenance

Patterns incorporated from three sources:

**claude_code_tooling/meta-process** (5 patterns adopted):
- Pattern #13 (Acceptance Gates) → Spec locking before dispatch
- Pattern #15 (Plan Workflow) → Task lifecycle with status tracking
- Pattern #16 (Plan Blocker Enforcement) → Dependency staleness check
- Pattern #17 (Verification Enforcement) → Mandatory validation with evidence recording
- Pattern #28 (Question-Driven Planning) → Investigation phase before execution
- Pattern #29 (Uncertainty Tracking) → Persisted uncertainty log across runs

**steno/finetuning-clones-v2** (4 patterns adopted):
- Experiment Registry → Every run is a structured experiment with hypothesis/outcome
- Scorer Reliability Validation → Validate the validators before trusting comparisons
- Multi-Dimensional Quality → Score tasks on multiple dimensions, not just pass/fail
- Hypothesis-Driven Model Routing → Model changes structured as experiments with rollback

**claude_code_tooling/METHODS.md** (principles adopted):
- External validation only (never trust agent self-reports)
- 1 instance, 1 task (no shared agent state between tasks)
- Don't explain mistakes to agents, just reset
- Git checkpoints for rollback

## References

- `~/.openclaw/bin/run_task.py` — Current flat task runner (480 lines)
- `~/projects/claude_code_tooling/meta-process/` — AI coordination patterns (26 patterns, 13 scripts)
- `~/projects/claude_code_tooling/METHODS.md` — 280 techniques catalog
- `~/projects/steno/finetuning-clones-v2/` — Experiment registry, scorer reliability, multi-dimensional quality
- `~/projects/llm_client/llm_client/models.py` — Model registry (difficulty router extends this)
- `~/projects/llm_client/llm_client/io_log.py` — I/O logging (experiment records extend this)
- `~/projects/project-meta/vision/MULTI_AGENT_ARCHITECTURE.md` — Multi-agent coordination design
- `~/projects/project-meta/vision/OPENCLAW_MASTER.md` — OpenClaw ecosystem map
