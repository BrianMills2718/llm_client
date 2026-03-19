# LLM Client

`llm_client` is a runtime substrate/control plane for multi-provider LLM calls,
observability, budgets, prompt identity, and agent routing.

## Repo Execution Contract

The canonical control surface for work in this repo is:

- [docs/plans/01_master-roadmap.md](docs/plans/01_master-roadmap.md) (Programs A/B/D complete, C complete)
- [docs/plans/02_client-boundary-hardening.md](docs/plans/02_client-boundary-hardening.md)
- [docs/plans/03_model-policy-modernization.md](docs/plans/03_model-policy-modernization.md)
- [docs/plans/04_workflow-layer-boundary.md](docs/plans/04_workflow-layer-boundary.md)
- [docs/plans/05_eval-boundary-cleanup.md](docs/plans/05_eval-boundary-cleanup.md)

### Required Working Mode

1. Anchor every code change to the master roadmap and one child plan.
2. Define acceptance criteria upfront before editing code.
3. Work in thin, independently verifiable slices.
4. After one slice passes, continue immediately to the next unblocked slice.
5. Do not stop just to say "I completed one step" when the roadmap already
   makes the next step clear.
6. Stop only for:
   - a real blocker,
   - a user reprioritization,
   - a design decision that cannot be resolved from repository context.
7. Update the roadmap/child plan when the default next step changes.

### Priority Order

Unless the user changes priority, follow:

1. Preserve the completed Program A/B/C/D boundaries; do not invent new
   cleanup slices without fresh evidence.
2. Behavior-changing model-policy work only when a benchmark-backed plan
   exists.
3. New work only when backed by: a benchmark-backed model-ranking plan,
   a concrete new workflow use case, or a live boundary leak from bug work
   or downstream usage.

## Architecture Boundary

**Core substrate** (stable, first-class):
- Call boundary: 14 functions (7 sync + 7 async)
- Mandatory metadata: `task=`, `trace_id=`, `max_budget=` on every call
- Observability: JSONL + SQLite logging, experiment runs, cost tracking
- Prompt identity: YAML/Jinja2 templates with provenance
- Agent SDK routing: `claude-code`, `codex`, `openai-agents/*`
- Model registry: task-based selection, deprecated-model warnings

**Optional runtime** (stable modules, not core — do not grow):
- `mcp_agent` — MCP tool-calling loops, artifact contracts, stagnation detection
- `workflow_langgraph` — LangGraph-backed durable workflows
- `tool_runtime_common` — shared tool-call helpers

**Optional eval layer** (stable modules, not core — do not grow):
- `scoring` — rubric-based LLM-as-judge
- `experiment_eval` — deterministic checks, gate policies
- `analyzer` — post-run failure classification
- `task_graph` — simple YAML DAG runner (not a workflow engine)
- `difficulty` — frozen compatibility layer for task_graph

## Required kwargs (mandatory on every call)

| Kwarg | Purpose | Example |
|-------|---------|---------|
| `task=` | What kind of work (tags observability DB) | `"extraction"`, `"synthesis"` |
| `trace_id=` | Correlates all calls in a unit of work (supports hierarchy via `/`) | `"sam_gov_research_abc123"` |
| `max_budget=` | Cost limit in USD for this trace (0 = unlimited) | `0`, `1.0`, `5.0` |

Omitting any of these raises `ValueError`.

## API Reference

For detailed API documentation, usage examples, structured output routing,
agent SDK details, MCP agent loops, streaming, batch operations, cost
dashboard CLI, model registry, and all other reference material, see
**[docs/API_REFERENCE.md](docs/API_REFERENCE.md)**.

## Multi-Agent Coordination

This repo uses worktree-based isolation for concurrent AI instances.

**Before starting work:**
1. Check existing claims: `python scripts/meta/worktree-coordination/check_claims.py --list`
2. Claim your work: `python scripts/meta/worktree-coordination/check_claims.py --claim --feature <name> --task "description"`
3. Create a worktree: `make worktree` (or `git worktree add worktrees/plan-N-desc`)
4. Work in the worktree, not the main directory

**Before committing:**
- Commits must use prefixes: `[Plan #N]`, `[Trivial]`, or `[Unplanned]`
- Release claims when done: `python scripts/meta/worktree-coordination/check_claims.py --release`

**Check for messages from other instances:**
`python scripts/meta/worktree-coordination/check_messages.py`
