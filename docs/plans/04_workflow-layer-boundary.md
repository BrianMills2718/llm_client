# Plan 04: Workflow Layer Boundary

**Status:** Complete
**Type:** design
**Priority:** Medium
**Blocked By:** None
**Blocks:** durable workflow implementation inside or above `llm_client`

---

## Gap

**Current:** `llm_client` has a simple DAG runner in `task_graph.py`, but the
ecosystem wants off-the-shelf durable workflow capabilities such as checkpoint,
resume, human-in-the-loop approvals, and long-lived state. Those requirements
are currently documented only across ADRs, uncertainties, and notebooks.

**Target:** one explicit workflow-boundary plan defines what belongs in the
shared substrate, what belongs in a LangGraph-backed workflow layer, and what
must not be added to `task_graph`.

**Why:** durable orchestration should get a real home without turning
`llm_client` into an accidental custom workflow engine.

---

## References Reviewed

- `docs/ECOSYSTEM_TOP_DOWN_ARCHITECTURE.md` - workflow-layer target boundary
- `docs/ECOSYSTEM_UNCERTAINTIES.md` - `task_graph` versus LangGraph
- `docs/adr/0010-cross-project-runtime-substrate.md` - substrate/workflow split
- `docs/TASK_GRAPH_DESIGN.md` - current simple-runner contract
- `docs/notebooks/01_ecosystem_runtime_eval_roadmap.ipynb` - provisional
  workflow phase notes

---

## Files Affected

- docs/plans/04_workflow-layer-boundary.md (create)
- docs/plans/CLAUDE.md (modify)
- docs/plans/01_master-roadmap.md (modify)
- AGENTS.md (modify)
- CLAUDE.md (modify)
- pyproject.toml (modify)
- README.md (modify)
- llm_client/workflow_langgraph.py (create)
- llm_client/prompt_assets/shared/workflow/summary_approval_draft/v1/manifest.yaml (create)
- llm_client/prompt_assets/shared/workflow/summary_approval_draft/v1/template.yaml (create)
- llm_client/prompt_assets/shared/workflow/summary_approval_revision/v1/manifest.yaml (create)
- llm_client/prompt_assets/shared/workflow/summary_approval_revision/v1/template.yaml (create)
- tests/test_workflow_langgraph.py (create)

---

## Program Guardrails

1. No durable-workflow features get added to `task_graph` under this plan.
2. The workflow layer must consume `llm_client` for LLM calls, prompt refs,
   budgets, and observability rather than inventing parallel plumbing.
3. No hidden state or silent fallbacks: checkpoints, resumes, approvals, and
   failures must be explicit artifacts or state transitions.
4. The first proving slice must be one real workflow, not an abstract
   framework-first rewrite.

---

## Overall Definition Of Done

This program is done only when all of the following are true:

1. The durable workflow layer has an explicit home and contract.
2. `task_graph` remains intentionally limited to simple DAG
   dispatch/validation/logging.
3. A LangGraph-backed proving slice shows checkpoint/resume or approval-state
   behavior while still using `llm_client` observability and prompt identity.
4. The coexistence rule between `task_graph` and the workflow layer is written
   down clearly enough that future work does not reopen the boundary by habit.

---

## Long-Term Phases

### Phase 1: Define The Workflow Boundary

**Status:** completed

**Purpose:** Specify the boundary before any durable-workflow implementation
lands.

**Input -> Output:** scattered workflow guidance -> one explicit contract for
state, checkpoints, approvals, observability, and artifact flow

**Passes if:**

- the contract says what lives in workflow state versus observability
- `llm_client` call requirements (`task`, `trace_id`, `max_budget`,
  `prompt_ref`) are preserved inside workflow nodes
- the plan names what `task_graph` must not absorb

**Fails if:**

- the plan leaves room for `task_graph` to keep growing into durable
  orchestration
- the workflow layer gets its own logging or prompt-resolution stack

### Phase 2: Prove One Real LangGraph-Backed Slice

**Status:** completed

**Purpose:** Earn the boundary with a real workflow instead of a speculative
framework layer.

**Input -> Output:** boundary contract -> one real multi-step workflow with
durability features

**Passes if:**

- one workflow proves at least one durable feature such as resume, checkpoint,
  or explicit approval state
- workflow nodes use `llm_client` calls and shared observability
- failures and resumes are explicit and inspectable

**Fails if:**

- the first slice requires a big-bang migration of existing projects
- the proof depends on hidden state or ad hoc local glue

**Proven slice:** `llm_client.workflow_langgraph` now provides one concrete
approval-gated summarization workflow:

- draft summary with `llm_client.call_llm(...)`
- pause with LangGraph `interrupt(...)`
- resume with explicit approval/revision payload
- revise or finalize while preserving `prompt_ref`, `trace_id`, and
  `max_budget`

**Pass evidence:**

- `tests/test_workflow_langgraph.py`
- focused workflow/eval/model public-surface verification

### Phase 3: Decide Task-Graph Coexistence

**Status:** completed

**Purpose:** Make the long-term relationship between `task_graph` and the
workflow layer explicit.

**Input -> Output:** two orchestration surfaces -> clear coexistence or adapter
rule

**Passes if:**

- `task_graph` either stays intentionally simple or has a clearly bounded
  adapter role
- no one needs to guess which workflow tool to use for a new project

**Fails if:**

- both orchestration systems keep gaining overlapping features without a rule

**Coexistence rule:**

- use `llm_client.task_graph` for simple YAML DAG dispatch, validation, and
  logging
- use `llm_client.workflow_langgraph` when you need durable workflow state,
  explicit approval pause/resume, or checkpoint-backed execution
- do not add those durable capabilities to `task_graph`

---

## Closeout

This plan is complete.

The first workflow slice stays intentionally narrow:

1. one concrete LangGraph-backed approval workflow,
2. no generic workflow framework abstraction,
3. no new durable features added to `task_graph`.

Future workflow work should build from this proven boundary rather than
reopening the `task_graph` question by habit.
