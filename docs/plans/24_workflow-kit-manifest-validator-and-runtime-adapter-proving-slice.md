# Plan 24: Workflow Kit Manifest, Validator, and Runtime Adapter Proving Slice

**Status:** Cancelled
**Type:** design
**Priority:** N/A — see redirect below
**Blocked By:** N/A
**Blocks:** N/A

> **Redirected (2026-04-04)**: After strategic review, the manifest-validator framing was
> solving the wrong problem for Brian's ecosystem. The correct problem is **execution strategy
> swappability** (make different execution approaches — single LLM call, map-reduce, agent SDK,
> critique loop — swappable at the task boundary, the way llm_client makes providers swappable).
>
> **Where this work moved:**
> - Architectural decision: `project-meta/docs/ops/ADR-2026-04-04-workflow-portability-revised-execution-strategies.md`
> - Deferred implementation: `~/projects/PROJECTS_DEFERRED/workflow_portability_swappable_execution.md`
> - Framework research: `project-meta/research_texts/agent_ecosystems/2026-04-04_workflow_pipeline_framework_landscape.md`
>
> **Why the framing changed**: The plan was inspired by Journey Kits (multi-tenant distribution)
> but Brian's ecosystem needs cross-strategy swappability, not cross-organization distribution.
> The manifest validator would govern content that doesn't exist yet. The execution strategy
> layer belongs in `agentic_scaffolding` (where EvaluatorOptimizerLoop already lives), not
> `llm_client`. Trigger for implementation: second project needs to swap strategies and
> copy-pastes stage code to do it.
>
> The original plan text is preserved below for reference.

---

## Gap

**Current:** `llm_client` already owns:

- LLM execution and observability,
- direct model calls,
- Claude/Codex agent SDK routing,
- one narrow LangGraph-backed workflow proof,
- prompt identity and budget propagation.

Separately, `project-meta` now defines a proposed `workflow_kit` manifest and a
package-level architecture for portable workflow packaging, but that proposal
is still only documentation. There is no runtime-facing validator, no local
adapter story, and no proving slice that shows how a package runs truthfully on
top of `llm_client`.

**Target:** Define the first real `llm_client`-owned slice for portable
workflow packaging:

1. a runtime-facing manifest model and validator;
2. a truthful runtime-support contract based on capabilities, not slogans;
3. one proving-slice workflow kit that runs through `llm_client`;
4. one explicit adapter story for multiple runtimes;
5. explicit non-goals: no public registry, no community/discovery layer, no
   enterprise resource-admin UX.

**Why:** If workflow packaging is real, the runtime-facing half belongs closest
to the execution substrate, not in prose alone. But `llm_client` should only
own the substrate-facing pieces. It should not absorb the registry/community
product layer or become an accidental marketplace.

---

## References Reviewed

- `README.md` — current `llm_client` public boundary and responsibilities
- `docs/ECOSYSTEM_TOP_DOWN_ARCHITECTURE.md` — substrate/workflow boundary and anti-patterns
- `docs/plans/04_workflow-layer-boundary.md` — workflow-layer coexistence contract
- `llm_client/workflow_langgraph.py` — current durable workflow proving slice
- `docs/guides/codex-integration.md` — current runtime-specific Codex execution surface
- `/home/brian/projects/project-meta/research_texts/agent_ecosystems/2026-04-04_journey_kits_workflow_portability_notes.md` — analyzed Journey/Journey Kits packaging concept and identified the manifest-first requirement
- `/home/brian/projects/project-meta/docs/ops/ADR-2026-04-04-workflow-kits-capability-manifests-and-runtime-portability.md` — current ecosystem architecture proposal for workflow kits
- `/home/brian/projects/project-meta/vision/schemas/workflow_kit_manifest_v0.schema.json` — first machine-readable package schema
- `/home/brian/projects/static_pipeline/README.md` — concrete local example of workflow-oriented control
- `/home/brian/projects/static_pipeline/docs/planning/END_TO_END_PLANNING.md` — config-driven stage registry and workflow definitions
- Anthropic, *Building effective agents* — workflow vs agent distinction
- Journey homepage and Claude setup pages:
  - https://www.journeykits.ai/
  - https://www.journeykits.ai/setup/claude
- CrewAI Flows documentation — external confirmation that workflow-vs-agent split is a common architecture pattern

---

## Files Affected

> Tentative; design-only plan, so some remain prospective.

- `docs/plans/24_workflow-kit-manifest-validator-and-runtime-adapter-proving-slice.md` (create)
- `docs/plans/CLAUDE.md` (modify)
- `KNOWLEDGE.md` (modify)
- `llm_client/workflow_kits/__init__.py` (create)
- `llm_client/workflow_kits/models.py` (create)
- `llm_client/workflow_kits/validator.py` (create)
- `llm_client/workflow_kits/runtime_support.py` (create)
- `llm_client/cli/workflow_kits.py` (create or extend existing CLI)
- `tests/test_workflow_kit_manifest.py` (create)
- `tests/test_workflow_kit_validator.py` (create)
- `tests/test_workflow_kit_runtime_support.py` (create)
- `tests/test_workflow_kit_examples.py` (create)
- `examples/workflow_kits/summary_approval/kit.yaml` (create)
- `examples/workflow_kits/summary_approval/README.md` (create)
- `examples/workflow_kits/summary_approval/overlay.local.example.yaml` (create)

---

## Target Architecture

`llm_client` should own the **runtime-facing half** of workflow packaging:

1. manifest loading and validation;
2. runtime-family support declarations;
3. capability vocabulary used by execution adapters;
4. local verification hooks and example packages;
5. integration with existing observability and workflow surfaces.

`llm_client` should **not** own:

1. public package discovery/registry;
2. community reputation or moderation;
3. org/team administration UX;
4. secret storage backends.

Those are product/community layers above the substrate boundary.

---

## Current Implementation Slice

The first slice should prove only this:

1. a workflow kit manifest can be parsed and validated;
2. the validator can reject portability claims that are not backed by required
   capabilities;
3. a single example workflow kit can run through one or more supported
   runtimes truthfully;
4. one local/private overlay pattern can express resource bindings without
   embedding secrets into the package itself.

The first slice should **not** attempt:

1. remote installation from a public registry;
2. generic auto-adaptation to every runtime;
3. full package publishing UX;
4. broad agent-kit support;
5. generalized community learnings ingestion.

---

## Plan

### Steps

1. Define the runtime-facing manifest contract inside `llm_client`.
2. Rename the earlier ambiguous “control mode” concept to a clearer term:
   prefer `autonomy_profile` or `control_policy`, with a small enum for the
   manifest even if the underlying reality is a spectrum.
3. Implement a validator that checks:
   - schema validity,
   - referenced files exist,
   - runtime support claims are structurally coherent,
   - required capabilities match the claimed supported runtimes.
4. Implement a small runtime-support model:
   - runtime family,
   - support level (`verified`, `experimental`, `adapter_required`),
   - required capabilities,
   - optional capabilities,
   - adapter reference.
5. Create one real example package around the existing summary-approval
   workflow in `workflow_langgraph.py`.
6. Add one local/private overlay example that supplies resource-binding
   configuration without storing secrets in the shared package.
7. Expose a CLI surface for validation and explanation only:
   - `python -m llm_client workflow-kits validate ...`
   - `python -m llm_client workflow-kits explain ...`
8. Delay installer/registry work until the proving slice is trusted.

---

## Required Tests

### New Tests (TDD)

| Test File | Test Function | What It Verifies |
|-----------|---------------|------------------|
| `tests/test_workflow_kit_manifest.py` | `test_manifest_parses_minimal_valid_example` | a valid example manifest loads into typed models |
| `tests/test_workflow_kit_validator.py` | `test_validator_rejects_missing_required_section` | malformed manifests fail loudly |
| `tests/test_workflow_kit_validator.py` | `test_validator_rejects_unsupported_portability_claim` | runtime support claims are checked against required capabilities |
| `tests/test_workflow_kit_runtime_support.py` | `test_direct_llm_requires_adapter_for_shell_based_workflow` | capability-based runtime truthfulness is enforced |
| `tests/test_workflow_kit_examples.py` | `test_summary_approval_example_is_valid` | the proving-slice package stays valid over time |

### Existing Tests (Must Pass)

| Test Pattern | Why |
|--------------|-----|
| `tests/test_workflow_langgraph.py` | existing workflow proof must not regress |
| `tests/test_agents.py` | Codex/Claude runtime surfaces are part of the adapter story |
| `tests/test_client.py` | public substrate behavior must remain stable |
| `tests/test_public_surface.py` | new package surface must integrate truthfully with the public API |

---

## Acceptance Criteria

- [ ] One `llm_client` plan clearly states the ownership boundary for workflow-kit tooling.
- [ ] A typed runtime-facing manifest model exists in `llm_client`.
- [ ] The validator can reject fake portability claims, not just malformed YAML/JSON.
- [ ] One real workflow kit example exists for the current summary-approval slice.
- [ ] One local/private overlay example exists for resource bindings.
- [ ] The first CLI surfaces are validation/explanation only, not installer/registry logic.
- [ ] Existing workflow and runtime tests still pass.

---

## Failure Modes

| Failure | How to detect | How to fix |
|---------|--------------|-----------|
| `llm_client` starts absorbing registry/community concerns | new files or docs drift toward publishing/search/reputation/admin logic | keep only runtime-facing manifest, validator, and adapters in scope |
| Portability claims stay slogan-based | manifests only list runtimes without capabilities or support levels | require capability-backed runtime support validation |
| The package layer duplicates LangGraph or task-graph orchestration | new code reimplements workflow engines instead of packaging a workflow definition | package existing workflow surfaces rather than rebuilding orchestration |
| The first example is too complex | example needs multiple services and hides the core contract problem | keep the proving slice small and local, using the summary-approval workflow |
| Overlay and package boundaries blur | shared manifest starts carrying private secret material | keep private bindings in a separate overlay example and validate that secret fields are references only |
| `autonomy_profile` becomes an overfit taxonomy | too many categories or unclear semantic differences | keep the enum narrow, document it as a pragmatic projection of a spectrum |

---

## Notes

### Recommendation on terminology

Use `workflow kit` for the packaged artifact for now.

Use `agent runtime` for Codex, Claude Code, direct LLM, MCP-loop, or other
execution environments.

Use `autonomy_profile` or `control_policy` inside runtime/package docs rather
than the earlier `control mode` phrase. The underlying reality is a spectrum,
but manifests may still need a bounded set of descriptive categories.

### Why `llm_client` is the likely home

This work looks like the natural extension of `llm_client` because it already
owns:

- plain LLM execution,
- agent SDK routing,
- prompt identity,
- budgets,
- observability,
- and a narrow durable workflow slice.

What is being proposed here is not “a new workflow engine.” It is a packaging
and runtime-truthfulness layer for things that already run on top of
`llm_client`.

### Why not just use `static_pipeline`

`static_pipeline` is a strong proof of workflow-oriented execution, especially:

- registry-driven stage definitions,
- explicit workflow selection,
- conditional and parallel steps.

But it is an application/workflow implementation, not the cross-project
packaging substrate. It is a valuable proving example, not the canonical home
for the portable manifest layer.

### Why not registry first

Journey is useful proof that discovery and installation are desirable, but it
does not remove the need for a local truthful contract. A registry before a
validator simply multiplies weak claims.

---

## Open Questions

1. Should `autonomy_profile` be the canonical manifest field name, or is
   `control_policy` clearer?
2. Should the first example package wrap `workflow_langgraph.py` only, or also
   include one simpler direct-LLM workflow example for contrast?
3. Should overlays be YAML manifests parallel to the kit, or a separate local
   config format?
4. Should `llm_client` own only validation and models, with execution adapters
   in a later repo, or should it own the first adapter slice too?
5. At what point should the future `agent kit` concept be separated from the
   `workflow kit` concept?

The plan intentionally leaves these as explicit uncertainties rather than
quietly choosing them through implementation inertia.
