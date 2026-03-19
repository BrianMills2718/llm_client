# Plan 05: Eval Boundary Cleanup

**Status:** Complete
**Type:** implementation
**Priority:** Medium
**Blocked By:** None
**Blocks:** future eval/package-boundary cleanup

---

## Gap

**Current:** `llm_client` correctly owns shared observability and experiment
logging, but repo-local eval helpers such as scoring, analyzer, and
experiment-eval still blur the line between runtime substrate and evaluation
layer. Some public-surface cleanup is already done, but the module-level fate
of these surfaces is still scattered across plan notes and audits.

**Target:** one explicit eval-boundary plan defines which eval surfaces remain
optional inside `llm_client`, which should migrate toward `prompt_eval` or a
sibling package, and how compatibility is preserved while the shared
observability backend remains authoritative.

**Why:** the shared observability substrate is strategic; eval-policy sprawl is
not. Future cleanup should be deliberate instead of happening piecemeal through
export tweaks alone.

---

## References Reviewed

- `docs/plans/02_client-boundary-hardening.md` - current public-surface and
  boundary-hardening slices
- `docs/PUBLIC_SURFACE_AUDIT.md` - namespace-level audit results
- `docs/adr/0008-task-graph-evaluation-contract-boundary.md` - current
  task-graph/eval contract
- `docs/ECOSYSTEM_TOP_DOWN_ARCHITECTURE.md` - substrate versus eval layer
- `docs/ECOSYSTEM_UNCERTAINTIES.md` - cross-project eval/prompt-eval boundary
- `prompt_eval` ADRs 0001 and 0002 - prompt-eval substrate contract and run
  family mapping

---

## Files Affected

- docs/plans/05_eval-boundary-cleanup.md (create)
- docs/plans/CLAUDE.md (modify)
- docs/plans/01_master-roadmap.md (modify)
- AGENTS.md (modify)
- CLAUDE.md (modify)
- llm_client/experiment_summary.py (create)
- llm_client/experiment_eval.py (modify)
- llm_client/observability/experiments.py (modify)
- llm_client/cli/experiments.py (modify)
- tests/test_experiment_summary.py (create)
- tests/test_experiment_eval.py (modify indirectly through compatibility path)
- tests/test_experiment_log.py (existing verification target)
- tests/test_cli_experiments.py (existing verification target)

---

## Program Guardrails

1. Shared observability, run/item/aggregate logging, and prompt provenance stay
   in `llm_client`.
2. Export cleanup alone does not count as boundary cleanup; module-level fate
   must be explicit.
3. No eval-module move may happen without compatibility notes and targeted
   downstream usage evidence.
4. `prompt_eval` should consume the substrate rather than merge wholesale into
   `llm_client`.

---

## Overall Definition Of Done

This program is done only when all of the following are true:

1. `llm_client` clearly owns shared observability and experiment plumbing.
2. Scoring, analyzer, and experiment-eval helpers are either explicitly
   optional inside `llm_client` or moved behind a clearer sibling boundary.
3. Public docs stop teaching eval helpers as equal peers of the runtime
   substrate.
4. Any physical move is preceded by a compatibility-preserving module and
   import plan.

---

## Long-Term Phases

### Phase 1: Audit Eval Surfaces And Coupling

**Purpose:** Measure what exists before changing any module boundary.

**Input -> Output:** scattered eval helpers -> explicit inventory of surfaces,
downstream imports, and dependency direction

**Passes if:**

- the current eval-related surfaces are classified by role
- downstream usage is measured before any move/removal claim
- the audit distinguishes shared observability from eval-specific policy

**Fails if:**

- planning treats all eval code as equally movable
- removal is proposed without a measured migration path

**Status:** completed

The audit evidence now exists across:

- `docs/PUBLIC_SURFACE_AUDIT.md` for scoring/experiment-eval top-level usage
- live module-namespace scans showing external usage for `task_graph` and
  analyzer, but not for the scoring/experiment-eval package-root shims
- the new `experiment_summary` seam, which proves that observability-owned
  summary bookkeeping can live outside the broader eval-review module

### Phase 2: Define The Stable Optional Boundary

**Purpose:** Decide what can remain as optional `llm_client` modules without
pretending they are core substrate.

**Input -> Output:** mixed package identity -> clear optional eval namespace
rule

**Passes if:**

- docs and import guidance point callers at module namespaces rather than
  package-root convenience exports
- the plan distinguishes optional-in-repo modules from future move candidates

**Fails if:**

- shared observability gets treated as part of the same move candidate group

**Status:** completed

Current explicit boundary:

- shared observability, run/item/aggregate lifecycle, prompt provenance, and
  cross-run comparisons remain core `llm_client`
- summary bookkeeping used by observability and CLI detail views now lives in
  `llm_client.experiment_summary`
- rubric review, deterministic checks, gate policy, and triage remain in
  `llm_client.experiment_eval` as optional eval helpers
- analyzer, scoring, and task-graph modules remain separate optional surfaces
  with explicit stable module namespaces and their own boundary questions

**Decision earned in this phase:**

- `llm_client.experiment_summary` stays the shared summary seam used by core
  observability and reporting surfaces
- `llm_client.experiment_eval` stays an optional in-repo eval module with a
  stable module namespace
- `llm_client.scoring` stays an optional in-repo eval module with a stable
  module namespace
- `llm_client.task_graph` and `llm_client.analyzer` stay stable optional
  module namespaces while workflow/analyzer architecture decisions remain
  separate
- docs and module docstrings now label these surfaces as optional rather than
  peers of the core runtime substrate

### Phase 3: Prove One Compatibility-Preserving Move Or Freeze Decision

**Purpose:** Earn the boundary with one real slice, not a blanket declaration.

**Input -> Output:** one ambiguous eval group -> one explicit outcome

**Passes if:**

- one module group either gets a compatibility-preserving namespace cleanup or
  an explicit freeze decision
- existing behavior stays the same while import guidance gets clearer

**Fails if:**

- the slice mixes code movement, behavior changes, and package-root cleanup in
  one pass

**Status:** completed (first proving slice)

**Proven slice:** extract shared outcome/adoption summary helpers into
`llm_client.experiment_summary` and make both observability and CLI detail
views depend on that seam instead of on `llm_client.experiment_eval`.

**Pass evidence:**

- `llm_client.observability.experiments` no longer imports
  `llm_client.experiment_eval` for run-summary bookkeeping
- `llm_client.experiment_eval` still re-exports the same summary helpers for
  compatibility
- focused verification passed in:
  - `tests/test_experiment_summary.py`
  - `tests/test_experiment_eval.py`
  - `tests/test_experiment_log.py`
  - `tests/test_cli_experiments.py`

---

## First Thin Slice

**Recommended first slice:** Phase 1 only.

1. Use the existing public-surface audit as input.
2. Perform module-level usage audits where evidence is still missing.
3. Write down which eval groups are optional-in-repo versus likely move
   candidates.

This is the smallest real slice because boundary cleanup without usage evidence
would just be churn.

## Closeout

This plan is complete for the current boundary-hardening tranche.

Remaining future work is intentionally deferred:

1. keep `experiment_summary` as the shared summary seam unless a stronger home
   emerges,
2. revisit physical moves for scoring/experiment-eval only if a sibling eval
   package earns itself with a compatibility-preserving migration plan,
3. revisit `task_graph` / `analyzer` module fate only through the workflow and
   analyzer architecture plans, not through ad hoc public-surface cleanup.
