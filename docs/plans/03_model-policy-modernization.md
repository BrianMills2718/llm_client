# Plan 03: Model Policy Modernization

**Status:** Complete
**Type:** implementation
**Priority:** High
**Blocked By:** None
**Blocks:** future data-driven model selection, easier registry audits, safer model-cost refreshes

---

## Gap

**Current:** `llm_client/models.py` embeds the default model registry and task
profiles directly in Python source. That keeps the package self-contained, but
it makes policy updates harder to audit, encourages hand-maintained market
intelligence drift, and mixes data maintenance with selection logic.

**Target:** model-selection logic stays in Python, but the default registry and
task profiles become explicit data with typed loading, parity tests, and a
clear future path toward observed-performance overlays.

**Why:** The next 10x value in `llm_client`’s model policy comes from the local
observability data it already collects. That requires the static registry to be
easy to inspect, diff, and refresh without rewriting selection logic every
time provider pricing or availability changes.

---

## References Reviewed

- `llm_client/models.py:1-220` - embedded default registry and task profiles
- `llm_client/difficulty.py:1-220` - adjacent model-policy surface with static tiers
- `tests/test_models.py:1-430` - current model-selection behavior coverage
- `tests/test_model_selection.py:1-120` - selection-task integration coverage
- `docs/ECOSYSTEM_UNCERTAINTIES.md:73-95` - current model-policy uncertainty

---

## Files Likely Affected

- docs/plans/03_model-policy-modernization.md (create)
- docs/plans/CLAUDE.md (modify)
- docs/ECOSYSTEM_UNCERTAINTIES.md (modify)
- llm_client/models.py (modify)
- llm_client/data/default_model_registry.json (create)
- llm_client/difficulty.py (modify, later phase only if justified)
- pyproject.toml (modify if packaged data files are added)
- tests/test_models.py (modify)
- tests/test_model_selection.py (modify if task-facing behavior changes)

---

## Program Guardrails

1. No behavior-changing policy rewrite may happen before the static registry is
   extracted into an auditable data shape and parity-tested.
2. The first slice must be no-behavior-change.
3. Selection semantics must stay fail-loud: unknown tasks, invalid data files,
   and malformed registry entries should raise immediately.
4. Observed-performance overlays must be additive first. They may demote or
   annotate candidates before they replace static ranking rules.
5. No network dependency is allowed in the selection path. Any future provider
   sync must happen offline or at build/update time, not at call time.

---

## Overall Definition Of Done

This program is done only when all of the following are true:

1. The default registry and task profiles are stored as explicit data rather
   than embedded Python literals.
2. `models.py` owns selection logic and typed loading, not large static tables.
3. The default registry is parity-tested against current behavior before any
   data-driven ranking changes land.
4. The path for future observed-performance guidance is documented and does not
   require another registry rewrite.
5. The remaining static-policy surfaces (`models.py`, `difficulty.py`) are
   clearly distinguished as either strategic or legacy.

---

## Long-Term Phases

### Phase 1: Extract The Static Registry Into Packaged Data

**Purpose:** Move embedded default models/tasks into an auditable data file
without changing selection behavior.

**Input -> Output:** Python literals in `models.py` -> packaged registry data
loaded through one typed loader

**Passes if:**

- `get_model()`, `list_models()`, and `query_performance()` behave the same
- invalid packaged data fails loudly during load
- focused tests prove registry parity for the current built-in tasks

**Fails if:**

- selection results drift for existing built-in tasks
- data loading adds hidden fallbacks or silent coercions

### Phase 2: Separate Static Policy From Observed-Performance Overlay

**Purpose:** Make it explicit which parts of model choice are curated defaults
and which parts come from observed cost/error/latency behavior.

**Input -> Output:** one mixed policy surface -> static registry plus observed
performance overlay

**Passes if:**

- static task requirements and preferences remain inspectable
- performance-based demotion stays optional and testable in isolation
- the overlay can be reasoned about without reading raw SQL first

**Fails if:**

- the overlay silently overrides task requirements
- callers cannot tell whether a choice was static or empirical

**Status:** completed

The additive overlay checkpoint is now proven:

- `ModelPerformanceObservation` and `PerformanceOverlayDecision` make the
  empirical overlay explicit data rather than hidden list mutation
- `_apply_performance_overlay(...)` applies only after static candidates are
  chosen
- focused tests prove demotion metadata is inspectable and neutral behavior is
  preserved when there is no performance data

#### Phase 2A: Static Candidate Pipeline

**Status:** completed

**Purpose:** Make the static policy path explicit before changing any empirical
overlay behavior.

**Input -> Output:** mixed filtering/sorting logic inside public functions ->
named static selection helpers reused by `get_model()` and `list_models()`

**Passes if:**

- static filtering and prefer-order sorting are implemented through one shared
  helper path
- `list_models()` and `get_model()` keep the same ordering behavior
- the performance overlay still applies only after static candidates are chosen

**Fails if:**

- static selection order drifts
- helper extraction changes availability filtering or task validation behavior

### Phase 3: Reassess `difficulty.py`

**Purpose:** Decide whether the difficulty router is strategic guidance,
legacy compatibility, or a candidate for deprecation.

**Input -> Output:** ambiguous second policy surface -> explicit status

**Passes if:**

- `difficulty.py` has a documented role or deprecation path
- model-governance guidance does not point callers at two competing systems

**Fails if:**

- both policy systems keep evolving independently

**Status:** completed

**Decision:** `difficulty.py` remains a frozen compatibility-guidance layer for
the simple `task_graph` runner and analyzer/model-floor logic. It is no longer
taught as the preferred model-governance path for new project code.

**Implications:**

- `llm_client.difficulty` remains the stable module namespace
- top-level root exports stay compatibility-only and already warn
- new product-facing code should prefer `get_model(task)` and
  `selection_task`-style config
- future removal or deeper deprecation must wait on separate decisions about
  `task_graph` and analyzer fate

---

## First Thin Slice

**Recommended first slice:** Phase 1 only.

1. Add packaged registry data for built-in models and task profiles.
2. Load it through one typed internal loader in `models.py`.
3. Keep current behavior identical.
4. Add parity tests that lock current built-in task winners and list ordering.

This is the smallest real slice because it improves auditability without
changing policy semantics.

### First Slice Status

**Status:** completed

The first no-behavior-change extraction is now proven:

- the built-in registry moved to packaged data in
  `llm_client/data/default_model_registry.json`
- `models.py` now loads the packaged defaults through a typed fail-loud loader
- focused parity verification passed in `tests/test_models.py` and
  `tests/test_model_selection.py`
- static candidate selection is now explicit and reused before any
  performance-based demotion step

### Closeout

The default no-behavior-change modernization tranche is now complete enough to
close this plan before any ranking changes:

- the static registry is packaged data
- the static candidate path is explicit
- the empirical overlay is inspectable
- `difficulty.py` has an explicit compatibility role instead of ambiguous
  parallel-policy status

No further behavior-changing work should proceed under this plan until a
separate benchmark-backed slice defines how empirical ranking changes will be
judged.
