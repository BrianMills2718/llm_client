# Plan 11: Program E Module Size Reduction

**Status:** In Progress
**Type:** implementation
**Priority:** High
**Blocked By:** None
**Blocks:** completion of Program E in [06_simplification-and-observability.md](./06_simplification-and-observability.md)

---

## Gap

**Current:** Program E's feature slices are materially ahead of its closeout
status. Langfuse callback wiring, replay/diff tooling, JSONL rotation, and the
models CLI already exist, but the program still fails its primary
maintainability criterion because several modules remain far above the plan's
size targets.

**Target:** reduce the remaining oversized modules into clearer, narrower
submodules until Program E's size/composition criteria are honestly satisfied,
or explicitly re-scope the criteria if a module's boundary is proven sound and
the threshold is wrong.

**Why:** leaving Program E marked active without an explicit child plan would
hide the real blocker. The remaining work is not feature invention. It is
structural decomposition and truthful closeout.

---

## References Reviewed

- `docs/plans/01_master-roadmap.md`
- `docs/plans/06_simplification-and-observability.md`
- `llm_client/client.py`
- `llm_client/mcp_agent.py`
- `llm_client/io_log.py`
- `llm_client/agents_codex.py`
- `llm_client/observability/experiments.py`
- `llm_client/agent_contracts.py`

---

## Current Audit Baseline

Fresh line-count audit on 2026-03-22:

1. `llm_client/client.py`: `4184`
2. `llm_client/mcp_agent.py`: `3335`
3. `llm_client/io_log.py`: `2102`
4. `llm_client/agents_codex.py`: `1931`
5. `llm_client/observability/experiments.py`: `1322`
6. `llm_client/agent_contracts.py`: `1228`

These counts are the current blocker for Program E completion.

---

## Files Affected

> This section declares the intended implementation surface for this plan. New
> slices must amend this section before touching additional files.

- `llm_client/client.py` (modify/extract)
- `llm_client/io_log.py` (modify/extract)
- `llm_client/observability/context.py` (new extracted module)
- `llm_client/observability/events.py` (compatibility facade wiring)
- `llm_client/observability/interventions.py` (new extracted module)
- `llm_client/mcp_agent.py` (modify/extract, later slice)
- `llm_client/agents_codex.py` (modify/extract, later slice)
- `llm_client/observability/experiments.py` (modify/extract if still needed after earlier tranches)
- `llm_client/agent_contracts.py` (modify/extract if still needed after earlier tranches)
- `docs/plans/06_simplification-and-observability.md` (update evidence/status)
- `docs/plans/01_master-roadmap.md` (update default next step as slices complete)
- `docs/API_REFERENCE.md` (generated)
- `docs/API_REFERENCE.html` (generated)

---

## Definition Of Done

This plan is done only when all of the following are true:

1. each targeted module is below the Program E hard threshold or has an
   explicit documented exception;
2. each extraction creates a narrower module with one clear responsibility;
3. existing behavior stays covered by focused regression tests at each wiring
   boundary; and
4. Program E can be evaluated honestly against its size/composition criteria
   without hand-waving about "already mostly done."

---

## Plan

## Thin Slices

### Phase 1: Boundary Map And First Tranche Selection

**Purpose:** choose the first extraction tranche from the current oversized set.

**Input -> Output:** raw size audit -> one concrete first write scope with
pass/fail tests and ownership

**Passes if:**

- the first tranche is concrete enough to implement without re-planning
- the tranche names source and destination modules explicitly
- affected tests and invariants are named up front

**Fails if:**

- the tranche is just "make client.py smaller"
- multiple unrelated mega-modules are edited at once without a boundary map

**Selected first tranche (2026-03-22):**

Start with `llm_client/io_log.py`, not `client.py`.

Write scope for the first implementation slice:

1. extract the experiment-run context / enforcement surface now living in
   `io_log.py` lines `282-673` into a dedicated observability-local module
2. extract the intervention logging/query/update surface now living in
   `io_log.py` lines `1934-2065` into a dedicated observability-local module
3. keep `io_log.py` as the compatibility facade so downstream imports remain
   truthful during the refactor

Why this tranche goes first:

1. it is meaningfully smaller and less coupled than attacking `client.py`
   first
2. it reduces one of the oversized modules without destabilizing the core call
   hot path
3. it already has strong regression coverage through:
   - `tests/test_experiment_log.py`
   - `tests/test_io_log.py`
   - `tests/test_io_log_compat.py`
   - experiment CLI coverage in `tests/test_cli_experiments.py`

### Phase 2: `client.py` / `io_log.py` First Decomposition Slice

**Purpose:** attack the most central oversized runtime modules first.

**Input -> Output:** oversized mixed-responsibility modules -> extracted
submodules with clearer seams

**Passes if:**

- one coherent concern is extracted from `client.py` or `io_log.py`
- public imports and compatibility behavior remain truthful
- focused regression tests pass

**Fails if:**

- extraction only moves code without improving responsibility boundaries
- compatibility surfaces silently drift

**Verified checkpoint 1 (2026-03-22):**

The intervention storage surface was extracted from `io_log.py` into
`llm_client/observability/interventions.py`, with `io_log.py` kept as the
compatibility facade.

What this checkpoint proved:

1. the tranche is viable without touching the core call hot path
2. the extraction surfaced and fixed a real latent bug in the old
   implementation (`log_intervention()` referenced missing `_resolve_project()`)
3. focused regression coverage remained green:
   - `pytest -q tests/test_experiment_log.py tests/test_io_log_compat.py`
   - result: `68 passed`

Effect on module size:

1. `llm_client/io_log.py`: `2102 -> 2011`
2. new module: `llm_client/observability/interventions.py` (`180` lines)

**Verified checkpoint 2 (2026-03-22):**

The experiment-context / feature-profile / AgentSpec guardrail surface was
extracted from `io_log.py` into `llm_client/observability/context.py`, with
`io_log.py` and `llm_client.observability.events` kept as compatibility
facades.

What this checkpoint proved:

1. the second half of the first tranche can move without changing downstream
   imports
2. the extraction reduced `io_log.py` materially while preserving the existing
   guardrail semantics used by call-contract and experiment surfaces
3. a public-surface regression was caught during test collection
   (`ActiveFeatureProfile` still had to be exported from
   `llm_client.observability.events`) and fixed before the slice was accepted
4. focused regression coverage remained green:
   - `pytest -q tests/test_experiment_log.py tests/test_io_log_compat.py tests/test_call_contracts.py`
   - result: `71 passed`

Effect on module size:

1. `llm_client/io_log.py`: `2011 -> 1600`
2. new module: `llm_client/observability/context.py` (`429` lines)

**Selected next tranche (2026-03-22):**

Finish `io_log.py` by replacing the remaining handwritten compatibility
wrappers for query / replay / experiment / intervention surfaces with direct
re-export bindings where the import graph is cycle-safe.

Passes if:

1. `io_log.py` drops below the hard threshold without changing the public API
2. the re-exports remain honest about where concrete behavior lives
3. focused compatibility, public-surface, and `io_log` regression tests stay
   green

Fails if:

1. the aliasing introduces import-cycle regressions
2. `io_log.py` still depends on large blocks of redundant wrapper code after
   the slice

**Verified checkpoint 3 (2026-03-22):**

The remaining `io_log.py` compatibility facade was reduced by replacing most
handwritten wrappers with truthful direct re-exports, while keeping dynamic
delegation only for the monkeypatch-sensitive compatibility surfaces that the
existing tests prove are load-bearing (`start_run`, `get_cost`, and
`get_background_mode_adoption`).

What this checkpoint proved:

1. `io_log.py` no longer blocks Program E's hard-threshold criterion
2. the compatibility surface can stay truthful without preserving hundreds of
   lines of redundant wrapper code
3. dynamic delegation remains in place where historical patchability is part
   of the effective contract
4. focused regression coverage remained green:
   - `pytest -q tests/test_io_log.py tests/test_io_log_compat.py tests/test_experiment_log.py tests/test_public_surface.py tests/test_api_reference_generation.py`
   - result: `137 passed`

Effect on module size:

1. `llm_client/io_log.py`: `1600 -> 1222`

**Selected next tranche (2026-03-22, post-io_log closeout):**

Move to `llm_client/agents_codex.py`.

Write scope for the next implementation slice:

1. extract Codex isolated-process transport helpers into a dedicated module
2. keep `agents_codex.py` as the orchestration facade so public entry points
   stay stable
3. prove the extracted boundary against both text and structured Codex paths

Why this tranche goes next:

1. `agents_codex.py` is now the smallest remaining hard-threshold violator
   after `io_log.py` was reduced below `1500`
2. the isolated-process transport path is a coherent responsibility that is
   visibly separable from CLI transport, result shaping, and streaming
3. it offers meaningful size reduction without starting with the most central
   modules (`client.py`, `mcp_agent.py`)

### Phase 3: Remaining Oversized Modules Or Explicit Re-Scope

**Purpose:** continue through the remaining oversized modules or document a
truthful exception if the threshold itself is wrong.

**Input -> Output:** unresolved oversize debt -> completed decompositions or
explicitly justified exceptions

**Passes if:**

- every module above threshold is either reduced or explicitly justified
- Program E closeout status becomes evidence-based rather than aspirational

**Fails if:**

- oversized modules are left implicit in conversation only
- completion depends on vague future cleanup

---

## Required Tests

### Existing Tests (Must Pass)

| Test Pattern | Why |
|--------------|-----|
| `tests/test_client.py` | core runtime behavior must stay stable during `client.py` extractions |
| `tests/test_io_log.py` | observability persistence must stay stable during `io_log.py` extractions |
| `tests/test_client_lifecycle.py` | lifecycle observability must not regress |
| targeted CLI / public-surface tests | extracted modules must preserve user-facing surfaces |

### Additional Verification

1. fresh module line-count audit after each tranche
2. roadmap and Program E plan status updates after each verified slice

---

## Acceptance Criteria

- [x] the remaining oversized modules are inventoried with a durable audit baseline
- [x] the first decomposition tranche is explicitly selected with pass/fail tests
- [x] the roadmap and Program E umbrella plan point to this child plan as the next default slice

---

## Notes

Do not treat this as a blanket refactor. Keep slices thin and responsibility
driven. The point is not to hit arbitrary line counts by shuffling code. The
point is to reduce oversized mixed-responsibility modules until Program E's
maintainability claim is true.
