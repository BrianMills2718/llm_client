# Ecosystem Uncertainties

This document tracks the unresolved architectural questions that remain after
the 2026-03-17 boundary decisions in ADR 0010, ADR 0011, and ADR 0012.

These are not undecided because the direction is unclear. They are undecided
because implementation detail and migration sequencing still matter.

## Scope

This file tracks cross-project uncertainties for:

1. `llm_client` as the shared runtime and observability substrate,
2. `prompt_eval` as the prompt-evaluation layer,
3. the future shared prompt asset layer,
4. the future shared data plane.

## Current Assumptions

If work proceeds before every question below is resolved, use these
assumptions:

1. Package boundaries matter more than repo boundaries.
2. `llm_client` remains the authoritative shared observability backend.
3. Prompt assets will use explicit identity and lineage rather than overrides.
4. Shared data should converge toward canonical cross-project sources of truth.
5. Commodity routing and durable orchestration should be wrapped, not rebuilt,
   when established libraries already solve the hard part.

## Open Questions

### U1: Repo Topology

**Status:** ✅ Resolved  
**Raised:** 2026-03-17  
**Context:** The package boundary is clear, but repo topology is not.  
**Current assumption:** Keep `llm_client` and `prompt_eval` as distinct
packages. They may remain separate repos or move into one monorepo later.

### U2: Home of the Shared Prompt Library

**Status:** ❓ Open  
**Raised:** 2026-03-17  
**Context:** Shared prompt assets should not be owned by `prompt_eval`, but the
physical home is still undecided.  
**Current options:** a dedicated prompt-asset repo/package, a subpackage in a
shared data layer, or a carefully bounded package inside `llm_client`.  
**Current assumption:** optimize for explicit asset identity first; do not block
other boundary cleanup on this choice.

### U3: Prompt Asset Metadata Schema

**Status:** 🔍 Investigating  
**Raised:** 2026-03-17  
**Context:** We agreed on explicit prompt identity and lineage, but not the
minimal metadata contract.  
**Known requirements:** asset ID, version, namespace/owner, `derived_from`,
template location, input/output schema references, tags, and status metadata.  
**Current assumption:** keep the first schema minimal and additive.

### U4: Shared Data Plane Backend

**Status:** ❓ Open  
**Raised:** 2026-03-17  
**Context:** We agreed on a separate shared data layer but not on its concrete
storage stack.  
**Current options:** SQLite plus content-addressed files, Postgres plus object
store, or a lightweight registry service over filesystem/object storage.  
**Current assumption:** start with the simplest registry that preserves
identity, lineage, and provenance without turning into a custom platform too
early.

### U5: `prompt_eval` Observability Migration

**Status:** 🔍 Investigating  
**Raised:** 2026-03-17  
**Context:** `prompt_eval` still persists JSON result files locally today.  
**Current options:** dual-write to JSON and `llm_client`, adapter export into
`llm_client`, or a direct cutover with compatibility helpers. The run-family
mapping itself is now defined in `prompt_eval` ADR 0002.  
**Current assumption:** local JSON artifacts can remain as exports, but the
authoritative analytics backend should move toward `llm_client`.

### U6: `task_graph` Versus External Workflow Runtime

**Status:** ✅ Resolved
**Raised:** 2026-03-17
**Resolved:** 2026-03-18 (Program C complete)
**Decision:** `task_graph` remains the intentionally simple DAG runner.
Durable workflows use `llm_client.workflow_langgraph` (LangGraph-backed,
proven with approval/resume slice). Coexistence rule: no durable-workflow
features get added to `task_graph`.

### U7: Extent of `llm_client` Model Policy Logic

**Status:** ❓ Open  
**Raised:** 2026-03-17  
**Context:** We agreed that `llm_client` should lean harder on LiteLLM for
commodity routing, but the exact boundary between shared policy and wrapped
library behavior is still unsettled.  
**Current assumption:** keep `llm_client` as the public wrapper/substrate,
lean harder on LiteLLM for commodity transport behavior, and stop pursuing a
middleware-only architecture as the primary surface. Keep task-based selection
and platform-specific guard rails where they add real value, but stop growing
hand-maintained market intelligence tables without clear evidence. Longer
term, prefer model-policy guidance derived from observed task performance,
cost, latency, and failure data over expanding static hand-maintained
rankings. The immediate next program is documented in
`docs/plans/03_model-policy-modernization.md`, and its first slice is now
proven: the built-in registry has moved to packaged data with parity-tested
selection behavior. The static candidate-selection path is also now explicit
before any performance overlay is applied. The additive performance overlay is
also now explicit and inspectable. `difficulty.py` remains a frozen
compatibility-guidance layer for `task_graph` and analyzer/model-floor logic;
new product-facing code should prefer task-based selection rather than growing
the difficulty surface.

### U8: Logical Boundary Hardening Versus Physical Package Reorg

**Status:** 🔍 Investigating  
**Raised:** 2026-03-18  
**Context:** The desired internal boundaries are becoming clearer, but the
timing of a physical package tree reorganization is not.  
**Current assumption:** define and prove logical boundaries first, including
import direction and public-surface cleanup. Delay broad filesystem/package
reorganization until the lower-risk internal seams are extracted and stable.
Because direct `from llm_client import ...` usage is widespread across the
workspace, treat public-export removal as a separately audited migration, not
as a casual cleanup side effect. The first symbol-level audit (`difficulty`)
found no non-`llm_client` workspace consumers. The next completed low-risk
pilots were `git_utils`, scoring/experiment-eval, and the **top-level**
re-exports for `task_graph` and analyzer. The current explicit boundary is:
`llm_client.scoring`, `llm_client.experiment_eval`, `llm_client.task_graph`,
and `llm_client.analyzer` remain stable optional module namespaces. The
remaining higher-risk question is future physical/package placement, not
whether those surfaces should still be taught at the package root.

### U9: Fate Of The Custom Agent Runtime

**Status:** ❓ Open  
**Raised:** 2026-03-18  
**Context:** The MCP/tool-loop runtime overlaps with external agent
frameworks, but it also carries custom concepts such as artifact contracts,
progressive disclosure, and loop control policies.  
**Current assumption:** isolate the custom agent runtime as an explicit
optional layer and stop growing it by default. Decide later, based on actual
ecosystem usage and value, whether it remains strategic or should be wrapped by
an external runtime. This does not apply to basic agent SDK routing at the
shared call boundary; multi-SDK dispatch remains part of the core substrate.
The first proven isolation slice now exists: shared tool-call records and
tool-description/contract helpers live in `tool_runtime_common.py` so
`tool_utils.py` no longer depends on `mcp_agent.py` internals. The same seam
now also carries the shared agent-result type and usage-count extraction so
`tool_shim.py` no longer imports private MCP helper functions for those
concerns. Core text-call routing now also uses explicit optional-runtime
adapter functions instead of importing private `_acall_with_mcp` /
`_acall_with_tools` entrypoints directly.

### U10: Runtime Split Topology Inside `llm_client`

**Status:** ✅ Resolved
**Raised:** 2026-03-18
**Resolved:** 2026-03-18 (Program A complete)
**Decision:** Split by workload family. `text_runtime` and
`structured_runtime` are proven and stable. Sync/async variants stay together
per workload. Reassess further fragmentation only if `client.py` bulk still
forms a coherent workload family needing extraction.

### U11: Quality Hardening Priorities Versus Stale Audit Claims

**Status:** 🔍 Investigating  
**Raised:** 2026-03-18  
**Context:** External strategic reviews can identify the right risk areas while
still carrying stale or overstated test-gap claims.  
**Current assumption:** future hardening work should be prioritized using live
repository evidence: current tests, executed targeted verification, and real
coverage artifacts where available. Do not treat unverified "0% coverage" or
similar claims as planning truth without checking the current tree first.
