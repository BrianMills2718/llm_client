# Plan #25: Provider Governance and Shared Coordination

**Status:** Planned
**Type:** design
**Priority:** Critical
**Blocked By:** llm_client PR #24 merge for the latest Gemini coordination baseline
**Blocks:** durable prevention of shared provider saturation and downstream provider-policy drift

---

## Gap

**Current:** `llm_client` has working tactical fixes for recent provider
incidents:

- `gpt-5.4` now routes toward the Codex SDK lane,
- Gemini has shared cooldowns and SQLite-backed shared leases,
- anomaly/operator tooling can acknowledge the resulting shared incident.

Those fixes are valuable but incomplete as long-term infrastructure because the
governing policy is still distributed across routing helpers, retry logic,
rate-limit coordination, and downstream operator interpretation.

**Target:** `llm_client` exposes provider governance as one explicit shared
runtime subsystem with:

1. a typed policy model,
2. a coordination backend abstraction,
3. structured provider-governance observability events,
4. cross-process validation coverage,
5. clear downstream/operator contracts.

**Why:** without one canonical governance layer, the same class of provider
incident will recur as new aliases, quotas, and execution patterns appear. The
repo needs a durable prevention model, not a growing pile of tactical patches.

---

## References Reviewed

- `CLAUDE.md` - repo workflow and maintenance posture
- `docs/plans/01_master-roadmap.md` - canonical roadmap and maintenance framing
- `docs/plans/CLAUDE.md` - plan index and numbering
- `docs/ops/CAPABILITY_DECOMPOSITION.md` - current ownership boundary
- `docs/adr/0010-cross-project-runtime-substrate.md` - substrate ownership
- `docs/adr/0002-routing-config-precedence.md` - routing-policy decision context
- `llm_client/core/routing.py` - current model canonicalization and policy normalization
- `llm_client/utils/rate_limit.py` - current semaphore, cooldown, and shared-lease behavior
- `llm_client/execution/execution_kernel.py` - cooldown publication path
- `/home/brian/projects/ecosystem-ops/anomaly_incidents.py` - downstream shared-incident grouping
- `/home/brian/projects/ecosystem-ops/anomaly_triage.py` - incident lifecycle requirements

---

## Files Affected

- `docs/adr/0015-provider-governance-and-shared-coordination.md` (create)
- `docs/adr/README.md` (modify)
- `docs/plans/25_provider-governance-and-shared-coordination.md` (create)
- `docs/plans/CLAUDE.md` (modify)
- `docs/plans/01_master-roadmap.md` (modify)
- `docs/ops/CAPABILITY_DECOMPOSITION.md` (modify)
- `README.md` (modify in later implementation or follow-up docs slice if runtime policy becomes user-facing)
- `llm_client/core/` and `llm_client/utils/` runtime files (future implementation phases)
- `tests/` provider-governance and coordination validation files (future implementation phases)

---

## Plan

### Phase 1: Policy Centralization

Create one typed provider-policy layer that declares:

- canonical model aliases,
- forced reroutes,
- hard-blocked models,
- provider route class,
- provider default caps,
- cooldown floors,
- optional task-class restrictions.

Deliverables:

- one Pydantic policy model,
- one canonical load/validation path,
- one current-policy representation replacing scattered literals.

### Phase 2: Coordination Backend Boundary

Extract shared coordination into a dedicated backend interface with a default
SQLite implementation.

Deliverables:

- explicit coordination interface,
- SQLite backend for leases/cooldowns,
- clean separation between policy and storage/enforcement mechanics,
- graceful observability on backend errors without silent policy bypass.

### Phase 3: Provider Event Surface

Emit structured provider-governance events that downstream observability can
consume directly.

Deliverables:

- event schema for route canonicalization, cap waits, cooldowns, and blocks,
- stable event names and fields,
- observability documentation for downstream consumers.

### Phase 4: Cross-Process Validation Harness

Add an integration-style validation layer that proves shared coordination
behavior across multiple local processes.

Deliverables:

- multi-process Gemini cap test,
- lease-expiry reclamation test,
- local-semaphore-before-shared-lease test,
- event-emission assertions for key provider-governance paths.

### Phase 5: Operator Surface Integration Contract

Define and prove the downstream contract from `llm_client` to
`ecosystem-ops`.

Deliverables:

- mapping from provider events to grouped incidents,
- lifecycle guidance for acknowledge vs resolve,
- operator-facing evidence fields for prevention status.

### Phase 6: Rollout and Resolution Gates

Roll out in stages and require fresh operational validation before resolving
shared incidents.

Deliverables:

- explicit rollout checklist,
- verification window for overnight Gemini-heavy runs,
- criteria for incident resolution and rollback.

---

## Required Tests

### New Tests (TDD / Required)

| Test File | Test Function | What It Verifies |
|-----------|---------------|------------------|
| `tests/test_provider_policy.py` | `test_canonicalizes_exact_aliases_before_route_selection` | `gpt-5.4` and similar aliases normalize before provider routing |
| `tests/test_provider_policy.py` | `test_policy_declares_forced_reroute_and_block_rules` | hard blocks and forced reroutes come from typed policy, not ad hoc conditionals |
| `tests/test_provider_coordination.py` | `test_shared_cap_enforced_across_processes` | shared provider cap is honored across concurrent local processes |
| `tests/test_provider_coordination.py` | `test_local_queueing_does_not_consume_shared_slots` | shared slots are acquired after local semaphore ownership |
| `tests/test_provider_events.py` | `test_emits_route_canonicalization_event` | route canonicalization is visible in observability |
| `tests/test_provider_events.py` | `test_emits_cooldown_and_cap_wait_events` | shared cooldown/cap decisions produce stable events |

### Existing Tests (Must Pass)

| Test Pattern | Why |
|--------------|-----|
| `tests/test_routing.py` | route normalization and Codex SDK behavior remain correct |
| `tests/test_rate_limit.py` | current shared cooldown and lease semantics stay intact during refactor |
| `tests/test_client.py -k "codex or routing"` | public client routing still reflects the governance layer |
| `git diff --check` | planning/docs slice remains syntactically clean |

---

## Acceptance Criteria

- [ ] provider-governance ownership is documented as part of the shared runtime substrate
- [ ] one typed provider-policy surface is defined before further provider patches land
- [ ] coordination behavior is separated from policy declaration
- [ ] structured provider-governance events are part of the design contract
- [ ] cross-process validation is required, not optional
- [ ] downstream operator lifecycle expectations are explicit: mitigation can be acknowledged before recurrence is resolved
- [ ] roadmap/index docs point to this as the next shared-infrastructure slice instead of leaving the work as unplanned anomaly fallout

---

## Notes

- The current SQLite backend is not a hack. It is the right default until
  multi-host concurrency is a demonstrated requirement.
- This plan should not widen into project scheduling/orchestration work inside
  `llm_client`; application/job staggering remains downstream execution policy.
- `ecosystem-ops` stays the operator console and lifecycle surface. It should
  consume governance facts, not become the source of provider policy truth.
- If PR #24 changes materially before merge, refresh this plan’s references and
  phase ordering before implementation begins.
