# Sprint: 2026-04-05 Plan 25 Provider Governance

**Status:** In Progress
**Owner:** codex
**Plan:** [../plans/25_provider-governance-and-shared-coordination.md](../plans/25_provider-governance-and-shared-coordination.md)
**Mission:** move recent anomaly-driven provider fixes out of tactical routing/rate-limit patches and into a durable shared provider-governance subsystem in `llm_client`.

## Operating Rules

1. Work only from dedicated worktrees until merge/push time.
2. Commit every verified slice before starting the next one.
3. Record every blocker, concern, or uncertainty in this file, then continue with the safer option unless a real stop condition applies.
4. Do not mark incidents resolved from code deployment alone; require fresh operational evidence.
5. Merge completed worktree branches into a clean integration branch, publish, then remove completed worktrees.

## Next 24 Hours

### Phase 0: Sprint Hardening

Goal: make the sprint executable without ambiguity.

Acceptance:

- `CLAUDE.md` points to this sprint tracker
- worktree claim exists for the implementation branch
- phase ordering and verification requirements are explicit here

### Phase 1: Typed Provider Policy

Goal: create one typed policy surface for canonical model identity, forced reroutes, hard blocks, shared-cap defaults, and cooldown defaults.

Acceptance:

- runtime code no longer relies on scattered provider-policy literals for the implemented paths
- exact aliases such as `gpt-5.4` are represented in the policy layer
- provider defaults for Gemini shared-cap and cooldown policy are represented in the policy layer

### Phase 2: Coordination Backend Boundary

Goal: separate policy from the shared coordination mechanics.

Acceptance:

- runtime code uses a coordination backend interface rather than direct SQLite-specific helpers in the policy path
- the default backend remains SQLite-backed and behaviorally equivalent for current Gemini coordination
- local queueing still acquires the local semaphore before the shared provider slot

### Phase 3: Provider Event Surface

Goal: make provider-governance decisions explicitly observable.

Acceptance:

- canonicalization, block/reroute, cap-wait, and cooldown paths emit structured provider-governance events or stable observability records
- event field names are documented in code or tests

### Phase 4: Validation Harness

Goal: prove the new layer rather than trusting the refactor.

Acceptance:

- new tests cover policy normalization
- new tests cover policy-declared reroutes/blocks
- coordination tests prove the boundary still enforces shared caps correctly

### Phase 5: Downstream Contract And Documentation

Goal: make downstream operator usage explicit.

Acceptance:

- docs describe what `ecosystem-ops` should consume from `llm_client`
- lifecycle rule remains explicit: mitigation can be acknowledged before an incident is resolved

### Phase 6: Merge, Publish, Cleanup

Goal: leave a clean and reversible state.

Acceptance:

- verified implementation branches are merged into a clean integration branch
- merged branch is pushed
- completed worktrees created by this sprint are removed
- sprint tracker records what shipped and what remains

## Verification Matrix

- `git diff --check`
- `pytest tests/test_provider_policy.py`
- `pytest tests/test_provider_coordination.py`
- `pytest tests/test_rate_limit.py`
- `pytest tests/test_routing.py`
- `pytest tests/test_client.py -k "codex or routing"`

## Concerns And Uncertainties

### 2026-04-05

- Plan 25 is documented as blocked on `llm_client` PR #24 merge. Safe default: implement on top of the sanctioned provider-governance plan branch that already includes the latest anomaly mitigation baseline, and keep the dependency visible in commits and tracker notes.
