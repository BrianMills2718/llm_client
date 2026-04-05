# ADR 0015: Provider Governance and Shared Coordination

Status: Accepted
Date: 2026-04-05
Applies to: Plan #25

## Context

`llm_client` already owns the shared runtime substrate for model execution,
routing, retries, observability, and agent-SDK dispatch. Recent anomaly work
confirmed that this ownership boundary is incomplete unless provider-governance
behavior is also treated as first-class shared infrastructure.

The concrete failures were:

1. exact `gpt-5.4` requests could drift into provider routing instead of the
   intended Codex SDK path,
2. Gemini first-attempt bursts could exceed shared quota even when each process
   respected its own local semaphore,
3. rate-limit mitigation state existed, but the policy that governs routing,
   caps, cooldowns, and incident publishing was spread across multiple modules
   and downstream operator tooling.

That pattern is operationally expensive:

- downstream repos experience the same failure class at the same time,
- `ecosystem-ops` has to infer shared incidents from raw failures instead of
  consuming explicit runtime governance signals,
- fixes land as tactical patches instead of durable runtime policy.

`llm_client` therefore needs one explicit provider-governance layer that turns
model identity, route selection, shared-cap coordination, and provider incident
signaling into a coherent shared contract.

## Decision

1. `llm_client` owns provider governance as part of the shared runtime
   substrate. This includes:
   - canonical model identity,
   - provider routing class,
   - hard blocks and forced reroutes,
   - provider-level shared concurrency caps,
   - provider cooldown floors and coordination state,
   - structured provider-governance events for observability.

2. Provider governance will be expressed through one typed policy surface
   rather than a collection of implicit rules distributed across routing,
   retries, and downstream anomaly logic.

3. Coordination mechanics will be separated from policy declaration.
   - policy answers what should happen,
   - coordination answers how shared processes enforce it.

4. The default coordination backend remains local-machine safe and
   reversible. SQLite-backed coordination is the default substrate for
   multi-worktree and multi-process control on one host.
   - This is sufficient for current overnight and worktree execution.
   - A distributed backend such as Redis or Postgres advisory locks is deferred
     until multi-host execution is a proven requirement.

5. `ecosystem-ops` remains the operator and lifecycle surface, not the
   provider-policy engine.
   - `llm_client` emits governance facts and events.
   - `ecosystem-ops` groups, ranks, acknowledges, and resolves incidents using
     those facts.

## Required Invariants

1. Canonical model identity must be resolved before provider routing.
   - exact aliases such as `gpt-5.4` must normalize to their authoritative
     runtime lane before provider selection.

2. Shared provider coordination must not be consumed by local queueing.
   - a process must acquire its local semaphore before claiming shared provider
     coordination state.

3. Provider-governance decisions must be observable.
   - if a route is canonicalized, a cap wait occurs, a cooldown is registered,
     or a model is blocked, that fact must be available in shared observability.

4. Provider incidents are not resolved by code deployment alone.
   - mitigation can be acknowledged once landed,
   - resolution requires fresh evidence that recurrence stopped.

5. Downstream repos must not encode competing copies of shared provider policy.
   - application repos may choose schedules and workloads,
   - they may not redefine canonical runtime routing or provider identity rules.

## Target Architecture

### 1. Provider Policy

One typed provider-policy surface declares:

- canonical model aliases and reroutes,
- route class (`codex_sdk`, `direct_provider`, `openrouter`, `agent_sdk`, etc.),
- provider default caps,
- cooldown floors,
- hard-block / allow / reroute decisions,
- optional task-class constraints.

### 2. Provider Coordination

One coordination interface enforces shared runtime behavior:

- shared lease acquisition/release,
- cooldown registration and waiting,
- expiry reclamation,
- provider-scoped wait/deny decisions.

SQLite is the default backend implementation.

### 3. Provider Events

`llm_client` emits structured events for:

- route canonicalization,
- policy block or reroute,
- shared-cap wait started/ended,
- cooldown registered/applied,
- coordination backend failures.

### 4. Operator Consumption

`ecosystem-ops` consumes these facts to:

- group incidents truthfully,
- show mitigation state,
- display prevention evidence in click-through detail,
- distinguish acknowledged mitigation from resolved recurrence.

## Consequences

Positive:

1. One place to prevent recurrence of shared provider failures across projects.
2. Cleaner separation between runtime policy and operator presentation.
3. Less downstream drift around model aliases and provider identity.
4. Stronger observability for provider-side saturation before it becomes a
   noisy anomaly burst.

Negative:

1. `llm_client` becomes more explicit about provider policy and therefore needs
   clearer public/runtime documentation.
2. Coordination and policy code must be tested both in-process and
   cross-process.
3. There is a risk of overfitting the policy layer to one provider if the
   interfaces are not kept generic.

## Off-The-Shelf vs Hand-Rolled Guidance

1. Keep using LiteLLM for commodity provider normalization and transport where
   it remains correct and observable.
2. Do not build a custom distributed coordination service unless a multi-host
   requirement is demonstrated.
3. Prefer SQLite now, because it is simple, inspectable, local, and already
   adequate for multi-worktree coordination.
4. Use typed Pydantic policy/config models rather than hand-maintained
   free-form env-var tables as the source of truth.

## Testing Contract

1. Unit tests must prove canonical model identity and route selection.
2. Unit tests must prove provider cooldown publication and reuse.
3. Cross-process tests must prove shared-cap enforcement, expiry reclamation,
   and non-consumption of shared slots by local queueing.
4. Observability tests must prove provider-governance events are emitted with
   stable fields.
5. Downstream operator tests must prove incidents can be acknowledged after
   mitigation lands and resolved only after verification evidence exists.
