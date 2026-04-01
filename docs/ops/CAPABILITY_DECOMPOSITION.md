# Capability Decomposition

Last updated: 2026-04-01

## Purpose

This document is the repo-local source of record for what `llm_client` owns as
shared infrastructure, what it intentionally consumes from adjacent shared
repos, and what it should not quietly absorb.

Use this together with:

- [`../plans/22_capability-ownership-and-sanctioned-worktree-alignment.md`](../plans/22_capability-ownership-and-sanctioned-worktree-alignment.md)
- [`../../README.md`](../../README.md)
- [`../../CLAUDE.md`](../../CLAUDE.md)

## Role

`llm_client` is the shared runtime substrate for Brian's project ecosystem.

It owns:

- model execution and transport routing
- required call metadata and budgets
- structured-output contract enforcement
- tool-call and LLM-call observability
- streaming, batch execution, and replay/query surfaces
- MCP loop and agent-runtime plumbing that belongs in the shared runtime layer

It does not own:

- prompt-evaluation semantics and optimization loops
- project-specific retrieval or application logic
- the cross-project meta-process/governance framework

Those stay in adjacent shared repos such as `prompt_eval` or `project-meta`, or
in consuming project repos.

## Capability Ledger

| Capability | Current owner | Intended owner | Class | Posture | Notes |
|---|---|---|---|---|---|
| Shared LLM execution, routing, retries, budgets, and structured-output enforcement | `llm_client` | `llm_client` | shared infrastructure | no move planned | Core runtime substrate. |
| Shared LLM/tool observability, run logging, replay, and diagnostics | `llm_client` | `llm_client` | shared infrastructure | no move planned | Other repos should consume this surface rather than rebuilding parallel logging stacks. |
| Prompt-centric evaluation semantics, statistical comparison, and optimization | `prompt_eval` | `prompt_eval` | consumed shared infrastructure | consume, do not re-own | `llm_client` should expose runtime primitives, not re-grow evaluation-framework logic. |
| Meta-process governance, rollout policy, and cross-repo ownership registry | `project-meta` | `project-meta` | consumed agent platform | consume, do not re-own | `llm_client` participates in governed rollout; it does not define the policy. |

## Known Consumers

Current known ecosystem consumers include:

- `prompt_eval`
- `Digimon_for_KG_application`
- `onto-canon6`
- `project-meta`
- `grounded-research`
- `open_web_retrieval`

This list is evidence-based and should expand only when real maintained
integrations exist.

## Boundary Rules

1. Keep `llm_client` focused on the shared runtime/control-plane layer.
2. `prompt_eval` owns prompt-evaluation semantics; do not move that boundary
   back into `llm_client`.
3. `project-meta` owns governance rollout policy and registry rules; consume
   them rather than duplicating them locally.
4. If a new feature would make `llm_client` a project-specific application or
   a prompt-evaluation framework, stop and document the boundary decision
   before implementing it.

## Open Uncertainties

- The exact long-term boundary between generic agent-runtime plumbing in
  `llm_client` and higher-level orchestration/runtime layers is still evolving.
- Some consumers still carry repo-local helper wrappers around `llm_client`,
  and the threshold for pulling those back into shared runtime remains
  evidence-driven.
- The right enforcement point for requiring consumer repos to prefer
  `llm_client` over ad hoc runtime wrappers is still unsettled.
