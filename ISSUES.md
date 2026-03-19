# Issues

Observed problems, concerns, and technical debt. Items start as **unconfirmed**
observations and get triaged through investigation into confirmed issues, plans,
or dismissed.

**Last reviewed:** (date)

---

## Status Key

| Status | Meaning | Next Step |
|--------|---------|-----------|
| `unconfirmed` | Observed, needs investigation | Investigate to confirm/dismiss |
| `monitoring` | Confirmed concern, watching for signals | Watch for trigger conditions |
| `confirmed` | Real problem, needs a fix | Create a plan |
| `planned` | Has a plan (link to plan) | Implement |
| `resolved` | Fixed | Record resolution |
| `dismissed` | Investigated, not a real problem | Record reasoning |

---

## Unconfirmed

(Add observations here with enough context to investigate later)

### ISSUE-001: (Title)

**Observed:** (date)
**Status:** `unconfirmed`

(What was observed. Why it might be a problem.)

**To investigate:** (What would confirm or dismiss this.)

---

## Monitoring

(Items confirmed as real but not yet urgent. Include trigger conditions.)

---

## Confirmed

(Items that need a fix but don't have a plan yet.)

### ISSUE-002: `workspace_agent` has no turn-level progress callback during execution

**Observed:** 2026-03-19  
**Status:** `confirmed`

`workspace_agent` calls currently expose `conversation_trace` only after the
agent call returns. During execution, callers have no turn-level progress
signal, no callback hook, and no built-in way to distinguish "still making
useful progress" from "stalled in the SDK/runtime return path" without
external process inspection or cancellation.

This showed up concretely in AC11 thesis experiments: agent-written workspaces
could already contain correct code that passed hidden acceptance, but the
awaiting caller still had no intermediate visibility while the agent loop was
running. `max_turns` helps bound iteration count, but it does not solve the
observability gap.

**Why it matters:**
- shared agent-runtime observability is a `llm_client` concern, not something
  each downstream repo should reimplement
- wall-clock timeouts are the wrong control for this class of problem
- downstream systems need progress signals to log, detect stagnation, and
  cancel intelligently

**Likely fix directions:**
- add an `on_turn` callback for `workspace_agent` dispatch
- optionally expose a progress-stream/event hook with the same semantics
- later consider wiring stagnation detection into the SDK-agent loop, not just
  MCP/tool loops

---

## Resolved

| ID | Description | Resolution | Date |
|----|-------------|------------|------|
| - | - | - | - |

---

## Dismissed

| ID | Description | Why Dismissed | Date |
|----|-------------|---------------|------|
| - | - | - | - |

---

## How to Use This File

1. **Observe something off?** Add under Unconfirmed with context and investigation steps
2. **Investigating?** Update the entry with findings, move to appropriate status
3. **Confirmed and needs a fix?** Create a plan, link it, move to Confirmed/Planned
4. **Not actually a problem?** Move to Dismissed with reasoning
5. **Watching a concern?** Move to Monitoring with trigger conditions
