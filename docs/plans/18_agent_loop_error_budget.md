# Plan #18: Agent Loop Error Budget and Retry Policy

**Status:** Planned
**Type:** implementation
**Priority:** High
**Blocked By:** None
**Blocks:** DIGIMON benchmark reliability

---

## Gap

**Current:** The agent loop has no aggregate error budget. Each LLM call has `RetryPolicy(max_retries=2)` and there's a `max_tool_calls=20` limit, but these don't compose correctly. When the primary model fails, fallback models each get their own retry budget. A single question can generate 318 LLM calls (observed in DIGIMON MuSiQue benchmark, 2026-03-24) consuming 1900+ seconds for what should be a 30-second question.

**Root cause:** The retry/fallback system treats each failure independently. There's no distinction between recoverable vs non-recoverable errors at the agent loop level, and no aggregate budget that caps total retry effort across all models.

**Target:** Configurable agent-level error budget with:
1. Classification of errors as recoverable vs non-recoverable
2. Aggregate retry budget (total attempts across all fallback models)
3. Configurable policy for when to stop retrying even recoverable errors
4. Per-question wall-clock timeout as a hard backstop

---

## Error Classification

### Non-recoverable (stop immediately)
- Authentication failures (401, 403)
- Quota/billing exhaustion
- Model not found (404)
- Content policy violations (safety filter, recitation)
- Tool function not found

### Potentially recoverable (retry with budget)
- Rate limits (429) — may clear after backoff
- Timeouts — transient network issue
- Server errors (500, 502, 503) — provider instability
- Malformed JSON in response — LLM output variance
- Empty response — provider glitch

### Recoverable but capped (retry N times then skip)
- JSON parse errors on tool results — might be a tool bug, not transient
- Tool execution errors — might be a data issue, not transient
- Malformed tool calls from LLM — model capability issue, unlikely to self-correct

---

## Plan

### Proposed Policy

```python
@dataclass
class AgentErrorBudget:
    """Aggregate error budget for the agent loop.

    Limits total retry effort across all models and fallbacks.
    """
    # Hard caps
    max_total_llm_calls: int = 100          # Total LLM calls per question (all models)
    max_wall_seconds: float = 300           # 5-minute wall clock per question
    max_consecutive_errors: int = 5         # Stop after N consecutive errors

    # Per-error-type caps
    max_retries_per_error_type: int = 3     # Same error pattern → stop after 3
    max_fallback_cascade_depth: int = 3     # Don't try more than 3 fallback models

    # Backoff escalation
    escalate_to_skip_after: int = 10        # After 10 total errors, skip to next question
```

### Integration Points

1. **In `_agent_loop` (mcp_turn_execution.py):** Track cumulative LLM calls and wall time. Check budget before each turn.

2. **In `execute_direct_tool_calls` (tool_utils.py):** Track consecutive errors. When `max_consecutive_errors` hit, break the tool execution loop.

3. **In fallback chain:** Track cascade depth. When `max_fallback_cascade_depth` exceeded, don't try more models — fail the call.

4. **In RetryPolicy:** Add `error_signature` tracking so the same error pattern doesn't retry indefinitely.

---

## Pre-made Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Where to put AgentErrorBudget | `llm_client/agent/agent_contracts.py` | Agent-level concern, not per-call |
| Default max_total_llm_calls | 100 | 10x the normal tool budget (20 tools × ~5 LLM calls each) |
| Default max_wall_seconds | 300 | 5 minutes per question is generous; 30 min is broken |
| Configure via | Kwarg to `acall_with_tools` / `_agent_loop` | Same pattern as existing kwargs |
| Backward compatible | Yes — default budget is generous enough that current working runs aren't affected |

---

## Acceptance Criteria

- [ ] AgentErrorBudget dataclass with configurable limits
- [ ] `_agent_loop` checks budget before each turn, breaks when exceeded
- [ ] Error classification: non-recoverable stops immediately, recoverable retries with budget
- [ ] max_consecutive_errors prevents infinite retry loops on the same error
- [ ] Existing benchmark runs (HotpotQAsmallest 3q) produce same results with default budget
- [ ] The 318-call pathological case would be capped at ~100 calls

---

## Budget

- Implementation: ~2 hours (dataclass + integration into agent loop)
- Testing: ~1 hour (verify existing benchmarks aren't affected, verify pathological case is capped)
- No LLM cost (code change only)
