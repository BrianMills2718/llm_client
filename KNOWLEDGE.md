# Operational Knowledge — llm_client

Shared findings from all agent sessions. Any agent brain can read and append.
Human-reviewed periodically.

## Findings

<!-- Append new findings below this line. Do not overwrite existing entries. -->
<!-- Format: ### YYYY-MM-DD — {agent} — {category}                          -->
<!-- Categories: bug-pattern, performance, schema-gotcha, integration-issue, -->
<!--             workaround, best-practice                                   -->
<!-- Agent names: claude-code, codex, openclaw                               -->

---

### 2026-04-01 — claude-code — best-practice

**Ecosystem audit findings (Phase 7 of infra sprint).**

Health: 1240 tests, 94% pass rate, no circular imports, clean architecture.

Issues found:
1. Plan #17 (text_runtime sync/async dedup) cancelled but 400 LOC duplication
   remains in execution/text_runtime.py. High maintenance risk — bug fixed in
   one path but missed in other. Consider un-cancelling.
2. KNOWLEDGE.md was empty until this entry. Agents should use /learned to
   record findings during debugging sessions.
3. 5 test_models.py failures due to stale model ranking expectations
   (gemini-2.5-flash vs gemini-3-flash-preview). Parameterize or update.
4. test_public_surface.py expects 118 exports, actual is 120. Trivial fix.
5. Gate-edit hook test overly strict on doc files. Gate should skip non-source.

### 2026-04-01 — codex — best-practice

**When a shared-infrastructure repo already has a registry row, the next
truthful step is a repo-local ownership source and sanctioned workflow
alignment, not more project-meta prose.**

The `llm_client` rollout showed a sharper failure mode than bootstrap-minimal
repos: the shared registry already knew about llm_client, but the repo still
lacked its own ownership source of record and its declared sanctioned worktree
policy did not match the actual Makefile/scripts surface.

Practical rule:

- if a shared repo is already registry-covered, add the local capability source
  and align declared workflow policy before expanding the registry further

### 2026-04-04 — codex — best-practice

**Portable workflow packaging should be staged as a runtime-facing manifest and validator layer before any registry or installer work.**

Journey, Anthropic's workflow-vs-agent framing, `static_pipeline`, and
`llm_client`'s existing LangGraph slice all point to the same boundary:
`llm_client` is the natural home for runtime-facing package truthfulness
(manifest models, capability-based runtime support, validation, and example
adapters), but not for registry/community/product concerns. A runtime list
without capability validation is not a truthful portability claim.

### 2026-04-04 — codex — bug-pattern

**Repeated submit-validator rejections that explicitly require the forced-terminal path should short-circuit the loop instead of burning the remaining tool budget.**

In DIGIMON's benchmark lane, the normal submit gate correctly rejected pending
semantic-plan atoms, but the generic MCP loop then kept searching and
re-submitting until budget exhaustion even after the validator signaled both
`new_evidence_required_before_retry` and `requires_forced_terminal_path`.
`mcp_turn_outcomes.py` now treats repeated rejections of that shape as early
control churn and forces finalization immediately. Verified in
`tests/test_mcp_agent.py::test_repeated_submit_rejections_can_force_final_early`.

### 2026-04-05 — codex — bug-pattern

**OpenRouter routing should normalize explicit `google/...` provider ids the same way it already normalizes other provider-prefixed models.**

`ac14` surfaced repeated provider-resolution failures from `MODEL=google/gemini-2.0-flash-001`.
Under OpenRouter policy, `llm_client.core.routing.normalize_model_for_policy()`
left `google/...` untouched, so the call reached LiteLLM without a provider
route and failed as "LLM Provider NOT provided". Fix: normalize `google/...`
to `openrouter/google/...` in the shared routing layer so stale but otherwise
valid OpenRouter Google aliases do not fail late inside structured-call
boundaries.

### 2026-04-05 — codex — bug-pattern

**In-process semaphores are not enough to stop shared quota bursts when multiple repos run concurrently; 429s need a cross-process provider cooldown.**

The April 3 Gemini anomaly cluster hit several active projects at once, which
means each process could respect its own semaphore and still collectively hammer
the same provider quota. `llm_client.utils.rate_limit` now combines a lower
default Google concurrency ceiling with a SQLite-backed shared cooldown window
that every process/worktree checks before starting the next Gemini call. The
execution kernel registers that cooldown on any 429-like failure, including
quota-exhausted errors that are intentionally not retried.
