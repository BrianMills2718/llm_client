# ADR 0009: Long-Thinking Responses Background Polling

Status: Accepted  
Date: 2026-02-23

## Context

`gpt-5.2-pro` can run long-thinking passes that exceed normal request timeouts
when `reasoning_effort` is high/xhigh. Without background execution + polling,
the client can fail or surface incomplete responses while work is still in
progress.

## Decision

1. `gpt-5.2-pro` is treated as a Responses-API model in `llm_client/client.py`.
2. For long-thinking effort levels (`high`, `xhigh`), Responses requests enable
   `background=true`.
3. When initial Responses status is non-terminal, the client polls by
   `response_id` until terminal completion/failure or timeout.
4. Polling controls are user-tunable via call kwargs:
   - `background_timeout` (default 900s),
   - `background_poll_interval` (default 15s).
5. Poll retrieval uses OpenAI SDK clients (`OpenAI` / `AsyncOpenAI`) because the
   current LiteLLM runtime exposes `responses()` as a function without a
   `.retrieve` method in this environment.
6. Background retrieval validates `api_base` and fails fast for non-OpenAI
   endpoints (for example OpenRouter), instead of retrying until timeout.
7. Routing traces expose `background_mode` to support lightweight adoption
   telemetry.

## Consequences

Positive:
1. Long-thinking calls are resilient to normal request timeout windows.
2. Deterministic behavior for sync and async paths.
3. Operators can tune polling latency/timeout tradeoffs per call.

Negative:
1. Polling introduces longer wall-clock runtimes for deep-review calls.
2. Retrieval currently depends on OpenAI SDK client semantics for background
   lookup.

## Uncertainties

1. LiteLLM may expose first-class background retrieval in future versions; if
   that happens, retrieval strategy should be revisited to reduce duplicate
   client logic.
2. Background semantics for non-OpenAI providers are intentionally out-of-scope
   for this ADR.

## Testing Contract

1. Unit tests must cover:
   - `gpt-5.2-pro` Responses detection,
   - background/reasoning kwargs emission,
   - sync and async polling handoff on non-terminal initial statuses.
2. Retrieval helper tests must validate:
   - key-presence requirements,
   - OpenAI client invocation shape (api key/base URL/timeout).
