# Plan 27 — Long Retry-Hint Failover

## Gap

Gemini quota `429 RESOURCE_EXHAUSTED` responses can include `google.rpc.RetryInfo`
windows measured in hours. Plan #25 stopped same-call retries for obvious spend-cap
and daily-cap messages, and Plan #26 suppressed recently exhausted models across
future calls. But quota-flavored `429`s with explicit retry hints still counted as
retryable inside the current call, so long-running batch jobs could sleep on a
single exhausted model for many minutes instead of failing over to the next model
in the chain.

## Decision

- Keep short provider retry hints retryable.
- Treat long retry hints as temporary model unavailability instead of in-call
  retry.
- Preserve fallback routing so the current request can continue on a backup
  model.
- Keep the threshold configurable via environment to avoid hardcoded policy.

## Acceptance Criteria

- A quota-like `429` with a short retry hint (for example `14s`) still retries.
- A quota-like `429` with a long retry hint (for example `34820s`) does not
  sleep/retry inside the same call and instead bubbles out for fallback.
- The exhausted model remains suppressible in routing for later calls.
- Targeted retry/routing tests pass.

## Implementation

1. Add a configurable in-call retry-hint ceiling in `execution/retry.py`.
2. Use uncapped provider retry hints for retryability classification so hour-scale
   waits can be rejected even though normal retry delays remain capped.
3. Extend model-unavailability cooldown logic to honor provider retry hints when
   they exceed the default cooldown.
4. Add tests for short-hint retry, long-hint failover classification, and
   provider-hint cooldown propagation.

## Verification

```bash
pytest -q tests/test_client.py -k 'quota_retry_delay'
pytest -q tests/test_routing.py tests/test_execution_kernel.py
```
