# Tests Directory

Pytest suite for `llm_client`.

## Running Tests

```bash
# Full suite (default in this repo)
pytest -q

# Full suite with verbose output
pytest tests/ -v

# Exclude integration-marked tests (offline-safe local run)
pytest -m "not integration" -q

# Run integration tests explicitly
LLM_CLIENT_INTEGRATION=1 pytest -m integration -q

# Run long-thinking smoke only (extra opt-in)
LLM_CLIENT_INTEGRATION=1 LLM_CLIENT_LONG_THINKING_SMOKE=1 pytest -m integration tests/integration_long_thinking_smoke_test.py -q

# Single test
pytest tests/test_client.py::test_call_llm_happy_path -q
```

## Conventions

1. Keep tests deterministic and offline-safe by default.
2. Mark real network tests with `@pytest.mark.integration`.
3. Prefer focused unit/contract assertions; avoid brittle snapshot-style checks.
4. When behavior contracts change, update both tests and relevant ADR/docs together.
