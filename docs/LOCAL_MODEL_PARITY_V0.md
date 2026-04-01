# Local Model Parity v0 — Design Contract

**Status**: Design (not yet implemented)
**Date**: 2026-04-01
**Plan ref**: project-meta Plan #13

---

## Scope

This document defines the first-class local-model contract for llm_client.
It covers what local providers must support, how they integrate with existing
routing and selection, and what parity tests are required.

This is a design document. Implementation is deferred until local-first
operation becomes a blocking priority.

---

## Current State

Local model support exists but is not first-class:

| Capability | Status |
|-----------|--------|
| Basic Ollama completion | Works (via litellm) |
| Difficulty tier integration | Tier 1 uses `ollama/llama3.1` first |
| Runtime availability detection | `_is_ollama_available()` exists |
| Task-based model selection | Not supported (registry is cloud-only) |
| Model registry entries | None — 0 local models in default registry |
| Custom API base config | Not supported (hardcoded to litellm default) |
| Offline-first routing policy | Not supported |
| Structured output | Unknown/untested for local models |
| Tool calling | Unknown/untested for local models |

---

## v0 Provider Scope

**In scope (v0):**
- Ollama (any model available via `ollama list`)

**Explicitly deferred:**
- vLLM
- LM Studio
- llama.cpp direct
- Other local inference servers

**Rationale:** Ollama is already partially supported and has the broadest
model compatibility. Expanding to other runtimes adds provider-specific
complexity without immediate ecosystem value.

---

## v0 Capability Matrix

The following call modes must be verified for the declared local provider:

| Surface | Required for v0? | Notes |
|---------|-----------------|-------|
| Basic completion | Yes | Already works |
| Structured output (json_schema) | Yes | Critical for extraction pipelines |
| Tool calling | No (deferred) | Few local models support it reliably |
| Streaming | Yes | Already works via litellm |
| Embeddings | No (deferred) | Most local embedding workflows use sentence-transformers, not Ollama |
| Observability/trace capture | Yes | Must log to same SQLite DB as cloud calls |

---

## Model Registry Integration

### Current gap

`get_model(task="extraction")` ignores local models because the default
registry (`data/default_model_registry.json`) has zero Ollama entries.

### Design decision

Add Ollama models to the default registry with realistic capability metadata:

```json
{
  "model_id": "ollama/llama3.1",
  "provider": "ollama",
  "intelligence": 2,
  "speed": 4,
  "cost_per_1k_input": 0.0,
  "cost_per_1k_output": 0.0,
  "supports_structured_output": true,
  "supports_tool_calling": false,
  "supports_streaming": true,
  "max_context": 128000,
  "local": true
}
```

The `local: true` flag enables routing policy to prefer or avoid local models.

### Routing policy

Add a routing preference to the model selection logic:

```python
class RoutingPreference(str, Enum):
    CLOUD_FIRST = "cloud_first"      # current default
    LOCAL_FIRST = "local_first"       # prefer local when available
    LOCAL_ONLY = "local_only"         # fail if local unavailable
    CLOUD_ONLY = "cloud_only"         # never use local
```

Controlled by env var `LLM_CLIENT_ROUTING_PREFERENCE=cloud_first` (default).

---

## Configuration

### New env vars

| Var | Default | Purpose |
|-----|---------|---------|
| `LLM_CLIENT_ROUTING_PREFERENCE` | `cloud_first` | Routing preference |
| `OLLAMA_API_BASE` | `http://localhost:11434` | Ollama server URL |

### Existing behavior preserved

- `ollama/model` strings continue to work directly
- Difficulty tier system continues to check Ollama availability
- Rate limiting at 5 concurrent requests preserved

---

## Availability Detection

### Current

`_is_ollama_available()` shells out to `ollama list` with a 5s timeout.

### v0 improvement

Replace subprocess check with HTTP health check:

```python
def _is_ollama_available() -> bool:
    """Check Ollama availability via HTTP."""
    try:
        resp = httpx.get(f"{OLLAMA_API_BASE}/api/tags", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False
```

Benefits: faster (2s vs 5s), no subprocess, works with remote Ollama servers.

---

## Observability Contract

Local provider calls must:

1. Log to the same `llm_observability.db` SQLite database
2. Include `provider: "ollama"` in the record
3. Report `cost_usd: 0.0` (local execution is free)
4. Report latency consistently with cloud calls
5. Classify failures into the same failure taxonomy

This is already mostly true (litellm handles logging). The explicit requirement
is that observability queries like `make cost` and `make errors` must include
local calls without special handling.

---

## Failure Semantics

| Scenario | Behavior |
|----------|----------|
| Ollama not running | Fail loud with `OllamaUnavailableError` |
| Model not downloaded | Fail loud with model-not-found error |
| Structured output fails | Fall back to cloud model if `local_first` policy |
| Rate limit exceeded | Queue (existing rate limiter applies) |
| Timeout | Standard timeout handling (configurable) |

**No silent fallbacks.** If a caller explicitly requests `ollama/model`, failure
must be explicit. If routing policy is `local_first`, fallback to cloud is
allowed with a warning log.

---

## Relationship to open_web_retrieval

`open_web_retrieval` may eventually use local models for extraction. The design
boundary is:

- **llm_client** owns: model registry, routing, observability, provider config
- **open_web_retrieval** owns: when to call, what to extract, domain logic

open_web_retrieval should call `llm_client.call_llm("ollama/model", ...)` with
standard kwargs. It should NOT implement its own Ollama client or availability
detection.

---

## Parity Tests

The v0 parity test suite must verify:

1. Basic completion works with an Ollama model
2. Structured output (json_schema) works or fails explicitly
3. Streaming works
4. Observability log entry is created with correct provider metadata
5. Failure when Ollama unavailable is explicit (not silent)
6. `get_model(task=...)` returns an Ollama model when `local_first` policy is set
7. Cost is recorded as $0.00

Tests should be marked `@pytest.mark.skipif(not _is_ollama_available())`
to avoid CI failures when Ollama is not installed.

---

## Implementation Phases (when prioritized)

1. Add Ollama models to default registry with `local: true`
2. Add `LLM_CLIENT_ROUTING_PREFERENCE` env var
3. Replace subprocess availability check with HTTP
4. Add `OLLAMA_API_BASE` env var
5. Write parity test suite
6. Verify structured output with >=2 Ollama models
7. Document in llm_client README
