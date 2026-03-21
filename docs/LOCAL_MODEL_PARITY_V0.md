# Local Model Parity v0

This document is the concrete design for first-class local-provider support in
`llm_client`.

It is not an ADR. ADR 0013 records the stable architectural decision. This
document explains the target contract in enough detail to guide future
implementation and verification.

## 1. Requirements

### Core Requirement

Any project should be able to use `llm_client` for local LLM and embedding work
without creating a second runtime stack, a second observability surface, or a
special-case bypass around the shared model-selection and failure-semantics
contract.

### Functional Requirements

1. Local providers are first-class runtime citizens, not compatibility
   exceptions.
2. Local execution uses the same top-level APIs:
   - `call_llm`
   - `call_llm_structured`
   - `call_llm_with_tools`
   - `stream_llm`
   - `embed`
3. Model selection can intentionally:
   - allow local or cloud,
   - prefer local,
   - require local,
   - forbid local.
4. Local execution is observable through the same shared run metadata and trace
   surfaces.
5. Unsupported local capabilities fail loudly and explicitly.

### Quality Requirements

1. No silent cloud fallback when the caller explicitly requested a local model.
2. Capability claims must be explicit per provider/model family rather than
   inferred from marketing names.
3. Availability detection must use runtime-relevant health checks, not shell
   convenience heuristics.
4. The design must preserve the existing `llm_client` rule: one task-based
   model policy surface, not a second independent local-policy subsystem.

## 2. Boundary

`llm_client` owns the local-provider runtime contract:

- provider identity
- model selection and locality policy
- local transport invocation
- capability metadata
- shared failure semantics
- shared observability

`llm_client` does not own:

- local search/fetch/extract infrastructure
- browser automation for retrieval
- project-specific decisions about when retrieved content should be summarized,
  ranked, or synthesized

That boundary matches the new `open_web_retrieval` architecture: retrieval owns
open-web data access; `llm_client` owns optional local or cloud LLM execution
used by higher-level systems.

## 3. Current Facts

Today `llm_client` already has partial local support:

1. direct `ollama/<model>` strings are valid call targets
2. `difficulty.py` can prefer Ollama when available
3. task-graph docs already mention local models for cheap/simple tasks

That support is not yet first-class because:

1. availability detection is mostly a compatibility heuristic (`ollama list`)
2. locality preference is not a governed shared policy surface
3. no explicit parity contract exists across completion, structured output,
   embeddings, streaming, and tool calling
4. local capability and failure metadata are not documented as a stable
   contract

## 4. v0 Provider Scope

### First-Class v0 Provider Family

**Ollama only.**

Why:

1. it already appears in the runtime surface
2. it already has a partial selection path
3. it is the simplest local-first provider to standardize first

### Explicitly Deferred From v0

1. vLLM
2. LM Studio
3. llama.cpp direct integration
4. custom OpenAI-compatible local endpoints treated as first-class named
   providers

These may be added later, but not before one provider family is fully governed.

## 5. Identity And Policy Surface

### Model Identity

The canonical model string remains:

- `ollama/<model_name>`

Examples:

- `ollama/llama3.1`
- `ollama/qwen2.5:7b`

The model string remains the runtime-facing identifier. The registry adds
metadata; it does not replace this identity.

### Locality Policy

Task-based selection should gain one explicit locality mode:

- `auto`: local and cloud are both allowed; choose by task policy
- `prefer_local`: prefer local candidates that satisfy the task contract, but
  fall back to cloud if none qualify
- `require_local`: local execution is mandatory; fail loudly if unavailable
- `cloud_only`: exclude local candidates

This must integrate into the existing task-based selection path rather than
creating a separate local-selection API.

### Explicit Override Rule

If the caller explicitly sets `model="ollama/..."`:

1. that is semantically equivalent to `require_local`
2. cloud fallback is forbidden
3. unavailability must raise a local-provider error, not silently route remote

## 6. Registry And Capability Contract

The registry must express local-provider capability explicitly.

### Required Registry Fields

Existing fields remain. v0 adds these semantics:

- `provider`: `ollama`
- `tags`: should include `local`

### Required New Metadata

The registry should add:

- `deployment`: `local | hosted | agent_sdk`
- `supports_embeddings`: `bool`
- `supports_streaming`: `bool`
- `supports_tool_calling_native`: `bool`
- `supports_structured_output_native`: `bool`
- `availability_probe`: symbolic probe name, not raw shell command

The critical distinction is:

- **native capability**: provider/model can do the thing directly
- **runtime-supported capability**: `llm_client` can still offer the surface by
  using validation, wrapping, or explicit failure

The design must record both truths instead of conflating them.

## 7. Availability Detection

### Rule

Availability must be checked at the provider-runtime boundary, not by shell
convenience.

### Ollama v0 Probe

Use the Ollama HTTP API as the authoritative availability check:

1. resolve base URL from config or environment
2. call a lightweight discovery endpoint such as `/api/tags`
3. verify connectivity and model presence separately

This replaces `ollama list` as the first-class contract.

### Failure Classes

The runtime must distinguish:

1. provider not reachable
2. provider reachable, model missing
3. provider reachable, model present, capability unsupported

## 8. Runtime Surface Expectations

### v0 Required

1. **Basic completion**
   - required
2. **Structured output**
   - required
   - native JSON mode is optional; schema-validated extraction is still allowed
3. **Streaming**
   - required for plain text
4. **Embeddings**
   - required if the provider/model is registered with
     `supports_embeddings=true`

### v0 Conditional

1. **Tool calling**
   - only for explicitly allowlisted Ollama models with
     `supports_tool_calling_native=true`
   - otherwise fail loudly
2. **Streaming with tools**
   - same rule as tool calling

### Explicit v0 Non-Goal

Do not emulate arbitrary tool-calling for local models that do not natively
support it. That would create a shadow agent runtime instead of a clean parity
contract.

## 9. Observability Contract

Local execution must emit the same shared observability envelope plus these
local-runtime fields:

- `execution_location`: `local | remote | agent_sdk`
- `provider_runtime`: e.g. `ollama`
- `provider_endpoint_hash`: hashed/redacted endpoint identity
- `capability_path`: `native | validated | unsupported`
- `locality_policy`: `auto | prefer_local | require_local | cloud_only`

Cost semantics remain honest:

- if cost is unknown or zero for local execution, record that truthfully
- do not fake cloud-like USD costs unless a documented estimation policy exists

## 10. Failure Semantics

v0 should add machine-readable failures for local-provider operations:

- `LLMC_ERR_LOCAL_PROVIDER_UNAVAILABLE`
- `LLMC_ERR_LOCAL_MODEL_UNAVAILABLE`
- `LLMC_ERR_LOCAL_CAPABILITY_UNSUPPORTED`
- `LLMC_ERR_LOCAL_POLICY_CONFLICT`

These should be surfaced through the same `LLMConfigurationError` /
`LLMCapabilityError` taxonomy rather than inventing a disconnected error layer.

## 11. Smallest Real Slice

The smallest real slice that proves the design is:

1. registry metadata for one Ollama model
2. HTTP availability check for Ollama
3. `resolve_model_selection(... locality_policy=...)` behavior
4. explicit no-cloud-fallback for `model="ollama/..."`
5. one verified basic completion path
6. one verified structured-output path
7. observability fields proving the call ran locally

Everything after that builds outward from a proven local baseline.

## 12. Verification Matrix

At minimum, parity proof must cover:

1. explicit local override succeeds when provider + model are available
2. explicit local override fails loudly when provider is unavailable
3. `prefer_local` falls back to cloud only when policy allows and no local
   candidate qualifies
4. `require_local` never falls back to cloud
5. structured-output local path returns validated results or explicit errors
6. embedding path records local execution metadata
7. unsupported tool-calling requests fail with explicit capability metadata

## 13. Migration Impact

Once the contract is implemented, downstream repos should stop:

1. shelling out to local model runtimes directly
2. inventing local-versus-cloud routing logic outside `llm_client`
3. treating local execution as an undocumented special case

They should instead use `llm_client` policy and model-selection surfaces the
same way they already use it for hosted providers.

## 14. Open Questions Deferred Beyond v0

1. Should locality policy live in call kwargs, task profiles, or both?
2. How should local cost estimation work, if at all?
3. When should second-provider support begin after Ollama?
4. Should provider availability be cached, and for how long?
