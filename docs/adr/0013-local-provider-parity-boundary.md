# ADR 0013: Local-Provider Parity Boundary

Status: Accepted
Date: 2026-03-21
Last verified: 2026-03-21

## Context

`llm_client` already exposes partial local-model support through direct
`ollama/...` model identifiers and compatibility routing helpers. That support
is real, but it is not yet a first-class architectural contract.

Without a durable boundary, local execution risks becoming:

1. an undocumented exception path,
2. a second policy system outside the shared task-based model registry,
3. an excuse for silent cloud fallback when locality actually matters,
4. a source of inconsistent observability and failure semantics.

At the same time, the broader workspace is standardizing local-first operation
in adjacent layers such as shared open-web retrieval. The shared LLM runtime
needs an equally explicit position on local execution.

## Decision

1. Local-provider support is a first-class responsibility of the `llm_client`
   runtime substrate.
2. The first governed local-provider family is **Ollama**. Other local
   providers are deferred until one provider family is fully specified and
   proven.
3. Explicit local model selection (for example `model="ollama/llama3.1"`) must
   not silently fall back to cloud execution.
4. Task-based model policy may expose explicit locality intent (`auto`,
   `prefer_local`, `require_local`, `cloud_only`) instead of treating local
   execution as an implicit side effect.
5. Local-provider capabilities must be explicit. Unsupported capabilities fail
   loudly with machine-readable metadata instead of being guessed or silently
   emulated.
6. Local execution must use the same shared observability envelope as cloud
   execution, with additional fields that truthfully identify local-runtime
   behavior.
7. The detailed v0 contract lives in
   [`../LOCAL_MODEL_PARITY_V0.md`](../LOCAL_MODEL_PARITY_V0.md).

## Consequences

Positive:
1. Local execution becomes part of the shared runtime contract instead of a
   private compatibility trick.
2. Projects can ask for local-first operation without inventing their own
   routing layer.
3. Failure semantics become clearer when locality is required for cost,
   privacy, or portability reasons.
4. The design aligns with the broader local-first direction without expanding
   `llm_client` into unrelated retrieval or browser infrastructure.

Negative:
1. The model registry and routing metadata will need to grow carefully.
2. Some current compatibility shortcuts, especially shell-based Ollama
   detection, should be retired or demoted.
3. Tool-calling parity for local models must remain conservative unless proven,
   which may disappoint callers expecting cloud-equivalent behavior by default.

## Testing Contract

1. Tests must prove that explicit local model overrides never fall back to
   cloud silently.
2. Tests must prove availability and capability failures are machine-readable.
3. Tests must prove local execution emits distinct locality metadata in shared
   observability records.
4. Any supported local capability must be covered by parity tests against the
   declared v0 contract in `LOCAL_MODEL_PARITY_V0.md`.
