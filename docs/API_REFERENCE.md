# API Reference

Generated from package docstrings and typed signatures.

Browser view: [API_REFERENCE.html](API_REFERENCE.html)

## Start Here

1. [README.md](../README.md) for installation, usage, routing, and examples.
2. [AGENTS.md](../AGENTS.md) for repo operating rules and architectural boundaries.
3. [docs/plans/01_master-roadmap.md](plans/01_master-roadmap.md) for the current long-term program state.

## Core Runtime Surface

- `acall_llm` - Async version of call_llm. Same three-tier routing (Agent SDK / Responses API / Completions).
- `acall_llm_batch` - Run multiple LLM calls concurrently with semaphore-based rate limiting.
- `acall_llm_structured` - Async version of call_llm_structured.
- `acall_llm_structured_batch` - Run multiple structured LLM calls concurrently.
- `acall_llm_with_tools` - Async version of call_llm_with_tools.
- `activate_experiment_run` - No docstring available.
- `activate_feature_profile` - Return the context manager that binds a feature profile for nested calls.
- `aembed` - Async version of embed(). See embed() for full docs.
- `ActiveExperimentRun` - Bind an existing run id as the active experiment context.
- `ActiveFeatureProfile` - Bind a feature profile as the active profile in the current context.
- `AsyncCachePolicy` - Protocol for async LLM response caches (Redis, etc.).
- `AsyncLLMStream` - Async streaming wrapper. Yields text chunks, then exposes ``.result``.

## Module Catalog

Generated from 127 importable modules under `llm_client`.

Open the HTML file for the full module-by-module docs surface.

## Source Of Truth

1. `pyproject.toml` is authoritative for package metadata and extras.
2. Module docstrings and public function signatures are authoritative for code behavior.
3. The roadmap and ADRs are authoritative for architectural boundaries.
