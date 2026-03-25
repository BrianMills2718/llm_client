# llm_client Package

This subtree owns the runtime substrate package. Keep changes here specific to
the package surface, not the repo-wide governance layer.

Read these first when working under `llm_client/`:

1. [`../CLAUDE.md`](../CLAUDE.md)
2. [`../docs/API_REFERENCE.md`](../docs/API_REFERENCE.md)
3. [`../docs/API_REFERENCE.html`](../docs/API_REFERENCE.html)
4. [`../scripts/meta/generate_api_reference.py`](../scripts/meta/generate_api_reference.py)
5. [`../docs/guides/`](../docs/guides/) for advanced usage guides

## Local Surfaces

| Subdir | Purpose |
|--------|---------|
| [`core/`](core/) | Types, config, errors, models, dispatch hub |
| [`execution/`](execution/) | Call lifecycle, runtimes, retry, streaming |
| [`agent/`](agent/) | MCP loop, contracts, tools, turn lifecycle |
| [`sdk/`](sdk/) | Agent SDK adapters (Claude, Codex) |
| [`tools/`](tools/) | Tool registry, result cleaning, shim, utils |
| [`utils/`](utils/) | Cost, git, OpenRouter, rate limiting |
| [`observability/`](observability/) | Event, run, and query adapters around `io_log` |
| [`cli/`](cli/) | CLI command modules for runtime and observability inspection |

Prompt assets live in `~/projects/prompts/` (external, shared across projects).
The `render_prompt()` engine and `prompt_assets.py` resolution logic live here.
Override with `LLM_CLIENT_PROMPT_ASSET_ROOT` env var.

## Working Rules

1. Keep package modules typed, documented, and boundary-focused.
2. Keep CLI modules thin; prefer routing to shared package APIs over re-
   implementing behavior in commands.
3. Keep data directories data-only unless a local routing doc is needed.
4. When a local rule changes, update this file and the leaf subtree file rather
   than repeating parent policy.
5. Regenerate the API reference after changing any public module docstring,
   signature, or export surface.
