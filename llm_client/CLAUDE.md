# llm_client Package

This subtree owns the runtime substrate package. Keep changes here specific to
the package surface, not the repo-wide governance layer.

Read these first when working under `llm_client/`:

1. [`../CLAUDE.md`](../CLAUDE.md)
2. [`../docs/API_REFERENCE.md`](../docs/API_REFERENCE.md)
3. [`../docs/ECOSYSTEM_TOP_DOWN_ARCHITECTURE.md`](../docs/ECOSYSTEM_TOP_DOWN_ARCHITECTURE.md)

## Local Surfaces

| Subdir | Purpose |
|--------|---------|
| [`cli/`](cli/) | CLI command modules for runtime and observability inspection |
| [`observability/`](observability/) | Event, run, and query adapters around `io_log` |
| [`prompts/`](prompts/) | Prompt asset data and templates |
| [`rubrics/`](rubrics/) | Versioned rubric definitions used by scoring and evaluation |

## Working Rules

1. Keep package modules typed, documented, and boundary-focused.
2. Keep CLI modules thin; prefer routing to shared package APIs over re-
   implementing behavior in commands.
3. Keep data directories data-only unless a local routing doc is needed.
4. When a local rule changes, update this file and the leaf subtree file rather
   than repeating parent policy.
