# llm_client/cli

This subtree contains CLI entrypoints and operator-facing command surfaces for
`llm_client`.

## Use This Directory For

- command-line tooling that exposes runtime or observability capabilities

## Working Rules

- Keep CLI behavior aligned with the underlying runtime contracts.
- Prefer thin command surfaces over duplicating business logic in CLI modules.
