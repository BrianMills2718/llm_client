# scripts/meta

This directory contains the `llm_client` implementations behind the
operator-facing wrappers and governance tooling.

## Use This Directory For

- required-reading, plan-validation, and relationships validation logic
- deterministic file-context resolution
- hook logging and governance helpers

## Working Rules

- Prefer changing implementations here over editing wrapper scripts.
- Keep `scripts/relationships.yaml` truthful when governance logic changes.
