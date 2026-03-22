# CLI Commands

This subtree contains the `python -m llm_client` command modules.

## Purpose

Keep command entrypoints thin, deterministic, and easy to trace back to the
underlying package APIs. The command surface should expose the runtime substrate
without becoming a second implementation layer.

## What Lives Here

- `common.py` for shared CLI helpers
- `adoption.py`, `backfill.py`, `cost.py`, `experiments.py`, `models.py`,
  `scores.py`, `tool_lint.py`, and `traces.py` for subcommand handlers
- `__init__.py` for subcommand registration

## Local Rules

1. Prefer calling package APIs over duplicating logic in the CLI layer.
2. Keep table/JSON formatting code local when it is purely presentation logic.
3. Keep read-only inspection commands separate from mutation or backfill
   commands.
4. Update the CLI help text when new subcommands or options are added.
