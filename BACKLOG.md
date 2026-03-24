# llm_client Backlog

## Resolved (2026-03-24)

### Task-Level Progress Observability
**Resolved by:** Plan #14 (BatchProgressTracker, stagnation detection, item_timeout_s)

### litellm JSON Schema Validation + Retryable
**Resolved by:** Commit `48ea0b6` — enabled `litellm.enable_json_schema_validation = True`, added `JSONSchemaValidationError` to retryable types.

### _StructuredValidationRetry Repair Prompt
**Resolved by:** Already committed at `8382968`.

### git_utils.py Module Import
**Resolved by:** Stub at root, canonical at `utils/git_utils.py`.

### Pre-commit Hook Race Condition
**Status:** Not reproduced in current session (all commits passed hooks). Monitor.
