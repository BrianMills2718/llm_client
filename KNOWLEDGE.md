# Operational Knowledge — llm_client

Shared findings from all agent sessions. Any agent brain can read and append.
Human-reviewed periodically.

## Findings

<!-- Append new findings below this line. Do not overwrite existing entries. -->
<!-- Format: ### YYYY-MM-DD — {agent} — {category}                          -->
<!-- Categories: bug-pattern, performance, schema-gotcha, integration-issue, -->
<!--             workaround, best-practice                                   -->
<!-- Agent names: claude-code, codex, openclaw                               -->

### 2026-03-31 — codex — integration-issue
The `llm_client.agent_spec` compatibility shim cannot rely on
`scripts.meta.agent_spec` being importable from `project-meta`, because
`project-meta/scripts/meta/` is not packaged as an importable subpackage.
When `llm_client` needs that shim during tests or worktree execution, the
truthful fallback is to load `project-meta/scripts/meta/agent_spec.py`
directly by file path instead of assuming package importability.

---
