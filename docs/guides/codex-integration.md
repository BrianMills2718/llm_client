# Codex Integration

## Workspace agent mode

For code-generation/editing workflows that depend on workspace side effects,
set `execution_mode="workspace_agent"` to prevent accidental routing to
chat-only models.

```python
result = call_llm(
    "codex/gpt-5",
    messages,
    execution_mode="workspace_agent",
    task="codex_demo",
    trace_id="codex_demo",
    max_budget=5.00,
)
```

Default agent settings:
- `codex`: `approval_policy="never"`, `skip_git_repo_check=True`
- `claude-code`: `permission_mode="bypassPermissions"`

For fully headless agent runs, `yolo_mode=True` is a convenience flag.

## Process isolation (hang containment)

For `codex` calls that occasionally become cancellation-unresponsive under long
tool loops, run non-streaming turns in a dedicated worker process:

```python
result = call_llm(
    "codex/gpt-5",
    messages,
    execution_mode="workspace_agent",
    task="codex_isolation",
    trace_id="codex_isolation",
    max_budget=5.00,
    codex_process_isolation=True,
    codex_process_start_method="fork",   # optional
    codex_process_grace_s=3.0,           # optional
)
```

Or via environment:

```bash
export LLM_CLIENT_CODEX_PROCESS_ISOLATION=1
export LLM_CLIENT_CODEX_PROCESS_START_METHOD=fork
export LLM_CLIENT_CODEX_PROCESS_GRACE_S=3.0
```

## Transport fallback

Three transport modes:
- `codex_transport="sdk"`: SDK only.
- `codex_transport="cli"`: `codex exec` directly.
- `codex_transport="auto"`: prefer SDK, fall back to CLI on failure.

If timeouts are globally disabled (`LLM_CLIENT_TIMEOUT_POLICY=ban`), pair
auto transport with `agent_hard_timeout`:

```python
result = call_llm(
    "codex",
    messages,
    execution_mode="workspace_agent",
    task="codex_transport",
    trace_id="codex_transport",
    max_budget=5.00,
    codex_transport="auto",
    agent_hard_timeout=300,
    model_reasoning_effort="medium",
)
```

## Reasoning effort

- `model_reasoning_effort=minimal` is often rejected on ChatGPT-account Codex
  lanes. `llm_client` coerces `minimal -> low` by default.
- Set `LLM_CLIENT_CODEX_ALLOW_MINIMAL_EFFORT=1` to force minimal unchanged.

## Agent billing and retry

- Default billing: `LLM_CLIENT_AGENT_BILLING_MODE=subscription` (cost=0.0, billing_mode="subscription_included")
- API-metered: `LLM_CLIENT_AGENT_BILLING_MODE=api`
- Retries disabled by default (avoid duplicate side effects). Enable with `agent_retry_safe=True` or `LLM_CLIENT_AGENT_RETRY_SAFE=1`.
