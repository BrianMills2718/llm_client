# llm_client package

This subtree is the production runtime substrate for model execution, tool
calling, MCP loops, observability, and agent routing.

## Use This Directory For

- call contract enforcement
- transport/runtime behavior
- agent runtime routing
- observability and result-shape contracts
- prompt and rubric assets shipped with the package

## Route Narrower Work

- CLI entrypoints and operator-facing command surfaces -> `cli/`
- observability storage and reconstruction surfaces -> `observability/`
- packaged prompt artifacts -> `prompts/`
- packaged rubric assets -> `rubrics/`

## Working Rules

- Preserve the required call contract: `task=`, `trace_id=`, and `max_budget=`.
- Add observability whenever control-plane behavior changes.
- Do not bypass `llm_client` with direct provider calls in project code.
