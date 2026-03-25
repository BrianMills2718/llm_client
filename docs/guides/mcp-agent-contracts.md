# MCP Agent Composability Contracts

When using MCP agent loops (`mcp_servers=...` or `mcp_sessions=...`), you can enforce
tool-chain legality and expose only currently legal tools.

## Basic usage

```python
result = await acall_llm(
    "openrouter/deepseek/deepseek-chat",
    messages,
    task="mcp_contracts",
    trace_id="mcp_contracts",
    max_budget=5.00,
    mcp_servers={...},
    enforce_tool_contracts=True,
    progressive_tool_disclosure=True,
    initial_artifacts=("QUERY_TEXT",),
    tool_contracts={
        "entity_onehop": {
            "requires_all": [{"kind": "ENTITY_SET", "ref_type": "id"}],
            "produces": [{"kind": "ENTITY_SET", "ref_type": "id"}],
        },
        "chunk_get_text_by_entity_ids": {
            "artifact_prereqs": "none",
            "requires_all": [{"kind": "ENTITY_SET", "ref_type": "id"}],
            "produces": [{"kind": "CHUNK_SET", "ref_type": "fulltext"}],
        },
        "chunk_get_text_by_chunk_ids": {
            "artifact_prereqs": "none",
            "produces": [{"kind": "CHUNK_SET", "ref_type": "fulltext"}],
        },
        "todo_write": {"is_control": True},
    },
    # Reliability controls:
    forced_final_max_attempts=3,
    forced_final_circuit_breaker_threshold=2,
    finalization_fallback_models=["openrouter/openai/gpt-5-mini"],
    retrieval_stagnation_turns=4,
)

raw = result.raw_response  # MCPAgentResult
print(raw.metadata["primary_failure_class"])
print(raw.metadata["first_terminal_failure_event_code"])
```

## Contract shape

- `artifact_prereqs: "none"` declares a self-contained tool for artifact-state checks only.
- There is no hidden tool-name fallback for artifact prereq bypass; if a tool should
  be callable from explicit IDs alone, declare that behavior in its contract.

### Multi-mode tools

For multi-mode tools, use declarative `call_modes` instead of runtime special cases:

```python
"chunk_get_text": {
    "requires_all": ["CHUNK_SET"],
    "call_modes": [
        {
            "name": "by_chunk_id",
            "when_arg_equals": {"mode": "by_chunk_id"},
            "artifact_prereqs": "none",
            "produces": [{"kind": "CHUNK_SET", "ref_type": "fulltext"}],
        },
        {
            "name": "by_entity_ids",
            "when_args_present_any": ["entity_ids", "entity_id"],
            "artifact_prereqs": "none",
            "produces": [{"kind": "CHUNK_SET", "ref_type": "fulltext"}],
        },
    ],
}
```

`call_modes` are resolved per call from arguments. Prefer explicit split tools at
the LLM boundary and reserve `call_modes` for discriminated-union style contracts.

### Handle inputs

Tools can consume previously emitted typed artifact handles declaratively:

```python
"extract_date_mentions_from_artifacts": {
    "artifact_prereqs": "none",
    "handle_inputs": [
        {
            "arg": "chunk_artifact_ids",
            "inject_arg": "chunk_artifacts",
            "representation": "payload",
            "accepts": [{"kind": "CHUNK_SET", "ref_type": "fulltext"}],
        }
    ],
    "produces": [{"kind": "CHUNK_SET", "ref_type": "fulltext"}],
}
```

The model passes stable `artifact_id` values. The runtime resolves handles from the
active artifact registry and injects typed artifact payloads before executor dispatch.

## Metadata fields

Results include:
- `primary_failure_class`, `first_terminal_failure_event_code`, `failure_event_code_counts`
- `available_capabilities_final`
- `hard_bindings_hash`, `full_bindings_hash`
- `lane_closure_analysis["lane_closed"]`
- `tool_disclosure_repair_suggestions`

## Reliability controls

- `finalization_fallback_models=[...]` — fallback models for finalization only (no tool calls)
- `forced_final_max_attempts` / `forced_final_circuit_breaker_threshold` — bound forced-final attempts
- `retrieval_stagnation_turns` — terminate long evidence loops producing no new evidence
- `active_artifact_context_enabled` — rolling artifact-context summaries (default: on)
- `runtime_artifact_read` — built-in tool to reopen prior typed artifacts by `artifact_id`
- MCP loop default completion cap: `8192` tokens (override with `LLM_CLIENT_MCP_MAX_COMPLETION_TOKENS`)

## Tool registry lint

```bash
python -m llm_client tool-lint \
  --module /path/to/tools.py \
  --tool-list-var DIRECT_TOOLS \
  --contracts-var TOOL_CONTRACTS \
  --fail-on-warning
```

Lint checks: missing descriptions, missing input examples, invalid call_modes, missing
handle_inputs args on direct tools.
