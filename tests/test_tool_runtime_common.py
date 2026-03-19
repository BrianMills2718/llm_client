"""Tests for shared tool-runtime support helpers.

These tests verify the narrow seam extracted from ``mcp_agent.py`` so direct
tool execution and MCP-backed execution can share the same record and contract
helpers without importing one another's runtime modules.
"""

from __future__ import annotations

import llm_client.mcp_agent as mcp_agent
from llm_client.tool_runtime_common import (
    MCPAgentResult,
    MCPToolCallRecord,
    append_input_examples_to_description,
    extract_usage_counts,
    normalize_tool_contracts,
    normalize_tool_input_examples,
    truncate_text,
)


def test_normalize_tool_input_examples_bounds_and_serializes() -> None:
    """Normalize at most two examples and serialize structured payloads."""
    examples = normalize_tool_input_examples(
        [
            {"query": "alpha", "limit": 3},
            " beta ",
            {"ignored": True},
        ]
    )

    assert len(examples) == 2
    assert '"query": "alpha"' in examples[0]
    assert examples[1] == "beta"


def test_append_input_examples_to_description_keeps_empty_base_clean() -> None:
    """Append examples without emitting empty preamble noise."""
    description = append_input_examples_to_description("", ["alpha", "beta"])
    assert description == "Input examples:\n- alpha\n- beta"


def test_truncate_text_adds_loud_marker() -> None:
    """Truncation should preserve the boundary and note the cut loudly."""
    assert truncate_text("abcdefghij", 5) == "abcde\n... [truncated at 5 chars]"


def test_normalize_tool_contracts_shared_alias_matches_mcp_runtime() -> None:
    """The shared helper and MCP runtime should expose the same contract behavior."""
    raw_contracts = {
        "search": {
            "call_modes": [
                {
                    "mode": "search",
                    "when_args_present_any": ["query"],
                    "artifact_prereqs": "none",
                    "produces": ["CHUNK_SET"],
                }
            ]
        }
    }

    assert normalize_tool_contracts(raw_contracts) == mcp_agent._normalize_tool_contracts(raw_contracts)


def test_extract_usage_counts_supports_openai_and_anthropic_keys() -> None:
    """Shared usage extraction should accept both common provider key styles."""
    assert extract_usage_counts({"prompt_tokens": 10, "completion_tokens": 4}) == (10, 4, 0, 0)
    assert extract_usage_counts({"input_tokens": 7, "output_tokens": 3, "cached_tokens": 2}) == (7, 3, 2, 0)


def test_mcp_agent_reexports_shared_tool_record_for_compatibility() -> None:
    """Existing MCP-runtime imports should still resolve the shared record type."""
    assert mcp_agent.MCPToolCallRecord is MCPToolCallRecord


def test_mcp_agent_reexports_shared_agent_result_for_compatibility() -> None:
    """Existing agent-result imports should still resolve the shared result type."""
    assert mcp_agent.MCPAgentResult is MCPAgentResult
