"""Common support for tool-execution runtimes.

This module holds the shared record types and normalization helpers used by
multiple tool-execution paths: the MCP loop, the direct Python-tool path, and
the structured-output tool shim. It intentionally excludes loop control,
provider dispatch, and agent policy so lower-level tool helpers do not need to
import the full MCP runtime just to share data contracts.
"""

from __future__ import annotations

import json as _json
from dataclasses import dataclass, field
from typing import Any

from llm_client.agent.agent_contracts import _normalize_tool_contracts as _agent_normalize_tool_contracts

DEFAULT_TOOL_INPUT_EXAMPLES_MAX_ITEMS: int = 2
"""Maximum tool input examples appended to tool descriptions."""

DEFAULT_TOOL_INPUT_EXAMPLE_MAX_CHARS: int = 240
"""Maximum chars retained per tool input example snippet."""

TOOL_REASONING_FIELD: str = "tool_reasoning"
"""Optional argument every tool call can include for action-level observability."""


@dataclass
class MCPToolCallRecord:
    """Record one tool call outcome in a provider-agnostic shape.

    The name stays MCP-prefixed for compatibility because existing callers and
    stored diagnostics already use that term, but the record is shared by both
    MCP-backed and direct in-process tool execution.
    """

    server: str
    tool: str
    arguments: dict[str, Any]
    tool_call_id: str | None = None
    tool_reasoning: str | None = None
    arg_coercions: list[dict[str, Any]] = field(default_factory=list)
    artifact_ids: list[str] = field(default_factory=list)
    artifact_handles: list[dict[str, Any]] = field(default_factory=list)
    result: str | None = None
    error: str | None = None
    latency_s: float = 0.0


@dataclass
class MCPAgentResult:
    """Accumulate agent-style runtime state in a shared result shape.

    This object is stored in ``LLMCallResult.raw_response`` by both the full
    MCP loop and the structured-output tool shim so downstream diagnostics can
    inspect turns, tool calls, warnings, and loop metadata through one stable
    contract.
    """

    tool_calls: list[MCPToolCallRecord] = field(default_factory=list)
    turns: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_tokens: int = 0
    total_cache_creation_tokens: int = 0
    conversation_trace: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    models_used: set[str] = field(default_factory=set)


def normalize_tool_input_examples(raw: Any) -> list[str]:
    """Normalize raw tool input examples into bounded display strings."""
    if raw is None:
        return []
    items: list[Any]
    if isinstance(raw, list):
        items = raw
    else:
        items = [raw]

    normalized: list[str] = []
    for item in items:
        if len(normalized) >= DEFAULT_TOOL_INPUT_EXAMPLES_MAX_ITEMS:
            break
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, (dict, list)):
            try:
                text = _json.dumps(item, ensure_ascii=False)
            except Exception:
                text = str(item).strip()
        else:
            text = str(item).strip()
        if not text:
            continue
        normalized.append(text[:DEFAULT_TOOL_INPUT_EXAMPLE_MAX_CHARS])
    return normalized


def append_input_examples_to_description(description: str, examples: list[str]) -> str:
    """Append bounded input examples to tool description text."""
    base = str(description or "").strip()
    if not examples:
        return base
    bullet_lines = "\n".join(f"- {example}" for example in examples)
    examples_block = "Input examples:\n" + bullet_lines
    if base:
        return f"{base}\n\n{examples_block}"
    return examples_block


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to ``max_length`` chars while preserving a loud marker."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + f"\n... [truncated at {max_length} chars]"


def normalize_tool_contracts(raw: Any) -> dict[str, dict[str, Any]]:
    """Normalize declarative tool contracts for any tool runtime.

    This keeps one stable import point for runtimes that need contract
    normalization without depending on wrappers defined inside ``mcp_agent``.
    """

    return _agent_normalize_tool_contracts(raw)


def extract_usage_counts(usage: dict[str, Any]) -> tuple[int, int, int, int]:
    """Extract input, output, cached, and cache-creation token counts.

    The helper accepts the normalized usage dict already attached to
    ``LLMCallResult`` and handles both OpenAI-style and Anthropic-style token
    keys.
    """

    inp = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
    out = usage.get("output_tokens") or usage.get("completion_tokens") or 0
    cached = usage.get("cached_tokens") or 0
    cache_creation = usage.get("cache_creation_tokens") or 0
    return int(inp), int(out), int(cached), int(cache_creation)
