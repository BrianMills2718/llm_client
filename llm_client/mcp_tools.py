"""Tool utility functions for the MCP agent loop.

Provides tool-call name normalization, budget accounting, argument
extraction/mutation, error signature generation, and runtime artifact
read helpers.  These are pure stateless helpers consumed by the main
agent loop and its decomposed sub-modules.
"""

from __future__ import annotations

import json as _json
import re
from typing import Any

from llm_client.agent_artifacts import (
    _runtime_artifact_read_contract as _agent_runtime_artifact_read_contract,
    _runtime_artifact_read_result as _agent_runtime_artifact_read_result,
    _runtime_artifact_read_tool_def as _agent_runtime_artifact_read_tool_def,
)
from llm_client.tool_runtime_common import (
    MCPToolCallRecord,
    TOOL_REASONING_FIELD,
    truncate_text as _truncate,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUDGET_EXEMPT_TOOL_NAMES: frozenset[str] = frozenset({
    "todo_write",
    "submit_answer",
    "runtime_artifact_read",
})
"""Tools exempt from max_tool_calls budgeting (planning + final submit)."""

AUTO_REASONING_TOOL_DEFAULTS: dict[str, str] = {
    "todo_write": "Replace TODO list to track reasoning progress across atoms.",
    "submit_answer": "Submit the current best factual answer from gathered evidence.",
    "runtime_artifact_read": "Recover previously produced typed artifacts by artifact_id when older tool payloads were cleared from active context.",
}
"""Deterministic fallback reasoning for idempotent planning tools."""

RUNTIME_ARTIFACT_READ_TOOL_NAME = "runtime_artifact_read"
"""Internal runtime tool for reopening typed artifact payloads by stable artifact_id."""


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------

def _is_budget_exempt_tool(tool_name: str) -> bool:
    """Return True when tool is excluded from max_tool_calls accounting."""
    return tool_name in BUDGET_EXEMPT_TOOL_NAMES


def _normalized_tool_name(raw_name: Any) -> str:
    """Normalize tool names for stable matching (budgeting, validation, execution)."""
    return str(raw_name or "").strip()


def _normalize_tool_call_name_inplace(tool_call: dict[str, Any]) -> str:
    """Normalize one tool call name in-place and return the normalized value."""
    fn = tool_call.get("function")
    if not isinstance(fn, dict):
        return ""
    normalized = _normalized_tool_name(fn.get("name"))
    fn["name"] = normalized
    return normalized


# ---------------------------------------------------------------------------
# Budget accounting
# ---------------------------------------------------------------------------

def _count_budgeted_records(records: list[MCPToolCallRecord]) -> int:
    """Count tool calls that consume max_tool_calls budget."""
    return sum(1 for r in records if not _is_budget_exempt_tool(r.tool))


def _count_budgeted_tool_calls(tool_calls: list[dict[str, Any]]) -> int:
    """Count proposed LLM tool calls that consume max_tool_calls budget."""
    used = 0
    for tc in tool_calls:
        tool_name = _normalized_tool_name(tc.get("function", {}).get("name", ""))
        if tool_name and not _is_budget_exempt_tool(tool_name):
            used += 1
    return used


def _trim_tool_calls_to_budget(
    tool_calls: list[dict[str, Any]],
    budget_remaining: int,
) -> tuple[list[dict[str, Any]], int]:
    """Keep all budget-exempt tools and trim only budgeted tools over cap."""
    kept: list[dict[str, Any]] = []
    kept_budgeted = 0
    dropped_budgeted = 0
    for tc in tool_calls:
        tool_name = _normalized_tool_name(tc.get("function", {}).get("name", ""))
        if _is_budget_exempt_tool(tool_name):
            kept.append(tc)
            continue
        if kept_budgeted < budget_remaining:
            kept.append(tc)
            kept_budgeted += 1
        else:
            dropped_budgeted += 1
    return kept, dropped_budgeted


# ---------------------------------------------------------------------------
# Error signatures
# ---------------------------------------------------------------------------

def _tool_error_signature(error: str) -> str:
    """Normalize tool error text so repeated failures can be detected."""
    text = (error or "").strip().lower()
    if not text:
        return "unknown error"
    # Keep stable semantic portion while avoiding noisy prefixes.
    if ":" in text:
        text = text.split(":", 1)[1].strip() or text
    text = " ".join(text.split())
    return text[:160]


# ---------------------------------------------------------------------------
# Argument extraction and mutation
# ---------------------------------------------------------------------------

def _extract_tool_call_args(tc: dict[str, Any]) -> dict[str, Any] | None:
    """Parse function-call arguments as dict when possible."""
    raw = tc.get("function", {}).get("arguments", {})
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        candidate = raw
        for _ in range(2):
            try:
                parsed = _json.loads(candidate)
            except Exception:
                return None
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, str):
                candidate = parsed
                continue
            break
    return None


def _set_tool_call_args(tc: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of tc with updated function arguments preserving argument wire type."""
    out = dict(tc)
    fn = dict(out.get("function", {}))
    raw = fn.get("arguments", {})
    if isinstance(raw, str):
        fn["arguments"] = _json.dumps(args)
    else:
        fn["arguments"] = args
    out["function"] = fn
    return out


# ---------------------------------------------------------------------------
# JSON result parsing
# ---------------------------------------------------------------------------

def _parse_record_result_json_value(record: MCPToolCallRecord) -> Any | None:
    """Best-effort parse of JSON tool result payloads preserving the top-level type."""
    if not record.result or not isinstance(record.result, str):
        return None
    try:
        parsed = _json.loads(record.result)
    except Exception:
        return None
    return parsed


def _parse_record_result_json(record: MCPToolCallRecord) -> dict[str, Any] | None:
    """Best-effort parse of JSON tool result payloads when a dict payload is required."""
    parsed = _parse_record_result_json_value(record)
    return parsed if isinstance(parsed, dict) else None


# ---------------------------------------------------------------------------
# Runtime artifact read helpers
# ---------------------------------------------------------------------------

def _runtime_artifact_read_tool_def() -> dict[str, Any]:
    """Return the OpenAI-format tool definition for the runtime artifact read tool."""
    return _agent_runtime_artifact_read_tool_def(
        tool_name=RUNTIME_ARTIFACT_READ_TOOL_NAME,
        tool_reasoning_field=TOOL_REASONING_FIELD,
    )


def _runtime_artifact_read_contract() -> dict[str, Any]:
    """Return the contract definition for the runtime artifact read tool."""
    return _agent_runtime_artifact_read_contract()


def _runtime_artifact_read_result(
    *,
    artifact_registry_by_id: dict[str, dict[str, Any]],
    tc: dict[str, Any],
    max_result_length: int,
    require_tool_reasoning: bool,
) -> tuple[MCPToolCallRecord, dict[str, Any]]:
    """Execute a runtime_artifact_read tool call against the in-memory artifact registry."""
    return _agent_runtime_artifact_read_result(
        artifact_registry_by_id=artifact_registry_by_id,
        tc=tc,
        max_result_length=max_result_length,
        require_tool_reasoning=require_tool_reasoning,
        tool_name=RUNTIME_ARTIFACT_READ_TOOL_NAME,
        tool_reasoning_field=TOOL_REASONING_FIELD,
        record_factory=MCPToolCallRecord,
        extract_tool_call_args=_extract_tool_call_args,
        truncate_text=_truncate,
    )


# ---------------------------------------------------------------------------
# Autofill tool reasoning
# ---------------------------------------------------------------------------

def _autofill_tool_reasoning(
    tc: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Autofill tool_reasoning for select deterministic tools when omitted."""
    tool_name = tc.get("function", {}).get("name", "")
    fallback = AUTO_REASONING_TOOL_DEFAULTS.get(tool_name)
    if not fallback:
        return tc, False

    args = _extract_tool_call_args(tc)
    if not isinstance(args, dict):
        return tc, False

    existing = args.get(TOOL_REASONING_FIELD)
    if isinstance(existing, str) and existing.strip():
        return tc, False

    patched_args = dict(args)
    patched_args[TOOL_REASONING_FIELD] = fallback
    return _set_tool_call_args(tc, patched_args), True
