"""MCP agent loop for llm_client.

Enables any litellm model to act as a tool-calling agent by connecting
to MCP servers, discovering their tools, and running an autonomous
tool-calling loop.

Usage (via call_llm/acall_llm — routing is automatic):
    result = await acall_llm(
        "gemini/gemini-3-flash-preview",
        messages,
        mcp_servers={
            "my-server": {
                "command": "python",
                "args": ["-u", "server.py"],
                "env": {"KEY": "value"},
            }
        },
        max_turns=20,
    )

The loop:
    1. Start MCP server subprocesses (stdio transport)
    2. Discover tools via session.list_tools()
    3. Convert MCP tool schemas to OpenAI function-calling format
    4. Call LLM with tools → if tool_calls → execute via MCP → repeat
    5. Return LLMCallResult with accumulated usage/cost
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import re
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from llm_client.client import LLMCallResult
from llm_client.foundation import (
    check_binding_conflicts,
    coerce_run_id,
    extract_bindings_from_tool_args,
    merge_binding_state,
    new_event_id,
    new_session_id,
    normalize_bindings,
    now_iso,
    sha256_json,
    sha256_text,
    validate_foundation_event,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configurable defaults — all overridable via kwargs to call_llm/acall_llm
# ---------------------------------------------------------------------------

DEFAULT_MAX_TURNS: int = 20
"""Maximum tool-calling loop iterations before forcing a final answer."""

DEFAULT_MAX_TOOL_CALLS: int | None = 20
"""Maximum tool calls before forcing a final answer. None disables tool budgeting."""

TOOL_REASONING_FIELD: str = "tool_reasoning"
"""Optional argument every tool call can include for action-level observability."""

DEFAULT_REQUIRE_TOOL_REASONING: bool = False
"""Hard-fail tool calls missing tool_reasoning when enabled."""

BUDGET_EXEMPT_TOOL_NAMES: frozenset[str] = frozenset({
    "todo_create",
    "todo_update",
    "todo_list",
    "todo_reset",
    "submit_answer",
})
"""Tools exempt from max_tool_calls budgeting (planning + final submit)."""

AUTO_REASONING_TOOL_DEFAULTS: dict[str, str] = {
    "todo_reset": "Reset TODO state for the current question before planning/execution.",
    "todo_list": "Read TODO states to decide the next unblock action.",
}
"""Deterministic fallback reasoning for idempotent planning tools."""

TURN_WARNING_THRESHOLD: int = 3
"""Inject a 'wrap up' system message this many turns before max_turns."""

DEFAULT_MCP_INIT_TIMEOUT: float = 30.0
"""Seconds to wait for each MCP server subprocess to initialize."""

DEFAULT_TOOL_RESULT_MAX_LENGTH: int = 50_000
"""Maximum character length for a single tool result. Longer results are truncated."""

DEFAULT_MAX_MESSAGE_CHARS: int = 260_000
"""Soft cap for serialized message-history size; old tool outputs are compacted above this."""

DEFAULT_ENFORCE_TOOL_CONTRACTS: bool = False
"""When enabled, reject tool calls that violate declared composability contracts."""

DEFAULT_PROGRESSIVE_TOOL_DISCLOSURE: bool = True
"""When enabled with contracts, expose only currently composable tools per turn."""

DEFAULT_INITIAL_ARTIFACTS: tuple[str, ...] = ("QUERY_TEXT",)
"""Default artifact kinds available before any tool call."""

# Kwargs consumed by the MCP agent loop (popped before passing to inner acall_llm)
MCP_LOOP_KWARGS = frozenset({
    "mcp_servers",
    "mcp_sessions",
    "max_turns",
    "max_tool_calls",
    "require_tool_reasoning",
    "mcp_init_timeout",
    "tool_result_max_length",
    "max_message_chars",
    "enforce_tool_contracts",
    "progressive_tool_disclosure",
    "tool_contracts",
    "initial_artifacts",
    "initial_bindings",
})

# Kwargs consumed by the direct tool loop
TOOL_LOOP_KWARGS = frozenset({
    "python_tools",
    "max_turns",
    "max_tool_calls",
    "require_tool_reasoning",
    "tool_result_max_length",
    "max_message_chars",
    "enforce_tool_contracts",
    "progressive_tool_disclosure",
    "tool_contracts",
    "initial_artifacts",
    "initial_bindings",
})


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class MCPToolCallRecord:
    """Record of a single MCP tool call during the agent loop."""

    server: str
    tool: str
    arguments: dict[str, Any]
    tool_reasoning: str | None = None
    arg_coercions: list[dict[str, Any]] = field(default_factory=list)
    result: str | None = None
    error: str | None = None
    latency_s: float = 0.0


@dataclass
class ToolCallValidation:
    """Normalized pre-execution validation outcome for a tool call."""

    is_valid: bool
    reason: str = ""
    error_code: str | None = None
    failure_phase: str | None = None
    call_bindings: dict[str, str | None] = field(default_factory=dict)


@dataclass
class MCPAgentResult:
    """Accumulated result from the MCP agent loop.

    Stored in LLMCallResult.raw_response for introspection.
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
    """Diagnostic warnings accumulated across turns (retries, fallbacks, sticky model switches)."""
    models_used: set[str] = field(default_factory=set)
    """Set of model strings actually used during the agent loop (tracks fallback switches)."""


# ---------------------------------------------------------------------------
# Session Pool — reuse MCP servers across multiple calls
# ---------------------------------------------------------------------------


class MCPSessionPool:
    """Persistent MCP server connections for reuse across multiple acall_llm() calls.

    Usage:
        async with MCPSessionPool(mcp_servers) as pool:
            for question in questions:
                result = await acall_llm(model, msgs, mcp_sessions=pool)
    """

    def __init__(
        self,
        mcp_servers: dict[str, dict[str, Any]],
        init_timeout: float = DEFAULT_MCP_INIT_TIMEOUT,
    ):
        self.mcp_servers = mcp_servers
        self.init_timeout = init_timeout
        self._stack: AsyncExitStack | None = None
        self.sessions: dict[str, Any] = {}
        self.tool_to_server: dict[str, str] = {}
        self.openai_tools: list[dict[str, Any]] = []

    async def __aenter__(self) -> "MCPSessionPool":
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        self.sessions, self.tool_to_server, self.openai_tools = await _start_servers(
            self.mcp_servers, self._stack, self.init_timeout,
        )
        logger.info(
            "MCPSessionPool: started %d servers, %d tools",
            len(self.sessions), len(self.openai_tools),
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._stack:
            await self._stack.__aexit__(*exc)
            self._stack = None
            self.sessions = {}
            self.tool_to_server = {}
            self.openai_tools = []


# ---------------------------------------------------------------------------
# Schema conversion
# ---------------------------------------------------------------------------


def _mcp_tool_to_openai(tool: Any) -> dict[str, Any]:
    """Convert an MCP Tool object to OpenAI function-calling format.

    MCP: {"name": "foo", "description": "...", "inputSchema": {...}}
    OpenAI: {"type": "function", "function": {"name": "foo", "description": "...", "parameters": {...}}}
    """
    parameters = dict(tool.inputSchema or {"type": "object", "properties": {}})
    if not isinstance(parameters, dict):
        parameters = {"type": "object", "properties": {}}
    parameters.setdefault("type", "object")
    properties = parameters.setdefault("properties", {})
    if not isinstance(properties, dict):
        properties = {}
        parameters["properties"] = properties
    properties.setdefault(
        TOOL_REASONING_FIELD,
        {
            "type": "string",
            "description": "Why this specific tool call is needed right now.",
        },
    )
    required = parameters.get("required")
    if not isinstance(required, list):
        required = []
    if TOOL_REASONING_FIELD not in required:
        required.append(TOOL_REASONING_FIELD)
    parameters["required"] = required

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": parameters,
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_usage(usage: dict[str, Any]) -> tuple[int, int, int, int]:
    """Extract (input_tokens, output_tokens, cached_tokens, cache_creation_tokens) from usage dict.

    Handles both OpenAI convention (prompt_tokens/completion_tokens)
    and Anthropic convention (input_tokens/output_tokens).
    Extracts provider-level prompt caching details when available.
    """
    inp = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
    out = usage.get("output_tokens") or usage.get("completion_tokens") or 0
    cached = usage.get("cached_tokens") or 0
    cache_creation = usage.get("cache_creation_tokens") or 0
    return int(inp), int(out), int(cached), int(cache_creation)


def _truncate(text: str, max_length: int) -> str:
    """Truncate text if it exceeds max_length, appending a notice."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + f"\n... [truncated at {max_length} chars]"


def _message_char_length(message: dict[str, Any]) -> int:
    """Best-effort serialized length for one chat message."""
    try:
        return len(_json.dumps(message, ensure_ascii=False, default=str))
    except Exception:
        return len(str(message))


def _compact_tool_history_for_context(
    messages: list[dict[str, Any]],
    max_message_chars: int,
) -> tuple[int, int, int]:
    """Compact verbose historical tool outputs when message history grows too large.

    Returns (compacted_message_count, chars_saved, resulting_chars).
    """
    if max_message_chars <= 0:
        total = sum(_message_char_length(m) for m in messages if isinstance(m, dict))
        return 0, 0, total

    total_chars = sum(_message_char_length(m) for m in messages if isinstance(m, dict))
    if total_chars <= max_message_chars:
        return 0, 0, total_chars

    target_chars = max(int(max_message_chars * 0.75), 32_000)
    compacted = 0
    saved = 0
    replacement = (
        '{"notice":"Earlier tool result compacted to fit context window. '
        'Re-run the tool if this evidence is needed again."}'
    )
    replacement_len = len(replacement)

    for msg in messages:
        if total_chars <= target_chars:
            break
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        content = msg.get("content")
        if not isinstance(content, str) or len(content) <= 512:
            continue
        old_len = len(content)
        if old_len <= replacement_len:
            continue
        msg["content"] = replacement
        delta = old_len - replacement_len
        total_chars -= delta
        saved += delta
        compacted += 1

    return compacted, saved, total_chars


def _is_budget_exempt_tool(tool_name: str) -> bool:
    """Return True when tool is excluded from max_tool_calls accounting."""
    return tool_name in BUDGET_EXEMPT_TOOL_NAMES


def _count_budgeted_records(records: list[MCPToolCallRecord]) -> int:
    """Count tool calls that consume max_tool_calls budget."""
    return sum(1 for r in records if not _is_budget_exempt_tool(r.tool))


def _count_budgeted_tool_calls(tool_calls: list[dict[str, Any]]) -> int:
    """Count proposed LLM tool calls that consume max_tool_calls budget."""
    used = 0
    for tc in tool_calls:
        tool_name = tc.get("function", {}).get("name", "")
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
        tool_name = tc.get("function", {}).get("name", "")
        if _is_budget_exempt_tool(tool_name):
            kept.append(tc)
            continue
        if kept_budgeted < budget_remaining:
            kept.append(tc)
            kept_budgeted += 1
        else:
            dropped_budgeted += 1
    return kept, dropped_budgeted


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


def _extract_unfinished_todo_ids(error_text: str) -> list[str]:
    """Extract TODO IDs from submit/todo validation messages."""
    text = (error_text or "").strip()
    if not text:
        return []
    # Common shape: "Unfinished TODOs: todo_2, todo_3."
    match = re.search(r"unfinished\s+todos?\s*:\s*([^\n]+)", text, flags=re.IGNORECASE)
    segment = match.group(1) if match else text
    found = re.findall(r"\btodo_[A-Za-z0-9_-]+\b", segment)
    # Preserve order while de-duplicating.
    deduped: list[str] = []
    seen: set[str] = set()
    for item in found:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _extract_tool_call_args(tc: dict[str, Any]) -> dict[str, Any] | None:
    """Parse function-call arguments as dict when possible."""
    raw = tc.get("function", {}).get("arguments", {})
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = _json.loads(raw)
        except Exception:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def _parse_record_result_json(record: MCPToolCallRecord) -> dict[str, Any] | None:
    """Best-effort parse of JSON tool result payloads."""
    if not record.result or not isinstance(record.result, str):
        return None
    try:
        parsed = _json.loads(record.result)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


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


def _normalize_artifact_kind(kind: Any) -> str | None:
    """Normalize artifact kind labels (e.g., SlotKind names) to upper snake case strings."""
    if isinstance(kind, str):
        normalized = kind.strip().upper()
        return normalized or None
    return None


def _normalize_tool_contracts(raw: Any) -> dict[str, dict[str, Any]]:
    """Normalize raw tool contract payload into a predictable dict shape."""
    if not isinstance(raw, dict):
        return {}

    normalized: dict[str, dict[str, Any]] = {}
    for tool_name, spec in raw.items():
        if not isinstance(tool_name, str) or not tool_name.strip():
            continue
        if not isinstance(spec, dict):
            continue

        requires_all = {
            k for k in (
                _normalize_artifact_kind(v) for v in (spec.get("requires_all") or [])
            )
            if k
        }
        requires_any = {
            k for k in (
                _normalize_artifact_kind(v) for v in (spec.get("requires_any") or [])
            )
            if k
        }
        produces = {
            k for k in (
                _normalize_artifact_kind(v) for v in (spec.get("produces") or [])
            )
            if k
        }

        normalized[tool_name.strip()] = {
            "requires_all": requires_all,
            "requires_any": requires_any,
            "produces": produces,
            "is_control": bool(spec.get("is_control", False)),
        }

    return normalized


def _effective_contract_requirements(
    tool_name: str,
    contract: dict[str, Any],
    parsed_args: dict[str, Any] | None,
) -> tuple[set[str], set[str]]:
    """Resolve dynamic per-call requirements for selected tools."""
    requires_all = set(contract.get("requires_all") or set())
    requires_any = set(contract.get("requires_any") or set())

    args = parsed_args or {}

    if tool_name == "chunk_get_text":
        has_chunk_ids = bool(args.get("chunk_id")) or bool(args.get("chunk_ids"))
        has_entity_ids = bool(args.get("entity_ids")) or bool(args.get("entity_names"))
        if has_chunk_ids and not has_entity_ids:
            return {"CHUNK_SET"}, set()
        if has_entity_ids and not has_chunk_ids:
            return {"ENTITY_SET"}, set()
        if has_chunk_ids and has_entity_ids:
            return {"CHUNK_SET", "ENTITY_SET"}, set()

    return requires_all, requires_any


def _validate_tool_contract_call(
    *,
    tool_name: str,
    contract: dict[str, Any],
    parsed_args: dict[str, Any] | None,
    available_artifacts: set[str],
    available_bindings: dict[str, str | None] | None = None,
) -> ToolCallValidation:
    """Validate whether a tool call is composable given currently available artifacts."""
    call_bindings = extract_bindings_from_tool_args(parsed_args)

    if contract.get("is_control"):
        return ToolCallValidation(
            is_valid=True,
            call_bindings=call_bindings,
        )

    if call_bindings:
        bindings_ok, bindings_reason, _conflicts, normalized_call_bindings = check_binding_conflicts(
            available_bindings=available_bindings,
            proposed_bindings=call_bindings,
        )
        if not bindings_ok:
            return ToolCallValidation(
                is_valid=False,
                reason=bindings_reason,
                error_code="binding_conflict",
                failure_phase="binding_validation",
                call_bindings=normalized_call_bindings,
            )

    requires_all, requires_any = _effective_contract_requirements(
        tool_name, contract, parsed_args,
    )

    missing_all = sorted(requires_all - available_artifacts)
    if missing_all:
        return ToolCallValidation(
            is_valid=False,
            reason=f"{tool_name} requires all of {sorted(requires_all)}; missing {missing_all}",
            error_code="missing_prerequisite",
            failure_phase="input_validation",
            call_bindings=call_bindings,
        )

    if requires_any and not (requires_any & available_artifacts):
        return ToolCallValidation(
            is_valid=False,
            reason=f"{tool_name} requires one of {sorted(requires_any)}; available {sorted(available_artifacts)}",
            error_code="missing_prerequisite",
            failure_phase="input_validation",
            call_bindings=call_bindings,
        )

    return ToolCallValidation(
        is_valid=True,
        call_bindings=call_bindings,
    )


def _filter_tools_for_disclosure(
    *,
    openai_tools: list[dict[str, Any]],
    normalized_tool_contracts: dict[str, dict[str, Any]],
    available_artifacts: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Return tools currently composable from artifact state + hidden reasons.

    Unknown tools (no contract declared) are kept visible to avoid false negatives.
    Control tools stay visible regardless of artifact state.
    """
    if not openai_tools:
        return [], []
    if not normalized_tool_contracts:
        return list(openai_tools), []

    visible: list[dict[str, Any]] = []
    hidden: list[dict[str, str]] = []
    for tool_def in openai_tools:
        fn = tool_def.get("function", {}) if isinstance(tool_def, dict) else {}
        tool_name = str(fn.get("name", "")).strip()
        if not tool_name:
            visible.append(tool_def)
            continue

        contract = normalized_tool_contracts.get(tool_name)
        if not isinstance(contract, dict):
            visible.append(tool_def)
            continue

        validation = _validate_tool_contract_call(
            tool_name=tool_name,
            contract=contract,
            parsed_args=None,
            available_artifacts=available_artifacts,
            available_bindings=None,
        )
        if validation.is_valid:
            visible.append(tool_def)
        else:
            hidden.append({"tool": tool_name, "reason": validation.reason})

    if not visible:
        # Failsafe: never present an empty tool surface if tools exist.
        return list(openai_tools), hidden
    return visible, hidden


def _contract_outputs(contract: dict[str, Any] | None) -> set[str]:
    """Return declared artifact outputs for a tool contract."""
    if not isinstance(contract, dict):
        return set()
    return set(contract.get("produces") or set())


def _is_responses_api_raw_response(raw_response: Any) -> bool:
    """Best-effort check for Responses API objects (vs chat completions)."""
    if raw_response is None:
        return False
    has_output = hasattr(raw_response, "output")
    has_choices = hasattr(raw_response, "choices")
    return bool(has_output and not has_choices)


# ---------------------------------------------------------------------------
# MCP Agent Loop
# ---------------------------------------------------------------------------


def _import_mcp() -> tuple[Any, ...]:
    """Lazily import mcp client components.

    Returns:
        (stdio_client, StdioServerParameters, ClientSession)
    """
    try:
        from mcp.client.stdio import (
            StdioServerParameters,
            stdio_client,
        )
        from mcp import ClientSession
    except ImportError:
        raise ImportError(
            "mcp package is required for MCP agent loop. "
            "Install with: pip install llm_client[mcp]"
        ) from None
    return stdio_client, StdioServerParameters, ClientSession


async def _start_servers(
    mcp_servers: dict[str, dict[str, Any]],
    stack: AsyncExitStack,
    init_timeout: float,
) -> tuple[dict[str, Any], dict[str, str], list[dict[str, Any]]]:
    """Start MCP servers and discover tools.

    Returns:
        (sessions, tool_to_server, openai_tools)
    """
    stdio_client, StdioServerParameters, ClientSession = _import_mcp()

    sessions: dict[str, Any] = {}
    tool_to_server: dict[str, str] = {}
    openai_tools: list[dict[str, Any]] = []

    for server_name, server_cfg in mcp_servers.items():
        params = StdioServerParameters(
            command=server_cfg["command"],
            args=server_cfg.get("args", []),
            env=server_cfg.get("env"),
            cwd=server_cfg.get("cwd"),
        )

        read_stream, write_stream = await stack.enter_async_context(
            stdio_client(params)
        )
        session = await stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await asyncio.wait_for(session.initialize(), timeout=init_timeout)
        sessions[server_name] = session

        # Discover tools
        tools_result = await session.list_tools()
        for tool in tools_result.tools:
            if tool.name in tool_to_server:
                logger.warning(
                    "Duplicate tool %r from server %r (already from %r)",
                    tool.name, server_name, tool_to_server[tool.name],
                )
                continue
            tool_to_server[tool.name] = server_name
            openai_tools.append(_mcp_tool_to_openai(tool))

    logger.info(
        "MCP agent loop: %d tools from %d servers",
        len(openai_tools), len(sessions),
    )
    return sessions, tool_to_server, openai_tools


async def _execute_tool_calls(
    tool_calls: list[dict[str, Any]],
    sessions: dict[str, Any],
    tool_to_server: dict[str, str],
    max_result_length: int,
    require_tool_reasoning: bool = DEFAULT_REQUIRE_TOOL_REASONING,
) -> tuple[list[MCPToolCallRecord], list[dict[str, Any]]]:
    """Execute tool calls against MCP servers.

    Returns:
        (records, tool_messages) — records for tracking, messages to append
    """
    records: list[MCPToolCallRecord] = []
    tool_messages: list[dict[str, Any]] = []

    for tc in tool_calls:
        fn = tc.get("function", {})
        tool_name = fn.get("name", "")
        arguments_str = fn.get("arguments", "{}")
        tc_id = tc.get("id", "")

        try:
            arguments = (
                _json.loads(arguments_str)
                if isinstance(arguments_str, str)
                else arguments_str
            )
        except _json.JSONDecodeError as exc:
            logger.error("Failed to parse tool call arguments for %s: %s", tool_name, str(arguments_str)[:200])
            record = MCPToolCallRecord(
                server=tool_to_server.get(tool_name) or "unknown",
                tool=tool_name,
                arguments={},
                error=f"JSON parse error: {exc}",
            )
            tool_messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": f"ERROR: Invalid JSON arguments: {exc}",
            })
            records.append(record)
            continue

        server_name = tool_to_server.get(tool_name)
        if not isinstance(arguments, dict):
            arguments = {}

        tool_reasoning_raw = arguments.pop(TOOL_REASONING_FIELD, None)
        tool_reasoning = None
        if isinstance(tool_reasoning_raw, str):
            stripped = tool_reasoning_raw.strip()
            if stripped:
                tool_reasoning = stripped

        record = MCPToolCallRecord(
            server=server_name or "unknown",
            tool=tool_name,
            arguments=arguments,
            tool_reasoning=tool_reasoning,
        )

        if require_tool_reasoning and not tool_reasoning:
            record.error = f"Missing required argument: {TOOL_REASONING_FIELD}"
            tool_messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": _json.dumps({"error": record.error}),
            })
            records.append(record)
            continue

        t0 = time.monotonic()
        if server_name is None:
            record.error = f"Unknown tool: {tool_name}"
            tool_content = _json.dumps({"error": record.error})
        else:
            try:
                session = sessions[server_name]
                mcp_result = await session.call_tool(tool_name, arguments)

                parts: list[str] = []
                for content_item in mcp_result.content or []:
                    if hasattr(content_item, "text"):
                        parts.append(content_item.text)
                    else:
                        parts.append(str(content_item))
                tool_content = "\n".join(parts)
                tool_content = _truncate(tool_content, max_result_length)

                if mcp_result.isError:
                    record.error = tool_content
                else:
                    record.result = tool_content
            except Exception as e:
                record.error = str(e)
                tool_content = _json.dumps({"error": str(e)})

        record.latency_s = round(time.monotonic() - t0, 3)
        records.append(record)

        tool_messages.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "content": tool_content,
        })

    return records, tool_messages


async def _inner_acall_llm(
    model: str,
    messages: list[dict[str, Any]],
    **kwargs: Any,
) -> LLMCallResult:
    """Call acall_llm from client module. Separate function for testability."""
    from llm_client.client import acall_llm
    return await acall_llm(model, messages, **kwargs)


async def _acall_with_mcp(
    model: str,
    messages: list[dict[str, Any]],
    mcp_servers: dict[str, dict[str, Any]] | None = None,
    *,
    mcp_sessions: MCPSessionPool | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_tool_calls: int | None = DEFAULT_MAX_TOOL_CALLS,
    require_tool_reasoning: bool = DEFAULT_REQUIRE_TOOL_REASONING,
    mcp_init_timeout: float = DEFAULT_MCP_INIT_TIMEOUT,
    tool_result_max_length: int = DEFAULT_TOOL_RESULT_MAX_LENGTH,
    max_message_chars: int = DEFAULT_MAX_MESSAGE_CHARS,
    enforce_tool_contracts: bool = DEFAULT_ENFORCE_TOOL_CONTRACTS,
    progressive_tool_disclosure: bool = DEFAULT_PROGRESSIVE_TOOL_DISCLOSURE,
    tool_contracts: dict[str, dict[str, Any]] | None = None,
    initial_artifacts: list[str] | tuple[str, ...] | None = DEFAULT_INITIAL_ARTIFACTS,
    initial_bindings: dict[str, Any] | None = None,
    timeout: int = 60,
    **kwargs: Any,
) -> LLMCallResult:
    """Run an MCP tool-calling agent loop with any litellm model.

    Starts MCP server subprocesses, discovers tools, then loops:
    1. Call LLM with tool definitions
    2. If LLM returns tool_calls → execute via MCP → append results
    3. Repeat until LLM returns text (no tool calls) or max_turns

    Args:
        model: Any litellm model string (NOT an agent model)
        messages: Initial messages (system + user)
        mcp_servers: Dict of server_name -> {command, args?, env?, cwd?}
            (ignored if mcp_sessions is provided)
        mcp_sessions: Pre-started MCPSessionPool for server reuse across calls.
            When provided, servers are NOT started/stopped per call.
        max_turns: Maximum loop iterations
        max_tool_calls: Maximum tool calls before final forced answer. None disables.
        require_tool_reasoning: If True, reject tool calls missing tool_reasoning.
        mcp_init_timeout: Seconds to wait for server startup
        tool_result_max_length: Max chars per tool result (truncated if longer)
        enforce_tool_contracts: If True, reject tool calls violating tool contracts.
        progressive_tool_disclosure: If True, hide tools whose contracts are not currently satisfiable.
        tool_contracts: Optional per-tool composability contracts.
        initial_artifacts: Artifact kinds available before any tool call.
        initial_bindings: Binding state available before any tool call.
        timeout: Per-turn LLM call timeout
        **kwargs: Passed through to acall_llm (retry, hooks, etc.)
    """
    agent_result = MCPAgentResult()
    messages = list(messages)  # don't mutate caller's list
    total_cost = 0.0
    final_content = ""
    final_finish_reason = "stop"

    if mcp_sessions is not None:
        # Reuse existing sessions — no server spawn/teardown
        sessions = mcp_sessions.sessions
        tool_to_server = mcp_sessions.tool_to_server
        openai_tools = mcp_sessions.openai_tools

        if not openai_tools:
            raise ValueError("MCPSessionPool has no tools — was it entered as context manager?")

        async def _mcp_executor(
            tool_calls: list[dict[str, Any]], max_len: int,
        ) -> tuple[list[MCPToolCallRecord], list[dict[str, Any]]]:
            return await _execute_tool_calls(
                tool_calls,
                sessions,
                tool_to_server,
                max_len,
                require_tool_reasoning=require_tool_reasoning,
            )

        final_content, final_finish_reason = await _agent_loop(
            model, messages, openai_tools,
            agent_result,
            _mcp_executor,
            max_turns,
            max_tool_calls,
            require_tool_reasoning,
            tool_result_max_length,
            max_message_chars,
            enforce_tool_contracts,
            progressive_tool_disclosure,
            tool_contracts,
            initial_artifacts,
            initial_bindings,
            timeout,
            kwargs,
        )
        total_cost = sum(r.latency_s for r in agent_result.tool_calls)  # placeholder; real cost tracked below

    elif mcp_servers is not None:
        async with AsyncExitStack() as stack:
            # Start servers and discover tools
            sessions, tool_to_server, openai_tools = await _start_servers(
                mcp_servers, stack, mcp_init_timeout,
            )

            if not openai_tools:
                raise ValueError(
                    f"No tools discovered from MCP servers: {list(mcp_servers.keys())}"
                )

            async def _mcp_executor(
                tool_calls: list[dict[str, Any]], max_len: int,
            ) -> tuple[list[MCPToolCallRecord], list[dict[str, Any]]]:
                return await _execute_tool_calls(
                    tool_calls,
                    sessions,
                    tool_to_server,
                    max_len,
                    require_tool_reasoning=require_tool_reasoning,
                )

            final_content, final_finish_reason = await _agent_loop(
                model, messages, openai_tools,
                agent_result,
                _mcp_executor,
                max_turns,
                max_tool_calls,
                require_tool_reasoning,
                tool_result_max_length,
                max_message_chars,
                enforce_tool_contracts,
                progressive_tool_disclosure,
                tool_contracts,
                initial_artifacts,
                initial_bindings,
                timeout,
                kwargs,
            )
    else:
        raise ValueError("Either mcp_servers or mcp_sessions must be provided")

    # Cost is accumulated during _agent_loop via agent_result metadata
    usage = {
            "input_tokens": agent_result.total_input_tokens,
            "output_tokens": agent_result.total_output_tokens,
            "total_tokens": (
                agent_result.total_input_tokens + agent_result.total_output_tokens
            ),
        }
    if agent_result.total_cached_tokens:
        usage["cached_tokens"] = agent_result.total_cached_tokens
    if agent_result.total_cache_creation_tokens:
        usage["cache_creation_tokens"] = agent_result.total_cache_creation_tokens
    return LLMCallResult(
        content=final_content,
        usage=usage,
        cost=agent_result.metadata.get("total_cost", 0.0),
        model=model,
        finish_reason=final_finish_reason,
        raw_response=agent_result,
        warnings=agent_result.warnings,
    )


async def _agent_loop(
    model: str,
    messages: list[dict[str, Any]],
    openai_tools: list[dict[str, Any]],
    agent_result: MCPAgentResult,
    executor: Any,  # Callable[[list, int], Awaitable[tuple[list[MCPToolCallRecord], list[dict]]]]
    max_turns: int,
    max_tool_calls: int | None,
    require_tool_reasoning: bool,
    tool_result_max_length: int,
    max_message_chars: int = DEFAULT_MAX_MESSAGE_CHARS,
    enforce_tool_contracts: bool = DEFAULT_ENFORCE_TOOL_CONTRACTS,
    progressive_tool_disclosure: bool = DEFAULT_PROGRESSIVE_TOOL_DISCLOSURE,
    tool_contracts: dict[str, dict[str, Any]] | None = None,
    initial_artifacts: list[str] | tuple[str, ...] | None = DEFAULT_INITIAL_ARTIFACTS,
    initial_bindings: dict[str, Any] | None = None,
    timeout: int = 60,
    kwargs: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Core agent loop shared by MCP, direct-tool, and session-pool paths.

    Args:
        executor: async callable (tool_calls, max_result_length) -> (records, tool_messages).
            For MCP: wraps _execute_tool_calls with bound sessions.
            For direct tools: wraps execute_direct_tool_calls with bound tool_map.

    Returns (final_content, final_finish_reason).
    """
    total_cost = 0.0
    final_content = ""
    final_finish_reason = "stop"
    effective_model = model
    tool_call_counts: dict[str, int] = {}
    tool_loop_nudges: dict[str, int] = {}
    tool_error_counts: dict[tuple[str, str], int] = {}
    tool_error_nudges: dict[tuple[str, str], int] = {}
    submit_requires_todo_progress = False
    todo_done_error_streak: dict[str, int] = {}
    control_loop_suppressed_calls = 0
    last_budget_remaining: int | None = None
    rejected_missing_reasoning_calls = 0
    tool_call_turns_total = 0
    tool_call_empty_text_turns = 0
    responses_tool_call_empty_text_turns = 0
    tool_arg_coercions = 0
    tool_arg_coercion_calls = 0
    tool_arg_validation_rejections = 0
    tool_disclosure_turns = 0
    tool_disclosure_hidden_total = 0
    context_compactions = 0
    context_compacted_messages = 0
    context_compacted_chars = 0
    contract_rejected_calls = 0
    contract_violation_events: list[dict[str, Any]] = []
    normalized_tool_contracts = _normalize_tool_contracts(tool_contracts)
    if initial_artifacts is None:
        initial_artifacts = DEFAULT_INITIAL_ARTIFACTS
    available_artifacts = {
        k for k in (_normalize_artifact_kind(v) for v in initial_artifacts) if k
    }
    if not available_artifacts:
        available_artifacts = set(DEFAULT_INITIAL_ARTIFACTS)
    initial_artifact_snapshot = sorted(available_artifacts)
    available_bindings = normalize_bindings(initial_bindings)
    initial_binding_snapshot = dict(available_bindings)
    artifact_timeline: list[dict[str, Any]] = [{
        "turn": 0,
        "phase": "initial",
        "available_artifacts": list(initial_artifact_snapshot),
    }]
    if enforce_tool_contracts and not normalized_tool_contracts:
        warning = (
            "TOOL_CONTRACTS: enforce_tool_contracts=True but no contracts were provided; "
            "composability validation is skipped."
        )
        agent_result.warnings.append(warning)
        logger.warning(warning)
    requires_submit_answer = any(
        t.get("function", {}).get("name") == "submit_answer"
        for t in openai_tools
        if isinstance(t, dict)
    )
    submit_answer_succeeded = False
    force_final_reason: str | None = None
    pending_submit_todo_ids: list[str] = []

    kwargs = dict(kwargs or {})
    trace_id = str(kwargs.get("trace_id", "")).strip() or None
    task = str(kwargs.get("task", "")).strip() or None
    foundation_session_id = new_session_id()
    foundation_actor_id = "agent:mcp_loop:default:1"
    foundation_events: list[dict[str, Any]] = []
    foundation_event_types: dict[str, int] = {}
    foundation_event_validation_errors = 0
    foundation_events_logged = 0
    try:
        from llm_client import io_log as _io_log  # local import to avoid module-cycle hazards
        active_run_id = _io_log.get_active_experiment_run_id()
    except Exception:
        _io_log = None
        active_run_id = None
    foundation_run_id = coerce_run_id(
        active_run_id if isinstance(active_run_id, str) else None,
        trace_id,
    )

    def _emit_foundation_event(payload: dict[str, Any]) -> None:
        nonlocal foundation_event_validation_errors, foundation_events_logged
        try:
            validated = validate_foundation_event(payload)
        except Exception as exc:
            foundation_event_validation_errors += 1
            warning = f"FOUNDATION_EVENT_INVALID: {type(exc).__name__}: {exc}"
            agent_result.warnings.append(warning)
            logger.warning(warning)
            return

        foundation_events.append(validated)
        event_type = str(validated.get("event_type", "unknown"))
        foundation_event_types[event_type] = foundation_event_types.get(event_type, 0) + 1

        if _io_log is None:
            return
        try:
            _io_log.log_foundation_event(
                event=validated,
                caller="llm_client.mcp_agent",
                task=task,
                trace_id=trace_id,
            )
            foundation_events_logged += 1
        except Exception as exc:
            warning = f"FOUNDATION_EVENT_LOG_FAILED: {type(exc).__name__}: {exc}"
            agent_result.warnings.append(warning)
            logger.warning(warning)

    if (
        "max_tokens" not in kwargs
        and "max_completion_tokens" not in kwargs
    ):
        # Tool-calling turns do not need long free-form completions; smaller
        # completions reduce context-window pressure on long agent traces.
        kwargs = dict(kwargs)
        kwargs["max_completion_tokens"] = 4096

    for turn in range(max_turns):
        if max_tool_calls is not None:
            budgeted_calls_used = _count_budgeted_records(agent_result.tool_calls)
            remaining_tool_calls = max_tool_calls - budgeted_calls_used
            if remaining_tool_calls <= 0:
                force_final_reason = "max_tool_calls"
                logger.warning(
                    "Agent loop exhausted max_tool_calls=%d after %d turns (%d budgeted, %d total tool calls); forcing final answer",
                    max_tool_calls,
                    turn,
                    budgeted_calls_used,
                    len(agent_result.tool_calls),
                )
                break
            if remaining_tool_calls != last_budget_remaining:
                budget_msg = {
                    "role": "user",
                    "content": (
                        f"[SYSTEM: Retrieval-tool budget: {budgeted_calls_used}/{max_tool_calls} used, "
                        f"{remaining_tool_calls} remaining. Plan carefully. "
                        f"Budget-exempt tools: {', '.join(sorted(BUDGET_EXEMPT_TOOL_NAMES))}. "
                        "If you already have enough evidence, submit your answer now.]"
                    ),
                }
                messages.append(budget_msg)
                agent_result.conversation_trace.append(budget_msg)
                last_budget_remaining = remaining_tool_calls

        agent_result.turns = turn + 1

        compacted_count, compacted_chars, current_chars = _compact_tool_history_for_context(
            messages, max_message_chars,
        )
        if compacted_count:
            context_compactions += 1
            context_compacted_messages += compacted_count
            context_compacted_chars += compacted_chars
            warning = (
                "CONTEXT_COMPACTION: compacted "
                f"{compacted_count} tool message(s), saved ~{compacted_chars} chars "
                f"(history ~{current_chars} chars)"
            )
            agent_result.warnings.append(warning)
            logger.warning(warning)

        disclosed_tools = openai_tools
        hidden_disclosure = []
        if progressive_tool_disclosure and normalized_tool_contracts:
            disclosed_tools, hidden_disclosure = _filter_tools_for_disclosure(
                openai_tools=openai_tools,
                normalized_tool_contracts=normalized_tool_contracts,
                available_artifacts=available_artifacts,
            )
            if hidden_disclosure:
                tool_disclosure_turns += 1
                tool_disclosure_hidden_total += len(hidden_disclosure)
                hidden_names = [h["tool"] for h in hidden_disclosure]
                logger.info(
                    "TOOL_DISCLOSURE turn=%d exposed=%d/%d hidden=%s available_artifacts=%s",
                    turn + 1,
                    len(disclosed_tools),
                    len(openai_tools),
                    hidden_names,
                    sorted(available_artifacts),
                )
                agent_result.warnings.append(
                    "TOOL_DISCLOSURE: hidden currently incompatible tools on turn "
                    f"{turn + 1}: {', '.join(hidden_names)}"
                )

        result = await _inner_acall_llm(
            effective_model, messages, timeout=timeout, tools=disclosed_tools, **kwargs,
        )

        # Track per-turn diagnostics
        agent_result.models_used.add(result.model)
        if result.warnings:
            agent_result.warnings.extend(result.warnings)

        # Sticky fallback: if inner call fell back to a different model,
        # use that model for remaining turns (avoids re-hitting dead primary).
        if result.model != effective_model:
            agent_result.warnings.append(
                f"STICKY_FALLBACK: {effective_model} failed, "
                f"using {result.model} for remaining turns"
            )
            effective_model = result.model

        _emit_foundation_event(
            {
                "event_id": new_event_id(),
                "event_type": "LLMCalled",
                "timestamp": now_iso(),
                "run_id": foundation_run_id,
                "session_id": foundation_session_id,
                "actor_id": foundation_actor_id,
                "operation": {"name": "_inner_acall_llm", "version": None},
                "inputs": {
                    "artifact_ids": sorted(available_artifacts),
                    "params": {
                        "turn": turn + 1,
                        "model": effective_model,
                    },
                    "bindings": dict(available_bindings),
                },
                "outputs": {
                    "artifact_ids": [],
                    "payload_hashes": [sha256_text(result.content or "")],
                },
                "llm": {
                    "model_id": result.model,
                    "content_persisted": "hash_only",
                    "prompt_sha256": sha256_json(messages),
                    "response_sha256": sha256_text(result.content or ""),
                    "token_usage": dict(result.usage or {}),
                    "cost_usd": float(result.cost or 0.0),
                },
            }
        )

        # Accumulate usage
        inp, out, cached, cache_create = _extract_usage(result.usage or {})
        agent_result.total_input_tokens += inp
        agent_result.total_output_tokens += out
        agent_result.total_cached_tokens += cached
        agent_result.total_cache_creation_tokens += cache_create
        total_cost += result.cost

        # No tool calls → done
        if not result.tool_calls:
            remaining_turns = max_turns - (turn + 1)
            if requires_submit_answer and not submit_answer_succeeded and remaining_turns > 0:
                # Benchmark loops require explicit submit_answer() so scorers can
                # reliably parse the final answer from tool calls.
                if result.content:
                    draft_msg = {
                        "role": "assistant",
                        "content": result.content,
                    }
                    messages.append(draft_msg)
                    agent_result.conversation_trace.append(draft_msg)

                submit_nudge = {
                    "role": "user",
                    "content": (
                        "[SYSTEM: Do NOT answer in plain text. You MUST call "
                        "submit_answer(reasoning, answer) now. Use a short factual "
                        "answer (<=8 words). No additional searches.]"
                    ),
                }
                messages.append(submit_nudge)
                agent_result.conversation_trace.append(submit_nudge)
                logger.warning(
                    "Agent loop: model returned plain text without submit_answer on turn %d/%d; nudging explicit submission.",
                    turn + 1, max_turns,
                )
                continue

            final_content = result.content
            final_finish_reason = result.finish_reason
            # Log visibility: empty content on first turn is almost always a model failure
            if not result.content and turn == 0:
                logger.error(
                    "Agent loop: model=%s returned empty content with 0 tool calls on turn 1 "
                    "(finish_reason=%s). All %d retries + fallback exhausted at the per-turn level.",
                    model, result.finish_reason, kwargs.get("num_retries", 2),
                )
            elif not result.content:
                logger.warning(
                    "Agent loop: model=%s returned empty content on turn %d/%d "
                    "(finish_reason=%s, %d tool calls so far).",
                    model, turn + 1, max_turns, result.finish_reason,
                    len(agent_result.tool_calls),
                )
            # Capture final assistant message in trace
            if result.content:
                agent_result.conversation_trace.append({
                    "role": "assistant",
                    "content": result.content,
                })
            break

        # Append assistant message with tool calls
        tool_calls_this_turn = list(result.tool_calls)
        autofilled_reasoning_tools: list[str] = []
        patched_calls: list[dict[str, Any]] = []
        for tc in tool_calls_this_turn:
            patched, changed = _autofill_tool_reasoning(tc)
            if changed:
                name = patched.get("function", {}).get("name", "")
                if name:
                    autofilled_reasoning_tools.append(name)
            patched_calls.append(patched)
        tool_calls_this_turn = patched_calls
        if autofilled_reasoning_tools:
            agent_result.warnings.append(
                "OBSERVABILITY: auto-filled missing tool_reasoning on tools: "
                + ", ".join(autofilled_reasoning_tools)
            )
        tool_call_turns_total += 1
        if not (result.content or "").strip():
            tool_call_empty_text_turns += 1
            is_responses_turn = _is_responses_api_raw_response(result.raw_response)
            if is_responses_turn:
                responses_tool_call_empty_text_turns += 1
            logger.info(
                "Agent loop metric: turn %d/%d model=%s returned %d tool call(s) with empty assistant text%s",
                turn + 1,
                max_turns,
                result.model,
                len(tool_calls_this_turn),
                " [responses-api]" if is_responses_turn else "",
            )

        if max_tool_calls is not None:
            budgeted_used = _count_budgeted_records(agent_result.tool_calls)
            remaining_tool_calls = max_tool_calls - budgeted_used
            budgeted_requested = _count_budgeted_tool_calls(tool_calls_this_turn)
            if budgeted_requested > remaining_tool_calls:
                tool_calls_this_turn, dropped = _trim_tool_calls_to_budget(
                    tool_calls_this_turn,
                    max(remaining_tool_calls, 0),
                )
                trim_msg = {
                    "role": "user",
                    "content": (
                        f"[SYSTEM: Retrieval-tool budget allows only {remaining_tool_calls} more budgeted call(s). "
                        f"Ignored {dropped} over-budget retrieval call(s). "
                        f"Budget-exempt tools ({', '.join(sorted(BUDGET_EXEMPT_TOOL_NAMES))}) are still allowed.]"
                    ),
                }
                messages.append(trim_msg)
                agent_result.conversation_trace.append(trim_msg)
                logger.warning(
                    "Agent loop trimmed %d over-budget retrieval call(s) on turn %d; budget remaining=%d",
                    dropped,
                    turn + 1,
                    remaining_tool_calls,
                )

        missing_reasoning_tools: list[str] = []
        missing_reasoning_call_ids: set[str] = set()
        for tc in tool_calls_this_turn:
            fn = tc.get("function", {})
            tc_name = fn.get("name", "")
            tc_id = tc.get("id", "")
            tc_args = _extract_tool_call_args(tc)
            if not isinstance(tc_args, dict):
                missing_reasoning_tools.append(tc_name or "<unknown>")
                if tc_id:
                    missing_reasoning_call_ids.add(tc_id)
                continue
            tc_reasoning = tc_args.get(TOOL_REASONING_FIELD)
            if not isinstance(tc_reasoning, str) or not tc_reasoning.strip():
                missing_reasoning_tools.append(tc_name or "<unknown>")
                if tc_id:
                    missing_reasoning_call_ids.add(tc_id)

        if missing_reasoning_tools:
            reasoning_nudge = {
                "role": "user",
                "content": (
                    "[SYSTEM: Observability requirement: every tool call must include "
                    f"'{TOOL_REASONING_FIELD}' with one concise sentence explaining why this call is needed."
                    + (
                        " Calls without it are rejected."
                        if require_tool_reasoning else
                        ""
                    )
                    + "]"
                ),
            }
            messages.append(reasoning_nudge)
            agent_result.conversation_trace.append(reasoning_nudge)
            agent_result.warnings.append(
                "OBSERVABILITY: missing tool_reasoning on tools: "
                + ", ".join(missing_reasoning_tools)
            )

        assistant_msg = {
            "role": "assistant",
            "content": result.content or None,
            "tool_calls": tool_calls_this_turn,
        }
        messages.append(assistant_msg)

        # Capture assistant message in trace (with tool call names for readability)
        agent_result.conversation_trace.append({
            "role": "assistant",
            "content": result.content or "",
            "tool_calls": [
                {
                    "id": tc.get("id", ""),
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": tc.get("function", {}).get("arguments", ""),
                }
                for tc in tool_calls_this_turn
            ],
        })

        tool_calls_to_execute = tool_calls_this_turn
        pending_repair_msg: dict[str, Any] | None = None
        if require_tool_reasoning and missing_reasoning_call_ids:
            tool_calls_to_execute = [
                tc for tc in tool_calls_this_turn
                if tc.get("id", "") not in missing_reasoning_call_ids
            ]

            rejected_missing_reasoning_calls += len(missing_reasoning_call_ids)
            rejected_tool_messages: list[dict[str, Any]] = []
            for tc in tool_calls_this_turn:
                tc_id = tc.get("id", "")
                if tc_id not in missing_reasoning_call_ids:
                    continue
                rejected_tool_messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": _json.dumps({
                        "error": f"Missing required argument: {TOOL_REASONING_FIELD}",
                    }),
                })
            messages.extend(rejected_tool_messages)
            for tmsg in rejected_tool_messages:
                agent_result.conversation_trace.append({
                    "role": "tool",
                    "tool_call_id": tmsg.get("tool_call_id", ""),
                    "content": tmsg.get("content", ""),
                })

            pending_repair_msg = {
                "role": "user",
                "content": (
                    "[SYSTEM: One or more tool calls were rejected because "
                    f"'{TOOL_REASONING_FIELD}' was missing. Re-issue corrected tool calls "
                    "only if still needed.]"
                ),
            }

        contract_rejected_records: list[MCPToolCallRecord] = []
        contract_rejected_messages: list[dict[str, Any]] = []
        pending_contract_msg: dict[str, Any] | None = None
        if enforce_tool_contracts and normalized_tool_contracts:
            filtered_contract_calls: list[dict[str, Any]] = []
            for tc in tool_calls_to_execute:
                tool_name = tc.get("function", {}).get("name", "")
                tc_id = tc.get("id", "")
                parsed_args = _extract_tool_call_args(tc)
                contract = normalized_tool_contracts.get(tool_name)
                if not isinstance(contract, dict):
                    filtered_contract_calls.append(tc)
                    continue

                validation = _validate_tool_contract_call(
                    tool_name=tool_name,
                    contract=contract,
                    parsed_args=parsed_args,
                    available_artifacts=available_artifacts,
                    available_bindings=available_bindings,
                )
                if validation.is_valid:
                    filtered_contract_calls.append(tc)
                    continue

                contract_rejected_calls += 1
                err = f"Tool contract violation: {validation.reason}"
                contract_violation_events.append({
                    "turn": turn + 1,
                    "tool": tool_name or "<unknown>",
                    "reason": validation.reason,
                    "error_code": validation.error_code or "contract_violation",
                    "failure_phase": validation.failure_phase or "input_validation",
                    "available_artifacts": sorted(available_artifacts),
                    "available_bindings": dict(available_bindings),
                    "call_bindings": dict(validation.call_bindings),
                    "arg_keys": sorted(parsed_args.keys()) if isinstance(parsed_args, dict) else [],
                })
                contract_rejected_records.append(
                    MCPToolCallRecord(
                        server="__contract__",
                        tool=tool_name or "<unknown>",
                        arguments=parsed_args if isinstance(parsed_args, dict) else {},
                        error=err,
                    )
                )
                contract_rejected_messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": _json.dumps(
                        {
                            "error": err,
                            "error_code": validation.error_code or "contract_violation",
                            "failure_phase": validation.failure_phase or "input_validation",
                            "call_bindings": validation.call_bindings,
                        }
                    ),
                })
                _emit_foundation_event(
                    {
                        "event_id": new_event_id(),
                        "event_type": "ToolFailed",
                        "timestamp": now_iso(),
                        "run_id": foundation_run_id,
                        "session_id": foundation_session_id,
                        "actor_id": foundation_actor_id,
                        "operation": {"name": tool_name or "<unknown>", "version": None},
                        "inputs": {
                            "artifact_ids": sorted(available_artifacts),
                            "params": parsed_args if isinstance(parsed_args, dict) else {},
                            "bindings": dict(available_bindings),
                        },
                        "outputs": {"artifact_ids": [], "payload_hashes": []},
                        "failure": {
                            "error_code": validation.error_code or "contract_violation",
                            "category": "validation",
                            "phase": validation.failure_phase or "input_validation",
                            "retryable": False,
                            "tool_name": tool_name or "<unknown>",
                            "user_message": err,
                            "debug_ref": None,
                        },
                    }
                )

            tool_calls_to_execute = filtered_contract_calls
            if contract_rejected_records:
                pending_contract_msg = {
                    "role": "user",
                    "content": (
                        "[SYSTEM: One or more tool calls were rejected by tool composability contracts. "
                        f"Use tools compatible with available artifacts: {sorted(available_artifacts)}. "
                        "Adjust plan and try again.]"
                    ),
                }

        # Control-loop suppression for repeated invalid control actions:
        # 1) block repeated submit_answer attempts until TODO state changes;
        # 2) block repeated todo_update(done) retries on the same TODO after
        #    repeated validation failures.
        suppressed_records: list[MCPToolCallRecord] = []
        suppressed_tool_messages: list[dict[str, Any]] = []
        filtered_tool_calls: list[dict[str, Any]] = []
        for tc in tool_calls_to_execute:
            tool_name = tc.get("function", {}).get("name", "")
            tc_id = tc.get("id", "")
            parsed_args = _extract_tool_call_args(tc) or {}

            if tool_name == "submit_answer" and submit_requires_todo_progress:
                unresolved_hint = (
                    f" Unfinished TODOs currently blocking submit: {', '.join(pending_submit_todo_ids)}. "
                    "Call todo_update(todo_id=<id>, status='done' with evidence_refs/confidence) "
                    "or status='blocked' with a concise note, then retry submit."
                    if pending_submit_todo_ids else
                    ""
                )
                err = (
                    "submit_answer suppressed: TODO state has not changed since the last unfinished-TODO "
                    "submission failure. Call todo_update(...) or todo_list() to unblock first."
                    + unresolved_hint
                )
                suppressed_records.append(
                    MCPToolCallRecord(
                        server="__agent__",
                        tool=tool_name,
                        arguments=parsed_args,
                        error=err,
                    )
                )
                suppressed_tool_messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": _json.dumps({"error": err}),
                })
                continue

            if tool_name == "todo_update":
                todo_id = str(parsed_args.get("todo_id", "")).strip()
                status = str(parsed_args.get("status", "")).strip().lower()
                if (
                    todo_id and status == "done"
                    and todo_done_error_streak.get(todo_id, 0) >= 2
                ):
                    err = (
                        "todo_update(done) suppressed: repeated completion failures for this TODO. "
                        "Change strategy first (status='blocked' or upstream bridge repair) before retrying done."
                    )
                    suppressed_records.append(
                        MCPToolCallRecord(
                            server="__agent__",
                            tool=tool_name,
                            arguments=parsed_args,
                            error=err,
                        )
                    )
                    suppressed_tool_messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": _json.dumps({"error": err}),
                    })
                    continue

            filtered_tool_calls.append(tc)

        tool_calls_to_execute = filtered_tool_calls
        pending_control_loop_msg: dict[str, Any] | None = None
        if suppressed_records:
            control_loop_suppressed_calls += len(suppressed_records)
            pending_control_loop_msg = {
                "role": "user",
                "content": (
                    "[SYSTEM: Repeated control-call loop detected. Some tool calls were suppressed. "
                    "Update TODO state or change hypotheses before retrying submit/completion.]"
                ),
            }
            for rec in suppressed_records:
                _emit_foundation_event(
                    {
                        "event_id": new_event_id(),
                        "event_type": "ToolFailed",
                        "timestamp": now_iso(),
                        "run_id": foundation_run_id,
                        "session_id": foundation_session_id,
                        "actor_id": foundation_actor_id,
                        "operation": {"name": rec.tool or "<unknown>", "version": None},
                        "inputs": {
                            "artifact_ids": sorted(available_artifacts),
                            "params": rec.arguments if isinstance(rec.arguments, dict) else {},
                            "bindings": dict(available_bindings),
                        },
                        "outputs": {"artifact_ids": [], "payload_hashes": []},
                        "failure": {
                            "error_code": "control_loop_suppressed",
                            "category": "policy",
                            "phase": "post_validation",
                            "retryable": False,
                            "tool_name": rec.tool or "<unknown>",
                            "user_message": rec.error or "suppressed",
                            "debug_ref": None,
                        },
                    }
                )

        for tc in tool_calls_to_execute:
            tool_name = tc.get("function", {}).get("name", "") or "<unknown>"
            parsed_args = _extract_tool_call_args(tc)
            _emit_foundation_event(
                {
                    "event_id": new_event_id(),
                    "event_type": "ToolCalled",
                    "timestamp": now_iso(),
                    "run_id": foundation_run_id,
                    "session_id": foundation_session_id,
                    "actor_id": foundation_actor_id,
                    "operation": {"name": tool_name, "version": None},
                    "inputs": {
                        "artifact_ids": sorted(available_artifacts),
                        "params": parsed_args if isinstance(parsed_args, dict) else {},
                        "bindings": dict(available_bindings),
                    },
                    "outputs": {"artifact_ids": [], "payload_hashes": []},
                }
            )

        # Execute valid tool calls via executor
        if tool_calls_to_execute:
            executed_records, executed_tool_messages = await executor(
                tool_calls_to_execute, tool_result_max_length,
            )
        else:
            executed_records, executed_tool_messages = [], []

        for record in executed_records:
            if record.error:
                _emit_foundation_event(
                    {
                        "event_id": new_event_id(),
                        "event_type": "ToolFailed",
                        "timestamp": now_iso(),
                        "run_id": foundation_run_id,
                        "session_id": foundation_session_id,
                        "actor_id": foundation_actor_id,
                        "operation": {"name": record.tool or "<unknown>", "version": None},
                        "inputs": {
                            "artifact_ids": sorted(available_artifacts),
                            "params": record.arguments if isinstance(record.arguments, dict) else {},
                            "bindings": dict(available_bindings),
                        },
                        "outputs": {"artifact_ids": [], "payload_hashes": []},
                        "failure": {
                            "error_code": "tool_runtime_error",
                            "category": "execution",
                            "phase": "execution",
                            "retryable": False,
                            "tool_name": record.tool or "<unknown>",
                            "user_message": record.error,
                            "debug_ref": None,
                        },
                    }
                )
                continue

            observed_bindings = extract_bindings_from_tool_args(record.arguments)
            merged_bindings = merge_binding_state(
                available_bindings=available_bindings,
                observed_bindings=observed_bindings,
            )
            if merged_bindings != available_bindings:
                old_bindings = dict(available_bindings)
                available_bindings = merged_bindings
                _emit_foundation_event(
                    {
                        "event_id": new_event_id(),
                        "event_type": "BindingChanged",
                        "timestamp": now_iso(),
                        "run_id": foundation_run_id,
                        "session_id": foundation_session_id,
                        "actor_id": foundation_actor_id,
                        "operation": {"name": record.tool or "<unknown>", "version": None},
                        "inputs": {
                            "artifact_ids": sorted(available_artifacts),
                            "params": record.arguments if isinstance(record.arguments, dict) else {},
                            "bindings": old_bindings,
                        },
                        "outputs": {"artifact_ids": [], "payload_hashes": []},
                        "binding_change": {
                            "old_bindings": old_bindings,
                            "new_bindings": dict(available_bindings),
                            "reason": "adopted_new_binding_from_successful_tool_call",
                        },
                    }
                )

            produced: set[str] = set()
            if enforce_tool_contracts and normalized_tool_contracts:
                produced = _contract_outputs(normalized_tool_contracts.get(record.tool))
                if produced:
                    available_artifacts.update(produced)
                    artifact_timeline.append({
                        "turn": turn + 1,
                        "phase": "tool_success",
                        "tool": record.tool,
                        "produced": sorted(produced),
                        "available_artifacts": sorted(available_artifacts),
                    })

            _emit_foundation_event(
                {
                    "event_id": new_event_id(),
                    "event_type": "ArtifactCreated",
                    "timestamp": now_iso(),
                    "run_id": foundation_run_id,
                    "session_id": foundation_session_id,
                    "actor_id": foundation_actor_id,
                    "operation": {"name": record.tool or "<unknown>", "version": None},
                    "inputs": {
                        "artifact_ids": sorted(available_artifacts),
                        "params": record.arguments if isinstance(record.arguments, dict) else {},
                        "bindings": dict(available_bindings),
                    },
                    "outputs": {
                        "artifact_ids": sorted(produced),
                        "payload_hashes": (
                            [sha256_text(record.result or "")]
                            if record.result is not None else []
                        ),
                    },
                }
            )

        records = contract_rejected_records + suppressed_records + executed_records
        tool_messages = contract_rejected_messages + suppressed_tool_messages + executed_tool_messages
        agent_result.tool_calls.extend(records)
        messages.extend(tool_messages)
        if pending_repair_msg is not None:
            messages.append(pending_repair_msg)
            agent_result.conversation_trace.append(pending_repair_msg)
        if pending_contract_msg is not None:
            messages.append(pending_contract_msg)
            agent_result.conversation_trace.append(pending_contract_msg)
        if pending_control_loop_msg is not None:
            messages.append(pending_control_loop_msg)
            agent_result.conversation_trace.append(pending_control_loop_msg)

        turn_arg_coercions = 0
        turn_arg_coercion_calls = 0
        turn_arg_validation_rejections = 0
        for record in records:
            coercions = getattr(record, "arg_coercions", None) or []
            if coercions:
                turn_arg_coercions += len(coercions)
                turn_arg_coercion_calls += 1
            if record.error and "validation error:" in record.error.lower():
                turn_arg_validation_rejections += 1

        if turn_arg_coercions:
            tool_arg_coercions += turn_arg_coercions
            tool_arg_coercion_calls += turn_arg_coercion_calls
            warning = (
                "TOOL_ARG_COERCION: turn "
                f"{turn + 1}/{max_turns} applied {turn_arg_coercions} coercion(s) "
                f"across {turn_arg_coercion_calls} call(s)"
            )
            agent_result.warnings.append(warning)
            logger.warning(warning)
        if turn_arg_validation_rejections:
            tool_arg_validation_rejections += turn_arg_validation_rejections
            warning = (
                "TOOL_ARG_VALIDATION: turn "
                f"{turn + 1}/{max_turns} rejected {turn_arg_validation_rejections} "
                "tool call(s) due to argument/schema mismatch"
            )
            agent_result.warnings.append(warning)
            logger.warning(warning)

        submitted_answer_value: str | None = None
        submit_error_this_turn = False
        submit_errors: list[str] = []
        for record in records:
            if record.tool != "submit_answer":
                continue
            if record.error:
                submit_error_this_turn = True
                submit_errors.append(str(record.error))
                continue
            parsed = _parse_record_result_json(record)
            if parsed is not None:
                status = str(parsed.get("status", "")).strip().lower()
                if status and status not in {"submitted", "ok", "success"}:
                    submit_error_this_turn = True
                    validation = parsed.get("validation_error")
                    reason_code = ""
                    detail = ""
                    if isinstance(validation, dict):
                        reason_code = str(validation.get("reason_code", "")).strip()
                        detail = str(validation.get("message", "")).strip()
                    err_parts = [f"submit_answer not accepted (status={status})"]
                    if reason_code:
                        err_parts.append(f"reason_code={reason_code}")
                    if detail:
                        err_parts.append(detail)
                    submit_errors.append(" | ".join(err_parts))
                    continue

            submit_answer_succeeded = True
            arg_answer = record.arguments.get("answer") if isinstance(record.arguments, dict) else None
            if isinstance(arg_answer, str) and arg_answer.strip():
                submitted_answer_value = arg_answer.strip()
                continue
            if isinstance(parsed, dict):
                parsed_answer = parsed.get("answer")
                if isinstance(parsed_answer, str) and parsed_answer.strip():
                    submitted_answer_value = parsed_answer.strip()

        todo_progress_this_turn = False
        for record in records:
            if record.tool != "todo_update":
                continue
            args = record.arguments if isinstance(record.arguments, dict) else {}
            todo_id = str(args.get("todo_id", "")).strip()
            status = str(args.get("status", "")).strip().lower()
            parsed = _parse_record_result_json(record)
            result_status = str(parsed.get("status", "")).strip().lower() if isinstance(parsed, dict) else ""
            needs_revision = result_status == "needs_revision"
            if record.error or needs_revision:
                if todo_id and status == "done":
                    todo_done_error_streak[todo_id] = todo_done_error_streak.get(todo_id, 0) + 1
                continue
            todo_progress_this_turn = True
            if todo_id:
                todo_done_error_streak[todo_id] = 0
        if todo_progress_this_turn:
            submit_requires_todo_progress = False

        if submit_error_this_turn:
            todo_blocked = any(
                ("todo" in err.lower()) or ("unfinished" in err.lower())
                for err in submit_errors
            )
            pending_submit_todo_ids = []
            for err in submit_errors:
                pending_submit_todo_ids.extend(_extract_unfinished_todo_ids(err))
            # Keep stable order and uniqueness
            if pending_submit_todo_ids:
                deduped_ids: list[str] = []
                seen_ids: set[str] = set()
                for todo_id in pending_submit_todo_ids:
                    key = todo_id.lower()
                    if key in seen_ids:
                        continue
                    seen_ids.add(key)
                    deduped_ids.append(todo_id)
                pending_submit_todo_ids = deduped_ids
            refusal_blocked = any(
                ("refusal-style" in err.lower()) or ("not acceptable" in err.lower())
                for err in submit_errors
            )
            if todo_blocked and not todo_progress_this_turn:
                submit_requires_todo_progress = True
            todo_fix_hint = ""
            if pending_submit_todo_ids:
                todo_fix_hint = (
                    " Unfinished TODO IDs: "
                    + ", ".join(pending_submit_todo_ids)
                    + ". For each unresolved leaf TODO, call "
                    "todo_update(todo_id=<id>, status='blocked', note='why evidence is insufficient') "
                    "or mark done with evidence refs before submit."
                )
            retry_submit_msg = {
                "role": "user",
                "content": (
                    "[SYSTEM: submit_answer failed. "
                    + (
                        "Create/update TODOs first: todo_create(...), "
                        "todo_update(..., status='done'), todo_list(). Then call submit_answer again. "
                        if todo_blocked else
                        ""
                    )
                    + todo_fix_hint
                    + (
                        "Do NOT use refusal text (cannot, unknown, insufficient, no such, not found). "
                        "Submit the single best factual guess from retrieved evidence. "
                        if refusal_blocked else
                        ""
                    )
                    + "Call submit_answer again with answer as a short fact only "
                    "(name/date/number/yes/no, <=8 words).]"
                ),
            }
            messages.append(retry_submit_msg)
            agent_result.conversation_trace.append(retry_submit_msg)
        elif todo_progress_this_turn:
            pending_submit_todo_ids = []

        # Repeated tool-error detection: nudge strategy change instead of
        # repeatedly retrying near-identical invalid calls.
        for record in records:
            if not record.error:
                continue
            sig = _tool_error_signature(str(record.error))
            key = (record.tool, sig)
            count = tool_error_counts.get(key, 0) + 1
            tool_error_counts[key] = count

            last_nudged = tool_error_nudges.get(key, 0)
            if count >= 2 and count > last_nudged:
                todo_hint = (
                    " If this is a TODO completion error, reopen upstream TODO(s) "
                    "and gather new evidence before retrying."
                    if record.tool == "todo_update" else
                    ""
                )
                err_msg = {
                    "role": "user",
                    "content": (
                        f"[SYSTEM: Repeated tool error from '{record.tool}': {sig}. "
                        "STOP retrying near-identical calls. Change strategy "
                        "(new evidence, alternate hypothesis, or upstream repair) before continuing.]"
                        + todo_hint
                    ),
                }
                messages.append(err_msg)
                agent_result.conversation_trace.append(err_msg)
                tool_error_nudges[key] = count
                logger.warning(
                    "Loop detection: repeated tool error from '%s' (count=%d): %s",
                    record.tool, count, sig,
                )

        # Track tool call frequency for loop detection
        for tc in tool_calls_to_execute:
            tc_name = tc.get("function", {}).get("name", "")
            if tc_name and not _is_budget_exempt_tool(tc_name):
                tool_call_counts[tc_name] = tool_call_counts.get(tc_name, 0) + 1

        # Loop detection: if same tool called 3+ times, nudge model to switch
        for tool_name, count in tool_call_counts.items():
            last_nudged = tool_loop_nudges.get(tool_name, 0)
            if count >= 3 and count % 3 == 0 and count > last_nudged:
                loop_msg = {
                    "role": "user",
                    "content": (
                        f"[SYSTEM: You have called '{tool_name}' {count} times. "
                        "STOP repeating the same searches. Read what you already have. "
                        "Try a DIFFERENT tool or submit your answer now.]"
                    ),
                }
                messages.append(loop_msg)
                agent_result.conversation_trace.append(loop_msg)
                tool_loop_nudges[tool_name] = count
                logger.warning(
                    "Loop detection: tool '%s' called %d times on turn %d",
                    tool_name, count, turn + 1,
                )

        # Capture tool results in trace
        for tmsg in tool_messages:
            agent_result.conversation_trace.append({
                "role": "tool",
                "tool_call_id": tmsg.get("tool_call_id", ""),
                "content": tmsg.get("content", ""),
            })

        if submit_answer_succeeded:
            final_content = submitted_answer_value or (result.content or "").strip() or "submitted"
            final_finish_reason = "submitted"
            logger.info(
                "Agent loop: submit_answer succeeded on turn %d/%d",
                turn + 1, max_turns,
            )
            break

        logger.debug(
            "Agent turn %d/%d: %d tool calls",
            turn + 1, max_turns, len(tool_calls_this_turn),
        )

        # Turn countdown: warn agent when it's running low on turns
        remaining = max_turns - (turn + 1)
        if remaining == TURN_WARNING_THRESHOLD:
            countdown_msg = {
                "role": "user",
                "content": (
                    f"[SYSTEM: You have {remaining} turns remaining. "
                    "Submit your best answer now based on the evidence you have. "
                    "Do not search for more information.]"
                ),
            }
            messages.append(countdown_msg)
            agent_result.conversation_trace.append(countdown_msg)
    else:
        force_final_reason = "max_turns"

    if force_final_reason is not None and not submit_answer_succeeded:
        if force_final_reason == "max_tool_calls":
            force_msg = {
                "role": "user",
                "content": (
                    "[SYSTEM: Tool-call budget exhausted. You MUST give your best answer now "
                    "using evidence already retrieved. Extract the best name/date/number and answer.]"
                ),
            }
        else:
            logger.warning(
                "Agent loop exhausted max_turns=%d, forcing final answer",
                max_turns,
            )
            force_msg = {
                "role": "user",
                "content": (
                    "[SYSTEM: All turns exhausted. You MUST give your best answer now "
                    "based on the evidence you found. Extract any name, date, or number "
                    "from the tool results. 'I don't know' is NOT acceptable — guess.]"
                ),
            }

        messages.append(force_msg)
        agent_result.conversation_trace.append(force_msg)
        compacted_count, compacted_chars, current_chars = _compact_tool_history_for_context(
            messages, max_message_chars,
        )
        if compacted_count:
            context_compactions += 1
            context_compacted_messages += compacted_count
            context_compacted_chars += compacted_chars
            warning = (
                "CONTEXT_COMPACTION: compacted "
                f"{compacted_count} tool message(s), saved ~{compacted_chars} chars "
                f"(history ~{current_chars} chars) before forced-final call"
            )
            agent_result.warnings.append(warning)
            logger.warning(warning)
        final_result = await _inner_acall_llm(
            effective_model, messages, timeout=timeout, **kwargs,
        )
        agent_result.models_used.add(final_result.model)
        if final_result.warnings:
            agent_result.warnings.extend(final_result.warnings)
        final_content = final_result.content
        final_finish_reason = final_result.finish_reason
        total_cost += final_result.cost
        inp, out, cached, cache_create = _extract_usage(final_result.usage or {})
        agent_result.total_input_tokens += inp
        agent_result.total_output_tokens += out
        agent_result.total_cached_tokens += cached
        agent_result.total_cache_creation_tokens += cache_create
        agent_result.turns += 1
        # Capture forced final answer in trace
        if final_content:
            agent_result.conversation_trace.append({
                "role": "assistant",
                "content": final_content,
            })

    agent_result.metadata["total_cost"] = total_cost
    agent_result.metadata["max_turns"] = max_turns
    agent_result.metadata["max_tool_calls"] = max_tool_calls
    agent_result.metadata["tool_calls_used"] = len(agent_result.tool_calls)
    agent_result.metadata["budgeted_tool_calls_used"] = _count_budgeted_records(agent_result.tool_calls)
    agent_result.metadata["budget_exempt_tools"] = sorted(BUDGET_EXEMPT_TOOL_NAMES)
    agent_result.metadata["require_tool_reasoning"] = require_tool_reasoning
    agent_result.metadata["rejected_missing_reasoning_calls"] = rejected_missing_reasoning_calls
    agent_result.metadata["control_loop_suppressed_calls"] = control_loop_suppressed_calls
    agent_result.metadata["tool_call_turns_total"] = tool_call_turns_total
    agent_result.metadata["tool_call_empty_text_turns"] = tool_call_empty_text_turns
    agent_result.metadata["responses_tool_call_empty_text_turns"] = responses_tool_call_empty_text_turns
    agent_result.metadata["tool_call_empty_text_turn_ratio"] = (
        (tool_call_empty_text_turns / tool_call_turns_total)
        if tool_call_turns_total else 0.0
    )
    agent_result.metadata["tool_arg_coercions"] = tool_arg_coercions
    agent_result.metadata["tool_arg_coercion_calls"] = tool_arg_coercion_calls
    agent_result.metadata["tool_arg_validation_rejections"] = tool_arg_validation_rejections
    agent_result.metadata["context_compactions"] = context_compactions
    agent_result.metadata["context_compacted_messages"] = context_compacted_messages
    agent_result.metadata["context_compacted_chars"] = context_compacted_chars
    agent_result.metadata["enforce_tool_contracts"] = enforce_tool_contracts
    agent_result.metadata["progressive_tool_disclosure"] = progressive_tool_disclosure
    agent_result.metadata["tool_contract_rejections"] = contract_rejected_calls
    agent_result.metadata["tool_contract_violation_events"] = contract_violation_events
    agent_result.metadata["tool_contracts_declared"] = sorted(normalized_tool_contracts.keys())
    agent_result.metadata["initial_artifacts"] = initial_artifact_snapshot
    agent_result.metadata["available_artifacts_final"] = sorted(available_artifacts)
    agent_result.metadata["artifact_timeline"] = artifact_timeline
    agent_result.metadata["initial_bindings"] = initial_binding_snapshot
    agent_result.metadata["available_bindings_final"] = dict(available_bindings)
    agent_result.metadata["tool_disclosure_turns"] = tool_disclosure_turns
    agent_result.metadata["tool_disclosure_hidden_total"] = tool_disclosure_hidden_total
    agent_result.metadata["foundation_event_count"] = len(foundation_events)
    agent_result.metadata["foundation_event_types"] = dict(foundation_event_types)
    agent_result.metadata["foundation_event_validation_errors"] = foundation_event_validation_errors
    agent_result.metadata["foundation_events_logged"] = foundation_events_logged
    agent_result.metadata["foundation_events"] = foundation_events
    if tool_call_turns_total > 0:
        agent_result.warnings.append(
            "METRIC: tool_call_empty_text_turns="
            f"{tool_call_empty_text_turns}/{tool_call_turns_total}, "
            f"responses_tool_call_empty_text_turns={responses_tool_call_empty_text_turns}"
        )
    if tool_arg_coercions or tool_arg_validation_rejections:
        agent_result.warnings.append(
            "METRIC: tool_arg_coercions="
            f"{tool_arg_coercions} across {tool_arg_coercion_calls} calls, "
            f"tool_arg_validation_rejections={tool_arg_validation_rejections}"
        )
    logger.info(
        "Agent loop metrics: tool_call_turns=%d empty_text_tool_call_turns=%d responses_empty_text_tool_call_turns=%d "
        "tool_arg_coercions=%d tool_arg_validation_rejections=%d",
        tool_call_turns_total,
        tool_call_empty_text_turns,
        responses_tool_call_empty_text_turns,
        tool_arg_coercions,
        tool_arg_validation_rejections,
    )
    return final_content, final_finish_reason


# ---------------------------------------------------------------------------
# Direct Python Tool Loop
# ---------------------------------------------------------------------------


async def _acall_with_tools(
    model: str,
    messages: list[dict[str, Any]],
    python_tools: list[Any],
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_tool_calls: int | None = DEFAULT_MAX_TOOL_CALLS,
    require_tool_reasoning: bool = DEFAULT_REQUIRE_TOOL_REASONING,
    tool_result_max_length: int = DEFAULT_TOOL_RESULT_MAX_LENGTH,
    max_message_chars: int = DEFAULT_MAX_MESSAGE_CHARS,
    enforce_tool_contracts: bool = DEFAULT_ENFORCE_TOOL_CONTRACTS,
    progressive_tool_disclosure: bool = DEFAULT_PROGRESSIVE_TOOL_DISCLOSURE,
    tool_contracts: dict[str, dict[str, Any]] | None = None,
    initial_artifacts: list[str] | tuple[str, ...] | None = DEFAULT_INITIAL_ARTIFACTS,
    initial_bindings: dict[str, Any] | None = None,
    timeout: int = 60,
    **kwargs: Any,
) -> LLMCallResult:
    """Run a tool-calling agent loop with direct Python functions.

    Same loop as _acall_with_mcp but calls Python functions in-process
    instead of going through MCP subprocess/stdio/JSON-RPC.

    Args:
        model: Any litellm model string (NOT an agent model).
        messages: Initial messages (system + user).
        python_tools: List of typed Python callables (sync or async).
        max_turns: Maximum loop iterations.
        max_tool_calls: Maximum tool calls before final forced answer. None disables.
        require_tool_reasoning: If True, reject tool calls missing tool_reasoning.
        tool_result_max_length: Max chars per tool result.
        progressive_tool_disclosure: If True, hide tools whose contracts are not currently satisfiable.
        initial_bindings: Binding state available before any tool call.
        timeout: Per-turn LLM call timeout.
        **kwargs: Passed through to acall_llm.
    """
    from llm_client.tool_utils import execute_direct_tool_calls, prepare_direct_tools

    tool_map, openai_tools = prepare_direct_tools(python_tools)

    if not openai_tools:
        raise ValueError("python_tools list is empty — no tools to call.")

    agent_result = MCPAgentResult()
    messages = list(messages)  # don't mutate caller's list

    async def _direct_executor(
        tool_calls: list[dict[str, Any]], max_len: int,
    ) -> tuple[list[MCPToolCallRecord], list[dict[str, Any]]]:
        return await execute_direct_tool_calls(
            tool_calls,
            tool_map,
            max_len,
            require_tool_reasoning=require_tool_reasoning,
        )

    final_content, final_finish_reason = await _agent_loop(
        model, messages, openai_tools,
        agent_result,
        _direct_executor,
        max_turns,
        max_tool_calls,
        require_tool_reasoning,
        tool_result_max_length,
        max_message_chars,
        enforce_tool_contracts,
        progressive_tool_disclosure,
        tool_contracts,
        initial_artifacts,
        initial_bindings,
        timeout,
        kwargs,
    )

    usage = {
            "input_tokens": agent_result.total_input_tokens,
            "output_tokens": agent_result.total_output_tokens,
            "total_tokens": (
                agent_result.total_input_tokens + agent_result.total_output_tokens
            ),
        }
    if agent_result.total_cached_tokens:
        usage["cached_tokens"] = agent_result.total_cached_tokens
    if agent_result.total_cache_creation_tokens:
        usage["cache_creation_tokens"] = agent_result.total_cache_creation_tokens
    return LLMCallResult(
        content=final_content,
        usage=usage,
        cost=agent_result.metadata.get("total_cost", 0.0),
        model=model,
        finish_reason=final_finish_reason,
        raw_response=agent_result,
        warnings=agent_result.warnings,
    )
