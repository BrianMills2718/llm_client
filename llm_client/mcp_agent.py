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
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from llm_client.client import LLMCallResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configurable defaults — all overridable via kwargs to call_llm/acall_llm
# ---------------------------------------------------------------------------

DEFAULT_MAX_TURNS: int = 20
"""Maximum tool-calling loop iterations before forcing a final answer."""

DEFAULT_MCP_INIT_TIMEOUT: float = 30.0
"""Seconds to wait for each MCP server subprocess to initialize."""

DEFAULT_TOOL_RESULT_MAX_LENGTH: int = 50_000
"""Maximum character length for a single tool result. Longer results are truncated."""

# Kwargs consumed by the MCP agent loop (popped before passing to inner acall_llm)
MCP_LOOP_KWARGS = frozenset({
    "mcp_servers",
    "mcp_sessions",
    "max_turns",
    "mcp_init_timeout",
    "tool_result_max_length",
})

# Kwargs consumed by the direct tool loop
TOOL_LOOP_KWARGS = frozenset({
    "python_tools",
    "max_turns",
    "tool_result_max_length",
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
    result: str | None = None
    error: str | None = None
    latency_s: float = 0.0


@dataclass
class MCPAgentResult:
    """Accumulated result from the MCP agent loop.

    Stored in LLMCallResult.raw_response for introspection.
    """

    tool_calls: list[MCPToolCallRecord] = field(default_factory=list)
    turns: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    conversation_trace: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


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
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_usage(usage: dict[str, Any]) -> tuple[int, int]:
    """Extract (input_tokens, output_tokens) from usage dict.

    Handles both OpenAI convention (prompt_tokens/completion_tokens)
    and Anthropic convention (input_tokens/output_tokens).
    """
    inp = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
    out = usage.get("output_tokens") or usage.get("completion_tokens") or 0
    return int(inp), int(out)


def _truncate(text: str, max_length: int) -> str:
    """Truncate text if it exceeds max_length, appending a notice."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + f"\n... [truncated at {max_length} chars]"


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
        except _json.JSONDecodeError:
            arguments = {}

        server_name = tool_to_server.get(tool_name)
        record = MCPToolCallRecord(
            server=server_name or "unknown",
            tool=tool_name,
            arguments=arguments,
        )

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
    mcp_init_timeout: float = DEFAULT_MCP_INIT_TIMEOUT,
    tool_result_max_length: int = DEFAULT_TOOL_RESULT_MAX_LENGTH,
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
        mcp_init_timeout: Seconds to wait for server startup
        tool_result_max_length: Max chars per tool result (truncated if longer)
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
            return await _execute_tool_calls(tool_calls, sessions, tool_to_server, max_len)

        final_content, final_finish_reason = await _agent_loop(
            model, messages, openai_tools,
            agent_result, _mcp_executor, max_turns, tool_result_max_length, timeout, kwargs,
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
                return await _execute_tool_calls(tool_calls, sessions, tool_to_server, max_len)

            final_content, final_finish_reason = await _agent_loop(
                model, messages, openai_tools,
                agent_result, _mcp_executor, max_turns, tool_result_max_length, timeout, kwargs,
            )
    else:
        raise ValueError("Either mcp_servers or mcp_sessions must be provided")

    # Cost is accumulated during _agent_loop via agent_result metadata
    return LLMCallResult(
        content=final_content,
        usage={
            "input_tokens": agent_result.total_input_tokens,
            "output_tokens": agent_result.total_output_tokens,
            "total_tokens": (
                agent_result.total_input_tokens + agent_result.total_output_tokens
            ),
        },
        cost=agent_result.metadata.get("total_cost", 0.0),
        model=model,
        finish_reason=final_finish_reason,
        raw_response=agent_result,
    )


async def _agent_loop(
    model: str,
    messages: list[dict[str, Any]],
    openai_tools: list[dict[str, Any]],
    agent_result: MCPAgentResult,
    executor: Any,  # Callable[[list, int], Awaitable[tuple[list[MCPToolCallRecord], list[dict]]]]
    max_turns: int,
    tool_result_max_length: int,
    timeout: int,
    kwargs: dict[str, Any],
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

    for turn in range(max_turns):
        agent_result.turns = turn + 1

        result = await _inner_acall_llm(
            model, messages, timeout=timeout, tools=openai_tools, **kwargs,
        )

        # Accumulate usage
        inp, out = _extract_usage(result.usage or {})
        agent_result.total_input_tokens += inp
        agent_result.total_output_tokens += out
        total_cost += result.cost

        # No tool calls → done
        if not result.tool_calls:
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
        assistant_msg = {
            "role": "assistant",
            "content": result.content or None,
            "tool_calls": result.tool_calls,
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
                for tc in result.tool_calls
            ],
        })

        # Execute tool calls via executor
        records, tool_messages = await executor(result.tool_calls, tool_result_max_length)
        agent_result.tool_calls.extend(records)
        messages.extend(tool_messages)

        # Capture tool results in trace
        for tmsg in tool_messages:
            agent_result.conversation_trace.append({
                "role": "tool",
                "tool_call_id": tmsg.get("tool_call_id", ""),
                "content": tmsg.get("content", ""),
            })

        logger.debug(
            "Agent turn %d/%d: %d tool calls",
            turn + 1, max_turns, len(result.tool_calls),
        )
    else:
        # max_turns exhausted — one final call without tools to get an answer
        logger.warning(
            "Agent loop exhausted max_turns=%d, forcing final answer",
            max_turns,
        )
        final_result = await _inner_acall_llm(
            model, messages, timeout=timeout, **kwargs,
        )
        final_content = final_result.content
        final_finish_reason = final_result.finish_reason
        total_cost += final_result.cost
        inp, out = _extract_usage(final_result.usage or {})
        agent_result.total_input_tokens += inp
        agent_result.total_output_tokens += out
        agent_result.turns += 1
        # Capture forced final answer in trace
        if final_content:
            agent_result.conversation_trace.append({
                "role": "assistant",
                "content": final_content,
            })

    agent_result.metadata["total_cost"] = total_cost
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
    tool_result_max_length: int = DEFAULT_TOOL_RESULT_MAX_LENGTH,
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
        tool_result_max_length: Max chars per tool result.
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
        return await execute_direct_tool_calls(tool_calls, tool_map, max_len)

    final_content, final_finish_reason = await _agent_loop(
        model, messages, openai_tools,
        agent_result, _direct_executor, max_turns, tool_result_max_length, timeout, kwargs,
    )

    return LLMCallResult(
        content=final_content,
        usage={
            "input_tokens": agent_result.total_input_tokens,
            "output_tokens": agent_result.total_output_tokens,
            "total_tokens": (
                agent_result.total_input_tokens + agent_result.total_output_tokens
            ),
        },
        cost=agent_result.metadata.get("total_cost", 0.0),
        model=model,
        finish_reason=final_finish_reason,
        raw_response=agent_result,
    )
