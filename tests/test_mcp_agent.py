"""Tests for MCP agent loop. All mocked (no real MCP servers or LLM calls).

Tests cover:
- _mcp_tool_to_openai() schema conversion
- _extract_usage() with OpenAI and Anthropic conventions
- _truncate() behavior
- _acall_with_mcp() full agent loop (start, discover, loop, cleanup)
- Routing: mcp_servers on non-agent model goes through MCP loop
- Routing: mcp_servers on agent model goes through existing agent path
- max_turns exhaustion
- Error handling: no tools, unknown tool, MCP tool error
- MCP_LOOP_KWARGS are popped before inner acall_llm
"""

# mock-ok: MCP servers require subprocess lifecycle; unit tests must mock

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_client import (
    DEFAULT_MAX_TURNS,
    DEFAULT_MCP_INIT_TIMEOUT,
    DEFAULT_TOOL_RESULT_MAX_LENGTH,
    LLMCallResult,
    MCPAgentResult,
    MCPToolCallRecord,
    acall_llm,
    call_llm,
)
from llm_client.mcp_agent import (
    MCP_LOOP_KWARGS,
    _extract_usage,
    _mcp_tool_to_openai,
    _truncate,
)


# ---------------------------------------------------------------------------
# Schema conversion
# ---------------------------------------------------------------------------


class TestMcpToolToOpenai:
    def test_basic_conversion(self) -> None:
        tool = MagicMock()
        tool.name = "search"
        tool.description = "Search for entities"
        tool.inputSchema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        result = _mcp_tool_to_openai(tool)
        assert result == {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search for entities",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }

    def test_missing_description(self) -> None:
        tool = MagicMock()
        tool.name = "foo"
        tool.description = None
        tool.inputSchema = {"type": "object", "properties": {}}
        result = _mcp_tool_to_openai(tool)
        assert result["function"]["description"] == ""

    def test_missing_schema(self) -> None:
        tool = MagicMock()
        tool.name = "bar"
        tool.description = "desc"
        tool.inputSchema = None
        result = _mcp_tool_to_openai(tool)
        assert result["function"]["parameters"] == {"type": "object", "properties": {}}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestExtractUsage:
    def test_openai_convention(self) -> None:
        inp, out = _extract_usage({"prompt_tokens": 100, "completion_tokens": 50})
        assert inp == 100
        assert out == 50

    def test_anthropic_convention(self) -> None:
        inp, out = _extract_usage({"input_tokens": 200, "output_tokens": 75})
        assert inp == 200
        assert out == 75

    def test_empty_usage(self) -> None:
        inp, out = _extract_usage({})
        assert inp == 0
        assert out == 0

    def test_input_tokens_takes_priority(self) -> None:
        """input_tokens is checked before prompt_tokens."""
        inp, out = _extract_usage({
            "input_tokens": 300,
            "prompt_tokens": 100,
            "output_tokens": 50,
        })
        assert inp == 300


class TestTruncate:
    def test_no_truncation(self) -> None:
        assert _truncate("hello", 100) == "hello"

    def test_exact_limit(self) -> None:
        assert _truncate("hello", 5) == "hello"

    def test_truncation(self) -> None:
        result = _truncate("hello world", 5)
        assert result.startswith("hello")
        assert "truncated at 5 chars" in result


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_default_max_turns(self) -> None:
        assert DEFAULT_MAX_TURNS == 20

    def test_default_init_timeout(self) -> None:
        assert DEFAULT_MCP_INIT_TIMEOUT == 30.0

    def test_default_tool_result_max_length(self) -> None:
        assert DEFAULT_TOOL_RESULT_MAX_LENGTH == 50_000

    def test_mcp_loop_kwargs(self) -> None:
        assert "mcp_servers" in MCP_LOOP_KWARGS
        assert "max_turns" in MCP_LOOP_KWARGS
        assert "mcp_init_timeout" in MCP_LOOP_KWARGS
        assert "tool_result_max_length" in MCP_LOOP_KWARGS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_tool(name: str, desc: str = "tool") -> MagicMock:
    t = MagicMock()
    t.name = name
    t.description = desc
    t.inputSchema = {"type": "object", "properties": {}}
    return t


def _make_tool_result(text: str, is_error: bool = False) -> MagicMock:
    content_item = MagicMock()
    content_item.text = text
    result = MagicMock()
    result.content = [content_item]
    result.isError = is_error
    return result


def _make_llm_result(
    content: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
    cost: float = 0.001,
    finish_reason: str = "stop",
) -> LLMCallResult:
    return LLMCallResult(
        content=content,
        usage=usage or {"input_tokens": 100, "output_tokens": 50},
        cost=cost,
        model="test-model",
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
    )


# ---------------------------------------------------------------------------
# Full agent loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAcallWithMcp:
    """Test _acall_with_mcp with fully mocked MCP and LLM."""

    async def test_single_turn_no_tools(self) -> None:
        """LLM returns a text answer immediately (no tool calls)."""
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(
            tools=[_make_tool("search")],
        ))

        with (
            patch("llm_client.mcp_agent._import_mcp") as mock_import,
            patch("llm_client.mcp_agent._inner_acall_llm") as mock_acall,
        ):
            # Setup MCP mocks
            mock_stdio = AsyncMock()
            mock_stdio.__aenter__ = AsyncMock(return_value=("read", "write"))
            mock_stdio.__aexit__ = AsyncMock(return_value=False)
            mock_import.return_value = (
                MagicMock(return_value=mock_stdio),  # stdio_client
                MagicMock,  # StdioServerParameters
                MagicMock(return_value=mock_session),  # ClientSession
            )

            # Setup session context manager
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            # LLM returns text answer
            mock_acall.return_value = _make_llm_result(content="Paris", finish_reason="stop")

            from llm_client.mcp_agent import _acall_with_mcp
            result = await _acall_with_mcp(
                "test-model",
                [{"role": "user", "content": "What is the capital of France?"}],
                mcp_servers={"server": {"command": "python", "args": ["s.py"]}},
            )

            assert result.content == "Paris"
            assert result.finish_reason == "stop"
            assert isinstance(result.raw_response, MCPAgentResult)
            assert result.raw_response.turns == 1
            assert result.raw_response.tool_calls == []

    async def test_multi_turn_with_tool_calls(self) -> None:
        """LLM calls a tool, gets result, then answers."""
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(
            tools=[_make_tool("search")],
        ))
        mock_session.call_tool = AsyncMock(
            return_value=_make_tool_result('{"entities": ["Paris"]}'),
        )

        with (
            patch("llm_client.mcp_agent._import_mcp") as mock_import,
            patch("llm_client.mcp_agent._inner_acall_llm") as mock_acall,
        ):
            mock_stdio = AsyncMock()
            mock_stdio.__aenter__ = AsyncMock(return_value=("read", "write"))
            mock_stdio.__aexit__ = AsyncMock(return_value=False)
            mock_import.return_value = (
                MagicMock(return_value=mock_stdio),
                MagicMock,
                MagicMock(return_value=mock_session),
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            # Turn 1: LLM calls a tool
            # Turn 2: LLM answers
            mock_acall.side_effect = [
                _make_llm_result(tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"query": "capital of France"}',
                    },
                }]),
                _make_llm_result(content="Paris"),
            ]

            from llm_client.mcp_agent import _acall_with_mcp
            result = await _acall_with_mcp(
                "test-model",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
            )

            assert result.content == "Paris"
            agent_result = result.raw_response
            assert isinstance(agent_result, MCPAgentResult)
            assert agent_result.turns == 2
            assert len(agent_result.tool_calls) == 1
            assert agent_result.tool_calls[0].tool == "search"
            assert agent_result.tool_calls[0].server == "srv"
            assert agent_result.tool_calls[0].result is not None
            assert agent_result.tool_calls[0].error is None

            # Verify the session.call_tool was called correctly
            mock_session.call_tool.assert_called_once_with(
                "search", {"query": "capital of France"},
            )

    async def test_max_turns_exhausted(self) -> None:
        """Loop reaches max_turns and makes a final call without tools."""
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(
            tools=[_make_tool("search")],
        ))
        mock_session.call_tool = AsyncMock(
            return_value=_make_tool_result("result"),
        )

        with (
            patch("llm_client.mcp_agent._import_mcp") as mock_import,
            patch("llm_client.mcp_agent._inner_acall_llm") as mock_acall,
        ):
            mock_stdio = AsyncMock()
            mock_stdio.__aenter__ = AsyncMock(return_value=("read", "write"))
            mock_stdio.__aexit__ = AsyncMock(return_value=False)
            mock_import.return_value = (
                MagicMock(return_value=mock_stdio),
                MagicMock,
                MagicMock(return_value=mock_session),
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            # Always return tool calls (never stops on its own)
            tool_call_result = _make_llm_result(tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": "{}"},
            }])
            final_result = _make_llm_result(content="forced answer")

            # max_turns=2: 2 loop iterations + 1 final call = 3 calls total
            mock_acall.side_effect = [tool_call_result, tool_call_result, final_result]

            from llm_client.mcp_agent import _acall_with_mcp
            result = await _acall_with_mcp(
                "test-model",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
                max_turns=2,
            )

            assert result.content == "forced answer"
            assert result.raw_response.turns == 3  # 2 loop + 1 forced

    async def test_no_tools_raises(self) -> None:
        """Error if MCP servers provide no tools."""
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=[]))

        with (
            patch("llm_client.mcp_agent._import_mcp") as mock_import,
        ):
            mock_stdio = AsyncMock()
            mock_stdio.__aenter__ = AsyncMock(return_value=("read", "write"))
            mock_stdio.__aexit__ = AsyncMock(return_value=False)
            mock_import.return_value = (
                MagicMock(return_value=mock_stdio),
                MagicMock,
                MagicMock(return_value=mock_session),
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            from llm_client.mcp_agent import _acall_with_mcp
            with pytest.raises(ValueError, match="No tools discovered"):
                await _acall_with_mcp(
                    "test-model",
                    [{"role": "user", "content": "Q"}],
                    mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
                )

    async def test_unknown_tool_handled(self) -> None:
        """Tool call for a tool not in any server is recorded as error."""
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(
            tools=[_make_tool("real_tool")],
        ))

        with (
            patch("llm_client.mcp_agent._import_mcp") as mock_import,
            patch("llm_client.mcp_agent._inner_acall_llm") as mock_acall,
        ):
            mock_stdio = AsyncMock()
            mock_stdio.__aenter__ = AsyncMock(return_value=("read", "write"))
            mock_stdio.__aexit__ = AsyncMock(return_value=False)
            mock_import.return_value = (
                MagicMock(return_value=mock_stdio),
                MagicMock,
                MagicMock(return_value=mock_session),
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_acall.side_effect = [
                _make_llm_result(tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "nonexistent_tool", "arguments": "{}"},
                }]),
                _make_llm_result(content="answer"),
            ]

            from llm_client.mcp_agent import _acall_with_mcp
            result = await _acall_with_mcp(
                "test-model",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
            )

            assert result.content == "answer"
            assert len(result.raw_response.tool_calls) == 1
            assert result.raw_response.tool_calls[0].error == "Unknown tool: nonexistent_tool"

    async def test_mcp_tool_error_handled(self) -> None:
        """MCP tool returning isError=true is recorded as error."""
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(
            tools=[_make_tool("search")],
        ))
        mock_session.call_tool = AsyncMock(
            return_value=_make_tool_result("something failed", is_error=True),
        )

        with (
            patch("llm_client.mcp_agent._import_mcp") as mock_import,
            patch("llm_client.mcp_agent._inner_acall_llm") as mock_acall,
        ):
            mock_stdio = AsyncMock()
            mock_stdio.__aenter__ = AsyncMock(return_value=("read", "write"))
            mock_stdio.__aexit__ = AsyncMock(return_value=False)
            mock_import.return_value = (
                MagicMock(return_value=mock_stdio),
                MagicMock,
                MagicMock(return_value=mock_session),
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_acall.side_effect = [
                _make_llm_result(tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }]),
                _make_llm_result(content="answer"),
            ]

            from llm_client.mcp_agent import _acall_with_mcp
            result = await _acall_with_mcp(
                "test-model",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
            )

            assert result.raw_response.tool_calls[0].error == "something failed"
            assert result.raw_response.tool_calls[0].result is None

    async def test_usage_accumulates(self) -> None:
        """Usage tokens accumulate across turns."""
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(
            tools=[_make_tool("search")],
        ))
        mock_session.call_tool = AsyncMock(
            return_value=_make_tool_result("data"),
        )

        with (
            patch("llm_client.mcp_agent._import_mcp") as mock_import,
            patch("llm_client.mcp_agent._inner_acall_llm") as mock_acall,
        ):
            mock_stdio = AsyncMock()
            mock_stdio.__aenter__ = AsyncMock(return_value=("read", "write"))
            mock_stdio.__aexit__ = AsyncMock(return_value=False)
            mock_import.return_value = (
                MagicMock(return_value=mock_stdio),
                MagicMock,
                MagicMock(return_value=mock_session),
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_acall.side_effect = [
                _make_llm_result(
                    tool_calls=[{
                        "id": "c1", "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }],
                    usage={"input_tokens": 100, "output_tokens": 20},
                    cost=0.01,
                ),
                _make_llm_result(
                    content="answer",
                    usage={"input_tokens": 200, "output_tokens": 30},
                    cost=0.02,
                ),
            ]

            from llm_client.mcp_agent import _acall_with_mcp
            result = await _acall_with_mcp(
                "test-model",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
            )

            assert result.usage["input_tokens"] == 300
            assert result.usage["output_tokens"] == 50
            assert result.usage["total_tokens"] == 350
            assert abs(result.cost - 0.03) < 0.001


# ---------------------------------------------------------------------------
# Routing through call_llm / acall_llm
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRouting:
    async def test_non_agent_with_mcp_routes_to_loop(self) -> None:
        """Non-agent model + mcp_servers → MCP agent loop."""
        with patch("llm_client.mcp_agent._acall_with_mcp") as mock_loop:
            mock_loop.return_value = _make_llm_result(content="answer")

            result = await acall_llm(
                "gemini/gemini-3-flash-preview",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
                max_turns=5,
            )

            mock_loop.assert_called_once()
            call_kwargs = mock_loop.call_args
            assert call_kwargs.kwargs.get("mcp_servers") == {
                "srv": {"command": "python", "args": ["s.py"]},
            }
            assert call_kwargs.kwargs.get("max_turns") == 5

    async def test_agent_model_with_mcp_skips_loop(self) -> None:
        """Agent model + mcp_servers → existing agent SDK path (not MCP loop)."""
        with (
            patch("llm_client.mcp_agent._acall_with_mcp") as mock_loop,
            patch("llm_client.agents._route_acall") as mock_route,
        ):
            mock_route.return_value = _make_llm_result(content="agent answer")

            result = await acall_llm(
                "codex",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
            )

            mock_loop.assert_not_called()
            mock_route.assert_called_once()

    async def test_no_mcp_servers_normal_routing(self) -> None:
        """No mcp_servers → normal litellm routing."""
        with (
            patch("llm_client.mcp_agent._acall_with_mcp") as mock_loop,
            patch("litellm.acompletion") as mock_completion,
        ):
            mock_msg = MagicMock()
            mock_msg.content = "hello"
            mock_msg.tool_calls = None
            mock_choice = MagicMock()
            mock_choice.message = mock_msg
            mock_choice.finish_reason = "stop"
            mock_resp = MagicMock()
            mock_resp.choices = [mock_choice]
            mock_resp.usage = MagicMock(
                prompt_tokens=10, completion_tokens=5, total_tokens=15,
            )
            mock_resp.model = "gemini/gemini-3-flash-preview"
            mock_completion.return_value = mock_resp

            result = await acall_llm(
                "gemini/gemini-3-flash-preview",
                [{"role": "user", "content": "Q"}],
            )

            mock_loop.assert_not_called()
            mock_completion.assert_called_once()

    def test_sync_call_llm_with_mcp(self) -> None:
        """Sync call_llm with mcp_servers routes to MCP loop."""
        with patch("llm_client.mcp_agent._acall_with_mcp") as mock_loop:
            mock_loop.return_value = _make_llm_result(content="sync answer")

            result = call_llm(
                "gemini/gemini-3-flash-preview",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
            )

            # _run_sync wraps the async call
            mock_loop.assert_called_once()

    async def test_mcp_kwargs_popped_from_inner_calls(self) -> None:
        """MCP-specific kwargs don't leak to inner acall_llm."""
        with patch("llm_client.mcp_agent._acall_with_mcp") as mock_loop:
            mock_loop.return_value = _make_llm_result(content="answer")

            await acall_llm(
                "gemini/gemini-3-flash-preview",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
                max_turns=10,
                mcp_init_timeout=60.0,
                tool_result_max_length=10000,
                temperature=0.5,  # regular litellm kwarg
            )

            call_kwargs = mock_loop.call_args.kwargs
            # MCP kwargs present
            assert call_kwargs["mcp_servers"] == {"srv": {"command": "python", "args": ["s.py"]}}
            assert call_kwargs["max_turns"] == 10
            assert call_kwargs["mcp_init_timeout"] == 60.0
            assert call_kwargs["tool_result_max_length"] == 10000
            # Regular kwargs also present (passed through)
            assert call_kwargs["temperature"] == 0.5
