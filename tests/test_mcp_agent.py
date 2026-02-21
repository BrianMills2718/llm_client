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
    DEFAULT_MAX_TOOL_CALLS,
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
        assert result["type"] == "function"
        assert result["function"]["name"] == "search"
        assert result["function"]["description"] == "Search for entities"
        required = result["function"]["parameters"]["required"]
        assert "query" in required
        assert "tool_reasoning" in required
        props = result["function"]["parameters"]["properties"]
        assert props["query"] == {"type": "string"}
        assert "tool_reasoning" in props

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
        params = result["function"]["parameters"]
        assert params["type"] == "object"
        assert "tool_reasoning" in params["properties"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestExtractUsage:
    def test_openai_convention(self) -> None:
        inp, out, cached, cache_create = _extract_usage({"prompt_tokens": 100, "completion_tokens": 50})
        assert inp == 100
        assert out == 50
        assert cached == 0
        assert cache_create == 0

    def test_anthropic_convention(self) -> None:
        inp, out, cached, cache_create = _extract_usage({"input_tokens": 200, "output_tokens": 75})
        assert inp == 200
        assert out == 75

    def test_empty_usage(self) -> None:
        inp, out, cached, cache_create = _extract_usage({})
        assert inp == 0
        assert out == 0
        assert cached == 0
        assert cache_create == 0

    def test_input_tokens_takes_priority(self) -> None:
        """input_tokens is checked before prompt_tokens."""
        inp, out, cached, cache_create = _extract_usage({
            "input_tokens": 300,
            "prompt_tokens": 100,
            "output_tokens": 50,
        })
        assert inp == 300

    def test_cached_tokens(self) -> None:
        """Provider-level prompt caching fields are extracted."""
        inp, out, cached, cache_create = _extract_usage({
            "input_tokens": 500,
            "output_tokens": 100,
            "cached_tokens": 400,
            "cache_creation_tokens": 50,
        })
        assert inp == 500
        assert out == 100
        assert cached == 400
        assert cache_create == 50


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
    def test_default_max_tool_calls(self) -> None:
        assert DEFAULT_MAX_TOOL_CALLS == 20

    def test_default_max_turns(self) -> None:
        assert DEFAULT_MAX_TURNS == 20

    def test_default_init_timeout(self) -> None:
        assert DEFAULT_MCP_INIT_TIMEOUT == 30.0

    def test_default_tool_result_max_length(self) -> None:
        assert DEFAULT_TOOL_RESULT_MAX_LENGTH == 50_000

    def test_mcp_loop_kwargs(self) -> None:
        assert "mcp_servers" in MCP_LOOP_KWARGS
        assert "max_turns" in MCP_LOOP_KWARGS
        assert "max_tool_calls" in MCP_LOOP_KWARGS
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
    model: str = "test-model",
    warnings: list[str] | None = None,
) -> LLMCallResult:
    return LLMCallResult(
        content=content,
        usage=usage or {"input_tokens": 100, "output_tokens": 50},
        cost=cost,
        model=model,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        warnings=warnings or [],
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
            # No tool calls → trace has just the final assistant message
            trace = result.raw_response.conversation_trace
            non_budget_trace = [
                msg for msg in trace
                if "budget:" not in str(msg.get("content", "")).lower()
            ]
            assert len(non_budget_trace) == 1
            assert non_budget_trace[0]["role"] == "assistant"
            assert non_budget_trace[0]["content"] == "Paris"

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

            # Verify conversation trace captures all messages
            trace = agent_result.conversation_trace
            non_budget_trace = [
                msg for msg in trace
                if ("budget:" not in str(msg.get("content", "")).lower())
                and ("Observability requirement" not in str(msg.get("content", "")))
            ]
            assert len(non_budget_trace) == 3  # assistant(tool_call) + tool_result + assistant(answer)
            assert non_budget_trace[0]["role"] == "assistant"
            assert len(non_budget_trace[0]["tool_calls"]) == 1
            assert non_budget_trace[0]["tool_calls"][0]["name"] == "search"
            assert non_budget_trace[1]["role"] == "tool"
            assert non_budget_trace[2]["role"] == "assistant"
            assert non_budget_trace[2]["content"] == "Paris"

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

    async def test_max_tool_calls_exhausted(self) -> None:
        """Loop stops when max_tool_calls is exhausted and forces final answer."""
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

            tool_call_result = _make_llm_result(tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": "{}"},
            }])
            final_result = _make_llm_result(content="forced by tool budget")
            mock_acall.side_effect = [tool_call_result, final_result]

            from llm_client.mcp_agent import _acall_with_mcp
            result = await _acall_with_mcp(
                "test-model",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
                max_turns=10,
                max_tool_calls=1,
            )

            assert result.content == "forced by tool budget"
            assert result.raw_response.turns == 2  # 1 loop + 1 forced final
            assert len(result.raw_response.tool_calls) == 1

    async def test_max_tool_calls_ignores_todo_tools(self) -> None:
        """todo_* calls do not consume max_tool_calls budget."""
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(
            tools=[_make_tool("todo_update"), _make_tool("search")],
        ))
        mock_session.call_tool = AsyncMock(
            side_effect=[_make_tool_result("todo-ok"), _make_tool_result("search-ok")],
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
                    "function": {"name": "todo_update", "arguments": "{}"},
                }]),
                _make_llm_result(tool_calls=[{
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }]),
                _make_llm_result(content="forced by retrieval budget"),
            ]

            from llm_client.mcp_agent import _acall_with_mcp
            result = await _acall_with_mcp(
                "test-model",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
                max_turns=10,
                max_tool_calls=1,
            )

            assert result.content == "forced by retrieval budget"
            assert result.raw_response.turns == 3  # 2 loop turns + 1 forced final
            assert len(result.raw_response.tool_calls) == 2
            assert result.raw_response.metadata["budgeted_tool_calls_used"] == 1
            assert mock_session.call_tool.call_count == 2

    async def test_require_tool_reasoning_rejects_mcp_call(self) -> None:
        """Strict mode rejects tool calls that omit tool_reasoning."""
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(
            tools=[_make_tool("search")],
        ))
        mock_session.call_tool = AsyncMock(return_value=_make_tool_result("result"))

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
                _make_llm_result(content="answer after rejection"),
            ]

            from llm_client.mcp_agent import _acall_with_mcp
            result = await _acall_with_mcp(
                "test-model",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
                require_tool_reasoning=True,
            )

            assert result.content == "answer after rejection"
            # Invalid call is rejected pre-execution and not recorded as executed tool call
            assert len(result.raw_response.tool_calls) == 0
            assert result.raw_response.metadata["rejected_missing_reasoning_calls"] == 1
            assert any("missing tool_reasoning" in w.lower() for w in (result.warnings or []))
            # Synthetic tool rejection is still visible in trace for observability
            assert any(
                "Missing required argument: tool_reasoning" in str(msg.get("content", ""))
                for msg in result.raw_response.conversation_trace
                if msg.get("role") == "tool"
            )
            mock_session.call_tool.assert_not_called()

    async def test_enforce_tool_contracts_rejects_incompatible_call(self) -> None:
        """Contract mode rejects tool calls when required artifacts are unavailable."""
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(
            tools=[_make_tool("search")],
        ))
        mock_session.call_tool = AsyncMock(return_value=_make_tool_result("result"))

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
                    "function": {
                        "name": "search",
                        "arguments": '{"tool_reasoning":"test"}',
                    },
                }]),
                _make_llm_result(content="answer after contract rejection"),
            ]

            from llm_client.mcp_agent import _acall_with_mcp
            result = await _acall_with_mcp(
                "test-model",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
                enforce_tool_contracts=True,
                tool_contracts={
                    "search": {
                        "requires_all": ["ENTITY_SET"],
                        "produces": ["CHUNK_SET"],
                    },
                },
                initial_artifacts=("QUERY_TEXT",),
            )

            assert result.content == "answer after contract rejection"
            assert result.raw_response.metadata["tool_contract_rejections"] == 1
            assert result.raw_response.metadata["available_artifacts_final"] == ["QUERY_TEXT"]
            assert any(
                "Tool contract violation" in str(msg.get("content", ""))
                for msg in result.raw_response.conversation_trace
                if msg.get("role") == "tool"
            )
            mock_session.call_tool.assert_not_called()

    async def test_enforce_tool_contracts_propagates_artifacts(self) -> None:
        """Successful contract-validated calls grow available artifacts."""
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(
            tools=[_make_tool("entity_tfidf"), _make_tool("entity_onehop")],
        ))
        mock_session.call_tool = AsyncMock(
            side_effect=[_make_tool_result("entity hit"), _make_tool_result("onehop hit")],
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
                    "function": {
                        "name": "entity_tfidf",
                        "arguments": '{"tool_reasoning":"seed entities from query"}',
                    },
                }]),
                _make_llm_result(tool_calls=[{
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "entity_onehop",
                        "arguments": '{"tool_reasoning":"expand entity neighborhood"}',
                    },
                }]),
                _make_llm_result(content="answer"),
            ]

            from llm_client.mcp_agent import _acall_with_mcp
            result = await _acall_with_mcp(
                "test-model",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
                enforce_tool_contracts=True,
                tool_contracts={
                    "entity_tfidf": {
                        "requires_all": ["QUERY_TEXT"],
                        "produces": ["ENTITY_SET"],
                    },
                    "entity_onehop": {
                        "requires_all": ["ENTITY_SET"],
                        "produces": ["ENTITY_SET"],
                    },
                },
                initial_artifacts=("QUERY_TEXT",),
            )

            assert result.content == "answer"
            assert mock_session.call_tool.call_count == 2
            assert result.raw_response.metadata["tool_contract_rejections"] == 0
            assert "ENTITY_SET" in result.raw_response.metadata["available_artifacts_final"]

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
                max_tool_calls=7,
                task="test",
                trace_id="test_non_agent_mcp_routing",
                max_budget=0,
            )

            mock_loop.assert_called_once()
            call_kwargs = mock_loop.call_args
            assert call_kwargs.kwargs.get("mcp_servers") == {
                "srv": {"command": "python", "args": ["s.py"]},
            }
            assert call_kwargs.kwargs.get("max_turns") == 5
            assert call_kwargs.kwargs.get("max_tool_calls") == 7

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
                task="test",
                trace_id="test_agent_model_skips_loop",
                max_budget=0,
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
                task="test",
                trace_id="test_no_mcp_normal_routing",
                max_budget=0,
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
                task="test",
                trace_id="test_sync_call_llm_with_mcp",
                max_budget=0,
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
                max_tool_calls=15,
                mcp_init_timeout=60.0,
                tool_result_max_length=10000,
                temperature=0.5,  # regular litellm kwarg
                task="test",
                trace_id="test_mcp_kwargs_popped",
                max_budget=0,
            )

            call_kwargs = mock_loop.call_args.kwargs
            # MCP kwargs present
            assert call_kwargs["mcp_servers"] == {"srv": {"command": "python", "args": ["s.py"]}}
            assert call_kwargs["max_turns"] == 10
            assert call_kwargs["max_tool_calls"] == 15
            assert call_kwargs["mcp_init_timeout"] == 60.0
            assert call_kwargs["tool_result_max_length"] == 10000
            # Regular kwargs also present (passed through)
            assert call_kwargs["temperature"] == 0.5


# ---------------------------------------------------------------------------
# Warnings, sticky fallback, models_used
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAgentDiagnostics:
    """Tests for warnings, models_used, and sticky fallback in agent loop."""

    async def test_sticky_fallback(self) -> None:
        """When inner call returns a different model (fallback), remaining turns use it."""
        from llm_client.mcp_agent import MCPAgentResult, _agent_loop

        call_count = 0

        async def mock_executor(
            tool_calls: list[dict[str, Any]], max_len: int,
        ) -> tuple[list, list]:
            records = [MCPToolCallRecord(server="s", tool="t", arguments={}, result="ok")]
            msgs = [{"role": "tool", "tool_call_id": "tc1", "content": "result"}]
            return records, msgs

        async def mock_inner_acall(model, messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: tool call, but model fell back
                return _make_llm_result(
                    content="",
                    model="fallback-model",
                    tool_calls=[{
                        "id": "tc1",
                        "function": {"name": "t", "arguments": "{}"},
                    }],
                    finish_reason="tool_calls",
                    warnings=["FALLBACK: test-model -> fallback-model (Exception: error)"],
                )
            else:
                # Second call: final answer with sticky model
                return _make_llm_result(content="answer", model="fallback-model")

        agent_result = MCPAgentResult()

        with patch("llm_client.mcp_agent._inner_acall_llm", side_effect=mock_inner_acall):
            content, finish = await _agent_loop(
                "test-model",
                [{"role": "user", "content": "q"}],
                [{"type": "function", "function": {"name": "t"}}],
                agent_result,
                mock_executor,
                5,
                None,
                False,
                50000,
                max_message_chars=60,
                timeout=60,
                kwargs={},
            )

        assert content == "answer"
        assert "fallback-model" in agent_result.models_used
        assert any("STICKY_FALLBACK" in w for w in agent_result.warnings)
        # Verify inner calls used the sticky model on turn 2
        assert call_count == 2

    async def test_warnings_propagated_from_turns(self) -> None:
        """Per-turn warnings from inner acall_llm accumulate in MCPAgentResult."""
        from llm_client.mcp_agent import MCPAgentResult, _agent_loop

        async def mock_executor(tc, ml):
            records = [MCPToolCallRecord(server="s", tool="t", arguments={}, result="ok")]
            msgs = [{"role": "tool", "tool_call_id": "tc1", "content": "result"}]
            return records, msgs

        results = [
            _make_llm_result(
                content="",
                tool_calls=[{"id": "tc1", "function": {"name": "t", "arguments": "{}"}}],
                finish_reason="tool_calls",
                warnings=["RETRY 1/3: test-model (Exception: rate limit)"],
            ),
            _make_llm_result(content="done", warnings=["RETRY 1/3: test-model (Exception: timeout)"]),
        ]

        agent_result = MCPAgentResult()

        with patch("llm_client.mcp_agent._inner_acall_llm", side_effect=results):
            await _agent_loop(
                "test-model",
                [{"role": "user", "content": "q"}],
                [{"type": "function", "function": {"name": "t"}}],
                agent_result,
                mock_executor,
                5,
                None,
                False,
                50000,
                max_message_chars=60,
                timeout=60,
                kwargs={},
            )

        assert len(agent_result.warnings) >= 2
        assert any("rate limit" in w for w in agent_result.warnings)
        assert any("timeout" in w for w in agent_result.warnings)

    async def test_models_used_tracked(self) -> None:
        """models_used set tracks all models that responded."""
        from llm_client.mcp_agent import MCPAgentResult, _agent_loop

        result = _make_llm_result(content="ok", model="gemini/gemini-2.5-flash")
        agent_result = MCPAgentResult()

        with patch("llm_client.mcp_agent._inner_acall_llm", return_value=result):
            await _agent_loop(
                "gemini/gemini-2.5-flash",
                [{"role": "user", "content": "q"}],
                [{"type": "function", "function": {"name": "t"}}],
                agent_result,
                AsyncMock(),
                5,
                None,
                False,
                50000,
                max_message_chars=60,
                timeout=60,
                kwargs={},
            )

        assert "gemini/gemini-2.5-flash" in agent_result.models_used

    async def test_submit_answer_enforced_when_tool_available(self) -> None:
        """When submit_answer exists, plain-text response is nudged into explicit submission."""
        from llm_client.mcp_agent import MCPAgentResult, _agent_loop

        llm_results = [
            _make_llm_result(content="June 1982"),
            _make_llm_result(
                content="",
                tool_calls=[{
                    "id": "tc_submit",
                    "function": {
                        "name": "submit_answer",
                        "arguments": '{"reasoning":"from chunk_1","answer":"June 1982"}',
                    },
                }],
                finish_reason="tool_calls",
            ),
        ]

        async def mock_executor(tc, ml):
            return (
                [
                    MCPToolCallRecord(
                        server="srv",
                        tool="submit_answer",
                        arguments={"reasoning": "from chunk_1", "answer": "June 1982"},
                        result='{"status":"submitted","answer":"June 1982"}',
                    ),
                ],
                [{
                    "role": "tool",
                    "tool_call_id": "tc_submit",
                    "content": '{"status":"submitted","answer":"June 1982"}',
                }],
            )

        agent_result = MCPAgentResult()
        with patch("llm_client.mcp_agent._inner_acall_llm", side_effect=llm_results):
            content, finish = await _agent_loop(
                "test-model",
                [{"role": "user", "content": "q"}],
                [{"type": "function", "function": {"name": "submit_answer"}}],
                agent_result,
                mock_executor,
                5,
                None,
                False,
                50000,
                max_message_chars=60,
                timeout=60,
                kwargs={},
            )

        assert content == "June 1982"
        assert finish == "submitted"
        assert any(
            msg.get("role") == "user" and "MUST call submit_answer" in msg.get("content", "")
            for msg in agent_result.conversation_trace
        )

    async def test_autofill_reasoning_for_todo_reset(self) -> None:
        """todo_reset missing tool_reasoning is auto-filled and executed in strict mode."""
        from llm_client.mcp_agent import MCPAgentResult, _agent_loop

        observed_tool_calls: list[dict[str, Any]] = []

        async def mock_executor(tool_calls, max_len):
            observed_tool_calls.extend(tool_calls)
            return (
                [MCPToolCallRecord(server="srv", tool="todo_reset", arguments={}, result='{"status":"reset"}')],
                [{"role": "tool", "tool_call_id": "tc_reset", "content": '{"status":"reset"}'}],
            )

        llm_results = [
            _make_llm_result(
                content="",
                tool_calls=[{
                    "id": "tc_reset",
                    "function": {"name": "todo_reset", "arguments": "{}"},
                }],
                finish_reason="tool_calls",
            ),
            _make_llm_result(content="done"),
        ]

        agent_result = MCPAgentResult()
        with patch("llm_client.mcp_agent._inner_acall_llm", side_effect=llm_results):
            content, finish = await _agent_loop(
                "test-model",
                [{"role": "user", "content": "q"}],
                [{"type": "function", "function": {"name": "todo_reset"}}],
                agent_result,
                mock_executor,
                3,
                None,
                True,
                50000,
                max_message_chars=60,
                timeout=60,
                kwargs={},
            )

        assert content == "done"
        assert finish == "stop"
        assert observed_tool_calls
        args = json.loads(observed_tool_calls[0]["function"]["arguments"])
        assert "tool_reasoning" in args
        assert agent_result.metadata["rejected_missing_reasoning_calls"] == 0
        assert not any(
            "observability: missing tool_reasoning on tools:" in w.lower()
            for w in agent_result.warnings
        )

    async def test_submit_answer_suppressed_until_todo_progress(self) -> None:
        """Repeated submit_answer calls are suppressed until TODO state changes."""
        from llm_client.mcp_agent import MCPAgentResult, _agent_loop

        executor_call_count = 0

        async def mock_executor(tool_calls, max_len):
            nonlocal executor_call_count
            executor_call_count += 1
            return (
                [
                    MCPToolCallRecord(
                        server="srv",
                        tool="submit_answer",
                        arguments={"reasoning": "r", "answer": "a"},
                        error="Cannot submit yet. Unfinished TODOs: todo_1.",
                    ),
                ],
                [
                    {
                        "role": "tool",
                        "tool_call_id": "tc_submit_1",
                        "content": '{"error":"Cannot submit yet. Unfinished TODOs: todo_1."}',
                    }
                ],
            )

        llm_results = [
            _make_llm_result(
                content="",
                tool_calls=[{
                    "id": "tc_submit_1",
                    "function": {
                        "name": "submit_answer",
                        "arguments": '{"reasoning":"r","answer":"a","tool_reasoning":"submit now"}',
                    },
                }],
                finish_reason="tool_calls",
            ),
            _make_llm_result(
                content="",
                tool_calls=[{
                    "id": "tc_submit_2",
                    "function": {
                        "name": "submit_answer",
                        "arguments": '{"reasoning":"r","answer":"a","tool_reasoning":"submit again"}',
                    },
                }],
                finish_reason="tool_calls",
            ),
            _make_llm_result(content="final"),
        ]

        agent_result = MCPAgentResult()
        with patch("llm_client.mcp_agent._inner_acall_llm", side_effect=llm_results):
            content, finish = await _agent_loop(
                "test-model",
                [{"role": "user", "content": "q"}],
                [{"type": "function", "function": {"name": "submit_answer"}}],
                agent_result,
                mock_executor,
                3,
                None,
                False,
                50000,
                max_message_chars=60,
                timeout=60,
                kwargs={},
            )

        assert content == "final"
        assert finish == "stop"
        # First submit call executes; second is suppressed before executor.
        assert executor_call_count == 1
        assert agent_result.metadata["control_loop_suppressed_calls"] >= 1
        assert any(
            "submit_answer suppressed" in (record.error or "")
            for record in agent_result.tool_calls
            if record.tool == "submit_answer"
        )
