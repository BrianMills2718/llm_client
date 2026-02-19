"""Tests for structured-output tool-calling shim.

Tests cover:
- Single tool call → final answer (2 turns)
- Multi-turn tool calls → final answer
- Immediate final answer (no tool calls)
- Malformed JSON recovery
- Unknown tool name → error → model corrects
- Max turns exhausted
- Routing: flash-lite model routes to shim
- Routing: normal model uses native _acall_with_tools
"""

# mock-ok: testing shim routing/parsing, not LLM quality

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from llm_client import LLMCallResult, MCPAgentResult
from llm_client.tool_shim import _acall_with_tool_shim, _build_tool_system_prompt
from llm_client.tool_utils import prepare_direct_tools


# ---------------------------------------------------------------------------
# Test tools
# ---------------------------------------------------------------------------


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


async def search(query: str, limit: int = 10) -> str:
    """Search for entities."""
    return f"Results for {query} (limit={limit})"


def failing_tool(x: str) -> str:
    """Always fails."""
    raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm_result(
    content: str = "",
    usage: dict[str, Any] | None = None,
    cost: float = 0.001,
    finish_reason: str = "stop",
) -> LLMCallResult:
    return LLMCallResult(
        content=content,
        usage=usage or {"input_tokens": 100, "output_tokens": 50},
        cost=cost,
        model="test-model",
        tool_calls=[],
        finish_reason=finish_reason,
    )


# ---------------------------------------------------------------------------
# System prompt generation
# ---------------------------------------------------------------------------


class TestBuildToolSystemPrompt:
    def test_includes_tool_names(self) -> None:
        _, openai_tools = prepare_direct_tools([add, search])
        prompt = _build_tool_system_prompt(openai_tools)
        assert "add(" in prompt
        assert "search(" in prompt
        assert "final_answer" in prompt
        assert "tool_call" in prompt

    def test_includes_parameter_schema(self) -> None:
        _, openai_tools = prepare_direct_tools([add])
        prompt = _build_tool_system_prompt(openai_tools)
        assert '"a"' in prompt
        assert '"b"' in prompt
        assert "integer" in prompt


# ---------------------------------------------------------------------------
# Shim loop tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestToolShimLoop:
    async def test_single_tool_call_and_answer(self) -> None:
        """tool_call → final_answer in 2 turns."""
        with patch("llm_client.tool_shim._inner_acall_llm") as mock_acall:
            mock_acall.side_effect = [
                # Turn 1: call add(3, 4)
                _make_llm_result(content=json.dumps({
                    "action": "tool_call",
                    "tool_name": "add",
                    "arguments": {"a": 3, "b": 4},
                })),
                # Turn 2: final answer
                _make_llm_result(content=json.dumps({
                    "action": "final_answer",
                    "content": "The sum is 7",
                })),
            ]

            result = await _acall_with_tool_shim(
                "test-model",
                [{"role": "user", "content": "What is 3+4?"}],
                [add],
                task="test", trace_id="test_shim_1", max_budget=0,
            )

            assert result.content == "The sum is 7"
            assert result.finish_reason == "stop"
            agent: MCPAgentResult = result.raw_response
            assert agent.turns == 2
            assert len(agent.tool_calls) == 1
            assert agent.tool_calls[0].tool == "add"
            assert agent.tool_calls[0].result == "7"

    async def test_multi_turn_tools(self) -> None:
        """3 tool calls → final answer."""
        with patch("llm_client.tool_shim._inner_acall_llm") as mock_acall:
            mock_acall.side_effect = [
                _make_llm_result(content=json.dumps({
                    "action": "tool_call",
                    "tool_name": "add",
                    "arguments": {"a": 1, "b": 2},
                })),
                _make_llm_result(content=json.dumps({
                    "action": "tool_call",
                    "tool_name": "search",
                    "arguments": {"query": "test"},
                })),
                _make_llm_result(content=json.dumps({
                    "action": "tool_call",
                    "tool_name": "add",
                    "arguments": {"a": 10, "b": 20},
                })),
                _make_llm_result(content=json.dumps({
                    "action": "final_answer",
                    "content": "Done: 3, results, 30",
                })),
            ]

            result = await _acall_with_tool_shim(
                "test-model",
                [{"role": "user", "content": "Do stuff"}],
                [add, search],
                task="test", trace_id="test_multi", max_budget=0,
            )

            assert result.content == "Done: 3, results, 30"
            agent: MCPAgentResult = result.raw_response
            assert agent.turns == 4
            assert len(agent.tool_calls) == 3
            assert agent.tool_calls[1].tool == "search"

    async def test_final_answer_immediately(self) -> None:
        """Model answers without calling any tools."""
        with patch("llm_client.tool_shim._inner_acall_llm") as mock_acall:
            mock_acall.return_value = _make_llm_result(content=json.dumps({
                "action": "final_answer",
                "content": "42",
            }))

            result = await _acall_with_tool_shim(
                "test-model",
                [{"role": "user", "content": "What is the answer?"}],
                [add],
                task="test", trace_id="test_immediate", max_budget=0,
            )

            assert result.content == "42"
            agent: MCPAgentResult = result.raw_response
            assert agent.turns == 1
            assert len(agent.tool_calls) == 0

    async def test_malformed_json_recovery(self) -> None:
        """Bad JSON → error message → model retries successfully."""
        with patch("llm_client.tool_shim._inner_acall_llm") as mock_acall:
            mock_acall.side_effect = [
                # Turn 1: invalid JSON
                _make_llm_result(content="not json at all"),
                # Turn 2: model corrects itself
                _make_llm_result(content=json.dumps({
                    "action": "final_answer",
                    "content": "recovered",
                })),
            ]

            result = await _acall_with_tool_shim(
                "test-model",
                [{"role": "user", "content": "test"}],
                [add],
                task="test", trace_id="test_malformed", max_budget=0,
            )

            assert result.content == "recovered"
            # Verify the error message was injected
            calls = mock_acall.call_args_list
            second_call_msgs = calls[1][0][1]  # messages arg of 2nd call
            assert any("not valid JSON" in m.get("content", "") for m in second_call_msgs)

    async def test_unknown_tool_name(self) -> None:
        """Model calls nonexistent tool → error → model corrects."""
        with patch("llm_client.tool_shim._inner_acall_llm") as mock_acall:
            mock_acall.side_effect = [
                # Turn 1: call nonexistent tool
                _make_llm_result(content=json.dumps({
                    "action": "tool_call",
                    "tool_name": "nonexistent",
                    "arguments": {},
                })),
                # Turn 2: model gives up and answers
                _make_llm_result(content=json.dumps({
                    "action": "final_answer",
                    "content": "sorry, used wrong tool",
                })),
            ]

            result = await _acall_with_tool_shim(
                "test-model",
                [{"role": "user", "content": "test"}],
                [add],
                task="test", trace_id="test_unknown_tool", max_budget=0,
            )

            assert result.content == "sorry, used wrong tool"
            agent: MCPAgentResult = result.raw_response
            assert len(agent.tool_calls) == 1
            assert agent.tool_calls[0].error is not None
            assert "Unknown tool" in agent.tool_calls[0].error

    async def test_max_turns_exhausted(self) -> None:
        """Loop terminates and forces answer when max_turns reached."""
        with patch("llm_client.tool_shim._inner_acall_llm") as mock_acall:
            # All turns return tool calls, then the forced final answer
            tool_response = _make_llm_result(content=json.dumps({
                "action": "tool_call",
                "tool_name": "add",
                "arguments": {"a": 1, "b": 1},
            }))
            forced_answer = _make_llm_result(content="forced final")
            mock_acall.side_effect = [tool_response, tool_response, forced_answer]

            result = await _acall_with_tool_shim(
                "test-model",
                [{"role": "user", "content": "keep going"}],
                [add],
                max_turns=2,
                task="test", trace_id="test_max_turns", max_budget=0,
            )

            assert result.content == "forced final"
            agent: MCPAgentResult = result.raw_response
            assert agent.turns == 3  # 2 tool turns + 1 forced

    async def test_unknown_action(self) -> None:
        """Unknown action → error nudge → model corrects."""
        with patch("llm_client.tool_shim._inner_acall_llm") as mock_acall:
            mock_acall.side_effect = [
                _make_llm_result(content=json.dumps({
                    "action": "think",
                    "thought": "hmm",
                })),
                _make_llm_result(content=json.dumps({
                    "action": "final_answer",
                    "content": "got it",
                })),
            ]

            result = await _acall_with_tool_shim(
                "test-model",
                [{"role": "user", "content": "test"}],
                [add],
                task="test", trace_id="test_unknown_action", max_budget=0,
            )

            assert result.content == "got it"

    async def test_system_message_preserved(self) -> None:
        """Tool prompt is appended to existing system message."""
        with patch("llm_client.tool_shim._inner_acall_llm") as mock_acall:
            mock_acall.return_value = _make_llm_result(content=json.dumps({
                "action": "final_answer",
                "content": "ok",
            }))

            await _acall_with_tool_shim(
                "test-model",
                [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "hi"},
                ],
                [add],
                task="test", trace_id="test_sysmsg", max_budget=0,
            )

            call_msgs = mock_acall.call_args[0][1]
            assert call_msgs[0]["role"] == "system"
            assert "You are helpful." in call_msgs[0]["content"]
            assert "tool_call" in call_msgs[0]["content"]

    async def test_usage_accumulates(self) -> None:
        """Usage tokens accumulate across turns."""
        with patch("llm_client.tool_shim._inner_acall_llm") as mock_acall:
            mock_acall.side_effect = [
                _make_llm_result(
                    content=json.dumps({
                        "action": "tool_call",
                        "tool_name": "add",
                        "arguments": {"a": 1, "b": 2},
                    }),
                    usage={"input_tokens": 100, "output_tokens": 20},
                    cost=0.01,
                ),
                _make_llm_result(
                    content=json.dumps({
                        "action": "final_answer",
                        "content": "3",
                    }),
                    usage={"input_tokens": 200, "output_tokens": 30},
                    cost=0.02,
                ),
            ]

            result = await _acall_with_tool_shim(
                "test-model",
                [{"role": "user", "content": "1+2"}],
                [add],
                task="test", trace_id="test_usage", max_budget=0,
            )

            assert result.usage["input_tokens"] == 300
            assert result.usage["output_tokens"] == 50
            assert abs(result.cost - 0.03) < 0.001

    async def test_tool_error_visible_to_model(self) -> None:
        """Tool execution error is sent back to the model as user message."""
        with patch("llm_client.tool_shim._inner_acall_llm") as mock_acall:
            mock_acall.side_effect = [
                _make_llm_result(content=json.dumps({
                    "action": "tool_call",
                    "tool_name": "failing_tool",
                    "arguments": {"x": "test"},
                })),
                _make_llm_result(content=json.dumps({
                    "action": "final_answer",
                    "content": "tool failed",
                })),
            ]

            result = await _acall_with_tool_shim(
                "test-model",
                [{"role": "user", "content": "test"}],
                [failing_tool],
                task="test", trace_id="test_tool_error", max_budget=0,
            )

            assert result.content == "tool failed"
            # Check the error was injected as a user message
            calls = mock_acall.call_args_list
            second_msgs = calls[1][0][1]
            error_msgs = [m for m in second_msgs if "ERROR" in m.get("content", "")]
            assert len(error_msgs) == 1
            assert "RuntimeError: boom" in error_msgs[0]["content"]


# ---------------------------------------------------------------------------
# Routing tests (via acall_llm / call_llm)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestShimRouting:
    async def test_routing_flash_lite(self) -> None:
        """acall_llm with python_tools + flash-lite model routes to shim."""
        with (
            patch("llm_client.tool_shim._inner_acall_llm") as mock_shim_inner,
            patch("llm_client.client._check_budget"),
            patch("llm_client.client._io_log"),
        ):
            mock_shim_inner.return_value = _make_llm_result(content=json.dumps({
                "action": "final_answer",
                "content": "shim answer",
            }))

            from llm_client import acall_llm
            result = await acall_llm(
                "gemini/gemini-2.5-flash-lite",
                [{"role": "user", "content": "test"}],
                python_tools=[add],
                task="test", trace_id="test_route_shim", max_budget=0,
            )

            assert result.content == "shim answer"
            # Verify _inner_acall_llm was called (shim path), NOT with tools= kwarg
            assert mock_shim_inner.called
            _, call_kwargs = mock_shim_inner.call_args
            assert "tools" not in call_kwargs

    async def test_routing_normal_model(self) -> None:
        """Normal model with python_tools uses _acall_with_tools, not shim."""
        with (
            patch("llm_client.mcp_agent._inner_acall_llm") as mock_native_inner,
            patch("llm_client.client._check_budget"),
            patch("llm_client.client._io_log"),
        ):
            mock_native_inner.return_value = _make_llm_result(content="native answer")

            from llm_client import acall_llm
            result = await acall_llm(
                "gemini/gemini-2.5-flash",
                [{"role": "user", "content": "test"}],
                python_tools=[add],
                task="test", trace_id="test_route_native", max_budget=0,
            )

            assert result.content == "native answer"
            # Verify native path was used — it passes tools= kwarg
            assert mock_native_inner.called
            _, call_kwargs = mock_native_inner.call_args
            assert "tools" in call_kwargs
