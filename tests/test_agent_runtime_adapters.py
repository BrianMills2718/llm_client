"""Tests for explicit optional-runtime adapters.

These tests prove that core runtime code can call the named adapter functions
without depending on private MCP runtime entrypoints directly, while existing
private-function patches remain effective because the adapters delegate at call
time.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from llm_client import LLMCallResult
from llm_client.agent.agent_planning import PlanningConfig
from llm_client.agent.mcp_agent import (
    acall_with_mcp_runtime,
    acall_with_python_tools_runtime,
)


def _make_result() -> LLMCallResult:
    """Build a minimal loop result for adapter delegation tests."""
    return LLMCallResult(
        content="ok",
        usage={"input_tokens": 1, "output_tokens": 1},
        cost=0.0,
        model="test-model",
    )


@pytest.mark.asyncio
async def test_acall_with_mcp_runtime_delegates_to_private_loop() -> None:
    """The MCP adapter should call the existing private loop implementation."""
    planning_config = PlanningConfig(enabled=True)
    with patch("llm_client.agent.mcp_agent._acall_with_mcp", new_callable=AsyncMock) as mock_loop:
        mock_loop.return_value = _make_result()
        result = await acall_with_mcp_runtime(
            "gemini/gemini-2.5-flash",
            [{"role": "user", "content": "hi"}],
            mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
            planning_config=planning_config,
            timeout=30,
            task="test",
            trace_id="trace",
            max_budget=1.0,
        )

    assert result.content == "ok"
    _, kwargs = mock_loop.call_args
    assert kwargs["mcp_servers"] == {"srv": {"command": "python", "args": ["s.py"]}}
    assert kwargs["planning_config"] is planning_config
    assert kwargs["timeout"] == 30


@pytest.mark.asyncio
async def test_acall_with_python_tools_runtime_delegates_to_private_loop() -> None:
    """The direct-tool adapter should call the existing private tool loop."""
    tool = object()
    planning_config = PlanningConfig(enabled=True)
    with patch("llm_client.agent.mcp_agent._acall_with_tools", new_callable=AsyncMock) as mock_loop:
        mock_loop.return_value = _make_result()
        result = await acall_with_python_tools_runtime(
            "gemini/gemini-2.5-flash",
            [{"role": "user", "content": "hi"}],
            python_tools=[tool],
            planning_config=planning_config,
            timeout=30,
            task="test",
            trace_id="trace",
            max_budget=1.0,
        )

    assert result.content == "ok"
    _, kwargs = mock_loop.call_args
    assert kwargs["python_tools"] == [tool]
    assert kwargs["planning_config"] is planning_config
    assert kwargs["timeout"] == 30
