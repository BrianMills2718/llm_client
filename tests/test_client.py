"""Tests for llm_client. All mock litellm.completion (no real API calls)."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from llm_client import LLMCallResult, call_llm, call_llm_with_tools


def _mock_response(content: str = "Hello!", tool_calls: list | None = None) -> MagicMock:
    """Build a mock litellm response."""
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = content
    mock.choices[0].message.tool_calls = tool_calls
    mock.usage.prompt_tokens = 10
    mock.usage.completion_tokens = 5
    mock.usage.total_tokens = 15
    return mock


class TestCallLLM:
    """Tests for call_llm."""

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_returns_result(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        mock_comp.return_value = _mock_response()
        result = call_llm("gpt-4", [{"role": "user", "content": "Hi"}])
        assert isinstance(result, LLMCallResult)
        assert result.content == "Hello!"
        assert result.cost == 0.001
        assert result.model == "gpt-4"

    @patch("llm_client.client.litellm.completion_cost", return_value=0.01)
    @patch("llm_client.client.litellm.completion")
    def test_passes_num_retries(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        mock_comp.return_value = _mock_response()
        call_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=5)
        kwargs = mock_comp.call_args.kwargs
        assert kwargs["num_retries"] == 5

    @patch("llm_client.client.litellm.completion_cost", return_value=0.01)
    @patch("llm_client.client.litellm.completion")
    def test_reasoning_effort_for_claude(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        mock_comp.return_value = _mock_response()
        call_llm("anthropic/claude-3-opus", [{"role": "user", "content": "Hi"}], reasoning_effort="high")
        kwargs = mock_comp.call_args.kwargs
        assert kwargs["reasoning_effort"] == "high"

    @patch("llm_client.client.litellm.completion_cost", return_value=0.01)
    @patch("llm_client.client.litellm.completion")
    def test_reasoning_effort_ignored_for_non_claude(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        mock_comp.return_value = _mock_response()
        call_llm("gpt-4", [{"role": "user", "content": "Hi"}], reasoning_effort="high")
        kwargs = mock_comp.call_args.kwargs
        assert "reasoning_effort" not in kwargs

    @patch("llm_client.client.litellm.completion")
    def test_raises_on_error(self, mock_comp: MagicMock) -> None:
        mock_comp.side_effect = Exception("API down")
        with pytest.raises(Exception, match="API down"):
            call_llm("gpt-4", [{"role": "user", "content": "Hi"}])

    @patch("llm_client.client.litellm.completion_cost", return_value=0.01)
    @patch("llm_client.client.litellm.completion")
    def test_extracts_usage(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        mock_comp.return_value = _mock_response()
        result = call_llm("gpt-4", [{"role": "user", "content": "Hi"}])
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["completion_tokens"] == 5
        assert result.usage["total_tokens"] == 15

    @patch("llm_client.client.litellm.completion_cost", side_effect=Exception("no pricing"))
    @patch("llm_client.client.litellm.completion")
    def test_cost_fallback(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        mock_comp.return_value = _mock_response()
        result = call_llm("gpt-4", [{"role": "user", "content": "Hi"}])
        assert result.cost > 0  # Fallback estimate
        assert result.cost == 15 * 0.000001  # total_tokens * $1/M


class TestCallLLMWithTools:
    """Tests for call_llm_with_tools."""

    @patch("llm_client.client.litellm.completion_cost", return_value=0.01)
    @patch("llm_client.client.litellm.completion")
    def test_passes_tools(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        mock_comp.return_value = _mock_response()
        tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
        call_llm_with_tools("gpt-4", [{"role": "user", "content": "Hi"}], tools)
        kwargs = mock_comp.call_args.kwargs
        assert kwargs["tools"] == tools

    @patch("llm_client.client.litellm.completion_cost", return_value=0.01)
    @patch("llm_client.client.litellm.completion")
    def test_extracts_tool_calls(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        mock_tc = MagicMock()
        mock_tc.id = "call_1"
        mock_tc.type = "function"
        mock_tc.function.name = "get_weather"
        mock_tc.function.arguments = '{"location": "NYC"}'
        mock_comp.return_value = _mock_response(tool_calls=[mock_tc])

        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
        result = call_llm_with_tools("gpt-4", [{"role": "user", "content": "Weather?"}], tools)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "get_weather"
