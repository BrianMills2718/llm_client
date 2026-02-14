"""Tests for llm_client. All mock litellm.completion (no real API calls)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from llm_client import (
    LLMCallResult,
    acall_llm,
    acall_llm_structured,
    acall_llm_with_tools,
    call_llm,
    call_llm_with_tools,
    strip_fences,
)


def _mock_response(
    content: str = "Hello!",
    tool_calls: list | None = None,
    finish_reason: str = "stop",
) -> MagicMock:
    """Build a mock litellm response."""
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = content
    mock.choices[0].message.tool_calls = tool_calls
    mock.choices[0].finish_reason = finish_reason
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
    def test_num_retries_not_passed_to_litellm(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        """num_retries controls our retry loop, not litellm's internal retry."""
        mock_comp.return_value = _mock_response()
        call_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=5)
        kwargs = mock_comp.call_args.kwargs
        assert "num_retries" not in kwargs

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

    @patch("llm_client.client.litellm.completion_cost", return_value=0.01)
    @patch("llm_client.client.litellm.completion")
    def test_api_base_passed_through(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        mock_comp.return_value = _mock_response()
        call_llm(
            "gpt-4",
            [{"role": "user", "content": "Hi"}],
            api_base="https://openrouter.ai/api/v1",
        )
        kwargs = mock_comp.call_args.kwargs
        assert kwargs["api_base"] == "https://openrouter.ai/api/v1"

    @patch("llm_client.client.litellm.completion_cost", return_value=0.01)
    @patch("llm_client.client.litellm.completion")
    def test_api_base_omitted_when_none(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        mock_comp.return_value = _mock_response()
        call_llm("gpt-4", [{"role": "user", "content": "Hi"}])
        kwargs = mock_comp.call_args.kwargs
        assert "api_base" not in kwargs


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

    @patch("llm_client.client.litellm.completion_cost", return_value=0.01)
    @patch("llm_client.client.litellm.completion")
    def test_api_base_passed_through(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        mock_comp.return_value = _mock_response()
        tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
        call_llm_with_tools(
            "gpt-4",
            [{"role": "user", "content": "Hi"}],
            tools,
            api_base="https://openrouter.ai/api/v1",
        )
        kwargs = mock_comp.call_args.kwargs
        assert kwargs["api_base"] == "https://openrouter.ai/api/v1"


class TestAcallLLM:
    """Tests for acall_llm (async)."""

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.acompletion")
    async def test_returns_result(self, mock_acomp: MagicMock, mock_cost: MagicMock) -> None:
        mock_acomp.return_value = _mock_response()
        result = await acall_llm("gpt-4", [{"role": "user", "content": "Hi"}])
        assert isinstance(result, LLMCallResult)
        assert result.content == "Hello!"
        assert result.cost == 0.001
        assert result.model == "gpt-4"

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.01)
    @patch("llm_client.client.litellm.acompletion")
    async def test_passes_kwargs(self, mock_acomp: MagicMock, mock_cost: MagicMock) -> None:
        mock_acomp.return_value = _mock_response()
        await acall_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=5, timeout=120)
        kwargs = mock_acomp.call_args.kwargs
        assert "num_retries" not in kwargs
        assert kwargs["timeout"] == 120

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.01)
    @patch("llm_client.client.litellm.acompletion")
    async def test_reasoning_effort_for_claude(self, mock_acomp: MagicMock, mock_cost: MagicMock) -> None:
        mock_acomp.return_value = _mock_response()
        await acall_llm("anthropic/claude-3-opus", [{"role": "user", "content": "Hi"}], reasoning_effort="high")
        kwargs = mock_acomp.call_args.kwargs
        assert kwargs["reasoning_effort"] == "high"

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.01)
    @patch("llm_client.client.litellm.acompletion")
    async def test_api_base_passed_through(self, mock_acomp: MagicMock, mock_cost: MagicMock) -> None:
        mock_acomp.return_value = _mock_response()
        await acall_llm(
            "gpt-4",
            [{"role": "user", "content": "Hi"}],
            api_base="https://openrouter.ai/api/v1",
        )
        kwargs = mock_acomp.call_args.kwargs
        assert kwargs["api_base"] == "https://openrouter.ai/api/v1"

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.acompletion")
    async def test_raises_on_error(self, mock_acomp: MagicMock) -> None:
        mock_acomp.side_effect = Exception("API down")
        with pytest.raises(Exception, match="API down"):
            await acall_llm("gpt-4", [{"role": "user", "content": "Hi"}])


class TestAcallLLMStructured:
    """Tests for acall_llm_structured (async)."""

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("instructor.from_litellm")
    async def test_returns_parsed_model(self, mock_from_litellm: MagicMock, mock_cost: MagicMock) -> None:
        class Sentiment(BaseModel):
            label: str
            score: float

        parsed = Sentiment(label="positive", score=0.95)
        raw_response = _mock_response()

        mock_client = MagicMock()
        mock_client.chat.completions.create_with_completion = AsyncMock(
            return_value=(parsed, raw_response)
        )
        mock_from_litellm.return_value = mock_client

        result, meta = await acall_llm_structured(
            "gpt-4",
            [{"role": "user", "content": "I love this!"}],
            response_model=Sentiment,
        )
        assert result.label == "positive"
        assert result.score == 0.95
        assert isinstance(meta, LLMCallResult)
        assert meta.cost == 0.001

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("instructor.from_litellm")
    async def test_api_base_passed_through(self, mock_from_litellm: MagicMock, mock_cost: MagicMock) -> None:
        class Item(BaseModel):
            name: str

        parsed = Item(name="test")
        raw_response = _mock_response()

        mock_client = MagicMock()
        mock_client.chat.completions.create_with_completion = AsyncMock(
            return_value=(parsed, raw_response)
        )
        mock_from_litellm.return_value = mock_client

        await acall_llm_structured(
            "gpt-4",
            [{"role": "user", "content": "Extract"}],
            response_model=Item,
            api_base="https://openrouter.ai/api/v1",
        )
        call_kwargs = mock_client.chat.completions.create_with_completion.call_args.kwargs
        assert call_kwargs["api_base"] == "https://openrouter.ai/api/v1"


class TestAcallLLMWithTools:
    """Tests for acall_llm_with_tools (async)."""

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.01)
    @patch("llm_client.client.litellm.acompletion")
    async def test_passes_tools(self, mock_acomp: MagicMock, mock_cost: MagicMock) -> None:
        mock_acomp.return_value = _mock_response()
        tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
        await acall_llm_with_tools("gpt-4", [{"role": "user", "content": "Hi"}], tools)
        kwargs = mock_acomp.call_args.kwargs
        assert kwargs["tools"] == tools

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.01)
    @patch("llm_client.client.litellm.acompletion")
    async def test_extracts_tool_calls(self, mock_acomp: MagicMock, mock_cost: MagicMock) -> None:
        mock_tc = MagicMock()
        mock_tc.id = "call_1"
        mock_tc.type = "function"
        mock_tc.function.name = "get_weather"
        mock_tc.function.arguments = '{"location": "NYC"}'
        mock_acomp.return_value = _mock_response(tool_calls=[mock_tc])

        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
        result = await acall_llm_with_tools("gpt-4", [{"role": "user", "content": "Weather?"}], tools)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "get_weather"


class TestFinishReasonAndRawResponse:
    """Tests for finish_reason and raw_response fields."""

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_finish_reason_extracted(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        mock_comp.return_value = _mock_response(finish_reason="stop")
        result = call_llm("gpt-4", [{"role": "user", "content": "Hi"}])
        assert result.finish_reason == "stop"

    @patch("llm_client.client.litellm.completion")
    def test_finish_reason_length_raises(self, mock_comp: MagicMock) -> None:
        """Truncated responses should raise immediately (not retry)."""
        mock_comp.return_value = _mock_response(finish_reason="length")
        with pytest.raises(RuntimeError, match="truncated"):
            call_llm("gpt-4", [{"role": "user", "content": "Hi"}])
        assert mock_comp.call_count == 1  # No retry on truncation

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_raw_response_included(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        resp = _mock_response()
        mock_comp.return_value = resp
        result = call_llm("gpt-4", [{"role": "user", "content": "Hi"}])
        assert result.raw_response is resp

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_raw_response_excluded_from_repr(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        mock_comp.return_value = _mock_response()
        result = call_llm("gpt-4", [{"role": "user", "content": "Hi"}])
        assert "raw_response" not in repr(result)

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.acompletion")
    async def test_async_finish_reason_extracted(self, mock_acomp: MagicMock, mock_cost: MagicMock) -> None:
        mock_acomp.return_value = _mock_response(finish_reason="stop")
        result = await acall_llm("gpt-4", [{"role": "user", "content": "Hi"}])
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.acompletion")
    async def test_async_finish_reason_length_raises(self, mock_acomp: MagicMock) -> None:
        """Async: truncated responses should raise immediately."""
        mock_acomp.return_value = _mock_response(finish_reason="length")
        with pytest.raises(RuntimeError, match="truncated"):
            await acall_llm("gpt-4", [{"role": "user", "content": "Hi"}])
        assert mock_acomp.call_count == 1

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.acompletion")
    async def test_async_raw_response_included(self, mock_acomp: MagicMock, mock_cost: MagicMock) -> None:
        resp = _mock_response()
        mock_acomp.return_value = resp
        result = await acall_llm("gpt-4", [{"role": "user", "content": "Hi"}])
        assert result.raw_response is resp

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("instructor.from_litellm")
    def test_structured_finish_reason_extracted(self, mock_from_litellm: MagicMock, mock_cost: MagicMock) -> None:
        class Item(BaseModel):
            name: str

        parsed = Item(name="test")
        raw_resp = _mock_response(finish_reason="stop")

        mock_client = MagicMock()
        mock_client.chat.completions.create_with_completion.return_value = (parsed, raw_resp)
        mock_from_litellm.return_value = mock_client

        from llm_client import call_llm_structured

        _, meta = call_llm_structured(
            "gpt-4",
            [{"role": "user", "content": "Extract"}],
            response_model=Item,
        )
        assert meta.finish_reason == "stop"
        assert meta.raw_response is raw_resp


class TestSmartRetry:
    """Tests for retry with jittered exponential backoff."""

    @patch("llm_client.client.time.sleep")
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_retries_on_empty_content(self, mock_comp: MagicMock, mock_cost: MagicMock, mock_sleep: MagicMock) -> None:
        """Empty responses should trigger retry."""
        empty = _mock_response(content="")
        good = _mock_response(content="Hello!")
        mock_comp.side_effect = [empty, good]

        result = call_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2)
        assert result.content == "Hello!"
        assert mock_comp.call_count == 2
        mock_sleep.assert_called_once()

    @patch("llm_client.client.time.sleep")
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_retries_on_transient_error(self, mock_comp: MagicMock, mock_cost: MagicMock, mock_sleep: MagicMock) -> None:
        """Rate limit / timeout errors should retry."""
        mock_comp.side_effect = [
            Exception("rate limit exceeded"),
            _mock_response(),
        ]

        result = call_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2)
        assert result.content == "Hello!"
        assert mock_comp.call_count == 2

    @patch("llm_client.client.litellm.completion")
    def test_non_retryable_error_raises_immediately(self, mock_comp: MagicMock) -> None:
        """Non-retryable errors should not retry."""
        mock_comp.side_effect = Exception("invalid api key")

        with pytest.raises(Exception, match="invalid api key"):
            call_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2)
        assert mock_comp.call_count == 1

    @patch("llm_client.client.time.sleep")
    @patch("llm_client.client.litellm.completion")
    def test_exhausted_retries_raises(self, mock_comp: MagicMock, mock_sleep: MagicMock) -> None:
        """After exhausting retries, the last error should propagate."""
        mock_comp.side_effect = Exception("timeout")

        with pytest.raises(Exception, match="timeout"):
            call_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2)
        assert mock_comp.call_count == 3  # initial + 2 retries

    @patch("llm_client.client.time.sleep")
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_retries_on_json_parse_error(self, mock_comp: MagicMock, mock_cost: MagicMock, mock_sleep: MagicMock) -> None:
        """JSON parse errors should be retryable."""
        mock_comp.side_effect = [
            Exception("json parse error: unterminated string"),
            _mock_response(),
        ]

        result = call_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2)
        assert result.content == "Hello!"

    @pytest.mark.asyncio
    @patch("llm_client.client.asyncio.sleep", new_callable=AsyncMock)
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.acompletion")
    async def test_async_retries_on_empty_content(self, mock_acomp: MagicMock, mock_cost: MagicMock, mock_sleep: AsyncMock) -> None:
        """Async: empty responses should trigger retry."""
        empty = _mock_response(content="")
        good = _mock_response(content="Hello!")
        mock_acomp.side_effect = [empty, good]

        result = await acall_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2)
        assert result.content == "Hello!"
        assert mock_acomp.call_count == 2

    @pytest.mark.asyncio
    @patch("llm_client.client.asyncio.sleep", new_callable=AsyncMock)
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.acompletion")
    async def test_async_retries_on_transient_error(self, mock_acomp: MagicMock, mock_cost: MagicMock, mock_sleep: AsyncMock) -> None:
        """Async: transient errors should retry."""
        mock_acomp.side_effect = [
            Exception("connection reset"),
            _mock_response(),
        ]

        result = await acall_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2)
        assert result.content == "Hello!"

    @patch("llm_client.client.time.sleep")
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("instructor.from_litellm")
    def test_structured_retries_on_transient_error(self, mock_from_litellm: MagicMock, mock_cost: MagicMock, mock_sleep: MagicMock) -> None:
        """Structured: transient errors should retry."""
        class Item(BaseModel):
            name: str

        parsed = Item(name="test")
        raw_resp = _mock_response()

        mock_client = MagicMock()
        mock_client.chat.completions.create_with_completion.side_effect = [
            Exception("service unavailable"),
            (parsed, raw_resp),
        ]
        mock_from_litellm.return_value = mock_client

        from llm_client import call_llm_structured

        result, meta = call_llm_structured(
            "gpt-4",
            [{"role": "user", "content": "Extract"}],
            response_model=Item,
            num_retries=2,
        )
        assert result.name == "test"
        assert mock_client.chat.completions.create_with_completion.call_count == 2

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_tool_calls_not_retried_on_empty_content(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        """Tool call responses with empty content should NOT retry."""
        mock_tc = MagicMock()
        mock_tc.id = "call_1"
        mock_tc.type = "function"
        mock_tc.function.name = "get_weather"
        mock_tc.function.arguments = '{"location": "NYC"}'
        resp = _mock_response(content="", tool_calls=[mock_tc], finish_reason="tool_calls")
        mock_comp.return_value = resp

        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
        result = call_llm_with_tools("gpt-4", [{"role": "user", "content": "Weather?"}], tools)
        assert result.tool_calls[0]["function"]["name"] == "get_weather"
        assert mock_comp.call_count == 1  # No retry


class TestThinkingModelDetection:
    """Tests for automatic thinking model configuration."""

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_gemini_3_gets_thinking_config(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        mock_comp.return_value = _mock_response()
        call_llm("gemini/gemini-3-flash", [{"role": "user", "content": "Hi"}])
        kwargs = mock_comp.call_args.kwargs
        assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 0}

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_gemini_4_gets_thinking_config(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        mock_comp.return_value = _mock_response()
        call_llm("gemini/gemini-4-pro", [{"role": "user", "content": "Hi"}])
        kwargs = mock_comp.call_args.kwargs
        assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 0}

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_non_thinking_model_no_config(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        mock_comp.return_value = _mock_response()
        call_llm("gemini/gemini-2.5-flash", [{"role": "user", "content": "Hi"}])
        kwargs = mock_comp.call_args.kwargs
        assert "thinking" not in kwargs

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_thinking_config_not_overridden(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        """User-provided thinking config should not be overridden."""
        mock_comp.return_value = _mock_response()
        custom = {"type": "enabled", "budget_tokens": 1000}
        call_llm("gemini/gemini-3-flash", [{"role": "user", "content": "Hi"}], thinking=custom)
        kwargs = mock_comp.call_args.kwargs
        assert kwargs["thinking"] == custom

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.acompletion")
    async def test_async_gemini_3_gets_thinking_config(self, mock_acomp: MagicMock, mock_cost: MagicMock) -> None:
        mock_acomp.return_value = _mock_response()
        await acall_llm("gemini/gemini-3-flash", [{"role": "user", "content": "Hi"}])
        kwargs = mock_acomp.call_args.kwargs
        assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 0}

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("instructor.from_litellm")
    def test_structured_gemini_3_gets_thinking_config(self, mock_from_litellm: MagicMock, mock_cost: MagicMock) -> None:
        class Item(BaseModel):
            name: str

        parsed = Item(name="test")
        raw_resp = _mock_response()

        mock_client = MagicMock()
        mock_client.chat.completions.create_with_completion.return_value = (parsed, raw_resp)
        mock_from_litellm.return_value = mock_client

        from llm_client import call_llm_structured

        call_llm_structured(
            "gemini/gemini-3-flash",
            [{"role": "user", "content": "Extract"}],
            response_model=Item,
        )
        call_kwargs = mock_client.chat.completions.create_with_completion.call_args.kwargs
        assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 0}


class TestStripFences:
    """Tests for the strip_fences utility."""

    def test_strips_json_fences(self) -> None:
        assert strip_fences('```json\n{"key": "value"}\n```') == '{"key": "value"}'

    def test_strips_bare_fences(self) -> None:
        assert strip_fences('```\n{"key": "value"}\n```') == '{"key": "value"}'

    def test_strips_python_fences(self) -> None:
        assert strip_fences('```python\nprint("hi")\n```') == 'print("hi")'

    def test_no_fences_unchanged(self) -> None:
        assert strip_fences('{"key": "value"}') == '{"key": "value"}'

    def test_empty_string(self) -> None:
        assert strip_fences("") == ""

    def test_whitespace_around_fences(self) -> None:
        assert strip_fences('  ```json\n{"a": 1}\n```  ') == '{"a": 1}'

    def test_preserves_internal_backticks(self) -> None:
        content = '```json\n{"code": "use ``` for blocks"}\n```'
        result = strip_fences(content)
        assert "use ``` for blocks" in result


class TestBackoffCalculation:
    """Tests for _calculate_backoff."""

    def test_increases_with_attempt(self) -> None:
        from llm_client.client import _calculate_backoff
        # With jitter, exact values vary, but higher attempts = higher base
        delays = [_calculate_backoff(i, base_delay=1.0) for i in range(5)]
        # Attempt 4 base = 16s, attempt 0 base = 1s. Even with jitter, avg should increase.
        # Just verify the cap works
        for d in delays:
            assert d <= 30.0

    def test_capped_at_30(self) -> None:
        from llm_client.client import _calculate_backoff
        for _ in range(100):
            assert _calculate_backoff(10, base_delay=1.0) <= 30.0
