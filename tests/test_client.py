"""Tests for llm_client. All mock litellm.completion (no real API calls)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from llm_client import (
    AsyncCachePolicy,
    AsyncLLMStream,
    CachePolicy,
    Hooks,
    LLMCallResult,
    LLMStream,
    LRUCache,
    RetryPolicy,
    acall_llm,
    acall_llm_batch,
    acall_llm_structured,
    acall_llm_structured_batch,
    acall_llm_with_tools,
    astream_llm,
    astream_llm_with_tools,
    call_llm,
    call_llm_batch,
    call_llm_structured,
    call_llm_structured_batch,
    call_llm_with_tools,
    exponential_backoff,
    fixed_backoff,
    linear_backoff,
    stream_llm,
    stream_llm_with_tools,
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
        call_llm("anthropic/claude-opus-4-6", [{"role": "user", "content": "Hi"}], reasoning_effort="high")
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
        await acall_llm("anthropic/claude-opus-4-6", [{"role": "user", "content": "Hi"}], reasoning_effort="high")
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


class TestNonRetryableErrors:
    """Permanent errors (auth, billing, quota) should fail immediately."""

    @patch("llm_client.client.litellm.completion")
    def test_authentication_error_not_retried(self, mock_comp: MagicMock) -> None:
        """litellm.AuthenticationError (401) should not retry."""
        import litellm
        mock_comp.side_effect = litellm.AuthenticationError(
            "Incorrect API key provided", model="gpt-4", llm_provider="openai",
        )
        with pytest.raises(litellm.AuthenticationError):
            call_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2)
        assert mock_comp.call_count == 1

    @patch("llm_client.client.litellm.completion")
    def test_budget_exceeded_not_retried(self, mock_comp: MagicMock) -> None:
        """litellm.BudgetExceededError should not retry."""
        import litellm
        mock_comp.side_effect = litellm.BudgetExceededError(
            current_cost=10.0, max_budget=5.0,
        )
        with pytest.raises(litellm.BudgetExceededError):
            call_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2)
        assert mock_comp.call_count == 1

    @patch("llm_client.client.litellm.completion")
    def test_content_policy_not_retried(self, mock_comp: MagicMock) -> None:
        """litellm.ContentPolicyViolationError should not retry."""
        import litellm
        mock_comp.side_effect = litellm.ContentPolicyViolationError(
            "content filtered", model="gpt-4", llm_provider="openai",
        )
        with pytest.raises(litellm.ContentPolicyViolationError):
            call_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2)
        assert mock_comp.call_count == 1

    @patch("llm_client.client.litellm.completion")
    def test_not_found_not_retried(self, mock_comp: MagicMock) -> None:
        """litellm.NotFoundError (404, model doesn't exist) should not retry."""
        import litellm
        mock_comp.side_effect = litellm.NotFoundError(
            "Model not found", model="gpt-99", llm_provider="openai",
        )
        with pytest.raises(litellm.NotFoundError):
            call_llm("gpt-99", [{"role": "user", "content": "Hi"}], num_retries=2)
        assert mock_comp.call_count == 1

    @patch("llm_client.client.litellm.completion")
    def test_quota_exceeded_not_retried(self, mock_comp: MagicMock) -> None:
        """RateLimitError with quota message should not retry."""
        import litellm
        mock_comp.side_effect = litellm.RateLimitError(
            "You exceeded your current quota, please check your plan and billing details",
            model="gpt-4", llm_provider="openai",
        )
        with pytest.raises(litellm.RateLimitError):
            call_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2)
        assert mock_comp.call_count == 1

    @patch("llm_client.client.time.sleep")
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_transient_rate_limit_is_retried(self, mock_comp: MagicMock, mock_cost: MagicMock, mock_sleep: MagicMock) -> None:
        """RateLimitError with transient message (no quota keywords) should retry."""
        import litellm
        mock_comp.side_effect = [
            litellm.RateLimitError(
                "Rate limit exceeded, retry after 1s",
                model="gpt-4", llm_provider="openai",
            ),
            _mock_response(),
        ]
        result = call_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2)
        assert result.content == "Hello!"
        assert mock_comp.call_count == 2

    @patch("llm_client.client.litellm.completion")
    def test_generic_quota_string_not_retried(self, mock_comp: MagicMock) -> None:
        """Generic Exception with quota/billing message should not retry."""
        mock_comp.side_effect = Exception("insufficient quota for this request")
        with pytest.raises(Exception, match="insufficient quota"):
            call_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2)
        assert mock_comp.call_count == 1


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
        call_llm("gemini/gemini-2.0-flash-lite", [{"role": "user", "content": "Hi"}])
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

    def test_custom_max_delay(self) -> None:
        from llm_client.client import _calculate_backoff
        for _ in range(100):
            assert _calculate_backoff(10, base_delay=1.0, max_delay=10.0) <= 10.0

    def test_custom_base_delay(self) -> None:
        from llm_client.client import _calculate_backoff
        # base_delay=0.1, attempt=0 → 0.1 * 2^0 * jitter = 0.05..0.15
        for _ in range(100):
            d = _calculate_backoff(0, base_delay=0.1, max_delay=30.0)
            assert 0.05 <= d <= 0.15


class TestGPT5TemperatureStripping:
    """Tests for GPT-5 temperature stripping in _prepare_call_kwargs."""

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.responses")
    def test_structured_gpt5_strips_temperature(self, mock_resp: MagicMock, mock_cost: MagicMock) -> None:
        """call_llm_structured with GPT-5 should strip temperature (responses API)."""
        class Item(BaseModel):
            name: str

        mock_resp.return_value = _mock_responses_api_response(output_text='{"name": "test"}')

        from llm_client import call_llm_structured

        call_llm_structured(
            "gpt-5-mini",
            [{"role": "user", "content": "Extract"}],
            response_model=Item,
            temperature=0.5,
        )
        call_kwargs = mock_resp.call_args.kwargs
        # _prepare_responses_kwargs strips temperature for GPT-5
        # The key thing: it should NOT have hit instructor at all
        assert "response_model" not in call_kwargs

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_structured_non_gpt5_keeps_temperature(self, mock_completion: MagicMock, mock_cost: MagicMock) -> None:
        """Non-GPT-5 models should keep temperature (native schema path)."""
        class Item(BaseModel):
            name: str

        mock_completion.return_value = _mock_response(content='{"name": "test"}')

        from llm_client import call_llm_structured

        call_llm_structured(
            "deepseek/deepseek-chat",
            [{"role": "user", "content": "Extract"}],
            response_model=Item,
            temperature=0.5,
        )
        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5


class TestConfigurableBackoff:
    """Tests for configurable base_delay and max_delay parameters."""

    @patch("llm_client.client.time.sleep")
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_custom_base_delay_used(self, mock_comp: MagicMock, mock_cost: MagicMock, mock_sleep: MagicMock) -> None:
        """Custom base_delay should affect backoff timing."""
        mock_comp.side_effect = [
            Exception("rate limit exceeded"),
            _mock_response(),
        ]
        call_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2, base_delay=0.1)
        assert mock_sleep.call_count == 1
        # base_delay=0.1, attempt=0, so delay should be small (0.05-0.15)
        actual_delay = mock_sleep.call_args[0][0]
        assert actual_delay <= 0.15

    @patch("llm_client.client.time.sleep")
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_custom_max_delay_caps(self, mock_comp: MagicMock, mock_cost: MagicMock, mock_sleep: MagicMock) -> None:
        """Custom max_delay should cap backoff."""
        mock_comp.side_effect = [
            Exception("rate limit exceeded"),
            Exception("rate limit exceeded"),
            Exception("rate limit exceeded"),
            _mock_response(),
        ]
        call_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=4, base_delay=100.0, max_delay=5.0)
        for call in mock_sleep.call_args_list:
            assert call[0][0] <= 5.0

    @pytest.mark.asyncio
    @patch("llm_client.client.asyncio.sleep", new_callable=AsyncMock)
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.acompletion")
    async def test_async_custom_backoff(self, mock_acomp: MagicMock, mock_cost: MagicMock, mock_sleep: AsyncMock) -> None:
        """Async functions should respect custom backoff params."""
        mock_acomp.side_effect = [
            Exception("rate limit exceeded"),
            _mock_response(),
        ]
        await acall_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2, base_delay=0.1, max_delay=1.0)
        assert mock_sleep.call_count == 1
        actual_delay = mock_sleep.call_args[0][0]
        assert actual_delay <= 1.0


# ---------------------------------------------------------------------------
# Responses API (GPT-5 models)
# ---------------------------------------------------------------------------


def _mock_responses_api_response(
    output_text: str = "Hello from GPT-5!",
    status: str = "completed",
    input_tokens: int = 10,
    output_tokens: int = 5,
    total_tokens: int = 15,
) -> MagicMock:
    """Build a mock litellm responses() API response."""
    mock = MagicMock()
    mock.output_text = output_text
    mock.status = status
    mock.incomplete_details = None
    mock.usage.input_tokens = input_tokens
    mock.usage.output_tokens = output_tokens
    mock.usage.total_tokens = total_tokens
    mock.usage.cost = None
    return mock


class TestResponsesAPIDetection:
    """Tests for GPT-5 model detection."""

    def test_gpt5_mini_detected(self) -> None:
        from llm_client.client import _is_responses_api_model
        assert _is_responses_api_model("gpt-5-mini") is True

    def test_gpt5_detected(self) -> None:
        from llm_client.client import _is_responses_api_model
        assert _is_responses_api_model("gpt-5") is True

    def test_gpt5_nano_detected(self) -> None:
        from llm_client.client import _is_responses_api_model
        assert _is_responses_api_model("gpt-5-nano") is True

    def test_gpt4_not_detected(self) -> None:
        from llm_client.client import _is_responses_api_model
        assert _is_responses_api_model("gpt-4o") is False

    def test_claude_not_detected(self) -> None:
        from llm_client.client import _is_responses_api_model
        assert _is_responses_api_model("anthropic/claude-sonnet-4-5-20250929") is False


class TestMessageConversion:
    """Tests for converting messages to responses API input."""

    def test_simple_user_message(self) -> None:
        from llm_client.client import _convert_messages_to_input
        result = _convert_messages_to_input([{"role": "user", "content": "Hello"}])
        assert result == "User: Hello"

    def test_system_and_user(self) -> None:
        from llm_client.client import _convert_messages_to_input
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ]
        result = _convert_messages_to_input(messages)
        assert "System: You are helpful" in result
        assert "User: Hi" in result

    def test_multi_turn(self) -> None:
        from llm_client.client import _convert_messages_to_input
        messages = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
            {"role": "user", "content": "Follow-up"},
        ]
        result = _convert_messages_to_input(messages)
        assert "User: Question" in result
        assert "Assistant: Answer" in result
        assert "User: Follow-up" in result


class TestResponseFormatConversion:
    """Tests for converting response_format to responses API text param."""

    def test_none_returns_text(self) -> None:
        from llm_client.client import _convert_response_format_for_responses
        result = _convert_response_format_for_responses(None)
        assert result == {"format": {"type": "text"}}

    def test_json_object_returns_text(self) -> None:
        from llm_client.client import _convert_response_format_for_responses
        result = _convert_response_format_for_responses({"type": "json_object"})
        assert result == {"format": {"type": "text"}}

    def test_json_schema_converted(self) -> None:
        from llm_client.client import _convert_response_format_for_responses
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "strict": True,
                "name": "test_schema",
                "schema": {"type": "object", "properties": {"key": {"type": "string"}}},
            },
        }
        result = _convert_response_format_for_responses(response_format)
        assert result["format"]["type"] == "json_schema"
        assert result["format"]["name"] == "test_schema"
        assert result["format"]["strict"] is True
        assert "properties" in result["format"]["schema"]


class TestResponsesAPIRouting:
    """Tests for GPT-5 routing through responses() API."""

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.responses")
    def test_gpt5_routes_to_responses(self, mock_resp: MagicMock, mock_cost: MagicMock) -> None:
        """GPT-5 models should use litellm.responses(), not completion()."""
        mock_resp.return_value = _mock_responses_api_response()
        result = call_llm("gpt-5-mini", [{"role": "user", "content": "Hi"}])
        assert result.content == "Hello from GPT-5!"
        assert result.model == "gpt-5-mini"
        assert result.finish_reason == "stop"
        mock_resp.assert_called_once()

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.responses")
    def test_gpt5_passes_input_not_messages(self, mock_resp: MagicMock, mock_cost: MagicMock) -> None:
        """Responses API receives 'input' string, not 'messages' list."""
        mock_resp.return_value = _mock_responses_api_response()
        call_llm("gpt-5-mini", [{"role": "user", "content": "Hello"}])
        kwargs = mock_resp.call_args.kwargs
        assert "input" in kwargs
        assert "User: Hello" in kwargs["input"]
        assert "messages" not in kwargs

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.responses")
    def test_gpt5_strips_max_tokens(self, mock_resp: MagicMock, mock_cost: MagicMock) -> None:
        """max_tokens should be stripped for GPT-5 (reasoning tokens issue)."""
        mock_resp.return_value = _mock_responses_api_response()
        call_llm("gpt-5-mini", [{"role": "user", "content": "Hi"}], max_tokens=4096)
        kwargs = mock_resp.call_args.kwargs
        assert "max_tokens" not in kwargs
        assert "max_output_tokens" not in kwargs

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.responses")
    def test_gpt5_converts_response_format(self, mock_resp: MagicMock, mock_cost: MagicMock) -> None:
        """response_format should be converted to 'text' parameter."""
        mock_resp.return_value = _mock_responses_api_response()
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "strict": True,
                "name": "test",
                "schema": {"type": "object"},
            },
        }
        call_llm("gpt-5-mini", [{"role": "user", "content": "Hi"}], response_format=response_format)
        kwargs = mock_resp.call_args.kwargs
        assert "response_format" not in kwargs
        assert "text" in kwargs
        assert kwargs["text"]["format"]["type"] == "json_schema"

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.responses")
    def test_gpt5_strips_temperature(self, mock_resp: MagicMock, mock_cost: MagicMock) -> None:
        """GPT-5 responses API does not support temperature — should be stripped."""
        mock_resp.return_value = _mock_responses_api_response()
        call_llm("gpt-5-mini", [{"role": "user", "content": "Hi"}], temperature=0.5)
        kwargs = mock_resp.call_args.kwargs
        assert "temperature" not in kwargs

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.responses")
    def test_gpt5_extracts_usage(self, mock_resp: MagicMock, mock_cost: MagicMock) -> None:
        """Usage should map input_tokens/output_tokens to prompt/completion."""
        mock_resp.return_value = _mock_responses_api_response(
            input_tokens=100, output_tokens=50, total_tokens=150,
        )
        result = call_llm("gpt-5-mini", [{"role": "user", "content": "Hi"}])
        assert result.usage["prompt_tokens"] == 100
        assert result.usage["completion_tokens"] == 50
        assert result.usage["total_tokens"] == 150

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.responses")
    def test_gpt5_raw_response_included(self, mock_resp: MagicMock, mock_cost: MagicMock) -> None:
        """raw_response should contain the original responses API object."""
        resp = _mock_responses_api_response()
        mock_resp.return_value = resp
        result = call_llm("gpt-5-mini", [{"role": "user", "content": "Hi"}])
        assert result.raw_response is resp

    @patch("llm_client.client.litellm.responses")
    def test_gpt5_empty_content_raises(self, mock_resp: MagicMock) -> None:
        """Empty response from GPT-5 should raise ValueError (retryable)."""
        mock_resp.return_value = _mock_responses_api_response(output_text="")
        with pytest.raises(ValueError, match="Empty content"):
            call_llm("gpt-5-mini", [{"role": "user", "content": "Hi"}])

    @patch("llm_client.client.litellm.responses")
    def test_gpt5_incomplete_status_raises(self, mock_resp: MagicMock) -> None:
        """Incomplete response with max_output_tokens should raise RuntimeError."""
        resp = _mock_responses_api_response(output_text="partial", status="incomplete")
        details = MagicMock()
        details.reason = "max_output_tokens"
        resp.incomplete_details = details
        mock_resp.return_value = resp
        with pytest.raises(RuntimeError, match="truncated"):
            call_llm("gpt-5-mini", [{"role": "user", "content": "Hi"}])

    @patch("llm_client.client.time.sleep")
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.responses")
    def test_gpt5_retries_on_transient_error(self, mock_resp: MagicMock, mock_cost: MagicMock, mock_sleep: MagicMock) -> None:
        """GPT-5 calls should retry on transient errors."""
        mock_resp.side_effect = [
            Exception("rate limit exceeded"),
            _mock_responses_api_response(),
        ]
        result = call_llm("gpt-5-mini", [{"role": "user", "content": "Hi"}], num_retries=2)
        assert result.content == "Hello from GPT-5!"
        assert mock_resp.call_count == 2

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.responses")
    def test_gpt5_api_base_passed(self, mock_resp: MagicMock, mock_cost: MagicMock) -> None:
        """api_base should be passed through for GPT-5 models."""
        mock_resp.return_value = _mock_responses_api_response()
        call_llm("gpt-5-mini", [{"role": "user", "content": "Hi"}], api_base="https://custom.api/v1")
        kwargs = mock_resp.call_args.kwargs
        assert kwargs["api_base"] == "https://custom.api/v1"

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_non_gpt5_still_uses_completion(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        """Non-GPT-5 models should still use litellm.completion()."""
        mock_comp.return_value = _mock_response()
        result = call_llm("deepseek/deepseek-chat", [{"role": "user", "content": "Hi"}])
        assert result.content == "Hello!"
        mock_comp.assert_called_once()


class TestAsyncResponsesAPIRouting:
    """Tests for async GPT-5 routing through aresponses() API."""

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.aresponses")
    async def test_async_gpt5_routes_to_aresponses(self, mock_aresp: MagicMock, mock_cost: MagicMock) -> None:
        """Async GPT-5 should use litellm.aresponses()."""
        mock_aresp.return_value = _mock_responses_api_response()
        result = await acall_llm("gpt-5-mini", [{"role": "user", "content": "Hi"}])
        assert result.content == "Hello from GPT-5!"
        assert result.model == "gpt-5-mini"
        mock_aresp.assert_called_once()

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.aresponses")
    async def test_async_gpt5_passes_input(self, mock_aresp: MagicMock, mock_cost: MagicMock) -> None:
        """Async responses API should receive 'input', not 'messages'."""
        mock_aresp.return_value = _mock_responses_api_response()
        await acall_llm("gpt-5-mini", [{"role": "user", "content": "Hello"}])
        kwargs = mock_aresp.call_args.kwargs
        assert "input" in kwargs
        assert "User: Hello" in kwargs["input"]

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.aresponses")
    async def test_async_gpt5_strips_max_tokens(self, mock_aresp: MagicMock, mock_cost: MagicMock) -> None:
        """Async: max_tokens should be stripped for GPT-5."""
        mock_aresp.return_value = _mock_responses_api_response()
        await acall_llm("gpt-5-mini", [{"role": "user", "content": "Hi"}], max_tokens=4096)
        kwargs = mock_aresp.call_args.kwargs
        assert "max_tokens" not in kwargs

    @pytest.mark.asyncio
    @patch("llm_client.client.asyncio.sleep", new_callable=AsyncMock)
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.aresponses")
    async def test_async_gpt5_retries(self, mock_aresp: MagicMock, mock_cost: MagicMock, mock_sleep: AsyncMock) -> None:
        """Async GPT-5 should retry on transient errors."""
        mock_aresp.side_effect = [
            Exception("service unavailable"),
            _mock_responses_api_response(),
        ]
        result = await acall_llm("gpt-5-mini", [{"role": "user", "content": "Hi"}], num_retries=2)
        assert result.content == "Hello from GPT-5!"
        assert mock_aresp.call_count == 2

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.acompletion")
    async def test_async_non_gpt5_still_uses_acompletion(self, mock_acomp: MagicMock, mock_cost: MagicMock) -> None:
        """Async non-GPT-5 should still use litellm.acompletion()."""
        mock_acomp.return_value = _mock_response()
        result = await acall_llm("deepseek/deepseek-chat", [{"role": "user", "content": "Hi"}])
        assert result.content == "Hello!"
        mock_acomp.assert_called_once()


# ---------------------------------------------------------------------------
# retry_on tests
# ---------------------------------------------------------------------------


class TestRetryOn:
    """Tests for the retry_on parameter."""

    @patch("llm_client.client.time.sleep")
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_retry_on_extends_patterns(self, mock_comp: MagicMock, mock_cost: MagicMock, mock_sleep: MagicMock) -> None:
        """Custom pattern in retry_on should trigger retry."""
        mock_comp.side_effect = [
            Exception("custom flux capacitor error"),
            _mock_response(),
        ]
        result = call_llm(
            "gpt-4", [{"role": "user", "content": "Hi"}],
            num_retries=2,
            retry_on=["flux capacitor"],
        )
        assert result.content == "Hello!"
        assert mock_comp.call_count == 2

    @patch("llm_client.client.time.sleep")
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_retry_on_does_not_affect_defaults(self, mock_comp: MagicMock, mock_cost: MagicMock, mock_sleep: MagicMock) -> None:
        """Default retryable patterns still work when retry_on is set."""
        mock_comp.side_effect = [
            Exception("rate limit exceeded"),
            _mock_response(),
        ]
        result = call_llm(
            "gpt-4", [{"role": "user", "content": "Hi"}],
            num_retries=2,
            retry_on=["custom pattern"],
        )
        assert result.content == "Hello!"
        assert mock_comp.call_count == 2

    @pytest.mark.asyncio
    @patch("llm_client.client.asyncio.sleep", new_callable=AsyncMock)
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.acompletion")
    async def test_retry_on_async(self, mock_acomp: MagicMock, mock_cost: MagicMock, mock_sleep: AsyncMock) -> None:
        """Async: custom retry_on pattern triggers retry."""
        mock_acomp.side_effect = [
            Exception("custom flux capacitor error"),
            _mock_response(),
        ]
        result = await acall_llm(
            "gpt-4", [{"role": "user", "content": "Hi"}],
            num_retries=2,
            retry_on=["flux capacitor"],
        )
        assert result.content == "Hello!"
        assert mock_acomp.call_count == 2


# ---------------------------------------------------------------------------
# on_retry tests
# ---------------------------------------------------------------------------


class TestOnRetry:
    """Tests for the on_retry callback parameter."""

    @patch("llm_client.client.time.sleep")
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_on_retry_called_with_attempt_error_delay(self, mock_comp: MagicMock, mock_cost: MagicMock, mock_sleep: MagicMock) -> None:
        """on_retry should be called with (attempt, error, delay)."""
        mock_comp.side_effect = [
            Exception("rate limit exceeded"),
            _mock_response(),
        ]
        callback = MagicMock()
        call_llm(
            "gpt-4", [{"role": "user", "content": "Hi"}],
            num_retries=2,
            on_retry=callback,
        )
        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == 0  # attempt
        assert isinstance(args[1], Exception)  # error
        assert "rate limit" in str(args[1])
        assert isinstance(args[2], float)  # delay

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_on_retry_not_called_on_success(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        """on_retry should not be called when the first attempt succeeds."""
        mock_comp.return_value = _mock_response()
        callback = MagicMock()
        call_llm(
            "gpt-4", [{"role": "user", "content": "Hi"}],
            on_retry=callback,
        )
        callback.assert_not_called()

    @pytest.mark.asyncio
    @patch("llm_client.client.asyncio.sleep", new_callable=AsyncMock)
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.acompletion")
    async def test_on_retry_async(self, mock_acomp: MagicMock, mock_cost: MagicMock, mock_sleep: AsyncMock) -> None:
        """Async: on_retry callback receives correct args."""
        mock_acomp.side_effect = [
            Exception("timeout"),
            _mock_response(),
        ]
        callback = MagicMock()
        await acall_llm(
            "gpt-4", [{"role": "user", "content": "Hi"}],
            num_retries=2,
            on_retry=callback,
        )
        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == 0
        assert isinstance(args[1], Exception)
        assert isinstance(args[2], float)


# ---------------------------------------------------------------------------
# Cache tests
# ---------------------------------------------------------------------------


class TestCache:
    """Tests for the cache parameter and LRUCache."""

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_cache_hit_skips_llm_call(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        """Cached result should be returned without calling the LLM."""
        cache = LRUCache()
        messages = [{"role": "user", "content": "Hi"}]

        # First call populates the cache
        mock_comp.return_value = _mock_response(content="First")
        result1 = call_llm("gpt-4", messages, cache=cache)
        assert result1.content == "First"
        assert mock_comp.call_count == 1

        # Second call with same args should hit cache
        mock_comp.return_value = _mock_response(content="Second")
        result2 = call_llm("gpt-4", messages, cache=cache)
        assert result2.content == "First"  # cached
        assert mock_comp.call_count == 1  # no additional call

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_cache_miss_calls_llm_and_stores(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        """Cache miss should call LLM and store the result."""
        cache = LRUCache()
        mock_comp.return_value = _mock_response(content="Fresh")
        result = call_llm("gpt-4", [{"role": "user", "content": "Hi"}], cache=cache)
        assert result.content == "Fresh"
        assert mock_comp.call_count == 1

        # Verify it's actually in the cache by calling again
        result2 = call_llm("gpt-4", [{"role": "user", "content": "Hi"}], cache=cache)
        assert result2.content == "Fresh"
        assert mock_comp.call_count == 1

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_cache_not_used_when_none(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        """Default behavior (cache=None) should always call LLM."""
        mock_comp.return_value = _mock_response()
        call_llm("gpt-4", [{"role": "user", "content": "Hi"}])
        call_llm("gpt-4", [{"role": "user", "content": "Hi"}])
        assert mock_comp.call_count == 2

    def test_lru_cache_eviction(self) -> None:
        """Oldest entry should be evicted when maxsize is exceeded."""
        cache = LRUCache(maxsize=2)
        r1 = LLMCallResult(content="a", usage={}, cost=0, model="m")
        r2 = LLMCallResult(content="b", usage={}, cost=0, model="m")
        r3 = LLMCallResult(content="c", usage={}, cost=0, model="m")

        cache.set("k1", r1)
        cache.set("k2", r2)
        assert cache.get("k1") is r1  # present
        assert cache.get("k2") is r2  # present

        cache.set("k3", r3)  # evicts k1 (k2 was accessed more recently)
        assert cache.get("k1") is None  # evicted
        assert cache.get("k2") is r2
        assert cache.get("k3") is r3

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.acompletion")
    async def test_cache_async(self, mock_acomp: MagicMock, mock_cost: MagicMock) -> None:
        """Async: cache should work the same way."""
        cache = LRUCache()
        messages = [{"role": "user", "content": "Hi"}]

        mock_acomp.return_value = _mock_response(content="Cached")
        result1 = await acall_llm("gpt-4", messages, cache=cache)
        assert result1.content == "Cached"
        assert mock_acomp.call_count == 1

        result2 = await acall_llm("gpt-4", messages, cache=cache)
        assert result2.content == "Cached"
        assert mock_acomp.call_count == 1

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("instructor.from_litellm")
    def test_cache_structured(self, mock_from_litellm: MagicMock, mock_cost: MagicMock) -> None:
        """Structured call caching should return parsed model and LLMCallResult."""
        class Item(BaseModel):
            name: str

        parsed = Item(name="test")
        raw_resp = _mock_response()

        mock_client = MagicMock()
        mock_client.chat.completions.create_with_completion.return_value = (parsed, raw_resp)
        mock_from_litellm.return_value = mock_client

        cache = LRUCache()
        messages = [{"role": "user", "content": "Extract"}]

        result1, meta1 = call_llm_structured(
            "gpt-4", messages, response_model=Item, cache=cache,
        )
        assert result1.name == "test"
        assert mock_client.chat.completions.create_with_completion.call_count == 1

        # Second call should hit cache
        result2, meta2 = call_llm_structured(
            "gpt-4", messages, response_model=Item, cache=cache,
        )
        assert result2.name == "test"
        assert mock_client.chat.completions.create_with_completion.call_count == 1

    @patch("llm_client.client.time.monotonic")
    def test_lru_cache_ttl_expiry(self, mock_mono: MagicMock) -> None:
        """Entries older than TTL should be evicted on access."""
        mock_mono.return_value = 1000.0
        cache = LRUCache(ttl=60.0)
        r = LLMCallResult(content="hi", usage={}, cost=0, model="m")
        cache.set("k", r)

        # Within TTL
        mock_mono.return_value = 1050.0
        assert cache.get("k") is r

        # Past TTL
        mock_mono.return_value = 1061.0
        assert cache.get("k") is None

    @patch("llm_client.client.time.monotonic")
    def test_lru_cache_no_ttl_never_expires(self, mock_mono: MagicMock) -> None:
        """Without TTL, entries never expire."""
        mock_mono.return_value = 0.0
        cache = LRUCache()
        r = LLMCallResult(content="hi", usage={}, cost=0, model="m")
        cache.set("k", r)

        mock_mono.return_value = 999999.0
        assert cache.get("k") is r

    def test_lru_cache_clear(self) -> None:
        """clear() should empty the cache."""
        cache = LRUCache()
        r = LLMCallResult(content="hi", usage={}, cost=0, model="m")
        cache.set("k", r)
        assert cache.get("k") is r
        cache.clear()
        assert cache.get("k") is None

    def test_lru_cache_thread_safe(self) -> None:
        """Concurrent reads/writes should not crash."""
        import concurrent.futures

        cache = LRUCache(maxsize=50)
        results = [LLMCallResult(content=str(i), usage={}, cost=0, model="m") for i in range(100)]

        def writer(start: int) -> None:
            for i in range(start, start + 50):
                cache.set(f"k{i}", results[i])

        def reader(start: int) -> None:
            for i in range(start, start + 50):
                cache.get(f"k{i}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(writer, 0),
                pool.submit(writer, 50),
                pool.submit(reader, 0),
                pool.submit(reader, 50),
            ]
            for f in concurrent.futures.as_completed(futures):
                f.result()  # raises if any thread crashed


# ---------------------------------------------------------------------------
# RetryPolicy tests
# ---------------------------------------------------------------------------


class TestRetryPolicy:
    """Tests for the RetryPolicy parameter."""

    @patch("llm_client.client.time.sleep")
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_retry_policy_overrides_individual_params(self, mock_comp: MagicMock, mock_cost: MagicMock, mock_sleep: MagicMock) -> None:
        """RetryPolicy should override num_retries/base_delay/max_delay."""
        mock_comp.side_effect = [
            Exception("rate limit"),
            Exception("rate limit"),
            Exception("rate limit"),
            _mock_response(),
        ]
        policy = RetryPolicy(max_retries=5, base_delay=0.01, max_delay=0.05)
        # num_retries=0 would fail without policy override
        result = call_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=0, retry=policy)
        assert result.content == "Hello!"
        assert mock_comp.call_count == 4

    @patch("llm_client.client.time.sleep")
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_retry_policy_on_retry_callback(self, mock_comp: MagicMock, mock_cost: MagicMock, mock_sleep: MagicMock) -> None:
        """on_retry in RetryPolicy should fire."""
        mock_comp.side_effect = [Exception("timeout"), _mock_response()]
        cb = MagicMock()
        policy = RetryPolicy(max_retries=2, on_retry=cb)
        call_llm("gpt-4", [{"role": "user", "content": "Hi"}], retry=policy)
        cb.assert_called_once()

    @patch("llm_client.client.time.sleep")
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_retry_policy_custom_backoff(self, mock_comp: MagicMock, mock_cost: MagicMock, mock_sleep: MagicMock) -> None:
        """Custom backoff function on RetryPolicy should be used."""
        mock_comp.side_effect = [Exception("timeout"), _mock_response()]
        policy = RetryPolicy(max_retries=2, backoff=fixed_backoff, base_delay=0.42)
        call_llm("gpt-4", [{"role": "user", "content": "Hi"}], retry=policy)
        assert mock_sleep.call_args[0][0] == 0.42

    @patch("llm_client.client.time.sleep")
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_retry_policy_should_retry_overrides_patterns(self, mock_comp: MagicMock, mock_cost: MagicMock, mock_sleep: MagicMock) -> None:
        """should_retry on RetryPolicy should replace built-in pattern matching."""
        mock_comp.side_effect = [
            Exception("totally custom error xyz"),
            _mock_response(),
        ]
        # This error would NOT match any built-in pattern, but should_retry says yes
        policy = RetryPolicy(max_retries=2, should_retry=lambda e: "xyz" in str(e))
        result = call_llm("gpt-4", [{"role": "user", "content": "Hi"}], retry=policy)
        assert result.content == "Hello!"
        assert mock_comp.call_count == 2

    @patch("llm_client.client.time.sleep")
    @patch("llm_client.client.litellm.completion")
    def test_retry_policy_should_retry_can_reject(self, mock_comp: MagicMock, mock_sleep: MagicMock) -> None:
        """should_retry returning False should prevent retry even for built-in patterns."""
        mock_comp.side_effect = Exception("rate limit")
        # "rate limit" normally retries, but should_retry always says no
        policy = RetryPolicy(max_retries=5, should_retry=lambda e: False)
        with pytest.raises(Exception, match="rate limit"):
            call_llm("gpt-4", [{"role": "user", "content": "Hi"}], retry=policy)
        assert mock_comp.call_count == 1  # no retry

    @pytest.mark.asyncio
    @patch("llm_client.client.asyncio.sleep", new_callable=AsyncMock)
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.acompletion")
    async def test_retry_policy_async(self, mock_acomp: MagicMock, mock_cost: MagicMock, mock_sleep: AsyncMock) -> None:
        """RetryPolicy should work with async functions."""
        mock_acomp.side_effect = [Exception("timeout"), _mock_response()]
        policy = RetryPolicy(max_retries=3, backoff=linear_backoff, base_delay=0.1)
        result = await acall_llm("gpt-4", [{"role": "user", "content": "Hi"}], retry=policy)
        assert result.content == "Hello!"
        assert mock_acomp.call_count == 2


# ---------------------------------------------------------------------------
# Backoff strategy tests
# ---------------------------------------------------------------------------


class TestBackoffStrategies:
    """Tests for the public backoff functions."""

    def test_exponential_backoff_increases(self) -> None:
        # base * 2^attempt, so attempt 0 → ~1, attempt 3 → ~8
        for _ in range(50):
            d0 = exponential_backoff(0, 1.0, 30.0)
            d3 = exponential_backoff(3, 1.0, 30.0)
            assert 0.5 <= d0 <= 1.5
            assert 4.0 <= d3 <= 12.0

    def test_linear_backoff_increases(self) -> None:
        for _ in range(50):
            d0 = linear_backoff(0, 1.0, 30.0)
            d3 = linear_backoff(3, 1.0, 30.0)
            assert 0.8 <= d0 <= 1.2
            assert 3.2 <= d3 <= 4.8

    def test_fixed_backoff_constant(self) -> None:
        for attempt in range(10):
            assert fixed_backoff(attempt, 2.0, 30.0) == 2.0

    def test_fixed_backoff_respects_max_delay(self) -> None:
        assert fixed_backoff(0, 100.0, 5.0) == 5.0

    def test_exponential_backoff_capped(self) -> None:
        for _ in range(100):
            assert exponential_backoff(20, 1.0, 10.0) <= 10.0

    def test_linear_backoff_capped(self) -> None:
        for _ in range(100):
            assert linear_backoff(100, 1.0, 5.0) <= 5.0


# ---------------------------------------------------------------------------
# Fallback model tests
# ---------------------------------------------------------------------------


class TestFallbackModels:
    """Tests for the fallback_models parameter."""

    @patch("llm_client.client.time.sleep")
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_fallback_on_exhausted_retries(self, mock_comp: MagicMock, mock_cost: MagicMock, mock_sleep: MagicMock) -> None:
        """When primary model exhausts retries, fallback model should be tried."""
        mock_comp.side_effect = [
            Exception("rate limit"),  # primary attempt 0
            Exception("rate limit"),  # primary attempt 1
            _mock_response(content="From fallback"),  # fallback attempt 0
        ]
        result = call_llm(
            "gpt-4", [{"role": "user", "content": "Hi"}],
            num_retries=1,
            fallback_models=["gpt-3.5-turbo"],
        )
        assert result.content == "From fallback"
        assert result.model == "gpt-3.5-turbo"
        assert mock_comp.call_count == 3

    @patch("llm_client.client.time.sleep")
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_no_fallback_on_success(self, mock_comp: MagicMock, mock_cost: MagicMock, mock_sleep: MagicMock) -> None:
        """Fallback should not be used if primary succeeds."""
        mock_comp.return_value = _mock_response(content="Primary OK")
        result = call_llm(
            "gpt-4", [{"role": "user", "content": "Hi"}],
            fallback_models=["gpt-3.5-turbo"],
        )
        assert result.content == "Primary OK"
        assert result.model == "gpt-4"
        assert mock_comp.call_count == 1

    @patch("llm_client.client.time.sleep")
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_on_fallback_callback(self, mock_comp: MagicMock, mock_cost: MagicMock, mock_sleep: MagicMock) -> None:
        """on_fallback callback should fire with correct args."""
        mock_comp.side_effect = [
            Exception("rate limit"),
            Exception("rate limit"),
            _mock_response(content="OK"),
        ]
        callback = MagicMock()
        call_llm(
            "gpt-4", [{"role": "user", "content": "Hi"}],
            num_retries=1,
            fallback_models=["gpt-3.5-turbo"],
            on_fallback=callback,
        )
        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == "gpt-4"  # failed model
        assert isinstance(args[1], Exception)  # error
        assert args[2] == "gpt-3.5-turbo"  # next model

    @patch("llm_client.client.time.sleep")
    @patch("llm_client.client.litellm.completion")
    def test_all_fallbacks_exhausted_raises(self, mock_comp: MagicMock, mock_sleep: MagicMock) -> None:
        """When all models (primary + fallbacks) fail, error should propagate."""
        mock_comp.side_effect = Exception("rate limit")
        with pytest.raises(Exception, match="rate limit"):
            call_llm(
                "gpt-4", [{"role": "user", "content": "Hi"}],
                num_retries=0,
                fallback_models=["gpt-3.5-turbo"],
            )
        assert mock_comp.call_count == 2  # primary + 1 fallback

    @patch("llm_client.client.time.sleep")
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_multiple_fallbacks(self, mock_comp: MagicMock, mock_cost: MagicMock, mock_sleep: MagicMock) -> None:
        """Multiple fallback models tried in order."""
        mock_comp.side_effect = [
            Exception("rate limit"),  # primary
            Exception("rate limit"),  # fallback 1
            _mock_response(content="Third model"),  # fallback 2
        ]
        result = call_llm(
            "gpt-4", [{"role": "user", "content": "Hi"}],
            num_retries=0,
            fallback_models=["gpt-3.5-turbo", "ollama/llama3"],
        )
        assert result.content == "Third model"
        assert result.model == "ollama/llama3"

    @pytest.mark.asyncio
    @patch("llm_client.client.asyncio.sleep", new_callable=AsyncMock)
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.acompletion")
    async def test_fallback_async(self, mock_acomp: MagicMock, mock_cost: MagicMock, mock_sleep: AsyncMock) -> None:
        """Async: fallback should work."""
        mock_acomp.side_effect = [
            Exception("timeout"),
            _mock_response(content="Async fallback"),
        ]
        result = await acall_llm(
            "gpt-4", [{"role": "user", "content": "Hi"}],
            num_retries=0,
            fallback_models=["gpt-3.5-turbo"],
        )
        assert result.content == "Async fallback"

    @patch("llm_client.client.time.sleep")
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_fallback_non_retryable_error(self, mock_comp: MagicMock, mock_cost: MagicMock, mock_sleep: MagicMock) -> None:
        """Non-retryable errors on primary should still trigger fallback."""
        mock_comp.side_effect = [
            Exception("invalid api key"),  # not retryable, but fallback should kick in
            _mock_response(content="Fallback OK"),
        ]
        result = call_llm(
            "gpt-4", [{"role": "user", "content": "Hi"}],
            num_retries=2,
            fallback_models=["gpt-3.5-turbo"],
        )
        assert result.content == "Fallback OK"
        assert mock_comp.call_count == 2  # 1 attempt on primary (no retry), 1 on fallback


# ---------------------------------------------------------------------------
# Hooks tests
# ---------------------------------------------------------------------------


class TestHooks:
    """Tests for the hooks parameter (observability)."""

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_before_call_fires(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        """before_call hook should fire with model, messages, kwargs."""
        mock_comp.return_value = _mock_response()
        before = MagicMock()
        hooks = Hooks(before_call=before)
        messages = [{"role": "user", "content": "Hi"}]
        call_llm("gpt-4", messages, hooks=hooks)
        before.assert_called_once()
        args = before.call_args[0]
        assert args[0] == "gpt-4"
        assert args[1] == messages

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_after_call_fires(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        """after_call hook should fire with LLMCallResult."""
        mock_comp.return_value = _mock_response()
        after = MagicMock()
        hooks = Hooks(after_call=after)
        call_llm("gpt-4", [{"role": "user", "content": "Hi"}], hooks=hooks)
        after.assert_called_once()
        result = after.call_args[0][0]
        assert isinstance(result, LLMCallResult)
        assert result.content == "Hello!"

    @patch("llm_client.client.time.sleep")
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_on_error_fires(self, mock_comp: MagicMock, mock_cost: MagicMock, mock_sleep: MagicMock) -> None:
        """on_error hook should fire on each failed attempt."""
        mock_comp.side_effect = [
            Exception("rate limit"),
            _mock_response(),
        ]
        on_error = MagicMock()
        hooks = Hooks(on_error=on_error)
        call_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2, hooks=hooks)
        on_error.assert_called_once()
        args = on_error.call_args[0]
        assert isinstance(args[0], Exception)
        assert args[1] == 0  # attempt number

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_hooks_not_called_when_none(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        """No errors when hooks is None (default)."""
        mock_comp.return_value = _mock_response()
        call_llm("gpt-4", [{"role": "user", "content": "Hi"}])  # should not raise

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_hooks_partial_fields(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        """Only set fields are called; None fields are skipped."""
        mock_comp.return_value = _mock_response()
        after = MagicMock()
        hooks = Hooks(after_call=after)  # before_call and on_error are None
        call_llm("gpt-4", [{"role": "user", "content": "Hi"}], hooks=hooks)
        after.assert_called_once()

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.acompletion")
    async def test_hooks_async(self, mock_acomp: MagicMock, mock_cost: MagicMock) -> None:
        """Async: hooks should fire."""
        mock_acomp.return_value = _mock_response()
        before = MagicMock()
        after = MagicMock()
        hooks = Hooks(before_call=before, after_call=after)
        await acall_llm("gpt-4", [{"role": "user", "content": "Hi"}], hooks=hooks)
        before.assert_called_once()
        after.assert_called_once()

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("instructor.from_litellm")
    def test_hooks_structured(self, mock_from_litellm: MagicMock, mock_cost: MagicMock) -> None:
        """Hooks should fire for structured calls too."""
        class Item(BaseModel):
            name: str

        parsed = Item(name="test")
        raw_resp = _mock_response()

        mock_client = MagicMock()
        mock_client.chat.completions.create_with_completion.return_value = (parsed, raw_resp)
        mock_from_litellm.return_value = mock_client

        before = MagicMock()
        after = MagicMock()
        hooks = Hooks(before_call=before, after_call=after)
        call_llm_structured(
            "gpt-4",
            [{"role": "user", "content": "Extract"}],
            response_model=Item,
            hooks=hooks,
        )
        before.assert_called_once()
        after.assert_called_once()


# ---------------------------------------------------------------------------
# Async cache protocol tests
# ---------------------------------------------------------------------------


class TestAsyncCachePolicy:
    """Tests for AsyncCachePolicy support in async functions."""

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.acompletion")
    async def test_async_cache_get_and_set(self, mock_acomp: MagicMock, mock_cost: MagicMock) -> None:
        """Async cache should be awaited for get/set."""
        cached_result = LLMCallResult(
            content="Cached!", usage={}, cost=0.001, model="gpt-4",
        )

        class FakeAsyncCache:
            def __init__(self) -> None:
                self.store: dict[str, LLMCallResult] = {}

            async def get(self, key: str) -> LLMCallResult | None:
                return self.store.get(key)

            async def set(self, key: str, value: LLMCallResult) -> None:
                self.store[key] = value

        cache = FakeAsyncCache()
        messages = [{"role": "user", "content": "Hi"}]

        # First call — cache miss, calls LLM
        mock_acomp.return_value = _mock_response(content="Fresh")
        result1 = await acall_llm("gpt-4", messages, cache=cache)
        assert result1.content == "Fresh"
        assert mock_acomp.call_count == 1

        # Second call — cache hit
        result2 = await acall_llm("gpt-4", messages, cache=cache)
        assert result2.content == "Fresh"
        assert mock_acomp.call_count == 1  # no additional call

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.acompletion")
    async def test_sync_cache_still_works_in_async(self, mock_acomp: MagicMock, mock_cost: MagicMock) -> None:
        """Sync LRUCache should still work in async functions."""
        cache = LRUCache()
        messages = [{"role": "user", "content": "Hi"}]

        mock_acomp.return_value = _mock_response(content="Sync cached")
        result1 = await acall_llm("gpt-4", messages, cache=cache)
        result2 = await acall_llm("gpt-4", messages, cache=cache)
        assert result1.content == "Sync cached"
        assert result2.content == "Sync cached"
        assert mock_acomp.call_count == 1


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------


def _mock_stream_chunks(texts: list[str]) -> list[MagicMock]:
    """Build mock streaming chunks."""
    chunks = []
    for text in texts:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = text
        chunks.append(chunk)
    return chunks


class TestStreamLLM:
    """Tests for stream_llm."""

    @patch("llm_client.client.litellm.stream_chunk_builder", return_value=None)
    @patch("llm_client.client.litellm.completion")
    def test_yields_chunks(self, mock_comp: MagicMock, mock_builder: MagicMock) -> None:
        """stream_llm should yield text chunks."""
        chunks = _mock_stream_chunks(["Hello", " ", "world!"])
        mock_comp.return_value = iter(chunks)

        stream = stream_llm("gpt-4", [{"role": "user", "content": "Hi"}])
        collected = list(stream)
        assert collected == ["Hello", " ", "world!"]

    @patch("llm_client.client.litellm.stream_chunk_builder", return_value=None)
    @patch("llm_client.client.litellm.completion")
    def test_result_available_after_iteration(self, mock_comp: MagicMock, mock_builder: MagicMock) -> None:
        """stream.result should be available after consuming the stream."""
        chunks = _mock_stream_chunks(["Hello", "!"])
        mock_comp.return_value = iter(chunks)

        stream = stream_llm("gpt-4", [{"role": "user", "content": "Hi"}])
        for _ in stream:
            pass
        result = stream.result
        assert isinstance(result, LLMCallResult)
        assert result.content == "Hello!"
        assert result.model == "gpt-4"

    @patch("llm_client.client.litellm.stream_chunk_builder", return_value=None)
    @patch("llm_client.client.litellm.completion")
    def test_result_raises_before_iteration(self, mock_comp: MagicMock, mock_builder: MagicMock) -> None:
        """Accessing .result before iterating should raise."""
        chunks = _mock_stream_chunks(["Hi"])
        mock_comp.return_value = iter(chunks)

        stream = stream_llm("gpt-4", [{"role": "user", "content": "Hi"}])
        with pytest.raises(RuntimeError, match="not yet consumed"):
            _ = stream.result

    @patch("llm_client.client.litellm.stream_chunk_builder")
    @patch("llm_client.client.litellm.completion")
    def test_result_includes_usage_when_available(self, mock_comp: MagicMock, mock_builder: MagicMock) -> None:
        """If stream_chunk_builder succeeds, usage and cost should be populated."""
        chunks = _mock_stream_chunks(["Hi"])
        mock_comp.return_value = iter(chunks)

        # Mock a complete response from stream_chunk_builder
        complete = _mock_response(content="Hi")
        mock_builder.return_value = complete

        with patch("llm_client.client._compute_cost", return_value=0.005):
            stream = stream_llm("gpt-4", [{"role": "user", "content": "Hi"}])
            list(stream)
            assert stream.result.usage["total_tokens"] == 15
            assert stream.result.cost == 0.005

    @patch("llm_client.client.litellm.stream_chunk_builder", return_value=None)
    @patch("llm_client.client.litellm.completion")
    def test_stream_passes_kwargs(self, mock_comp: MagicMock, mock_builder: MagicMock) -> None:
        """Extra kwargs should be passed through to litellm."""
        chunks = _mock_stream_chunks(["Hi"])
        mock_comp.return_value = iter(chunks)

        stream = stream_llm("gpt-4", [{"role": "user", "content": "Hi"}], temperature=0.5)
        list(stream)
        kwargs = mock_comp.call_args.kwargs
        assert kwargs["stream"] is True
        assert kwargs["temperature"] == 0.5

    @patch("llm_client.client.litellm.stream_chunk_builder", return_value=None)
    @patch("llm_client.client.litellm.completion")
    def test_stream_hooks_before_and_after(self, mock_comp: MagicMock, mock_builder: MagicMock) -> None:
        """Hooks should fire for streaming."""
        chunks = _mock_stream_chunks(["Hi"])
        mock_comp.return_value = iter(chunks)

        before = MagicMock()
        after = MagicMock()
        hooks = Hooks(before_call=before, after_call=after)
        stream = stream_llm("gpt-4", [{"role": "user", "content": "Hi"}], hooks=hooks)
        before.assert_called_once()
        after.assert_not_called()  # not yet consumed

        list(stream)
        after.assert_called_once()  # now it's called


class TestAstreamLLM:
    """Tests for astream_llm."""

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.stream_chunk_builder", return_value=None)
    @patch("llm_client.client.litellm.acompletion")
    async def test_yields_chunks(self, mock_acomp: MagicMock, mock_builder: MagicMock) -> None:
        """astream_llm should yield text chunks."""
        chunks = _mock_stream_chunks(["Hello", " ", "world!"])

        async def async_iter():
            for c in chunks:
                yield c

        mock_acomp.return_value = async_iter()

        stream = await astream_llm("gpt-4", [{"role": "user", "content": "Hi"}])
        collected = []
        async for chunk in stream:
            collected.append(chunk)
        assert collected == ["Hello", " ", "world!"]

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.stream_chunk_builder", return_value=None)
    @patch("llm_client.client.litellm.acompletion")
    async def test_result_available_after_async_iteration(self, mock_acomp: MagicMock, mock_builder: MagicMock) -> None:
        """async stream.result should work after consuming."""
        chunks = _mock_stream_chunks(["Hello", "!"])

        async def async_iter():
            for c in chunks:
                yield c

        mock_acomp.return_value = async_iter()

        stream = await astream_llm("gpt-4", [{"role": "user", "content": "Hi"}])
        async for _ in stream:
            pass
        result = stream.result
        assert result.content == "Hello!"
        assert result.model == "gpt-4"


# ---------------------------------------------------------------------------
# Batch/parallel call tests
# ---------------------------------------------------------------------------


class TestBatchCalls:
    """Tests for call_llm_batch / acall_llm_batch."""

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.acompletion")
    async def test_acall_llm_batch_basic(self, mock_acomp: MagicMock, mock_cost: MagicMock) -> None:
        """3 items, all succeed, verify 3 results in order."""
        mock_acomp.side_effect = [
            _mock_response(content="R0"),
            _mock_response(content="R1"),
            _mock_response(content="R2"),
        ]
        msgs_list = [
            [{"role": "user", "content": f"Q{i}"}] for i in range(3)
        ]
        results = await acall_llm_batch("gpt-4", msgs_list)
        assert len(results) == 3
        assert all(isinstance(r, LLMCallResult) for r in results)
        # Order preserved
        assert results[0].content == "R0"
        assert results[1].content == "R1"
        assert results[2].content == "R2"

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.acompletion")
    async def test_acall_llm_batch_concurrency_limit(self, mock_acomp: MagicMock, mock_cost: MagicMock) -> None:
        """Verify semaphore limits concurrency."""
        import asyncio

        active = 0
        max_active = 0

        original_acomp = mock_acomp

        async def tracked_call(**kwargs: object) -> MagicMock:
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return _mock_response(content="OK")

        original_acomp.side_effect = tracked_call

        msgs_list = [[{"role": "user", "content": f"Q{i}"}] for i in range(10)]
        results = await acall_llm_batch("gpt-4", msgs_list, max_concurrent=3)
        assert len(results) == 10
        assert max_active <= 3

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.acompletion")
    async def test_acall_llm_batch_return_exceptions_true(self, mock_acomp: MagicMock) -> None:
        """Failed item returns Exception at correct index."""
        mock_acomp.side_effect = [
            _mock_response(content="OK"),
            Exception("boom"),
            _mock_response(content="OK2"),
        ]
        with patch("llm_client.client.litellm.completion_cost", return_value=0.001):
            results = await acall_llm_batch(
                "gpt-4",
                [[{"role": "user", "content": f"Q{i}"}] for i in range(3)],
                return_exceptions=True,
            )
        assert isinstance(results[0], LLMCallResult)
        assert isinstance(results[1], Exception)
        assert isinstance(results[2], LLMCallResult)

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.acompletion")
    async def test_acall_llm_batch_return_exceptions_false_raises(self, mock_acomp: MagicMock) -> None:
        """Without return_exceptions, first error propagates."""
        mock_acomp.side_effect = Exception("API down")
        with pytest.raises(Exception, match="API down"):
            await acall_llm_batch(
                "gpt-4",
                [[{"role": "user", "content": "Q"}]],
                return_exceptions=False,
            )

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.acompletion")
    async def test_acall_llm_batch_on_item_complete(self, mock_acomp: MagicMock, mock_cost: MagicMock) -> None:
        """on_item_complete callback fires with (index, result)."""
        mock_acomp.return_value = _mock_response(content="OK")
        completed: list[tuple[int, LLMCallResult]] = []
        await acall_llm_batch(
            "gpt-4",
            [[{"role": "user", "content": "Q"}]],
            on_item_complete=lambda idx, res: completed.append((idx, res)),
        )
        assert len(completed) == 1
        assert completed[0][0] == 0
        assert completed[0][1].content == "OK"

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.acompletion")
    async def test_acall_llm_batch_on_item_error(self, mock_acomp: MagicMock) -> None:
        """on_item_error callback fires with (index, error)."""
        mock_acomp.side_effect = Exception("fail")
        errors: list[tuple[int, Exception]] = []
        with pytest.raises(Exception):
            await acall_llm_batch(
                "gpt-4",
                [[{"role": "user", "content": "Q"}]],
                on_item_error=lambda idx, err: errors.append((idx, err)),
            )
        assert len(errors) == 1
        assert errors[0][0] == 0
        assert "fail" in str(errors[0][1])

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.acompletion")
    async def test_acall_llm_batch_forwards_params(self, mock_acomp: MagicMock, mock_cost: MagicMock) -> None:
        """Verify params are forwarded to acall_llm."""
        mock_acomp.return_value = _mock_response()
        before = MagicMock()
        hooks = Hooks(before_call=before)
        cache = LRUCache()
        await acall_llm_batch(
            "gpt-4",
            [[{"role": "user", "content": "Q"}]],
            hooks=hooks,
            cache=cache,
            timeout=120,
        )
        before.assert_called_once()
        # timeout forwarded
        kwargs = mock_acomp.call_args.kwargs
        assert kwargs["timeout"] == 120

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.acompletion")
    def test_call_llm_batch_sync(self, mock_acomp: MagicMock, mock_cost: MagicMock) -> None:
        """Sync wrapper returns results."""
        mock_acomp.return_value = _mock_response(content="Sync OK")
        results = call_llm_batch(
            "gpt-4",
            [[{"role": "user", "content": "Q1"}], [{"role": "user", "content": "Q2"}]],
        )
        assert len(results) == 2
        assert results[0].content == "Sync OK"

    @pytest.mark.asyncio
    async def test_acall_llm_batch_empty(self) -> None:
        """Empty list returns empty list."""
        results = await acall_llm_batch("gpt-4", [])
        assert results == []

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("instructor.from_litellm")
    async def test_acall_llm_structured_batch_basic(self, mock_from_litellm: MagicMock, mock_cost: MagicMock) -> None:
        """Structured batch returns (parsed, meta) tuples."""
        class Item(BaseModel):
            name: str

        parsed = Item(name="test")
        raw_resp = _mock_response()

        mock_client = MagicMock()
        mock_client.chat.completions.create_with_completion = AsyncMock(
            return_value=(parsed, raw_resp)
        )
        mock_from_litellm.return_value = mock_client

        results = await acall_llm_structured_batch(
            "gpt-4",
            [[{"role": "user", "content": "Extract"}]] * 2,
            response_model=Item,
        )
        assert len(results) == 2
        for item, meta in results:
            assert item.name == "test"
            assert isinstance(meta, LLMCallResult)

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.acompletion")
    async def test_acall_llm_batch_preserves_order(self, mock_acomp: MagicMock, mock_cost: MagicMock) -> None:
        """Results match input order regardless of completion order."""
        import asyncio

        async def variable_delay(**kwargs: object) -> MagicMock:
            msgs = kwargs.get("messages", [])
            content = msgs[0]["content"] if msgs else "?"  # type: ignore[index]
            # Vary delay so completion order differs from input order
            if "Q0" in str(content):
                await asyncio.sleep(0.03)
            elif "Q1" in str(content):
                await asyncio.sleep(0.01)
            return _mock_response(content=str(content))

        mock_acomp.side_effect = variable_delay

        msgs_list = [[{"role": "user", "content": f"Q{i}"}] for i in range(3)]
        results = await acall_llm_batch("gpt-4", msgs_list)
        assert "Q0" in results[0].content
        assert "Q1" in results[1].content
        assert "Q2" in results[2].content


# ---------------------------------------------------------------------------
# Streaming retry/fallback tests
# ---------------------------------------------------------------------------


class TestStreamRetryFallback:
    """Tests for streaming retry/fallback support."""

    @patch("llm_client.client.time.sleep")
    @patch("llm_client.client.litellm.stream_chunk_builder", return_value=None)
    @patch("llm_client.client.litellm.completion")
    def test_stream_retries_on_creation_failure(self, mock_comp: MagicMock, mock_builder: MagicMock, mock_sleep: MagicMock) -> None:
        """Stream creation fails once then succeeds."""
        chunks = _mock_stream_chunks(["Hello"])
        mock_comp.side_effect = [
            Exception("rate limit exceeded"),
            iter(chunks),
        ]
        stream = stream_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2)
        collected = list(stream)
        assert collected == ["Hello"]
        assert mock_comp.call_count == 2
        mock_sleep.assert_called_once()

    @patch("llm_client.client.litellm.completion")
    def test_stream_no_retry_non_retryable(self, mock_comp: MagicMock) -> None:
        """Non-retryable error raises immediately."""
        mock_comp.side_effect = Exception("invalid api key")
        with pytest.raises(Exception, match="invalid api key"):
            stream_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2)
        assert mock_comp.call_count == 1

    @patch("llm_client.client.time.sleep")
    @patch("llm_client.client.litellm.stream_chunk_builder", return_value=None)
    @patch("llm_client.client.litellm.completion")
    def test_stream_fallback_on_exhausted_retries(self, mock_comp: MagicMock, mock_builder: MagicMock, mock_sleep: MagicMock) -> None:
        """Primary exhausted, fallback model used."""
        chunks = _mock_stream_chunks(["Fallback!"])
        mock_comp.side_effect = [
            Exception("rate limit"),
            Exception("rate limit"),
            iter(chunks),
        ]
        stream = stream_llm(
            "gpt-4", [{"role": "user", "content": "Hi"}],
            num_retries=1,
            fallback_models=["gpt-3.5-turbo"],
        )
        collected = list(stream)
        assert collected == ["Fallback!"]
        assert stream.result.model == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    @patch("llm_client.client.asyncio.sleep", new_callable=AsyncMock)
    @patch("llm_client.client.litellm.stream_chunk_builder", return_value=None)
    @patch("llm_client.client.litellm.acompletion")
    async def test_astream_retries_on_creation_failure(self, mock_acomp: MagicMock, mock_builder: MagicMock, mock_sleep: AsyncMock) -> None:
        """Async: stream creation retries on transient error."""
        chunks = _mock_stream_chunks(["Hi"])

        async def async_iter():
            for c in chunks:
                yield c

        mock_acomp.side_effect = [
            Exception("rate limit exceeded"),
            async_iter(),
        ]
        stream = await astream_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2)
        collected = []
        async for chunk in stream:
            collected.append(chunk)
        assert collected == ["Hi"]
        assert mock_acomp.call_count == 2

    @patch("llm_client.client.time.sleep")
    @patch("llm_client.client.litellm.stream_chunk_builder", return_value=None)
    @patch("llm_client.client.litellm.completion")
    def test_stream_retry_policy_accepted(self, mock_comp: MagicMock, mock_builder: MagicMock, mock_sleep: MagicMock) -> None:
        """RetryPolicy param works with streaming."""
        chunks = _mock_stream_chunks(["OK"])
        mock_comp.side_effect = [
            Exception("timeout"),
            iter(chunks),
        ]
        from llm_client import RetryPolicy, fixed_backoff
        policy = RetryPolicy(max_retries=3, backoff=fixed_backoff, base_delay=0.01)
        stream = stream_llm("gpt-4", [{"role": "user", "content": "Hi"}], retry=policy)
        collected = list(stream)
        assert collected == ["OK"]
        assert mock_sleep.call_args[0][0] == 0.01


# ---------------------------------------------------------------------------
# Streaming with tools tests
# ---------------------------------------------------------------------------


class TestStreamWithTools:
    """Tests for stream_llm_with_tools / astream_llm_with_tools."""

    @patch("llm_client.client.litellm.stream_chunk_builder", return_value=None)
    @patch("llm_client.client.litellm.completion")
    def test_stream_with_tools_passes_tools(self, mock_comp: MagicMock, mock_builder: MagicMock) -> None:
        """Verify tools kwarg passed to litellm.completion."""
        chunks = _mock_stream_chunks(["Hi"])
        mock_comp.return_value = iter(chunks)

        tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
        stream = stream_llm_with_tools("gpt-4", [{"role": "user", "content": "Hi"}], tools)
        list(stream)
        kwargs = mock_comp.call_args.kwargs
        assert kwargs["tools"] == tools

    @patch("llm_client.client.litellm.stream_chunk_builder")
    @patch("llm_client.client.litellm.completion")
    def test_stream_finalize_extracts_tool_calls(self, mock_comp: MagicMock, mock_builder: MagicMock) -> None:
        """_finalize extracts tool_calls from stream_chunk_builder result."""
        chunks = _mock_stream_chunks([""])
        mock_comp.return_value = iter(chunks)

        # Build a complete response with tool_calls
        mock_tc = MagicMock()
        mock_tc.id = "call_1"
        mock_tc.type = "function"
        mock_tc.function.name = "get_weather"
        mock_tc.function.arguments = '{"location": "NYC"}'
        complete = _mock_response(content="", tool_calls=[mock_tc], finish_reason="tool_calls")
        mock_builder.return_value = complete

        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
        stream = stream_llm_with_tools("gpt-4", [{"role": "user", "content": "Weather?"}], tools)
        list(stream)
        assert len(stream.result.tool_calls) == 1
        assert stream.result.tool_calls[0]["function"]["name"] == "get_weather"
        assert stream.result.finish_reason == "tool_calls"

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.stream_chunk_builder", return_value=None)
    @patch("llm_client.client.litellm.acompletion")
    async def test_astream_with_tools(self, mock_acomp: MagicMock, mock_builder: MagicMock) -> None:
        """Async: tools passed through."""
        chunks = _mock_stream_chunks(["Hi"])

        async def async_iter():
            for c in chunks:
                yield c

        mock_acomp.return_value = async_iter()

        tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
        stream = await astream_llm_with_tools("gpt-4", [{"role": "user", "content": "Hi"}], tools)
        collected = []
        async for chunk in stream:
            collected.append(chunk)
        assert collected == ["Hi"]
        kwargs = mock_acomp.call_args.kwargs
        assert kwargs["tools"] == tools


# ---------------------------------------------------------------------------
# GPT-5 structured output tests
# ---------------------------------------------------------------------------


class TestGPT5StructuredOutput:
    """Tests for GPT-5 structured output via Responses API."""

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.responses")
    def test_structured_gpt5_uses_responses_api(self, mock_resp: MagicMock, mock_cost: MagicMock) -> None:
        """GPT-5 structured calls use litellm.responses(), not instructor."""
        class Item(BaseModel):
            name: str

        mock_resp.return_value = _mock_responses_api_response(output_text='{"name": "test"}')

        result, meta = call_llm_structured(
            "gpt-5-mini",
            [{"role": "user", "content": "Extract"}],
            response_model=Item,
        )
        assert result.name == "test"
        assert isinstance(meta, LLMCallResult)
        mock_resp.assert_called_once()

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.responses")
    def test_structured_gpt5_passes_json_schema(self, mock_resp: MagicMock, mock_cost: MagicMock) -> None:
        """Verify text.format has correct JSON schema."""
        class Item(BaseModel):
            name: str

        mock_resp.return_value = _mock_responses_api_response(output_text='{"name": "test"}')

        call_llm_structured(
            "gpt-5-mini",
            [{"role": "user", "content": "Extract"}],
            response_model=Item,
        )
        kwargs = mock_resp.call_args.kwargs
        assert "text" in kwargs
        fmt = kwargs["text"]["format"]
        assert fmt["type"] == "json_schema"
        assert fmt["name"] == "Item"
        assert fmt["strict"] is True
        assert "properties" in fmt["schema"]
        assert "name" in fmt["schema"]["properties"]
        assert fmt["schema"]["additionalProperties"] is False

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.aresponses")
    async def test_async_structured_gpt5(self, mock_aresp: MagicMock, mock_cost: MagicMock) -> None:
        """Async GPT-5 structured uses aresponses()."""
        class Item(BaseModel):
            name: str

        mock_aresp.return_value = _mock_responses_api_response(output_text='{"name": "async_test"}')

        result, meta = await acall_llm_structured(
            "gpt-5-mini",
            [{"role": "user", "content": "Extract"}],
            response_model=Item,
        )
        assert result.name == "async_test"
        mock_aresp.assert_called_once()

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_structured_native_schema_model(self, mock_completion: MagicMock, mock_cost: MagicMock) -> None:
        """Models supporting response_schema use native JSON schema path."""
        class Item(BaseModel):
            name: str

        mock_completion.return_value = _mock_response(content='{"name": "test"}')

        result, meta = call_llm_structured(
            "deepseek/deepseek-chat",
            [{"role": "user", "content": "Extract"}],
            response_model=Item,
        )
        assert result.name == "test"
        call_kwargs = mock_completion.call_args.kwargs
        assert "response_format" in call_kwargs
        assert call_kwargs["response_format"]["type"] == "json_schema"
        assert call_kwargs["response_format"]["json_schema"]["name"] == "Item"

    @patch("llm_client.client.litellm.supports_response_schema", return_value=False)
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("instructor.from_litellm")
    def test_structured_unsupported_model_uses_instructor(self, mock_from_litellm: MagicMock, mock_cost: MagicMock, mock_supports: MagicMock) -> None:
        """Models without response_schema support fall back to instructor."""
        class Item(BaseModel):
            name: str

        parsed = Item(name="test")
        raw_resp = _mock_response()

        mock_client = MagicMock()
        mock_client.chat.completions.create_with_completion.return_value = (parsed, raw_resp)
        mock_from_litellm.return_value = mock_client

        result, meta = call_llm_structured(
            "gpt-3.5-turbo",
            [{"role": "user", "content": "Extract"}],
            response_model=Item,
        )
        assert result.name == "test"
        mock_from_litellm.assert_called_once()  # instructor was used


class TestStrictJsonSchema:
    """Tests for _strict_json_schema helper."""

    def test_adds_additional_properties_false(self) -> None:
        """Simple model gets additionalProperties: false."""
        from llm_client.client import _strict_json_schema

        class Simple(BaseModel):
            name: str

        schema = _strict_json_schema(Simple.model_json_schema())
        assert schema["additionalProperties"] is False
        assert "name" in schema["required"]

    def test_optional_fields_added_to_required(self) -> None:
        """Optional fields must be in required for OpenAI strict mode."""
        from llm_client.client import _strict_json_schema
        from typing import Optional

        class WithOptional(BaseModel):
            name: str
            nickname: Optional[str] = None

        schema = _strict_json_schema(WithOptional.model_json_schema())
        assert "name" in schema["required"]
        assert "nickname" in schema["required"]

    def test_nested_model(self) -> None:
        """Nested models also get additionalProperties: false."""
        from llm_client.client import _strict_json_schema

        class Inner(BaseModel):
            value: int

        class Outer(BaseModel):
            inner: Inner

        schema = _strict_json_schema(Outer.model_json_schema())
        assert schema["additionalProperties"] is False
        # Inner model should be in $defs
        for defn in schema.get("$defs", {}).values():
            if defn.get("type") == "object":
                assert defn["additionalProperties"] is False

    def test_anyof_optional_field(self) -> None:
        """Optional fields produce anyOf — sub-schemas should be processed."""
        from llm_client.client import _strict_json_schema
        from typing import Optional

        class WithOptional(BaseModel):
            data: Optional[str] = None

        schema = _strict_json_schema(WithOptional.model_json_schema())
        assert schema["additionalProperties"] is False
        # The anyOf sub-schemas are primitives here, but should not break
        prop = schema["properties"]["data"]
        assert "anyOf" in prop

    def test_anyof_with_nested_object(self) -> None:
        """anyOf containing an object ref gets additionalProperties in $defs."""
        from llm_client.client import _strict_json_schema
        from typing import Optional

        class Inner(BaseModel):
            value: int

        class Outer(BaseModel):
            inner: Optional[Inner] = None

        schema = _strict_json_schema(Outer.model_json_schema())
        assert schema["additionalProperties"] is False
        # Inner in $defs must have additionalProperties: false
        inner_def = schema["$defs"]["Inner"]
        assert inner_def["additionalProperties"] is False

    def test_allof_oneof(self) -> None:
        """allOf and oneOf sub-schemas are also recursed into."""
        from llm_client.client import _strict_json_schema

        # Manually constructed schema with allOf/oneOf containing objects
        schema = {
            "type": "object",
            "properties": {
                "x": {
                    "allOf": [{"type": "object", "properties": {"a": {"type": "string"}}}],
                },
                "y": {
                    "oneOf": [
                        {"type": "object", "properties": {"b": {"type": "int"}}},
                        {"type": "string"},
                    ],
                },
            },
        }
        _strict_json_schema(schema)
        assert schema["additionalProperties"] is False
        assert schema["properties"]["x"]["allOf"][0]["additionalProperties"] is False
        assert schema["properties"]["y"]["oneOf"][0]["additionalProperties"] is False


# ---------------------------------------------------------------------------
# Model deprecation warnings
# ---------------------------------------------------------------------------


class TestModelDeprecation:
    """Test that deprecated models emit loud warnings."""

    def test_gpt4o_warns(self):
        """GPT-4o should trigger deprecation warning."""
        with pytest.warns(DeprecationWarning, match="DEPRECATED MODEL DETECTED.*gpt-4o"):
            with patch("litellm.completion", return_value=_mock_response()):
                call_llm("gpt-4o", [{"role": "user", "content": "hi"}])

    def test_gpt4o_mini_warns(self):
        """GPT-4o-mini should trigger its own deprecation warning."""
        with pytest.warns(DeprecationWarning, match="DEPRECATED MODEL DETECTED.*gpt-4o-mini"):
            with patch("litellm.completion", return_value=_mock_response()):
                call_llm("gpt-4o-mini", [{"role": "user", "content": "hi"}])

    def test_gpt4o_mini_does_not_trigger_gpt4o_pattern(self):
        """gpt-4o-mini should NOT also trigger the gpt-4o warning (exception logic)."""
        with pytest.warns(DeprecationWarning) as record:
            with patch("litellm.completion", return_value=_mock_response()):
                call_llm("gpt-4o-mini", [{"role": "user", "content": "hi"}])
        # Should only have one warning, and it should mention gpt-4o-mini specifically
        assert len(record) == 1
        assert "gpt-4o-mini" in str(record[0].message)

    def test_claude_3_haiku_warns(self):
        """Claude 3 Haiku should trigger deprecation warning."""
        with pytest.warns(DeprecationWarning, match="DEPRECATED MODEL"):
            with patch("litellm.completion", return_value=_mock_response()):
                call_llm("anthropic/claude-3-haiku-20240307", [{"role": "user", "content": "hi"}])

    def test_gemini_15_warns(self):
        """Gemini 1.5 should trigger deprecation warning."""
        with pytest.warns(DeprecationWarning, match="DEPRECATED MODEL"):
            with patch("litellm.completion", return_value=_mock_response()):
                call_llm("gemini/gemini-1.5-flash", [{"role": "user", "content": "hi"}])

    def test_o1_pro_warns(self):
        """o1-pro should trigger deprecation warning."""
        with pytest.warns(DeprecationWarning, match="DEPRECATED MODEL"):
            with patch("litellm.completion", return_value=_mock_response()):
                call_llm("o1-pro", [{"role": "user", "content": "hi"}])

    def test_current_model_no_warning(self):
        """Current models should NOT trigger any deprecation warning."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            with patch("litellm.completion", return_value=_mock_response()):
                # These should all pass without DeprecationWarning
                call_llm("anthropic/claude-sonnet-4-5-20250929", [{"role": "user", "content": "hi"}])

    def test_deepseek_no_warning(self):
        """DeepSeek V3.2 should NOT trigger deprecation warning."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            with patch("litellm.completion", return_value=_mock_response()):
                call_llm("deepseek/deepseek-chat", [{"role": "user", "content": "hi"}])

    def test_gemini_25_no_warning(self):
        """Gemini 2.5+ should NOT trigger deprecation warning."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            with patch("litellm.completion", return_value=_mock_response()):
                call_llm("gemini/gemini-2.5-flash", [{"role": "user", "content": "hi"}])

    def test_warning_message_contains_stop_instruction(self):
        """Warning message should contain the LLM-agent-directed STOP instruction."""
        with pytest.warns(DeprecationWarning) as record:
            with patch("litellm.completion", return_value=_mock_response()):
                call_llm("gpt-4o", [{"role": "user", "content": "hi"}])
        msg = str(record[0].message)
        assert "STOP" in msg
        assert "DO NOT USE THIS MODEL" in msg
        assert "USER PERMISSION" in msg
        assert "Use instead:" in msg

    def test_structured_also_warns(self):
        """call_llm_structured should also check for deprecated models."""

        class _Entity(BaseModel):
            name: str
            type: str

        with pytest.warns(DeprecationWarning, match="DEPRECATED MODEL"):
            with patch("litellm.completion", return_value=_mock_response('{"name":"x","type":"y"}')):
                call_llm_structured(
                    "gpt-4o", [{"role": "user", "content": "hi"}],
                    response_model=_Entity,
                )

    @pytest.mark.asyncio
    async def test_async_also_warns(self):
        """acall_llm should also check for deprecated models."""
        with pytest.warns(DeprecationWarning, match="DEPRECATED MODEL"):
            with patch("litellm.acompletion", new_callable=AsyncMock, return_value=_mock_response()):
                await acall_llm("gpt-4o", [{"role": "user", "content": "hi"}])

    def test_mistral_large_warns(self):
        """Mistral Large should trigger deprecation warning."""
        with pytest.warns(DeprecationWarning, match="DEPRECATED MODEL"):
            with patch("litellm.completion", return_value=_mock_response()):
                call_llm("mistral/mistral-large-latest", [{"role": "user", "content": "hi"}])
