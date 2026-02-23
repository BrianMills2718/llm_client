"""Tests for llm_client. All mock litellm.completion (no real API calls)."""

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

import llm_client.client as client_mod
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
from llm_client.errors import (
    LLMAuthError,
    LLMCapabilityError,
    LLMContentFilterError,
    LLMError,
    LLMModelNotFoundError,
    LLMQuotaExhaustedError,
)


@pytest.fixture(autouse=True)
def _explicit_test_routing_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Week-1 invariant: routing policy must be explicit in tests."""
    monkeypatch.setenv("LLM_CLIENT_OPENROUTER_ROUTING", "off")


class TestRequiredTags:
    def test_calls_experiment_enforcement_hook(self) -> None:
        with (
            patch("llm_client.client._io_log.enforce_feature_profile") as mock_feature_enforce,
            patch("llm_client.client._io_log.enforce_experiment_context") as mock_experiment_enforce,
        ):
            client_mod._require_tags(
                "digimon.benchmark",
                "trace.required.tags",
                0,
                caller="test_required_tags",
            )
        mock_feature_enforce.assert_called_once_with("digimon.benchmark", caller="llm_client.client")
        mock_experiment_enforce.assert_called_once_with("digimon.benchmark", caller="llm_client.client")

    def test_missing_tags_raise_before_enforcement(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_CLIENT_REQUIRE_TAGS", "1")
        with (
            patch("llm_client.client._io_log.enforce_feature_profile") as mock_feature_enforce,
            patch("llm_client.client._io_log.enforce_experiment_context") as mock_experiment_enforce,
        ):
            with pytest.raises(ValueError, match="Missing required kwargs"):
                client_mod._require_tags(
                    task=None,
                    trace_id="trace.required.tags",
                    max_budget=0,
                    caller="test_required_tags",
                )
        mock_feature_enforce.assert_not_called()
        mock_experiment_enforce.assert_not_called()


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
    mock.choices[0].message.refusal = None
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
        result = call_llm("gpt-4", [{"role": "user", "content": "Hi"}], task="test", trace_id="test_returns_result", max_budget=0)
        assert isinstance(result, LLMCallResult)
        assert result.content == "Hello!"
        assert result.cost == 0.001
        assert result.model == "gpt-4"

    @patch("llm_client.client.litellm.completion_cost", return_value=0.01)
    @patch("llm_client.client.litellm.completion")
    def test_num_retries_not_passed_to_litellm(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        """num_retries controls our retry loop, not litellm's internal retry."""
        mock_comp.return_value = _mock_response()
        call_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=5, task="test", trace_id="test_num_retries", max_budget=0)
        kwargs = mock_comp.call_args.kwargs
        assert "num_retries" not in kwargs

    @patch("llm_client.client.litellm.completion_cost", return_value=0.01)
    @patch("llm_client.client.litellm.completion")
    def test_reasoning_effort_for_claude(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        mock_comp.return_value = _mock_response()
        call_llm("anthropic/claude-opus-4-6", [{"role": "user", "content": "Hi"}], reasoning_effort="high", task="test", trace_id="test_reasoning_claude", max_budget=0)
        kwargs = mock_comp.call_args.kwargs
        assert kwargs["reasoning_effort"] == "high"

    @patch("llm_client.client.litellm.completion_cost", return_value=0.01)
    @patch("llm_client.client.litellm.completion")
    def test_reasoning_effort_ignored_for_non_claude(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        mock_comp.return_value = _mock_response()
        call_llm("gpt-4", [{"role": "user", "content": "Hi"}], reasoning_effort="high", task="test", trace_id="test_reasoning_non_claude", max_budget=0)
        kwargs = mock_comp.call_args.kwargs
        assert "reasoning_effort" not in kwargs

    @patch("llm_client.client.litellm.completion")
    def test_raises_on_error(self, mock_comp: MagicMock) -> None:
        mock_comp.side_effect = Exception("API down")
        with pytest.raises(Exception, match="API down"):
            call_llm("gpt-4", [{"role": "user", "content": "Hi"}], task="test", trace_id="test_raises_error", max_budget=0)

    @patch("llm_client.client.litellm.completion_cost", return_value=0.01)
    @patch("llm_client.client.litellm.completion")
    def test_extracts_usage(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        mock_comp.return_value = _mock_response()
        result = call_llm("gpt-4", [{"role": "user", "content": "Hi"}], task="test", trace_id="test_extracts_usage", max_budget=0)
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["completion_tokens"] == 5
        assert result.usage["total_tokens"] == 15

    @patch("llm_client.client.litellm.completion_cost", side_effect=Exception("no pricing"))
    @patch("llm_client.client.litellm.completion")
    def test_cost_fallback(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        mock_comp.return_value = _mock_response()
        result = call_llm("gpt-4", [{"role": "user", "content": "Hi"}], task="test", trace_id="test_cost_fallback", max_budget=0)
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
            task="test",
            trace_id="test_api_base_passed",
            max_budget=0,
        )
        kwargs = mock_comp.call_args.kwargs
        assert kwargs["api_base"] == "https://openrouter.ai/api/v1"

    @patch("llm_client.client.litellm.completion_cost", return_value=0.01)
    @patch("llm_client.client.litellm.completion")
    def test_api_base_omitted_when_none(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        mock_comp.return_value = _mock_response()
        call_llm("gpt-4", [{"role": "user", "content": "Hi"}], task="test", trace_id="test_api_base_omitted", max_budget=0)
        kwargs = mock_comp.call_args.kwargs
        assert "api_base" not in kwargs


class TestCallLLMWithTools:
    """Tests for call_llm_with_tools."""

    @patch("llm_client.client.litellm.completion_cost", return_value=0.01)
    @patch("llm_client.client.litellm.completion")
    def test_passes_tools(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        mock_comp.return_value = _mock_response()
        tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
        call_llm_with_tools("gpt-4", [{"role": "user", "content": "Hi"}], tools, task="test", trace_id="test_passes_tools", max_budget=0)
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
        result = call_llm_with_tools("gpt-4", [{"role": "user", "content": "Weather?"}], tools, task="test", trace_id="test_extracts_tool_calls", max_budget=0)
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
            task="test",
            trace_id="test_tools_api_base",
            max_budget=0,
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
        result = await acall_llm("gpt-4", [{"role": "user", "content": "Hi"}], task="test", trace_id="test_async_returns_result", max_budget=0)
        assert isinstance(result, LLMCallResult)
        assert result.content == "Hello!"
        assert result.cost == 0.001
        assert result.model == "gpt-4"

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.01)
    @patch("llm_client.client.litellm.acompletion")
    async def test_passes_kwargs(self, mock_acomp: MagicMock, mock_cost: MagicMock) -> None:
        mock_acomp.return_value = _mock_response()
        await acall_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=5, timeout=120, task="test", trace_id="test_async_passes_kwargs", max_budget=0)
        kwargs = mock_acomp.call_args.kwargs
        assert "num_retries" not in kwargs
        assert kwargs["timeout"] == 120

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.01)
    @patch("llm_client.client.litellm.acompletion")
    async def test_reasoning_effort_for_claude(self, mock_acomp: MagicMock, mock_cost: MagicMock) -> None:
        mock_acomp.return_value = _mock_response()
        await acall_llm("anthropic/claude-opus-4-6", [{"role": "user", "content": "Hi"}], reasoning_effort="high", task="test", trace_id="test_async_reasoning_claude", max_budget=0)
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
            task="test",
            trace_id="test_async_api_base",
            max_budget=0,
        )
        kwargs = mock_acomp.call_args.kwargs
        assert kwargs["api_base"] == "https://openrouter.ai/api/v1"

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.acompletion")
    async def test_raises_on_error(self, mock_acomp: MagicMock) -> None:
        mock_acomp.side_effect = Exception("API down")
        with pytest.raises(Exception, match="API down"):
            await acall_llm("gpt-4", [{"role": "user", "content": "Hi"}], task="test", trace_id="test_async_raises_error", max_budget=0)


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
            task="test",
            trace_id="test_async_structured_returns",
            max_budget=0,
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
            task="test",
            trace_id="test_async_structured_api_base",
            max_budget=0,
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
        await acall_llm_with_tools("gpt-4", [{"role": "user", "content": "Hi"}], tools, task="test", trace_id="test_async_tools_passes", max_budget=0)
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
        result = await acall_llm_with_tools("gpt-4", [{"role": "user", "content": "Weather?"}], tools, task="test", trace_id="test_async_tools_extracts", max_budget=0)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "get_weather"


class TestFinishReasonAndRawResponse:
    """Tests for finish_reason and raw_response fields."""

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_finish_reason_extracted(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        mock_comp.return_value = _mock_response(finish_reason="stop")
        result = call_llm("gpt-4", [{"role": "user", "content": "Hi"}], task="test", trace_id="test_finish_reason", max_budget=0)
        assert result.finish_reason == "stop"

    @patch("llm_client.client.litellm.completion")
    def test_finish_reason_length_raises(self, mock_comp: MagicMock) -> None:
        """Truncated responses should raise immediately (not retry)."""
        mock_comp.return_value = _mock_response(finish_reason="length")
        with pytest.raises(LLMError, match="truncated"):
            call_llm("gpt-4", [{"role": "user", "content": "Hi"}], task="test", trace_id="test_finish_length", max_budget=0)
        assert mock_comp.call_count == 1  # No retry on truncation

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_raw_response_included(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        resp = _mock_response()
        mock_comp.return_value = resp
        result = call_llm("gpt-4", [{"role": "user", "content": "Hi"}], task="test", trace_id="test_raw_response", max_budget=0)
        assert result.raw_response is resp

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_raw_response_excluded_from_repr(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        mock_comp.return_value = _mock_response()
        result = call_llm("gpt-4", [{"role": "user", "content": "Hi"}], task="test", trace_id="test_raw_repr", max_budget=0)
        assert "raw_response" not in repr(result)

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.acompletion")
    async def test_async_finish_reason_extracted(self, mock_acomp: MagicMock, mock_cost: MagicMock) -> None:
        mock_acomp.return_value = _mock_response(finish_reason="stop")
        result = await acall_llm("gpt-4", [{"role": "user", "content": "Hi"}], task="test", trace_id="test_async_finish_reason", max_budget=0)
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.acompletion")
    async def test_async_finish_reason_length_raises(self, mock_acomp: MagicMock) -> None:
        """Async: truncated responses should raise immediately."""
        mock_acomp.return_value = _mock_response(finish_reason="length")
        with pytest.raises(LLMError, match="truncated"):
            await acall_llm("gpt-4", [{"role": "user", "content": "Hi"}], task="test", trace_id="test_async_finish_length", max_budget=0)
        assert mock_acomp.call_count == 1

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.acompletion")
    async def test_async_raw_response_included(self, mock_acomp: MagicMock, mock_cost: MagicMock) -> None:
        resp = _mock_response()
        mock_acomp.return_value = resp
        result = await acall_llm("gpt-4", [{"role": "user", "content": "Hi"}], task="test", trace_id="test_async_raw_response", max_budget=0)
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
            task="test",
            trace_id="test_structured_finish_reason",
            max_budget=0,
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

        result = call_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2, task="test", trace_id="test_retry_empty", max_budget=0)
        assert result.content == "Hello!"
        assert mock_comp.call_count == 2
        mock_sleep.assert_called_once()

    @patch("llm_client.client.time.sleep")
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_retries_on_empty_choices(self, mock_comp: MagicMock, mock_cost: MagicMock, mock_sleep: MagicMock) -> None:
        """Empty choices[] should be treated as retryable empty responses."""
        empty_choices = MagicMock()
        empty_choices.choices = []
        empty_choices.usage.prompt_tokens = 0
        empty_choices.usage.completion_tokens = 0
        empty_choices.usage.total_tokens = 0
        good = _mock_response(content="Hello!")
        mock_comp.side_effect = [empty_choices, good]

        result = call_llm(
            "gpt-4",
            [{"role": "user", "content": "Hi"}],
            num_retries=2,
            task="test",
            trace_id="test_retry_empty_choices",
            max_budget=0,
        )
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

        result = call_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2, task="test", trace_id="test_retry_transient", max_budget=0)
        assert result.content == "Hello!"
        assert mock_comp.call_count == 2

    @patch("llm_client.client.litellm.completion")
    def test_non_retryable_error_raises_immediately(self, mock_comp: MagicMock) -> None:
        """Non-retryable errors should not retry."""
        mock_comp.side_effect = Exception("invalid api key")

        with pytest.raises(Exception, match="invalid api key"):
            call_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2, task="test", trace_id="test_non_retryable", max_budget=0)
        assert mock_comp.call_count == 1

    @patch("llm_client.client.time.sleep")
    @patch("llm_client.client.litellm.completion")
    def test_exhausted_retries_raises(self, mock_comp: MagicMock, mock_sleep: MagicMock) -> None:
        """After exhausting retries, the last error should propagate."""
        mock_comp.side_effect = Exception("timeout")

        with pytest.raises(Exception, match="timeout"):
            call_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2, task="test", trace_id="test_exhausted_retries", max_budget=0)
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

        result = call_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2, task="test", trace_id="test_retry_json_parse", max_budget=0)
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

        result = await acall_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2, task="test", trace_id="test_async_retry_empty", max_budget=0)
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

        result = await acall_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2, task="test", trace_id="test_async_retry_transient", max_budget=0)
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
            task="test",
            trace_id="test_structured_retry_transient",
            max_budget=0,
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
        result = call_llm_with_tools("gpt-4", [{"role": "user", "content": "Weather?"}], tools, task="test", trace_id="test_tool_calls_no_retry", max_budget=0)
        assert result.tool_calls[0]["function"]["name"] == "get_weather"
        assert mock_comp.call_count == 1  # No retry

    @patch("llm_client.client.time.sleep")
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_finish_reason_tool_calls_but_no_tools_retries(
        self, mock_comp: MagicMock, mock_cost: MagicMock, mock_sleep: MagicMock,
    ) -> None:
        """finish_reason='tool_calls' with no actual tool calls should retry.

        Some models (e.g. deepseek) return finish_reason='tool_calls' but
        with an empty tool_calls list and empty content. This should be
        treated as an empty response and retried.
        """
        # First call: model claims tool_calls but sends nothing
        bogus = _mock_response(content="", tool_calls=None, finish_reason="tool_calls")
        # Second call: model responds correctly
        good = _mock_response(content="Hello!")
        mock_comp.side_effect = [bogus, good]

        result = call_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2, task="test", trace_id="test_bogus_tool_calls", max_budget=0)
        assert result.content == "Hello!"
        assert mock_comp.call_count == 2  # Retried once


class TestNonRetryableErrors:
    """Permanent errors (auth, billing, quota) should fail immediately."""

    @patch("llm_client.client.litellm.completion")
    def test_authentication_error_not_retried(self, mock_comp: MagicMock) -> None:
        """litellm.AuthenticationError (401) should not retry."""
        import litellm
        mock_comp.side_effect = litellm.AuthenticationError(
            "Incorrect API key provided", model="gpt-4", llm_provider="openai",
        )
        with pytest.raises(LLMAuthError):
            call_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2, task="test", trace_id="test_auth_error", max_budget=0)
        assert mock_comp.call_count == 1

    @patch("llm_client.client.litellm.completion")
    def test_budget_exceeded_not_retried(self, mock_comp: MagicMock) -> None:
        """litellm.BudgetExceededError should not retry."""
        import litellm
        mock_comp.side_effect = litellm.BudgetExceededError(
            current_cost=10.0, max_budget=5.0,
        )
        with pytest.raises(LLMQuotaExhaustedError):
            call_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2, task="test", trace_id="test_budget_exceeded", max_budget=0)
        assert mock_comp.call_count == 1

    @patch("llm_client.client.litellm.completion")
    def test_content_policy_not_retried(self, mock_comp: MagicMock) -> None:
        """litellm.ContentPolicyViolationError should not retry."""
        import litellm
        mock_comp.side_effect = litellm.ContentPolicyViolationError(
            "content filtered", model="gpt-4", llm_provider="openai",
        )
        with pytest.raises(LLMContentFilterError):
            call_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2, task="test", trace_id="test_content_policy", max_budget=0)
        assert mock_comp.call_count == 1

    @patch("llm_client.client.litellm.completion")
    def test_not_found_not_retried(self, mock_comp: MagicMock) -> None:
        """litellm.NotFoundError (404, model doesn't exist) should not retry."""
        import litellm
        mock_comp.side_effect = litellm.NotFoundError(
            "Model not found", model="gpt-99", llm_provider="openai",
        )
        with pytest.raises(LLMModelNotFoundError):
            call_llm("gpt-99", [{"role": "user", "content": "Hi"}], num_retries=2, task="test", trace_id="test_not_found", max_budget=0)
        assert mock_comp.call_count == 1

    @patch("llm_client.client.litellm.completion")
    def test_quota_exceeded_not_retried(self, mock_comp: MagicMock) -> None:
        """RateLimitError with quota message should not retry."""
        import litellm
        mock_comp.side_effect = litellm.RateLimitError(
            "You exceeded your current quota, please check your plan and billing details",
            model="gpt-4", llm_provider="openai",
        )
        with pytest.raises(LLMQuotaExhaustedError):
            call_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2, task="test", trace_id="test_quota_exceeded", max_budget=0)
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
        result = call_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2, task="test", trace_id="test_transient_rate_limit", max_budget=0)
        assert result.content == "Hello!"
        assert mock_comp.call_count == 2

    @patch("llm_client.client.time.sleep")
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_rate_limit_quota_with_retry_delay_is_retried(
        self, mock_comp: MagicMock, mock_cost: MagicMock, mock_sleep: MagicMock,
    ) -> None:
        """Quota-like 429 with explicit retryDelay should retry."""
        import litellm
        mock_comp.side_effect = [
            litellm.RateLimitError(
                (
                    "Quota exceeded for metric: foo. "
                    "Please retry in 14.673264491s. "
                    '{"details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"14s"}]}'
                ),
                model="gemini-2.5-flash",
                llm_provider="gemini",
            ),
            _mock_response(),
        ]

        result = call_llm(
            "gemini/gemini-2.5-flash",
            [{"role": "user", "content": "Hi"}],
            num_retries=2,
            task="test",
            trace_id="test_quota_retry_delay",
            max_budget=0,
        )
        assert result.content == "Hello!"
        assert mock_comp.call_count == 2
        assert mock_sleep.call_count == 1
        assert mock_sleep.call_args.args[0] >= 14.0
        assert any("retry_delay_source=parsed" in warning for warning in result.warnings)

    @patch("llm_client.client.time.sleep")
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_structured_retry_after_hint_is_used(
        self, mock_comp: MagicMock, mock_cost: MagicMock, mock_sleep: MagicMock,
    ) -> None:
        """Structured retry_after hints should be preferred over text parsing."""

        class StructuredRetryError(Exception):
            retry_after = 3.0

        mock_comp.side_effect = [
            StructuredRetryError("rate limit"),
            _mock_response(),
        ]

        result = call_llm(
            "gpt-4",
            [{"role": "user", "content": "Hi"}],
            num_retries=2,
            task="test",
            trace_id="test_structured_retry_after",
            max_budget=0,
        )
        assert result.content == "Hello!"
        assert mock_sleep.call_count == 1
        assert mock_sleep.call_args.args[0] >= 3.0
        assert any("retry_delay_source=structured" in warning for warning in result.warnings)

    @patch("llm_client.client.litellm.completion")
    def test_generic_quota_string_not_retried(self, mock_comp: MagicMock) -> None:
        """Generic Exception with quota/billing message should not retry."""
        mock_comp.side_effect = Exception("insufficient quota for this request")
        with pytest.raises(Exception, match="insufficient quota"):
            call_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2, task="test", trace_id="test_generic_quota", max_budget=0)
        assert mock_comp.call_count == 1


class TestOpenRouterKeyRotation:
    """OpenRouter key-limit handling should rotate keys when possible."""

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_rotates_key_on_key_limit_403(
        self,
        mock_comp: MagicMock,
        mock_cost: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key-aaa1")
        monkeypatch.setenv("OPENROUTER_API_KEY_2", "or-key-bbb2")
        monkeypatch.delenv("OPENROUTER_API_KEYS", raising=False)
        client_mod._reset_openrouter_key_rotation_state()

        class OpenRouterKeyLimitError(Exception):
            status_code = 403
            llm_provider = "openrouter"
            model = "openrouter/openai/gpt-5-mini"

        seen_keys: list[str | None] = []

        def _side_effect(**kwargs: object) -> MagicMock:
            seen_keys.append(os.environ.get("OPENROUTER_API_KEY"))
            if len(seen_keys) == 1:
                raise OpenRouterKeyLimitError("OpenRouter error: Key limit exceeded (total limit)")
            return _mock_response()

        mock_comp.side_effect = _side_effect

        result = call_llm(
            "openrouter/openai/gpt-5-mini",
            [{"role": "user", "content": "Hi"}],
            num_retries=2,
            task="test",
            trace_id="test_openrouter_rotate",
            max_budget=0,
        )
        assert result.content == "Hello!"
        assert mock_comp.call_count == 2
        assert seen_keys == ["or-key-aaa1", "or-key-bbb2"]
        assert os.environ.get("OPENROUTER_API_KEY") == "or-key-bbb2"
        assert any("OPENROUTER_KEY_ROTATED" in warning for warning in result.warnings)

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_rotates_key_on_insufficient_credits_402(
        self,
        mock_comp: MagicMock,
        mock_cost: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key-aaa1")
        monkeypatch.setenv("OPENROUTER_API_KEY_2", "or-key-bbb2")
        monkeypatch.delenv("OPENROUTER_API_KEYS", raising=False)
        client_mod._reset_openrouter_key_rotation_state()

        class OpenRouterInsufficientCreditsError(Exception):
            status_code = 402
            llm_provider = "openrouter"
            model = "openrouter/deepseek/deepseek-chat"

        seen_keys: list[str | None] = []

        def _side_effect(**kwargs: object) -> MagicMock:
            seen_keys.append(os.environ.get("OPENROUTER_API_KEY"))
            if len(seen_keys) == 1:
                raise OpenRouterInsufficientCreditsError(
                    'OpenrouterException - {"error":{"message":"Insufficient credits","code":402}}',
                )
            return _mock_response()

        mock_comp.side_effect = _side_effect

        result = call_llm(
            "openrouter/deepseek/deepseek-chat",
            [{"role": "user", "content": "Hi"}],
            num_retries=2,
            task="test",
            trace_id="test_openrouter_rotate_credits_402",
            max_budget=0,
        )
        assert result.content == "Hello!"
        assert mock_comp.call_count == 2
        assert seen_keys == ["or-key-aaa1", "or-key-bbb2"]
        assert os.environ.get("OPENROUTER_API_KEY") == "or-key-bbb2"
        assert any("OPENROUTER_KEY_ROTATED" in warning for warning in result.warnings)

    @patch("llm_client.client.litellm.completion")
    def test_key_limit_without_backup_key_fails_without_retry(
        self,
        mock_comp: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key-aaa1")
        monkeypatch.delenv("OPENROUTER_API_KEYS", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY_2", raising=False)
        client_mod._reset_openrouter_key_rotation_state()

        class OpenRouterKeyLimitError(Exception):
            status_code = 403
            llm_provider = "openrouter"
            model = "openrouter/openai/gpt-5-mini"

        mock_comp.side_effect = OpenRouterKeyLimitError(
            "OpenRouter error: Key limit exceeded (total limit)",
        )

        with pytest.raises(Exception, match="Key limit exceeded"):
            call_llm(
                "openrouter/openai/gpt-5-mini",
                [{"role": "user", "content": "Hi"}],
                num_retries=2,
                task="test",
                trace_id="test_openrouter_rotate_unavailable",
                max_budget=0,
            )
        assert mock_comp.call_count == 1

    @patch("llm_client.client.litellm.completion")
    def test_explicit_api_key_disables_rotation(
        self,
        mock_comp: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key-aaa1")
        monkeypatch.setenv("OPENROUTER_API_KEY_2", "or-key-bbb2")
        monkeypatch.delenv("OPENROUTER_API_KEYS", raising=False)
        client_mod._reset_openrouter_key_rotation_state()

        class OpenRouterKeyLimitError(Exception):
            status_code = 403
            llm_provider = "openrouter"
            model = "openrouter/openai/gpt-5-mini"

        seen_api_keys: list[str | None] = []

        def _side_effect(**kwargs: object) -> MagicMock:
            seen_api_keys.append(kwargs.get("api_key") if isinstance(kwargs, dict) else None)
            raise OpenRouterKeyLimitError("OpenRouter error: Key limit exceeded (total limit)")

        mock_comp.side_effect = _side_effect

        with pytest.raises(Exception, match="Key limit exceeded"):
            call_llm(
                "openrouter/openai/gpt-5-mini",
                [{"role": "user", "content": "Hi"}],
                api_key="manual-key-9999",
                num_retries=2,
                task="test",
                trace_id="test_openrouter_explicit_key",
                max_budget=0,
            )
        assert mock_comp.call_count == 1
        assert seen_api_keys == ["manual-key-9999"]
        assert os.environ.get("OPENROUTER_API_KEY") == "or-key-aaa1"


class TestThinkingModelDetection:
    """Tests for automatic thinking model configuration."""

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_gemini_3_gets_thinking_config(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        mock_comp.return_value = _mock_response()
        call_llm("gemini/gemini-3-flash", [{"role": "user", "content": "Hi"}], task="test", trace_id="test_gemini3_thinking", max_budget=0)
        kwargs = mock_comp.call_args.kwargs
        assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 0}

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_gemini_4_gets_thinking_config(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        mock_comp.return_value = _mock_response()
        call_llm("gemini/gemini-4-pro", [{"role": "user", "content": "Hi"}], task="test", trace_id="test_gemini4_thinking", max_budget=0)
        kwargs = mock_comp.call_args.kwargs
        assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 0}

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_non_thinking_model_no_config(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        mock_comp.return_value = _mock_response()
        call_llm("gemini/gemini-2.0-flash-lite", [{"role": "user", "content": "Hi"}], task="test", trace_id="test_non_thinking", max_budget=0)
        kwargs = mock_comp.call_args.kwargs
        assert "thinking" not in kwargs

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_thinking_config_not_overridden(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        """User-provided thinking config should not be overridden."""
        mock_comp.return_value = _mock_response()
        custom = {"type": "enabled", "budget_tokens": 1000}
        call_llm("gemini/gemini-3-flash", [{"role": "user", "content": "Hi"}], thinking=custom, task="test", trace_id="test_thinking_not_overridden", max_budget=0)
        kwargs = mock_comp.call_args.kwargs
        assert kwargs["thinking"] == custom

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.acompletion")
    async def test_async_gemini_3_gets_thinking_config(self, mock_acomp: MagicMock, mock_cost: MagicMock) -> None:
        mock_acomp.return_value = _mock_response()
        await acall_llm("gemini/gemini-3-flash", [{"role": "user", "content": "Hi"}], task="test", trace_id="test_async_gemini3_thinking", max_budget=0)
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
            task="test",
            trace_id="test_structured_gemini3_thinking",
            max_budget=0,
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
            task="test",
            trace_id="test_gpt5_strips_temp",
            max_budget=0,
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
            task="test",
            trace_id="test_non_gpt5_keeps_temp",
            max_budget=0,
        )
        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_openrouter_gpt5_strips_sampling_controls(self, mock_completion: MagicMock, mock_cost: MagicMock) -> None:
        """Provider-prefixed GPT-5 calls should also strip incompatible sampling args."""
        mock_completion.return_value = _mock_response(content="ok")

        call_llm(
            "openrouter/openai/gpt-5",
            [{"role": "user", "content": "Hi"}],
            temperature=0.3,
            top_p=0.8,
            logprobs=True,
            task="test",
            trace_id="test_openrouter_gpt5_sampling_strip",
            max_budget=0,
        )
        call_kwargs = mock_completion.call_args.kwargs
        assert "temperature" not in call_kwargs
        assert "top_p" not in call_kwargs
        assert "logprobs" not in call_kwargs

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_openrouter_gpt5_coerce_and_warn_populates_result_warnings(
        self,
        mock_completion: MagicMock,
        mock_cost: MagicMock,
    ) -> None:
        """Coercion should be visible in LLMCallResult.warnings by default."""
        mock_completion.return_value = _mock_response(content="ok")

        result = call_llm(
            "openrouter/openai/gpt-5",
            [{"role": "user", "content": "Hi"}],
            temperature=0.25,
            top_p=0.9,
            unsupported_param_policy="coerce_and_warn",
            task="test",
            trace_id="test_openrouter_gpt5_coerce_warn",
            max_budget=0,
        )
        call_kwargs = mock_completion.call_args.kwargs
        assert "temperature" not in call_kwargs
        assert "top_p" not in call_kwargs
        assert any("COERCE_PARAMS" in w for w in result.warnings)
        assert any("gpt5_sampling_compatibility" in w for w in result.warnings)

    def test_openrouter_gpt5_strict_policy_raises(self) -> None:
        """unsupported_param_policy=error should fail loud instead of coercing."""
        with pytest.raises(LLMCapabilityError, match="Unsupported params for model"):
            call_llm(
                "openrouter/openai/gpt-5",
                [{"role": "user", "content": "Hi"}],
                temperature=0.25,
                unsupported_param_policy="error",
                task="test",
                trace_id="test_openrouter_gpt5_strict_error",
                max_budget=0,
            )


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
        call_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2, base_delay=0.1, task="test", trace_id="test_custom_base_delay", max_budget=0)
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
        call_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=4, base_delay=100.0, max_delay=5.0, task="test", trace_id="test_custom_max_delay", max_budget=0)
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
        await acall_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2, base_delay=0.1, max_delay=1.0, task="test", trace_id="test_async_custom_backoff", max_budget=0)
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
    output: list | None = None,
) -> MagicMock:
    """Build a mock litellm responses() API response."""
    mock = MagicMock()
    mock.output_text = output_text
    mock.output = output or []
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

    def test_gpt52_pro_detected(self) -> None:
        from llm_client.client import _is_responses_api_model
        assert _is_responses_api_model("gpt-5.2-pro") is True

    def test_gpt4_not_detected(self) -> None:
        from llm_client.client import _is_responses_api_model
        assert _is_responses_api_model("gpt-4o") is False

    def test_claude_not_detected(self) -> None:
        from llm_client.client import _is_responses_api_model
        assert _is_responses_api_model("anthropic/claude-sonnet-4-5-20250929") is False


class TestToolSchemaConversion:
    """Tests for ChatCompletions → Responses API tool schema conversion."""

    def test_nested_format_flattened(self) -> None:
        from llm_client.client import _convert_tools_for_responses_api
        tools = [{"type": "function", "function": {"name": "foo", "description": "d", "parameters": {}}}]
        result = _convert_tools_for_responses_api(tools)
        assert result == [{"type": "function", "name": "foo", "description": "d", "parameters": {}}]

    def test_already_flat_passthrough(self) -> None:
        from llm_client.client import _convert_tools_for_responses_api
        tools = [{"type": "function", "name": "foo", "description": "d", "parameters": {}}]
        result = _convert_tools_for_responses_api(tools)
        assert result == tools

    def test_multiple_tools(self) -> None:
        from llm_client.client import _convert_tools_for_responses_api
        tools = [
            {"type": "function", "function": {"name": "a", "description": "x", "parameters": {}}},
            {"type": "function", "function": {"name": "b", "description": "y", "parameters": {"type": "object"}}},
        ]
        result = _convert_tools_for_responses_api(tools)
        assert len(result) == 2
        assert result[0]["name"] == "a"
        assert result[1]["name"] == "b"
        assert result[1]["parameters"] == {"type": "object"}

    def test_idempotent(self) -> None:
        from llm_client.client import _convert_tools_for_responses_api
        tools = [{"type": "function", "function": {"name": "foo", "description": "d", "parameters": {}}}]
        once = _convert_tools_for_responses_api(tools)
        twice = _convert_tools_for_responses_api(once)
        assert once == twice


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
        result = call_llm("gpt-5-mini", [{"role": "user", "content": "Hi"}], task="test", trace_id="test_gpt5_routes", max_budget=0)
        assert result.content == "Hello from GPT-5!"
        assert result.model == "gpt-5-mini"
        assert result.finish_reason == "stop"
        mock_resp.assert_called_once()

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.responses")
    def test_gpt5_passes_input_not_messages(self, mock_resp: MagicMock, mock_cost: MagicMock) -> None:
        """Responses API receives 'input' string, not 'messages' list."""
        mock_resp.return_value = _mock_responses_api_response()
        call_llm("gpt-5-mini", [{"role": "user", "content": "Hello"}], task="test", trace_id="test_gpt5_input", max_budget=0)
        kwargs = mock_resp.call_args.kwargs
        assert "input" in kwargs
        assert "User: Hello" in kwargs["input"]
        assert "messages" not in kwargs

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.responses")
    def test_gpt5_strips_max_tokens(self, mock_resp: MagicMock, mock_cost: MagicMock) -> None:
        """max_tokens should be stripped for GPT-5 (reasoning tokens issue)."""
        mock_resp.return_value = _mock_responses_api_response()
        call_llm("gpt-5-mini", [{"role": "user", "content": "Hi"}], max_tokens=4096, task="test", trace_id="test_gpt5_strips_max", max_budget=0)
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
        call_llm("gpt-5-mini", [{"role": "user", "content": "Hi"}], response_format=response_format, task="test", trace_id="test_gpt5_resp_format", max_budget=0)
        kwargs = mock_resp.call_args.kwargs
        assert "response_format" not in kwargs
        assert "text" in kwargs
        assert kwargs["text"]["format"]["type"] == "json_schema"

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.responses")
    def test_gpt5_converts_tools_to_flat_format(self, mock_resp: MagicMock, mock_cost: MagicMock) -> None:
        """Tools passed to GPT-5 should be flattened from ChatCompletions to Responses API format."""
        mock_resp.return_value = _mock_responses_api_response()
        chat_completions_tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search entities",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                },
            }
        ]
        call_llm("gpt-5-mini", [{"role": "user", "content": "Hi"}], tools=chat_completions_tools, task="test", trace_id="test_gpt5_flat_tools", max_budget=0)
        kwargs = mock_resp.call_args.kwargs
        tools = kwargs["tools"]
        assert len(tools) == 1
        # Should be flat (Responses API format), not nested
        assert "function" not in tools[0], "tools should be flattened for Responses API"
        assert tools[0]["name"] == "search"
        assert tools[0]["description"] == "Search entities"
        assert tools[0]["parameters"]["properties"]["q"]["type"] == "string"

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.responses")
    def test_gpt52_pro_xhigh_enables_background_with_reasoning(
        self,
        mock_resp: MagicMock,
        mock_cost: MagicMock,
    ) -> None:
        """gpt-5.2-pro with xhigh reasoning should use background mode and reasoning payload."""
        mock_resp.return_value = _mock_responses_api_response()
        result = call_llm(
            "gpt-5.2-pro",
            [{"role": "user", "content": "Deep review"}],
            reasoning_effort="xhigh",
            task="test",
            trace_id="test_gpt52_background_reasoning",
            max_budget=0,
        )
        kwargs = mock_resp.call_args.kwargs
        assert kwargs["background"] is True
        assert kwargs["reasoning"] == {"effort": "xhigh"}
        assert isinstance(result.routing_trace, dict)
        assert result.routing_trace.get("background_mode") is True

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client._poll_background_response")
    @patch("llm_client.client.litellm.responses")
    def test_gpt52_background_in_progress_polls_until_complete(
        self,
        mock_resp: MagicMock,
        mock_poll: MagicMock,
        mock_cost: MagicMock,
    ) -> None:
        """If initial background response is pending, sync poller should be used."""
        initial = _mock_responses_api_response(output_text="", status="in_progress")
        initial.id = "resp_sync_123"
        completed = _mock_responses_api_response(output_text="done", status="completed")
        mock_resp.return_value = initial
        mock_poll.return_value = completed

        result = call_llm(
            "gpt-5.2-pro",
            [{"role": "user", "content": "Deep review"}],
            reasoning_effort="xhigh",
            background_timeout=123,
            background_poll_interval=7,
            task="test",
            trace_id="test_gpt52_sync_background_poll",
            max_budget=0,
        )

        assert result.content == "done"
        mock_poll.assert_called_once()
        poll_args = mock_poll.call_args
        assert poll_args.args[0] == "resp_sync_123"
        assert poll_args.kwargs["timeout"] == 123
        assert poll_args.kwargs["poll_interval"] == 7
        assert poll_args.kwargs["request_timeout"] == 60

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.responses")
    def test_gpt5_strips_temperature(self, mock_resp: MagicMock, mock_cost: MagicMock) -> None:
        """GPT-5 responses API does not support temperature — should be stripped."""
        mock_resp.return_value = _mock_responses_api_response()
        call_llm("gpt-5-mini", [{"role": "user", "content": "Hi"}], temperature=0.5, task="test", trace_id="test_gpt5_strips_temp", max_budget=0)
        kwargs = mock_resp.call_args.kwargs
        assert "temperature" not in kwargs

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.responses")
    def test_gpt5_strips_top_p_and_logprobs(self, mock_resp: MagicMock, mock_cost: MagicMock) -> None:
        """GPT-5 requests should not forward unsupported sampling extras."""
        mock_resp.return_value = _mock_responses_api_response()
        call_llm(
            "gpt-5-mini",
            [{"role": "user", "content": "Hi"}],
            top_p=0.7,
            logprobs=True,
            top_logprobs=3,
            task="test",
            trace_id="test_gpt5_strips_extra_sampling",
            max_budget=0,
        )
        kwargs = mock_resp.call_args.kwargs
        assert "top_p" not in kwargs
        assert "logprobs" not in kwargs
        assert "top_logprobs" not in kwargs

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.responses")
    def test_gpt5_extracts_usage(self, mock_resp: MagicMock, mock_cost: MagicMock) -> None:
        """Usage should map input_tokens/output_tokens to prompt/completion."""
        mock_resp.return_value = _mock_responses_api_response(
            input_tokens=100, output_tokens=50, total_tokens=150,
        )
        result = call_llm("gpt-5-mini", [{"role": "user", "content": "Hi"}], task="test", trace_id="test_gpt5_usage", max_budget=0)
        assert result.usage["prompt_tokens"] == 100
        assert result.usage["completion_tokens"] == 50
        assert result.usage["total_tokens"] == 150

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.responses")
    def test_gpt5_raw_response_included(self, mock_resp: MagicMock, mock_cost: MagicMock) -> None:
        """raw_response should contain the original responses API object."""
        resp = _mock_responses_api_response()
        mock_resp.return_value = resp
        result = call_llm("gpt-5-mini", [{"role": "user", "content": "Hi"}], task="test", trace_id="test_gpt5_raw_resp", max_budget=0)
        assert result.raw_response is resp

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.responses")
    def test_gpt5_tool_call_output_without_text(self, mock_resp: MagicMock, mock_cost: MagicMock) -> None:
        """Responses API function_call output should map to OpenAI-style tool_calls."""
        tool_item = SimpleNamespace(
            type="function_call",
            call_id="call_abc123",
            name="get_weather",
            arguments='{"location":"NYC"}',
        )
        mock_resp.return_value = _mock_responses_api_response(output_text="", output=[tool_item])
        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
        result = call_llm_with_tools(
            "gpt-5-mini",
            [{"role": "user", "content": "Weather?"}],
            tools,
            task="test",
            trace_id="test_gpt5_tool_call_output_without_text",
            max_budget=0,
        )
        assert result.finish_reason == "tool_calls"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["id"] == "call_abc123"
        assert result.tool_calls[0]["function"]["name"] == "get_weather"
        assert '"location":"NYC"' in result.tool_calls[0]["function"]["arguments"]

    @patch("llm_client.client.litellm.responses")
    def test_gpt5_empty_content_raises(self, mock_resp: MagicMock) -> None:
        """Empty response from GPT-5 should raise ValueError (retryable)."""
        mock_resp.return_value = _mock_responses_api_response(output_text="")
        with pytest.raises(LLMError, match="Empty content"):
            call_llm("gpt-5-mini", [{"role": "user", "content": "Hi"}], task="test", trace_id="test_gpt5_empty", max_budget=0)

    @patch("llm_client.client.litellm.responses")
    def test_gpt5_incomplete_status_raises(self, mock_resp: MagicMock) -> None:
        """Incomplete response with max_output_tokens should raise RuntimeError."""
        resp = _mock_responses_api_response(output_text="partial", status="incomplete")
        details = MagicMock()
        details.reason = "max_output_tokens"
        resp.incomplete_details = details
        mock_resp.return_value = resp
        with pytest.raises(LLMError, match="truncated"):
            call_llm("gpt-5-mini", [{"role": "user", "content": "Hi"}], task="test", trace_id="test_gpt5_incomplete", max_budget=0)

    @patch("llm_client.client.time.sleep")
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.responses")
    def test_gpt5_retries_on_transient_error(self, mock_resp: MagicMock, mock_cost: MagicMock, mock_sleep: MagicMock) -> None:
        """GPT-5 calls should retry on transient errors."""
        mock_resp.side_effect = [
            Exception("rate limit exceeded"),
            _mock_responses_api_response(),
        ]
        result = call_llm("gpt-5-mini", [{"role": "user", "content": "Hi"}], num_retries=2, task="test", trace_id="test_gpt5_retry_transient", max_budget=0)
        assert result.content == "Hello from GPT-5!"
        assert mock_resp.call_count == 2

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.responses")
    def test_gpt5_api_base_passed(self, mock_resp: MagicMock, mock_cost: MagicMock) -> None:
        """api_base should be passed through for GPT-5 models."""
        mock_resp.return_value = _mock_responses_api_response()
        call_llm("gpt-5-mini", [{"role": "user", "content": "Hi"}], api_base="https://custom.api/v1", task="test", trace_id="test_gpt5_api_base", max_budget=0)
        kwargs = mock_resp.call_args.kwargs
        assert kwargs["api_base"] == "https://custom.api/v1"

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_non_gpt5_still_uses_completion(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        """Non-GPT-5 models should still use litellm.completion()."""
        mock_comp.return_value = _mock_response()
        result = call_llm("deepseek/deepseek-chat", [{"role": "user", "content": "Hi"}], task="test", trace_id="test_non_gpt5_completion", max_budget=0)
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
        result = await acall_llm("gpt-5-mini", [{"role": "user", "content": "Hi"}], task="test", trace_id="test_async_gpt5_routes", max_budget=0)
        assert result.content == "Hello from GPT-5!"
        assert result.model == "gpt-5-mini"
        mock_aresp.assert_called_once()

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.aresponses")
    async def test_async_gpt5_passes_input(self, mock_aresp: MagicMock, mock_cost: MagicMock) -> None:
        """Async responses API should receive 'input', not 'messages'."""
        mock_aresp.return_value = _mock_responses_api_response()
        await acall_llm("gpt-5-mini", [{"role": "user", "content": "Hello"}], task="test", trace_id="test_async_gpt5_input", max_budget=0)
        kwargs = mock_aresp.call_args.kwargs
        assert "input" in kwargs
        assert "User: Hello" in kwargs["input"]

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.aresponses")
    async def test_async_gpt5_strips_max_tokens(self, mock_aresp: MagicMock, mock_cost: MagicMock) -> None:
        """Async: max_tokens should be stripped for GPT-5."""
        mock_aresp.return_value = _mock_responses_api_response()
        await acall_llm("gpt-5-mini", [{"role": "user", "content": "Hi"}], max_tokens=4096, task="test", trace_id="test_async_gpt5_strips_max", max_budget=0)
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
        result = await acall_llm("gpt-5-mini", [{"role": "user", "content": "Hi"}], num_retries=2, task="test", trace_id="test_async_gpt5_retries", max_budget=0)
        assert result.content == "Hello from GPT-5!"
        assert mock_aresp.call_count == 2

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client._apoll_background_response", new_callable=AsyncMock)
    @patch("llm_client.client.litellm.aresponses")
    async def test_async_gpt52_background_in_progress_polls_until_complete(
        self,
        mock_aresp: MagicMock,
        mock_apoll: AsyncMock,
        mock_cost: MagicMock,
    ) -> None:
        """Async path should poll until completed when background response is pending."""
        initial = _mock_responses_api_response(output_text="", status="in_progress")
        initial.id = "resp_async_123"
        completed = _mock_responses_api_response(output_text="done", status="completed")
        mock_aresp.return_value = initial
        mock_apoll.return_value = completed

        result = await acall_llm(
            "gpt-5.2-pro",
            [{"role": "user", "content": "Deep review"}],
            reasoning_effort="xhigh",
            background_timeout=240,
            background_poll_interval=11,
            task="test",
            trace_id="test_gpt52_async_background_poll",
            max_budget=0,
        )

        assert result.content == "done"
        mock_apoll.assert_awaited_once()
        poll_args = mock_apoll.call_args
        assert poll_args.args[0] == "resp_async_123"
        assert poll_args.kwargs["timeout"] == 240
        assert poll_args.kwargs["poll_interval"] == 11
        assert poll_args.kwargs["request_timeout"] == 60
        assert isinstance(result.routing_trace, dict)
        assert result.routing_trace.get("background_mode") is True

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.aresponses")
    async def test_async_gpt5_tool_call_output_without_text(self, mock_aresp: MagicMock, mock_cost: MagicMock) -> None:
        """Async Responses API function_call output should map to tool_calls."""
        tool_item = SimpleNamespace(
            type="function_call",
            call_id="call_async_1",
            name="lookup",
            arguments='{"entity":"Israel"}',
        )
        mock_aresp.return_value = _mock_responses_api_response(output_text="", output=[tool_item])
        result = await acall_llm_with_tools(
            "gpt-5-mini",
            [{"role": "user", "content": "Lookup entity"}],
            [{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
            task="test",
            trace_id="test_async_gpt5_tool_call_output_without_text",
            max_budget=0,
        )
        assert result.finish_reason == "tool_calls"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "lookup"

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.acompletion")
    async def test_async_non_gpt5_still_uses_acompletion(self, mock_acomp: MagicMock, mock_cost: MagicMock) -> None:
        """Async non-GPT-5 should still use litellm.acompletion()."""
        mock_acomp.return_value = _mock_response()
        result = await acall_llm("deepseek/deepseek-chat", [{"role": "user", "content": "Hi"}], task="test", trace_id="test_async_non_gpt5", max_budget=0)
        assert result.content == "Hello!"
        mock_acomp.assert_called_once()


class TestLongThinkingBackgroundRetrieval:
    """Tests for explicit background response retrieval helpers."""

    def test_retrieve_background_response_requires_openai_key(self) -> None:
        from llm_client.client import _BackgroundRetrievalConfigurationError, _retrieve_background_response

        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            with pytest.raises(_BackgroundRetrievalConfigurationError, match="OPENAI_API_KEY is required"):
                _retrieve_background_response(
                    response_id="resp_123",
                    api_base=None,
                    request_timeout=60,
                )

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    def test_retrieve_background_response_rejects_non_openai_api_base(self) -> None:
        from llm_client.client import _BackgroundRetrievalConfigurationError, _retrieve_background_response

        with pytest.raises(_BackgroundRetrievalConfigurationError, match="OpenAI endpoints only"):
            _retrieve_background_response(
                response_id="resp_123",
                api_base="https://openrouter.ai/api/v1",
                request_timeout=60,
            )

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    @patch("openai.OpenAI")
    def test_retrieve_background_response_uses_openai_client(
        self,
        mock_openai: MagicMock,
    ) -> None:
        from llm_client.client import _retrieve_background_response

        fake_response = MagicMock()
        client = MagicMock()
        client.responses.retrieve.return_value = fake_response
        mock_openai.return_value = client

        result = _retrieve_background_response(
            response_id="resp_123",
            api_base="https://api.openai.com/v1",
            request_timeout=77,
        )

        assert result is fake_response
        mock_openai.assert_called_once()
        assert mock_openai.call_args.kwargs["api_key"] == "test-key"
        assert mock_openai.call_args.kwargs["base_url"] == "https://api.openai.com/v1"
        assert mock_openai.call_args.kwargs["timeout"] == 77
        client.responses.retrieve.assert_called_once_with("resp_123")

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    @patch("openai.AsyncOpenAI")
    async def test_aretrieve_background_response_uses_async_openai_client(
        self,
        mock_async_openai: MagicMock,
    ) -> None:
        from llm_client.client import _aretrieve_background_response

        fake_response = MagicMock()
        client = MagicMock()
        client.responses.retrieve = AsyncMock(return_value=fake_response)
        mock_async_openai.return_value = client

        result = await _aretrieve_background_response(
            response_id="resp_async_123",
            api_base="https://api.openai.com/v1",
            request_timeout=88,
        )

        assert result is fake_response
        mock_async_openai.assert_called_once()
        assert mock_async_openai.call_args.kwargs["api_key"] == "test-key"
        assert mock_async_openai.call_args.kwargs["base_url"] == "https://api.openai.com/v1"
        assert mock_async_openai.call_args.kwargs["timeout"] == 88
        client.responses.retrieve.assert_awaited_once_with("resp_async_123")

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    async def test_aretrieve_background_response_rejects_non_openai_api_base(self) -> None:
        from llm_client.client import _BackgroundRetrievalConfigurationError, _aretrieve_background_response

        with pytest.raises(_BackgroundRetrievalConfigurationError, match="OpenAI endpoints only"):
            await _aretrieve_background_response(
                response_id="resp_async_123",
                api_base="https://openrouter.ai/api/v1",
                request_timeout=60,
            )

    def test_poll_background_response_fails_fast_on_configuration_error(self) -> None:
        from llm_client.client import _BackgroundRetrievalConfigurationError, _poll_background_response

        with (
            patch(
                "llm_client.client._retrieve_background_response",
                side_effect=_BackgroundRetrievalConfigurationError("unsupported api_base"),
            ),
            patch("time.sleep") as mock_sleep,
        ):
            with pytest.raises(_BackgroundRetrievalConfigurationError, match="unsupported api_base"):
                _poll_background_response(
                    "resp_123",
                    api_base="https://openrouter.ai/api/v1",
                    poll_interval=1,
                    timeout=30,
                )
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_apoll_background_response_fails_fast_on_configuration_error(self) -> None:
        from llm_client.client import _BackgroundRetrievalConfigurationError, _apoll_background_response

        with (
            patch(
                "llm_client.client._aretrieve_background_response",
                side_effect=_BackgroundRetrievalConfigurationError("unsupported api_base"),
            ),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            with pytest.raises(_BackgroundRetrievalConfigurationError, match="unsupported api_base"):
                await _apoll_background_response(
                    "resp_async_123",
                    api_base="https://openrouter.ai/api/v1",
                    poll_interval=1,
                    timeout=30,
                )
        mock_sleep.assert_not_awaited()


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
            task="test",
            trace_id="test_retry_on_extends",
            max_budget=0,
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
            task="test",
            trace_id="test_retry_on_defaults",
            max_budget=0,
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
            task="test",
            trace_id="test_async_retry_on",
            max_budget=0,
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
            task="test",
            trace_id="test_on_retry_args",
            max_budget=0,
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
            task="test",
            trace_id="test_on_retry_success",
            max_budget=0,
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
            task="test",
            trace_id="test_async_on_retry",
            max_budget=0,
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
        result1 = call_llm("gpt-4", messages, cache=cache, task="test", trace_id="test_cache_hit", max_budget=0)
        assert result1.content == "First"
        assert mock_comp.call_count == 1

        # Second call with same args should hit cache
        mock_comp.return_value = _mock_response(content="Second")
        result2 = call_llm("gpt-4", messages, cache=cache, task="test", trace_id="test_cache_hit", max_budget=0)
        assert result2.content == "First"  # cached
        assert mock_comp.call_count == 1  # no additional call

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_cache_miss_calls_llm_and_stores(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        """Cache miss should call LLM and store the result."""
        cache = LRUCache()
        mock_comp.return_value = _mock_response(content="Fresh")
        result = call_llm("gpt-4", [{"role": "user", "content": "Hi"}], cache=cache, task="test", trace_id="test_cache_miss", max_budget=0)
        assert result.content == "Fresh"
        assert mock_comp.call_count == 1

        # Verify it's actually in the cache by calling again
        result2 = call_llm("gpt-4", [{"role": "user", "content": "Hi"}], cache=cache, task="test", trace_id="test_cache_miss", max_budget=0)
        assert result2.content == "Fresh"
        assert mock_comp.call_count == 1

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_cache_not_used_when_none(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        """Default behavior (cache=None) should always call LLM."""
        mock_comp.return_value = _mock_response()
        call_llm("gpt-4", [{"role": "user", "content": "Hi"}], task="test", trace_id="test_cache_none_1", max_budget=0)
        call_llm("gpt-4", [{"role": "user", "content": "Hi"}], task="test", trace_id="test_cache_none_2", max_budget=0)
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
        result1 = await acall_llm("gpt-4", messages, cache=cache, task="test", trace_id="test_async_cache", max_budget=0)
        assert result1.content == "Cached"
        assert mock_acomp.call_count == 1

        result2 = await acall_llm("gpt-4", messages, cache=cache, task="test", trace_id="test_async_cache", max_budget=0)
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
            task="test", trace_id="test_cache_structured",
            max_budget=0,
        )
        assert result1.name == "test"
        assert mock_client.chat.completions.create_with_completion.call_count == 1

        # Second call should hit cache
        result2, meta2 = call_llm_structured(
            "gpt-4", messages, response_model=Item, cache=cache,
            task="test", trace_id="test_cache_structured",
            max_budget=0,
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
        result = call_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=0, retry=policy, task="test", trace_id="test_policy_overrides", max_budget=0)
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
        call_llm("gpt-4", [{"role": "user", "content": "Hi"}], retry=policy, task="test", trace_id="test_policy_callback", max_budget=0)
        cb.assert_called_once()

    @patch("llm_client.client.time.sleep")
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_retry_policy_custom_backoff(self, mock_comp: MagicMock, mock_cost: MagicMock, mock_sleep: MagicMock) -> None:
        """Custom backoff function on RetryPolicy should be used."""
        mock_comp.side_effect = [Exception("timeout"), _mock_response()]
        policy = RetryPolicy(max_retries=2, backoff=fixed_backoff, base_delay=0.42)
        call_llm("gpt-4", [{"role": "user", "content": "Hi"}], retry=policy, task="test", trace_id="test_policy_backoff", max_budget=0)
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
        result = call_llm("gpt-4", [{"role": "user", "content": "Hi"}], retry=policy, task="test", trace_id="test_policy_should_retry", max_budget=0)
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
            call_llm("gpt-4", [{"role": "user", "content": "Hi"}], retry=policy, task="test", trace_id="test_policy_reject", max_budget=0)
        assert mock_comp.call_count == 1  # no retry

    @pytest.mark.asyncio
    @patch("llm_client.client.asyncio.sleep", new_callable=AsyncMock)
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.acompletion")
    async def test_retry_policy_async(self, mock_acomp: MagicMock, mock_cost: MagicMock, mock_sleep: AsyncMock) -> None:
        """RetryPolicy should work with async functions."""
        mock_acomp.side_effect = [Exception("timeout"), _mock_response()]
        policy = RetryPolicy(max_retries=3, backoff=linear_backoff, base_delay=0.1)
        result = await acall_llm("gpt-4", [{"role": "user", "content": "Hi"}], retry=policy, task="test", trace_id="test_async_policy", max_budget=0)
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
            task="test",
            trace_id="test_fallback_exhausted",
            max_budget=0,
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
            task="test",
            trace_id="test_no_fallback",
            max_budget=0,
        )
        assert result.content == "Primary OK"
        assert result.model == "gpt-4"
        assert mock_comp.call_count == 1

    @patch("llm_client.client.time.sleep")
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_duplicate_primary_fallback_is_deduplicated(
        self,
        mock_comp: MagicMock,
        mock_cost: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        """Duplicate primary in fallback list should not trigger self-fallback churn."""
        mock_comp.return_value = _mock_response(content="Primary OK")
        result = call_llm(
            "gemini/gemini-2.5-flash", [{"role": "user", "content": "Hi"}],
            fallback_models=["gemini/gemini-2.5-flash", "gemini/gemini-2.5-flash"],
            task="test",
            trace_id="test_dedup_primary_fallback",
            max_budget=0,
        )
        assert result.content == "Primary OK"
        assert result.model == "gemini/gemini-2.5-flash"
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
            task="test",
            trace_id="test_on_fallback_cb",
            max_budget=0,
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
                task="test",
                trace_id="test_all_fallbacks_fail",
                max_budget=0,
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
            task="test",
            trace_id="test_multiple_fallbacks",
            max_budget=0,
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
            task="test",
            trace_id="test_async_fallback",
            max_budget=0,
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
            task="test",
            trace_id="test_fallback_non_retryable",
            max_budget=0,
        )
        assert result.content == "Fallback OK"
        assert mock_comp.call_count == 2  # 1 attempt on primary (no retry), 1 on fallback

    @patch("llm_client.client.time.sleep")
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_fallback_populates_warnings(self, mock_comp: MagicMock, mock_cost: MagicMock, mock_sleep: MagicMock) -> None:
        """Fallback should populate result.warnings with FALLBACK message."""
        mock_comp.side_effect = [
            Exception("rate limit"),
            Exception("rate limit"),
            _mock_response(content="Fallback OK"),
        ]
        result = call_llm(
            "gpt-4", [{"role": "user", "content": "Hi"}],
            num_retries=1,
            fallback_models=["gpt-3.5-turbo"],
            task="test",
            trace_id="test_fallback_warnings",
            max_budget=0,
        )
        assert result.content == "Fallback OK"
        assert len(result.warnings) >= 2  # at least 1 retry + 1 fallback
        assert any("RETRY" in w for w in result.warnings)
        assert any("FALLBACK" in w for w in result.warnings)
        assert any("gpt-4" in w and "gpt-3.5-turbo" in w for w in result.warnings)

    @patch("llm_client.client.time.sleep")
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_retry_populates_warnings(self, mock_comp: MagicMock, mock_cost: MagicMock, mock_sleep: MagicMock) -> None:
        """Retries should populate result.warnings with RETRY messages."""
        mock_comp.side_effect = [
            Exception("rate limit"),
            _mock_response(content="OK after retry"),
        ]
        result = call_llm(
            "gpt-4", [{"role": "user", "content": "Hi"}],
            num_retries=2,
            task="test",
            trace_id="test_retry_warnings",
            max_budget=0,
        )
        assert result.content == "OK after retry"
        assert len(result.warnings) == 1
        assert "RETRY 1/3" in result.warnings[0]
        assert "gpt-4" in result.warnings[0]
        assert "rate limit" in result.warnings[0]
        assert "retry_delay_source=none" in result.warnings[0]

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_no_warnings_on_success(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        """Clean call should have empty warnings."""
        mock_comp.return_value = _mock_response(content="Clean")
        result = call_llm(
            "gpt-4", [{"role": "user", "content": "Hi"}],
            task="test",
            trace_id="test_no_warnings",
            max_budget=0,
        )
        assert result.content == "Clean"
        assert result.warnings == []


class TestExecutionModeContracts:
    """Capability-contract guards for model/kwargs compatibility."""

    def test_workspace_agent_requires_agent_model_sync(self) -> None:
        with pytest.raises(LLMCapabilityError, match="workspace_agent"):
            call_llm(
                "gpt-4",
                [{"role": "user", "content": "Hi"}],
                execution_mode="workspace_agent",
                task="test",
                trace_id="test_workspace_agent_sync",
                max_budget=0,
            )

    @pytest.mark.asyncio
    async def test_workspace_agent_requires_agent_model_async(self) -> None:
        with pytest.raises(LLMCapabilityError, match="workspace_agent"):
            await acall_llm(
                "gpt-4",
                [{"role": "user", "content": "Hi"}],
                execution_mode="workspace_agent",
                task="test",
                trace_id="test_workspace_agent_async",
                max_budget=0,
            )

    def test_workspace_agent_rejects_non_agent_fallback(self) -> None:
        with pytest.raises(LLMCapabilityError, match="Incompatible models"):
            call_llm(
                "codex",
                [{"role": "user", "content": "Hi"}],
                execution_mode="workspace_agent",
                fallback_models=["gpt-4o"],
                task="test",
                trace_id="test_workspace_agent_fallback",
                max_budget=0,
            )

    def test_agent_only_kwargs_rejected_for_non_agent(self) -> None:
        with pytest.raises(LLMCapabilityError, match="agent-only kwargs"):
            call_llm(
                "gpt-4",
                [{"role": "user", "content": "Hi"}],
                working_directory="/tmp",
                task="test",
                trace_id="test_agent_kwargs_non_agent",
                max_budget=0,
            )

    def test_workspace_tools_requires_non_agent_model(self) -> None:
        with pytest.raises(LLMCapabilityError, match="workspace_tools"):
            call_llm(
                "codex",
                [{"role": "user", "content": "Hi"}],
                execution_mode="workspace_tools",
                python_tools=[lambda x: x],
                task="test",
                trace_id="test_workspace_tools_non_agent_required",
                max_budget=0,
            )

    def test_workspace_tools_requires_tooling_inputs(self) -> None:
        with pytest.raises(LLMCapabilityError, match="requires python_tools"):
            call_llm(
                "gpt-4",
                [{"role": "user", "content": "Hi"}],
                execution_mode="workspace_tools",
                task="test",
                trace_id="test_workspace_tools_requires_inputs",
                max_budget=0,
            )

    def test_workspace_tools_rejects_agent_fallback(self) -> None:
        with pytest.raises(LLMCapabilityError, match="Incompatible models"):
            call_llm(
                "gpt-4",
                [{"role": "user", "content": "Hi"}],
                execution_mode="workspace_tools",
                python_tools=[lambda x: x],
                fallback_models=["codex"],
                task="test",
                trace_id="test_workspace_tools_fallback",
                max_budget=0,
            )


class TestResponsesAPIRouting:
    """Tests for _is_responses_api_model explicit set."""

    def test_excludes_openrouter(self) -> None:
        from llm_client.client import _is_responses_api_model
        assert _is_responses_api_model("openrouter/openai/gpt-5-mini") is False

    def test_excludes_any_provider_prefix(self) -> None:
        from llm_client.client import _is_responses_api_model
        assert _is_responses_api_model("azure/gpt-5") is False
        assert _is_responses_api_model("custom/gpt-5-mini") is False

    def test_matches_bare_gpt5(self) -> None:
        from llm_client.client import _is_responses_api_model
        assert _is_responses_api_model("gpt-5") is True
        assert _is_responses_api_model("gpt-5-mini") is True
        assert _is_responses_api_model("gpt-5-nano") is True

    def test_no_substring_match(self) -> None:
        """Ensure 'gpt-5' substring in non-GPT-5 model names doesn't match."""
        from llm_client.client import _is_responses_api_model
        assert _is_responses_api_model("gpt-5-turbo-custom") is False
        assert _is_responses_api_model("my-gpt-5") is False


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
        call_llm("gpt-4", messages, hooks=hooks, task="test", trace_id="test_before_call", max_budget=0)
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
        call_llm("gpt-4", [{"role": "user", "content": "Hi"}], hooks=hooks, task="test", trace_id="test_after_call", max_budget=0)
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
        call_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2, hooks=hooks, task="test", trace_id="test_on_error_hook", max_budget=0)
        on_error.assert_called_once()
        args = on_error.call_args[0]
        assert isinstance(args[0], Exception)
        assert args[1] == 0  # attempt number

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_hooks_not_called_when_none(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        """No errors when hooks is None (default)."""
        mock_comp.return_value = _mock_response()
        call_llm("gpt-4", [{"role": "user", "content": "Hi"}], task="test", trace_id="test_hooks_none", max_budget=0)  # should not raise

    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("llm_client.client.litellm.completion")
    def test_hooks_partial_fields(self, mock_comp: MagicMock, mock_cost: MagicMock) -> None:
        """Only set fields are called; None fields are skipped."""
        mock_comp.return_value = _mock_response()
        after = MagicMock()
        hooks = Hooks(after_call=after)  # before_call and on_error are None
        call_llm("gpt-4", [{"role": "user", "content": "Hi"}], hooks=hooks, task="test", trace_id="test_hooks_partial", max_budget=0)
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
        await acall_llm("gpt-4", [{"role": "user", "content": "Hi"}], hooks=hooks, task="test", trace_id="test_async_hooks", max_budget=0)
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
            task="test",
            trace_id="test_hooks_structured",
            max_budget=0,
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
        result1 = await acall_llm("gpt-4", messages, cache=cache, task="test", trace_id="test_async_cache_get_set", max_budget=0)
        assert result1.content == "Fresh"
        assert mock_acomp.call_count == 1

        # Second call — cache hit
        result2 = await acall_llm("gpt-4", messages, cache=cache, task="test", trace_id="test_async_cache_get_set", max_budget=0)
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
        result1 = await acall_llm("gpt-4", messages, cache=cache, task="test", trace_id="test_sync_cache_in_async", max_budget=0)
        result2 = await acall_llm("gpt-4", messages, cache=cache, task="test", trace_id="test_sync_cache_in_async", max_budget=0)
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

        stream = stream_llm("gpt-4", [{"role": "user", "content": "Hi"}], task="test", trace_id="test_stream_chunks", max_budget=0)
        collected = list(stream)
        assert collected == ["Hello", " ", "world!"]

    @patch("llm_client.client.litellm.stream_chunk_builder", return_value=None)
    @patch("llm_client.client.litellm.completion")
    def test_result_available_after_iteration(self, mock_comp: MagicMock, mock_builder: MagicMock) -> None:
        """stream.result should be available after consuming the stream."""
        chunks = _mock_stream_chunks(["Hello", "!"])
        mock_comp.return_value = iter(chunks)

        stream = stream_llm("gpt-4", [{"role": "user", "content": "Hi"}], task="test", trace_id="test_stream_result", max_budget=0)
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

        stream = stream_llm("gpt-4", [{"role": "user", "content": "Hi"}], task="test", trace_id="test_stream_before_iter", max_budget=0)
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
            stream = stream_llm("gpt-4", [{"role": "user", "content": "Hi"}], task="test", trace_id="test_stream_usage", max_budget=0)
            list(stream)
            assert stream.result.usage["total_tokens"] == 15
            assert stream.result.cost == 0.005

    @patch("llm_client.client.litellm.stream_chunk_builder", return_value=None)
    @patch("llm_client.client.litellm.completion")
    def test_stream_passes_kwargs(self, mock_comp: MagicMock, mock_builder: MagicMock) -> None:
        """Extra kwargs should be passed through to litellm."""
        chunks = _mock_stream_chunks(["Hi"])
        mock_comp.return_value = iter(chunks)

        stream = stream_llm("gpt-4", [{"role": "user", "content": "Hi"}], temperature=0.5, task="test", trace_id="test_stream_kwargs", max_budget=0)
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
        stream = stream_llm("gpt-4", [{"role": "user", "content": "Hi"}], hooks=hooks, task="test", trace_id="test_stream_hooks", max_budget=0)
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

        stream = await astream_llm("gpt-4", [{"role": "user", "content": "Hi"}], task="test", trace_id="test_astream_chunks", max_budget=0)
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

        stream = await astream_llm("gpt-4", [{"role": "user", "content": "Hi"}], task="test", trace_id="test_astream_result", max_budget=0)
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
        results = await acall_llm_batch("gpt-4", msgs_list, task="test", trace_id="test_batch_basic", max_budget=0)
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
        results = await acall_llm_batch("gpt-4", msgs_list, max_concurrent=3, task="test", trace_id="test_batch_concurrency", max_budget=0)
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
                task="test",
                trace_id="test_batch_return_exc",
                max_budget=0,
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
                task="test",
                trace_id="test_batch_raises",
                max_budget=0,
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
            task="test",
            trace_id="test_batch_on_complete",
            max_budget=0,
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
                task="test",
                trace_id="test_batch_on_error",
                max_budget=0,
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
            task="test",
            trace_id="test_batch_forwards",
            max_budget=0,
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
            task="test",
            trace_id="test_batch_sync",
            max_budget=0,
        )
        assert len(results) == 2
        assert results[0].content == "Sync OK"

    @pytest.mark.asyncio
    async def test_acall_llm_batch_empty(self) -> None:
        """Empty list returns empty list."""
        results = await acall_llm_batch("gpt-4", [], task="test", trace_id="test_batch_empty", max_budget=0)
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
            task="test",
            trace_id="test_structured_batch",
            max_budget=0,
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
        results = await acall_llm_batch("gpt-4", msgs_list, task="test", trace_id="test_batch_order", max_budget=0)
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
        stream = stream_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2, task="test", trace_id="test_stream_retry", max_budget=0)
        collected = list(stream)
        assert collected == ["Hello"]
        assert mock_comp.call_count == 2
        mock_sleep.assert_called_once()

    @patch("llm_client.client.litellm.completion")
    def test_stream_no_retry_non_retryable(self, mock_comp: MagicMock) -> None:
        """Non-retryable error raises immediately."""
        mock_comp.side_effect = Exception("invalid api key")
        with pytest.raises(Exception, match="invalid api key"):
            stream_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2, task="test", trace_id="test_stream_non_retryable", max_budget=0)
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
            task="test",
            trace_id="test_stream_fallback",
            max_budget=0,
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
        stream = await astream_llm("gpt-4", [{"role": "user", "content": "Hi"}], num_retries=2, task="test", trace_id="test_astream_retry", max_budget=0)
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
        stream = stream_llm("gpt-4", [{"role": "user", "content": "Hi"}], retry=policy, task="test", trace_id="test_stream_retry_policy", max_budget=0)
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
        stream = stream_llm_with_tools("gpt-4", [{"role": "user", "content": "Hi"}], tools, task="test", trace_id="test_stream_tools_passes", max_budget=0)
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
        stream = stream_llm_with_tools("gpt-4", [{"role": "user", "content": "Weather?"}], tools, task="test", trace_id="test_stream_tools_extract", max_budget=0)
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
        stream = await astream_llm_with_tools("gpt-4", [{"role": "user", "content": "Hi"}], tools, task="test", trace_id="test_astream_tools", max_budget=0)
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
            task="test",
            trace_id="test_gpt5_structured_resp",
            max_budget=0,
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
            task="test",
            trace_id="test_gpt5_structured_schema",
            max_budget=0,
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
            task="test",
            trace_id="test_async_gpt5_structured",
            max_budget=0,
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
            task="test",
            trace_id="test_native_schema",
            max_budget=0,
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
            task="test",
            trace_id="test_instructor_fallback",
            max_budget=0,
        )
        assert result.name == "test"
        mock_from_litellm.assert_called_once()  # instructor was used

    @patch("llm_client.client.litellm.supports_response_schema", return_value=True)
    @patch("llm_client.client.litellm.completion_cost", return_value=0.001)
    @patch("instructor.from_litellm")
    @patch("llm_client.client.litellm.completion")
    def test_structured_schema_rejection_falls_back_to_instructor(
        self,
        mock_completion: MagicMock,
        mock_from_litellm: MagicMock,
        mock_cost: MagicMock,
        mock_supports: MagicMock,
    ) -> None:
        """Provider schema rejection should fall through to instructor fallback."""

        class Item(BaseModel):
            name: str

        mock_completion.side_effect = Exception(
            "INVALID_ARGUMENT: response_schema nesting depth exceeds limit"
        )

        parsed = Item(name="fallback")
        raw_resp = _mock_response()

        mock_client = MagicMock()
        mock_client.chat.completions.create_with_completion.return_value = (parsed, raw_resp)
        mock_from_litellm.return_value = mock_client

        result, meta = call_llm_structured(
            "gemini/gemini-2.5-flash-lite",
            [{"role": "user", "content": "Extract"}],
            response_model=Item,
            task="test",
            trace_id="test_schema_rejection_fallback",
            max_budget=0,
        )

        assert result.name == "fallback"
        assert meta.model == "gemini/gemini-2.5-flash-lite"
        mock_completion.assert_called_once()
        mock_from_litellm.assert_called_once()


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
        """GPT-4o should trigger outclassed-model warning."""
        with pytest.warns(UserWarning, match="OUTCLASSED MODEL.*gpt-4o"):
            with patch("litellm.completion", return_value=_mock_response()):
                call_llm("gpt-4o", [{"role": "user", "content": "hi"}], task="test", trace_id="test_depr_gpt4o", max_budget=0)

    def test_gpt4o_mini_warns(self):
        """GPT-4o-mini should trigger its own deprecation warning."""
        with pytest.warns(DeprecationWarning, match="DEPRECATED MODEL DETECTED.*gpt-4o-mini"):
            with patch("litellm.completion", return_value=_mock_response()):
                call_llm("gpt-4o-mini", [{"role": "user", "content": "hi"}], task="test", trace_id="test_depr_gpt4o_mini", max_budget=0)

    def test_gpt4o_mini_does_not_trigger_gpt4o_pattern(self):
        """gpt-4o-mini should NOT also trigger the gpt-4o warning (exception logic)."""
        with pytest.warns(DeprecationWarning) as record:
            with patch("litellm.completion", return_value=_mock_response()):
                call_llm("gpt-4o-mini", [{"role": "user", "content": "hi"}], task="test", trace_id="test_depr_gpt4o_mini_only", max_budget=0)
        # Should only have one warning, and it should mention gpt-4o-mini specifically
        assert len(record) == 1
        assert "gpt-4o-mini" in str(record[0].message)

    def test_claude_3_haiku_warns(self):
        """Claude 3 Haiku should trigger deprecation warning."""
        with pytest.warns(DeprecationWarning, match="DEPRECATED MODEL"):
            with patch("litellm.completion", return_value=_mock_response()):
                call_llm("anthropic/claude-3-haiku-20240307", [{"role": "user", "content": "hi"}], task="test", trace_id="test_depr_claude3", max_budget=0)

    def test_gemini_15_warns(self):
        """Gemini 1.5 should trigger deprecation warning."""
        with pytest.warns(DeprecationWarning, match="DEPRECATED MODEL"):
            with patch("litellm.completion", return_value=_mock_response()):
                call_llm("gemini/gemini-1.5-flash", [{"role": "user", "content": "hi"}], task="test", trace_id="test_depr_gemini15", max_budget=0)

    def test_o1_pro_warns(self):
        """o1-pro should trigger deprecation warning."""
        with pytest.warns(DeprecationWarning, match="DEPRECATED MODEL"):
            with patch("litellm.completion", return_value=_mock_response()):
                call_llm("o1-pro", [{"role": "user", "content": "hi"}], task="test", trace_id="test_depr_o1pro", max_budget=0)

    def test_current_model_no_warning(self):
        """Current models should NOT trigger any deprecation warning."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            with patch("litellm.completion", return_value=_mock_response()):
                call_llm("anthropic/claude-sonnet-4-5-20250929", [{"role": "user", "content": "hi"}], task="test", trace_id="test_depr_current", max_budget=0)

    def test_deepseek_no_warning(self):
        """DeepSeek V3.2 should NOT trigger deprecation warning."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            with patch("litellm.completion", return_value=_mock_response()):
                call_llm("deepseek/deepseek-chat", [{"role": "user", "content": "hi"}], task="test", trace_id="test_depr_deepseek", max_budget=0)

    def test_gemini_25_no_warning(self):
        """Gemini 2.5+ should NOT trigger deprecation warning."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            with patch("litellm.completion", return_value=_mock_response()):
                call_llm("gemini/gemini-2.5-flash", [{"role": "user", "content": "hi"}], task="test", trace_id="test_depr_gemini25", max_budget=0)

    def test_warning_message_contains_outclassed_guidance(self):
        """Outclassed warning should include guidance and replacement."""
        with pytest.warns(UserWarning) as record:
            with patch("litellm.completion", return_value=_mock_response()):
                call_llm("gpt-4o", [{"role": "user", "content": "hi"}], task="test", trace_id="test_depr_stop_msg", max_budget=0)
        msg = str(record[0].message)
        assert "OUTCLASSED MODEL" in msg
        assert "Reason:" in msg
        assert "Use instead:" in msg

    def test_structured_also_warns(self):
        """call_llm_structured should also check for deprecated models."""

        class _Entity(BaseModel):
            name: str
            type: str

        with pytest.warns(UserWarning, match="OUTCLASSED MODEL"):
            with (
                patch("litellm.supports_response_schema", return_value=True),
                patch("litellm.completion", return_value=_mock_response('{"name":"x","type":"y"}')),
            ):
                call_llm_structured(
                    "gpt-4o", [{"role": "user", "content": "hi"}],
                    response_model=_Entity,
                    task="test", trace_id="test_depr_structured",
                    max_budget=0,
                )

    @pytest.mark.asyncio
    async def test_async_also_warns(self):
        """acall_llm should also check for deprecated models."""
        with pytest.warns(UserWarning, match="OUTCLASSED MODEL"):
            with patch("litellm.acompletion", new_callable=AsyncMock, return_value=_mock_response()):
                await acall_llm("gpt-4o", [{"role": "user", "content": "hi"}], task="test", trace_id="test_depr_async", max_budget=0)

    def test_mistral_large_warns(self):
        """Mistral Large should trigger deprecation warning."""
        with pytest.warns(DeprecationWarning, match="DEPRECATED MODEL"):
            with patch("litellm.completion", return_value=_mock_response()):
                call_llm("mistral/mistral-large-latest", [{"role": "user", "content": "hi"}], task="test", trace_id="test_depr_mistral", max_budget=0)
