"""Tests for feature-flagged native Gemini routing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llm_client.client import (
    _build_result_from_gemini_native,
    _is_retryable,
    acall_llm,
    call_llm,
)
from llm_client.errors import LLMEmptyResponseError


def _mock_chat_completion_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None

    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"

    usage = MagicMock()
    usage.prompt_tokens = 3
    usage.completion_tokens = 2
    usage.total_tokens = 5

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


@patch.dict("os.environ", {"LLM_CLIENT_GEMINI_NATIVE_MODE": "on", "GEMINI_API_KEY": "test-key"}, clear=False)
@patch("llm_client.client._call_gemini_native")
@patch("llm_client.client.litellm.completion")
def test_call_llm_routes_to_native_gemini(
    mock_completion: MagicMock,
    mock_native: MagicMock,
) -> None:
    mock_native.return_value = {
        "candidates": [{
            "content": {"parts": [{"text": "native hello"}]},
            "finishReason": "STOP",
        }],
        "usageMetadata": {
            "promptTokenCount": 3,
            "candidatesTokenCount": 2,
            "totalTokenCount": 5,
        },
    }

    result = call_llm(
        "gemini/gemini-2.5-flash",
        [{"role": "user", "content": "Hi"}],
        task="test",
        trace_id="test_gemini_native_sync",
        max_budget=0,
    )

    assert result.content == "native hello"
    mock_native.assert_called_once()
    mock_completion.assert_not_called()


@patch.dict("os.environ", {"LLM_CLIENT_GEMINI_NATIVE_MODE": "on", "GEMINI_API_KEY": "test-key"}, clear=False)
@patch("llm_client.client.litellm.completion_cost", return_value=0.001)
@patch("llm_client.client._call_gemini_native")
@patch("llm_client.client.litellm.completion")
def test_call_llm_skips_native_on_unsupported_kwargs(
    mock_completion: MagicMock,
    mock_native: MagicMock,
    _mock_cost: MagicMock,
) -> None:
    mock_completion.return_value = _mock_chat_completion_response("litellm path")

    result = call_llm(
        "gemini/gemini-2.5-flash",
        [{"role": "user", "content": "Hi"}],
        response_format={"type": "json_object"},  # unsupported by native path
        task="test",
        trace_id="test_gemini_native_skip",
        max_budget=0,
    )

    assert result.content == "litellm path"
    mock_native.assert_not_called()
    mock_completion.assert_called_once()


@pytest.mark.asyncio
@patch.dict("os.environ", {"LLM_CLIENT_GEMINI_NATIVE_MODE": "on", "GEMINI_API_KEY": "test-key"}, clear=False)
@patch("llm_client.client._acall_gemini_native")
@patch("llm_client.client.litellm.acompletion")
async def test_acall_llm_routes_to_native_gemini(
    mock_acompletion: MagicMock,
    mock_native_async: MagicMock,
) -> None:
    mock_native_async.return_value = {
        "candidates": [{
            "content": {"parts": [{"text": "native async"}]},
            "finishReason": "STOP",
        }],
        "usageMetadata": {
            "promptTokenCount": 3,
            "candidatesTokenCount": 2,
            "totalTokenCount": 5,
        },
    }

    result = await acall_llm(
        "gemini/gemini-2.5-flash",
        [{"role": "user", "content": "Hi"}],
        task="test",
        trace_id="test_gemini_native_async",
        max_budget=0,
    )

    assert result.content == "native async"
    mock_native_async.assert_called_once()
    mock_acompletion.assert_not_called()


def test_build_result_from_gemini_native_parses_tool_calls() -> None:
    response = {
        "candidates": [{
            "content": {
                "parts": [{
                    "functionCall": {
                        "name": "search_docs",
                        "args": {"query": "Lady Godiva"},
                    },
                }],
            },
            "finishReason": "STOP",
        }],
        "usageMetadata": {
            "promptTokenCount": 3,
            "candidatesTokenCount": 2,
            "totalTokenCount": 5,
        },
    }

    result = _build_result_from_gemini_native(response, "gemini/gemini-2.5-flash")

    assert result.finish_reason == "tool_calls"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["function"]["name"] == "search_docs"
    assert '"query": "Lady Godiva"' in result.tool_calls[0]["function"]["arguments"]


def test_build_result_from_gemini_native_policy_block_is_non_retryable() -> None:
    response = {
        "candidates": [{
            "content": {"parts": []},
            "finishReason": "SAFETY",
            "safetyRatings": [{"category": "HARM_CATEGORY_HATE_SPEECH", "blocked": True}],
        }],
        "promptFeedback": {
            "blockReason": "SAFETY",
            "blockReasonMessage": "blocked by policy",
        },
    }

    with pytest.raises(LLMEmptyResponseError) as exc:
        _build_result_from_gemini_native(response, "gemini/gemini-2.5-flash")

    err = exc.value
    assert err.classification == "provider_policy_block"
    assert err.retryable is False
    assert _is_retryable(err) is False
    assert err.diagnostics.get("finish_reason") == "SAFETY"


def test_build_result_from_gemini_native_unknown_empty_is_retryable() -> None:
    response = {
        "candidates": [{
            "content": {"parts": []},
            "finishReason": "STOP",
        }],
    }

    with pytest.raises(LLMEmptyResponseError) as exc:
        _build_result_from_gemini_native(response, "gemini/gemini-2.5-flash")

    err = exc.value
    assert err.classification == "provider_empty_unknown"
    assert err.retryable is True
    assert _is_retryable(err) is True
