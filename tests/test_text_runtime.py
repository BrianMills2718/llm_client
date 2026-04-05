"""Focused tests for the internal text-call runtime split.

# mock-ok: validates the runtime seam against patched provider transports
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_client import LRUCache
from llm_client.execution.text_runtime import _acall_llm_impl, _call_llm_impl


def _mock_response(content: str = "Hello!") -> MagicMock:
    """Build a minimal completion response for text-runtime tests."""
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = content
    mock.choices[0].message.tool_calls = None
    mock.choices[0].message.refusal = None
    mock.choices[0].finish_reason = "stop"
    mock.usage.prompt_tokens = 10
    mock.usage.completion_tokens = 5
    mock.usage.total_tokens = 15
    return mock


@pytest.fixture(autouse=True)
def _explicit_test_runtime_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep runtime-split tests independent from ambient process policy."""
    monkeypatch.setenv("LLM_CLIENT_OPENROUTER_ROUTING", "off")
    monkeypatch.setenv("LLM_CLIENT_TIMEOUT_POLICY", "allow")


@patch("llm_client.core.client.litellm.completion_cost", return_value=0.001)
@patch("llm_client.core.client.litellm.acompletion", new_callable=AsyncMock)
def test_text_runtime_sync_preserves_cache_and_identity_contracts(
    mock_acomp: AsyncMock,
    _mock_cost: MagicMock,
) -> None:
    """Sync runtime delegates to async impl — patches acompletion, not completion."""
    cache = LRUCache()
    messages = [{"role": "user", "content": "Hi"}]
    mock_acomp.return_value = _mock_response(content="First")

    result1 = _call_llm_impl("gpt-4", messages, cache=cache, task="test", trace_id="text.runtime.sync", max_budget=0)
    result2 = _call_llm_impl("gpt-4", messages, cache=cache, task="test", trace_id="text.runtime.sync", max_budget=0)

    assert mock_acomp.call_count == 1
    assert result1.cache_hit is False
    assert result2.cache_hit is True
    assert result2.cost_source == "cache_hit"
    assert result2.requested_model == "gpt-4"
    assert result2.resolved_model == "gpt-4"
    assert result2.routing_trace is not None
    assert result2.routing_trace["attempted_models"] == ["gpt-4"]


@pytest.mark.asyncio
@patch("llm_client.core.client.litellm.completion_cost", return_value=0.001)
@patch("llm_client.core.client.litellm.acompletion", new_callable=AsyncMock)
async def test_text_runtime_async_preserves_cache_and_identity_contracts(
    mock_acompletion: AsyncMock,
    _mock_cost: MagicMock,
) -> None:
    """Direct async runtime calls should preserve the text-call return contract."""
    cache = LRUCache()
    messages = [{"role": "user", "content": "Hi"}]
    mock_acompletion.return_value = _mock_response(content="First")

    result1 = await _acall_llm_impl(
        "gpt-4",
        messages,
        cache=cache,
        task="test",
        trace_id="text.runtime.async",
        max_budget=0,
    )
    result2 = await _acall_llm_impl(
        "gpt-4",
        messages,
        cache=cache,
        task="test",
        trace_id="text.runtime.async",
        max_budget=0,
    )

    assert mock_acompletion.call_count == 1
    assert result1.cache_hit is False
    assert result2.cache_hit is True
    assert result2.cost_source == "cache_hit"
    assert result2.requested_model == "gpt-4"
    assert result2.resolved_model == "gpt-4"
    assert result2.routing_trace is not None
    assert result2.routing_trace["attempted_models"] == ["gpt-4"]
