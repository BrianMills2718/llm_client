"""Focused tests for the internal structured-call runtime split.

# mock-ok: validates the runtime seam against patched provider transports
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from llm_client import LRUCache
from llm_client.core.errors import LLMCapabilityError
from llm_client.execution.structured_runtime import _acall_llm_structured_impl, _call_llm_structured_impl


class _City(BaseModel):
    """Minimal schema used to exercise the structured runtime seam."""

    name: str


def _mock_structured_response(content: str = '{"name":"Tokyo"}') -> MagicMock:
    """Build a minimal structured completion response."""
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = content
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
@patch("llm_client.core.client.litellm.supports_response_schema", return_value=True)
@patch("llm_client.core.client.litellm.completion")
def test_structured_runtime_sync_preserves_cache_and_identity_contracts(
    mock_comp: MagicMock,
    _mock_supports_schema: MagicMock,
    _mock_cost: MagicMock,
) -> None:
    """Direct sync runtime calls should preserve the structured-call return contract."""
    cache = LRUCache()
    messages = [{"role": "user", "content": "Name a city"}]
    mock_comp.return_value = _mock_structured_response()

    parsed1, meta1 = _call_llm_structured_impl(
        "gpt-4",
        messages,
        _City,
        cache=cache,
        task="test",
        trace_id="structured.runtime.sync",
        max_budget=0,
    )
    parsed2, meta2 = _call_llm_structured_impl(
        "gpt-4",
        messages,
        _City,
        cache=cache,
        task="test",
        trace_id="structured.runtime.sync",
        max_budget=0,
    )

    assert mock_comp.call_count == 1
    assert parsed1.name == "Tokyo"
    assert parsed2.name == "Tokyo"
    assert meta1.cache_hit is False
    assert meta2.cache_hit is True
    assert meta2.cost_source == "cache_hit"
    assert meta2.requested_model == "gpt-4"
    assert meta2.resolved_model == "gpt-4"
    assert meta2.routing_trace is not None
    assert meta2.routing_trace["attempted_models"] == ["gpt-4"]


@pytest.mark.asyncio
@patch("llm_client.core.client.litellm.completion_cost", return_value=0.001)
@patch("llm_client.core.client.litellm.supports_response_schema", return_value=True)
@patch("llm_client.core.client.litellm.acompletion", new_callable=AsyncMock)
async def test_structured_runtime_async_preserves_cache_and_identity_contracts(
    mock_acompletion: AsyncMock,
    _mock_supports_schema: MagicMock,
    _mock_cost: MagicMock,
) -> None:
    """Direct async runtime calls should preserve the structured-call return contract."""
    cache = LRUCache()
    messages = [{"role": "user", "content": "Name a city"}]
    mock_acompletion.return_value = _mock_structured_response()

    parsed1, meta1 = await _acall_llm_structured_impl(
        "gpt-4",
        messages,
        _City,
        cache=cache,
        task="test",
        trace_id="structured.runtime.async",
        max_budget=0,
    )
    parsed2, meta2 = await _acall_llm_structured_impl(
        "gpt-4",
        messages,
        _City,
        cache=cache,
        task="test",
        trace_id="structured.runtime.async",
        max_budget=0,
    )

    assert mock_acompletion.call_count == 1
    assert parsed1.name == "Tokyo"
    assert parsed2.name == "Tokyo"
    assert meta1.cache_hit is False
    assert meta2.cache_hit is True
    assert meta2.cost_source == "cache_hit"
    assert meta2.requested_model == "gpt-4"
    assert meta2.resolved_model == "gpt-4"
    assert meta2.routing_trace is not None
    assert meta2.routing_trace["attempted_models"] == ["gpt-4"]


@patch("llm_client.core.client.litellm.supports_response_schema", return_value=True)
@patch(
    "llm_client.core.client.litellm.completion",
    side_effect=RuntimeError(
        "Invalid schema for response_format 'City': extra required key 'name' "
        "(invalid_json_schema)"
    ),
)
def test_structured_runtime_sync_raises_capability_error_for_gpt5_schema_rejection(
    _mock_comp: MagicMock,
    _mock_supports_schema: MagicMock,
) -> None:
    """Provider-side GPT-5 schema rejection should fail loudly as a capability error."""
    messages = [{"role": "user", "content": "Name a city"}]

    with pytest.raises(LLMCapabilityError, match="provider rejected structured JSON-schema output"):
        _call_llm_structured_impl(
            "openai/gpt-5-mini",
            messages,
            _City,
            task="test",
            trace_id="structured.runtime.sync.gpt5_schema",
            max_budget=0,
        )


@pytest.mark.asyncio
@patch(
    "llm_client.core.client.litellm.aresponses",
    new_callable=AsyncMock,
    side_effect=RuntimeError(
        "Invalid schema for response_format 'City': extra required key 'name' "
        "(invalid_json_schema)"
    ),
)
async def test_structured_runtime_async_raises_capability_error_for_gpt5_schema_rejection(
    _mock_aresponses: AsyncMock,
) -> None:
    """Bare GPT-5 structured responses should surface schema rejection as capability errors."""
    messages = [{"role": "user", "content": "Name a city"}]

    with pytest.raises(LLMCapabilityError, match="provider rejected structured JSON-schema output"):
        await _acall_llm_structured_impl(
            "gpt-5-mini",
            messages,
            _City,
            task="test",
            trace_id="structured.runtime.async.gpt5_schema",
            max_budget=0,
        )
