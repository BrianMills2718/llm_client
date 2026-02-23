"""Characterization tests for additive model-identity fields."""

from __future__ import annotations

from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from llm_client import (
    LLMCallResult,
    acall_llm,
    acall_llm_structured,
    astream_llm,
    call_llm,
    call_llm_structured,
    stream_llm,
)
from llm_client.config import ClientConfig
from llm_client.mcp_agent import _acall_with_tools


def _mock_response(
    content: str = "Hello!",
    *,
    finish_reason: str = "stop",
) -> MagicMock:
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = content
    mock.choices[0].message.tool_calls = None
    mock.choices[0].finish_reason = finish_reason
    mock.usage.prompt_tokens = 10
    mock.usage.completion_tokens = 5
    mock.usage.total_tokens = 15
    return mock


def _mock_responses_api_output(raw_json: str) -> MagicMock:
    response = MagicMock()
    response.output_text = raw_json
    response.output = []
    response.status = "completed"
    response.usage.input_tokens = 10
    response.usage.output_tokens = 5
    response.usage.total_tokens = 15
    response.usage.input_tokens_details.cached_tokens = 0
    response.usage.cost = None
    return response


def _mock_stream_chunks(texts: list[str]) -> list[MagicMock]:
    chunks = []
    for text in texts:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = text
        chunks.append(chunk)
    return chunks


class _StructuredPayload(BaseModel):
    message: str


class TestModelIdentityContract:
    @patch("llm_client.client.litellm.completion_cost", return_value=0.01)
    @patch("llm_client.client.litellm.completion")
    def test_sync_identity_fields_with_explicit_routing_off(
        self,
        mock_completion: MagicMock,
        _mock_cost: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_CLIENT_OPENROUTER_ROUTING", "off")
        mock_completion.return_value = _mock_response()

        result = call_llm(
            "gpt-4",
            [{"role": "user", "content": "Hi"}],
            task="test",
            trace_id="identity.sync.off",
            max_budget=0,
        )

        assert result.model == "gpt-4"
        assert result.requested_model == "gpt-4"
        assert result.resolved_model == "gpt-4"
        assert result.routing_trace is not None
        assert result.routing_trace["routing_policy"] == "openrouter_off"
        assert result.routing_trace["attempted_models"] == ["gpt-4"]

    @patch("llm_client.client.litellm.completion_cost", return_value=0.01)
    @patch("llm_client.client.litellm.completion")
    def test_sync_identity_fields_with_explicit_routing_on(
        self,
        mock_completion: MagicMock,
        _mock_cost: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_CLIENT_OPENROUTER_ROUTING", "on")
        mock_completion.return_value = _mock_response()

        result = call_llm(
            "gpt-4",
            [{"role": "user", "content": "Hi"}],
            task="test",
            trace_id="identity.sync.on",
            max_budget=0,
        )

        assert result.model == "openrouter/openai/gpt-4"
        assert result.requested_model == "gpt-4"
        assert result.resolved_model == "openrouter/openai/gpt-4"
        assert result.routing_trace is not None
        assert result.routing_trace["routing_policy"] == "openrouter_on"
        assert result.routing_trace["normalized_from"] == "gpt-4"
        assert result.routing_trace["normalized_to"] == "openrouter/openai/gpt-4"

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.01)
    @patch("llm_client.client.litellm.acompletion", new_callable=AsyncMock)
    async def test_async_identity_fields_with_explicit_routing_off(
        self,
        mock_acompletion: AsyncMock,
        _mock_cost: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_CLIENT_OPENROUTER_ROUTING", "off")
        mock_acompletion.return_value = _mock_response()

        result = await acall_llm(
            "gpt-4",
            [{"role": "user", "content": "Hi"}],
            task="test",
            trace_id="identity.async.off",
            max_budget=0,
        )

        assert result.model == "gpt-4"
        assert result.requested_model == "gpt-4"
        assert result.resolved_model == "gpt-4"
        assert result.routing_trace is not None
        assert result.routing_trace["routing_policy"] == "openrouter_off"
        assert result.routing_trace["attempted_models"] == ["gpt-4"]

    @patch("llm_client.client.litellm.completion_cost", return_value=0.01)
    @patch("llm_client.client.litellm.responses")
    def test_structured_identity_fields_with_explicit_routing_off(
        self,
        mock_responses: MagicMock,
        _mock_cost: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_CLIENT_OPENROUTER_ROUTING", "off")
        mock_responses.return_value = _mock_responses_api_output('{"message":"Hi"}')

        parsed, result = call_llm_structured(
            "gpt-5",
            [{"role": "user", "content": "Hi"}],
            _StructuredPayload,
            task="test",
            trace_id="identity.structured.sync.off",
            max_budget=0,
        )

        assert parsed.message == "Hi"
        assert result.model == "gpt-5"
        assert result.requested_model == "gpt-5"
        assert result.resolved_model == "gpt-5"
        assert result.routing_trace is not None
        assert result.routing_trace["routing_policy"] == "openrouter_off"
        assert result.routing_trace["attempted_models"] == ["gpt-5"]

    @patch("llm_client.client.litellm.completion_cost", return_value=0.01)
    @patch("llm_client.client.litellm.supports_response_schema", return_value=True)
    @patch("llm_client.client.litellm.completion")
    def test_structured_identity_fields_with_explicit_routing_on(
        self,
        mock_completion: MagicMock,
        _mock_supports_schema: MagicMock,
        _mock_cost: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_CLIENT_OPENROUTER_ROUTING", "on")
        mock_completion.return_value = _mock_response(content='{"message":"Hi"}')

        parsed, result = call_llm_structured(
            "gpt-4",
            [{"role": "user", "content": "Hi"}],
            _StructuredPayload,
            task="test",
            trace_id="identity.structured.sync.on",
            max_budget=0,
        )

        assert parsed.message == "Hi"
        assert result.model == "openrouter/openai/gpt-4"
        assert result.requested_model == "gpt-4"
        assert result.resolved_model == "openrouter/openai/gpt-4"
        assert result.routing_trace is not None
        assert result.routing_trace["routing_policy"] == "openrouter_on"
        assert result.routing_trace["normalized_from"] == "gpt-4"
        assert result.routing_trace["normalized_to"] == "openrouter/openai/gpt-4"

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.01)
    @patch("llm_client.client.litellm.aresponses", new_callable=AsyncMock)
    async def test_async_structured_identity_fields_with_explicit_routing_off(
        self,
        mock_aresponses: AsyncMock,
        _mock_cost: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_CLIENT_OPENROUTER_ROUTING", "off")
        mock_aresponses.return_value = _mock_responses_api_output('{"message":"Hi"}')

        parsed, result = await acall_llm_structured(
            "gpt-5",
            [{"role": "user", "content": "Hi"}],
            _StructuredPayload,
            task="test",
            trace_id="identity.structured.async.off",
            max_budget=0,
        )

        assert parsed.message == "Hi"
        assert result.model == "gpt-5"
        assert result.requested_model == "gpt-5"
        assert result.resolved_model == "gpt-5"
        assert result.routing_trace is not None
        assert result.routing_trace["routing_policy"] == "openrouter_off"
        assert result.routing_trace["attempted_models"] == ["gpt-5"]

    @patch("llm_client.client.litellm.completion_cost", return_value=0.01)
    @patch("llm_client.client.litellm.supports_response_schema", return_value=True)
    @patch("llm_client.client.litellm.completion")
    def test_structured_fallback_chain_records_attempted_models(
        self,
        mock_completion: MagicMock,
        _mock_supports_schema: MagicMock,
        _mock_cost: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_CLIENT_OPENROUTER_ROUTING", "on")
        mock_completion.side_effect = [
            RuntimeError("primary failed"),
            _mock_response(content='{"message":"Recovered"}'),
        ]

        parsed, result = call_llm_structured(
            "gpt-4",
            [{"role": "user", "content": "Hi"}],
            _StructuredPayload,
            num_retries=0,
            fallback_models=["gpt-3.5-turbo"],
            task="test",
            trace_id="identity.structured.sync.fallback",
            max_budget=0,
        )

        assert parsed.message == "Recovered"
        assert result.model == "openrouter/openai/gpt-3.5-turbo"
        assert result.requested_model == "gpt-4"
        assert result.resolved_model == "openrouter/openai/gpt-3.5-turbo"
        assert result.routing_trace is not None
        assert result.routing_trace["attempted_models"] == [
            "openrouter/openai/gpt-4",
            "openrouter/openai/gpt-3.5-turbo",
        ]
        assert result.routing_trace["normalized_from"] == "gpt-4"
        assert result.routing_trace["normalized_to"] == "openrouter/openai/gpt-4"
        assert any(w.startswith("FALLBACK:") for w in (result.warnings or []))

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.completion_cost", return_value=0.01)
    @patch("llm_client.client.litellm.supports_response_schema", return_value=True)
    @patch("llm_client.client.litellm.acompletion", new_callable=AsyncMock)
    async def test_async_structured_fallback_chain_records_attempted_models(
        self,
        mock_acompletion: AsyncMock,
        _mock_supports_schema: MagicMock,
        _mock_cost: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_CLIENT_OPENROUTER_ROUTING", "on")
        mock_acompletion.side_effect = [
            RuntimeError("primary failed"),
            _mock_response(content='{"message":"Recovered"}'),
        ]

        parsed, result = await acall_llm_structured(
            "gpt-4",
            [{"role": "user", "content": "Hi"}],
            _StructuredPayload,
            num_retries=0,
            fallback_models=["gpt-3.5-turbo"],
            task="test",
            trace_id="identity.structured.async.fallback",
            max_budget=0,
        )

        assert parsed.message == "Recovered"
        assert result.model == "openrouter/openai/gpt-3.5-turbo"
        assert result.requested_model == "gpt-4"
        assert result.resolved_model == "openrouter/openai/gpt-3.5-turbo"
        assert result.routing_trace is not None
        assert result.routing_trace["attempted_models"] == [
            "openrouter/openai/gpt-4",
            "openrouter/openai/gpt-3.5-turbo",
        ]
        assert result.routing_trace["normalized_from"] == "gpt-4"
        assert result.routing_trace["normalized_to"] == "openrouter/openai/gpt-4"
        assert any(w.startswith("FALLBACK:") for w in (result.warnings or []))

    @patch("llm_client.client.litellm.stream_chunk_builder", return_value=None)
    @patch("llm_client.client.litellm.completion")
    def test_stream_identity_fields_with_explicit_routing_off(
        self,
        mock_completion: MagicMock,
        _mock_stream_builder: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_CLIENT_OPENROUTER_ROUTING", "off")
        mock_completion.return_value = iter(_mock_stream_chunks(["Hello", "!"]))

        stream = stream_llm(
            "gpt-4",
            [{"role": "user", "content": "Hi"}],
            task="test",
            trace_id="identity.stream.sync.off",
            max_budget=0,
        )
        list(stream)
        result = stream.result

        assert result.model == "gpt-4"
        assert result.requested_model == "gpt-4"
        assert result.resolved_model == "gpt-4"
        assert result.routing_trace is not None
        assert result.routing_trace["routing_policy"] == "openrouter_off"
        assert result.routing_trace["attempted_models"] == ["gpt-4"]

    @pytest.mark.asyncio
    @patch("llm_client.client.litellm.stream_chunk_builder", return_value=None)
    @patch("llm_client.client.litellm.acompletion")
    async def test_async_stream_identity_fields_with_explicit_routing_on(
        self,
        mock_acompletion: AsyncMock,
        _mock_stream_builder: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_CLIENT_OPENROUTER_ROUTING", "on")

        async def _async_iter() -> AsyncGenerator[MagicMock, None]:
            for chunk in _mock_stream_chunks(["Hello", "!"]):
                yield chunk

        mock_acompletion.return_value = _async_iter()

        stream = await astream_llm(
            "gpt-4",
            [{"role": "user", "content": "Hi"}],
            task="test",
            trace_id="identity.stream.async.on",
            max_budget=0,
        )
        async for _ in stream:
            pass
        result = stream.result

        assert result.model == "openrouter/openai/gpt-4"
        assert result.requested_model == "gpt-4"
        assert result.resolved_model == "openrouter/openai/gpt-4"
        assert result.routing_trace is not None
        assert result.routing_trace["routing_policy"] == "openrouter_on"
        assert result.routing_trace["normalized_from"] == "gpt-4"
        assert result.routing_trace["normalized_to"] == "openrouter/openai/gpt-4"

    @pytest.mark.asyncio
    async def test_mcp_tool_loop_sets_executed_model_and_identity_fields(self) -> None:
        async def fake_agent_loop(
            model: str,
            messages: list[dict[str, Any]],
            openai_tools: list[dict[str, Any]],
            agent_result: Any,
            executor: Any,
            *args: Any,
            **kwargs: Any,
        ) -> tuple[str, str]:
            agent_result.metadata["total_cost"] = 0.5
            agent_result.metadata["resolved_model"] = "fallback-model"
            agent_result.metadata["attempted_models"] = [model, "fallback-model"]
            agent_result.metadata["sticky_fallback"] = True
            agent_result.warnings.append(
                f"STICKY_FALLBACK: {model} failed, using fallback-model for remaining turns"
            )
            return "done", "stop"

        with (
            patch(
                "llm_client.tool_utils.prepare_direct_tools",
                return_value=(
                    {"noop": lambda: "ok"},
                    [{"type": "function", "function": {"name": "noop"}}],
                ),
            ),
            patch("llm_client.mcp_agent._agent_loop", side_effect=fake_agent_loop),
        ):
            result = await _acall_with_tools(
                "requested-model",
                [{"role": "user", "content": "hi"}],
                python_tools=[object()],
            )

        assert result.model == "fallback-model"
        assert result.requested_model == "requested-model"
        assert result.resolved_model == "fallback-model"
        assert result.routing_trace is not None
        assert result.routing_trace["attempted_models"] == ["requested-model", "fallback-model"]
        assert result.routing_trace["sticky_fallback"] is True

    @patch("llm_client.models.supports_tool_calling", return_value=True)
    @patch("llm_client.mcp_agent._acall_with_tools")
    def test_sync_call_llm_tool_loop_preserves_agent_routing_trace(
        self,
        mock_tool_loop: MagicMock,
        _mock_supports_tools: MagicMock,
    ) -> None:
        mock_tool_loop.return_value = LLMCallResult(
            content="done",
            usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            cost=0.0,
            model="fallback-model",
            finish_reason="stop",
            raw_response={"ok": True},
            warnings=["STICKY_FALLBACK: gpt-4 -> fallback-model"],
        )
        mock_tool_loop.return_value.requested_model = "gpt-4"
        mock_tool_loop.return_value.resolved_model = "fallback-model"
        mock_tool_loop.return_value.routing_trace = {
            "attempted_models": ["gpt-4", "fallback-model"],
            "sticky_fallback": True,
        }

        result = call_llm(
            "gpt-4",
            [{"role": "user", "content": "Hi"}],
            python_tools=[object()],
            task="test",
            trace_id="identity.sync.tools.loop.trace",
            max_budget=0,
            config=ClientConfig(routing_policy="direct"),
        )

        assert result.model == "fallback-model"
        assert result.requested_model == "gpt-4"
        assert result.resolved_model == "fallback-model"
        assert result.routing_trace is not None
        assert result.routing_trace["attempted_models"] == ["gpt-4", "fallback-model"]
        assert result.routing_trace["sticky_fallback"] is True
        assert result.routing_trace["selected_model"] == "fallback-model"

    @pytest.mark.asyncio
    @patch("llm_client.models.supports_tool_calling", return_value=True)
    @patch("llm_client.mcp_agent._acall_with_tools")
    async def test_async_call_llm_tool_loop_preserves_agent_routing_trace(
        self,
        mock_tool_loop: AsyncMock,
        _mock_supports_tools: MagicMock,
    ) -> None:
        mock_tool_loop.return_value = LLMCallResult(
            content="done",
            usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            cost=0.0,
            model="fallback-model",
            finish_reason="stop",
            raw_response={"ok": True},
            warnings=["STICKY_FALLBACK: gpt-4 -> fallback-model"],
        )
        mock_tool_loop.return_value.requested_model = "gpt-4"
        mock_tool_loop.return_value.resolved_model = "fallback-model"
        mock_tool_loop.return_value.routing_trace = {
            "attempted_models": ["gpt-4", "fallback-model"],
            "sticky_fallback": True,
        }

        result = await acall_llm(
            "gpt-4",
            [{"role": "user", "content": "Hi"}],
            python_tools=[object()],
            task="test",
            trace_id="identity.async.tools.loop.trace",
            max_budget=0,
            config=ClientConfig(routing_policy="direct"),
        )

        assert result.model == "fallback-model"
        assert result.requested_model == "gpt-4"
        assert result.resolved_model == "fallback-model"
        assert result.routing_trace is not None
        assert result.routing_trace["attempted_models"] == ["gpt-4", "fallback-model"]
        assert result.routing_trace["sticky_fallback"] is True
        assert result.routing_trace["selected_model"] == "fallback-model"

    @patch("llm_client.mcp_agent._acall_with_mcp")
    def test_sync_call_llm_mcp_loop_preserves_agent_routing_trace(
        self,
        mock_mcp_loop: MagicMock,
    ) -> None:
        mock_mcp_loop.return_value = LLMCallResult(
            content="done",
            usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            cost=0.0,
            model="fallback-model",
            finish_reason="stop",
            raw_response={"ok": True},
            warnings=["STICKY_FALLBACK: gpt-4 -> fallback-model"],
        )
        mock_mcp_loop.return_value.requested_model = "gpt-4"
        mock_mcp_loop.return_value.resolved_model = "fallback-model"
        mock_mcp_loop.return_value.routing_trace = {
            "attempted_models": ["gpt-4", "fallback-model"],
            "sticky_fallback": True,
        }

        result = call_llm(
            "gpt-4",
            [{"role": "user", "content": "Hi"}],
            mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
            task="test",
            trace_id="identity.sync.mcp.loop.trace",
            max_budget=0,
            config=ClientConfig(routing_policy="direct"),
        )

        assert result.model == "fallback-model"
        assert result.requested_model == "gpt-4"
        assert result.resolved_model == "fallback-model"
        assert result.routing_trace is not None
        assert result.routing_trace["attempted_models"] == ["gpt-4", "fallback-model"]
        assert result.routing_trace["sticky_fallback"] is True
        assert result.routing_trace["selected_model"] == "fallback-model"

    @pytest.mark.asyncio
    @patch("llm_client.mcp_agent._acall_with_mcp")
    async def test_async_call_llm_mcp_loop_preserves_agent_routing_trace(
        self,
        mock_mcp_loop: AsyncMock,
    ) -> None:
        mock_mcp_loop.return_value = LLMCallResult(
            content="done",
            usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            cost=0.0,
            model="fallback-model",
            finish_reason="stop",
            raw_response={"ok": True},
            warnings=["STICKY_FALLBACK: gpt-4 -> fallback-model"],
        )
        mock_mcp_loop.return_value.requested_model = "gpt-4"
        mock_mcp_loop.return_value.resolved_model = "fallback-model"
        mock_mcp_loop.return_value.routing_trace = {
            "attempted_models": ["gpt-4", "fallback-model"],
            "sticky_fallback": True,
        }

        result = await acall_llm(
            "gpt-4",
            [{"role": "user", "content": "Hi"}],
            mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
            task="test",
            trace_id="identity.async.mcp.loop.trace",
            max_budget=0,
            config=ClientConfig(routing_policy="direct"),
        )

        assert result.model == "fallback-model"
        assert result.requested_model == "gpt-4"
        assert result.resolved_model == "fallback-model"
        assert result.routing_trace is not None
        assert result.routing_trace["attempted_models"] == ["gpt-4", "fallback-model"]
        assert result.routing_trace["sticky_fallback"] is True
        assert result.routing_trace["selected_model"] == "fallback-model"

    @patch("llm_client.client.litellm.completion_cost", return_value=0.01)
    @patch("llm_client.client.litellm.completion")
    def test_explicit_config_still_controls_routing_policy(
        self,
        mock_completion: MagicMock,
        _mock_cost: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Env says normalize to OpenRouter; explicit config still controls routing.
        monkeypatch.setenv("LLM_CLIENT_OPENROUTER_ROUTING", "on")
        mock_completion.return_value = _mock_response()

        result = call_llm(
            "gpt-4",
            [{"role": "user", "content": "Hi"}],
            task="test",
            trace_id="identity.explicit.routing",
            max_budget=0,
            config=ClientConfig(
                routing_policy="direct",
            ),
        )

        assert result.model == "gpt-4"
        assert result.requested_model == "gpt-4"
        assert result.resolved_model == "gpt-4"
        assert result.execution_model == "gpt-4"

    @patch("llm_client.client.litellm.completion_cost", return_value=0.01)
    @patch("llm_client.client.litellm.completion")
    def test_warning_records_include_stable_model_code(
        self,
        mock_completion: MagicMock,
        _mock_cost: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_CLIENT_OPENROUTER_ROUTING", "off")
        mock_completion.return_value = _mock_response()

        result = call_llm(
            "gpt-4o",
            [{"role": "user", "content": "Hi"}],
            task="test",
            trace_id="identity.warning.codes",
            max_budget=0,
        )

        assert any(
            rec.get("code") == "LLMC_WARN_MODEL_OUTCLASSED"
            for rec in result.warning_records
        )
