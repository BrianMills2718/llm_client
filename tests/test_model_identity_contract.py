"""Characterization tests for additive model-identity fields."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_client import acall_llm, call_llm
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

    @pytest.mark.asyncio
    async def test_mcp_tool_loop_keeps_legacy_model_and_populates_new_fields(self) -> None:
        async def fake_agent_loop(
            model,
            messages,
            openai_tools,
            agent_result,
            executor,
            *args,
            **kwargs,
        ):
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

        assert result.model == "requested-model"
        assert result.requested_model == "requested-model"
        assert result.resolved_model == "fallback-model"
        assert result.routing_trace is not None
        assert result.routing_trace["attempted_models"] == ["requested-model", "fallback-model"]
        assert result.routing_trace["sticky_fallback"] is True

    @patch("llm_client.client.litellm.completion_cost", return_value=0.01)
    @patch("llm_client.client.litellm.completion")
    def test_explicit_config_requested_model_semantics(
        self,
        mock_completion: MagicMock,
        _mock_cost: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Env says normalize to OpenRouter; explicit config still controls
        # the public result.model semantics.
        monkeypatch.setenv("LLM_CLIENT_OPENROUTER_ROUTING", "on")
        mock_completion.return_value = _mock_response()

        result = call_llm(
            "gpt-4",
            [{"role": "user", "content": "Hi"}],
            task="test",
            trace_id="identity.requested.semantics",
            max_budget=0,
            config=ClientConfig(
                routing_policy="openrouter",
                result_model_semantics="requested",
            ),
        )

        assert result.model == "gpt-4"
        assert result.requested_model == "gpt-4"
        assert result.resolved_model == "openrouter/openai/gpt-4"
        assert result.execution_model == "openrouter/openai/gpt-4"

    @patch("llm_client.client.litellm.completion_cost", return_value=0.01)
    @patch("llm_client.client.litellm.completion")
    def test_explicit_config_resolved_model_semantics(
        self,
        mock_completion: MagicMock,
        _mock_cost: MagicMock,
    ) -> None:
        mock_completion.return_value = _mock_response()

        result = call_llm(
            "gpt-4",
            [{"role": "user", "content": "Hi"}],
            task="test",
            trace_id="identity.resolved.semantics",
            max_budget=0,
            config=ClientConfig(
                routing_policy="openrouter",
                result_model_semantics="resolved",
            ),
        )

        assert result.model == "openrouter/openai/gpt-4"
        assert result.requested_model == "gpt-4"
        assert result.resolved_model == "openrouter/openai/gpt-4"
        assert result.execution_model == "openrouter/openai/gpt-4"

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

    @patch("llm_client.client._io_log.log_foundation_event")
    @patch("llm_client.client.litellm.completion_cost", return_value=0.01)
    @patch("llm_client.client.litellm.completion")
    def test_semantics_telemetry_event_for_explicit_config(
        self,
        mock_completion: MagicMock,
        _mock_cost: MagicMock,
        mock_log_foundation_event: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_CLIENT_OPENROUTER_ROUTING", "off")
        mock_completion.return_value = _mock_response()

        call_llm(
            "gpt-4",
            [{"role": "user", "content": "Hi"}],
            task="test",
            trace_id="telemetry.explicit.config",
            max_budget=0,
            config=ClientConfig(
                routing_policy="direct",
                result_model_semantics="requested",
            ),
        )

        assert mock_log_foundation_event.called
        event = mock_log_foundation_event.call_args.kwargs["event"]
        assert event["event_type"] == "ConfigChanged"
        assert event["operation"]["name"] == "result_model_semantics_adoption"
        params = event["inputs"]["params"]
        assert params["caller"] == "call_llm"
        assert params["config_source"] == "explicit_config"
        assert params["result_model_semantics"] == "requested"

    @patch("llm_client.client._io_log.log_foundation_event")
    @patch("llm_client.client.litellm.completion_cost", return_value=0.01)
    @patch("llm_client.client.litellm.completion")
    def test_semantics_telemetry_event_for_env_default_config(
        self,
        mock_completion: MagicMock,
        _mock_cost: MagicMock,
        mock_log_foundation_event: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_CLIENT_OPENROUTER_ROUTING", "off")
        monkeypatch.setenv("LLM_CLIENT_RESULT_MODEL_SEMANTICS", "resolved")
        mock_completion.return_value = _mock_response()

        call_llm(
            "gpt-4",
            [{"role": "user", "content": "Hi"}],
            task="test",
            trace_id="telemetry.env.default",
            max_budget=0,
        )

        assert mock_log_foundation_event.called
        event = mock_log_foundation_event.call_args.kwargs["event"]
        params = event["inputs"]["params"]
        assert params["caller"] == "call_llm"
        assert params["config_source"] == "env_or_default"
        assert params["result_model_semantics"] == "resolved"
