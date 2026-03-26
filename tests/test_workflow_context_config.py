"""Tests for the workflow module -- context and config (no langgraph required).

Ported from llm_client v2 (~/projects/archive/llm_client_v2/tests/test_workflow.py).
Builder tests that require langgraph are in test_workflow_builder.py.
"""

from __future__ import annotations

import textwrap
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llm_client.workflow.config import StageConfig, WorkflowConfig
from llm_client.workflow.context import (
    WF_MAX_BUDGET_KEY,
    WF_TASK_PREFIX_KEY,
    WF_TRACE_ID_KEY,
    WorkflowContext,
)


# ---------------------------------------------------------------------------
# WorkflowContext tests
# ---------------------------------------------------------------------------


class TestWorkflowContext:
    def test_task_derivation(self) -> None:
        ctx = WorkflowContext(
            trace_id="t-001", max_budget=5.0, task_prefix="research", stage="decompose",
        )
        assert ctx.task == "research.decompose"

    def test_task_without_stage(self) -> None:
        ctx = WorkflowContext(trace_id="t-001", max_budget=5.0, task_prefix="research")
        assert ctx.task == "research"

    def test_inject_into_state(self) -> None:
        ctx = WorkflowContext(trace_id="t-001", max_budget=5.0, task_prefix="research")
        state = ctx.inject_into_state({"query": "test"})
        assert state["query"] == "test"
        assert state[WF_TRACE_ID_KEY] == "t-001"
        assert state[WF_MAX_BUDGET_KEY] == 5.0
        assert state[WF_TASK_PREFIX_KEY] == "research"

    def test_current_recovers_from_state(self) -> None:
        state = {
            WF_TRACE_ID_KEY: "t-002",
            WF_MAX_BUDGET_KEY: 3.0,
            WF_TASK_PREFIX_KEY: "pipeline",
        }
        ctx = WorkflowContext.current(state, stage="analyze")
        assert ctx.trace_id == "t-002"
        assert ctx.max_budget == 3.0
        assert ctx.task_prefix == "pipeline"
        assert ctx.task == "pipeline.analyze"

    def test_current_raises_on_missing_context(self) -> None:
        with pytest.raises(KeyError, match="build_workflow"):
            WorkflowContext.current({"query": "test"})

    def test_call_llm_delegates_with_correct_kwargs(self) -> None:
        """Verify call_llm passes task/trace_id/max_budget from context."""
        ctx = WorkflowContext(
            trace_id="t-003", max_budget=2.0, task_prefix="test", stage="extract",
        )
        # mock-ok: testing delegation pattern, not actual LLM call
        with patch("llm_client.core.client.call_llm") as mock_call:
            mock_call.return_value = MagicMock(content="result")
            result = ctx.call_llm("gpt-5-mini", [{"role": "user", "content": "hi"}])

            mock_call.assert_called_once()
            kwargs = mock_call.call_args
            assert kwargs.kwargs["task"] == "test.extract"
            assert kwargs.kwargs["trace_id"] == "t-003"
            assert kwargs.kwargs["max_budget"] == 2.0

    def test_call_llm_structured_delegates(self) -> None:
        """Verify call_llm_structured passes contracts from context."""
        from pydantic import BaseModel

        class Output(BaseModel):
            answer: str

        ctx = WorkflowContext(
            trace_id="t-004", max_budget=1.0, task_prefix="test", stage="parse",
        )
        # mock-ok: testing delegation pattern
        with patch("llm_client.core.client.call_llm_structured") as mock_call:
            mock_call.return_value = (Output(answer="42"), MagicMock())
            parsed, meta = ctx.call_llm_structured(
                "gpt-5-mini", [{"role": "user", "content": "hi"}], Output,
            )

            mock_call.assert_called_once()
            kwargs = mock_call.call_args
            assert kwargs.kwargs["task"] == "test.parse"
            assert kwargs.kwargs["trace_id"] == "t-004"
            assert parsed.answer == "42"

    def test_call_llm_allows_kwarg_override(self) -> None:
        """Callers can override task/trace_id if needed."""
        ctx = WorkflowContext(
            trace_id="t-005", max_budget=1.0, task_prefix="test", stage="s1",
        )
        # mock-ok: testing override behavior
        with patch("llm_client.core.client.call_llm") as mock_call:
            mock_call.return_value = MagicMock()
            ctx.call_llm("m", [{}], task="custom_task", trace_id="custom_trace")

            kwargs = mock_call.call_args.kwargs
            assert kwargs["task"] == "custom_task"
            assert kwargs["trace_id"] == "custom_trace"


# ---------------------------------------------------------------------------
# WorkflowConfig tests
# ---------------------------------------------------------------------------


class TestWorkflowConfig:
    def test_from_dict(self) -> None:
        cfg = WorkflowConfig.from_dict({
            "task_prefix": "research",
            "max_budget": 5.0,
            "default_model": "gemini/gemini-2.5-flash-lite",
            "stages": {
                "decompose": {"model": "gemini/gemini-3-flash"},
                "synthesize": {},
            },
        })
        assert cfg.task_prefix == "research"
        assert cfg.max_budget == 5.0
        assert cfg.model_for_stage("decompose") == "gemini/gemini-3-flash"
        assert cfg.model_for_stage("synthesize") == "gemini/gemini-2.5-flash-lite"
        assert cfg.model_for_stage("unknown") == "gemini/gemini-2.5-flash-lite"

    def test_from_yaml(self, tmp_path: Any) -> None:
        yaml_content = textwrap.dedent("""\
            task_prefix: pipeline
            max_budget: 3.0
            default_model: gpt-5-mini
            stages:
              step_a:
                model: gemini/gemini-3-flash
                retry:
                  max_retries: 3
              step_b:
                fallback_models:
                  - gpt-5-mini
                  - gemini/gemini-2.5-flash
        """)
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content)

        cfg = WorkflowConfig.from_yaml(config_file)
        assert cfg.task_prefix == "pipeline"
        assert cfg.stage("step_a").retry.max_retries == 3
        assert cfg.stage("step_b").fallback_models == ["gpt-5-mini", "gemini/gemini-2.5-flash"]

    def test_from_yaml_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            WorkflowConfig.from_yaml("/nonexistent/path.yaml")

    def test_stage_returns_defaults_for_unknown(self) -> None:
        cfg = WorkflowConfig.from_dict({"stages": {}})
        stage = cfg.stage("anything")
        assert stage.model is None
        assert stage.retry.max_retries == 2
