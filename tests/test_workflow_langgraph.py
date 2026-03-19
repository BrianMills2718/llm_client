"""Tests for the LangGraph-backed approval workflow proving slice."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from llm_client.client import LLMCallResult
from llm_client.workflow_langgraph import (
    DRAFT_PROMPT_REF,
    REVISION_PROMPT_REF,
    WORKFLOW_SELECTION_TASK,
    build_summary_approval_workflow,
    resume_summary_approval_workflow,
    start_summary_approval_workflow,
    workflow_config,
)


def _result(content: str) -> LLMCallResult:
    return LLMCallResult(content=content, usage={}, cost=0.0, model="stub-model")


def test_summary_approval_workflow_revises_then_approves() -> None:
    """The proving slice should pause, revise, and resume to approval."""

    app = build_summary_approval_workflow()

    # mock-ok: The workflow logic is real LangGraph state/checkpoint behavior;
    # the external LLM call is mocked to keep the test deterministic and local.
    with patch(
        "llm_client.workflow_langgraph.call_llm",
        side_effect=[_result("Draft v1"), _result("Draft v2")],
    ) as mock_call:
        first = start_summary_approval_workflow(
            app,
            workflow_id="wf-approve",
            source_text="Alpha event led to Beta outcome.",
            requested_style="concise",
            max_budget=0.5,
            max_revision_rounds=1,
        )
        assert first["draft_summary"] == "Draft v1"
        assert first["workflow_status"] == "awaiting_approval"
        assert "__interrupt__" in first
        interrupt_payload = first["__interrupt__"][0].value
        assert interrupt_payload["draft_summary"] == "Draft v1"
        assert interrupt_payload["revision_round"] == 0

        state_after_first = app.get_state(workflow_config("wf-approve")).values
        assert state_after_first["draft_summary"] == "Draft v1"
        assert state_after_first["workflow_status"] == "awaiting_approval"

        second = resume_summary_approval_workflow(
            app,
            workflow_id="wf-approve",
            approved=False,
            review_note="Make it tighter.",
        )
        assert second["draft_summary"] == "Draft v2"
        assert second["workflow_status"] == "awaiting_approval"
        assert second["revision_round"] == 1
        assert "__interrupt__" in second
        second_interrupt_payload = second["__interrupt__"][0].value
        assert second_interrupt_payload["draft_summary"] == "Draft v2"
        assert second_interrupt_payload["revision_round"] == 1

        third = resume_summary_approval_workflow(
            app,
            workflow_id="wf-approve",
            approved=True,
        )
        assert third["final_summary"] == "Draft v2"
        assert third["workflow_status"] == "approved"
        assert third["approved"] is True

    assert mock_call.call_count == 2
    first_call = mock_call.call_args_list[0]
    assert isinstance(first_call.args[0], str)
    assert first_call.args[0]
    assert first_call.kwargs["task"] == "workflow_summary_draft"
    assert first_call.kwargs["trace_id"] == "workflow.summary_approval.wf-approve"
    assert first_call.kwargs["max_budget"] == 0.5
    assert first_call.kwargs["prompt_ref"] == DRAFT_PROMPT_REF

    second_call = mock_call.call_args_list[1]
    assert second_call.kwargs["task"] == "workflow_summary_revision"
    assert second_call.kwargs["trace_id"] == "workflow.summary_approval.wf-approve"
    assert second_call.kwargs["max_budget"] == 0.5
    assert second_call.kwargs["prompt_ref"] == REVISION_PROMPT_REF


def test_summary_approval_workflow_rejects_when_revision_budget_exhausted() -> None:
    """A rejected draft should terminate explicitly when no revisions remain."""

    app = build_summary_approval_workflow()

    # mock-ok: The workflow checkpoint/resume path is real; the LLM edge is
    # mocked to keep the proof slice deterministic.
    with patch(
        "llm_client.workflow_langgraph.call_llm",
        return_value=_result("Draft v1"),
    ) as mock_call:
        first = start_summary_approval_workflow(
            app,
            workflow_id="wf-reject",
            source_text="Gamma causes Delta.",
            requested_style="brief",
            max_budget=0.25,
            max_revision_rounds=0,
        )
        assert first["workflow_status"] == "awaiting_approval"
        final = resume_summary_approval_workflow(
            app,
            workflow_id="wf-reject",
            approved=False,
            review_note="Still not good enough.",
        )
        assert final["workflow_status"] == "rejected"
        assert final["final_summary"] == "Draft v1"

    assert mock_call.call_count == 1


def test_summary_approval_requires_feedback_for_revision_request() -> None:
    """Requesting a revision without feedback should fail loudly."""

    app = build_summary_approval_workflow()

    # mock-ok: This test targets workflow validation around resume payloads,
    # not external model behavior.
    with patch(
        "llm_client.workflow_langgraph.call_llm",
        return_value=_result("Draft v1"),
    ):
        start_summary_approval_workflow(
            app,
            workflow_id="wf-invalid",
            source_text="Epsilon text.",
            max_revision_rounds=1,
        )
        with pytest.raises(ValueError, match="review_note is required"):
            resume_summary_approval_workflow(
                app,
                workflow_id="wf-invalid",
                approved=False,
            )


def test_summary_approval_uses_task_based_model_selection() -> None:
    """The workflow should select its model via the shared task registry."""

    app = build_summary_approval_workflow()

    # mock-ok: This test isolates model-selection policy wiring without making
    # an external API call.
    with patch("llm_client.workflow_langgraph.get_model", return_value="selected-model") as mock_get_model:
        with patch(
            "llm_client.workflow_langgraph.call_llm",
            return_value=_result("Draft v1"),
        ):
            start_summary_approval_workflow(
                app,
                workflow_id="wf-policy",
                source_text="Policy wiring text.",
            )

    mock_get_model.assert_called_once_with(WORKFLOW_SELECTION_TASK)
