"""LangGraph-backed approval workflow proving slice.

This module provides one concrete durable workflow: draft a summary with
`llm_client`, pause for explicit human approval, and either revise or finalize
on resume. It is intentionally not a general workflow framework. Its job is to
prove that durable workflow state can live in a LangGraph-backed layer while
LLM calls, prompt identity, budgets, and observability still flow through
`llm_client`.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, ConfigDict, Field

from llm_client.core.client import call_llm
from llm_client.core.models import get_model
from llm_client.prompts import render_prompt

logger = logging.getLogger(__name__)

DRAFT_PROMPT_REF = "shared.workflow.summary_approval_draft@1"
REVISION_PROMPT_REF = "shared.workflow.summary_approval_revision@1"
WORKFLOW_SELECTION_TASK = "synthesis"


class SummaryApprovalState(TypedDict, total=False):
    """State carried through the LangGraph summary approval workflow."""

    workflow_id: str
    trace_id: str
    source_text: str
    requested_style: str
    max_budget: float
    max_revision_rounds: int
    revision_round: int
    draft_summary: str
    final_summary: str
    approved: bool
    review_note: str
    workflow_status: str
    last_model: str


class SummaryApprovalDecision(BaseModel):
    """Validated human review payload used to resume the workflow."""

    model_config = ConfigDict(extra="forbid")

    approved: bool = Field(description="Whether the reviewer approved the current draft.")
    review_note: str | None = Field(
        default=None,
        description="Revision guidance when approval is withheld.",
    )


def workflow_config(workflow_id: str) -> dict[str, dict[str, str]]:
    """Return the LangGraph thread configuration for one workflow instance."""

    normalized = workflow_id.strip()
    if not normalized:
        raise ValueError("workflow_id must not be empty.")
    return {"configurable": {"thread_id": normalized}}


def build_summary_approval_workflow(*, checkpointer: Any | None = None) -> Any:
    """Build the concrete LangGraph summary approval workflow.

    Args:
        checkpointer: Optional LangGraph checkpointer. When omitted, an
            in-memory saver is used so checkpoint/resume works in tests and
            small local runs.

    Returns:
        Compiled LangGraph app for the approval workflow.
    """

    builder = StateGraph(SummaryApprovalState)
    builder.add_node("draft_summary", _draft_summary)
    builder.add_node("review_summary", _review_summary)
    builder.add_node("route_after_review", _route_after_review)
    builder.add_node("revise_summary", _revise_summary)
    builder.add_node("finalize_summary", _finalize_summary)
    builder.add_node("reject_summary", _reject_summary)
    builder.add_edge(START, "draft_summary")
    builder.add_edge("draft_summary", "review_summary")
    builder.add_edge("review_summary", "route_after_review")
    builder.add_edge("revise_summary", "review_summary")
    builder.add_edge("finalize_summary", END)
    builder.add_edge("reject_summary", END)
    return builder.compile(checkpointer=checkpointer or InMemorySaver())


def start_summary_approval_workflow(
    app: Any,
    *,
    workflow_id: str,
    source_text: str,
    requested_style: str = "concise",
    max_budget: float = 0.25,
    max_revision_rounds: int = 1,
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Start the concrete summary approval workflow and run until pause or end."""

    normalized_workflow_id = workflow_id.strip()
    if not normalized_workflow_id:
        raise ValueError("workflow_id must not be empty.")
    normalized_source = source_text.strip()
    if not normalized_source:
        raise ValueError("source_text must not be empty.")
    normalized_style = requested_style.strip()
    if not normalized_style:
        raise ValueError("requested_style must not be empty.")
    if max_budget < 0:
        raise ValueError("max_budget must be >= 0.")
    if max_revision_rounds < 0:
        raise ValueError("max_revision_rounds must be >= 0.")

    initial_state: SummaryApprovalState = {
        "workflow_id": normalized_workflow_id,
        "trace_id": trace_id or f"workflow.summary_approval.{normalized_workflow_id}",
        "source_text": normalized_source,
        "requested_style": normalized_style,
        "max_budget": float(max_budget),
        "max_revision_rounds": int(max_revision_rounds),
        "revision_round": 0,
        "workflow_status": "starting",
    }
    logger.info(
        "summary approval workflow start workflow_id=%s trace_id=%s max_revision_rounds=%d",
        initial_state["workflow_id"],
        initial_state["trace_id"],
        initial_state["max_revision_rounds"],
    )
    return _invoke_workflow(app, initial_state, workflow_config(normalized_workflow_id))


def resume_summary_approval_workflow(
    app: Any,
    *,
    workflow_id: str,
    approved: bool,
    review_note: str | None = None,
) -> dict[str, Any]:
    """Resume the summary approval workflow with an explicit review decision."""

    payload: dict[str, Any] = {"approved": approved}
    if review_note is not None:
        payload["review_note"] = review_note
    logger.info("summary approval workflow resume workflow_id=%s approved=%s", workflow_id, approved)
    return _invoke_workflow(app, Command(resume=payload), workflow_config(workflow_id))


def _draft_summary(state: SummaryApprovalState) -> dict[str, Any]:
    """Generate the first summary draft with `llm_client` and prompt provenance."""

    workflow_id = state["workflow_id"]
    trace_id = state["trace_id"]
    messages = render_prompt(
        prompt_ref=DRAFT_PROMPT_REF,
        source_text=state["source_text"],
        requested_style=state["requested_style"],
    )
    model = get_model(WORKFLOW_SELECTION_TASK)
    logger.info("summary approval draft workflow_id=%s trace_id=%s", workflow_id, trace_id)
    result = call_llm(
        model,
        messages,
        task="workflow_summary_draft",
        trace_id=trace_id,
        max_budget=state["max_budget"],
        prompt_ref=DRAFT_PROMPT_REF,
    )
    draft_summary = result.content.strip()
    if not draft_summary:
        raise ValueError("Draft summary node produced empty content.")
    return {
        "draft_summary": draft_summary,
        "workflow_status": "awaiting_approval",
        "last_model": result.model,
    }


def _review_summary(state: SummaryApprovalState) -> dict[str, Any]:
    """Pause for explicit human approval and validate the resume payload."""

    revision_round = int(state.get("revision_round", 0))
    max_revision_rounds = int(state["max_revision_rounds"])
    payload = {
        "workflow_id": state["workflow_id"],
        "trace_id": state["trace_id"],
        "revision_round": revision_round,
        "max_revision_rounds": max_revision_rounds,
        "draft_summary": state["draft_summary"],
        "requested_style": state["requested_style"],
    }
    logger.info(
        "summary approval pause workflow_id=%s revision_round=%d",
        state["workflow_id"],
        revision_round,
    )
    decision = SummaryApprovalDecision.model_validate(interrupt(payload))
    review_note = (decision.review_note or "").strip()
    if not decision.approved and revision_round < max_revision_rounds and not review_note:
        raise ValueError(
            "review_note is required when requesting a revision before max_revision_rounds is exhausted."
        )
    return {
        "approved": decision.approved,
        "review_note": review_note,
        "workflow_status": "approved" if decision.approved else "revision_requested",
    }


def _route_after_review(
    state: SummaryApprovalState,
) -> Command[Literal["finalize_summary", "revise_summary", "reject_summary"]]:
    """Route approved drafts to finalize and rejected drafts to revise or reject."""

    if state.get("approved"):
        return Command(goto="finalize_summary")
    if int(state.get("revision_round", 0)) >= int(state["max_revision_rounds"]):
        return Command(goto="reject_summary")
    return Command(goto="revise_summary")


def _revise_summary(state: SummaryApprovalState) -> dict[str, Any]:
    """Revise the summary draft from reviewer feedback using `llm_client`."""

    review_note = str(state.get("review_note") or "").strip()
    if not review_note:
        raise ValueError("revise_summary requires a non-empty review_note.")
    workflow_id = state["workflow_id"]
    trace_id = state["trace_id"]
    messages = render_prompt(
        prompt_ref=REVISION_PROMPT_REF,
        source_text=state["source_text"],
        requested_style=state["requested_style"],
        current_draft=state["draft_summary"],
        review_note=review_note,
    )
    model = get_model(WORKFLOW_SELECTION_TASK)
    logger.info(
        "summary approval revise workflow_id=%s trace_id=%s revision_round=%d",
        workflow_id,
        trace_id,
        int(state.get("revision_round", 0)) + 1,
    )
    result = call_llm(
        model,
        messages,
        task="workflow_summary_revision",
        trace_id=trace_id,
        max_budget=state["max_budget"],
        prompt_ref=REVISION_PROMPT_REF,
    )
    draft_summary = result.content.strip()
    if not draft_summary:
        raise ValueError("Revision node produced empty content.")
    return {
        "draft_summary": draft_summary,
        "revision_round": int(state.get("revision_round", 0)) + 1,
        "workflow_status": "awaiting_approval",
        "last_model": result.model,
    }


def _finalize_summary(state: SummaryApprovalState) -> dict[str, Any]:
    """Finalize an approved summary without any additional LLM call."""

    logger.info("summary approval finalize workflow_id=%s", state["workflow_id"])
    return {
        "final_summary": state["draft_summary"],
        "workflow_status": "approved",
    }


def _reject_summary(state: SummaryApprovalState) -> dict[str, Any]:
    """Finalize the workflow explicitly as rejected after revision budget is exhausted."""

    logger.info("summary approval reject workflow_id=%s", state["workflow_id"])
    return {
        "final_summary": state["draft_summary"],
        "workflow_status": "rejected",
    }


def _invoke_workflow(app: Any, payload: Any, config: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Invoke a compiled workflow and fail loudly if LangGraph returns a non-mapping."""

    result = app.invoke(payload, config=config)
    if not isinstance(result, dict):
        raise TypeError(f"Workflow returned non-mapping result: {type(result).__name__}")
    return dict(result)


__all__ = [
    "DRAFT_PROMPT_REF",
    "REVISION_PROMPT_REF",
    "WORKFLOW_SELECTION_TASK",
    "build_summary_approval_workflow",
    "resume_summary_approval_workflow",
    "start_summary_approval_workflow",
    "workflow_config",
]
