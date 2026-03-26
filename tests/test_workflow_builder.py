"""Tests for the workflow builder (requires langgraph).

Ported from llm_client v2 (~/projects/archive/llm_client_v2/tests/test_workflow.py).
Context and config tests that don't require langgraph are in test_workflow_context_config.py.
"""

from __future__ import annotations

from typing import Any, TypedDict

import pytest

try:
    import langgraph  # noqa: F401
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False

pytestmark = pytest.mark.skipif(not HAS_LANGGRAPH, reason="langgraph not installed")

from llm_client.workflow.config import WorkflowConfig
from llm_client.workflow.context import WorkflowContext


class SimpleState(TypedDict, total=False):
    query: str
    result: str
    _wf_trace_id: str
    _wf_max_budget: float
    _wf_task_prefix: str
    _wf_current_stage: str


class TestBuildWorkflow:
    def test_build_and_invoke_simple_pipeline(self) -> None:
        """Two-node pipeline: upper -> reverse."""
        from llm_client.workflow.builder import build_workflow

        def upper(state: dict[str, Any]) -> dict[str, Any]:
            return {"result": state["query"].upper()}

        def reverse(state: dict[str, Any]) -> dict[str, Any]:
            return {"result": state["result"][::-1]}

        config = WorkflowConfig.from_dict({"task_prefix": "test"})
        app = build_workflow(
            state_schema=SimpleState,
            config=config,
            nodes={"upper": upper, "reverse": reverse},
            edges=[("upper", "reverse")],
            finish_points=["reverse"],
        )

        ctx = WorkflowContext(trace_id="t-build", max_budget=0.0, task_prefix="test")
        initial = ctx.inject_into_state({"query": "hello"})
        result = app.invoke(initial, config={"configurable": {"thread_id": "t1"}})
        assert result["result"] == "OLLEH"

    def test_conditional_edges(self) -> None:
        """Router sends to different nodes based on state."""
        from llm_client.workflow.builder import build_workflow

        def check(state: dict[str, Any]) -> dict[str, Any]:
            return {"result": "checked"}

        def path_a(state: dict[str, Any]) -> dict[str, Any]:
            return {"result": "A"}

        def path_b(state: dict[str, Any]) -> dict[str, Any]:
            return {"result": "B"}

        def router(state: dict[str, Any]) -> str:
            return "path_a" if state["query"] == "go_a" else "path_b"

        config = WorkflowConfig.from_dict({"task_prefix": "test"})
        app = build_workflow(
            state_schema=SimpleState,
            config=config,
            nodes={"check": check, "path_a": path_a, "path_b": path_b},
            conditional_edges={"check": router},
            entry_point="check",
            finish_points=["path_a", "path_b"],
        )

        ctx = WorkflowContext(trace_id="t-cond", max_budget=0.0, task_prefix="test")

        result_a = app.invoke(
            ctx.inject_into_state({"query": "go_a"}),
            config={"configurable": {"thread_id": "t-a"}},
        )
        assert result_a["result"] == "A"

        result_b = app.invoke(
            ctx.inject_into_state({"query": "go_b"}),
            config={"configurable": {"thread_id": "t-b"}},
        )
        assert result_b["result"] == "B"

    def test_invalid_edge_source_raises(self) -> None:
        from llm_client.workflow.builder import build_workflow

        config = WorkflowConfig.from_dict({})
        with pytest.raises(ValueError, match="not a defined node"):
            build_workflow(
                state_schema=SimpleState,
                config=config,
                nodes={"a": lambda s: s},
                edges=[("nonexistent", "a")],
            )

    def test_invalid_edge_target_raises(self) -> None:
        from llm_client.workflow.builder import build_workflow

        config = WorkflowConfig.from_dict({})
        with pytest.raises(ValueError, match="not a defined node"):
            build_workflow(
                state_schema=SimpleState,
                config=config,
                nodes={"a": lambda s: s},
                edges=[("a", "nonexistent")],
            )

    def test_interrupt_and_resume(self) -> None:
        """Verify interrupt_before pauses and Command resumes."""
        from langgraph.types import Command, interrupt

        from llm_client.workflow.builder import build_workflow

        def step_1(state: dict[str, Any]) -> dict[str, Any]:
            return {"result": "step1_done"}

        def step_2(state: dict[str, Any]) -> dict[str, Any]:
            user_input = interrupt("Need user input")
            return {"result": f"got: {user_input}"}

        config = WorkflowConfig.from_dict({"task_prefix": "test"})
        app = build_workflow(
            state_schema=SimpleState,
            config=config,
            nodes={"step_1": step_1, "step_2": step_2},
            edges=[("step_1", "step_2")],
            finish_points=["step_2"],
        )

        ctx = WorkflowContext(trace_id="t-int", max_budget=0.0, task_prefix="test")
        thread = {"configurable": {"thread_id": "t-interrupt"}}

        # First invoke -- should pause at step_2's interrupt
        result = app.invoke(ctx.inject_into_state({"query": "test"}), config=thread)
        snapshot = app.get_state(thread)
        assert snapshot.next  # there are pending nodes

        # Resume with user input
        result = app.invoke(Command(resume="user_value"), config=thread)
        assert result["result"] == "got: user_value"
