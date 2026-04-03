"""Tests for agent planning and working memory (Plan #19).

Verifies PlanState, PlanningConfig, tool definitions, context injection,
and edge cases (cycles, aliases, truncation).
"""

import pytest

from llm_client.agent.agent_planning import (
    AgentPlan,
    PlanState,
    PlanStep,
    PlanningConfig,
    build_plan_tools,
    execute_plan_tool,
)
from llm_client.agent.mcp_tools import _count_budgeted_records, _count_budgeted_tool_calls
from llm_client.tools.tool_runtime_common import MCPToolCallRecord


class TestPlanState:
    """Core plan state management."""

    def test_no_plan_initially(self) -> None:
        state = PlanState()
        assert not state.has_plan
        assert state.plan is None
        assert state.format_context() == ""

    def test_create_plan_basic(self) -> None:
        state = PlanState()
        plan = state.create_plan("Find the capital of France", [
            {"step_id": "s1", "description": "Search for France"},
            {"step_id": "s2", "description": "Extract capital", "depends_on": ["s1"]},
        ])
        assert state.has_plan
        assert len(plan.steps) == 2
        assert plan.steps[0].status == "pending"
        assert plan.steps[1].depends_on == ["s1"]

    def test_create_plan_auto_ids(self) -> None:
        state = PlanState()
        plan = state.create_plan("task", [
            {"description": "First step"},
            {"description": "Second step"},
        ])
        assert plan.steps[0].step_id == "s1"
        assert plan.steps[1].step_id == "s2"

    def test_create_plan_rejects_empty(self) -> None:
        state = PlanState()
        with pytest.raises(ValueError, match="at least one step"):
            state.create_plan("task", [])

    def test_create_plan_rejects_duplicate_ids(self) -> None:
        state = PlanState()
        with pytest.raises(ValueError, match="Duplicate step_id"):
            state.create_plan("task", [
                {"step_id": "s1", "description": "A"},
                {"step_id": "s1", "description": "B"},
            ])

    def test_create_plan_rejects_unknown_deps(self) -> None:
        state = PlanState()
        with pytest.raises(ValueError, match="unknown step"):
            state.create_plan("task", [
                {"step_id": "s1", "description": "A", "depends_on": ["s99"]},
            ])

    def test_create_plan_rejects_cycles(self) -> None:
        state = PlanState()
        with pytest.raises(ValueError, match="Circular dependency"):
            state.create_plan("task", [
                {"step_id": "s1", "description": "A", "depends_on": ["s2"]},
                {"step_id": "s2", "description": "B", "depends_on": ["s1"]},
            ])

    def test_update_step_status(self) -> None:
        state = PlanState()
        state.create_plan("task", [{"step_id": "s1", "description": "Do thing"}])
        step = state.update_step("s1", "in_progress")
        assert step.status == "in_progress"
        assert step.attempts == 1
        step = state.update_step("s1", "done", result="Found it")
        assert step.status == "done"
        assert step.result == "Found it"
        assert step.attempts == 2

    def test_update_step_aliases(self) -> None:
        state = PlanState()
        state.create_plan("task", [{"step_id": "s1", "description": "Do thing"}])
        state.update_step("s1", "started")
        assert state.plan.steps[0].status == "in_progress"
        state.update_step("s1", "complete", result="done")
        assert state.plan.steps[0].status == "done"

    def test_update_step_invalid_status(self) -> None:
        state = PlanState()
        state.create_plan("task", [{"step_id": "s1", "description": "Do thing"}])
        with pytest.raises(ValueError, match="Invalid status"):
            state.update_step("s1", "yolo")

    def test_update_step_not_found(self) -> None:
        state = PlanState()
        state.create_plan("task", [{"step_id": "s1", "description": "Do thing"}])
        with pytest.raises(ValueError, match="not found"):
            state.update_step("s99", "done")

    def test_update_step_no_plan(self) -> None:
        state = PlanState()
        with pytest.raises(ValueError, match="No plan exists"):
            state.update_step("s1", "done")


class TestFormatContext:
    """Plan context formatting for injection."""

    def test_compact_format(self) -> None:
        state = PlanState()
        state.create_plan("Find X", [
            {"step_id": "s1", "description": "Search"},
            {"step_id": "s2", "description": "Extract", "depends_on": ["s1"]},
        ])
        state.update_step("s1", "done", result="Found in DB")
        ctx = state.format_context(max_chars=500, fmt="compact")
        assert "[PLAN PROGRESS: 1/2 steps done]" in ctx
        assert "[x] s1" in ctx
        assert "[ ] s2" in ctx
        assert "Depends:" in ctx

    def test_truncation(self) -> None:
        state = PlanState()
        state.create_plan("task", [
            {"step_id": f"s{i}", "description": f"Very long step description number {i} with details"}
            for i in range(1, 9)
        ])
        ctx = state.format_context(max_chars=200)
        assert len(ctx) <= 200
        assert ctx.endswith("...")

    def test_in_progress_icon(self) -> None:
        state = PlanState()
        state.create_plan("task", [{"step_id": "s1", "description": "Working"}])
        state.update_step("s1", "in_progress")
        ctx = state.format_context()
        assert "[>] s1" in ctx


class TestSummary:
    """Plan summary for observability."""

    def test_no_plan_summary(self) -> None:
        state = PlanState()
        assert state.summary() == {"has_plan": False}

    def test_plan_summary(self) -> None:
        state = PlanState()
        state.create_plan("task", [
            {"step_id": "s1", "description": "A"},
            {"step_id": "s2", "description": "B"},
        ])
        state.update_step("s1", "done", result="ok")
        summary = state.summary()
        assert summary["has_plan"] is True
        assert summary["total_steps"] == 2
        assert summary["done_steps"] == 1
        assert len(summary["steps"]) == 2


class TestBuildPlanTools:
    """Tool definition generation."""

    def test_enabled_returns_two_tools(self) -> None:
        config = PlanningConfig(enabled=True)
        state = PlanState()
        tools = build_plan_tools(state, config)
        assert len(tools) == 2
        names = {t["function"]["name"] for t in tools}
        assert names == {"create_plan", "update_plan"}

    def test_disabled_returns_empty(self) -> None:
        config = PlanningConfig(enabled=False)
        state = PlanState()
        assert build_plan_tools(state, config) == []

    def test_tool_schema_valid(self) -> None:
        config = PlanningConfig()
        state = PlanState()
        tools = build_plan_tools(state, config)
        for tool in tools:
            assert tool["type"] == "function"
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]


class TestPlanningBudgetExemption:
    """Planning tools should not consume retrieval-tool budget."""

    def test_planning_tool_calls_are_not_budgeted(self) -> None:
        tool_calls = [
            {"function": {"name": "create_plan"}},
            {"function": {"name": "update_plan"}},
            {"function": {"name": "search"}},
        ]

        assert _count_budgeted_tool_calls(tool_calls) == 1

    def test_planning_tool_records_are_not_budgeted(self) -> None:
        records = [
            MCPToolCallRecord(server="__agent__", tool="create_plan", arguments={}),
            MCPToolCallRecord(server="__agent__", tool="update_plan", arguments={}),
            MCPToolCallRecord(server="srv", tool="search", arguments={}),
        ]

        assert _count_budgeted_records(records) == 1


class TestExecutePlanTool:
    """Tool execution."""

    def test_create_plan_via_tool(self) -> None:
        state = PlanState()
        result = execute_plan_tool(
            "create_plan",
            {"steps": [
                {"step_id": "s1", "description": "Search"},
                {"step_id": "s2", "description": "Extract"},
            ]},
            state, question="Find X", turn=0,
        )
        assert state.has_plan
        assert "PLAN PROGRESS" in result

    def test_update_plan_via_tool(self) -> None:
        state = PlanState()
        execute_plan_tool("create_plan", {"steps": [{"step_id": "s1", "description": "Do"}]},
                          state, question="task", turn=0)
        result = execute_plan_tool("update_plan", {"step_id": "s1", "status": "done", "result": "ok"},
                                   state, question="task", turn=1)
        assert "[x] s1" in result

    def test_error_returns_message(self) -> None:
        state = PlanState()
        result = execute_plan_tool("create_plan", {"steps": []},
                                   state, question="task", turn=0)
        assert "Error" in result

    def test_unknown_tool(self) -> None:
        state = PlanState()
        result = execute_plan_tool("delete_plan", {}, state, question="task", turn=0)
        assert "Unknown" in result
