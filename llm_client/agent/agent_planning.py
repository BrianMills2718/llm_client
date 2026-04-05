"""Agent planning and working memory for the MCP agent loop.

Provides built-in plan creation and progress tracking tools that any agent loop
consumer gets automatically when PlanningConfig.enabled=True. Plan progress is
auto-injected into every turn's context so the agent always sees where it is.

Plan 19: Agent Planning and Working Memory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from pydantic import Field

try:
    from data_contracts import boundary, BoundaryModel
except ImportError:
    def boundary(*args, **kwargs):  # type: ignore[misc]
        def decorator(fn):  # type: ignore[misc]
            return fn
        return decorator
    from pydantic import BaseModel as BoundaryModel  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class ToolSignatureContract(BoundaryModel):
    """Describes the required callable shape for a DIGIMON-provided custom plan or todo tool."""

    model_config = {"extra": "forbid"}

    tool_name: str = Field(description="'plan_tool' or 'todo_tool' — which slot this contract describes")
    accepts_text: bool = Field(description="Whether the tool accepts a plain text/string input")
    returns_structured: bool = Field(description="Whether the tool returns structured data (dict/Pydantic) rather than a raw string")


@boundary(
    name="llm_client.digimon_tool_integration",
    producer="llm_client",
    consumers=["Digimon_for_KG_application"],
)
def validate_custom_tool_contract(tool_name: str, tool: Callable[..., Any]) -> ToolSignatureContract:
    """Validate and describe the contract for a DIGIMON-supplied custom plan or todo tool.

    DIGIMON passes custom_plan_tool / custom_todo_tool via PlanningConfig.
    This boundary records the expected calling contract so the integration
    surface is visible to ecosystem tooling.

    Args:
        tool_name: Either 'plan_tool' or 'todo_tool'.
        tool: The callable provided by DIGIMON.

    Returns:
        A ToolSignatureContract describing the tool's expected interface.
    """
    import inspect
    sig = inspect.signature(tool)
    params = list(sig.parameters.values())
    accepts_text = any(
        p.annotation in (str, inspect.Parameter.empty)
        for p in params
        if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    )
    # Treat dict/BaseModel return annotations as structured; str/None/empty as plain
    ret = sig.return_annotation
    returns_structured = ret not in (str, None, inspect.Parameter.empty)
    return ToolSignatureContract(
        tool_name=tool_name,
        accepts_text=accepts_text,
        returns_structured=returns_structured,
    )


@dataclass
class PlanningConfig:
    """Configuration for agent planning and working memory.

    Consumers declare this to enable built-in plan/todo tools in the agent loop.
    DIGIMON can override with custom tools via custom_plan_tool/custom_todo_tool.

    Attributes:
        enabled: Whether planning tools are registered in the agent loop.
        auto_inject_context: Prepend plan progress to every turn's messages.
        context_format: "compact" (one line per step) or "full" (with results).
        max_context_chars: Budget for the injected plan summary.
        custom_plan_tool: Project-specific plan creation tool (replaces built-in).
        custom_todo_tool: Project-specific status update tool (replaces built-in).
        require_plan_before_tools: If True, first action must be create_plan.
    """

    enabled: bool = True
    auto_inject_context: bool = True
    context_format: Literal["compact", "full"] = "compact"
    max_context_chars: int = 500
    custom_plan_tool: Any | None = None
    custom_todo_tool: Any | None = None
    require_plan_before_tools: bool = False


@dataclass
class PlanStep:
    """A single step in an agent plan.

    Attributes:
        step_id: Short identifier (s1, s2, ...).
        description: What this step does.
        depends_on: Step IDs that must complete before this one.
        status: Current status.
        result: Evidence or finding when done.
        attempts: How many times this step was attempted.
    """

    step_id: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    status: Literal["pending", "in_progress", "done", "blocked"] = "pending"
    result: str = ""
    attempts: int = 0


_STATUS_ALIASES: dict[str, str] = {
    "started": "in_progress",
    "complete": "done",
    "completed": "done",
    "waiting": "blocked",
    "skip": "done",
    "skipped": "done",
}


@dataclass
class AgentPlan:
    """A structured plan for an agent task.

    Created by the agent via create_plan, updated via update_plan.
    """

    plan_id: str
    question: str
    steps: list[PlanStep] = field(default_factory=list)
    created_turn: int = 0


class PlanState:
    """Mutable state tracker for an active agent plan.

    Created once per agent loop invocation. Shared between the built-in
    plan/todo tools and the context injection logic.
    """

    def __init__(self) -> None:
        self._plan: AgentPlan | None = None

    @property
    def has_plan(self) -> bool:
        """Whether a plan has been created."""
        return self._plan is not None

    @property
    def plan(self) -> AgentPlan | None:
        """The current plan, if any."""
        return self._plan

    def create_plan(self, question: str, steps: list[dict[str, Any]], turn: int = 0) -> AgentPlan:
        """Create a new plan from a list of step dicts.

        Args:
            question: The original task.
            steps: List of dicts with 'step_id', 'description', optional 'depends_on'.
            turn: The turn number when the plan was created.

        Returns:
            The created AgentPlan.

        Raises:
            ValueError: If steps are empty, have duplicate IDs, or circular deps.
        """
        if not steps:
            raise ValueError("Plan must have at least one step.")

        plan_steps = []
        seen_ids: set[str] = set()
        for i, s in enumerate(steps):
            step_id = s.get("step_id", f"s{i + 1}")
            if step_id in seen_ids:
                raise ValueError(f"Duplicate step_id: {step_id}")
            seen_ids.add(step_id)
            depends = s.get("depends_on", [])
            plan_steps.append(PlanStep(
                step_id=step_id,
                description=s.get("description", f"Step {i + 1}"),
                depends_on=depends if isinstance(depends, list) else [depends],
            ))

        # Validate dependency references
        for step in plan_steps:
            for dep in step.depends_on:
                if dep not in seen_ids:
                    raise ValueError(f"Step {step.step_id} depends on unknown step {dep}")

        # Simple cycle detection (topological sort attempt)
        remaining = {s.step_id for s in plan_steps}
        deps_map = {s.step_id: set(s.depends_on) for s in plan_steps}
        resolved: set[str] = set()
        changed = True
        while changed:
            changed = False
            for sid in list(remaining):
                if deps_map[sid] <= resolved:
                    resolved.add(sid)
                    remaining.discard(sid)
                    changed = True
        if remaining:
            raise ValueError(f"Circular dependency detected involving: {remaining}")

        plan_id = f"plan_{turn}"
        self._plan = AgentPlan(
            plan_id=plan_id,
            question=question,
            steps=plan_steps,
            created_turn=turn,
        )
        logger.info("Created plan %s with %d steps for: %s", plan_id, len(plan_steps), question[:80])
        return self._plan

    def update_step(self, step_id: str, status: str, result: str = "") -> PlanStep:
        """Update a plan step's status and optional result.

        Args:
            step_id: Which step to update.
            status: New status (or alias like "started", "complete").
            result: Evidence or finding (required when status=done).

        Returns:
            The updated PlanStep.

        Raises:
            ValueError: If no plan exists, step not found, or invalid status.
        """
        if self._plan is None:
            raise ValueError("No plan exists. Call create_plan first.")

        # Resolve aliases
        status = _STATUS_ALIASES.get(status, status)
        if status not in ("pending", "in_progress", "done", "blocked"):
            raise ValueError(f"Invalid status: {status!r}. Use: pending, in_progress, done, blocked")

        step = None
        for s in self._plan.steps:
            if s.step_id == step_id:
                step = s
                break
        if step is None:
            raise ValueError(f"Step {step_id!r} not found. Available: {[s.step_id for s in self._plan.steps]}")

        step.status = status
        step.attempts += 1
        if result:
            step.result = result

        logger.debug("Updated step %s → %s%s", step_id, status, f" ({result[:50]})" if result else "")
        return step

    def format_context(self, max_chars: int = 500, fmt: str = "compact") -> str:
        """Format plan progress as a compact context string.

        Args:
            max_chars: Maximum characters for the summary.
            fmt: "compact" or "full".

        Returns:
            Formatted plan progress string, or empty string if no plan.
        """
        if self._plan is None:
            return ""

        done_count = sum(1 for s in self._plan.steps if s.status == "done")
        total = len(self._plan.steps)

        lines = [f"[PLAN PROGRESS: {done_count}/{total} steps done]"]

        for step in self._plan.steps:
            icon = {"done": "[x]", "in_progress": "[>]", "blocked": "[!]"}.get(step.status, "[ ]")
            line = f"{icon} {step.step_id}: {step.description}"
            if step.result and fmt == "compact":
                # Truncate result to fit
                max_result = 40
                result_preview = step.result[:max_result]
                if len(step.result) > max_result:
                    result_preview += "..."
                line += f" → {result_preview!r}"
            elif step.result and fmt == "full":
                line += f"\n    Result: {step.result}"
            lines.append(line)

        # Dependency summary (compact)
        deps = []
        for step in self._plan.steps:
            if step.depends_on:
                dep_status = []
                for dep_id in step.depends_on:
                    dep_step = next((s for s in self._plan.steps if s.step_id == dep_id), None)
                    check = "✓" if dep_step and dep_step.status == "done" else "…"
                    dep_status.append(f"{dep_id} {check}")
                deps.append(f"{step.step_id} needs {', '.join(dep_status)}")
        if deps:
            lines.append(f"Depends: {'; '.join(deps)}")

        result = "\n".join(lines)
        if len(result) > max_chars:
            result = result[:max_chars - 3] + "..."
        return result

    def summary(self) -> dict[str, Any]:
        """Return a summary dict for observability."""
        if self._plan is None:
            return {"has_plan": False}
        return {
            "has_plan": True,
            "plan_id": self._plan.plan_id,
            "total_steps": len(self._plan.steps),
            "done_steps": sum(1 for s in self._plan.steps if s.status == "done"),
            "in_progress_steps": sum(1 for s in self._plan.steps if s.status == "in_progress"),
            "blocked_steps": sum(1 for s in self._plan.steps if s.status == "blocked"),
            "steps": [
                {
                    "step_id": s.step_id,
                    "status": s.status,
                    "attempts": s.attempts,
                    "has_result": bool(s.result),
                }
                for s in self._plan.steps
            ],
        }


def build_plan_tools(plan_state: PlanState, config: PlanningConfig) -> list[dict[str, Any]]:
    """Build OpenAI-format tool definitions for the plan/todo tools.

    If config has custom tools, those are used instead of the built-in ones.

    Args:
        plan_state: Shared plan state (tools operate on this).
        config: Planning configuration.

    Returns:
        List of OpenAI-format tool dicts.
    """
    tools: list[dict[str, Any]] = []

    if not config.enabled:
        return tools

    tools.append({
        "type": "function",
        "function": {
            "name": "create_plan",
            "description": (
                "Create a structured plan for the task. Call FIRST before using other tools. "
                "Decompose the task into 2-8 concrete steps with dependencies."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "description": "List of plan steps. Each step has step_id, description, and optional depends_on.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "step_id": {"type": "string", "description": "Short ID like s1, s2, ..."},
                                "description": {"type": "string", "description": "What this step does"},
                                "depends_on": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Step IDs that must complete first",
                                },
                            },
                            "required": ["step_id", "description"],
                        },
                        "minItems": 1,
                        "maxItems": 8,
                    },
                },
                "required": ["steps"],
            },
        },
    })

    tools.append({
        "type": "function",
        "function": {
            "name": "update_plan",
            "description": (
                "Update a plan step's status and record results. "
                "Call after completing or starting a step."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "step_id": {
                        "type": "string",
                        "description": "Which step to update (s1, s2, ...)",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["in_progress", "done", "blocked"],
                        "description": "New status for the step",
                    },
                    "result": {
                        "type": "string",
                        "description": "What was found/achieved (required when status=done)",
                    },
                },
                "required": ["step_id", "status"],
            },
        },
    })

    return tools


def execute_plan_tool(
    tool_name: str,
    arguments: dict[str, Any],
    plan_state: PlanState,
    question: str,
    turn: int,
) -> str:
    """Execute a plan/todo tool call and return the result string.

    Args:
        tool_name: "create_plan" or "update_plan".
        arguments: Tool call arguments.
        plan_state: Shared plan state.
        question: The original task (for create_plan).
        turn: Current turn number.

    Returns:
        Tool result string (plan summary or error message).
    """
    if tool_name == "create_plan":
        steps = arguments.get("steps", [])
        try:
            plan_state.create_plan(question=question, steps=steps, turn=turn)
            return plan_state.format_context(max_chars=1000, fmt="full")
        except ValueError as e:
            return f"Error creating plan: {e}"

    elif tool_name == "update_plan":
        step_id = arguments.get("step_id", "")
        status = arguments.get("status", "")
        result = arguments.get("result", "")
        try:
            plan_state.update_step(step_id, status, result)
            return plan_state.format_context(max_chars=1000, fmt="compact")
        except ValueError as e:
            return f"Error updating plan: {e}"

    return f"Unknown plan tool: {tool_name}"
