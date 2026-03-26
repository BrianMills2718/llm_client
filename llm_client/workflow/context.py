"""Workflow context -- carries observability and budget through LangGraph state.

WorkflowContext is the integration seam between LangGraph nodes and
llm_client's call contracts. It provides ``call_llm()`` and
``call_llm_structured()`` wrappers that auto-inject ``task``, ``trace_id``,
and ``max_budget`` so node functions don't need to pass them manually.

Context travels as plain string fields in the LangGraph state dict
(``_wf_trace_id``, ``_wf_max_budget``, ``_wf_task_prefix``) so it
survives checkpointing without requiring custom serialization.
"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

# State keys used to carry context through LangGraph state
WF_TRACE_ID_KEY = "_wf_trace_id"
WF_MAX_BUDGET_KEY = "_wf_max_budget"
WF_TASK_PREFIX_KEY = "_wf_task_prefix"
WF_STAGE_KEY = "_wf_current_stage"


class WorkflowContext:
    """Carries llm_client call contracts through a LangGraph workflow run.

    Created once at workflow start and injected into state as plain fields.
    Nodes recover it via ``WorkflowContext.current(state)`` and use its
    ``call_llm()`` / ``call_llm_structured()`` methods which auto-inject
    the shared trace_id, max_budget, and per-stage task name.
    """

    def __init__(
        self,
        *,
        trace_id: str,
        max_budget: float,
        task_prefix: str,
        stage: str = "",
    ) -> None:
        """Initialize workflow context.

        Args:
            trace_id: Shared across all nodes -- groups all calls in one trace.
            max_budget: USD budget for the entire workflow run.
            task_prefix: Prepended to stage name for per-node task labels.
            stage: Current stage name (set automatically by build_workflow).
        """
        self.trace_id = trace_id
        self.max_budget = max_budget
        self.task_prefix = task_prefix
        self.stage = stage

    @property
    def task(self) -> str:
        """Derived task label: '{task_prefix}.{stage}' or just task_prefix."""
        if self.stage:
            return f"{self.task_prefix}.{self.stage}"
        return self.task_prefix

    def inject_into_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of state with context fields injected."""
        return {
            **state,
            WF_TRACE_ID_KEY: self.trace_id,
            WF_MAX_BUDGET_KEY: self.max_budget,
            WF_TASK_PREFIX_KEY: self.task_prefix,
        }

    @classmethod
    def current(cls, state: dict[str, Any], *, stage: str = "") -> WorkflowContext:
        """Recover context from LangGraph state inside a node function.

        Args:
            state: The LangGraph state dict passed to the node.
            stage: Override stage name (usually set by the node wrapper).

        Raises:
            KeyError: If context fields are missing -- means the workflow
                was not built with build_workflow() or context was not injected.
        """
        trace_id = state.get(WF_TRACE_ID_KEY)
        if trace_id is None:
            raise KeyError(
                f"Missing {WF_TRACE_ID_KEY} in state. "
                "Was this workflow built with build_workflow()?"
            )
        return cls(
            trace_id=str(trace_id),
            max_budget=float(state.get(WF_MAX_BUDGET_KEY, 0.0)),
            task_prefix=str(state.get(WF_TASK_PREFIX_KEY, "")),
            stage=stage or str(state.get(WF_STAGE_KEY, "")),
        )

    def call_llm(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        """Call llm_client.call_llm with auto-injected workflow contracts.

        All kwargs are forwarded. task, trace_id, and max_budget are set
        from the workflow context but can be overridden explicitly.
        """
        from llm_client.core.client import call_llm

        kwargs.setdefault("task", self.task)
        kwargs.setdefault("trace_id", self.trace_id)
        kwargs.setdefault("max_budget", self.max_budget)
        return call_llm(model, messages, **kwargs)

    async def acall_llm(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        """Async version of call_llm with auto-injected workflow contracts."""
        from llm_client.core.client import acall_llm

        kwargs.setdefault("task", self.task)
        kwargs.setdefault("trace_id", self.trace_id)
        kwargs.setdefault("max_budget", self.max_budget)
        return await acall_llm(model, messages, **kwargs)

    def call_llm_structured(
        self,
        model: str,
        messages: list[dict[str, Any]],
        response_model: type[T],
        **kwargs: Any,
    ) -> tuple[T, Any]:
        """Call llm_client.call_llm_structured with auto-injected contracts.

        Returns (parsed_model, LLMCallResult) tuple.
        """
        from llm_client.core.client import call_llm_structured

        kwargs.setdefault("task", self.task)
        kwargs.setdefault("trace_id", self.trace_id)
        kwargs.setdefault("max_budget", self.max_budget)
        return call_llm_structured(model, messages, response_model, **kwargs)

    async def acall_llm_structured(
        self,
        model: str,
        messages: list[dict[str, Any]],
        response_model: type[T],
        **kwargs: Any,
    ) -> tuple[T, Any]:
        """Async version of call_llm_structured with auto-injected contracts."""
        from llm_client.core.client import acall_llm_structured

        kwargs.setdefault("task", self.task)
        kwargs.setdefault("trace_id", self.trace_id)
        kwargs.setdefault("max_budget", self.max_budget)
        return await acall_llm_structured(model, messages, response_model, **kwargs)
