"""Workflow builder -- thin wrapper around LangGraph StateGraph.

Provides ``build_workflow()`` which compiles a LangGraph StateGraph with
automatic WorkflowContext injection. Validates node/edge wiring at build
time (fail-loud) and sets up checkpointing.

Requires langgraph: ``pip install llm-client[workflow]``
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Hashable, Sequence, TypeVar

try:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    _HAS_LANGGRAPH = True
except ImportError:
    _HAS_LANGGRAPH = False

from llm_client.workflow.config import WorkflowConfig
from llm_client.workflow.context import WF_STAGE_KEY


def _require_langgraph() -> None:
    """Raise a clear error if langgraph is not installed."""
    if not _HAS_LANGGRAPH:
        raise ImportError(
            "langgraph is required for llm_client.workflow.builder. "
            "Install it with: pip install llm-client[workflow]"
        )

logger = logging.getLogger(__name__)

StateT = TypeVar("StateT")


def _wrap_node(
    name: str,
    fn: Callable[..., Any],
    config: WorkflowConfig,
) -> Callable[..., Any]:
    """Wrap a node function to inject the current stage name into state.

    This lets WorkflowContext.current(state) know which stage is executing
    so it can derive the correct task label (e.g., 'research.decompose').
    """
    @functools.wraps(fn)
    def wrapper(state: dict[str, Any]) -> Any:
        state[WF_STAGE_KEY] = name
        return fn(state)

    return wrapper


def build_workflow(
    *,
    state_schema: type[Any],
    config: WorkflowConfig,
    nodes: dict[str, Callable[..., Any]],
    edges: Sequence[tuple[str, str]] | None = None,
    conditional_edges: dict[str, Callable[..., Hashable | Sequence[Hashable]]] | None = None,
    entry_point: str | None = None,
    finish_points: Sequence[str] | None = None,
    checkpointer: Any | None = None,
    interrupt_before: Sequence[str] | None = None,
    interrupt_after: Sequence[str] | None = None,
) -> Any:
    """Build and compile a LangGraph workflow with llm_client integration.

    This is a thin wrapper around StateGraph that:
    1. Wraps each node to inject the current stage name
    2. Validates all edge references at build time
    3. Compiles with a checkpointer (InMemorySaver by default)

    The caller is responsible for injecting WorkflowContext into the initial
    state via ``ctx.inject_into_state(initial_state)`` before invoking.

    Args:
        state_schema: TypedDict or similar defining the LangGraph state.
        config: WorkflowConfig with per-stage settings.
        nodes: Mapping of stage name to node function.
        edges: List of (source, target) edges. Use "END" as target for terminal edges.
        conditional_edges: Mapping of source node to router function.
        entry_point: First node to execute (defaults to first node in edges).
        finish_points: Nodes that terminate the workflow (adds edge to END).
        checkpointer: LangGraph checkpointer (InMemorySaver if None).
        interrupt_before: Node names to interrupt before execution.
        interrupt_after: Node names to interrupt after execution.

    Returns:
        Compiled LangGraph app (CompiledStateGraph) with invoke/stream/get_state.

    Raises:
        ValueError: If edges reference undefined nodes.
        ImportError: If langgraph is not installed.
    """
    _require_langgraph()
    graph = StateGraph(state_schema)

    # Add wrapped nodes
    for name, fn in nodes.items():
        wrapped = _wrap_node(name, fn, config)
        graph.add_node(name, wrapped)

    # Validate and add edges
    all_node_names = set(nodes.keys())
    edges = edges or []

    for source, target in edges:
        if source not in all_node_names:
            raise ValueError(
                f"Edge source '{source}' is not a defined node. "
                f"Available: {sorted(all_node_names)}"
            )
        if target != "END" and target not in all_node_names:
            raise ValueError(
                f"Edge target '{target}' is not a defined node. "
                f"Available: {sorted(all_node_names)} + 'END'"
            )
        actual_target = END if target == "END" else target
        graph.add_edge(source, actual_target)

    # Add conditional edges
    if conditional_edges:
        for source, router in conditional_edges.items():
            if source not in all_node_names:
                raise ValueError(
                    f"Conditional edge source '{source}' is not a defined node. "
                    f"Available: {sorted(all_node_names)}"
                )
            graph.add_conditional_edges(source, router)

    # Set entry point
    if entry_point:
        if entry_point not in all_node_names:
            raise ValueError(
                f"Entry point '{entry_point}' is not a defined node. "
                f"Available: {sorted(all_node_names)}"
            )
        graph.add_edge(START, entry_point)
    elif edges:
        # Default: first edge's source
        graph.add_edge(START, edges[0][0])

    # Set finish points
    if finish_points:
        for fp in finish_points:
            if fp not in all_node_names:
                raise ValueError(
                    f"Finish point '{fp}' is not a defined node. "
                    f"Available: {sorted(all_node_names)}"
                )
            graph.add_edge(fp, END)

    # Compile
    if checkpointer is None:
        checkpointer = InMemorySaver()

    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=list(interrupt_before) if interrupt_before else None,
        interrupt_after=list(interrupt_after) if interrupt_after else None,
    )

    logger.info(
        "Built workflow with %d nodes, %d edges, checkpointer=%s",
        len(nodes),
        len(edges) + len(conditional_edges or {}),
        type(checkpointer).__name__,
    )

    return compiled
