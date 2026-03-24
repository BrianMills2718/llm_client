"""Evidence tracking and stagnation detection for the MCP agent loop.

Classifies evidence-producing tool calls, collects evidence pointer labels,
computes evidence digests, and monitors retrieval stagnation across turns.
"""

from __future__ import annotations

from typing import Any

from llm_client.agent.agent_artifacts import (
    _collect_evidence_pointer_labels as _agent_collect_evidence_pointer_labels,
    _evidence_digest as _agent_evidence_digest,
    _tool_evidence_pointer_labels as _agent_tool_evidence_pointer_labels,
)
from llm_client.agent.mcp_tools import (
    BUDGET_EXEMPT_TOOL_NAMES,
    _is_budget_exempt_tool,
)
from llm_client.tools.tool_runtime_common import MCPToolCallRecord


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_RETRIEVAL_STAGNATION_TURNS: int = 4
"""Force final answer after this many consecutive evidence turns with zero new evidence."""

DEFAULT_RETRIEVAL_STAGNATION_ACTION: str = "force_final"
"""Stagnation behavior: 'force_final' (default) or 'observe' (diagnostic only)."""

RETRIEVAL_STAGNATION_ACTIONS: frozenset[str] = frozenset({"force_final", "observe"})
"""Allowed stagnation actions."""


# ---------------------------------------------------------------------------
# Evidence classification
# ---------------------------------------------------------------------------

def _is_evidence_tool_name(tool_name: str) -> bool:
    """Best-effort classification for evidence-producing non-control tools."""
    name = (tool_name or "").strip().lower()
    if not name or _is_budget_exempt_tool(name):
        return False
    return name.startswith(("entity_", "relationship_", "chunk_", "subgraph_", "community_", "bridge_"))


# ---------------------------------------------------------------------------
# Evidence pointer delegation
# ---------------------------------------------------------------------------

def _collect_evidence_pointer_labels(payload: Any, out: set[str]) -> None:
    """Collect evidence pointer labels from a tool result payload."""
    _agent_collect_evidence_pointer_labels(payload, out)


def _tool_evidence_pointer_labels(record: MCPToolCallRecord) -> set[str]:
    """Extract evidence pointer labels from a tool call record."""
    return _agent_tool_evidence_pointer_labels(
        record,
        budget_exempt_tool_names=BUDGET_EXEMPT_TOOL_NAMES,
    )


def _evidence_digest(evidence_labels: set[str]) -> str:
    """Compute a stable digest of the current evidence pointer label set."""
    return _agent_evidence_digest(evidence_labels)
