"""Context-budget-based deferred tool loading for agent runtimes.

Supports server-level ``defer_loading`` config that hides tool definitions
from the initial context to save tokens.  A synthetic ``search_available_tools``
tool is injected so the agent can discover deferred tools on demand.

This is orthogonal to the artifact-based progressive disclosure in
``disclosure.py``, which hides tools based on prerequisite state.
"""

from __future__ import annotations

import json as _json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

SEARCH_TOOL_NAME = "search_available_tools"
"""Synthetic tool name for on-demand deferred tool discovery."""

# Rough estimate: 4 chars per token (matches DEFAULT_TOOL_DISCLOSURE_TOKEN_CHARS).
_CHARS_PER_TOKEN_ESTIMATE = 4


def search_tool_definition() -> dict[str, Any]:
    """Return the OpenAI-format schema for the synthetic search tool."""
    return {
        "type": "function",
        "function": {
            "name": SEARCH_TOOL_NAME,
            "description": (
                "Search for available tools by keyword or capability description. "
                "Use this when you need a tool that isn't in your current visible set."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query describing what capability you need",
                    },
                },
                "required": ["query"],
            },
        },
    }


def _tool_name(tool_def: dict[str, Any]) -> str:
    """Extract normalized tool name from an OpenAI tool definition."""
    fn = tool_def.get("function", {})
    return str(fn.get("name", "")).strip()


def _tool_description(tool_def: dict[str, Any]) -> str:
    """Extract description text from an OpenAI tool definition."""
    fn = tool_def.get("function", {})
    return str(fn.get("description", "")).strip()


def _tool_search_text(tool_def: dict[str, Any]) -> str:
    """Build a combined search string from tool name and description."""
    name = _tool_name(tool_def)
    desc = _tool_description(tool_def)
    return f"{name} {desc}".lower()


class DeferredToolRegistry:
    """Registry of deferred tools with keyword search and activation tracking.

    All deferred tool definitions are stored here.  The agent loop consults
    this registry when processing ``search_available_tools`` calls and when
    building final metrics.
    """

    def __init__(self) -> None:
        """Initialize an empty deferred tool registry."""
        self._deferred: dict[str, dict[str, Any]] = {}
        self._activated: set[str] = set()

    @property
    def deferred_total(self) -> int:
        """Total number of deferred tool definitions."""
        return len(self._deferred)

    @property
    def discovered_count(self) -> int:
        """Number of deferred tools activated via search."""
        return len(self._activated)

    def add(self, tool_def: dict[str, Any]) -> None:
        """Add a tool definition to the deferred set."""
        name = _tool_name(tool_def)
        if not name:
            logger.warning("Skipping deferred tool with empty name")
            return
        self._deferred[name] = tool_def

    def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Search deferred tools by keyword matching on name and description.

        Returns up to ``max_results`` matching tool definitions sorted by
        relevance (number of query terms matched).
        """
        if not query or not query.strip():
            return []

        terms = [t.lower() for t in re.split(r'\s+', query.strip()) if t]
        if not terms:
            return []

        scored: list[tuple[int, str, dict[str, Any]]] = []
        for name, tool_def in self._deferred.items():
            search_text = _tool_search_text(tool_def)
            score = sum(1 for term in terms if term in search_text)
            if score > 0:
                scored.append((score, name, tool_def))

        scored.sort(key=lambda x: (-x[0], x[1]))
        return [tool_def for _score, _name, tool_def in scored[:max_results]]

    def activate(self, tool_defs: list[dict[str, Any]]) -> list[str]:
        """Mark tools as activated and return their names.

        Called after search results are returned to the agent so the tools
        become available in subsequent turns.
        """
        activated_names: list[str] = []
        for tool_def in tool_defs:
            name = _tool_name(tool_def)
            if name and name in self._deferred:
                self._activated.add(name)
                activated_names.append(name)
        return activated_names

    def get_activated_tools(self) -> list[dict[str, Any]]:
        """Return all tool definitions that have been activated via search."""
        return [
            self._deferred[name]
            for name in sorted(self._activated)
            if name in self._deferred
        ]

    def estimated_token_savings(self) -> int:
        """Estimate tokens saved by deferring tools that were never activated.

        Counts characters in deferred tool definitions that remain unactivated,
        divided by the chars-per-token estimate.
        """
        savings_chars = 0
        for name, tool_def in self._deferred.items():
            if name not in self._activated:
                savings_chars += len(_json.dumps(tool_def))
        return max(0, savings_chars // _CHARS_PER_TOKEN_ESTIMATE)

    def has_deferred(self) -> bool:
        """Return True when any tools are deferred."""
        return bool(self._deferred)

    def deferred_tool_names(self) -> list[str]:
        """Return sorted list of all deferred tool names."""
        return sorted(self._deferred.keys())


def partition_tools_for_deferral(
    openai_tools: list[dict[str, Any]],
    server_name: str,
    server_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split discovered tools into always-loaded and deferred sets.

    Args:
        openai_tools: All OpenAI-format tool definitions from a single server.
        server_name: Name of the MCP server (for logging).
        server_cfg: Server config dict, which may contain ``defer_loading``
            and ``always_loaded_tools``.

    Returns:
        (always_loaded, deferred) — two lists of tool definitions.
    """
    defer_loading = server_cfg.get("defer_loading", False)
    if not defer_loading:
        return list(openai_tools), []

    always_loaded_names: set[str] = set()
    raw = server_cfg.get("always_loaded_tools")
    if isinstance(raw, (list, tuple)):
        always_loaded_names = {str(n).strip() for n in raw if isinstance(n, str) and n.strip()}

    always_loaded: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    for tool_def in openai_tools:
        name = _tool_name(tool_def)
        if name in always_loaded_names:
            always_loaded.append(tool_def)
        else:
            deferred.append(tool_def)

    logger.info(
        "DEFERRED_TOOLS server=%s always_loaded=%d deferred=%d always_loaded_names=%s",
        server_name,
        len(always_loaded),
        len(deferred),
        sorted(always_loaded_names),
    )
    return always_loaded, deferred


def execute_search_tool_call(
    registry: DeferredToolRegistry,
    arguments: dict[str, Any],
) -> str:
    """Execute a search_available_tools call against the deferred registry.

    Returns a JSON string with matching tool names and descriptions.
    """
    query = str(arguments.get("query", "")).strip()
    if not query:
        return _json.dumps({"error": "query parameter is required", "results": []})

    matches = registry.search(query, max_results=5)
    activated = registry.activate(matches)

    results: list[dict[str, str]] = []
    for tool_def in matches:
        results.append({
            "name": _tool_name(tool_def),
            "description": _tool_description(tool_def),
        })

    return _json.dumps({
        "results": results,
        "activated": activated,
        "message": (
            f"Found {len(results)} matching tool(s). "
            "They are now available for use in your next tool call."
            if results
            else "No matching tools found. Try different search terms."
        ),
    })


def build_deferred_metrics(registry: DeferredToolRegistry) -> dict[str, Any]:
    """Build metadata dict for deferred tool metrics.

    Returned keys:
        deferred_tools_total: how many tools were deferred
        deferred_tools_discovered: how many were activated via search
        deferred_tools_token_savings: estimated tokens saved by deferring
    """
    return {
        "deferred_tools_total": registry.deferred_total,
        "deferred_tools_discovered": registry.discovered_count,
        "deferred_tools_token_savings": registry.estimated_token_savings(),
    }
