"""Tests for context-budget-based deferred tool loading (Plan 02, Slice D).

Tests the DeferredToolRegistry, partition_tools_for_deferral, search tool
injection, and deferred metric reporting.  These tests exercise the deferred
loading module directly without requiring MCP server subprocess infrastructure.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from llm_client.agent.deferred_tools import (
    SEARCH_TOOL_NAME,
    DeferredToolRegistry,
    build_deferred_metrics,
    execute_search_tool_call,
    partition_tools_for_deferral,
    search_tool_definition,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool(name: str, description: str = "") -> dict[str, Any]:
    """Build a minimal OpenAI-format tool definition for testing."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description or f"Tool {name}",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    }


def _tool_names(tools: list[dict[str, Any]]) -> set[str]:
    """Extract tool names from a list of OpenAI tool defs."""
    return {
        str(t.get("function", {}).get("name", "")).strip()
        for t in tools
    }


# ---------------------------------------------------------------------------
# Test: defer_loading hides tools
# ---------------------------------------------------------------------------

class TestDeferLoadingHidesTools:
    """Verify that defer_loading=True hides tools from the initial set."""

    def test_defer_loading_hides_tools(self) -> None:
        """Configure a server with defer_loading=True.
        Verify only always_loaded_tools appear in initial tool set."""
        tools = [
            _make_tool("alpha", "Alpha search"),
            _make_tool("beta", "Beta search"),
            _make_tool("gamma", "Gamma search"),
            _make_tool("delta", "Delta search"),
        ]
        server_cfg: dict[str, Any] = {
            "command": "python",
            "args": ["server.py"],
            "defer_loading": True,
            "always_loaded_tools": ["alpha"],
        }

        always_loaded, deferred = partition_tools_for_deferral(
            tools, "test-server", server_cfg,
        )

        assert _tool_names(always_loaded) == {"alpha"}
        assert _tool_names(deferred) == {"beta", "gamma", "delta"}
        assert len(always_loaded) == 1
        assert len(deferred) == 3

    def test_no_defer_keeps_all_tools(self) -> None:
        """When defer_loading is False (default), all tools are visible."""
        tools = [
            _make_tool("alpha"),
            _make_tool("beta"),
        ]
        server_cfg: dict[str, Any] = {
            "command": "python",
            "args": ["server.py"],
        }

        always_loaded, deferred = partition_tools_for_deferral(
            tools, "test-server", server_cfg,
        )

        assert _tool_names(always_loaded) == {"alpha", "beta"}
        assert deferred == []

    def test_defer_all_when_no_always_loaded(self) -> None:
        """When defer_loading=True with no always_loaded_tools, all are deferred."""
        tools = [_make_tool("a"), _make_tool("b")]
        server_cfg: dict[str, Any] = {
            "command": "python",
            "defer_loading": True,
        }

        always_loaded, deferred = partition_tools_for_deferral(
            tools, "test-server", server_cfg,
        )

        assert always_loaded == []
        assert len(deferred) == 2


# ---------------------------------------------------------------------------
# Test: always_loaded_tools visible
# ---------------------------------------------------------------------------

class TestAlwaysLoadedToolsVisible:
    """Verify always_loaded_tools appear in the initial set."""

    def test_always_loaded_tools_visible(self) -> None:
        """Configure defer_loading=True with 2 always_loaded_tools.
        Verify both are in the initial set."""
        tools = [
            _make_tool("list_operators", "List available operators"),
            _make_tool("meta_generate_answer", "Generate final answer"),
            _make_tool("entity_search", "Search entities"),
            _make_tool("chunk_search", "Search chunks"),
        ]
        server_cfg: dict[str, Any] = {
            "command": "python",
            "defer_loading": True,
            "always_loaded_tools": ["list_operators", "meta_generate_answer"],
        }

        always_loaded, deferred = partition_tools_for_deferral(
            tools, "digimon", server_cfg,
        )

        assert _tool_names(always_loaded) == {"list_operators", "meta_generate_answer"}
        assert _tool_names(deferred) == {"entity_search", "chunk_search"}


# ---------------------------------------------------------------------------
# Test: search tool added when deferred
# ---------------------------------------------------------------------------

class TestSearchToolAddedWhenDeferred:
    """Verify search_available_tools is auto-added when any server defers tools."""

    def test_search_tool_added_when_deferred(self) -> None:
        """When any server defers tools, verify search_available_tools is
        auto-added to the registry."""
        registry = DeferredToolRegistry()
        registry.add(_make_tool("hidden_tool", "A hidden capability"))

        assert registry.has_deferred()

        # The real _start_servers injects the search tool; simulate here
        visible_tools: list[dict[str, Any]] = [_make_tool("visible_tool")]
        if registry.has_deferred():
            visible_tools.append(search_tool_definition())

        names = _tool_names(visible_tools)
        assert SEARCH_TOOL_NAME in names
        assert "visible_tool" in names
        assert "hidden_tool" not in names

    def test_search_tool_schema_valid(self) -> None:
        """Verify the search tool definition has required fields."""
        defn = search_tool_definition()
        assert defn["type"] == "function"
        fn = defn["function"]
        assert fn["name"] == SEARCH_TOOL_NAME
        assert "description" in fn
        assert "parameters" in fn
        params = fn["parameters"]
        assert "query" in params["properties"]
        assert "query" in params["required"]


# ---------------------------------------------------------------------------
# Test: search finds deferred tool
# ---------------------------------------------------------------------------

class TestSearchFindsDeferredTool:
    """Verify search_available_tools returns matching deferred tools."""

    def test_search_finds_deferred_tool(self) -> None:
        """Call search_available_tools with a query.
        Verify it returns matching deferred tools."""
        registry = DeferredToolRegistry()
        registry.add(_make_tool("entity_vdb_search", "Search entities in vector database"))
        registry.add(_make_tool("chunk_search", "Search document chunks"))
        registry.add(_make_tool("relationship_query", "Query entity relationships"))

        matches = registry.search("entity search")
        assert len(matches) >= 1
        matched_names = _tool_names(matches)
        assert "entity_vdb_search" in matched_names

    def test_search_returns_max_5(self) -> None:
        """Verify search returns at most 5 results by default."""
        registry = DeferredToolRegistry()
        for i in range(10):
            registry.add(_make_tool(f"tool_{i}", f"Search variant {i}"))

        matches = registry.search("search")
        assert len(matches) <= 5

    def test_search_empty_query(self) -> None:
        """Empty query returns no results."""
        registry = DeferredToolRegistry()
        registry.add(_make_tool("entity_search", "Search entities"))

        matches = registry.search("")
        assert matches == []

    def test_search_no_match(self) -> None:
        """Query with no matching terms returns empty list."""
        registry = DeferredToolRegistry()
        registry.add(_make_tool("entity_search", "Search entities"))

        matches = registry.search("zzz_nonexistent_capability")
        assert matches == []


# ---------------------------------------------------------------------------
# Test: search adds tools to active set
# ---------------------------------------------------------------------------

class TestSearchAddsToolsToActiveSet:
    """After search discovers tools, verify they're available for subsequent calls."""

    def test_search_adds_tools_to_active_set(self) -> None:
        """After search discovers tools, verify they're available."""
        registry = DeferredToolRegistry()
        tool_a = _make_tool("entity_search", "Search for entities in database")
        tool_b = _make_tool("chunk_search", "Search document chunks by keyword")
        registry.add(tool_a)
        registry.add(tool_b)

        # Simulate search and activation
        matches = registry.search("entity")
        activated = registry.activate(matches)
        assert "entity_search" in activated

        # Get activated tools for merging into visible set
        active = registry.get_activated_tools()
        assert _tool_names(active) == {"entity_search"}

    def test_execute_search_activates_tools(self) -> None:
        """execute_search_tool_call activates matching tools."""
        registry = DeferredToolRegistry()
        registry.add(_make_tool("entity_search", "Search entities in vector DB"))
        registry.add(_make_tool("relationship_query", "Query relationships"))

        result_text = execute_search_tool_call(
            registry, {"query": "entity search"},
        )
        result = json.loads(result_text)

        assert len(result["results"]) >= 1
        assert any(r["name"] == "entity_search" for r in result["results"])
        assert "entity_search" in result["activated"]

        # Verify the tool is now in activated set
        assert registry.discovered_count >= 1

    def test_activated_tools_persist_across_searches(self) -> None:
        """Tools activated in earlier searches remain activated."""
        registry = DeferredToolRegistry()
        registry.add(_make_tool("tool_a", "Alpha capability"))
        registry.add(_make_tool("tool_b", "Beta capability"))

        # First search activates tool_a
        execute_search_tool_call(registry, {"query": "alpha"})
        assert "tool_a" in [
            t.get("function", {}).get("name")
            for t in registry.get_activated_tools()
        ]

        # Second search activates tool_b
        execute_search_tool_call(registry, {"query": "beta"})

        # Both should be activated
        active_names = _tool_names(registry.get_activated_tools())
        assert "tool_a" in active_names
        assert "tool_b" in active_names


# ---------------------------------------------------------------------------
# Test: no defer = no search
# ---------------------------------------------------------------------------

class TestNoDeferNoSearch:
    """When no servers defer, verify search_available_tools is NOT added."""

    def test_no_defer_no_search(self) -> None:
        """When no servers defer, verify search tool is NOT added."""
        registry = DeferredToolRegistry()
        # Don't add any tools to the registry

        assert not registry.has_deferred()

        # Simulate _start_servers logic: only add search if deferred
        visible_tools: list[dict[str, Any]] = [
            _make_tool("tool_a"),
            _make_tool("tool_b"),
        ]
        if registry.has_deferred():
            visible_tools.append(search_tool_definition())

        names = _tool_names(visible_tools)
        assert SEARCH_TOOL_NAME not in names


# ---------------------------------------------------------------------------
# Test: deferred metrics in result
# ---------------------------------------------------------------------------

class TestDeferredMetricsInResult:
    """Verify metadata contains deferred tool counts."""

    def test_deferred_metrics_in_result(self) -> None:
        """Verify build_deferred_metrics contains expected keys and values."""
        registry = DeferredToolRegistry()
        # Add 5 tools, activate 2
        for i in range(5):
            registry.add(_make_tool(f"tool_{i}", f"Capability {i}"))

        # Activate 2 via search
        matches = registry.search("capability")
        registry.activate(matches[:2])

        metrics = build_deferred_metrics(registry)

        assert metrics["deferred_tools_total"] == 5
        assert metrics["deferred_tools_discovered"] == 2
        assert isinstance(metrics["deferred_tools_token_savings"], int)
        assert metrics["deferred_tools_token_savings"] > 0

    def test_no_deferred_metrics(self) -> None:
        """Empty registry reports zero metrics."""
        registry = DeferredToolRegistry()
        metrics = build_deferred_metrics(registry)

        assert metrics["deferred_tools_total"] == 0
        assert metrics["deferred_tools_discovered"] == 0
        assert metrics["deferred_tools_token_savings"] == 0

    def test_all_activated_no_savings(self) -> None:
        """When all deferred tools are activated, token savings is 0."""
        registry = DeferredToolRegistry()
        for i in range(3):
            registry.add(_make_tool(f"tool_{i}", f"Capability {i}"))

        # Activate all
        all_tools = registry.search("capability", max_results=10)
        registry.activate(all_tools)

        metrics = build_deferred_metrics(registry)
        assert metrics["deferred_tools_total"] == 3
        assert metrics["deferred_tools_discovered"] == 3
        assert metrics["deferred_tools_token_savings"] == 0


# ---------------------------------------------------------------------------
# Test: keyword matching on name
# ---------------------------------------------------------------------------

class TestAllDeferredKeywordMatch:
    """Search for a tool by name keyword. Verify it matches."""

    def test_keyword_match_on_name(self) -> None:
        """Search by exact name fragment finds the right tool."""
        registry = DeferredToolRegistry()
        registry.add(_make_tool("entity_vdb_search", "Vector database entity search"))
        registry.add(_make_tool("chunk_full_text_search", "Full text chunk search"))
        registry.add(_make_tool("graph_traversal", "Traverse knowledge graph"))

        # Search by name keyword "vdb"
        matches = registry.search("vdb")
        assert len(matches) == 1
        assert _tool_names(matches) == {"entity_vdb_search"}

    def test_keyword_match_on_description(self) -> None:
        """Search by description keyword finds matching tools."""
        registry = DeferredToolRegistry()
        registry.add(_make_tool("tool_a", "Search for entities in a knowledge graph"))
        registry.add(_make_tool("tool_b", "Generate a summary report"))

        matches = registry.search("knowledge graph")
        assert len(matches) >= 1
        assert "tool_a" in _tool_names(matches)

    def test_multi_term_scoring(self) -> None:
        """Tools matching more query terms rank higher."""
        registry = DeferredToolRegistry()
        registry.add(_make_tool("entity_search", "Search entities"))
        registry.add(_make_tool("entity_vdb_search", "Search entities in vector database"))
        registry.add(_make_tool("unrelated", "Something completely different"))

        matches = registry.search("entity vector search")
        # entity_vdb_search matches all 3 terms; entity_search matches 2
        assert len(matches) >= 2
        first_name = matches[0].get("function", {}).get("name", "")
        assert first_name == "entity_vdb_search"

    def test_execute_search_missing_query(self) -> None:
        """execute_search_tool_call handles missing query gracefully."""
        registry = DeferredToolRegistry()
        registry.add(_make_tool("tool_a", "Some tool"))

        result_text = execute_search_tool_call(registry, {})
        result = json.loads(result_text)
        assert "error" in result
        assert result["results"] == []
