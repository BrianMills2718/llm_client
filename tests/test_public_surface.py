"""Focused tests for the grouped top-level public surface.

These tests verify that ``llm_client.__init__`` now records its public surface
explicitly in grouped categories without changing which names are exported.
"""

from __future__ import annotations

import warnings

import pytest


def test_grouped_exports_flatten_to_public_surface_without_duplicates() -> None:
    """Grouped export categories should flatten to the public surface exactly once."""
    import llm_client

    flattened = [
        *llm_client._CORE_SUBSTRATE_EXPORTS,
        *llm_client._COMPAT_HOLD_EXPORTS,
        *llm_client._CANDIDATE_MOVE_EXPORTS,
    ]

    assert len(flattened) == len(set(flattened))
    assert llm_client.__all__ == flattened
    assert len(flattened) == 124

    assert "call_llm" in llm_client._CORE_SUBSTRATE_EXPORTS
    assert "compare_call_snapshots" in llm_client._CORE_SUBSTRATE_EXPORTS
    assert "get_active_llm_calls" in llm_client._COMPAT_HOLD_EXPORTS
    assert "MCPAgentResult" in llm_client._COMPAT_HOLD_EXPORTS
    assert "PlanningConfig" in llm_client._COMPAT_HOLD_EXPORTS
    assert "AgentPlan" in llm_client._COMPAT_HOLD_EXPORTS
    assert "DifficultyTier" in llm_client._CANDIDATE_MOVE_EXPORTS


def test_top_level_declared_exports_resolve_for_star_import_compatibility() -> None:
    """All advertised top-level exports should resolve for star-import compatibility."""
    import llm_client

    namespace: dict[str, object] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        exec("from llm_client import *", {}, namespace)

    assert "configure_logging" in namespace
    assert namespace["configure_logging"] is llm_client.configure_logging
    assert "get_active_llm_calls" in namespace
    assert namespace["get_active_llm_calls"] is llm_client.get_active_llm_calls
    assert "compare_call_snapshots" in namespace
    assert namespace["compare_call_snapshots"] is llm_client.compare_call_snapshots
    assert "call_llm" in namespace
    assert namespace["call_llm"] is llm_client.call_llm
    assert "PlanningConfig" in namespace
    assert namespace["PlanningConfig"] is llm_client.PlanningConfig
    assert "AgentPlan" in namespace
    assert namespace["AgentPlan"] is llm_client.AgentPlan


def test_top_level_difficulty_export_warns_and_resolves() -> None:
    """Deprecated difficulty exports should still resolve with explicit guidance."""
    import llm_client
    from llm_client.difficulty import get_model_for_difficulty as module_export

    llm_client.__dict__.pop("get_model_for_difficulty", None)

    with pytest.warns(
        DeprecationWarning,
        match=r"`llm_client\.get_model_for_difficulty` is deprecated",
    ):
        exported = llm_client.get_model_for_difficulty

    assert exported is module_export
    assert llm_client.__dict__["get_model_for_difficulty"] is module_export


def test_explicit_import_of_non_all_difficulty_helper_warns_and_resolves() -> None:
    """De facto top-level difficulty helpers should remain compatible during migration."""
    import llm_client
    from llm_client.difficulty import (
        get_model_candidates_for_difficulty as module_export,
    )

    llm_client.__dict__.pop("get_model_candidates_for_difficulty", None)
    namespace: dict[str, object] = {}

    with pytest.warns(
        DeprecationWarning,
        match=r"`llm_client\.get_model_candidates_for_difficulty` is deprecated",
    ):
        exec(
            "from llm_client import get_model_candidates_for_difficulty",
            {},
            namespace,
        )

    assert namespace["get_model_candidates_for_difficulty"] is module_export


# Tests for git_utils, scoring, experiment_eval, task_graph, analyzer
# removed — those modules relocated (Plan #17).
