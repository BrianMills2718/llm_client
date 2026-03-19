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
    assert len(llm_client.__all__) == 147

    assert "call_llm" in llm_client._CORE_SUBSTRATE_EXPORTS
    assert "MCPAgentResult" in llm_client._COMPAT_HOLD_EXPORTS
    assert "score_output" in llm_client._CANDIDATE_MOVE_EXPORTS


def test_top_level_declared_exports_resolve_for_star_import_compatibility() -> None:
    """All advertised top-level exports should resolve for star-import compatibility."""
    import llm_client

    namespace: dict[str, object] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        exec("from llm_client import *", {}, namespace)

    assert "configure_logging" in namespace
    assert namespace["configure_logging"] is llm_client.configure_logging
    assert "call_llm" in namespace
    assert namespace["call_llm"] is llm_client.call_llm


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


def test_top_level_git_utils_export_warns_and_resolves() -> None:
    """Deprecated git-utils exports should still resolve with explicit guidance."""
    import llm_client
    from llm_client.git_utils import get_git_head as module_export

    llm_client.__dict__.pop("get_git_head", None)

    with pytest.warns(
        DeprecationWarning,
        match=r"`llm_client\.get_git_head` is deprecated",
    ):
        exported = llm_client.get_git_head

    assert exported is module_export
    assert llm_client.__dict__["get_git_head"] is module_export


def test_explicit_import_of_git_utils_constant_warns_and_resolves() -> None:
    """Deprecated top-level git-utils constants should remain compatible."""
    import llm_client
    from llm_client.git_utils import PROMPT_CHANGE as module_export

    llm_client.__dict__.pop("PROMPT_CHANGE", None)
    namespace: dict[str, object] = {}

    with pytest.warns(
        DeprecationWarning,
        match=r"`llm_client\.PROMPT_CHANGE` is deprecated",
    ):
        exec("from llm_client import PROMPT_CHANGE", {}, namespace)

    assert namespace["PROMPT_CHANGE"] == module_export


def test_top_level_scoring_export_warns_and_resolves() -> None:
    """Deprecated top-level scoring exports should still resolve with guidance."""
    import llm_client
    from llm_client.scoring import score_output as module_export

    llm_client.__dict__.pop("score_output", None)

    with pytest.warns(
        DeprecationWarning,
        match=r"`llm_client\.score_output` is deprecated",
    ):
        exported = llm_client.score_output

    assert exported is module_export
    assert llm_client.__dict__["score_output"] is module_export


def test_explicit_import_of_experiment_eval_export_warns_and_resolves() -> None:
    """Deprecated top-level experiment-eval exports should remain compatible."""
    import llm_client
    from llm_client.experiment_eval import build_gate_signals as module_export

    llm_client.__dict__.pop("build_gate_signals", None)
    namespace: dict[str, object] = {}

    with pytest.warns(
        DeprecationWarning,
        match=r"`llm_client\.build_gate_signals` is deprecated",
    ):
        exec("from llm_client import build_gate_signals", {}, namespace)

    assert namespace["build_gate_signals"] is module_export


def test_top_level_task_graph_export_warns_and_resolves() -> None:
    """Deprecated top-level task-graph exports should still resolve with guidance."""
    import llm_client
    from llm_client.task_graph import load_graph as module_export

    llm_client.__dict__.pop("load_graph", None)

    with pytest.warns(
        DeprecationWarning,
        match=r"`llm_client\.load_graph` is deprecated",
    ):
        exported = llm_client.load_graph

    assert exported is module_export
    assert llm_client.__dict__["load_graph"] is module_export


def test_explicit_import_of_analyzer_export_warns_and_resolves() -> None:
    """Deprecated top-level analyzer exports should remain compatible."""
    import llm_client
    from llm_client.analyzer import analyze_run as module_export

    llm_client.__dict__.pop("analyze_run", None)
    namespace: dict[str, object] = {}

    with pytest.warns(
        DeprecationWarning,
        match=r"`llm_client\.analyze_run` is deprecated",
    ):
        exec("from llm_client import analyze_run", {}, namespace)

    assert namespace["analyze_run"] is module_export
