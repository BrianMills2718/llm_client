from __future__ import annotations

import os

import pytest

from llm_client.core.model_selection import (
    resolve_model_chain,
    resolve_model_selection,
    strict_model_policy,
)


def test_resolve_model_selection_uses_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "llm_client.core.model_selection.get_model",
        lambda task, available_only=False, use_performance=True: f"resolved::{task}",
    )

    selection = resolve_model_selection("graph_building")

    assert selection.model == "resolved::graph_building"
    assert selection.source == "task"
    assert selection.strict_models is True


def test_resolve_model_selection_preserves_override() -> None:
    selection = resolve_model_selection(
        "graph_building",
        override_model="openrouter/x-ai/grok-4.1-fast",
        strict_models=False,
    )

    assert selection.model == "openrouter/x-ai/grok-4.1-fast"
    assert selection.source == "override"
    assert selection.strict_models is False


def test_resolve_model_selection_rejects_empty_task() -> None:
    with pytest.raises(ValueError, match="task must be non-empty"):
        resolve_model_selection("   ")


def test_resolve_model_chain_resolves_primary_and_fallback_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get_model(task: str, available_only: bool = False, use_performance: bool = True) -> str:
        del available_only, use_performance
        return {
            "extraction": "gpt-5.2-pro",
            "budget_extraction": "openrouter/deepseek/deepseek-chat",
        }[task]

    monkeypatch.setattr("llm_client.core.model_selection.get_model", fake_get_model)

    chain = resolve_model_chain(
        "extraction",
        fallback_tasks=["budget_extraction", "extraction"],
        fallback_models=["openrouter/deepseek/deepseek-chat", "gemini/gemini-3-flash-preview"],
    )

    assert chain.primary.model == "gpt-5.2-pro"
    assert chain.fallback_tasks == ["budget_extraction", "extraction"]
    assert chain.fallback_models == [
        "openrouter/deepseek/deepseek-chat",
        "gemini/gemini-3-flash-preview",
    ]


def test_strict_model_policy_sets_and_restores_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_CLIENT_STRICT_MODELS", raising=False)

    with strict_model_policy(True):
        assert os.environ["LLM_CLIENT_STRICT_MODELS"] == "1"

    assert "LLM_CLIENT_STRICT_MODELS" not in os.environ


def test_strict_model_policy_restores_prior_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_CLIENT_STRICT_MODELS", "0")

    with strict_model_policy(True):
        assert os.environ["LLM_CLIENT_STRICT_MODELS"] == "1"

    assert os.environ["LLM_CLIENT_STRICT_MODELS"] == "0"
