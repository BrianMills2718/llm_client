"""Tests for observability defaults and agent retry safety switches."""

from llm_client.client import (
    _agent_retry_safe_enabled,
    _build_model_chain,
    _require_tags,
)


def test_require_tags_defaults_when_not_strict(monkeypatch) -> None:
    monkeypatch.delenv("LLM_CLIENT_REQUIRE_TAGS", raising=False)
    monkeypatch.delenv("CI", raising=False)

    task, trace_id, max_budget, warnings = _require_tags(
        None, None, None, caller="test_call",
    )

    assert task == "adhoc"
    assert trace_id.startswith("auto/test_call/")
    assert max_budget == 0.0
    assert "AUTO_TAG: task=adhoc" in warnings
    assert "AUTO_TAG: max_budget=0 (unlimited)" in warnings


def test_require_tags_strict_raises(monkeypatch) -> None:
    monkeypatch.setenv("LLM_CLIENT_REQUIRE_TAGS", "1")
    try:
        _require_tags(None, None, None, caller="test_call")
    except ValueError as exc:
        assert "Missing required kwargs" in str(exc)
    else:
        raise AssertionError("Expected ValueError in strict tag mode")


def test_openrouter_default_routing_for_bare_gpt5(monkeypatch) -> None:
    monkeypatch.delenv("LLM_CLIENT_OPENROUTER_ROUTING", raising=False)
    assert _build_model_chain("gpt-5-mini", None) == ["openrouter/openai/gpt-5-mini"]


def test_agent_retry_safe_switch(monkeypatch) -> None:
    monkeypatch.delenv("LLM_CLIENT_AGENT_RETRY_SAFE", raising=False)
    assert _agent_retry_safe_enabled(None) is False
    assert _agent_retry_safe_enabled(True) is True

    monkeypatch.setenv("LLM_CLIENT_AGENT_RETRY_SAFE", "1")
    assert _agent_retry_safe_enabled(None) is True
