"""Tests for shared timeout-policy helpers."""

from __future__ import annotations

from llm_client.execution.timeout_policy import (
    default_timeout_for_caller,
    normalize_timeout,
)


def test_normalize_timeout_ban_appends_warning_and_zeroes_timeout(
    monkeypatch,
) -> None:
    """Global timeout ban should zero the timeout and emit one stable warning."""
    monkeypatch.setenv("LLM_CLIENT_TIMEOUT_POLICY", "ban")
    warnings: list[str] = []

    normalized = normalize_timeout(
        120,
        caller="test_timeout_policy",
        warning_sink=warnings,
    )

    assert normalized == 0
    assert warnings == [
        "TIMEOUT_DISABLED[test_timeout_policy]: timeout=120s ignored "
        "(set LLM_CLIENT_TIMEOUT_POLICY=allow to re-enable)."
    ]


def test_normalize_timeout_negative_values_clamp_to_zero(monkeypatch) -> None:
    """Negative timeout values should normalize to zero without warnings."""
    monkeypatch.delenv("LLM_CLIENT_TIMEOUT_POLICY", raising=False)
    warnings: list[str] = []

    normalized = normalize_timeout(
        -5,
        caller="test_timeout_policy",
        warning_sink=warnings,
    )

    assert normalized == 0
    assert warnings == []


def test_default_timeout_for_structured_calls_is_finite(monkeypatch) -> None:
    """Structured calls should inherit a longer finite shared default."""

    monkeypatch.delenv("LLM_CLIENT_DEFAULT_TIMEOUT", raising=False)
    monkeypatch.delenv("LLM_CLIENT_DEFAULT_STRUCTURED_TIMEOUT", raising=False)

    assert default_timeout_for_caller(caller="call_llm_structured") == 180
    assert default_timeout_for_caller(caller="acall_llm_structured") == 180


def test_default_timeout_for_structured_calls_honors_env_override(monkeypatch) -> None:
    """Structured default timeout should stay configurable from env."""

    monkeypatch.setenv("LLM_CLIENT_DEFAULT_STRUCTURED_TIMEOUT", "240")

    assert default_timeout_for_caller(caller="call_llm_structured") == 240
