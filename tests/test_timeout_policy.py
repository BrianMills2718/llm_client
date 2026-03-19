"""Tests for shared timeout-policy helpers."""

from __future__ import annotations

from llm_client.timeout_policy import normalize_timeout


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
