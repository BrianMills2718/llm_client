"""Opt-in integration smoke for long-thinking background mode.

This test is intentionally gated because it can run for multiple minutes and
incur real API cost.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration

from llm_client import ClientConfig, call_llm


def _env_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _long_thinking_smoke_enabled() -> tuple[bool, str]:
    if os.environ.get("LLM_CLIENT_INTEGRATION", "").strip() != "1":
        return False, "Set LLM_CLIENT_INTEGRATION=1 to enable integration tests."
    if os.environ.get("LLM_CLIENT_LONG_THINKING_SMOKE", "").strip() != "1":
        return False, "Set LLM_CLIENT_LONG_THINKING_SMOKE=1 to enable long-thinking smoke."
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        return False, "OPENAI_API_KEY is required for long-thinking smoke test."
    return True, ""


_ENABLED, _SKIP_REASON = _long_thinking_smoke_enabled()


@pytest.mark.skipif(not _ENABLED, reason=_SKIP_REASON)
def test_gpt52_pro_long_thinking_background_smoke() -> None:
    """Smoke: gpt-5.2-pro with high effort should execute with background mode."""
    timeout = _env_positive_int("LLM_CLIENT_LONG_THINKING_TIMEOUT", 900)
    poll_interval = _env_positive_int("LLM_CLIENT_LONG_THINKING_POLL_INTERVAL", 15)

    result = call_llm(
        "gpt-5.2-pro",
        [
            {
                "role": "user",
                "content": (
                    "In 3 short bullets, explain why test doubles reduce flaky tests. "
                    "Keep total response under 80 words."
                ),
            }
        ],
        reasoning_effort="high",
        background_timeout=timeout,
        background_poll_interval=poll_interval,
        task="integration.long_thinking_smoke",
        trace_id="integration.long_thinking_smoke.gpt52",
        max_budget=0,
        config=ClientConfig(routing_policy="direct"),
    )

    assert isinstance(result.content, str)
    assert len(result.content.strip()) > 0
    assert isinstance(result.routing_trace, dict)
    assert result.routing_trace.get("background_mode") is True
    assert result.requested_model == "gpt-5.2-pro"
