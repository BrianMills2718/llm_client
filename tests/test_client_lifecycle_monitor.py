"""Tests for lifecycle monitor shutdown behavior.

The async heartbeat monitor is observability infrastructure. Its teardown
should never override a real LLM result or a real model error.
"""

from __future__ import annotations

import asyncio
import logging
import time

import pytest

from llm_client.client import _AsyncLLMCallHeartbeatMonitor


@pytest.mark.asyncio
async def test_async_monitor_stop_logs_and_swallows_monitor_timeout(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Monitor-shutdown failures should be logged, not surfaced as call failures."""

    monitor = _AsyncLLMCallHeartbeatMonitor(
        call_id="call-1",
        call_kind="text",
        caller="test",
        task="unit.test",
        trace_id="trace-1",
        requested_model="gemini/gemini-2.5-flash",
        provider_timeout_s=0,
        prompt_ref=None,
        heartbeat_interval_s=1.0,
        stall_after_s=1.0,
        started_at=time.monotonic(),
    )

    async def _boom() -> None:
        raise asyncio.TimeoutError()

    monitor._task = asyncio.create_task(_boom())

    with caplog.at_level(logging.WARNING):
        await monitor.stop()

    assert "Async heartbeat monitor stop ignored monitor failure" in caplog.text
