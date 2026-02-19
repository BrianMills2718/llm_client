"""Tests for llm_client.rate_limit — per-provider concurrency limiting."""

import asyncio
import threading
import time

import pytest

from llm_client import rate_limit as rl
from llm_client.rate_limit import (
    _get_provider,
    aacquire,
    acquire,
    configure,
    _async_sems,
    _sync_sems,
    _limits,
    _DEFAULT_LIMITS,
)


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset rate limiter state between tests."""
    old_limits = dict(_limits)
    old_enabled = rl._enabled
    _async_sems.clear()
    _sync_sems.clear()
    rl._enabled = True
    yield
    _async_sems.clear()
    _sync_sems.clear()
    _limits.clear()
    _limits.update(old_limits)
    rl._enabled = old_enabled


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------


class TestProviderDetection:
    def test_google(self):
        assert _get_provider("gemini/gemini-3-flash") == "google"

    def test_openai(self):
        assert _get_provider("gpt-5-mini") == "openai"
        assert _get_provider("gpt-5") == "openai"
        assert _get_provider("o4-mini") == "openai"
        assert _get_provider("text-embedding-3-small") == "openai"

    def test_anthropic(self):
        assert _get_provider("anthropic/claude-sonnet-4-5") == "anthropic"

    def test_openrouter(self):
        assert _get_provider("openrouter/openai/gpt-5") == "openrouter"

    def test_deepseek(self):
        assert _get_provider("deepseek/deepseek-chat") == "deepseek"

    def test_agent_models(self):
        assert _get_provider("claude-code") == "agent"
        assert _get_provider("codex") == "agent"
        assert _get_provider("claude-code/opus") == "agent"

    def test_unknown_default(self):
        assert _get_provider("some-unknown-model") == "default"


# ---------------------------------------------------------------------------
# Sync acquire
# ---------------------------------------------------------------------------


class TestSyncAcquire:
    def test_basic_acquire_release(self):
        with acquire("gpt-5-mini"):
            pass  # should not raise

    def test_concurrency_limited(self):
        """Verify the semaphore actually limits concurrency."""
        configure(limits={"openai": 2})

        acquired = []
        barrier = threading.Barrier(3, timeout=2)

        def worker():
            with acquire("gpt-5-mini"):
                acquired.append(True)
                try:
                    barrier.wait()
                except threading.BrokenBarrierError:
                    pass
                time.sleep(0.05)

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()

        # Give threads time to try acquiring
        time.sleep(0.2)
        # With limit=2, only 2 should acquire simultaneously
        # (the 3rd waits). All 3 eventually complete.
        for t in threads:
            t.join(timeout=3)

        assert len(acquired) == 3  # all eventually acquired


# ---------------------------------------------------------------------------
# Async acquire
# ---------------------------------------------------------------------------


class TestAsyncAcquire:
    @pytest.mark.asyncio
    async def test_basic_async_acquire(self):
        async with aacquire("gemini/gemini-3-flash"):
            pass  # should not raise

    @pytest.mark.asyncio
    async def test_concurrency_limited_async(self):
        """Verify async semaphore limits concurrency."""
        configure(limits={"google": 2})

        peak_concurrent = 0
        current = 0

        async def worker():
            nonlocal peak_concurrent, current
            async with aacquire("gemini/gemini-3-flash"):
                current += 1
                peak_concurrent = max(peak_concurrent, current)
                await asyncio.sleep(0.05)
                current -= 1

        await asyncio.gather(*(worker() for _ in range(5)))

        assert peak_concurrent <= 2


# ---------------------------------------------------------------------------
# Configure
# ---------------------------------------------------------------------------


class TestConfigure:
    def test_disable(self):
        configure(enabled=False)
        # Should be a no-op (doesn't block)
        with acquire("gpt-5-mini"):
            pass

    def test_custom_limits(self):
        configure(limits={"openai": 1})
        assert _limits["openai"] == 1

    def test_configure_clears_semaphores(self):
        # Create a semaphore
        with acquire("gpt-5-mini"):
            pass
        assert "openai" in _sync_sems

        # Reconfigure — should clear
        configure(limits={"openai": 99})
        assert "openai" not in _sync_sems
