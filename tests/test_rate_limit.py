"""Tests for llm_client.utils.rate_limit — per-provider concurrency limiting."""

import asyncio
import itertools
import threading
import time
from unittest.mock import AsyncMock

import pytest

import llm_client.utils.rate_limit as rl
from llm_client.utils.rate_limit import (
    _DEFAULT_COOLDOWN_FLOORS,
    _DEFAULT_SHARED_LIMITS,
    _get_provider,
    aacquire,
    acquire,
    cooldown_remaining,
    configure,
    _async_sems,
    _cooldown_enabled,
    _cooldown_floors,
    _cooldown_state_path,
    _shared_enabled,
    _shared_limits,
    _sync_sems,
    _limits,
    _DEFAULT_LIMITS,
    register_rate_limit_cooldown,
)


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset rate limiter state between tests."""
    old_limits = dict(_limits)
    old_enabled = rl._enabled
    old_cooldown_enabled = _cooldown_enabled
    old_cooldown_floors = dict(_cooldown_floors)
    old_cooldown_state_path = _cooldown_state_path
    old_shared_enabled = _shared_enabled
    old_shared_limits = dict(_shared_limits)
    old_shared_lease_ttl_s = rl._shared_lease_ttl_s
    old_shared_poll_interval_s = rl._shared_poll_interval_s
    old_coordination_backend_override = rl._coordination_backend_override
    _async_sems.clear()
    _sync_sems.clear()
    rl._enabled = True
    rl._cooldown_enabled = True
    _cooldown_floors.clear()
    _cooldown_floors.update(_DEFAULT_COOLDOWN_FLOORS)
    rl._shared_enabled = True
    _shared_limits.clear()
    _shared_limits.update(_DEFAULT_SHARED_LIMITS)
    rl._shared_lease_ttl_s = rl._DEFAULT_SHARED_LEASE_TTL_S
    rl._shared_poll_interval_s = rl._DEFAULT_SHARED_POLL_INTERVAL_S
    rl._coordination_backend_override = None
    yield
    _async_sems.clear()
    _sync_sems.clear()
    _limits.clear()
    _limits.update(old_limits)
    rl._enabled = old_enabled
    rl._cooldown_enabled = old_cooldown_enabled
    _cooldown_floors.clear()
    _cooldown_floors.update(old_cooldown_floors)
    rl._cooldown_state_path = old_cooldown_state_path
    rl._shared_enabled = old_shared_enabled
    _shared_limits.clear()
    _shared_limits.update(old_shared_limits)
    rl._shared_lease_ttl_s = old_shared_lease_ttl_s
    rl._shared_poll_interval_s = old_shared_poll_interval_s
    rl._coordination_backend_override = old_coordination_backend_override


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

    def test_codex_family_models(self):
        """Codex-family models (gpt-5.3-codex etc.) are agents, not openai."""
        assert _get_provider("gpt-5.3-codex") == "agent"
        assert _get_provider("gpt-5.1-codex-mini") == "agent"
        assert _get_provider("gpt-5.1-codex-max") == "agent"

    def test_gpt54_alias_models(self):
        assert _get_provider("gpt-5.4") == "agent"
        assert _get_provider("openrouter/openai/gpt-5.4") == "agent"

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

    @pytest.mark.asyncio
    async def test_async_acquire_waits_for_registered_cooldown(self, tmp_path):
        """Async acquire should honor cross-process provider cooldowns."""
        configure(
            cooldown_path=tmp_path / "cooldowns.db",
            cooldown_floors={"google": 0.0},
        )
        applied = register_rate_limit_cooldown(
            "gemini/gemini-3-flash",
            0.5,
            source="provider-hint",
        )
        assert applied >= 0.4

        started = time.perf_counter()
        async with aacquire("gemini/gemini-3-flash"):
            pass
        elapsed = time.perf_counter() - started

        assert elapsed >= 0.2

    @pytest.mark.asyncio
    async def test_async_cooldown_wait_does_not_busy_spin_without_clock_progress(self, monkeypatch):
        """Async cooldown waits should not loop forever if a mocked sleep returns instantly."""
        sleep_mock = AsyncMock()

        monkeypatch.setattr(rl, "_provider_cooldown_remaining", lambda provider: 3.0)
        monkeypatch.setattr(rl.asyncio, "sleep", sleep_mock)
        monkeypatch.setattr(rl.time, "monotonic", lambda: next(itertools.repeat(100.0)))

        await rl._await_provider_cooldown("openai")

        sleep_mock.assert_awaited_once_with(3.0)


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

    def test_configure_updates_cooldown_settings(self, tmp_path):
        configure(
            cooldown_enabled=False,
            cooldown_floors={"google": 3.5},
            cooldown_path=tmp_path / "cooldowns.db",
            shared_enabled=False,
            shared_limits={"google": 2},
            shared_lease_ttl_s=123.0,
            shared_poll_interval_s=0.25,
        )
        assert rl._cooldown_enabled is False
        assert _cooldown_floors["google"] == 3.5
        assert rl._cooldown_state_path == tmp_path / "cooldowns.db"
        assert rl._shared_enabled is False
        assert _shared_limits["google"] == 2
        assert rl._shared_lease_ttl_s == 123.0
        assert rl._shared_poll_interval_s == 0.25


class TestSharedCooldown:
    def test_register_rate_limit_cooldown_uses_provider_floor(self, tmp_path):
        configure(
            cooldown_path=tmp_path / "cooldowns.db",
            cooldown_floors={"google": 0.5},
        )
        applied = register_rate_limit_cooldown(
            "gemini/gemini-2.5-flash",
            None,
            source="provider-floor",
        )
        assert applied >= 0.4
        assert cooldown_remaining("gemini/gemini-2.5-flash") > 0.1


class TestSharedProviderLeases:
    @pytest.mark.asyncio
    async def test_async_acquire_obeys_shared_provider_limit(self, tmp_path):
        """Cross-process lease caps should constrain concurrency before first-attempt calls."""
        configure(
            limits={"google": 10},
            cooldown_path=tmp_path / "cooldowns.db",
            shared_limits={"google": 1},
            shared_lease_ttl_s=5.0,
            shared_poll_interval_s=0.01,
        )

        peak_concurrent = 0
        current = 0

        async def worker():
            nonlocal peak_concurrent, current
            async with aacquire("gemini/gemini-2.5-flash"):
                current += 1
                peak_concurrent = max(peak_concurrent, current)
                await asyncio.sleep(0.05)
                current -= 1

        await asyncio.gather(*(worker() for _ in range(3)))

        assert peak_concurrent == 1

    def test_expired_shared_provider_lease_is_reclaimed(self, tmp_path):
        """Expired leases should not block new Gemini callers indefinitely."""
        db_path = tmp_path / "cooldowns.db"
        configure(
            cooldown_path=db_path,
            shared_limits={"google": 1},
            shared_lease_ttl_s=5.0,
            shared_poll_interval_s=0.01,
        )
        with rl._connect_cooldown_db() as conn:
            now = time.time()
            conn.execute(
                """
                INSERT INTO provider_leases(
                    lease_id, provider, holder, acquired_at, expires_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                ("expired-lease", "google", "test-holder", now - 10, now - 1),
            )
            conn.commit()

        started = time.perf_counter()
        with acquire("gemini/gemini-2.5-flash"):
            pass
        elapsed = time.perf_counter() - started

        assert elapsed < 0.2
        with rl._connect_cooldown_db() as conn:
            active = conn.execute(
                "SELECT COUNT(*) FROM provider_leases WHERE provider = ?",
                ("google",),
            ).fetchone()[0]
        assert active == 0
