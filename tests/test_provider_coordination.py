"""Tests for provider coordination backends."""

import time

from llm_client.utils.provider_coordination import SQLiteProviderCoordinationBackend


def test_shared_cap_enforced_across_processes(tmp_path) -> None:
    backend = SQLiteProviderCoordinationBackend(
        tmp_path / "coordination.db",
        busy_timeout_s=0.1,
    )

    first = backend.try_acquire_lease(
        "google",
        shared_limit=1,
        lease_ttl_s=5.0,
        holder="holder-1",
        poll_interval_s=0.01,
    )
    second = backend.try_acquire_lease(
        "google",
        shared_limit=1,
        lease_ttl_s=5.0,
        holder="holder-2",
        poll_interval_s=0.01,
    )

    assert first.lease_id is not None
    assert second.lease_id is None
    assert second.wait_s > 0

    backend.release_lease(first.lease_id)


def test_expired_leases_are_reclaimed(tmp_path) -> None:
    backend = SQLiteProviderCoordinationBackend(
        tmp_path / "coordination.db",
        busy_timeout_s=0.1,
    )

    with backend._connect() as conn:  # noqa: SLF001 - verifying SQLite backend behavior directly
        now = time.time()
        conn.execute(
            """
            INSERT INTO provider_leases(
                lease_id, provider, holder, acquired_at, expires_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            ("expired", "google", "holder", now - 10, now - 1),
        )
        conn.commit()

    attempt = backend.try_acquire_lease(
        "google",
        shared_limit=1,
        lease_ttl_s=5.0,
        holder="holder-2",
        poll_interval_s=0.01,
    )

    assert attempt.lease_id is not None
    backend.release_lease(attempt.lease_id)


def test_cooldown_registration_is_monotonic(tmp_path) -> None:
    backend = SQLiteProviderCoordinationBackend(
        tmp_path / "coordination.db",
        busy_timeout_s=0.1,
    )

    first = backend.register_cooldown("google", 0.2, source="test-floor")
    second = backend.register_cooldown("google", 0.1, source="test-shorter")

    assert first > 0
    assert second >= 0.05
    assert backend.cooldown_remaining("google") > 0
