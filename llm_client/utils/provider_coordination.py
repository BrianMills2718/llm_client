"""Shared provider-coordination backends for cooldowns and leases."""

from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class LeaseAttempt:
    """Result of trying to acquire one shared provider lease."""

    lease_id: str | None
    wait_s: float


class ProviderCoordinationBackend(Protocol):
    """Storage/enforcement boundary for provider cooldown and lease state."""

    def cooldown_remaining(self, provider: str) -> float:
        """Return remaining cooldown seconds for *provider*."""

    def register_cooldown(self, provider: str, delay_s: float, *, source: str) -> float:
        """Record a cooldown and return remaining cooldown seconds after the write."""

    def try_acquire_lease(
        self,
        provider: str,
        *,
        shared_limit: int,
        lease_ttl_s: float,
        holder: str,
        poll_interval_s: float,
    ) -> LeaseAttempt:
        """Try to acquire one shared lease for *provider*."""

    def release_lease(self, lease_id: str | None) -> None:
        """Release one previously acquired shared lease."""


class SQLiteProviderCoordinationBackend:
    """SQLite-backed provider coordination for one-machine multi-process runners."""

    def __init__(self, path: Path, *, busy_timeout_s: float) -> None:
        self._path = Path(path)
        self._busy_timeout_s = max(0.0, float(busy_timeout_s))

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._path), timeout=self._busy_timeout_s)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS provider_cooldowns (
                provider TEXT PRIMARY KEY,
                cooldown_until REAL NOT NULL,
                source TEXT,
                updated_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS provider_leases (
                lease_id TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                holder TEXT,
                acquired_at REAL NOT NULL,
                expires_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_provider_leases_provider_expires "
            "ON provider_leases(provider, expires_at)"
        )
        conn.commit()
        return conn

    def cooldown_remaining(self, provider: str) -> float:
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT cooldown_until FROM provider_cooldowns WHERE provider = ?",
                    (provider,),
                ).fetchone()
        except sqlite3.Error:
            return 0.0
        if row is None:
            return 0.0
        try:
            cooldown_until = float(row[0])
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, cooldown_until - time.time())

    def register_cooldown(self, provider: str, delay_s: float, *, source: str) -> float:
        delay = max(0.0, float(delay_s))
        if delay <= 0:
            return 0.0
        now = time.time()
        requested_until = now + delay
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT cooldown_until FROM provider_cooldowns WHERE provider = ?",
                    (provider,),
                ).fetchone()
                existing_until = float(row[0]) if row is not None else 0.0
                cooldown_until = max(existing_until, requested_until)
                conn.execute(
                    """
                    INSERT INTO provider_cooldowns(provider, cooldown_until, source, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(provider) DO UPDATE SET
                        cooldown_until = excluded.cooldown_until,
                        source = excluded.source,
                        updated_at = excluded.updated_at
                    """,
                    (provider, cooldown_until, source, now),
                )
                conn.commit()
        except sqlite3.Error:
            return delay
        return max(0.0, cooldown_until - now)

    def try_acquire_lease(
        self,
        provider: str,
        *,
        shared_limit: int,
        lease_ttl_s: float,
        holder: str,
        poll_interval_s: float,
    ) -> LeaseAttempt:
        limit = max(0, int(shared_limit))
        if limit <= 0:
            return LeaseAttempt(lease_id=None, wait_s=0.0)

        now = time.time()
        wait_s = max(0.01, float(poll_interval_s))
        try:
            conn = self._connect()
        except sqlite3.Error:
            return LeaseAttempt(lease_id=None, wait_s=wait_s)

        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM provider_leases WHERE expires_at <= ?", (now,))
            rows = conn.execute(
                "SELECT expires_at FROM provider_leases WHERE provider = ? ORDER BY expires_at ASC",
                (provider,),
            ).fetchall()
            active = len(rows)
            if active < limit:
                lease_id = uuid.uuid4().hex
                conn.execute(
                    """
                    INSERT INTO provider_leases(
                        lease_id,
                        provider,
                        holder,
                        acquired_at,
                        expires_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        lease_id,
                        provider,
                        holder,
                        now,
                        now + max(1.0, float(lease_ttl_s)),
                    ),
                )
                conn.commit()
                return LeaseAttempt(lease_id=lease_id, wait_s=0.0)
            if rows:
                soonest_expiry = min(float(row[0]) for row in rows)
                wait_s = max(wait_s, soonest_expiry - now)
            conn.commit()
            return LeaseAttempt(lease_id=None, wait_s=wait_s)
        except sqlite3.Error:
            try:
                conn.rollback()
            except sqlite3.Error:
                pass
            return LeaseAttempt(lease_id=None, wait_s=wait_s)
        finally:
            conn.close()

    def release_lease(self, lease_id: str | None) -> None:
        if not lease_id:
            return
        try:
            with self._connect() as conn:
                conn.execute("DELETE FROM provider_leases WHERE lease_id = ?", (lease_id,))
                conn.commit()
        except sqlite3.Error:
            return
