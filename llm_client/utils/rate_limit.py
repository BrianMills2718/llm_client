"""Per-provider concurrency limiting and cooldown coordination for LLM calls.

Prevents overwhelming any single provider when multiple projects,
batch calls, or concurrent tasks hit the same API simultaneously.

Uses asyncio.Semaphore per provider for async calls, with a threading
wrapper for sync callers. Also coordinates provider cooldown windows
across processes after shared rate-limit errors so separate worktrees
do not keep hammering the same quota surface.

Limits are configurable via configure() or environment variable
``LLM_CLIENT_RATE_LIMITS``. Cooldown behavior is configurable via
``LLM_CLIENT_RATE_LIMIT_COOLDOWN_*`` env vars or configure().

Usage::

    from llm_client.utils.rate_limit import acquire, aacquire

    # Async (preferred)
    async with aacquire("gemini/gemini-3-flash"):
        response = await litellm.acompletion(...)

    # Sync
    with acquire("gpt-5-mini"):
        response = litellm.completion(...)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

_PROVIDER_PREFIXES = {
    "gemini/": "google",
    "anthropic/": "anthropic",
    "deepseek/": "deepseek",
    "xai/": "xai",
    "openrouter/": "openrouter",
    "ollama/": "ollama",
    "together/": "together",
}

_OPENAI_PREFIXES = ("gpt-", "o1-", "o3", "o4-", "text-embedding-")


def _get_provider(model: str) -> str:
    """Extract provider from a model string."""
    # Agent models first — not rate-limited (they manage their own concurrency).
    # Must be checked before OpenAI prefixes because Codex-family models like
    # "gpt-5.3-codex" start with "gpt-" but route to the Codex SDK.
    if model.startswith(("claude-code", "codex")):
        return "agent"
    from llm_client.execution.call_contracts import (
        _is_codex_alias_model,
        _is_codex_family_model,
    )
    if _is_codex_alias_model(model) or _is_codex_family_model(model):
        return "agent"
    for prefix, provider in _PROVIDER_PREFIXES.items():
        if model.startswith(prefix):
            return provider
    if any(model.startswith(p) for p in _OPENAI_PREFIXES):
        return "openai"
    return "default"


# ---------------------------------------------------------------------------
# Default limits (concurrent requests per provider)
# ---------------------------------------------------------------------------

_DEFAULT_LIMITS: dict[str, int] = {
    "openai": 50,
    "google": 10,
    "anthropic": 20,
    "openrouter": 40,
    "deepseek": 20,
    "xai": 20,
    "ollama": 5,
    "together": 20,
    "agent": 999,  # agents manage their own concurrency
    "default": 30,
}

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_async_sems: dict[str, asyncio.Semaphore] = {}
_sync_sems: dict[str, threading.Semaphore] = {}
_limits: dict[str, int] = dict(_DEFAULT_LIMITS)
_enabled = True

_COOLDOWN_ENABLED_ENV = "LLM_CLIENT_RATE_LIMIT_COOLDOWN_ENABLED"
_COOLDOWN_FLOORS_ENV = "LLM_CLIENT_RATE_LIMIT_COOLDOWN_FLOORS"
_COOLDOWN_STATE_PATH_ENV = "LLM_CLIENT_RATE_LIMIT_STATE_PATH"
_COOLDOWN_DB_BUSY_TIMEOUT_MS_ENV = "LLM_CLIENT_RATE_LIMIT_DB_BUSY_TIMEOUT_MS"
_SHARED_LIMITS_ENV = "LLM_CLIENT_RATE_LIMIT_SHARED_LIMITS"
_SHARED_ENABLED_ENV = "LLM_CLIENT_RATE_LIMIT_SHARED_ENABLED"
_SHARED_LEASE_TTL_S_ENV = "LLM_CLIENT_RATE_LIMIT_SHARED_LEASE_TTL_S"
_SHARED_POLL_INTERVAL_S_ENV = "LLM_CLIENT_RATE_LIMIT_SHARED_POLL_INTERVAL_S"
_DEFAULT_COOLDOWN_DB_BUSY_TIMEOUT_MS = 1000
_DEFAULT_COOLDOWN_FLOORS: dict[str, float] = {
    "google": 15.0,
}
_DEFAULT_SHARED_LIMITS: dict[str, int] = {
    "google": 4,
}
_DEFAULT_SHARED_LEASE_TTL_S = 600.0
_DEFAULT_SHARED_POLL_INTERVAL_S = 0.5
_COOLDOWN_WAIT_EPSILON_S = 0.001
_default_data_root = Path(
    os.environ.get("LLM_CLIENT_DATA_ROOT", str(Path.home() / "projects" / "data"))
)
_cooldown_enabled = os.environ.get(_COOLDOWN_ENABLED_ENV, "1") == "1"
_cooldown_floors: dict[str, float] = dict(_DEFAULT_COOLDOWN_FLOORS)
_shared_enabled = os.environ.get(_SHARED_ENABLED_ENV, "1") == "1"
_shared_limits: dict[str, int] = dict(_DEFAULT_SHARED_LIMITS)
_shared_lease_ttl_s = _DEFAULT_SHARED_LEASE_TTL_S
_shared_poll_interval_s = _DEFAULT_SHARED_POLL_INTERVAL_S
_cooldown_state_path = Path(
    os.environ.get(
        _COOLDOWN_STATE_PATH_ENV,
        str(_default_data_root / "llm_rate_limit_state.sqlite3"),
    )
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def configure(
    *,
    enabled: bool | None = None,
    limits: dict[str, int] | None = None,
    cooldown_enabled: bool | None = None,
    cooldown_floors: dict[str, float] | None = None,
    cooldown_path: str | os.PathLike[str] | None = None,
    shared_enabled: bool | None = None,
    shared_limits: dict[str, int] | None = None,
    shared_lease_ttl_s: float | None = None,
    shared_poll_interval_s: float | None = None,
) -> None:
    """Configure rate limiting.

    Args:
        enabled: Enable/disable rate limiting globally.
        limits: Override per-provider concurrent request limits.
            Keys are provider names (openai, google, anthropic, etc.)
            Values are max concurrent requests.
        cooldown_enabled: Enable/disable cross-process provider cooldowns.
        cooldown_floors: Minimum cooldown seconds per provider after a 429.
        cooldown_path: SQLite path for shared cooldown state.
    """
    global _enabled, _cooldown_enabled, _cooldown_state_path  # noqa: PLW0603
    global _shared_enabled, _shared_lease_ttl_s, _shared_poll_interval_s  # noqa: PLW0603
    if enabled is not None:
        _enabled = enabled
    if cooldown_enabled is not None:
        _cooldown_enabled = cooldown_enabled
    if shared_enabled is not None:
        _shared_enabled = shared_enabled
    if limits is not None:
        with _lock:
            _limits.update(limits)
            # Clear cached semaphores so they get recreated with new limits
            _async_sems.clear()
            _sync_sems.clear()
    if cooldown_floors is not None:
        with _lock:
            _cooldown_floors.clear()
            for provider, delay in cooldown_floors.items():
                try:
                    _cooldown_floors[str(provider)] = max(0.0, float(delay))
                except (TypeError, ValueError):
                    logger.warning(
                        "Ignoring invalid cooldown floor for provider %s: %r",
                        provider,
                        delay,
                    )
    if shared_limits is not None:
        with _lock:
            _shared_limits.clear()
            for provider, limit in shared_limits.items():
                try:
                    _shared_limits[str(provider)] = max(0, int(limit))
                except (TypeError, ValueError):
                    logger.warning(
                        "Ignoring invalid shared limit for provider %s: %r",
                        provider,
                        limit,
                    )
    if cooldown_path is not None:
        _cooldown_state_path = Path(cooldown_path)
    if shared_lease_ttl_s is not None:
        _shared_lease_ttl_s = max(1.0, float(shared_lease_ttl_s))
    if shared_poll_interval_s is not None:
        _shared_poll_interval_s = max(0.01, float(shared_poll_interval_s))


def _load_env_limits() -> None:
    """Load limits from LLM_CLIENT_RATE_LIMITS env var (JSON)."""
    raw = os.environ.get("LLM_CLIENT_RATE_LIMITS")
    if raw:
        import json
        try:
            overrides = json.loads(raw)
            if isinstance(overrides, dict):
                _limits.update(overrides)
                logger.debug("Loaded rate limits from env: %s", overrides)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Invalid LLM_CLIENT_RATE_LIMITS env var: %s", raw)


_load_env_limits()


def _load_env_cooldown_floors() -> None:
    """Load provider cooldown floors from JSON env var."""
    raw = os.environ.get(_COOLDOWN_FLOORS_ENV)
    if not raw:
        return
    try:
        overrides = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Invalid %s env var: %s", _COOLDOWN_FLOORS_ENV, raw)
        return
    if not isinstance(overrides, dict):
        logger.warning("Invalid %s env var (expected object): %s", _COOLDOWN_FLOORS_ENV, raw)
        return
    for provider, delay in overrides.items():
        try:
            _cooldown_floors[str(provider)] = max(0.0, float(delay))
        except (TypeError, ValueError):
            logger.warning(
                "Ignoring invalid cooldown floor from env for provider %s: %r",
                provider,
                delay,
            )


_load_env_cooldown_floors()


def _load_env_shared_limits() -> None:
    """Load cross-process provider concurrency caps from JSON env var."""
    raw = os.environ.get(_SHARED_LIMITS_ENV)
    if not raw:
        return
    try:
        overrides = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Invalid %s env var: %s", _SHARED_LIMITS_ENV, raw)
        return
    if not isinstance(overrides, dict):
        logger.warning("Invalid %s env var (expected object): %s", _SHARED_LIMITS_ENV, raw)
        return
    for provider, limit in overrides.items():
        try:
            _shared_limits[str(provider)] = max(0, int(limit))
        except (TypeError, ValueError):
            logger.warning(
                "Ignoring invalid shared limit from env for provider %s: %r",
                provider,
                limit,
            )


def _load_env_shared_runtime_settings() -> None:
    """Load shared-lease TTL and poll cadence from env vars."""
    global _shared_lease_ttl_s, _shared_poll_interval_s  # noqa: PLW0603

    raw_ttl = os.environ.get(_SHARED_LEASE_TTL_S_ENV, "").strip()
    if raw_ttl:
        try:
            _shared_lease_ttl_s = max(1.0, float(raw_ttl))
        except ValueError:
            logger.warning("Invalid %s env var: %s", _SHARED_LEASE_TTL_S_ENV, raw_ttl)

    raw_poll = os.environ.get(_SHARED_POLL_INTERVAL_S_ENV, "").strip()
    if raw_poll:
        try:
            _shared_poll_interval_s = max(0.01, float(raw_poll))
        except ValueError:
            logger.warning("Invalid %s env var: %s", _SHARED_POLL_INTERVAL_S_ENV, raw_poll)


_load_env_shared_limits()
_load_env_shared_runtime_settings()


# ---------------------------------------------------------------------------
# Semaphore accessors
# ---------------------------------------------------------------------------


def _get_limit(provider: str) -> int:
    return _limits.get(provider, _limits.get("default", 30))


def _get_cooldown_floor(provider: str) -> float:
    return max(0.0, float(_cooldown_floors.get(provider, 0.0)))


def _get_shared_limit(provider: str) -> int:
    return max(0, int(_shared_limits.get(provider, 0)))


def _cooldown_db_timeout_s() -> float:
    raw = os.environ.get(_COOLDOWN_DB_BUSY_TIMEOUT_MS_ENV, "").strip()
    if raw.isdigit():
        return max(0.0, int(raw) / 1000.0)
    return _DEFAULT_COOLDOWN_DB_BUSY_TIMEOUT_MS / 1000.0


def _connect_cooldown_db() -> sqlite3.Connection:
    """Open the shared cooldown DB and ensure the schema exists."""
    _cooldown_state_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_cooldown_state_path), timeout=_cooldown_db_timeout_s())
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


def _provider_cooldown_remaining(provider: str) -> float:
    """Return remaining cooldown seconds for *provider* from shared state."""
    if not _cooldown_enabled:
        return 0.0
    if provider in {"agent"}:
        return 0.0
    try:
        with _connect_cooldown_db() as conn:
            row = conn.execute(
                "SELECT cooldown_until FROM provider_cooldowns WHERE provider = ?",
                (provider,),
            ).fetchone()
    except sqlite3.Error:
        logger.debug("Failed reading provider cooldown state", exc_info=True)
        return 0.0
    if row is None:
        return 0.0
    try:
        cooldown_until = float(row[0])
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, cooldown_until - time.time())


def cooldown_remaining(model: str) -> float:
    """Return shared cooldown seconds remaining for the model's provider."""
    return _provider_cooldown_remaining(_get_provider(model))


def _shared_lease_holder() -> str:
    """Return a stable holder identity for observability/debugging."""

    return f"pid:{os.getpid()}-thread:{threading.get_ident()}"


def _provider_uses_shared_limit(provider: str) -> bool:
    """Return whether the provider participates in cross-process leases."""

    return _shared_enabled and provider not in {"agent"} and _get_shared_limit(provider) > 0


def _cleanup_expired_provider_leases(conn: sqlite3.Connection, now: float) -> None:
    """Delete expired provider leases before counting active holders."""

    conn.execute("DELETE FROM provider_leases WHERE expires_at <= ?", (now,))


def _try_acquire_provider_lease(provider: str) -> tuple[str | None, float]:
    """Try to claim one cross-process provider lease.

    Returns:
        (lease_id, wait_s). ``lease_id`` is present on success; otherwise the
        caller should wait ``wait_s`` seconds before trying again.
    """

    if not _provider_uses_shared_limit(provider):
        return None, 0.0

    now = time.time()
    wait_s = max(0.01, _shared_poll_interval_s)
    try:
        conn = _connect_cooldown_db()
    except sqlite3.Error:
        logger.debug("Failed opening shared provider lease DB", exc_info=True)
        return None, wait_s

    try:
        conn.execute("BEGIN IMMEDIATE")
        _cleanup_expired_provider_leases(conn, now)
        limit = _get_shared_limit(provider)
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
                    _shared_lease_holder(),
                    now,
                    now + _shared_lease_ttl_s,
                ),
            )
            conn.commit()
            return lease_id, 0.0
        if rows:
            soonest_expiry = min(float(row[0]) for row in rows)
            wait_s = max(wait_s, soonest_expiry - now)
        conn.commit()
        return None, wait_s
    except sqlite3.Error:
        logger.debug("Failed acquiring shared provider lease", exc_info=True)
        try:
            conn.rollback()
        except sqlite3.Error:
            pass
        return None, wait_s
    finally:
        conn.close()


def _release_provider_lease(lease_id: str | None) -> None:
    """Release one cross-process provider lease, if present."""

    if not lease_id:
        return
    try:
        with _connect_cooldown_db() as conn:
            conn.execute("DELETE FROM provider_leases WHERE lease_id = ?", (lease_id,))
            conn.commit()
    except sqlite3.Error:
        logger.debug("Failed releasing shared provider lease", exc_info=True)


def _wait_for_provider_lease(provider: str) -> str | None:
    """Block until one shared provider lease is available."""

    if not _provider_uses_shared_limit(provider):
        return None
    while True:
        lease_id, wait_s = _try_acquire_provider_lease(provider)
        if lease_id is not None:
            logger.info(
                "Acquired shared provider lease: provider=%s limit=%d lease_ttl=%.1fs",
                provider,
                _get_shared_limit(provider),
                _shared_lease_ttl_s,
            )
            return lease_id
        logger.warning(
            "Waiting for shared provider lease: provider=%s limit=%d wait=%.2fs",
            provider,
            _get_shared_limit(provider),
            wait_s,
        )
        time.sleep(wait_s)


async def _await_provider_lease(provider: str) -> str | None:
    """Async wait until one shared provider lease is available."""

    if not _provider_uses_shared_limit(provider):
        return None
    while True:
        lease_id, wait_s = _try_acquire_provider_lease(provider)
        if lease_id is not None:
            logger.info(
                "Acquired shared provider lease: provider=%s limit=%d lease_ttl=%.1fs",
                provider,
                _get_shared_limit(provider),
                _shared_lease_ttl_s,
            )
            return lease_id
        logger.warning(
            "Waiting for shared provider lease: provider=%s limit=%d wait=%.2fs",
            provider,
            _get_shared_limit(provider),
            wait_s,
        )
        await asyncio.sleep(wait_s)


def register_rate_limit_cooldown(
    model: str,
    delay_s: float | None,
    *,
    source: str = "none",
) -> float:
    """Record a shared provider cooldown after a rate-limit signal.

    Returns the applied cooldown in seconds after combining the explicit delay
    with any configured provider floor.
    """
    provider = _get_provider(model)
    if not _cooldown_enabled or provider == "agent":
        return 0.0

    requested_delay = max(0.0, float(delay_s or 0.0))
    applied_delay = max(requested_delay, _get_cooldown_floor(provider))
    if applied_delay <= 0:
        return 0.0

    now = time.time()
    requested_until = now + applied_delay
    try:
        with _connect_cooldown_db() as conn:
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
        logger.debug("Failed writing provider cooldown state", exc_info=True)
        return applied_delay

    final_delay = max(0.0, cooldown_until - now)
    logger.warning(
        "Registered provider cooldown: provider=%s model=%s delay=%.1fs source=%s",
        provider,
        model,
        final_delay,
        source,
    )
    return final_delay


def _wait_for_provider_cooldown(provider: str) -> None:
    """Block until any shared provider cooldown expires."""
    while True:
        remaining = _provider_cooldown_remaining(provider)
        if remaining <= _COOLDOWN_WAIT_EPSILON_S:
            return
        logger.warning(
            "Waiting for provider cooldown: provider=%s remaining=%.1fs",
            provider,
            remaining,
        )
        started = time.monotonic()
        time.sleep(remaining)
        if time.monotonic() - started < _COOLDOWN_WAIT_EPSILON_S:
            logger.debug(
                "Provider cooldown sleep returned without observable clock progress; "
                "breaking wait loop to avoid busy spin",
            )
            return


async def _await_provider_cooldown(provider: str) -> None:
    """Async wait until any shared provider cooldown expires."""
    while True:
        remaining = _provider_cooldown_remaining(provider)
        if remaining <= _COOLDOWN_WAIT_EPSILON_S:
            return
        logger.warning(
            "Waiting for provider cooldown: provider=%s remaining=%.1fs",
            provider,
            remaining,
        )
        started = time.monotonic()
        await asyncio.sleep(remaining)
        if time.monotonic() - started < _COOLDOWN_WAIT_EPSILON_S:
            logger.debug(
                "Provider cooldown sleep returned without observable clock progress; "
                "breaking async wait loop to avoid busy spin",
            )
            return


def _get_async_sem(model: str) -> asyncio.Semaphore:
    provider = _get_provider(model)
    with _lock:
        if provider not in _async_sems:
            limit = _get_limit(provider)
            _async_sems[provider] = asyncio.Semaphore(limit)
        return _async_sems[provider]


def _get_sync_sem(model: str) -> threading.Semaphore:
    provider = _get_provider(model)
    with _lock:
        if provider not in _sync_sems:
            limit = _get_limit(provider)
            _sync_sems[provider] = threading.Semaphore(limit)
        return _sync_sems[provider]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@asynccontextmanager
async def aacquire(model: str) -> AsyncIterator[None]:
    """Async context manager: acquire a slot for the model's provider."""
    if not _enabled:
        yield
        return
    provider = _get_provider(model)
    await _await_provider_cooldown(provider)
    sem = _get_async_sem(model)
    await sem.acquire()
    lease_id = await _await_provider_lease(provider)
    try:
        yield
    finally:
        _release_provider_lease(lease_id)
        sem.release()


@contextmanager
def acquire(model: str) -> Iterator[None]:
    """Sync context manager: acquire a slot for the model's provider."""
    if not _enabled:
        yield
        return
    provider = _get_provider(model)
    _wait_for_provider_cooldown(provider)
    sem = _get_sync_sem(model)
    sem.acquire()
    lease_id = _wait_for_provider_lease(provider)
    try:
        yield
    finally:
        _release_provider_lease(lease_id)
        sem.release()
