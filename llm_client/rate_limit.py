"""Per-provider concurrency limiting for LLM API calls.

Prevents overwhelming any single provider when multiple projects,
batch calls, or concurrent tasks hit the same API simultaneously.

Uses asyncio.Semaphore per provider for async calls, with a threading
wrapper for sync callers. Limits are configurable via configure() or
environment variable LLM_CLIENT_RATE_LIMITS.

Usage::

    from llm_client.rate_limit import acquire, aacquire

    # Async (preferred)
    async with aacquire("gemini/gemini-3-flash"):
        response = await litellm.acompletion(...)

    # Sync
    with acquire("gpt-5-mini"):
        response = litellm.completion(...)
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from contextlib import asynccontextmanager, contextmanager
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
    for prefix, provider in _PROVIDER_PREFIXES.items():
        if model.startswith(prefix):
            return provider
    if any(model.startswith(p) for p in _OPENAI_PREFIXES):
        return "openai"
    # Agent models — not rate-limited (they manage their own concurrency)
    if model.startswith(("claude-code", "codex")):
        return "agent"
    return "default"


# ---------------------------------------------------------------------------
# Default limits (concurrent requests per provider)
# ---------------------------------------------------------------------------

_DEFAULT_LIMITS: dict[str, int] = {
    "openai": 50,
    "google": 30,
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def configure(
    *,
    enabled: bool | None = None,
    limits: dict[str, int] | None = None,
) -> None:
    """Configure rate limiting.

    Args:
        enabled: Enable/disable rate limiting globally.
        limits: Override per-provider concurrent request limits.
            Keys are provider names (openai, google, anthropic, etc.)
            Values are max concurrent requests.
    """
    global _enabled  # noqa: PLW0603
    if enabled is not None:
        _enabled = enabled
    if limits is not None:
        with _lock:
            _limits.update(limits)
            # Clear cached semaphores so they get recreated with new limits
            _async_sems.clear()
            _sync_sems.clear()


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


# ---------------------------------------------------------------------------
# Semaphore accessors
# ---------------------------------------------------------------------------


def _get_limit(provider: str) -> int:
    return _limits.get(provider, _limits.get("default", 30))


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
    sem = _get_async_sem(model)
    await sem.acquire()
    try:
        yield
    finally:
        sem.release()


@contextmanager
def acquire(model: str) -> Iterator[None]:
    """Sync context manager: acquire a slot for the model's provider."""
    if not _enabled:
        yield
        return
    sem = _get_sync_sem(model)
    sem.acquire()
    try:
        yield
    finally:
        sem.release()
