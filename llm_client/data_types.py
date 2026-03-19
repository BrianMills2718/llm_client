"""Core data types and cache infrastructure for LLM client.

Houses the fundamental value types (LLMCallResult, EmbeddingResult),
cache protocols (CachePolicy, AsyncCachePolicy), and the built-in
LRU cache implementation used across all call paths.

This module is a leaf dependency — it must not import from client.py
or any runtime module to avoid circular imports.
"""

from __future__ import annotations

import hashlib
import inspect
import json as _json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class LLMCallResult:
    """Result from an LLM call. Returned by all call_llm* functions.

    Attributes:
        content: The text response from the model
        usage: Token counts (prompt_tokens, completion_tokens, total_tokens)
        cost: Cost in USD for this call
        model: The model string that was used
        tool_calls: List of tool calls if the model invoked tools, else empty
        finish_reason: Why the model stopped: "stop", "length", "tool_calls",
                       "content_filter", etc. Empty string if unavailable.
        raw_response: The full litellm response object for edge cases
                      (e.g., accessing provider-specific data like Claude
                      thinking blocks). Excluded from repr to keep logs clean.
    """

    content: str
    usage: dict[str, Any]
    cost: float
    model: str
    requested_model: str | None = None
    """Raw model string provided at the public API boundary."""
    resolved_model: str | None = None
    """Best-effort model string used for the successful terminal attempt."""
    execution_model: str | None = None
    """Alias for resolved terminal model, kept additive for migration clarity."""
    routing_trace: dict[str, Any] | None = field(default=None, repr=False)
    """Optional routing/fallback trace for contract characterization and debugging."""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = ""
    raw_response: Any = field(default=None, repr=False)
    warnings: list[str] = field(default_factory=list)
    """Diagnostic warnings accumulated during retry/fallback/routing.
    Empty list on clean calls. Populated with RETRY/FALLBACK/STICKY_FALLBACK
    messages when non-obvious decisions occurred."""
    warning_records: list[dict[str, Any]] = field(default_factory=list, repr=False)
    """Machine-readable warning records (code/category/message/remediation)."""
    full_text: str | None = field(default=None, repr=False)
    """For agent SDKs: full conversation text (all assistant messages).
    ``content`` holds only the final assistant message.
    None for non-agent calls."""
    cost_source: str = "unspecified"
    """How cost was determined: provider_reported, computed, fallback_estimate, cache_hit, etc."""
    billing_mode: str = "api_metered"
    """Billing mode: api_metered, subscription_included, or unknown."""
    marginal_cost: float | None = None
    """Incremental cost attributed to this call; defaults to ``cost`` when omitted."""
    cache_hit: bool = False
    """Whether this result came from cache instead of a model call."""

    def __post_init__(self) -> None:
        if self.marginal_cost is None:
            self.marginal_cost = 0.0 if self.cache_hit else float(self.cost)
        if self.execution_model is None and self.resolved_model is not None:
            self.execution_model = self.resolved_model


@dataclass
class EmbeddingResult:
    """Result from an embedding call.

    Attributes:
        embeddings: List of embedding vectors (one per input text)
        usage: Token counts (prompt_tokens, total_tokens)
        cost: Cost in USD for this call
        model: The model string that was used
    """

    embeddings: list[list[float]]
    usage: dict[str, Any]
    cost: float
    model: str


# ---------------------------------------------------------------------------
# Cache infrastructure
# ---------------------------------------------------------------------------


@runtime_checkable
class CachePolicy(Protocol):
    """Protocol for LLM response caches. Implement get/set for custom backends."""

    def get(self, key: str) -> LLMCallResult | None: ...
    def set(self, key: str, value: LLMCallResult) -> None: ...


@runtime_checkable
class AsyncCachePolicy(Protocol):
    """Protocol for async LLM response caches (Redis, etc.).

    Async functions accept either ``CachePolicy`` or ``AsyncCachePolicy``.
    When an ``AsyncCachePolicy`` is detected, ``await`` is used for get/set
    so the event loop is never blocked.
    """

    async def get(self, key: str) -> LLMCallResult | None: ...
    async def set(self, key: str, value: LLMCallResult) -> None: ...


class LRUCache:
    """Thread-safe in-memory LRU cache for LLM responses.

    Args:
        maxsize: Maximum number of entries. Oldest evicted on overflow.
        ttl: Time-to-live in seconds. ``None`` means entries never expire.
    """

    def __init__(self, maxsize: int = 128, ttl: float | None = None) -> None:
        self._cache: OrderedDict[str, tuple[LLMCallResult, float]] = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl
        self._lock = threading.Lock()

    def get(self, key: str) -> LLMCallResult | None:
        with self._lock:
            if key not in self._cache:
                return None
            value, ts = self._cache[key]
            if self._ttl is not None and time.monotonic() - ts > self._ttl:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return value

    def set(self, key: str, value: LLMCallResult) -> None:
        with self._lock:
            self._cache[key] = (value, time.monotonic())
            self._cache.move_to_end(key)
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


def _cache_key(model: str, messages: list[dict[str, Any]], **kwargs: Any) -> str:
    """Build a deterministic cache key from call parameters."""
    key_data = _json.dumps({"model": model, "messages": messages, **kwargs}, sort_keys=True)
    return hashlib.sha256(key_data.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Async cache helpers
# ---------------------------------------------------------------------------


async def _async_cache_get(cache: Any, key: str) -> LLMCallResult | None:
    """Get from cache, awaiting if the cache is async."""
    result = cache.get(key)
    if inspect.isawaitable(result):
        awaited = await result
        return awaited if isinstance(awaited, LLMCallResult) else None
    return result if isinstance(result, LLMCallResult) else None


async def _async_cache_set(cache: Any, key: str, value: LLMCallResult) -> None:
    """Set into cache, awaiting if the cache is async."""
    result = cache.set(key, value)
    if inspect.isawaitable(result):
        await result
