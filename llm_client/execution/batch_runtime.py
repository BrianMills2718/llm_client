"""Batch execution internals extracted from client.py.

Includes batch-level progress tracking, stagnation detection, and
per-item timeout. See Plan #14.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, TypeVar, cast

from pydantic import BaseModel, Field

try:
    from data_contracts import boundary, BoundaryModel
except ImportError:
    def boundary(*args, **kwargs):  # type: ignore[misc]
        def decorator(fn):  # type: ignore[misc]
            return fn
        return decorator
    from pydantic import BaseModel as BoundaryModel  # type: ignore[assignment]

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class BatchStagnationEvent(BoundaryModel):
    """Typed event emitted when a batch run detects consecutive identical errors."""

    model_config = {"extra": "forbid"}

    error_hash: str = Field(description="Hash of the repeated error (type:message[:200])")
    stagnation_window: int = Field(description="Number of consecutive identical errors that triggered this event")
    items_completed: int = Field(description="Items successfully completed before stagnation was detected")
    items_total: int = Field(description="Total items in the batch")


# ---------------------------------------------------------------------------
# Batch progress tracking (Plan #14, Step 1)
# ---------------------------------------------------------------------------


@dataclass
class BatchProgressTracker:
    """Aggregates per-item results into batch-level progress."""

    total: int
    completed: int = 0
    errored: int = 0
    total_latency_s: float = 0.0
    started_at: float = field(default_factory=time.monotonic)
    _error_hashes: list[str] = field(default_factory=list)
    _stagnation_window: int = 5

    @property
    def pending(self) -> int:
        return self.total - self.completed - self.errored

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def avg_latency_s(self) -> float | None:
        return (self.total_latency_s / self.completed) if self.completed > 0 else None

    @property
    def completion_rate(self) -> float:
        return (self.completed + self.errored) / self.total if self.total > 0 else 0.0

    def record_completion(self, latency_s: float) -> None:
        """Record a successful item completion."""
        self.completed += 1
        self.total_latency_s += latency_s
        self._error_hashes.clear()

    def record_error(self, error: Exception) -> bool:
        """Record an item error. Returns True if stagnation detected."""
        self.errored += 1
        error_hash = str(type(error).__name__) + ":" + str(error)[:200]
        self._error_hashes.append(error_hash)
        if len(self._error_hashes) > self._stagnation_window:
            self._error_hashes = self._error_hashes[-self._stagnation_window:]
        if len(self._error_hashes) >= self._stagnation_window:
            return len(set(self._error_hashes)) == 1
        return False

    def summary(self) -> dict[str, Any]:
        """Return a snapshot dict for logging."""
        return {
            "total": self.total,
            "completed": self.completed,
            "errored": self.errored,
            "pending": self.pending,
            "elapsed_s": round(self.elapsed_s, 2),
            "avg_latency_s": round(self.avg_latency_s, 3) if self.avg_latency_s else None,
            "completion_rate": round(self.completion_rate, 4),
        }


@boundary(
    name="llm_client.batch_stagnation_event",
    producer="llm_client",
    consumers=["agentic_scaffolding"],
)
def emit_batch_stagnation_event(tracker: BatchProgressTracker) -> BatchStagnationEvent:
    """Extract and return a typed stagnation event from a BatchProgressTracker.

    Called when stagnation is detected (N consecutive identical errors).
    Consumers can inspect this event to trigger circuit-breaker logic.
    """
    error_hash = tracker._error_hashes[-1] if tracker._error_hashes else ""
    return BatchStagnationEvent(
        error_hash=error_hash,
        stagnation_window=tracker._stagnation_window,
        items_completed=tracker.completed,
        items_total=tracker.total,
    )

if TYPE_CHECKING:
    from llm_client.core.config import ClientConfig
    from llm_client.core.client import (
        AsyncCachePolicy,
        CachePolicy,
        Hooks,
        LLMCallResult,
        RetryPolicy,
    )


async def acall_llm_batch_impl(
    model: str,
    messages_list: list[list[dict[str, Any]]],
    *,
    max_concurrent: int = 5,
    return_exceptions: bool = False,
    on_item_complete: Callable[[int, LLMCallResult], None] | None = None,
    on_item_error: Callable[[int, Exception], None] | None = None,
    timeout: int = 60,
    num_retries: int = 2,
    reasoning_effort: str | None = None,
    api_base: str | None = None,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_on: list[str] | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    cache: CachePolicy | AsyncCachePolicy | None = None,
    retry: RetryPolicy | None = None,
    fallback_models: list[str] | None = None,
    on_fallback: Callable[[str, Exception, str], None] | None = None,
    hooks: Hooks | None = None,
    config: ClientConfig | None = None,
    # Plan #14: batch progress & stagnation
    progress_interval: int = 100,
    on_batch_progress: Callable[[BatchProgressTracker], None] | None = None,
    stagnation_window: int = 5,
    abort_on_stagnation: bool = False,
    item_timeout_s: float | None = None,
    **kwargs: Any,
) -> list[LLMCallResult | Exception]:
    """Implementation for acall_llm_batch extracted out of client facade.

    Args:
        progress_interval: Log batch progress every N items.
        on_batch_progress: Optional callback receiving BatchProgressTracker snapshots.
        stagnation_window: Consecutive identical errors before stagnation alert.
        abort_on_stagnation: If True, cancel remaining items on stagnation.
        item_timeout_s: Per-item wall-clock timeout (includes retries). None = no limit.
    """
    if not messages_list:
        return []

    from llm_client.core.client import acall_llm

    tracker = BatchProgressTracker(
        total=len(messages_list),
        _stagnation_window=stagnation_window,
    )
    _stagnation_abort = asyncio.Event()
    sem = asyncio.Semaphore(max_concurrent)

    def _maybe_log_progress() -> None:
        done = tracker.completed + tracker.errored
        if done > 0 and done % progress_interval == 0:
            logger.info("Batch progress: %s", tracker.summary())
            if on_batch_progress is not None:
                on_batch_progress(tracker)

    async def _call_one(idx: int, messages: list[dict[str, Any]]) -> LLMCallResult:
        if _stagnation_abort.is_set():
            raise RuntimeError(f"Batch aborted due to stagnation (item {idx} skipped)")

        async with sem:
            t0 = time.monotonic()
            try:
                coro = acall_llm(
                    model,
                    messages,
                    timeout=timeout,
                    num_retries=num_retries,
                    reasoning_effort=reasoning_effort,
                    api_base=api_base,
                    base_delay=base_delay,
                    max_delay=max_delay,
                    retry_on=retry_on,
                    on_retry=on_retry,
                    cache=cache,
                    retry=retry,
                    fallback_models=fallback_models,
                    on_fallback=on_fallback,
                    hooks=hooks,
                    config=config,
                    **kwargs,
                )
                if item_timeout_s is not None:
                    result = await asyncio.wait_for(coro, timeout=item_timeout_s)
                else:
                    result = await coro

                tracker.record_completion(time.monotonic() - t0)
                if on_item_complete is not None:
                    on_item_complete(idx, result)
                _maybe_log_progress()
                return result
            except asyncio.TimeoutError:
                exc = TimeoutError(
                    f"Item {idx} timed out after {item_timeout_s}s (item_timeout_s)"
                )
                is_stagnant = tracker.record_error(exc)
                if on_item_error is not None:
                    on_item_error(idx, exc)
                if is_stagnant:
                    event = emit_batch_stagnation_event(tracker)
                    logger.warning(
                        "BATCH_STAGNATION: %d consecutive identical errors. %s",
                        event.stagnation_window,
                        tracker.summary(),
                    )
                    if abort_on_stagnation:
                        _stagnation_abort.set()
                _maybe_log_progress()
                raise exc
            except Exception as e:
                is_stagnant = tracker.record_error(e)
                if on_item_error is not None:
                    on_item_error(idx, e)
                if is_stagnant:
                    event = emit_batch_stagnation_event(tracker)
                    logger.warning(
                        "BATCH_STAGNATION: %d consecutive identical errors. %s",
                        event.stagnation_window,
                        tracker.summary(),
                    )
                    if abort_on_stagnation:
                        _stagnation_abort.set()
                _maybe_log_progress()
                raise

    tasks = [_call_one(i, msgs) for i, msgs in enumerate(messages_list)]
    results = cast(
        "list[LLMCallResult | Exception]",
        await asyncio.gather(*tasks, return_exceptions=return_exceptions),
    )
    logger.info("Batch complete: %s", tracker.summary())
    if on_batch_progress is not None:
        on_batch_progress(tracker)
    return results


def call_llm_batch_impl(
    model: str,
    messages_list: list[list[dict[str, Any]]],
    *,
    max_concurrent: int = 5,
    return_exceptions: bool = False,
    on_item_complete: Callable[[int, LLMCallResult], None] | None = None,
    on_item_error: Callable[[int, Exception], None] | None = None,
    timeout: int = 60,
    num_retries: int = 2,
    reasoning_effort: str | None = None,
    api_base: str | None = None,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_on: list[str] | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    cache: CachePolicy | None = None,
    retry: RetryPolicy | None = None,
    fallback_models: list[str] | None = None,
    on_fallback: Callable[[str, Exception, str], None] | None = None,
    hooks: Hooks | None = None,
    config: ClientConfig | None = None,
    progress_interval: int = 100,
    on_batch_progress: Callable[[BatchProgressTracker], None] | None = None,
    stagnation_window: int = 5,
    abort_on_stagnation: bool = False,
    item_timeout_s: float | None = None,
    **kwargs: Any,
) -> list[LLMCallResult | Exception]:
    """Implementation for call_llm_batch extracted out of client facade."""
    coro = acall_llm_batch_impl(
        model,
        messages_list,
        max_concurrent=max_concurrent,
        return_exceptions=return_exceptions,
        on_item_complete=on_item_complete,
        on_item_error=on_item_error,
        timeout=timeout,
        num_retries=num_retries,
        reasoning_effort=reasoning_effort,
        api_base=api_base,
        base_delay=base_delay,
        max_delay=max_delay,
        retry_on=retry_on,
        on_retry=on_retry,
        cache=cache,
        retry=retry,
        fallback_models=fallback_models,
        on_fallback=on_fallback,
        hooks=hooks,
        config=config,
        progress_interval=progress_interval,
        on_batch_progress=on_batch_progress,
        stagnation_window=stagnation_window,
        abort_on_stagnation=abort_on_stagnation,
        item_timeout_s=item_timeout_s,
        **kwargs,
    )
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


async def acall_llm_structured_batch_impl(
    model: str,
    messages_list: list[list[dict[str, Any]]],
    response_model: type[T],
    *,
    max_concurrent: int = 5,
    return_exceptions: bool = False,
    on_item_complete: Callable[[int, tuple[T, LLMCallResult]], None] | None = None,
    on_item_error: Callable[[int, Exception], None] | None = None,
    timeout: int | None = None,
    num_retries: int = 2,
    reasoning_effort: str | None = None,
    api_base: str | None = None,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_on: list[str] | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    cache: CachePolicy | AsyncCachePolicy | None = None,
    retry: RetryPolicy | None = None,
    fallback_models: list[str] | None = None,
    on_fallback: Callable[[str, Exception, str], None] | None = None,
    hooks: Hooks | None = None,
    config: ClientConfig | None = None,
    **kwargs: Any,
) -> list[tuple[T, LLMCallResult] | Exception]:
    """Implementation for acall_llm_structured_batch extracted from facade."""
    if not messages_list:
        return []

    from llm_client.core.client import acall_llm_structured

    sem = asyncio.Semaphore(max_concurrent)

    async def _call_one(idx: int, messages: list[dict[str, Any]]) -> tuple[T, LLMCallResult]:
        async with sem:
            try:
                result = await acall_llm_structured(
                    model,
                    messages,
                    response_model,
                    timeout=timeout,
                    num_retries=num_retries,
                    reasoning_effort=reasoning_effort,
                    api_base=api_base,
                    base_delay=base_delay,
                    max_delay=max_delay,
                    retry_on=retry_on,
                    on_retry=on_retry,
                    cache=cache,
                    retry=retry,
                    fallback_models=fallback_models,
                    on_fallback=on_fallback,
                    hooks=hooks,
                    config=config,
                    **kwargs,
                )
                if on_item_complete is not None:
                    on_item_complete(idx, result)
                return result
            except Exception as e:
                if on_item_error is not None:
                    on_item_error(idx, e)
                raise

    tasks = [_call_one(i, msgs) for i, msgs in enumerate(messages_list)]
    return cast(
        "list[tuple[T, LLMCallResult] | Exception]",
        await asyncio.gather(*tasks, return_exceptions=return_exceptions),
    )


def call_llm_structured_batch_impl(
    model: str,
    messages_list: list[list[dict[str, Any]]],
    response_model: type[T],
    *,
    max_concurrent: int = 5,
    return_exceptions: bool = False,
    on_item_complete: Callable[[int, tuple[T, LLMCallResult]], None] | None = None,
    on_item_error: Callable[[int, Exception], None] | None = None,
    timeout: int | None = None,
    num_retries: int = 2,
    reasoning_effort: str | None = None,
    api_base: str | None = None,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_on: list[str] | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    cache: CachePolicy | None = None,
    retry: RetryPolicy | None = None,
    fallback_models: list[str] | None = None,
    on_fallback: Callable[[str, Exception, str], None] | None = None,
    hooks: Hooks | None = None,
    config: ClientConfig | None = None,
    **kwargs: Any,
) -> list[tuple[T, LLMCallResult] | Exception]:
    """Implementation for call_llm_structured_batch extracted from facade."""
    coro = acall_llm_structured_batch_impl(
        model,
        messages_list,
        response_model,
        max_concurrent=max_concurrent,
        return_exceptions=return_exceptions,
        on_item_complete=on_item_complete,
        on_item_error=on_item_error,
        timeout=timeout,
        num_retries=num_retries,
        reasoning_effort=reasoning_effort,
        api_base=api_base,
        base_delay=base_delay,
        max_delay=max_delay,
        retry_on=retry_on,
        on_retry=on_retry,
        cache=cache,
        retry=retry,
        fallback_models=fallback_models,
        on_fallback=on_fallback,
        hooks=hooks,
        config=config,
        **kwargs,
    )
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)
