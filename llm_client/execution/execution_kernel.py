"""Shared retry/fallback execution primitives for llm_client call paths."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")


def _error_text(exc: Exception) -> str:
    """Return non-empty text for logging warnings/diagnostics."""
    text = str(exc).strip()
    if text:
        return text
    return type(exc).__name__


def _maybe_register_provider_cooldown(
    *,
    model: str,
    exc: Exception,
    warning_sink: list[str],
    logger: logging.Logger,
) -> None:
    """Publish shared cooldown state for 429-like provider failures."""
    try:
        from llm_client.execution.retry import _is_rate_limit_error, _retry_delay_hint
        from llm_client.utils import rate_limit as _rate_limit
    except Exception:
        return

    if not _is_rate_limit_error(exc):
        return

    hint_delay, hint_source = _retry_delay_hint(exc)
    source = hint_source if hint_source != "none" else "provider-floor"
    applied_delay = _rate_limit.register_rate_limit_cooldown(
        model,
        hint_delay,
        source=source,
    )
    if applied_delay <= 0:
        return

    provider = _rate_limit._get_provider(model)
    warning_sink.append(
        f"PROVIDER_COOLDOWN: {provider} cooling down for {applied_delay:.1f}s "
        f"[source={source}]"
    )
    logger.warning(
        "Registered shared provider cooldown for %s after %s: %.1fs (source=%s)",
        provider,
        type(exc).__name__,
        applied_delay,
        source,
    )


def run_sync_with_retry(
    *,
    caller: str,
    model: str,
    max_retries: int,
    invoke: Callable[[int], T],
    should_retry: Callable[[Exception], bool],
    compute_delay: Callable[[int, Exception], tuple[float, str]],
    warning_sink: list[str],
    logger: logging.Logger,
    on_error: Callable[[Exception, int], None] | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    maybe_retry_hook: Callable[[Exception, int, int], bool] | None = None,
) -> T:
    """Execute sync attempts with shared retry behavior."""
    for attempt in range(max_retries + 1):
        try:
            return invoke(attempt)
        except Exception as exc:
            if on_error is not None:
                on_error(exc, attempt)
            _maybe_register_provider_cooldown(
                model=model,
                exc=exc,
                warning_sink=warning_sink,
                logger=logger,
            )
            if maybe_retry_hook is not None and maybe_retry_hook(exc, attempt, max_retries):
                continue
            if not should_retry(exc) or attempt >= max_retries:
                raise

            delay, retry_delay_source = compute_delay(attempt, exc)
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            warning_sink.append(
                f"RETRY {attempt + 1}/{max_retries + 1}: "
                f"{model} ({type(exc).__name__}: {_error_text(exc)}) "
                f"[retry_delay_source={retry_delay_source}]"
            )
            logger.warning(
                "%s attempt %d/%d failed (retrying in %.1fs, source=%s): %s",
                caller,
                attempt + 1,
                max_retries + 1,
                delay,
                retry_delay_source,
                _error_text(exc),
            )
            time.sleep(delay)

    raise RuntimeError("run_sync_with_retry exhausted without returning")


async def run_async_with_retry(
    *,
    caller: str,
    model: str,
    max_retries: int,
    invoke: Callable[[int], Awaitable[T]],
    should_retry: Callable[[Exception], bool],
    compute_delay: Callable[[int, Exception], tuple[float, str]],
    warning_sink: list[str],
    logger: logging.Logger,
    on_error: Callable[[Exception, int], None] | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    maybe_retry_hook: Callable[[Exception, int, int], bool] | None = None,
) -> T:
    """Execute async attempts with shared retry behavior."""
    for attempt in range(max_retries + 1):
        try:
            return await invoke(attempt)
        except Exception as exc:
            if on_error is not None:
                on_error(exc, attempt)
            _maybe_register_provider_cooldown(
                model=model,
                exc=exc,
                warning_sink=warning_sink,
                logger=logger,
            )
            if maybe_retry_hook is not None and maybe_retry_hook(exc, attempt, max_retries):
                continue
            if not should_retry(exc) or attempt >= max_retries:
                raise

            delay, retry_delay_source = compute_delay(attempt, exc)
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            warning_sink.append(
                f"RETRY {attempt + 1}/{max_retries + 1}: "
                f"{model} ({type(exc).__name__}: {_error_text(exc)}) "
                f"[retry_delay_source={retry_delay_source}]"
            )
            logger.warning(
                "%s attempt %d/%d failed (retrying in %.1fs, source=%s): %s",
                caller,
                attempt + 1,
                max_retries + 1,
                delay,
                retry_delay_source,
                _error_text(exc),
            )
            await asyncio.sleep(delay)

    raise RuntimeError("run_async_with_retry exhausted without returning")


def run_sync_with_fallback(
    *,
    models: list[str],
    execute_model: Callable[[int, str], T],
    on_fallback: Callable[[str, Exception, str], Any] | None = None,
    warning_sink: list[str] | None = None,
    logger: logging.Logger | None = None,
) -> T:
    """Execute sync model chain with fallback behavior."""
    last_error: Exception | None = None
    for model_idx, current_model in enumerate(models):
        try:
            return execute_model(model_idx, current_model)
        except Exception as exc:
            last_error = exc
            if model_idx < len(models) - 1:
                next_model = models[model_idx + 1]
                if on_fallback is not None:
                    on_fallback(current_model, exc, next_model)
                if warning_sink is not None:
                    warning_sink.append(
                        f"FALLBACK: {current_model} -> {next_model} "
                        f"({type(exc).__name__}: {_error_text(exc)})"
                    )
                if logger is not None:
                    logger.warning(
                        "Falling back from %s to %s: %s",
                        current_model,
                        next_model,
                        _error_text(exc),
                    )
                continue
            raise
    if last_error is not None:
        raise last_error
    raise RuntimeError("run_sync_with_fallback received empty model chain")


async def run_async_with_fallback(
    *,
    models: list[str],
    execute_model: Callable[[int, str], Awaitable[T]],
    on_fallback: Callable[[str, Exception, str], Any] | None = None,
    warning_sink: list[str] | None = None,
    logger: logging.Logger | None = None,
) -> T:
    """Execute async model chain with fallback behavior."""
    last_error: Exception | None = None
    for model_idx, current_model in enumerate(models):
        try:
            return await execute_model(model_idx, current_model)
        except Exception as exc:
            last_error = exc
            if model_idx < len(models) - 1:
                next_model = models[model_idx + 1]
                if on_fallback is not None:
                    on_fallback(current_model, exc, next_model)
                if warning_sink is not None:
                    warning_sink.append(
                        f"FALLBACK: {current_model} -> {next_model} "
                        f"({type(exc).__name__}: {_error_text(exc)})"
                    )
                if logger is not None:
                    logger.warning(
                        "Falling back from %s to %s: %s",
                        current_model,
                        next_model,
                        _error_text(exc),
                    )
                continue
            raise
    if last_error is not None:
        raise last_error
    raise RuntimeError("run_async_with_fallback received empty model chain")
