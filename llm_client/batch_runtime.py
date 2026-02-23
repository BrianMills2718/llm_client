"""Batch execution internals extracted from client.py."""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import TYPE_CHECKING, Any, Callable, TypeVar, cast

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

if TYPE_CHECKING:
    from llm_client.config import ClientConfig
    from llm_client.client import (
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
    **kwargs: Any,
) -> list[LLMCallResult | Exception]:
    """Implementation for acall_llm_batch extracted out of client facade."""
    if not messages_list:
        return []

    from llm_client.client import acall_llm

    sem = asyncio.Semaphore(max_concurrent)

    async def _call_one(idx: int, messages: list[dict[str, Any]]) -> LLMCallResult:
        async with sem:
            try:
                result = await acall_llm(
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
                if on_item_complete is not None:
                    on_item_complete(idx, result)
                return result
            except Exception as e:
                if on_item_error is not None:
                    on_item_error(idx, e)
                raise

    tasks = [_call_one(i, msgs) for i, msgs in enumerate(messages_list)]
    return cast(
        "list[LLMCallResult | Exception]",
        await asyncio.gather(*tasks, return_exceptions=return_exceptions),
    )


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
    **kwargs: Any,
) -> list[tuple[T, LLMCallResult] | Exception]:
    """Implementation for acall_llm_structured_batch extracted from facade."""
    if not messages_list:
        return []

    from llm_client.client import acall_llm_structured

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
