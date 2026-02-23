"""Embedding execution internals extracted from client.py."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

import llm_client.client as _client_mod

_client: Any = _client_mod

if TYPE_CHECKING:
    from llm_client.client import EmbeddingResult


def _resolve_embedding_route(
    *,
    model: str,
    api_base: str | None,
) -> tuple[str, str | None]:
    """Resolve embedding model/api_base via llm_client routing policy."""
    try:
        cfg = _client.ClientConfig.from_env()
        plan = _client._resolve_call_plan(
            model=model,
            fallback_models=None,
            api_base=api_base,
            config=cfg,
        )
        resolved_model = str(plan.primary_model or model)
        resolved_api_base = _client._resolve_api_base_for_model(
            resolved_model,
            api_base,
            cfg,
        )
        return resolved_model, resolved_api_base
    except Exception:
        return model, api_base


def _resolve_embedding_retry_policy(kwargs: dict[str, Any]) -> Any:
    """Resolve embedding retry policy from kwargs or defaults."""
    retry = kwargs.pop("retry", None)
    num_retries = int(kwargs.pop("num_retries", 2))
    base_delay = float(kwargs.pop("base_delay", 1.0))
    max_delay = float(kwargs.pop("max_delay", 30.0))
    retry_on = kwargs.pop("retry_on", None)
    on_retry = kwargs.pop("on_retry", None)
    return _client._effective_retry(
        retry,
        num_retries,
        base_delay,
        max_delay,
        retry_on,
        on_retry,
    )


def embed_impl(
    model: str,
    input: str | list[str],
    *,
    dimensions: int | None = None,
    timeout: int = 60,
    api_base: str | None = None,
    api_key: str | None = None,
    task: str | None = None,
    trace_id: str | None = None,
    **kwargs: Any,
) -> EmbeddingResult:
    """Implementation for embed extracted out of client facade."""
    texts = [input] if isinstance(input, str) else list(input)
    _log_t0 = time.monotonic()
    resolved_model, resolved_api_base = _resolve_embedding_route(model=model, api_base=api_base)
    retry_policy = _resolve_embedding_retry_policy(kwargs)
    retry_warnings: list[str] = []

    call_kwargs: dict[str, Any] = {"model": resolved_model, "input": texts, "timeout": timeout}
    if dimensions is not None:
        call_kwargs["dimensions"] = dimensions
    if resolved_api_base is not None:
        call_kwargs["api_base"] = resolved_api_base
    if api_key is not None:
        call_kwargs["api_key"] = api_key
    call_kwargs.update(kwargs)

    _client._check_model_deprecation(resolved_model)
    try:
        backoff_fn = retry_policy.backoff or _client.exponential_backoff

        def _invoke_attempt(_: int) -> Any:
            with _client._rate_limit.acquire(resolved_model):
                return _client.litellm.embedding(**call_kwargs)

        response = _client.run_sync_with_retry(
            caller="embed",
            model=resolved_model,
            max_retries=retry_policy.max_retries,
            invoke=_invoke_attempt,
            should_retry=lambda exc: _client._check_retryable(exc, retry_policy),
            compute_delay=lambda attempt, exc: _client._compute_retry_delay(
                attempt=attempt,
                error=exc,
                policy=retry_policy,
                backoff_fn=backoff_fn,
            ),
            warning_sink=retry_warnings,
            logger=_client.logger,
            on_retry=retry_policy.on_retry,
            maybe_retry_hook=lambda exc, attempt, max_retries: _client._maybe_retry_with_openrouter_key_rotation(
                error=exc,
                attempt=attempt,
                max_retries=max_retries,
                current_model=resolved_model,
                current_api_base=resolved_api_base,
                user_kwargs=call_kwargs,
                warning_sink=retry_warnings,
                on_retry=retry_policy.on_retry,
                caller="embed",
            ),
        )
        embeddings = [item["embedding"] for item in response.data]
        usage = dict(response.usage) if hasattr(response, "usage") and response.usage else {}
        try:
            cost = _client.litellm.completion_cost(completion_response=response)
        except Exception:
            cost = 0.0

        result = _client.EmbeddingResult(
            embeddings=embeddings,
            usage=usage,
            cost=cost,
            model=resolved_model,
        )
        _client._io_log.log_embedding(
            model=resolved_model,
            input_count=len(texts),
            input_chars=sum(len(t) for t in texts),
            dimensions=len(result.embeddings[0]) if result.embeddings else None,
            usage=result.usage,
            cost=result.cost,
            latency_s=time.monotonic() - _log_t0,
            caller="embed",
            task=task,
            trace_id=trace_id,
        )
        return cast("EmbeddingResult", result)
    except Exception as e:
        _client._io_log.log_embedding(
            model=resolved_model,
            input_count=len(texts),
            input_chars=sum(len(t) for t in texts),
            dimensions=None,
            usage=None,
            cost=None,
            latency_s=time.monotonic() - _log_t0,
            error=e,
            caller="embed",
            task=task,
            trace_id=trace_id,
        )
        raise


async def aembed_impl(
    model: str,
    input: str | list[str],
    *,
    dimensions: int | None = None,
    timeout: int = 60,
    api_base: str | None = None,
    api_key: str | None = None,
    task: str | None = None,
    trace_id: str | None = None,
    **kwargs: Any,
) -> EmbeddingResult:
    """Implementation for aembed extracted out of client facade."""
    texts = [input] if isinstance(input, str) else list(input)
    _log_t0 = time.monotonic()
    resolved_model, resolved_api_base = _resolve_embedding_route(model=model, api_base=api_base)
    retry_policy = _resolve_embedding_retry_policy(kwargs)
    retry_warnings: list[str] = []

    call_kwargs: dict[str, Any] = {"model": resolved_model, "input": texts, "timeout": timeout}
    if dimensions is not None:
        call_kwargs["dimensions"] = dimensions
    if resolved_api_base is not None:
        call_kwargs["api_base"] = resolved_api_base
    if api_key is not None:
        call_kwargs["api_key"] = api_key
    call_kwargs.update(kwargs)

    _client._check_model_deprecation(resolved_model)
    try:
        backoff_fn = retry_policy.backoff or _client.exponential_backoff

        async def _invoke_attempt(_: int) -> Any:
            async with _client._rate_limit.aacquire(resolved_model):
                return await _client.litellm.aembedding(**call_kwargs)

        response = await _client.run_async_with_retry(
            caller="aembed",
            model=resolved_model,
            max_retries=retry_policy.max_retries,
            invoke=_invoke_attempt,
            should_retry=lambda exc: _client._check_retryable(exc, retry_policy),
            compute_delay=lambda attempt, exc: _client._compute_retry_delay(
                attempt=attempt,
                error=exc,
                policy=retry_policy,
                backoff_fn=backoff_fn,
            ),
            warning_sink=retry_warnings,
            logger=_client.logger,
            on_retry=retry_policy.on_retry,
            maybe_retry_hook=lambda exc, attempt, max_retries: _client._maybe_retry_with_openrouter_key_rotation(
                error=exc,
                attempt=attempt,
                max_retries=max_retries,
                current_model=resolved_model,
                current_api_base=resolved_api_base,
                user_kwargs=call_kwargs,
                warning_sink=retry_warnings,
                on_retry=retry_policy.on_retry,
                caller="aembed",
            ),
        )
        embeddings = [item["embedding"] for item in response.data]
        usage = dict(response.usage) if hasattr(response, "usage") and response.usage else {}
        try:
            cost = _client.litellm.completion_cost(completion_response=response)
        except Exception:
            cost = 0.0

        result = _client.EmbeddingResult(
            embeddings=embeddings,
            usage=usage,
            cost=cost,
            model=resolved_model,
        )
        _client._io_log.log_embedding(
            model=resolved_model,
            input_count=len(texts),
            input_chars=sum(len(t) for t in texts),
            dimensions=len(result.embeddings[0]) if result.embeddings else None,
            usage=result.usage,
            cost=result.cost,
            latency_s=time.monotonic() - _log_t0,
            caller="aembed",
            task=task,
            trace_id=trace_id,
        )
        return cast("EmbeddingResult", result)
    except Exception as e:
        _client._io_log.log_embedding(
            model=resolved_model,
            input_count=len(texts),
            input_chars=sum(len(t) for t in texts),
            dimensions=None,
            usage=None,
            cost=None,
            latency_s=time.monotonic() - _log_t0,
            error=e,
            caller="aembed",
            task=task,
            trace_id=trace_id,
        )
        raise
