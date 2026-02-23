"""Embedding execution internals extracted from client.py."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

import llm_client.client as _client_mod

_client: Any = _client_mod

if TYPE_CHECKING:
    from llm_client.client import EmbeddingResult


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

    call_kwargs: dict[str, Any] = {"model": model, "input": texts, "timeout": timeout}
    if dimensions is not None:
        call_kwargs["dimensions"] = dimensions
    if api_base is not None:
        call_kwargs["api_base"] = api_base
    if api_key is not None:
        call_kwargs["api_key"] = api_key
    call_kwargs.update(kwargs)

    _client._check_model_deprecation(model)
    try:
        with _client._rate_limit.acquire(model):
            response = _client.litellm.embedding(**call_kwargs)

        embeddings = [item["embedding"] for item in response.data]
        usage = dict(response.usage) if hasattr(response, "usage") and response.usage else {}
        try:
            cost = _client.litellm.completion_cost(completion_response=response)
        except Exception:
            cost = 0.0

        result = _client.EmbeddingResult(embeddings=embeddings, usage=usage, cost=cost, model=model)
        _client._io_log.log_embedding(
            model=model,
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
            model=model,
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

    call_kwargs: dict[str, Any] = {"model": model, "input": texts, "timeout": timeout}
    if dimensions is not None:
        call_kwargs["dimensions"] = dimensions
    if api_base is not None:
        call_kwargs["api_base"] = api_base
    if api_key is not None:
        call_kwargs["api_key"] = api_key
    call_kwargs.update(kwargs)

    _client._check_model_deprecation(model)
    try:
        async with _client._rate_limit.aacquire(model):
            response = await _client.litellm.aembedding(**call_kwargs)

        embeddings = [item["embedding"] for item in response.data]
        usage = dict(response.usage) if hasattr(response, "usage") and response.usage else {}
        try:
            cost = _client.litellm.completion_cost(completion_response=response)
        except Exception:
            cost = 0.0

        result = _client.EmbeddingResult(embeddings=embeddings, usage=usage, cost=cost, model=model)
        _client._io_log.log_embedding(
            model=model,
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
            model=model,
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
