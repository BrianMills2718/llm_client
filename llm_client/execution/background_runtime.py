"""Background-polling helpers for long-thinking Responses API calls.

This module owns the long-thinking/background-mode runtime mechanics that were
previously embedded in ``llm_client.client``: mode gating, polling config,
sync/async poll loops, and direct response retrieval helpers. The public client
module keeps the monkeypatch-sensitive wrapper names that existing tests patch,
while the concrete implementation lives here.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import urllib.parse
from typing import Any, Awaitable, Callable

from llm_client.core.errors import LLMConfigurationError
from llm_client.core.model_detection import _base_model_name
from llm_client.openrouter import (
    OPENROUTER_API_KEY_ENV,
    _normalize_api_key_value,
    _openrouter_key_candidates_from_env,
)

logger = logging.getLogger(__name__)

_LONG_THINKING_MODELS = {"gpt-5.2-pro"}
_LONG_THINKING_REASONING_EFFORTS = {"high", "xhigh"}
_BACKGROUND_POLL_INTERVAL = 15
_BACKGROUND_DEFAULT_TIMEOUT = 900

_BACKGROUND_ERR_ENDPOINT_UNSUPPORTED = "LLMC_ERR_BACKGROUND_ENDPOINT_UNSUPPORTED"
_BACKGROUND_ERR_MISSING_OPENAI_KEY = "LLMC_ERR_BACKGROUND_OPENAI_KEY_REQUIRED"
_BACKGROUND_ERR_MISSING_OPENROUTER_KEY = "LLMC_ERR_BACKGROUND_OPENROUTER_KEY_REQUIRED"


def _validate_background_retrieval_api_base(api_base: str | None) -> str:
    """Return endpoint kind for background retrieval ("openai" or "openrouter")."""

    if api_base is None:
        return "openai"
    base = str(api_base).strip()
    if not base:
        return "openai"

    parsed = urllib.parse.urlparse(base)
    hostname = (parsed.hostname or "").strip().lower()
    if hostname == "api.openai.com" or hostname.endswith(".api.openai.com"):
        return "openai"
    if "openai.com" in hostname and "openrouter" not in hostname:
        return "openai"
    if hostname == "openrouter.ai" or hostname.endswith(".openrouter.ai"):
        return "openrouter"

    raise LLMConfigurationError(
        "Background response retrieval for long-thinking models currently supports "
        f"OpenAI/OpenRouter endpoints only. Received api_base={base!r}. "
        "Use https://api.openai.com/v1, https://openrouter.ai/api/v1, or default.",
        error_code=_BACKGROUND_ERR_ENDPOINT_UNSUPPORTED,
        details={"api_base": base},
    )


def _needs_background_mode(model: str, reasoning_effort: str | None) -> bool:
    """Check if a model+reasoning_effort combination needs background polling."""

    base = _base_model_name(model)
    return (
        base in _LONG_THINKING_MODELS
        and reasoning_effort is not None
        and reasoning_effort.lower() in _LONG_THINKING_REASONING_EFFORTS
    )


def _coerce_positive_int(value: Any, default: int) -> int:
    """Best-effort int coercion with positive guard and fallback."""

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _background_mode_for_model(
    *,
    model: str,
    use_responses: bool,
    reasoning_effort: str | None,
) -> bool | None:
    """Return background mode flag for routing trace / execution policy."""

    if not use_responses:
        return None
    return _needs_background_mode(model, reasoning_effort)


def _background_polling_config(model_kwargs: dict[str, Any]) -> tuple[int, int]:
    """Return validated (timeout, poll_interval) for background polling."""

    timeout = _coerce_positive_int(
        model_kwargs.get("background_timeout"),
        _BACKGROUND_DEFAULT_TIMEOUT,
    )
    poll_interval = _coerce_positive_int(
        model_kwargs.get("background_poll_interval"),
        _BACKGROUND_POLL_INTERVAL,
    )
    return timeout, poll_interval


def _maybe_poll_background_response_impl(
    response: Any,
    *,
    api_base: str | None,
    request_timeout: int | None,
    model_kwargs: dict[str, Any],
    poll_response: Callable[..., Any],
) -> Any:
    """Poll a non-terminal background response to completion when possible."""

    bg_status = getattr(response, "status", None)
    if not bg_status or bg_status == "completed":
        return response

    response_id = getattr(response, "id", None)
    if not response_id:
        return response

    bg_timeout, bg_poll_interval = _background_polling_config(model_kwargs)
    return poll_response(
        response_id,
        api_base=api_base,
        request_timeout=request_timeout,
        timeout=bg_timeout,
        poll_interval=bg_poll_interval,
    )


async def _maybe_apoll_background_response_impl(
    response: Any,
    *,
    api_base: str | None,
    request_timeout: int | None,
    model_kwargs: dict[str, Any],
    poll_response: Callable[..., Awaitable[Any]],
) -> Any:
    """Async variant of background polling helper."""

    bg_status = getattr(response, "status", None)
    if not bg_status or bg_status == "completed":
        return response

    response_id = getattr(response, "id", None)
    if not response_id:
        return response

    bg_timeout, bg_poll_interval = _background_polling_config(model_kwargs)
    return await poll_response(
        response_id,
        api_base=api_base,
        request_timeout=request_timeout,
        timeout=bg_timeout,
        poll_interval=bg_poll_interval,
    )


def _poll_background_response_impl(
    response_id: str,
    *,
    api_base: str | None = None,
    request_timeout: int | None = None,
    poll_interval: int = _BACKGROUND_POLL_INTERVAL,
    timeout: int = _BACKGROUND_DEFAULT_TIMEOUT,
    retrieve_response: Callable[..., Any],
) -> Any:
    """Poll a background Responses API response until completed."""

    deadline = time.monotonic() + timeout
    attempt = 0
    while time.monotonic() < deadline:
        try:
            response = retrieve_response(
                response_id=response_id,
                api_base=api_base,
                request_timeout=request_timeout,
            )
        except LLMConfigurationError:
            raise
        except Exception as error:
            logger.warning("Background poll attempt %d failed: %s", attempt, error)
            time.sleep(poll_interval)
            attempt += 1
            continue

        status = getattr(response, "status", None)
        if status == "completed":
            logger.info(
                "Background response %s completed after %d polls",
                response_id,
                attempt + 1,
            )
            return response
        if status in ("failed", "cancelled"):
            error = getattr(response, "error", None)
            raise RuntimeError(f"Background response {response_id} {status}: {error}")

        logger.debug(
            "Background response %s status=%s, poll %d",
            response_id,
            status,
            attempt + 1,
        )
        time.sleep(poll_interval)
        attempt += 1

    raise TimeoutError(
        f"Background response {response_id} did not complete within {timeout}s"
    )


async def _apoll_background_response_impl(
    response_id: str,
    *,
    api_base: str | None = None,
    request_timeout: int | None = None,
    poll_interval: int = _BACKGROUND_POLL_INTERVAL,
    timeout: int = _BACKGROUND_DEFAULT_TIMEOUT,
    retrieve_response: Callable[..., Awaitable[Any]],
) -> Any:
    """Poll a background Responses API response until completed asynchronously."""

    deadline = time.monotonic() + timeout
    attempt = 0
    while time.monotonic() < deadline:
        try:
            response = await retrieve_response(
                response_id=response_id,
                api_base=api_base,
                request_timeout=request_timeout,
            )
        except LLMConfigurationError:
            raise
        except Exception as error:
            logger.warning("Background poll attempt %d failed: %s", attempt, error)
            await asyncio.sleep(poll_interval)
            attempt += 1
            continue

        status = getattr(response, "status", None)
        if status == "completed":
            logger.info(
                "Background response %s completed after %d polls",
                response_id,
                attempt + 1,
            )
            return response
        if status in ("failed", "cancelled"):
            error = getattr(response, "error", None)
            raise RuntimeError(f"Background response {response_id} {status}: {error}")

        logger.debug(
            "Background response %s status=%s, poll %d",
            response_id,
            status,
            attempt + 1,
        )
        await asyncio.sleep(poll_interval)
        attempt += 1

    raise TimeoutError(
        f"Background response {response_id} did not complete within {timeout}s"
    )


def _retrieve_background_response_impl(
    *,
    response_id: str,
    api_base: str | None,
    request_timeout: int | None,
) -> Any:
    """Retrieve a background response by ID via the OpenAI SDK client."""

    from openai import OpenAI

    endpoint_kind = _validate_background_retrieval_api_base(api_base)
    if endpoint_kind == "openrouter":
        api_key = _normalize_api_key_value(os.environ.get(OPENROUTER_API_KEY_ENV))
        if not api_key:
            ring = _openrouter_key_candidates_from_env()
            api_key = ring[0] if ring else ""
        if not api_key:
            raise LLMConfigurationError(
                "OPENROUTER_API_KEY is required to retrieve background responses "
                "for long-thinking models via OpenRouter",
                error_code=_BACKGROUND_ERR_MISSING_OPENROUTER_KEY,
                details={"api_base": api_base},
            )
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise LLMConfigurationError(
                "OPENAI_API_KEY is required to retrieve background responses for long-thinking models",
                error_code=_BACKGROUND_ERR_MISSING_OPENAI_KEY,
                details={"api_base": api_base},
            )

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if api_base:
        client_kwargs["base_url"] = api_base
    if request_timeout is not None:
        client_kwargs["timeout"] = request_timeout
    client = OpenAI(**client_kwargs)
    return client.responses.retrieve(response_id)


async def _aretrieve_background_response_impl(
    *,
    response_id: str,
    api_base: str | None,
    request_timeout: int | None,
) -> Any:
    """Retrieve a background response by ID via the async OpenAI SDK client."""

    from openai import AsyncOpenAI

    endpoint_kind = _validate_background_retrieval_api_base(api_base)
    if endpoint_kind == "openrouter":
        api_key = _normalize_api_key_value(os.environ.get(OPENROUTER_API_KEY_ENV))
        if not api_key:
            ring = _openrouter_key_candidates_from_env()
            api_key = ring[0] if ring else ""
        if not api_key:
            raise LLMConfigurationError(
                "OPENROUTER_API_KEY is required to retrieve background responses "
                "for long-thinking models via OpenRouter",
                error_code=_BACKGROUND_ERR_MISSING_OPENROUTER_KEY,
                details={"api_base": api_base},
            )
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise LLMConfigurationError(
                "OPENAI_API_KEY is required to retrieve background responses for long-thinking models",
                error_code=_BACKGROUND_ERR_MISSING_OPENAI_KEY,
                details={"api_base": api_base},
            )

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if api_base:
        client_kwargs["base_url"] = api_base
    if request_timeout is not None:
        client_kwargs["timeout"] = request_timeout
    client = AsyncOpenAI(**client_kwargs)
    return await client.responses.retrieve(response_id)
