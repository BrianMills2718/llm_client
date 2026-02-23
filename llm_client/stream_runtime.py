"""Stream execution internals extracted from client.py."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, cast

import llm_client.client as _client_mod

_client: Any = _client_mod

if TYPE_CHECKING:
    from llm_client.client import AsyncLLMStream, Hooks, LLMStream, RetryPolicy
    from llm_client.config import ClientConfig


def stream_llm_impl(
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout: int = 60,
    num_retries: int = 2,
    reasoning_effort: str | None = None,
    api_base: str | None = None,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_on: list[str] | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    retry: RetryPolicy | None = None,
    fallback_models: list[str] | None = None,
    on_fallback: Callable[[str, Exception, str], None] | None = None,
    hooks: Hooks | None = None,
    config: ClientConfig | None = None,
    **kwargs: Any,
) -> LLMStream:
    """Implementation for stream_llm extracted out of client facade."""
    _client._check_model_deprecation(model)
    cfg = config or _client.ClientConfig.from_env()
    task = kwargs.pop("task", None)
    trace_id = kwargs.pop("trace_id", None)
    max_budget: float | None = kwargs.pop("max_budget", None)
    task, trace_id, max_budget, _entry_warnings = _client._require_tags(
        task, trace_id, max_budget, caller="stream_llm",
    )
    _client._check_budget(trace_id, max_budget)
    if _client._is_agent_model(model):
        from llm_client.agents import _route_stream

        return cast(
            "LLMStream",
            _route_stream(model, messages, hooks=hooks, timeout=timeout, **kwargs),
        )
    r = _client._effective_retry(retry, num_retries, base_delay, max_delay, retry_on, on_retry)
    plan = _client._resolve_call_plan(
        model=model,
        fallback_models=fallback_models,
        api_base=api_base,
        config=cfg,
    )
    models = plan.models
    routing_policy = str(plan.routing_trace.get("routing_policy", _client._routing_policy_label(cfg)))
    _warnings: list[str] = list(_entry_warnings)
    backoff_fn = r.backoff or _client.exponential_backoff

    def _execute_stream_model(model_idx: int, current_model: str) -> LLMStream:
        current_api_base = _client._resolve_api_base_for_model(current_model, api_base, cfg)
        call_kwargs = _client._prepare_call_kwargs(
            current_model, messages,
            timeout=timeout,
            num_retries=0,
            reasoning_effort=reasoning_effort,
            api_base=current_api_base,
            kwargs=kwargs,
            warning_sink=_warnings,
        )
        call_kwargs["stream"] = True

        if hooks and hooks.before_call:
            hooks.before_call(current_model, messages, kwargs)

        def _invoke_attempt(attempt: int) -> LLMStream:
            with _client._rate_limit.acquire(current_model):
                response = _client.litellm.completion(**call_kwargs)
            if attempt > 0:
                _client.logger.info("stream_llm succeeded after %d retries", attempt)
            return cast(
                "LLMStream",
                _client.LLMStream(
                response,
                current_model,
                hooks=hooks,
                messages=messages,
                task=task,
                trace_id=trace_id,
                warnings=_warnings,
                requested_model=model,
                resolved_model=current_model,
                routing_trace=_client._build_routing_trace(
                    requested_model=model,
                    attempted_models=models[:model_idx + 1],
                    selected_model=current_model,
                    requested_api_base=api_base,
                    effective_api_base=current_api_base,
                    routing_policy=routing_policy,
                ),
                ),
            )

        return cast(
            "LLMStream",
            _client.run_sync_with_retry(
            caller="stream_llm",
            model=current_model,
            max_retries=r.max_retries,
            invoke=_invoke_attempt,
            should_retry=lambda exc: _client._check_retryable(exc, r),
            compute_delay=lambda attempt, exc: _client._compute_retry_delay(
                attempt=attempt,
                error=exc,
                policy=r,
                backoff_fn=backoff_fn,
            ),
            warning_sink=_warnings,
            logger=_client.logger,
            on_error=(hooks.on_error if hooks and hooks.on_error else None),
            on_retry=r.on_retry,
            maybe_retry_hook=lambda exc, attempt, max_retries: _client._maybe_retry_with_openrouter_key_rotation(
                error=exc,
                attempt=attempt,
                max_retries=max_retries,
                current_model=current_model,
                current_api_base=current_api_base,
                user_kwargs=kwargs,
                warning_sink=_warnings,
                on_retry=r.on_retry,
                caller="stream_llm",
            ),
            ),
        )

    try:
        return cast(
            "LLMStream",
            _client.run_sync_with_fallback(
            models=models,
            execute_model=_execute_stream_model,
            on_fallback=on_fallback,
            warning_sink=_warnings,
            logger=_client.logger,
            ),
        )
    except Exception as e:
        raise _client.wrap_error(e) from e


async def astream_llm_impl(
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout: int = 60,
    num_retries: int = 2,
    reasoning_effort: str | None = None,
    api_base: str | None = None,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_on: list[str] | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    retry: RetryPolicy | None = None,
    fallback_models: list[str] | None = None,
    on_fallback: Callable[[str, Exception, str], None] | None = None,
    hooks: Hooks | None = None,
    config: ClientConfig | None = None,
    **kwargs: Any,
) -> AsyncLLMStream:
    """Implementation for astream_llm extracted out of client facade."""
    _client._check_model_deprecation(model)
    cfg = config or _client.ClientConfig.from_env()
    task = kwargs.pop("task", None)
    trace_id = kwargs.pop("trace_id", None)
    max_budget: float | None = kwargs.pop("max_budget", None)
    task, trace_id, max_budget, _entry_warnings = _client._require_tags(
        task, trace_id, max_budget, caller="astream_llm",
    )
    _client._check_budget(trace_id, max_budget)
    if _client._is_agent_model(model):
        from llm_client.agents import _route_astream

        return cast(
            "AsyncLLMStream",
            await _route_astream(model, messages, hooks=hooks, timeout=timeout, **kwargs),
        )
    r = _client._effective_retry(retry, num_retries, base_delay, max_delay, retry_on, on_retry)
    plan = _client._resolve_call_plan(
        model=model,
        fallback_models=fallback_models,
        api_base=api_base,
        config=cfg,
    )
    models = plan.models
    routing_policy = str(plan.routing_trace.get("routing_policy", _client._routing_policy_label(cfg)))
    _warnings: list[str] = list(_entry_warnings)
    backoff_fn = r.backoff or _client.exponential_backoff

    async def _execute_stream_model(model_idx: int, current_model: str) -> AsyncLLMStream:
        current_api_base = _client._resolve_api_base_for_model(current_model, api_base, cfg)
        call_kwargs = _client._prepare_call_kwargs(
            current_model, messages,
            timeout=timeout,
            num_retries=0,
            reasoning_effort=reasoning_effort,
            api_base=current_api_base,
            kwargs=kwargs,
            warning_sink=_warnings,
        )
        call_kwargs["stream"] = True

        if hooks and hooks.before_call:
            hooks.before_call(current_model, messages, kwargs)

        async def _invoke_attempt(attempt: int) -> AsyncLLMStream:
            async with _client._rate_limit.aacquire(current_model):
                response = await _client.litellm.acompletion(**call_kwargs)
            if attempt > 0:
                _client.logger.info("astream_llm succeeded after %d retries", attempt)
            return cast(
                "AsyncLLMStream",
                _client.AsyncLLMStream(
                response,
                current_model,
                hooks=hooks,
                messages=messages,
                task=task,
                trace_id=trace_id,
                warnings=_warnings,
                requested_model=model,
                resolved_model=current_model,
                routing_trace=_client._build_routing_trace(
                    requested_model=model,
                    attempted_models=models[:model_idx + 1],
                    selected_model=current_model,
                    requested_api_base=api_base,
                    effective_api_base=current_api_base,
                    routing_policy=routing_policy,
                ),
                ),
            )

        return cast(
            "AsyncLLMStream",
            await _client.run_async_with_retry(
            caller="astream_llm",
            model=current_model,
            max_retries=r.max_retries,
            invoke=_invoke_attempt,
            should_retry=lambda exc: _client._check_retryable(exc, r),
            compute_delay=lambda attempt, exc: _client._compute_retry_delay(
                attempt=attempt,
                error=exc,
                policy=r,
                backoff_fn=backoff_fn,
            ),
            warning_sink=_warnings,
            logger=_client.logger,
            on_error=(hooks.on_error if hooks and hooks.on_error else None),
            on_retry=r.on_retry,
            maybe_retry_hook=lambda exc, attempt, max_retries: _client._maybe_retry_with_openrouter_key_rotation(
                error=exc,
                attempt=attempt,
                max_retries=max_retries,
                current_model=current_model,
                current_api_base=current_api_base,
                user_kwargs=kwargs,
                warning_sink=_warnings,
                on_retry=r.on_retry,
                caller="astream_llm",
            ),
            ),
        )

    try:
        return cast(
            "AsyncLLMStream",
            await _client.run_async_with_fallback(
            models=models,
            execute_model=_execute_stream_model,
            on_fallback=on_fallback,
            warning_sink=_warnings,
            logger=_client.logger,
            ),
        )
    except Exception as e:
        raise _client.wrap_error(e) from e
