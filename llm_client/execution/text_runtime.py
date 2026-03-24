"""Internal runtimes for text-completion entrypoints.

This module owns the implementation behind the public ``call_llm`` and
``acall_llm`` facades. The public API remains in ``client.py``; this module
holds the text-call control flow so the runtime can be split by workload family
without changing caller-facing signatures.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable, cast

from llm_client.core.client import AsyncCachePolicy, CachePolicy, ExecutionMode, Hooks, LLMCallResult, RetryPolicy
from llm_client.core.config import ClientConfig
from llm_client.langfuse_callbacks import inject_metadata as _inject_langfuse_metadata

_client: Any = import_module("llm_client.client")


def _call_llm_impl(
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
    cache: CachePolicy | None = None,
    retry: RetryPolicy | None = None,
    fallback_models: list[str] | None = None,
    on_fallback: Callable[[str, Exception, str], None] | None = None,
    hooks: Hooks | None = None,
    execution_mode: ExecutionMode = "text",
    config: ClientConfig | None = None,
    **kwargs: Any,
) -> LLMCallResult:
    """Run the synchronous text-call runtime behind ``client.call_llm``."""
    time = _client.time
    logger = _client.logger
    litellm = _client.litellm
    _rate_limit = _client._rate_limit
    _check_model_deprecation = _client._check_model_deprecation
    _normalize_prompt_ref = _client._normalize_prompt_ref
    _require_tags = _client._require_tags
    _normalize_timeout = _client._normalize_timeout
    _check_budget = _client._check_budget
    _build_inner_named_call_kwargs = _client._build_inner_named_call_kwargs
    _resolve_call_plan = _client._resolve_call_plan
    _routing_policy_label = _client._routing_policy_label
    _validate_execution_contract = _client._validate_execution_contract
    _is_agent_model = _client._is_agent_model
    _split_agent_loop_kwargs = _client._split_agent_loop_kwargs
    _finalize_agent_loop_result = _client._finalize_agent_loop_result
    _effective_retry = _client._effective_retry
    _agent_retry_safe_enabled = _client._agent_retry_safe_enabled
    _coerce_model_kwargs_for_execution = _client._coerce_model_kwargs_for_execution
    _is_responses_api_model = _client._is_responses_api_model
    _background_mode_for_model = _client._background_mode_for_model
    _resolve_api_base_for_model = _client._resolve_api_base_for_model
    _prepare_responses_kwargs = _client._prepare_responses_kwargs
    _prepare_call_kwargs = _client._prepare_call_kwargs
    _cache_key = _client._cache_key
    _finalize_result = _client._finalize_result
    _build_routing_trace = _client._build_routing_trace
    _log_call_event = _client._log_call_event
    exponential_backoff = _client.exponential_backoff
    _maybe_poll_background_response = _client._maybe_poll_background_response
    _build_result_from_responses = _client._build_result_from_responses
    _build_result_from_response = _client._build_result_from_response
    _check_retryable = _client._check_retryable
    _compute_retry_delay = _client._compute_retry_delay
    _maybe_retry_with_openrouter_key_rotation = _client._maybe_retry_with_openrouter_key_rotation
    run_sync_with_retry = _client.run_sync_with_retry
    run_sync_with_fallback = _client.run_sync_with_fallback
    AGENT_RETRY_SAFE_ENV = _client.AGENT_RETRY_SAFE_ENV
    wrap_error = _client.wrap_error

    _check_model_deprecation(model)
    cfg = config or ClientConfig.from_env()
    _log_t0 = time.monotonic()
    task = kwargs.pop("task", None)
    trace_id = kwargs.pop("trace_id", None)
    max_budget: float | None = kwargs.pop("max_budget", None)
    prompt_ref = _normalize_prompt_ref(kwargs.pop("prompt_ref", None))
    agent_retry_safe = kwargs.pop("agent_retry_safe", None)
    task, trace_id, max_budget, _entry_warnings = _require_tags(
        task, trace_id, max_budget, caller="call_llm",
    )
    timeout = _normalize_timeout(
        timeout,
        caller="call_llm",
        warning_sink=_entry_warnings,
        logger=logger,
        log_policy_once_enabled=True,
    )
    _check_budget(trace_id, max_budget)

    snapshot_runtime_kwargs = _client._strip_llm_internal_kwargs(dict(kwargs))
    # Inject task/trace_id into litellm metadata for callback propagation
    # (e.g. Langfuse). Harmless when no callbacks are configured.
    _inject_langfuse_metadata(kwargs, task=task, trace_id=trace_id)
    public_runtime_kwargs = _client._strip_llm_internal_kwargs(dict(kwargs))
    from llm_client.observability.replay import build_call_snapshot

    call_snapshot = build_call_snapshot(
        public_api="call_llm",
        call_kind="text",
        requested_model=model,
        messages=messages,
        prompt_ref=prompt_ref,
        timeout=timeout,
        num_retries=num_retries,
        reasoning_effort=reasoning_effort,
        api_base=api_base,
        base_delay=base_delay,
        max_delay=max_delay,
        retry_on=retry_on,
        fallback_models=fallback_models,
        public_kwargs=snapshot_runtime_kwargs,
        execution_mode=execution_mode,
    )

    _inner_named = _build_inner_named_call_kwargs(
        num_retries=num_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        retry_on=retry_on,
        on_retry=on_retry,
        retry=retry,
        fallback_models=fallback_models,
        on_fallback=on_fallback,
        reasoning_effort=reasoning_effort,
        api_base=api_base,
        hooks=hooks,
        execution_mode=execution_mode,
        config=cfg,
    )

    plan = _resolve_call_plan(
        model=model,
        fallback_models=fallback_models,
        api_base=api_base,
        config=cfg,
    )
    models = plan.models
    primary_model = plan.primary_model
    fallback_chain = plan.fallback_models or None
    routing_policy = str(plan.routing_trace.get("routing_policy", _routing_policy_label(cfg)))
    if fallback_chain is not None:
        _inner_named["fallback_models"] = fallback_chain
    else:
        _inner_named.pop("fallback_models", None)
    _validate_execution_contract(
        models=models,
        execution_mode=execution_mode,
        kwargs=kwargs,
        caller="call_llm",
    )

    if ("mcp_servers" in public_runtime_kwargs or "mcp_sessions" in public_runtime_kwargs) and not _is_agent_model(model):
        from llm_client.agents import _run_sync
        from llm_client.mcp_agent import MCP_LOOP_KWARGS, acall_with_mcp_runtime

        mcp_kw, remaining = _split_agent_loop_kwargs(
            kwargs=public_runtime_kwargs,
            loop_kwargs=MCP_LOOP_KWARGS,
            task=task,
            trace_id=trace_id,
            max_budget=max_budget,
        )
        result = _run_sync(acall_with_mcp_runtime(
            primary_model, messages, timeout=timeout, **_inner_named, **mcp_kw, **remaining,
        ))
        result = cast(LLMCallResult, _finalize_agent_loop_result(
            result=result,
            requested_model=model,
            primary_model=primary_model,
            requested_api_base=api_base,
            config=cfg,
            routing_policy=routing_policy,
            caller="call_llm",
            messages=messages,
            log_started_at=_log_t0,
            task=task,
            trace_id=trace_id,
            prompt_ref=prompt_ref,
            call_snapshot=call_snapshot,
        ))
        return result

    if "python_tools" in public_runtime_kwargs and not _is_agent_model(model):
        from llm_client.agents import _run_sync
        from llm_client.mcp_agent import TOOL_LOOP_KWARGS, acall_with_python_tools_runtime
        from llm_client.core.models import supports_tool_calling

        if "mcp_servers" in public_runtime_kwargs or "mcp_sessions" in public_runtime_kwargs:
            raise ValueError("python_tools and mcp_servers/mcp_sessions are mutually exclusive.")
        tool_kw, remaining = _split_agent_loop_kwargs(
            kwargs=public_runtime_kwargs,
            loop_kwargs=TOOL_LOOP_KWARGS,
            task=task,
            trace_id=trace_id,
            max_budget=max_budget,
        )
        if not supports_tool_calling(model):
            from llm_client.tool_shim import _acall_with_tool_shim

            result = _run_sync(_acall_with_tool_shim(
                primary_model, messages, timeout=timeout, **_inner_named, **tool_kw, **remaining,
            ))
        else:
            result = _run_sync(acall_with_python_tools_runtime(
                primary_model, messages, timeout=timeout, **_inner_named, **tool_kw, **remaining,
            ))
        result = cast(LLMCallResult, _finalize_agent_loop_result(
            result=result,
            requested_model=model,
            primary_model=primary_model,
            requested_api_base=api_base,
            config=cfg,
            routing_policy=routing_policy,
            caller="call_llm",
            messages=messages,
            log_started_at=_log_t0,
            task=task,
            trace_id=trace_id,
            prompt_ref=prompt_ref,
            call_snapshot=call_snapshot,
        ))
        return result

    r = _effective_retry(retry, num_retries, base_delay, max_delay, retry_on, on_retry)
    if cache is not None and _is_agent_model(model):
        raise ValueError("Caching not supported for agent models — they have side effects.")
    _warnings: list[str] = list(_entry_warnings)
    agent_retry_safe_enabled = _agent_retry_safe_enabled(agent_retry_safe)
    last_model_attempted = model

    def _execute_model(model_idx: int, current_model: str) -> LLMCallResult:
        nonlocal last_model_attempted
        last_model_attempted = current_model
        is_agent = _is_agent_model(current_model)
        model_kwargs = _coerce_model_kwargs_for_execution(
            current_model=current_model,
            kwargs=kwargs,
            warning_sink=_warnings,
        )
        public_kwargs = _client._strip_llm_internal_kwargs(model_kwargs)

        if not is_agent and ("mcp_servers" in public_kwargs or "mcp_sessions" in public_kwargs):
            from llm_client.agents import _run_sync
            from llm_client.mcp_agent import MCP_LOOP_KWARGS, acall_with_mcp_runtime

            mcp_kw, remaining = _split_agent_loop_kwargs(
                kwargs=public_kwargs,
                loop_kwargs=MCP_LOOP_KWARGS,
                task=task,
                trace_id=trace_id,
                max_budget=max_budget,
            )
            inner_named_loop = dict(_inner_named)
            inner_named_loop.pop("fallback_models", None)
            result = _run_sync(acall_with_mcp_runtime(
                current_model,
                messages,
                timeout=timeout,
                **inner_named_loop,
                **mcp_kw,
                **remaining,
            ))
            if model_idx > 0:
                logger.info("call_llm fallback leg %d used MCP loop on %s", model_idx + 1, current_model)
            return cast(LLMCallResult, result)

        if not is_agent and "python_tools" in public_kwargs:
            from llm_client.agents import _run_sync
            from llm_client.mcp_agent import TOOL_LOOP_KWARGS, acall_with_python_tools_runtime
            from llm_client.core.models import supports_tool_calling

            tool_kw, remaining = _split_agent_loop_kwargs(
                kwargs=public_kwargs,
                loop_kwargs=TOOL_LOOP_KWARGS,
                task=task,
                trace_id=trace_id,
                max_budget=max_budget,
            )
            inner_named_loop = dict(_inner_named)
            inner_named_loop.pop("fallback_models", None)
            if not supports_tool_calling(current_model):
                from llm_client.tool_shim import _acall_with_tool_shim

                result = _run_sync(_acall_with_tool_shim(
                    current_model,
                    messages,
                    timeout=timeout,
                    **inner_named_loop,
                    **tool_kw,
                    **remaining,
                ))
            else:
                result = _run_sync(acall_with_python_tools_runtime(
                    current_model,
                    messages,
                    timeout=timeout,
                    **inner_named_loop,
                    **tool_kw,
                    **remaining,
                ))
            if model_idx > 0:
                logger.info("call_llm fallback leg %d used Python tool loop on %s", model_idx + 1, current_model)
            return cast(LLMCallResult, result)

        use_responses = not is_agent and _is_responses_api_model(current_model)
        background_mode = _background_mode_for_model(
            model=current_model,
            use_responses=use_responses,
            reasoning_effort=reasoning_effort,
        )
        current_api_base = _resolve_api_base_for_model(current_model, api_base, cfg)

        if is_agent:
            pass
        elif use_responses:
            call_kwargs = _prepare_responses_kwargs(
                current_model, messages,
                timeout=timeout,
                reasoning_effort=reasoning_effort,
                api_base=current_api_base,
                kwargs=public_kwargs,
                warning_sink=_warnings,
            )
        else:
            call_kwargs = _prepare_call_kwargs(
                current_model, messages,
                timeout=timeout,
                num_retries=r.max_retries,
                reasoning_effort=reasoning_effort,
                api_base=current_api_base,
                kwargs=public_kwargs,
                warning_sink=_warnings,
            )

        key: str | None = None
        if cache is not None:
            key = _cache_key(current_model, messages, **public_kwargs)
            cached = cache.get(key)
            if cached is not None:
                cached_result = cast(LLMCallResult, _finalize_result(
                    cached,
                    cache_hit=True,
                    requested_model=model,
                    resolved_model=current_model,
                    routing_trace=_build_routing_trace(
                        requested_model=model,
                        attempted_models=models[:model_idx + 1],
                        selected_model=current_model,
                        requested_api_base=api_base,
                        effective_api_base=current_api_base,
                        background_mode=background_mode,
                        routing_policy=routing_policy,
                    ),
                ))
                _log_call_event(
                    model=current_model,
                    messages=messages,
                    result=cached_result,
                    latency_s=time.monotonic() - _log_t0,
                    caller="call_llm",
                    task=task,
                    trace_id=trace_id,
                    prompt_ref=prompt_ref,
                    call_snapshot=call_snapshot,
                )
                return cached_result

        if hooks and hooks.before_call:
            hooks.before_call(current_model, messages, public_kwargs)

        backoff_fn = r.backoff or exponential_backoff
        if is_agent and not agent_retry_safe_enabled:
            effective_retries = 0
            if r.max_retries > 0:
                msg = (
                    "AGENT_RETRY_DISABLED: retries for agent models are disabled by default "
                    "to avoid duplicate side effects. Set agent_retry_safe=True (or "
                    f"{AGENT_RETRY_SAFE_ENV}=1) only for explicitly safe/read-only runs."
                )
                if msg not in _warnings:
                    _warnings.append(msg)
                    logger.warning(msg)
        else:
            effective_retries = r.max_retries

        def _invoke_attempt(attempt: int) -> LLMCallResult:
            if is_agent:
                from llm_client.agents import _route_call

                result = _route_call(
                    current_model, messages,
                    timeout=timeout, **public_kwargs,
                )
            elif use_responses:
                with _rate_limit.acquire(current_model):
                    response = litellm.responses(**call_kwargs)
                response = _maybe_poll_background_response(
                    response,
                    api_base=current_api_base,
                    request_timeout=(timeout if timeout > 0 else None),
                    model_kwargs=public_kwargs,
                )
                result = _build_result_from_responses(response, current_model, warnings=_warnings)
            else:
                with _rate_limit.acquire(current_model):
                    response = litellm.completion(**call_kwargs)
                result = _build_result_from_response(response, current_model, warnings=_warnings)
            if attempt > 0:
                logger.info("call_llm succeeded after %d retries", attempt)
            resolved_model = result.resolved_model if is_agent else current_model
            result = cast(LLMCallResult, _finalize_result(
                result,
                requested_model=model,
                resolved_model=resolved_model,
                routing_trace=_build_routing_trace(
                    requested_model=model,
                    attempted_models=models[:model_idx + 1],
                    selected_model=resolved_model,
                    requested_api_base=api_base,
                    effective_api_base=current_api_base,
                    sticky_fallback=any("STICKY_FALLBACK" in w for w in (result.warnings or [])),
                    background_mode=background_mode,
                    routing_policy=routing_policy,
                ),
            ))
            if hooks and hooks.after_call:
                hooks.after_call(result)
            if cache is not None and key is not None:
                cache.set(key, result)
            _log_call_event(
                model=current_model,
                messages=messages,
                result=result,
                latency_s=time.monotonic() - _log_t0,
                caller="call_llm",
                task=task,
                trace_id=trace_id,
                prompt_ref=prompt_ref,
                call_snapshot=call_snapshot,
            )
            return result

        return cast(LLMCallResult, run_sync_with_retry(
            caller="call_llm",
            model=current_model,
            max_retries=effective_retries,
            invoke=_invoke_attempt,
            should_retry=lambda exc: _check_retryable(exc, r),
            compute_delay=lambda attempt, exc: _compute_retry_delay(
                attempt=attempt,
                error=exc,
                policy=r,
                backoff_fn=backoff_fn,
            ),
            warning_sink=_warnings,
            logger=logger,
            on_error=(hooks.on_error if hooks and hooks.on_error else None),
            on_retry=r.on_retry,
                maybe_retry_hook=lambda exc, attempt, max_retries: _maybe_retry_with_openrouter_key_rotation(
                    error=exc,
                    attempt=attempt,
                    max_retries=max_retries,
                    current_model=current_model,
                    current_api_base=current_api_base,
                    user_kwargs=public_kwargs,
                    warning_sink=_warnings,
                    on_retry=r.on_retry,
                    caller="call_llm",
            ),
        ))

    try:
        return cast(LLMCallResult, run_sync_with_fallback(
            models=models,
            execute_model=_execute_model,
            on_fallback=on_fallback,
            warning_sink=_warnings,
            logger=logger,
        ))
    except Exception as e:
        _log_call_event(
            model=last_model_attempted,
            messages=messages,
            error=e,
            latency_s=time.monotonic() - _log_t0,
            caller="call_llm",
            task=task,
            trace_id=trace_id,
            prompt_ref=prompt_ref,
            call_snapshot=call_snapshot,
        )
        raise wrap_error(e) from e


async def _acall_llm_impl(
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
    cache: CachePolicy | AsyncCachePolicy | None = None,
    retry: RetryPolicy | None = None,
    fallback_models: list[str] | None = None,
    on_fallback: Callable[[str, Exception, str], None] | None = None,
    hooks: Hooks | None = None,
    execution_mode: ExecutionMode = "text",
    config: ClientConfig | None = None,
    **kwargs: Any,
) -> LLMCallResult:
    """Run the asynchronous text-call runtime behind ``client.acall_llm``."""
    time = _client.time
    logger = _client.logger
    litellm = _client.litellm
    _rate_limit = _client._rate_limit
    _check_model_deprecation = _client._check_model_deprecation
    _normalize_prompt_ref = _client._normalize_prompt_ref
    _require_tags = _client._require_tags
    _normalize_timeout = _client._normalize_timeout
    _check_budget = _client._check_budget
    _build_inner_named_call_kwargs = _client._build_inner_named_call_kwargs
    _resolve_call_plan = _client._resolve_call_plan
    _routing_policy_label = _client._routing_policy_label
    _validate_execution_contract = _client._validate_execution_contract
    _is_agent_model = _client._is_agent_model
    _split_agent_loop_kwargs = _client._split_agent_loop_kwargs
    _finalize_agent_loop_result = _client._finalize_agent_loop_result
    _effective_retry = _client._effective_retry
    _agent_retry_safe_enabled = _client._agent_retry_safe_enabled
    _coerce_model_kwargs_for_execution = _client._coerce_model_kwargs_for_execution
    _is_responses_api_model = _client._is_responses_api_model
    _background_mode_for_model = _client._background_mode_for_model
    _resolve_api_base_for_model = _client._resolve_api_base_for_model
    _prepare_responses_kwargs = _client._prepare_responses_kwargs
    _prepare_call_kwargs = _client._prepare_call_kwargs
    _cache_key = _client._cache_key
    _async_cache_get = _client._async_cache_get
    _async_cache_set = _client._async_cache_set
    _finalize_result = _client._finalize_result
    _build_routing_trace = _client._build_routing_trace
    _log_call_event = _client._log_call_event
    exponential_backoff = _client.exponential_backoff
    _maybe_apoll_background_response = _client._maybe_apoll_background_response
    _build_result_from_responses = _client._build_result_from_responses
    _build_result_from_response = _client._build_result_from_response
    _check_retryable = _client._check_retryable
    _compute_retry_delay = _client._compute_retry_delay
    _maybe_retry_with_openrouter_key_rotation = _client._maybe_retry_with_openrouter_key_rotation
    run_async_with_retry = _client.run_async_with_retry
    run_async_with_fallback = _client.run_async_with_fallback
    AGENT_RETRY_SAFE_ENV = _client.AGENT_RETRY_SAFE_ENV
    wrap_error = _client.wrap_error

    _check_model_deprecation(model)
    cfg = config or ClientConfig.from_env()
    _log_t0 = time.monotonic()
    task = kwargs.pop("task", None)
    trace_id = kwargs.pop("trace_id", None)
    max_budget: float | None = kwargs.pop("max_budget", None)
    prompt_ref = _normalize_prompt_ref(kwargs.pop("prompt_ref", None))
    agent_retry_safe = kwargs.pop("agent_retry_safe", None)
    task, trace_id, max_budget, _entry_warnings = _require_tags(
        task, trace_id, max_budget, caller="acall_llm",
    )
    timeout = _normalize_timeout(
        timeout,
        caller="acall_llm",
        warning_sink=_entry_warnings,
        logger=logger,
        log_policy_once_enabled=True,
    )
    _check_budget(trace_id, max_budget)

    snapshot_runtime_kwargs = _client._strip_llm_internal_kwargs(dict(kwargs))
    # Inject task/trace_id into litellm metadata for callback propagation
    # (e.g. Langfuse). Harmless when no callbacks are configured.
    _inject_langfuse_metadata(kwargs, task=task, trace_id=trace_id)
    public_runtime_kwargs = _client._strip_llm_internal_kwargs(dict(kwargs))
    from llm_client.observability.replay import build_call_snapshot

    call_snapshot = build_call_snapshot(
        public_api="acall_llm",
        call_kind="text",
        requested_model=model,
        messages=messages,
        prompt_ref=prompt_ref,
        timeout=timeout,
        num_retries=num_retries,
        reasoning_effort=reasoning_effort,
        api_base=api_base,
        base_delay=base_delay,
        max_delay=max_delay,
        retry_on=retry_on,
        fallback_models=fallback_models,
        public_kwargs=snapshot_runtime_kwargs,
        execution_mode=execution_mode,
    )

    _inner_named = _build_inner_named_call_kwargs(
        num_retries=num_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        retry_on=retry_on,
        on_retry=on_retry,
        retry=retry,
        fallback_models=fallback_models,
        on_fallback=on_fallback,
        reasoning_effort=reasoning_effort,
        api_base=api_base,
        hooks=hooks,
        execution_mode=execution_mode,
        config=cfg,
    )

    plan = _resolve_call_plan(
        model=model,
        fallback_models=fallback_models,
        api_base=api_base,
        config=cfg,
    )
    models = plan.models
    primary_model = plan.primary_model
    fallback_chain = plan.fallback_models or None
    routing_policy = str(plan.routing_trace.get("routing_policy", _routing_policy_label(cfg)))
    if fallback_chain is not None:
        _inner_named["fallback_models"] = fallback_chain
    else:
        _inner_named.pop("fallback_models", None)
    _validate_execution_contract(
        models=models,
        execution_mode=execution_mode,
        kwargs=kwargs,
        caller="acall_llm",
    )

    if ("mcp_servers" in public_runtime_kwargs or "mcp_sessions" in public_runtime_kwargs) and not _is_agent_model(model):
        from llm_client.mcp_agent import MCP_LOOP_KWARGS, acall_with_mcp_runtime

        mcp_kw, remaining = _split_agent_loop_kwargs(
            kwargs=public_runtime_kwargs,
            loop_kwargs=MCP_LOOP_KWARGS,
            task=task,
            trace_id=trace_id,
            max_budget=max_budget,
        )
        result = await acall_with_mcp_runtime(
            primary_model, messages, timeout=timeout, **_inner_named, **mcp_kw, **remaining,
        )
        result = cast(LLMCallResult, _finalize_agent_loop_result(
            result=result,
            requested_model=model,
            primary_model=primary_model,
            requested_api_base=api_base,
            config=cfg,
            routing_policy=routing_policy,
            caller="acall_llm",
            messages=messages,
            log_started_at=_log_t0,
            task=task,
            trace_id=trace_id,
            prompt_ref=prompt_ref,
            call_snapshot=call_snapshot,
        ))
        return result

    if "python_tools" in public_runtime_kwargs and not _is_agent_model(model):
        from llm_client.mcp_agent import TOOL_LOOP_KWARGS, acall_with_python_tools_runtime
        from llm_client.core.models import supports_tool_calling

        if "mcp_servers" in public_runtime_kwargs or "mcp_sessions" in public_runtime_kwargs:
            raise ValueError("python_tools and mcp_servers/mcp_sessions are mutually exclusive.")
        tool_kw, remaining = _split_agent_loop_kwargs(
            kwargs=public_runtime_kwargs,
            loop_kwargs=TOOL_LOOP_KWARGS,
            task=task,
            trace_id=trace_id,
            max_budget=max_budget,
        )
        if not supports_tool_calling(model):
            from llm_client.tool_shim import _acall_with_tool_shim

            result = await _acall_with_tool_shim(
                primary_model, messages, timeout=timeout, **_inner_named, **tool_kw, **remaining,
            )
        else:
            result = await acall_with_python_tools_runtime(
                primary_model, messages, timeout=timeout, **_inner_named, **tool_kw, **remaining,
            )
        result = cast(LLMCallResult, _finalize_agent_loop_result(
            result=result,
            requested_model=model,
            primary_model=primary_model,
            requested_api_base=api_base,
            config=cfg,
            routing_policy=routing_policy,
            caller="acall_llm",
            messages=messages,
            log_started_at=_log_t0,
            task=task,
            trace_id=trace_id,
            prompt_ref=prompt_ref,
            call_snapshot=call_snapshot,
        ))
        return result

    r = _effective_retry(retry, num_retries, base_delay, max_delay, retry_on, on_retry)
    if cache is not None and _is_agent_model(model):
        raise ValueError("Caching not supported for agent models — they have side effects.")
    _warnings: list[str] = list(_entry_warnings)
    agent_retry_safe_enabled = _agent_retry_safe_enabled(agent_retry_safe)
    last_model_attempted = model

    async def _execute_model(model_idx: int, current_model: str) -> LLMCallResult:
        nonlocal last_model_attempted
        last_model_attempted = current_model
        is_agent = _is_agent_model(current_model)
        model_kwargs = _coerce_model_kwargs_for_execution(
            current_model=current_model,
            kwargs=kwargs,
            warning_sink=_warnings,
        )
        public_kwargs = _client._strip_llm_internal_kwargs(model_kwargs)

        if not is_agent and ("mcp_servers" in public_kwargs or "mcp_sessions" in public_kwargs):
            from llm_client.mcp_agent import MCP_LOOP_KWARGS, acall_with_mcp_runtime

            mcp_kw, remaining = _split_agent_loop_kwargs(
                kwargs=public_kwargs,
                loop_kwargs=MCP_LOOP_KWARGS,
                task=task,
                trace_id=trace_id,
                max_budget=max_budget,
            )
            inner_named_loop = dict(_inner_named)
            inner_named_loop.pop("fallback_models", None)
            result = await acall_with_mcp_runtime(
                current_model,
                messages,
                timeout=timeout,
                **inner_named_loop,
                **mcp_kw,
                **remaining,
            )
            if model_idx > 0:
                logger.info("acall_llm fallback leg %d used MCP loop on %s", model_idx + 1, current_model)
            return result

        if not is_agent and "python_tools" in public_kwargs:
            from llm_client.mcp_agent import TOOL_LOOP_KWARGS, acall_with_python_tools_runtime
            from llm_client.core.models import supports_tool_calling

            tool_kw, remaining = _split_agent_loop_kwargs(
                kwargs=public_kwargs,
                loop_kwargs=TOOL_LOOP_KWARGS,
                task=task,
                trace_id=trace_id,
                max_budget=max_budget,
            )
            inner_named_loop = dict(_inner_named)
            inner_named_loop.pop("fallback_models", None)
            if not supports_tool_calling(current_model):
                from llm_client.tool_shim import _acall_with_tool_shim

                result = await _acall_with_tool_shim(
                    current_model,
                    messages,
                    timeout=timeout,
                    **inner_named_loop,
                    **tool_kw,
                    **remaining,
                )
            else:
                result = await acall_with_python_tools_runtime(
                    current_model,
                    messages,
                    timeout=timeout,
                    **inner_named_loop,
                    **tool_kw,
                    **remaining,
                )
            if model_idx > 0:
                logger.info("acall_llm fallback leg %d used Python tool loop on %s", model_idx + 1, current_model)
            return result

        use_responses = not is_agent and _is_responses_api_model(current_model)
        background_mode = _background_mode_for_model(
            model=current_model,
            use_responses=use_responses,
            reasoning_effort=reasoning_effort,
        )
        current_api_base = _resolve_api_base_for_model(current_model, api_base, cfg)

        if is_agent:
            pass
        elif use_responses:
            call_kwargs = _prepare_responses_kwargs(
                current_model, messages,
                timeout=timeout,
                reasoning_effort=reasoning_effort,
                api_base=current_api_base,
                kwargs=public_kwargs,
                warning_sink=_warnings,
            )
        else:
            call_kwargs = _prepare_call_kwargs(
                current_model, messages,
                timeout=timeout,
                num_retries=r.max_retries,
                reasoning_effort=reasoning_effort,
                api_base=current_api_base,
                kwargs=public_kwargs,
                warning_sink=_warnings,
            )

        key: str | None = None
        if cache is not None:
            key = _cache_key(current_model, messages, **public_kwargs)
            cached = await _async_cache_get(cache, key)
            if cached is not None:
                cached_result = cast(LLMCallResult, _finalize_result(
                    cached,
                    cache_hit=True,
                    requested_model=model,
                    resolved_model=current_model,
                    routing_trace=_build_routing_trace(
                        requested_model=model,
                        attempted_models=models[:model_idx + 1],
                        selected_model=current_model,
                        requested_api_base=api_base,
                        effective_api_base=current_api_base,
                        background_mode=background_mode,
                        routing_policy=routing_policy,
                    ),
                ))
                _log_call_event(
                    model=current_model,
                    messages=messages,
                    result=cached_result,
                    latency_s=time.monotonic() - _log_t0,
                    caller="acall_llm",
                    task=task,
                    trace_id=trace_id,
                    prompt_ref=prompt_ref,
                    call_snapshot=call_snapshot,
                )
                return cached_result

        if hooks and hooks.before_call:
            hooks.before_call(current_model, messages, public_kwargs)

        backoff_fn = r.backoff or exponential_backoff
        if is_agent and not agent_retry_safe_enabled:
            effective_retries = 0
            if r.max_retries > 0:
                msg = (
                    "AGENT_RETRY_DISABLED: retries for agent models are disabled by default "
                    "to avoid duplicate side effects. Set agent_retry_safe=True (or "
                    f"{AGENT_RETRY_SAFE_ENV}=1) only for explicitly safe/read-only runs."
                )
                if msg not in _warnings:
                    _warnings.append(msg)
                    logger.warning(msg)
        else:
            effective_retries = r.max_retries

        async def _invoke_attempt(attempt: int) -> LLMCallResult:
            if is_agent:
                from llm_client.agents import _route_acall

                result = await _route_acall(
                    current_model, messages,
                    timeout=timeout, **public_kwargs,
                )
            elif use_responses:
                async with _rate_limit.aacquire(current_model):
                    response = await litellm.aresponses(**call_kwargs)
                response = await _maybe_apoll_background_response(
                    response,
                    api_base=current_api_base,
                    request_timeout=(timeout if timeout > 0 else None),
                    model_kwargs=public_kwargs,
                )
                result = _build_result_from_responses(response, current_model, warnings=_warnings)
            else:
                async with _rate_limit.aacquire(current_model):
                    response = await litellm.acompletion(**call_kwargs)
                result = _build_result_from_response(response, current_model, warnings=_warnings)
            if attempt > 0:
                logger.info("acall_llm succeeded after %d retries", attempt)
            resolved_model = result.resolved_model if is_agent else current_model
            result = cast(LLMCallResult, _finalize_result(
                result,
                requested_model=model,
                resolved_model=resolved_model,
                routing_trace=_build_routing_trace(
                    requested_model=model,
                    attempted_models=models[:model_idx + 1],
                    selected_model=resolved_model,
                    requested_api_base=api_base,
                    effective_api_base=current_api_base,
                    sticky_fallback=any("STICKY_FALLBACK" in w for w in (result.warnings or [])),
                    background_mode=background_mode,
                    routing_policy=routing_policy,
                ),
            ))
            if hooks and hooks.after_call:
                hooks.after_call(result)
            if cache is not None and key is not None:
                await _async_cache_set(cache, key, result)
            _log_call_event(
                model=current_model,
                messages=messages,
                result=result,
                latency_s=time.monotonic() - _log_t0,
                caller="acall_llm",
                task=task,
                trace_id=trace_id,
                prompt_ref=prompt_ref,
                call_snapshot=call_snapshot,
            )
            return result

        return cast(LLMCallResult, await run_async_with_retry(
            caller="acall_llm",
            model=current_model,
            max_retries=effective_retries,
            invoke=_invoke_attempt,
            should_retry=lambda exc: _check_retryable(exc, r),
            compute_delay=lambda attempt, exc: _compute_retry_delay(
                attempt=attempt,
                error=exc,
                policy=r,
                backoff_fn=backoff_fn,
            ),
            warning_sink=_warnings,
            logger=logger,
            on_error=(hooks.on_error if hooks and hooks.on_error else None),
            on_retry=r.on_retry,
                maybe_retry_hook=lambda exc, attempt, max_retries: _maybe_retry_with_openrouter_key_rotation(
                    error=exc,
                    attempt=attempt,
                    max_retries=max_retries,
                    current_model=current_model,
                    current_api_base=current_api_base,
                    user_kwargs=public_kwargs,
                    warning_sink=_warnings,
                    on_retry=r.on_retry,
                    caller="acall_llm",
                ),
        ))

    try:
        return cast(LLMCallResult, await run_async_with_fallback(
            models=models,
            execute_model=_execute_model,
            on_fallback=on_fallback,
            warning_sink=_warnings,
            logger=logger,
        ))
    except Exception as e:
        _log_call_event(
            model=last_model_attempted,
            messages=messages,
            error=e,
            latency_s=time.monotonic() - _log_t0,
            caller="acall_llm",
            task=task,
            trace_id=trace_id,
            prompt_ref=prompt_ref,
            call_snapshot=call_snapshot,
        )
        raise wrap_error(e) from e
