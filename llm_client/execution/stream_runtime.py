"""Stream execution internals extracted from client.py."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable, cast

import llm_client.client as _client_mod
from llm_client.langfuse_callbacks import inject_metadata as _inject_langfuse_metadata

_client: Any = _client_mod

if TYPE_CHECKING:
    from llm_client.core.client import (
        AsyncLLMStream,
        Hooks,
        LLMStream,
        RetryPolicy,
    )
    from llm_client.core.config import ClientConfig


def _stream_terminal_payload(
    stream: Any,
    resolved_model: str | None,
) -> str | None:
    """Resolve a stable resolved model id for lifecycle completion/failure payloads."""

    if stream is not None:
        try:
            result = stream.result
        except Exception:
            return resolved_model
        return getattr(result, "resolved_model", resolved_model)
    return resolved_model


def _emit_stream_terminal_event(
    *,
    stream: Any,
    error: Exception | None,
    call_id: str,
    caller: str,
    task: str,
    trace_id: str,
    requested_model: str,
    resolved_model: str | None,
    provider_timeout_s: int,
    prompt_ref: str | None,
    heartbeat_interval_s: float,
    stall_after_s: float,
    started_at: float,
    monitor: Any,
) -> None:
    """Emit completed/failed lifecycle row for a stream using current monitor state."""

    snapshot = monitor.snapshot() if monitor is not None else None

    elapsed_s = time.monotonic() - started_at
    if error is None:
        resolved = _stream_terminal_payload(
            stream,
            resolved_model=resolved_model,
        )
        try:
            _client._emit_llm_call_lifecycle_event(
                call_id=call_id,
                phase="completed",
                call_kind="text",
                caller=caller,
                task=task,
                trace_id=trace_id,
                requested_model=requested_model,
                provider_timeout_s=provider_timeout_s,
                prompt_ref=prompt_ref,
                resolved_model=resolved,
                elapsed_s=elapsed_s,
                latency_s=elapsed_s,
                heartbeat_interval_s=heartbeat_interval_s,
                stall_after_s=stall_after_s,
                progress_observable=(
                    snapshot.progress_observable if snapshot is not None else None
                ),
                progress_source=(
                    snapshot.progress_source if snapshot is not None else None
                ),
                progress_event_count=(
                    snapshot.progress_event_count if snapshot is not None else None
                ),
            )
            return
        except Exception as exc:
            error = exc

    _client._emit_llm_call_lifecycle_event(
        call_id=call_id,
        phase="failed",
        call_kind="text",
        caller=caller,
        task=task,
        trace_id=trace_id,
        requested_model=requested_model,
        provider_timeout_s=provider_timeout_s,
        prompt_ref=prompt_ref,
        resolved_model=resolved_model,
        elapsed_s=elapsed_s,
        latency_s=elapsed_s,
        heartbeat_interval_s=heartbeat_interval_s,
        stall_after_s=stall_after_s,
        progress_observable=(
            snapshot.progress_observable if snapshot is not None else None
        ),
        progress_source=(
            snapshot.progress_source if snapshot is not None else None
        ),
        progress_event_count=(
            snapshot.progress_event_count if snapshot is not None else None
        ),
        error=error,
    )


class _SyncStreamLifecycleAdapter:
    """Wrap a sync stream and emit lifecycle terminal events when consumed."""

    def __init__(
        self,
        stream: Any,
        *,
        call_id: str,
        requested_model: str,
        resolved_model: str | None,
        monitor: Any,
        caller: str,
        task: str,
        trace_id: str,
        provider_timeout_s: int,
        prompt_ref: str | None,
        heartbeat_interval_s: float,
        stall_after_s: float,
        started_at: float,
    ) -> None:
        self._stream = stream
        self._call_id = call_id
        self._requested_model = requested_model
        self._resolved_model = resolved_model
        self._monitor = monitor
        self._caller = caller
        self._task = task
        self._trace_id = trace_id
        self._provider_timeout_s = provider_timeout_s
        self._prompt_ref = prompt_ref
        self._heartbeat_interval_s = heartbeat_interval_s
        self._stall_after_s = stall_after_s
        self._started_at = started_at
        self._finalized = False

    def __iter__(self) -> "_SyncStreamLifecycleAdapter":
        return self

    def __next__(self) -> str:
        try:
            chunk = next(self._stream)
        except StopIteration:
            self._finalize()
            raise
        except Exception as exc:
            self._finalize(error=exc)
            raise
        if self._monitor is not None:
            self._monitor.mark_progress(source="stream_chunk")
        return chunk

    def _finalize(self, *, error: Exception | None = None) -> None:
        if self._finalized:
            return
        self._finalized = True
        if self._monitor is not None:
            self._monitor.stop()
        _emit_stream_terminal_event(
            stream=self._stream,
            error=error,
            call_id=self._call_id,
            caller=self._caller,
            task=self._task,
            trace_id=self._trace_id,
            requested_model=self._requested_model,
            resolved_model=self._resolved_model,
            provider_timeout_s=self._provider_timeout_s,
            prompt_ref=self._prompt_ref,
            heartbeat_interval_s=self._heartbeat_interval_s,
            stall_after_s=self._stall_after_s,
            started_at=self._started_at,
            monitor=self._monitor,
        )

    @property
    def result(self) -> Any:
        """Expose result through the same interface as wrapped streams."""
        return self._stream.result


class _AsyncStreamLifecycleAdapter:
    """Wrap an async stream and emit lifecycle terminal events when consumed."""

    def __init__(
        self,
        stream: Any,
        *,
        call_id: str,
        requested_model: str,
        resolved_model: str | None,
        monitor: Any,
        caller: str,
        task: str,
        trace_id: str,
        provider_timeout_s: int,
        prompt_ref: str | None,
        heartbeat_interval_s: float,
        stall_after_s: float,
        started_at: float,
    ) -> None:
        self._stream = stream
        self._call_id = call_id
        self._requested_model = requested_model
        self._resolved_model = resolved_model
        self._monitor = monitor
        self._caller = caller
        self._task = task
        self._trace_id = trace_id
        self._provider_timeout_s = provider_timeout_s
        self._prompt_ref = prompt_ref
        self._heartbeat_interval_s = heartbeat_interval_s
        self._stall_after_s = stall_after_s
        self._started_at = started_at
        self._finalized = False

    def __aiter__(self) -> "_AsyncStreamLifecycleAdapter":
        return self

    async def __anext__(self) -> str:
        try:
            chunk = await self._stream.__anext__()
        except StopAsyncIteration:
            await self._finalize()
            raise
        except Exception as exc:
            await self._finalize(error=exc)
            raise
        if self._monitor is not None:
            self._monitor.mark_progress(source="stream_chunk")
        return chunk

    async def _finalize(self, *, error: Exception | None = None) -> None:
        if self._finalized:
            return
        self._finalized = True
        if self._monitor is not None:
            await self._monitor.stop()
        _emit_stream_terminal_event(
            stream=self._stream,
            error=error,
            call_id=self._call_id,
            caller=self._caller,
            task=self._task,
            trace_id=self._trace_id,
            requested_model=self._requested_model,
            resolved_model=self._resolved_model,
            provider_timeout_s=self._provider_timeout_s,
            prompt_ref=self._prompt_ref,
            heartbeat_interval_s=self._heartbeat_interval_s,
            stall_after_s=self._stall_after_s,
            started_at=self._started_at,
            monitor=self._monitor,
        )

    @property
    def result(self) -> Any:
        """Expose result through the same interface as wrapped streams."""
        return self._stream.result


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
    timeout = _client._normalize_timeout(timeout, caller="stream_llm")
    task = kwargs.pop("task", None)
    trace_id = kwargs.pop("trace_id", None)
    max_budget: float | None = kwargs.pop("max_budget", None)
    prompt_ref = _client._normalize_prompt_ref(kwargs.pop("prompt_ref", None))
    task, trace_id, max_budget, _entry_warnings = _client._require_tags(
        task,
        trace_id,
        max_budget,
        caller="stream_llm",
    )
    _client._check_budget(trace_id, max_budget)
    _inject_langfuse_metadata(kwargs, task=task, trace_id=trace_id)

    runtime_kwargs = dict(kwargs)
    heartbeat_interval_s, stall_after_s = _client._resolve_lifecycle_monitoring_settings(
        heartbeat_interval=runtime_kwargs.pop("lifecycle_heartbeat_interval_s", None),
        stall_after=runtime_kwargs.pop("lifecycle_stall_after_s", None),
    )
    public_kwargs = _client._strip_llm_internal_kwargs(runtime_kwargs)
    provider_timeout_s = _client._provider_timeout_for_lifecycle(timeout)

    r = _client._effective_retry(retry, num_retries, base_delay, max_delay, retry_on, on_retry)
    plan = _client._resolve_call_plan(
        model=model,
        fallback_models=fallback_models,
        api_base=api_base,
        config=cfg,
    )
    models = plan.models
    routing_policy = str(plan.routing_trace.get("routing_policy", _client._routing_policy_label(cfg)))
    _warnings = list(_entry_warnings)
    backoff_fn = r.backoff or _client.exponential_backoff

    call_id = _client._new_llm_call_lifecycle_id()
    started_at = time.monotonic()
    monitor = _client._SyncLLMCallHeartbeatMonitor(
        call_id=call_id,
        call_kind="text",
        caller="stream_llm",
        task=task,
        trace_id=trace_id,
        requested_model=model,
        provider_timeout_s=provider_timeout_s,
        prompt_ref=prompt_ref,
        heartbeat_interval_s=heartbeat_interval_s,
        stall_after_s=stall_after_s,
        started_at=started_at,
    )

    started_snapshot = monitor.snapshot()
    _client._emit_llm_call_lifecycle_event(
        call_id=call_id,
        phase="started",
        call_kind="text",
        caller="stream_llm",
        task=task,
        trace_id=trace_id,
        requested_model=model,
        provider_timeout_s=provider_timeout_s,
        prompt_ref=prompt_ref,
        heartbeat_interval_s=heartbeat_interval_s,
        stall_after_s=stall_after_s,
        progress_observable=started_snapshot.progress_observable,
        progress_source=started_snapshot.progress_source,
        progress_event_count=started_snapshot.progress_event_count,
    )
    monitor.start()

    def _execute_stream_model(model_idx: int, current_model: str) -> LLMStream:
        current_api_base = _client._resolve_api_base_for_model(current_model, api_base, cfg)
        call_kwargs = _client._prepare_call_kwargs(
            current_model,
            messages,
            timeout=timeout,
            num_retries=0,
            reasoning_effort=reasoning_effort,
            api_base=current_api_base,
            kwargs=runtime_kwargs,
            warning_sink=_warnings,
        )
        call_kwargs["stream"] = True

        if hooks and hooks.before_call:
            hooks.before_call(current_model, messages, public_kwargs)

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
                    prompt_ref=prompt_ref,
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
                    user_kwargs=public_kwargs,
                    warning_sink=_warnings,
                    on_retry=r.on_retry,
                    caller="stream_llm",
                ),
            ),
        )

    try:
        if _client._is_agent_model(model):
            from llm_client.sdk.agents import _route_stream

            stream = cast(
                "LLMStream",
                _route_stream(model, messages, hooks=hooks, timeout=timeout, **runtime_kwargs),
            )
        else:
            runtime_kwargs["_lifecycle_monitor"] = monitor
            stream = cast(
                "LLMStream",
                _client.run_sync_with_fallback(
                    models=models,
                    execute_model=_execute_stream_model,
                    on_fallback=on_fallback,
                    warning_sink=_warnings,
                    logger=_client.logger,
                ),
            )

        return _SyncStreamLifecycleAdapter(
            stream=stream,
            call_id=call_id,
            requested_model=model,
            resolved_model=(
                getattr(stream, "_resolved_model", None)
                if hasattr(stream, "_resolved_model")
                else getattr(stream, "resolved_model", None)
            ),
            monitor=monitor,
            caller="stream_llm",
            task=task,
            trace_id=trace_id,
            provider_timeout_s=provider_timeout_s,
            prompt_ref=prompt_ref,
            heartbeat_interval_s=heartbeat_interval_s,
            stall_after_s=stall_after_s,
            started_at=started_at,
        )
    except Exception as exc:
        _emit_stream_terminal_event(
            stream=None,
            error=exc,
            call_id=call_id,
            caller="stream_llm",
            task=task,
            trace_id=trace_id,
            requested_model=model,
            resolved_model=model,
            provider_timeout_s=provider_timeout_s,
            prompt_ref=prompt_ref,
            heartbeat_interval_s=heartbeat_interval_s,
            stall_after_s=stall_after_s,
            started_at=started_at,
            monitor=monitor,
        )
        raise _client.wrap_error(exc) from exc


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
    timeout = _client._normalize_timeout(timeout, caller="astream_llm")
    task = kwargs.pop("task", None)
    trace_id = kwargs.pop("trace_id", None)
    max_budget: float | None = kwargs.pop("max_budget", None)
    prompt_ref = _client._normalize_prompt_ref(kwargs.pop("prompt_ref", None))
    task, trace_id, max_budget, _entry_warnings = _client._require_tags(
        task,
        trace_id,
        max_budget,
        caller="astream_llm",
    )
    _client._check_budget(trace_id, max_budget)
    _inject_langfuse_metadata(kwargs, task=task, trace_id=trace_id)

    runtime_kwargs = dict(kwargs)
    heartbeat_interval_s, stall_after_s = _client._resolve_lifecycle_monitoring_settings(
        heartbeat_interval=runtime_kwargs.pop("lifecycle_heartbeat_interval_s", None),
        stall_after=runtime_kwargs.pop("lifecycle_stall_after_s", None),
    )
    public_kwargs = _client._strip_llm_internal_kwargs(runtime_kwargs)
    provider_timeout_s = _client._provider_timeout_for_lifecycle(timeout)

    r = _client._effective_retry(retry, num_retries, base_delay, max_delay, retry_on, on_retry)
    plan = _client._resolve_call_plan(
        model=model,
        fallback_models=fallback_models,
        api_base=api_base,
        config=cfg,
    )
    models = plan.models
    routing_policy = str(plan.routing_trace.get("routing_policy", _client._routing_policy_label(cfg)))
    _warnings = list(_entry_warnings)
    backoff_fn = r.backoff or _client.exponential_backoff

    call_id = _client._new_llm_call_lifecycle_id()
    started_at = time.monotonic()
    monitor = _client._AsyncLLMCallHeartbeatMonitor(
        call_id=call_id,
        call_kind="text",
        caller="astream_llm",
        task=task,
        trace_id=trace_id,
        requested_model=model,
        provider_timeout_s=provider_timeout_s,
        prompt_ref=prompt_ref,
        heartbeat_interval_s=heartbeat_interval_s,
        stall_after_s=stall_after_s,
        started_at=started_at,
    )

    started_snapshot = monitor.snapshot()
    _client._emit_llm_call_lifecycle_event(
        call_id=call_id,
        phase="started",
        call_kind="text",
        caller="astream_llm",
        task=task,
        trace_id=trace_id,
        requested_model=model,
        provider_timeout_s=provider_timeout_s,
        prompt_ref=prompt_ref,
        heartbeat_interval_s=heartbeat_interval_s,
        stall_after_s=stall_after_s,
        progress_observable=started_snapshot.progress_observable,
        progress_source=started_snapshot.progress_source,
        progress_event_count=started_snapshot.progress_event_count,
    )
    monitor.start()

    async def _execute_stream_model(model_idx: int, current_model: str) -> AsyncLLMStream:
        current_api_base = _client._resolve_api_base_for_model(current_model, api_base, cfg)
        call_kwargs = _client._prepare_call_kwargs(
            current_model,
            messages,
            timeout=timeout,
            num_retries=0,
            reasoning_effort=reasoning_effort,
            api_base=current_api_base,
            kwargs=runtime_kwargs,
            warning_sink=_warnings,
        )
        call_kwargs["stream"] = True

        if hooks and hooks.before_call:
            hooks.before_call(current_model, messages, public_kwargs)

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
                    prompt_ref=prompt_ref,
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
                    user_kwargs=public_kwargs,
                    warning_sink=_warnings,
                    on_retry=r.on_retry,
                    caller="astream_llm",
                ),
            ),
        )

    try:
        if _client._is_agent_model(model):
            from llm_client.sdk.agents import _route_astream

            stream = await _route_astream(
                model,
                messages,
                hooks=hooks,
                timeout=timeout,
                **runtime_kwargs,
            )
        else:
            stream = cast(
                "AsyncLLMStream",
                await _client.run_async_with_fallback(
                    models=models,
                    execute_model=_execute_stream_model,
                    on_fallback=on_fallback,
                    warning_sink=_warnings,
                    logger=_client.logger,
                ),
            )

        return _AsyncStreamLifecycleAdapter(
            stream=stream,
            call_id=call_id,
            requested_model=model,
            resolved_model=(
                getattr(stream, "_resolved_model", None)
                if hasattr(stream, "_resolved_model")
                else getattr(stream, "resolved_model", None)
            ),
            monitor=monitor,
            caller="astream_llm",
            task=task,
            trace_id=trace_id,
            provider_timeout_s=provider_timeout_s,
            prompt_ref=prompt_ref,
            heartbeat_interval_s=heartbeat_interval_s,
            stall_after_s=stall_after_s,
            started_at=started_at,
        )
    except Exception as exc:
        _emit_stream_terminal_event(
            stream=None,
            error=exc,
            call_id=call_id,
            caller="astream_llm",
            task=task,
            trace_id=trace_id,
            requested_model=model,
            resolved_model=model,
            provider_timeout_s=provider_timeout_s,
            prompt_ref=prompt_ref,
            heartbeat_interval_s=heartbeat_interval_s,
            stall_after_s=stall_after_s,
            started_at=started_at,
            monitor=monitor,
        )
        raise _client.wrap_error(exc) from exc
