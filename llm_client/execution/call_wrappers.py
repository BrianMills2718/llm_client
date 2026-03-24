"""Shared public-call wrapper scaffolding for ``llm_client.client``.

This module owns the repeated envelope around the public text and structured
call entrypoints: tag normalization, budget enforcement, lifecycle monitor
setup, terminal lifecycle emission, and runtime-kwargs preparation. The public
signatures remain in ``client.py`` while this module centralizes the repeated
wrapper mechanics that were previously copied across four entrypoints.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, TypeVar

from llm_client.execution.call_contracts import (
    check_budget as _check_budget,
    normalize_prompt_ref as _normalize_prompt_ref,
    require_tags as _require_tags,
)
from llm_client.execution.call_lifecycle import (
    _AsyncLLMCallHeartbeatMonitor,
    _SyncLLMCallHeartbeatMonitor,
    _emit_llm_call_lifecycle_event,
    _new_llm_call_lifecycle_id,
    _provider_timeout_for_lifecycle,
    _resolve_lifecycle_monitoring_settings,
)

T = TypeVar("T")


@dataclass(frozen=True)
class PreparedPublicCallEnvelope:
    """Resolved public-call envelope before runtime dispatch begins."""

    normalized_prompt_ref: str | None
    resolved_task: str
    resolved_trace_id: str
    resolved_max_budget: float
    effective_provider_timeout: int
    heartbeat_interval_s: float
    stall_after_s: float
    runtime_kwargs: dict[str, Any]


def _prepare_public_call_envelope(
    *,
    caller: str,
    timeout: int,
    kwargs: dict[str, Any],
) -> PreparedPublicCallEnvelope:
    """Resolve call tags, lifecycle settings, and provider-safe runtime kwargs."""

    normalized_prompt_ref = _normalize_prompt_ref(kwargs.get("prompt_ref"))
    resolved_task, resolved_trace_id, resolved_max_budget, _ = _require_tags(
        kwargs.get("task"),
        kwargs.get("trace_id"),
        kwargs.get("max_budget"),
        caller=caller,
    )
    _check_budget(resolved_trace_id, resolved_max_budget)
    effective_provider_timeout = _provider_timeout_for_lifecycle(timeout)

    runtime_kwargs = dict(kwargs)
    heartbeat_interval_s, stall_after_s = _resolve_lifecycle_monitoring_settings(
        heartbeat_interval=runtime_kwargs.pop("lifecycle_heartbeat_interval_s", None),
        stall_after=runtime_kwargs.pop("lifecycle_stall_after_s", None),
    )
    runtime_kwargs["task"] = resolved_task
    runtime_kwargs["trace_id"] = resolved_trace_id
    runtime_kwargs["max_budget"] = resolved_max_budget
    runtime_kwargs["prompt_ref"] = normalized_prompt_ref

    return PreparedPublicCallEnvelope(
        normalized_prompt_ref=normalized_prompt_ref,
        resolved_task=resolved_task,
        resolved_trace_id=resolved_trace_id,
        resolved_max_budget=resolved_max_budget,
        effective_provider_timeout=effective_provider_timeout,
        heartbeat_interval_s=heartbeat_interval_s,
        stall_after_s=stall_after_s,
        runtime_kwargs=runtime_kwargs,
    )


def _run_sync_public_call(
    *,
    model: str,
    call_kind: Literal["text", "structured"],
    caller: str,
    timeout: int,
    envelope: PreparedPublicCallEnvelope,
    invoke: Callable[[dict[str, Any]], T],
    resolve_model: Callable[[T], str | None],
) -> T:
    """Run one sync public-call wrapper with lifecycle monitoring and events."""

    call_id = _new_llm_call_lifecycle_id()
    started_at = time.monotonic()
    monitor = _SyncLLMCallHeartbeatMonitor(
        call_id=call_id,
        call_kind=call_kind,
        caller=caller,
        task=envelope.resolved_task,
        trace_id=envelope.resolved_trace_id,
        requested_model=model,
        provider_timeout_s=envelope.effective_provider_timeout,
        prompt_ref=envelope.normalized_prompt_ref,
        heartbeat_interval_s=envelope.heartbeat_interval_s,
        stall_after_s=envelope.stall_after_s,
        started_at=started_at,
    )
    started_snapshot = monitor.snapshot()
    _emit_llm_call_lifecycle_event(
        call_id=call_id,
        phase="started",
        call_kind=call_kind,
        caller=caller,
        task=envelope.resolved_task,
        trace_id=envelope.resolved_trace_id,
        requested_model=model,
        provider_timeout_s=envelope.effective_provider_timeout,
        prompt_ref=envelope.normalized_prompt_ref,
        heartbeat_interval_s=envelope.heartbeat_interval_s,
        stall_after_s=envelope.stall_after_s,
        progress_observable=started_snapshot.progress_observable,
        progress_source=started_snapshot.progress_source,
        progress_event_count=started_snapshot.progress_event_count,
    )
    runtime_kwargs = dict(envelope.runtime_kwargs)
    runtime_kwargs["_lifecycle_monitor"] = monitor
    monitor.start()
    try:
        result = invoke(runtime_kwargs)
    except Exception as exc:
        monitor.stop()
        snapshot = monitor.snapshot()
        _emit_llm_call_lifecycle_event(
            call_id=call_id,
            phase="failed",
            call_kind=call_kind,
            caller=caller,
            task=envelope.resolved_task,
            trace_id=envelope.resolved_trace_id,
            requested_model=model,
            provider_timeout_s=envelope.effective_provider_timeout,
            prompt_ref=envelope.normalized_prompt_ref,
            latency_s=time.monotonic() - started_at,
            error=exc,
            heartbeat_interval_s=envelope.heartbeat_interval_s,
            stall_after_s=envelope.stall_after_s,
            progress_observable=snapshot.progress_observable,
            progress_source=snapshot.progress_source,
            progress_event_count=snapshot.progress_event_count,
        )
        raise
    monitor.stop()
    elapsed_s = time.monotonic() - started_at
    completed_snapshot = monitor.snapshot()
    _emit_llm_call_lifecycle_event(
        call_id=call_id,
        phase="completed",
        call_kind=call_kind,
        caller=caller,
        task=envelope.resolved_task,
        trace_id=envelope.resolved_trace_id,
        requested_model=model,
        provider_timeout_s=envelope.effective_provider_timeout,
        prompt_ref=envelope.normalized_prompt_ref,
        resolved_model=resolve_model(result),
        elapsed_s=elapsed_s,
        latency_s=elapsed_s,
        heartbeat_interval_s=envelope.heartbeat_interval_s,
        stall_after_s=envelope.stall_after_s,
        progress_observable=completed_snapshot.progress_observable,
        progress_source=completed_snapshot.progress_source,
        progress_event_count=completed_snapshot.progress_event_count,
    )
    return result


async def _run_async_public_call(
    *,
    model: str,
    call_kind: Literal["text", "structured"],
    caller: str,
    timeout: int,
    envelope: PreparedPublicCallEnvelope,
    invoke: Callable[[dict[str, Any]], Awaitable[T]],
    resolve_model: Callable[[T], str | None],
) -> T:
    """Run one async public-call wrapper with lifecycle monitoring and events."""

    call_id = _new_llm_call_lifecycle_id()
    started_at = time.monotonic()
    monitor = _AsyncLLMCallHeartbeatMonitor(
        call_id=call_id,
        call_kind=call_kind,
        caller=caller,
        task=envelope.resolved_task,
        trace_id=envelope.resolved_trace_id,
        requested_model=model,
        provider_timeout_s=envelope.effective_provider_timeout,
        prompt_ref=envelope.normalized_prompt_ref,
        heartbeat_interval_s=envelope.heartbeat_interval_s,
        stall_after_s=envelope.stall_after_s,
        started_at=started_at,
    )
    started_snapshot = monitor.snapshot()
    _emit_llm_call_lifecycle_event(
        call_id=call_id,
        phase="started",
        call_kind=call_kind,
        caller=caller,
        task=envelope.resolved_task,
        trace_id=envelope.resolved_trace_id,
        requested_model=model,
        provider_timeout_s=envelope.effective_provider_timeout,
        prompt_ref=envelope.normalized_prompt_ref,
        heartbeat_interval_s=envelope.heartbeat_interval_s,
        stall_after_s=envelope.stall_after_s,
        progress_observable=started_snapshot.progress_observable,
        progress_source=started_snapshot.progress_source,
        progress_event_count=started_snapshot.progress_event_count,
    )
    runtime_kwargs = dict(envelope.runtime_kwargs)
    runtime_kwargs["_lifecycle_monitor"] = monitor
    monitor.start()
    try:
        result = await invoke(runtime_kwargs)
    except Exception as exc:
        await monitor.stop()
        snapshot = monitor.snapshot()
        _emit_llm_call_lifecycle_event(
            call_id=call_id,
            phase="failed",
            call_kind=call_kind,
            caller=caller,
            task=envelope.resolved_task,
            trace_id=envelope.resolved_trace_id,
            requested_model=model,
            provider_timeout_s=envelope.effective_provider_timeout,
            prompt_ref=envelope.normalized_prompt_ref,
            latency_s=time.monotonic() - started_at,
            error=exc,
            heartbeat_interval_s=envelope.heartbeat_interval_s,
            stall_after_s=envelope.stall_after_s,
            progress_observable=snapshot.progress_observable,
            progress_source=snapshot.progress_source,
            progress_event_count=snapshot.progress_event_count,
        )
        raise
    await monitor.stop()
    elapsed_s = time.monotonic() - started_at
    completed_snapshot = monitor.snapshot()
    _emit_llm_call_lifecycle_event(
        call_id=call_id,
        phase="completed",
        call_kind=call_kind,
        caller=caller,
        task=envelope.resolved_task,
        trace_id=envelope.resolved_trace_id,
        requested_model=model,
        provider_timeout_s=envelope.effective_provider_timeout,
        prompt_ref=envelope.normalized_prompt_ref,
        resolved_model=resolve_model(result),
        elapsed_s=elapsed_s,
        latency_s=elapsed_s,
        heartbeat_interval_s=envelope.heartbeat_interval_s,
        stall_after_s=envelope.stall_after_s,
        progress_observable=completed_snapshot.progress_observable,
        progress_source=completed_snapshot.progress_source,
        progress_event_count=completed_snapshot.progress_event_count,
    )
    return result
