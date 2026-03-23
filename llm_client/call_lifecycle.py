"""Lifecycle observability helpers for public LLM call entrypoints.

This module owns the progress, heartbeat, and stall-monitoring machinery used
by the public call wrappers in ``llm_client.client`` and the streaming runtime.
Keeping the implementation here lets ``client.py`` remain the public facade
while the lifecycle-specific state, event emission, and monitor logic live in a
single observability-local boundary.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from llm_client import io_log as _io_log
from llm_client.errors import LLMError
from llm_client.foundation import (
    coerce_run_id as _coerce_foundation_run_id,
    new_event_id as _new_foundation_event_id,
    now_iso as _foundation_now_iso,
)
from llm_client.timeout_policy import timeout_policy_label as _timeout_policy_label

logger = logging.getLogger(__name__)

_LLM_CALL_RUNTIME_ACTOR_ID = "service:llm_client:call_runtime:1"
_LIFECYCLE_HEARTBEAT_INTERVAL_ENV = "LLM_CLIENT_LIFECYCLE_HEARTBEAT_INTERVAL_S"
_LIFECYCLE_STALL_AFTER_ENV = "LLM_CLIENT_LIFECYCLE_STALL_AFTER_S"
_DEFAULT_LIFECYCLE_HEARTBEAT_INTERVAL_S = 15.0
_DEFAULT_LIFECYCLE_STALL_AFTER_S = 300.0


def _process_host_name() -> str | None:
    """Return the current host name for same-host lifecycle correlation."""

    try:
        hostname = socket.gethostname().strip()
    except Exception:
        return None
    return hostname or None


def _linux_process_start_token(pid: int) -> str | None:
    """Return a Linux procfs start token for one process when available."""

    if pid <= 0:
        return None
    try:
        stat_text = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
    except OSError:
        return None
    _, _, remainder = stat_text.partition(") ")
    if not remainder:
        return None
    fields = remainder.split()
    if len(fields) <= 19:
        return None
    start_ticks = fields[19].strip()
    return f"linux-proc-start:{start_ticks}" if start_ticks else None


_PROCESS_HOST_NAME = _process_host_name()
_PROCESS_ID = os.getpid()
_PROCESS_START_TOKEN = _linux_process_start_token(_PROCESS_ID)


@dataclass(frozen=True)
class _LLMCallProgressSnapshot:
    """Capture truthful progress-observability state for one in-flight call."""

    progress_observable: bool
    progress_source: str | None
    progress_event_count: int
    last_progress_at_monotonic: float | None


class _LLMCallProgressReporter(Protocol):
    """Minimal contract for lifecycles that can observe real call progress."""

    def enable_progress_tracking(self, *, default_source: str | None = None) -> None:
        """Declare that the call path exposes truthful progress signals."""

    def mark_progress(self, *, source: str) -> None:
        """Record one observed unit of forward progress for the in-flight call."""


def _new_llm_call_lifecycle_id() -> str:
    """Return a stable correlation id shared across lifecycle events for one call."""

    return f"llmcall_{uuid.uuid4().hex}"


def _llm_call_lifecycle_session_id(call_id: str) -> str:
    """Derive a deterministic Foundation session id for one public call lifecycle."""

    suffix = call_id.removeprefix("llmcall_")
    return f"sess_{suffix}"


def _llm_lifecycle_error_message(error: Exception) -> str:
    """Return a non-empty error message for lifecycle failure records."""

    if isinstance(error, LLMError) and error.original is not None:
        error = error.original
    message = str(error).strip()
    if message:
        return message
    return error.__class__.__name__


def _llm_lifecycle_error_type(error: Exception) -> str:
    """Return the most informative error type for lifecycle failure records."""

    if isinstance(error, LLMError) and error.original is not None:
        return error.original.__class__.__name__
    return error.__class__.__name__


def _provider_timeout_for_lifecycle(timeout: Any) -> int:
    """Compute the effective provider-timeout value for lifecycle observability."""

    if _timeout_policy_label() == "ban":
        return 0
    try:
        parsed = int(timeout)
    except (TypeError, ValueError):
        return 0
    return max(parsed, 0)


def _normalize_lifecycle_seconds(value: Any, *, default: float) -> float:
    """Normalize lifecycle heartbeat/stall thresholds from caller or env values."""

    if value is None:
        return max(default, 0.0)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if parsed <= 0:
        return 0.0
    return parsed


def _resolve_lifecycle_monitoring_settings(
    *,
    heartbeat_interval: Any,
    stall_after: Any,
) -> tuple[float, float]:
    """Resolve public liveness settings without forwarding them to providers."""

    heartbeat_default = _normalize_lifecycle_seconds(
        os.environ.get(
            _LIFECYCLE_HEARTBEAT_INTERVAL_ENV,
            _DEFAULT_LIFECYCLE_HEARTBEAT_INTERVAL_S,
        ),
        default=_DEFAULT_LIFECYCLE_HEARTBEAT_INTERVAL_S,
    )
    stall_default = _normalize_lifecycle_seconds(
        os.environ.get(_LIFECYCLE_STALL_AFTER_ENV, _DEFAULT_LIFECYCLE_STALL_AFTER_S),
        default=_DEFAULT_LIFECYCLE_STALL_AFTER_S,
    )
    resolved_heartbeat = _normalize_lifecycle_seconds(
        heartbeat_interval,
        default=heartbeat_default,
    )
    resolved_stall = _normalize_lifecycle_seconds(
        stall_after,
        default=stall_default,
    )
    return resolved_heartbeat, resolved_stall


def _emit_llm_call_lifecycle_event(
    *,
    call_id: str,
    phase: Literal["started", "heartbeat", "progress", "stalled", "completed", "failed"],
    call_kind: Literal["text", "structured"],
    caller: str,
    task: str,
    trace_id: str,
    requested_model: str,
    provider_timeout_s: int,
    prompt_ref: str | None,
    resolved_model: str | None = None,
    latency_s: float | None = None,
    elapsed_s: float | None = None,
    heartbeat_interval_s: float | None = None,
    stall_after_s: float | None = None,
    progress_observable: bool | None = None,
    progress_source: str | None = None,
    progress_event_count: int | None = None,
    error: Exception | None = None,
) -> None:
    """Emit a Foundation-backed lifecycle event for one public LLM call."""

    params: dict[str, Any] = {
        "task": task,
        "trace_id": trace_id,
        "call_kind": call_kind,
    }
    if prompt_ref is not None:
        params["prompt_ref"] = prompt_ref
    if resolved_model is not None:
        params["resolved_model"] = resolved_model

    _io_log.log_foundation_event(
        event={
            "event_id": _new_foundation_event_id(),
            "event_type": "LLMCallLifecycle",
            "timestamp": _foundation_now_iso(),
            "run_id": _coerce_foundation_run_id(None, trace_id),
            "session_id": _llm_call_lifecycle_session_id(call_id),
            "actor_id": _LLM_CALL_RUNTIME_ACTOR_ID,
            "operation": {"name": caller, "version": None},
            "inputs": {
                "artifact_ids": [],
                "params": params,
                "bindings": {},
            },
            "outputs": {
                "artifact_ids": [],
                "payload_hashes": [],
            },
            "llm_call_lifecycle": {
                "call_id": call_id,
                "phase": phase,
                "call_kind": call_kind,
                "requested_model_id": requested_model,
                "resolved_model_id": resolved_model,
                "provider_timeout_s": provider_timeout_s if provider_timeout_s > 0 else None,
                "timeout_policy": _timeout_policy_label(),
                "prompt_ref": prompt_ref,
                "host_name": _PROCESS_HOST_NAME,
                "process_id": _PROCESS_ID if _PROCESS_ID > 0 else None,
                "process_start_token": _PROCESS_START_TOKEN,
                "progress_observable": progress_observable,
                "progress_source": progress_source,
                "progress_event_count": progress_event_count,
                "elapsed_s": elapsed_s,
                "latency_s": latency_s,
                "heartbeat_interval_s": (
                    heartbeat_interval_s
                    if heartbeat_interval_s and heartbeat_interval_s > 0
                    else None
                ),
                "stall_after_s": stall_after_s if stall_after_s and stall_after_s > 0 else None,
                "error_type": _llm_lifecycle_error_type(error) if error is not None else None,
                "error_message": _llm_lifecycle_error_message(error) if error is not None else None,
            },
        },
        caller=caller,
        task=task,
        trace_id=trace_id,
    )


class _SyncLLMCallHeartbeatMonitor:
    """Emit lifecycle updates for one sync call, including real progress when available."""

    def __init__(
        self,
        *,
        call_id: str,
        call_kind: Literal["text", "structured"],
        caller: str,
        task: str,
        trace_id: str,
        requested_model: str,
        provider_timeout_s: int,
        prompt_ref: str | None,
        heartbeat_interval_s: float,
        stall_after_s: float,
        started_at: float,
        progress_observable: bool = False,
    ) -> None:
        self.call_id = call_id
        self.call_kind = call_kind
        self.caller = caller
        self.task = task
        self.trace_id = trace_id
        self.requested_model = requested_model
        self.provider_timeout_s = provider_timeout_s
        self.prompt_ref = prompt_ref
        self.heartbeat_interval_s = heartbeat_interval_s
        self.stall_after_s = stall_after_s
        self.started_at = started_at
        self._state_lock = threading.Lock()
        self._progress_observable = progress_observable
        self._progress_source: str | None = None
        self._progress_event_count = 0
        self._last_progress_at_monotonic: float | None = None
        self._stalled_emitted = False
        self._last_progress_event_emitted_at: float | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background liveness monitor when thresholds are enabled."""

        if self.heartbeat_interval_s <= 0 and self.stall_after_s <= 0:
            return
        self._thread = threading.Thread(
            target=self._run,
            name=f"llm-call-heartbeat-{self.call_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the monitor and wait briefly for clean exit."""

        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=1.0)

    def enable_progress_tracking(self, *, default_source: str | None = None) -> None:
        """Declare that this call path exposes truthful observable progress."""

        with self._state_lock:
            self._progress_observable = True
            if default_source:
                self._progress_source = default_source

    def mark_progress(self, *, source: str) -> None:
        """Record one unit of observed progress and rate-limit event emission."""

        now = time.monotonic()
        with self._state_lock:
            self._progress_observable = True
            self._progress_source = source
            self._progress_event_count += 1
            self._last_progress_at_monotonic = now
            self._stalled_emitted = False
            should_emit = (
                self._last_progress_event_emitted_at is None
                or self.heartbeat_interval_s <= 0
                or (now - self._last_progress_event_emitted_at) >= self.heartbeat_interval_s
            )
            if should_emit:
                self._last_progress_event_emitted_at = now
            snapshot = self._snapshot_locked()
        if should_emit:
            self._emit_phase("progress", elapsed_s=now - self.started_at, snapshot=snapshot)

    def snapshot(self) -> _LLMCallProgressSnapshot:
        """Return the current progress-observability state for this call."""

        with self._state_lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> _LLMCallProgressSnapshot:
        """Build a progress snapshot while the internal state lock is held."""

        return _LLMCallProgressSnapshot(
            progress_observable=self._progress_observable,
            progress_source=self._progress_source,
            progress_event_count=self._progress_event_count,
            last_progress_at_monotonic=self._last_progress_at_monotonic,
        )

    def _emit_phase(
        self,
        phase: Literal["heartbeat", "progress", "stalled"],
        *,
        elapsed_s: float,
        snapshot: _LLMCallProgressSnapshot | None = None,
    ) -> None:
        """Emit one non-terminal lifecycle phase with the latest progress metadata."""

        effective_snapshot = snapshot or self.snapshot()
        _emit_llm_call_lifecycle_event(
            call_id=self.call_id,
            phase=phase,
            call_kind=self.call_kind,
            caller=self.caller,
            task=self.task,
            trace_id=self.trace_id,
            requested_model=self.requested_model,
            provider_timeout_s=self.provider_timeout_s,
            prompt_ref=self.prompt_ref,
            elapsed_s=elapsed_s,
            heartbeat_interval_s=self.heartbeat_interval_s,
            stall_after_s=self.stall_after_s,
            progress_observable=effective_snapshot.progress_observable,
            progress_source=effective_snapshot.progress_source,
            progress_event_count=effective_snapshot.progress_event_count,
        )

    def _next_stall_wait(
        self,
        *,
        now: float,
        snapshot: _LLMCallProgressSnapshot,
    ) -> float | None:
        """Return seconds until the next eligible stall event, if any."""

        if self.stall_after_s <= 0:
            return None
        with self._state_lock:
            stalled_emitted = self._stalled_emitted
        if stalled_emitted:
            return None
        if snapshot.progress_observable:
            last_progress = snapshot.last_progress_at_monotonic
            if last_progress is None:
                return None
            idle_for_s = now - last_progress
            return max(self.stall_after_s - idle_for_s, 0.001)
        elapsed_s = now - self.started_at
        return max(self.stall_after_s - elapsed_s, 0.001)

    def _should_emit_stalled(
        self,
        *,
        now: float,
        snapshot: _LLMCallProgressSnapshot,
    ) -> bool:
        """Return whether the current call state has crossed the stall threshold."""

        if self.stall_after_s <= 0:
            return False
        with self._state_lock:
            if self._stalled_emitted:
                return False
        if snapshot.progress_observable:
            last_progress = snapshot.last_progress_at_monotonic
            if last_progress is None:
                return False
            return (now - last_progress) >= self.stall_after_s
        return (now - self.started_at) >= self.stall_after_s

    def _mark_stalled_emitted(self) -> None:
        """Remember that the current idle period already emitted a stall marker."""

        with self._state_lock:
            self._stalled_emitted = True

    def _run(self) -> None:
        """Emit in-flight lifecycle markers until the wrapped call terminates."""

        next_heartbeat = self.heartbeat_interval_s if self.heartbeat_interval_s > 0 else None
        while True:
            now = time.monotonic()
            elapsed = now - self.started_at
            snapshot = self.snapshot()
            waits: list[float] = []
            if next_heartbeat is not None:
                waits.append(max(next_heartbeat - elapsed, 0.001))
            stall_wait = self._next_stall_wait(now=now, snapshot=snapshot)
            if stall_wait is not None:
                waits.append(stall_wait)
            if not waits:
                return
            if self._stop_event.wait(min(waits)):
                return
            now = time.monotonic()
            elapsed = now - self.started_at
            snapshot = self.snapshot()
            if next_heartbeat is not None and elapsed >= next_heartbeat:
                self._emit_phase("heartbeat", elapsed_s=elapsed, snapshot=snapshot)
                while next_heartbeat is not None and elapsed >= next_heartbeat:
                    next_heartbeat += self.heartbeat_interval_s
            if self._should_emit_stalled(now=now, snapshot=snapshot):
                self._emit_phase("stalled", elapsed_s=elapsed, snapshot=snapshot)
                self._mark_stalled_emitted()


class _AsyncLLMCallHeartbeatMonitor:
    """Emit lifecycle updates for one async call, including real progress when available."""

    def __init__(
        self,
        *,
        call_id: str,
        call_kind: Literal["text", "structured"],
        caller: str,
        task: str,
        trace_id: str,
        requested_model: str,
        provider_timeout_s: int,
        prompt_ref: str | None,
        heartbeat_interval_s: float,
        stall_after_s: float,
        started_at: float,
        progress_observable: bool = False,
    ) -> None:
        self.call_id = call_id
        self.call_kind = call_kind
        self.caller = caller
        self.task = task
        self.trace_id = trace_id
        self.requested_model = requested_model
        self.provider_timeout_s = provider_timeout_s
        self.prompt_ref = prompt_ref
        self.heartbeat_interval_s = heartbeat_interval_s
        self.stall_after_s = stall_after_s
        self.started_at = started_at
        self._state_lock = threading.Lock()
        self._progress_observable = progress_observable
        self._progress_source: str | None = None
        self._progress_event_count = 0
        self._last_progress_at_monotonic: float | None = None
        self._stalled_emitted = False
        self._last_progress_event_emitted_at: float | None = None
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Start the async heartbeat task when thresholds are enabled."""

        if self.heartbeat_interval_s <= 0 and self.stall_after_s <= 0:
            return
        self._task = asyncio.create_task(
            self._run(),
            name=f"llm-call-heartbeat-{self.call_id}",
        )

    async def stop(self) -> None:
        """Stop the heartbeat task without letting monitor failures clobber the call.

        Lifecycle heartbeats are observability aids, not the source of truth for
        call success. If the monitor task itself times out or tears down
        noisily during shutdown, that failure must be logged loudly but must not
        replace the real LLM result or real model error that triggered stop.
        """

        if self._task is None:
            return
        task = self._task
        self._task = None
        self._stop_event.set()
        try:
            await task
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Async heartbeat monitor stop ignored monitor failure for call %s: %r",
                self.call_id,
                exc,
            )

    def enable_progress_tracking(self, *, default_source: str | None = None) -> None:
        """Declare that this async call path exposes truthful observable progress."""

        with self._state_lock:
            self._progress_observable = True
            if default_source:
                self._progress_source = default_source

    def mark_progress(self, *, source: str) -> None:
        """Record one unit of observed progress and rate-limit event emission."""

        now = time.monotonic()
        with self._state_lock:
            self._progress_observable = True
            self._progress_source = source
            self._progress_event_count += 1
            self._last_progress_at_monotonic = now
            self._stalled_emitted = False
            should_emit = (
                self._last_progress_event_emitted_at is None
                or self.heartbeat_interval_s <= 0
                or (now - self._last_progress_event_emitted_at) >= self.heartbeat_interval_s
            )
            if should_emit:
                self._last_progress_event_emitted_at = now
            snapshot = self._snapshot_locked()
        if should_emit:
            self._emit_phase("progress", elapsed_s=now - self.started_at, snapshot=snapshot)

    def snapshot(self) -> _LLMCallProgressSnapshot:
        """Return the current progress-observability state for this call."""

        with self._state_lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> _LLMCallProgressSnapshot:
        """Build a progress snapshot while the internal state lock is held."""

        return _LLMCallProgressSnapshot(
            progress_observable=self._progress_observable,
            progress_source=self._progress_source,
            progress_event_count=self._progress_event_count,
            last_progress_at_monotonic=self._last_progress_at_monotonic,
        )

    def _emit_phase(
        self,
        phase: Literal["heartbeat", "progress", "stalled"],
        *,
        elapsed_s: float,
        snapshot: _LLMCallProgressSnapshot | None = None,
    ) -> None:
        """Emit one non-terminal lifecycle phase with the latest progress metadata."""

        effective_snapshot = snapshot or self.snapshot()
        _emit_llm_call_lifecycle_event(
            call_id=self.call_id,
            phase=phase,
            call_kind=self.call_kind,
            caller=self.caller,
            task=self.task,
            trace_id=self.trace_id,
            requested_model=self.requested_model,
            provider_timeout_s=self.provider_timeout_s,
            prompt_ref=self.prompt_ref,
            elapsed_s=elapsed_s,
            heartbeat_interval_s=self.heartbeat_interval_s,
            stall_after_s=self.stall_after_s,
            progress_observable=effective_snapshot.progress_observable,
            progress_source=effective_snapshot.progress_source,
            progress_event_count=effective_snapshot.progress_event_count,
        )

    def _next_stall_wait(
        self,
        *,
        now: float,
        snapshot: _LLMCallProgressSnapshot,
    ) -> float | None:
        """Return seconds until the next eligible stall event, if any."""

        if self.stall_after_s <= 0:
            return None
        with self._state_lock:
            stalled_emitted = self._stalled_emitted
        if stalled_emitted:
            return None
        if snapshot.progress_observable:
            last_progress = snapshot.last_progress_at_monotonic
            if last_progress is None:
                return None
            idle_for_s = now - last_progress
            return max(self.stall_after_s - idle_for_s, 0.001)
        elapsed_s = now - self.started_at
        return max(self.stall_after_s - elapsed_s, 0.001)

    def _should_emit_stalled(
        self,
        *,
        now: float,
        snapshot: _LLMCallProgressSnapshot,
    ) -> bool:
        """Return whether the current call state has crossed the stall threshold."""

        if self.stall_after_s <= 0:
            return False
        with self._state_lock:
            if self._stalled_emitted:
                return False
        if snapshot.progress_observable:
            last_progress = snapshot.last_progress_at_monotonic
            if last_progress is None:
                return False
            return (now - last_progress) >= self.stall_after_s
        return (now - self.started_at) >= self.stall_after_s

    def _mark_stalled_emitted(self) -> None:
        """Remember that the current idle period already emitted a stall marker."""

        with self._state_lock:
            self._stalled_emitted = True

    async def _run(self) -> None:
        """Emit in-flight lifecycle markers until the wrapped async call terminates."""

        next_heartbeat = self.heartbeat_interval_s if self.heartbeat_interval_s > 0 else None
        while True:
            now = time.monotonic()
            elapsed = now - self.started_at
            snapshot = self.snapshot()
            waits: list[float] = []
            if next_heartbeat is not None:
                waits.append(max(next_heartbeat - elapsed, 0.001))
            stall_wait = self._next_stall_wait(now=now, snapshot=snapshot)
            if stall_wait is not None:
                waits.append(stall_wait)
            if not waits:
                return
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=min(waits))
                return
            except (TimeoutError, asyncio.TimeoutError):
                pass
            now = time.monotonic()
            elapsed = now - self.started_at
            snapshot = self.snapshot()
            if next_heartbeat is not None and elapsed >= next_heartbeat:
                self._emit_phase("heartbeat", elapsed_s=elapsed, snapshot=snapshot)
                while next_heartbeat is not None and elapsed >= next_heartbeat:
                    next_heartbeat += self.heartbeat_interval_s
            if self._should_emit_stalled(now=now, snapshot=snapshot):
                self._emit_phase("stalled", elapsed_s=elapsed, snapshot=snapshot)
                self._mark_stalled_emitted()
