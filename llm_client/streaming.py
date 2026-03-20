"""Streaming wrapper classes for LLM responses.

Houses LLMStream (sync) and AsyncLLMStream (async), which yield text
chunks as they arrive and expose the accumulated .result after the
stream is fully consumed.

These classes reference cost/usage helpers and finalization functions from
client.py at runtime to avoid circular imports during module extraction.
"""

from __future__ import annotations

import time
from typing import Any, Literal

import litellm

from llm_client.data_types import LLMCallResult
from llm_client.retry import Hooks


class LLMStream:
    """Sync streaming wrapper. Yields text chunks, then exposes ``.result``.

    Example::

        stream = stream_llm("gpt-4o", messages)
        for chunk in stream:
            print(chunk, end="", flush=True)
        print()
        print(stream.result.usage)
    """

    def __init__(
        self, response_iter: Any, model: str, hooks: Hooks | None = None,
        messages: list[dict[str, Any]] | None = None,
        task: str | None = None,
        trace_id: str | None = None,
        prompt_ref: str | None = None,
        warnings: list[str] | None = None,
        requested_model: str | None = None,
        resolved_model: str | None = None,
        routing_trace: dict[str, Any] | None = None,
        lifecycle_call_id: str | None = None,
        lifecycle_caller: str = "stream_llm",
        provider_timeout_s: int = 0,
        heartbeat_interval_s: float = 0.0,
        stall_after_s: float = 0.0,
    ) -> None:
        self._iter = response_iter
        self._model = model
        self._hooks = hooks
        self._messages = messages
        self._task = task
        self._trace_id = trace_id
        self._prompt_ref = prompt_ref
        self._warnings = warnings or []
        self._requested_model = requested_model
        self._resolved_model = resolved_model
        self._routing_trace = routing_trace
        self._t0 = time.monotonic()
        self._chunks_text: list[str] = []
        self._raw_chunks: list[Any] = []
        self._result: LLMCallResult | None = None
        self._lifecycle_call_id = lifecycle_call_id
        self._lifecycle_caller = lifecycle_caller
        self._provider_timeout_s = provider_timeout_s
        self._heartbeat_interval_s = heartbeat_interval_s
        self._stall_after_s = stall_after_s
        self._lifecycle_monitor: Any | None = None
        self._lifecycle_started_at: float | None = None
        self._lifecycle_terminal_emitted = False

    def __iter__(self) -> LLMStream:
        return self

    def __next__(self) -> str:
        self._ensure_lifecycle_started()
        try:
            chunk = next(self._iter)
        except StopIteration:
            self._finalize()
            raise
        except Exception as exc:
            self._finalize_failed(exc)
            raise
        self._raw_chunks.append(chunk)
        if self._lifecycle_monitor is not None:
            self._lifecycle_monitor.mark_progress(source="stream_chunk")
        text = ""
        if chunk.choices:
            delta = chunk.choices[0].delta
            text = (delta.content if delta and delta.content else "") or ""
        self._chunks_text.append(text)
        return text

    def _ensure_lifecycle_started(self) -> None:
        """Start lifecycle tracking lazily when the caller first consumes the stream."""
        if self._lifecycle_call_id is None or self._lifecycle_monitor is not None:
            return
        from llm_client import client as _client_mod

        _client: Any = _client_mod

        started_at = time.monotonic()
        self._lifecycle_started_at = started_at
        self._lifecycle_monitor = _client._SyncLLMCallHeartbeatMonitor(
            call_id=self._lifecycle_call_id,
            call_kind="text",
            caller=self._lifecycle_caller,
            task=self._task or "untagged",
            trace_id=self._trace_id or "untagged",
            requested_model=self._requested_model or self._model,
            provider_timeout_s=self._provider_timeout_s,
            prompt_ref=self._prompt_ref,
            heartbeat_interval_s=self._heartbeat_interval_s,
            stall_after_s=self._stall_after_s,
            started_at=started_at,
            progress_observable=True,
        )
        snapshot = self._lifecycle_monitor.snapshot()
        _client._emit_llm_call_lifecycle_event(
            call_id=self._lifecycle_call_id,
            phase="started",
            call_kind="text",
            caller=self._lifecycle_caller,
            task=self._task or "untagged",
            trace_id=self._trace_id or "untagged",
            requested_model=self._requested_model or self._model,
            provider_timeout_s=self._provider_timeout_s,
            prompt_ref=self._prompt_ref,
            heartbeat_interval_s=self._heartbeat_interval_s,
            stall_after_s=self._stall_after_s,
            progress_observable=snapshot.progress_observable,
            progress_source=snapshot.progress_source,
            progress_event_count=snapshot.progress_event_count,
        )
        self._lifecycle_monitor.start()

    def _emit_lifecycle_terminal(
        self,
        *,
        phase: Literal["completed", "failed"],
        error: Exception | None = None,
    ) -> None:
        """Emit one terminal lifecycle event and stop the in-flight monitor."""
        if self._lifecycle_call_id is None or self._lifecycle_terminal_emitted:
            return
        from llm_client import client as _client_mod

        _client: Any = _client_mod

        if self._lifecycle_monitor is not None:
            self._lifecycle_monitor.stop()
            snapshot = self._lifecycle_monitor.snapshot()
        else:
            snapshot = _client._LLMCallProgressSnapshot(
                progress_observable=True,
                progress_source=None,
                progress_event_count=0,
                last_progress_at_monotonic=None,
            )
        started_at = self._lifecycle_started_at or self._t0
        elapsed_s = time.monotonic() - started_at
        _client._emit_llm_call_lifecycle_event(
            call_id=self._lifecycle_call_id,
            phase=phase,
            call_kind="text",
            caller=self._lifecycle_caller,
            task=self._task or "untagged",
            trace_id=self._trace_id or "untagged",
            requested_model=self._requested_model or self._model,
            provider_timeout_s=self._provider_timeout_s,
            prompt_ref=self._prompt_ref,
            resolved_model=self._resolved_model or self._model if phase == "completed" else None,
            elapsed_s=elapsed_s,
            latency_s=elapsed_s,
            heartbeat_interval_s=self._heartbeat_interval_s,
            stall_after_s=self._stall_after_s,
            progress_observable=snapshot.progress_observable,
            progress_source=snapshot.progress_source,
            progress_event_count=snapshot.progress_event_count,
            error=error,
        )
        self._lifecycle_terminal_emitted = True

    def _finalize(self) -> None:
        if self._result is not None:
            return
        # Deferred import to avoid circular dependencies during module extraction.
        from llm_client import client as _client_mod

        _client: Any = _client_mod

        content = "".join(self._chunks_text)
        usage: dict[str, Any] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        cost = 0.0
        finish_reason = "stop"
        tool_calls: list[dict[str, Any]] = []
        try:
            complete = litellm.stream_chunk_builder(self._raw_chunks)
            if complete:
                usage = _client._extract_usage(complete)
                cost, cost_source = _client._parse_cost_result(_client._compute_cost(complete))
                first_choice = complete.choices[0]
                finish_reason = str(getattr(first_choice, "finish_reason", "") or "stop")
                message_obj = getattr(first_choice, "message", None)
                if message_obj is not None and getattr(message_obj, "tool_calls", None):
                    tool_calls = _client._extract_tool_calls(message_obj)
            else:
                cost_source = "unavailable"
        except Exception:
            cost_source = "unavailable"
        self._result = LLMCallResult(
            content=content,
            usage=usage,
            cost=cost,
            model=self._model,
            requested_model=self._requested_model,
            resolved_model=self._resolved_model or self._model,
            routing_trace=self._routing_trace,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            raw_response=self._raw_chunks[-1] if self._raw_chunks else None,
            warnings=self._warnings,
            cost_source=cost_source,
        )
        self._result = _client._finalize_result(
            self._result,
            requested_model=self._requested_model or self._model,
            resolved_model=self._resolved_model or self._model,
            routing_trace=self._routing_trace,
        )
        if self._hooks and self._hooks.after_call:
            self._hooks.after_call(self._result)
        _client._log_call_event(
            model=self._model,
            messages=self._messages,
            result=self._result,
            latency_s=time.monotonic() - self._t0,
            caller="stream_llm",
            task=self._task,
            trace_id=self._trace_id,
            prompt_ref=self._prompt_ref,
        )
        self._emit_lifecycle_terminal(phase="completed")

    def _finalize_failed(self, error: Exception) -> None:
        """Emit a terminal failure event if stream consumption raises mid-flight."""
        self._emit_lifecycle_terminal(phase="failed", error=error)

    @property
    def result(self) -> LLMCallResult:
        """The accumulated result. Available after the stream is fully consumed."""
        if self._result is None:
            raise RuntimeError("Stream not yet consumed. Iterate first.")
        return self._result


class AsyncLLMStream:
    """Async streaming wrapper. Yields text chunks, then exposes ``.result``.

    Example::

        stream = await astream_llm("gpt-4o", messages)
        async for chunk in stream:
            print(chunk, end="", flush=True)
        print()
        print(stream.result.usage)
    """

    def __init__(
        self, response_iter: Any, model: str, hooks: Hooks | None = None,
        messages: list[dict[str, Any]] | None = None,
        task: str | None = None,
        trace_id: str | None = None,
        prompt_ref: str | None = None,
        warnings: list[str] | None = None,
        requested_model: str | None = None,
        resolved_model: str | None = None,
        routing_trace: dict[str, Any] | None = None,
        lifecycle_call_id: str | None = None,
        lifecycle_caller: str = "astream_llm",
        provider_timeout_s: int = 0,
        heartbeat_interval_s: float = 0.0,
        stall_after_s: float = 0.0,
    ) -> None:
        self._iter = response_iter
        self._model = model
        self._hooks = hooks
        self._messages = messages
        self._task = task
        self._trace_id = trace_id
        self._prompt_ref = prompt_ref
        self._warnings = warnings or []
        self._requested_model = requested_model
        self._resolved_model = resolved_model
        self._routing_trace = routing_trace
        self._t0 = time.monotonic()
        self._chunks_text: list[str] = []
        self._raw_chunks: list[Any] = []
        self._result: LLMCallResult | None = None
        self._lifecycle_call_id = lifecycle_call_id
        self._lifecycle_caller = lifecycle_caller
        self._provider_timeout_s = provider_timeout_s
        self._heartbeat_interval_s = heartbeat_interval_s
        self._stall_after_s = stall_after_s
        self._lifecycle_monitor: Any | None = None
        self._lifecycle_started_at: float | None = None
        self._lifecycle_terminal_emitted = False

    def __aiter__(self) -> AsyncLLMStream:
        return self

    async def __anext__(self) -> str:
        await self._ensure_lifecycle_started()
        try:
            chunk = await self._iter.__anext__()
        except StopAsyncIteration:
            await self._finalize()
            raise
        except Exception as exc:
            await self._finalize_failed(exc)
            raise
        self._raw_chunks.append(chunk)
        if self._lifecycle_monitor is not None:
            self._lifecycle_monitor.mark_progress(source="stream_chunk")
        text = ""
        if chunk.choices:
            delta = chunk.choices[0].delta
            text = (delta.content if delta and delta.content else "") or ""
        self._chunks_text.append(text)
        return text

    async def _ensure_lifecycle_started(self) -> None:
        """Start lifecycle tracking lazily when the caller first consumes the stream."""
        if self._lifecycle_call_id is None or self._lifecycle_monitor is not None:
            return
        from llm_client import client as _client_mod

        _client: Any = _client_mod

        started_at = time.monotonic()
        self._lifecycle_started_at = started_at
        self._lifecycle_monitor = _client._AsyncLLMCallHeartbeatMonitor(
            call_id=self._lifecycle_call_id,
            call_kind="text",
            caller=self._lifecycle_caller,
            task=self._task or "untagged",
            trace_id=self._trace_id or "untagged",
            requested_model=self._requested_model or self._model,
            provider_timeout_s=self._provider_timeout_s,
            prompt_ref=self._prompt_ref,
            heartbeat_interval_s=self._heartbeat_interval_s,
            stall_after_s=self._stall_after_s,
            started_at=started_at,
            progress_observable=True,
        )
        snapshot = self._lifecycle_monitor.snapshot()
        _client._emit_llm_call_lifecycle_event(
            call_id=self._lifecycle_call_id,
            phase="started",
            call_kind="text",
            caller=self._lifecycle_caller,
            task=self._task or "untagged",
            trace_id=self._trace_id or "untagged",
            requested_model=self._requested_model or self._model,
            provider_timeout_s=self._provider_timeout_s,
            prompt_ref=self._prompt_ref,
            heartbeat_interval_s=self._heartbeat_interval_s,
            stall_after_s=self._stall_after_s,
            progress_observable=snapshot.progress_observable,
            progress_source=snapshot.progress_source,
            progress_event_count=snapshot.progress_event_count,
        )
        self._lifecycle_monitor.start()

    async def _emit_lifecycle_terminal(
        self,
        *,
        phase: Literal["completed", "failed"],
        error: Exception | None = None,
    ) -> None:
        """Emit one terminal lifecycle event and stop the in-flight monitor."""
        if self._lifecycle_call_id is None or self._lifecycle_terminal_emitted:
            return
        from llm_client import client as _client_mod

        _client: Any = _client_mod

        if self._lifecycle_monitor is not None:
            await self._lifecycle_monitor.stop()
            snapshot = self._lifecycle_monitor.snapshot()
        else:
            snapshot = _client._LLMCallProgressSnapshot(
                progress_observable=True,
                progress_source=None,
                progress_event_count=0,
                last_progress_at_monotonic=None,
            )
        started_at = self._lifecycle_started_at or self._t0
        elapsed_s = time.monotonic() - started_at
        _client._emit_llm_call_lifecycle_event(
            call_id=self._lifecycle_call_id,
            phase=phase,
            call_kind="text",
            caller=self._lifecycle_caller,
            task=self._task or "untagged",
            trace_id=self._trace_id or "untagged",
            requested_model=self._requested_model or self._model,
            provider_timeout_s=self._provider_timeout_s,
            prompt_ref=self._prompt_ref,
            resolved_model=self._resolved_model or self._model if phase == "completed" else None,
            elapsed_s=elapsed_s,
            latency_s=elapsed_s,
            heartbeat_interval_s=self._heartbeat_interval_s,
            stall_after_s=self._stall_after_s,
            progress_observable=snapshot.progress_observable,
            progress_source=snapshot.progress_source,
            progress_event_count=snapshot.progress_event_count,
            error=error,
        )
        self._lifecycle_terminal_emitted = True

    async def _finalize(self) -> None:
        if self._result is not None:
            return
        # Deferred import to avoid circular dependencies during module extraction.
        from llm_client import client as _client_mod

        _client: Any = _client_mod

        content = "".join(self._chunks_text)
        usage: dict[str, Any] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        cost = 0.0
        finish_reason = "stop"
        tool_calls: list[dict[str, Any]] = []
        try:
            complete = litellm.stream_chunk_builder(self._raw_chunks)
            if complete:
                usage = _client._extract_usage(complete)
                cost, cost_source = _client._parse_cost_result(_client._compute_cost(complete))
                first_choice = complete.choices[0]
                finish_reason = str(getattr(first_choice, "finish_reason", "") or "stop")
                message_obj = getattr(first_choice, "message", None)
                if message_obj is not None and getattr(message_obj, "tool_calls", None):
                    tool_calls = _client._extract_tool_calls(message_obj)
            else:
                cost_source = "unavailable"
        except Exception:
            cost_source = "unavailable"
        self._result = LLMCallResult(
            content=content,
            usage=usage,
            cost=cost,
            model=self._model,
            requested_model=self._requested_model,
            resolved_model=self._resolved_model or self._model,
            routing_trace=self._routing_trace,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            raw_response=self._raw_chunks[-1] if self._raw_chunks else None,
            warnings=self._warnings,
            cost_source=cost_source,
        )
        self._result = _client._finalize_result(
            self._result,
            requested_model=self._requested_model or self._model,
            resolved_model=self._resolved_model or self._model,
            routing_trace=self._routing_trace,
        )
        if self._hooks and self._hooks.after_call:
            self._hooks.after_call(self._result)
        _client._log_call_event(
            model=self._model,
            messages=self._messages,
            result=self._result,
            latency_s=time.monotonic() - self._t0,
            caller="astream_llm",
            task=self._task,
            trace_id=self._trace_id,
            prompt_ref=self._prompt_ref,
        )
        await self._emit_lifecycle_terminal(phase="completed")

    async def _finalize_failed(self, error: Exception) -> None:
        """Emit a terminal failure event if async stream consumption raises mid-flight."""
        await self._emit_lifecycle_terminal(phase="failed", error=error)

    @property
    def result(self) -> LLMCallResult:
        """The accumulated result. Available after the stream is fully consumed."""
        if self._result is None:
            raise RuntimeError("Stream not yet consumed. Iterate first.")
        return self._result
