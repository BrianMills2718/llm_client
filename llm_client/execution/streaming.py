"""Streaming wrapper classes for LLM responses.

Houses LLMStream (sync) and AsyncLLMStream (async), which yield text
chunks as they arrive and expose the accumulated .result after the
stream is fully consumed.

These classes reference cost/usage helpers and finalization functions from
client.py at runtime to avoid circular imports during module extraction.
"""

from __future__ import annotations

import time
from typing import Any

import litellm

from llm_client.core.data_types import LLMCallResult
from llm_client.execution.retry import Hooks


class LLMStream:
    """Sync streaming wrapper. Yields text chunks, then exposes ``.result``.

    Example::

        stream = stream_llm("gpt-5-mini", messages)
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

    def __iter__(self) -> LLMStream:
        return self

    def __next__(self) -> str:
        try:
            chunk = next(self._iter)
        except StopIteration:
            self._finalize()
            raise
        self._raw_chunks.append(chunk)
        text = ""
        if chunk.choices:
            delta = chunk.choices[0].delta
            text = (delta.content if delta and delta.content else "") or ""
        self._chunks_text.append(text)
        return text

    def _finalize(self) -> None:
        # Deferred import to avoid circular dependencies during module extraction.
        from llm_client.core import client as _client

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

    @property
    def result(self) -> LLMCallResult:
        """The accumulated result. Available after the stream is fully consumed."""
        if self._result is None:
            raise RuntimeError("Stream not yet consumed. Iterate first.")
        return self._result


class AsyncLLMStream:
    """Async streaming wrapper. Yields text chunks, then exposes ``.result``.

    Example::

        stream = await astream_llm("gpt-5-mini", messages)
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

    def __aiter__(self) -> AsyncLLMStream:
        return self

    async def __anext__(self) -> str:
        try:
            chunk = await self._iter.__anext__()
        except StopAsyncIteration:
            self._finalize()
            raise
        self._raw_chunks.append(chunk)
        text = ""
        if chunk.choices:
            delta = chunk.choices[0].delta
            text = (delta.content if delta and delta.content else "") or ""
        self._chunks_text.append(text)
        return text

    def _finalize(self) -> None:
        # Deferred import to avoid circular dependencies during module extraction.
        from llm_client.core import client as _client

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

    @property
    def result(self) -> LLMCallResult:
        """The accumulated result. Available after the stream is fully consumed."""
        if self._result is None:
            raise RuntimeError("Stream not yet consumed. Iterate first.")
        return self._result
