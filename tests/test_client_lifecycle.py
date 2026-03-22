"""Focused regression tests for public-call lifecycle emission.

These tests isolate the wrapper boundary in ``llm_client.client`` and prove
that the public text/structured entrypoints emit lifecycle events again after
the wrapper-side liveness logic was restored.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from llm_client import client, io_log
from llm_client.data_types import LLMCallResult


@pytest.fixture(autouse=True)
def _isolate_io_log(tmp_path: Path):
    """Keep lifecycle logging isolated in a temp DB for each test."""

    old_enabled = io_log._enabled
    old_root = io_log._data_root
    old_project = io_log._project
    old_db_path = io_log._db_path
    old_db_conn = io_log._db_conn
    old_last_cleanup = io_log._last_cleanup_date

    io_log._enabled = True
    io_log._data_root = tmp_path
    io_log._project = "test_project"
    io_log._db_path = tmp_path / "test.db"
    io_log._db_conn = None
    io_log._last_cleanup_date = None

    yield

    io_log._enabled = old_enabled
    io_log._data_root = old_root
    io_log._project = old_project
    io_log._db_path = old_db_path
    if io_log._db_conn is not None:
        io_log._db_conn.close()
    io_log._db_conn = old_db_conn
    io_log._last_cleanup_date = old_last_cleanup


class _ResponseModel(BaseModel):
    """Small structured-output contract used to test wrapper lifecycle emission."""

    label: str


def _lifecycle_rows() -> list[tuple[str, dict[str, Any]]]:
    """Return Foundation lifecycle rows from the isolated observability DB."""

    db = io_log._get_db()
    rows = db.execute(
        """
        SELECT event_type, payload
        FROM foundation_events
        WHERE event_type = 'LLMCallLifecycle'
        ORDER BY id ASC
        """
    ).fetchall()
    out: list[tuple[str, dict[str, Any]]] = []
    for event_type, payload_text in rows:
        out.append((event_type, json.loads(payload_text)))
    return out


def _mock_stream_chunk(text: str) -> Any:
    """Build one sync/async stream chunk compatible with LLMStream finalization."""

    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta = MagicMock()
    chunk.choices[0].delta.content = text
    return chunk


class _MockAsyncStream:
    """Async iterator shaped like litellm.async streaming responses."""

    def __init__(self, chunks: list[Any], fail_after: bool = False) -> None:
        self._chunks = chunks
        self._fail_after = fail_after
        self._index = 0

    def __aiter__(self) -> "_MockAsyncStream":
        return self

    async def __anext__(self) -> Any:
        if self._index < len(self._chunks):
            chunk = self._chunks[self._index]
            self._index += 1
            return chunk
        if self._fail_after:
            raise RuntimeError("boom")
        raise StopAsyncIteration


def test_call_llm_structured_emits_started_and_completed_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sync structured wrapper should emit lifecycle rows and pass the monitor through."""

    # mock-ok: isolates public wrapper lifecycle emission without provider calls.
    def _fake_impl(
        model: str,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel],
        **kwargs: Any,
    ) -> tuple[BaseModel, LLMCallResult]:
        monitor = kwargs["_lifecycle_monitor"]
        monitor.enable_progress_tracking(default_source="unit_test")
        monitor.mark_progress(source="unit_test")
        parsed = response_model(label="ok")
        result = LLMCallResult(
            content=parsed.model_dump_json(),
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            cost=0.0,
            model=model,
            resolved_model=model,
            finish_reason="stop",
            raw_response={"ok": True},
            warnings=[],
            cost_source="computed",
        )
        return parsed, result

    monkeypatch.setattr(
        "llm_client.structured_runtime._call_llm_structured_impl",
        _fake_impl,
    )

    parsed, result = client.call_llm_structured(
        "gemini/gemini-2.5-flash",
        [{"role": "user", "content": "hello"}],
        _ResponseModel,
        task="test.lifecycle",
        trace_id="trace.lifecycle.sync",
        max_budget=0.1,
        lifecycle_heartbeat_interval_s=0,
        lifecycle_stall_after_s=0,
    )

    assert parsed.label == "ok"
    assert result.resolved_model == "gemini/gemini-2.5-flash"

    rows = _lifecycle_rows()
    assert [payload["llm_call_lifecycle"]["phase"] for _, payload in rows] == [
        "started",
        "progress",
        "completed",
    ]
    completed = rows[-1][1]["llm_call_lifecycle"]
    assert completed["progress_observable"] is True
    assert completed["progress_source"] == "unit_test"
    assert completed["progress_event_count"] == 1


def test_stream_llm_emits_started_progress_completed_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sync streaming emits lifecycle rows when stream reaches natural end."""

    # mock-ok: keeps provider out of the test while validating lifecycle.
    monkeypatch.setattr(
        "llm_client.client.litellm.stream_chunk_builder",
        lambda chunks: None,
    )
    monkeypatch.setattr(
        "llm_client.client.litellm.completion",
        lambda **kwargs: iter([_mock_stream_chunk("hello")]),
    )

    stream = client.stream_llm(
        "gpt-4",
        [{"role": "user", "content": "Hi"}],
        task="test.lifecycle.stream",
        trace_id="trace.lifecycle.stream.sync",
        max_budget=0.1,
        lifecycle_heartbeat_interval_s=0,
        lifecycle_stall_after_s=0,
    )
    assert list(stream) == ["hello"]

    rows = _lifecycle_rows()
    assert [payload["llm_call_lifecycle"]["phase"] for _, payload in rows] == [
        "started",
        "progress",
        "completed",
    ]
    completed = rows[-1][1]["llm_call_lifecycle"]
    assert completed["progress_event_count"] == 1
    assert completed["progress_observable"] is True
    assert completed.get("error_type") is None


def test_stream_llm_emits_failed_lifecycle_on_iteration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sync stream iteration errors emit progress then failed lifecycle row."""

    # mock-ok: provider mocked; lifecycle behavior is the integration under test.
    class _FailingStream:
        def __iter__(self) -> "_FailingStream":
            return self

        def __next__(self) -> Any:
            if hasattr(self, "_yielded"):
                raise RuntimeError("boom")
            self._yielded = True
            return _mock_stream_chunk("part")

    monkeypatch.setattr(
        "llm_client.client.litellm.stream_chunk_builder",
        lambda chunks: None,
    )
    monkeypatch.setattr("llm_client.client.litellm.completion", lambda **kwargs: _FailingStream())

    stream = client.stream_llm(
        "gpt-4",
        [{"role": "user", "content": "Hi"}],
        task="test.lifecycle.stream",
        trace_id="trace.lifecycle.stream.sync.fail",
        max_budget=0.1,
        lifecycle_heartbeat_interval_s=0,
        lifecycle_stall_after_s=0,
    )
    with pytest.raises(RuntimeError, match="boom"):
        list(stream)

    rows = _lifecycle_rows()
    assert [payload["llm_call_lifecycle"]["phase"] for _, payload in rows] == [
        "started",
        "progress",
        "failed",
    ]
    failed = rows[-1][1]["llm_call_lifecycle"]
    assert failed["error_type"] == "RuntimeError"
    assert failed["error_message"] == "boom"
    assert failed["progress_event_count"] == 1


@pytest.mark.asyncio
async def test_acall_llm_structured_emits_failed_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Async structured wrapper should emit a terminal failed lifecycle row."""

    # mock-ok: isolates public wrapper lifecycle emission without provider calls.
    async def _fake_impl(
        model: str,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel],
        **kwargs: Any,
    ) -> tuple[BaseModel, LLMCallResult]:
        monitor = kwargs["_lifecycle_monitor"]
        monitor.enable_progress_tracking(default_source="unit_test_async")
        monitor.mark_progress(source="unit_test_async")
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "llm_client.structured_runtime._acall_llm_structured_impl",
        _fake_impl,
    )

    with pytest.raises(RuntimeError, match="boom"):
        await client.acall_llm_structured(
            "gemini/gemini-2.5-flash",
            [{"role": "user", "content": "hello"}],
            _ResponseModel,
            task="test.lifecycle",
            trace_id="trace.lifecycle.async",
            max_budget=0.1,
            lifecycle_heartbeat_interval_s=0,
            lifecycle_stall_after_s=0,
        )

    rows = _lifecycle_rows()
    assert [payload["llm_call_lifecycle"]["phase"] for _, payload in rows] == [
        "started",
        "progress",
        "failed",
    ]
    failed = rows[-1][1]["llm_call_lifecycle"]
    assert failed["progress_observable"] is True
    assert failed["progress_source"] == "unit_test_async"
    assert failed["progress_event_count"] == 1
    assert failed["error_type"] == "RuntimeError"
    assert failed["error_message"] == "boom"


@pytest.mark.asyncio
async def test_astream_llm_emits_started_progress_completed_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Async streaming emits lifecycle rows when stream reaches natural end."""

    async def _stream(**_: Any) -> _MockAsyncStream:
        return _MockAsyncStream([_mock_stream_chunk("hello")])

    monkeypatch.setattr(
        "llm_client.client.litellm.stream_chunk_builder",
        lambda chunks: None,
    )
    monkeypatch.setattr("llm_client.client.litellm.acompletion", _stream)

    stream = await client.astream_llm(
        "gpt-4",
        [{"role": "user", "content": "Hi"}],
        task="test.lifecycle.istream",
        trace_id="trace.lifecycle.stream.async",
        max_budget=0.1,
        lifecycle_heartbeat_interval_s=0,
        lifecycle_stall_after_s=0,
    )
    out: list[str] = []
    async for chunk in stream:
        out.append(chunk)
    assert out == ["hello"]

    rows = _lifecycle_rows()
    assert [payload["llm_call_lifecycle"]["phase"] for _, payload in rows] == [
        "started",
        "progress",
        "completed",
    ]
    completed = rows[-1][1]["llm_call_lifecycle"]
    assert completed["progress_observable"] is True
    assert completed["progress_event_count"] == 1


@pytest.mark.asyncio
async def test_astream_llm_emits_failed_lifecycle_on_iteration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Async iteration errors emit progress then failed lifecycle row."""

    async def _stream(**_: Any) -> _MockAsyncStream:
        return _MockAsyncStream([_mock_stream_chunk("part")], fail_after=True)

    monkeypatch.setattr(
        "llm_client.client.litellm.stream_chunk_builder",
        lambda chunks: None,
    )
    monkeypatch.setattr("llm_client.client.litellm.acompletion", _stream)

    stream = await client.astream_llm(
        "gpt-4",
        [{"role": "user", "content": "Hi"}],
        task="test.lifecycle.istream",
        trace_id="trace.lifecycle.stream.async.fail",
        max_budget=0.1,
        lifecycle_heartbeat_interval_s=0,
        lifecycle_stall_after_s=0,
    )
    out: list[str] = []
    with pytest.raises(RuntimeError, match="boom"):
        async for chunk in stream:
            out.append(chunk)

    rows = _lifecycle_rows()
    assert [payload["llm_call_lifecycle"]["phase"] for _, payload in rows] == [
        "started",
        "progress",
        "failed",
    ]
    failed = rows[-1][1]["llm_call_lifecycle"]
    assert failed["error_type"] == "RuntimeError"
    assert failed["error_message"] == "boom"
    assert failed["progress_observable"] is True
