"""Focused regression tests for public-call lifecycle emission.

These tests isolate the wrapper boundary in ``llm_client.client`` and prove
that the public text/structured entrypoints emit lifecycle events again after
the wrapper-side liveness logic was restored.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
