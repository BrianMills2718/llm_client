"""Tests for shared call snapshot comparison and replay helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llm_client import io_log
from llm_client.observability import replay as replay_module


@pytest.fixture(autouse=True)
def _isolate_io_log(tmp_path: Path):
    """Isolate observability state for replay-helper tests."""

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

    yield tmp_path

    io_log._enabled = old_enabled
    io_log._data_root = old_root
    io_log._project = old_project
    io_log._db_path = old_db_path
    if io_log._db_conn is not None:
        io_log._db_conn.close()
    io_log._db_conn = old_db_conn
    io_log._last_cleanup_date = old_last_cleanup


class ReplayItem:
    """Minimal module-level structured model for replay-path tests."""

    @classmethod
    def model_json_schema(cls) -> dict[str, object]:
        return {
            "title": "ReplayItem",
            "type": "object",
            "properties": {
                "value": {"type": "string", "description": "Replay value."},
            },
            "required": ["value"],
        }


def _insert_call(snapshot: dict[str, object], *, response: str = "ok", error: str | None = None) -> int:
    """Insert one call row with snapshot metadata and return its call id."""

    result = MagicMock(
        content=response,
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        cost=0.0,
        finish_reason="stop",
    )
    io_log.log_call(
        model="resolved-model",
        messages=[{"role": "user", "content": "hi"}],
        result=result if error is None else None,
        error=RuntimeError(error) if error is not None else None,
        latency_s=0.1,
        caller="call_llm",
        task="test.task",
        trace_id="trace.original",
        prompt_ref="prompt@1",
        call_snapshot=snapshot,
        call_fingerprint=replay_module.snapshot_fingerprint(snapshot),
    )
    db = io_log._get_db()
    row = db.execute("SELECT MAX(id) FROM llm_calls").fetchone()
    assert row is not None and row[0] is not None
    return int(row[0])


def test_snapshot_fingerprint_ignores_ephemeral_metadata() -> None:
    left = replay_module.build_call_snapshot(
        public_api="call_llm",
        call_kind="text",
        requested_model="gpt-5",
        messages=[{"role": "user", "content": "hi"}],
        prompt_ref="prompt@1",
        timeout=60,
        num_retries=2,
        reasoning_effort=None,
        api_base=None,
        base_delay=1.0,
        max_delay=30.0,
        retry_on=None,
        fallback_models=None,
        public_kwargs={"temperature": 0},
    )
    right = replay_module.build_call_snapshot(
        public_api="acall_llm",
        call_kind="text",
        requested_model="gpt-5",
        messages=[{"role": "user", "content": "hi"}],
        prompt_ref="prompt@1",
        timeout=60,
        num_retries=2,
        reasoning_effort=None,
        api_base=None,
        base_delay=1.0,
        max_delay=30.0,
        retry_on=None,
        fallback_models=None,
        public_kwargs={"temperature": 0},
    )

    assert replay_module.snapshot_fingerprint(left) == replay_module.snapshot_fingerprint(right)


def test_snapshot_marks_non_json_kwargs_as_replay_unsupported() -> None:
    snapshot = replay_module.build_call_snapshot(
        public_api="call_llm",
        call_kind="text",
        requested_model="gpt-5",
        messages=[{"role": "user", "content": "hi"}],
        prompt_ref=None,
        timeout=60,
        num_retries=0,
        reasoning_effort=None,
        api_base=None,
        base_delay=1.0,
        max_delay=30.0,
        retry_on=None,
        fallback_models=None,
        public_kwargs={"non_json": object()},
    )

    replay = snapshot["replay"]
    assert isinstance(replay, dict)
    assert replay["unsupported_keys"] == ["non_json"]


def test_compare_call_snapshots_reports_compact_differences() -> None:
    left = replay_module.build_call_snapshot(
        public_api="call_llm",
        call_kind="text",
        requested_model="gpt-5",
        messages=[{"role": "user", "content": "hi"}],
        prompt_ref="prompt@1",
        timeout=60,
        num_retries=2,
        reasoning_effort=None,
        api_base=None,
        base_delay=1.0,
        max_delay=30.0,
        retry_on=None,
        fallback_models=None,
        public_kwargs={"temperature": 0},
    )
    right = replay_module.build_call_snapshot(
        public_api="call_llm",
        call_kind="text",
        requested_model="gpt-5",
        messages=[{"role": "user", "content": "different"}],
        prompt_ref="prompt@1",
        timeout=60,
        num_retries=2,
        reasoning_effort=None,
        api_base=None,
        base_delay=1.0,
        max_delay=30.0,
        retry_on=None,
        fallback_models=None,
        public_kwargs={"temperature": 0},
    )

    left_id = _insert_call(left, response="left")
    right_id = _insert_call(right, response="right")

    report = replay_module.compare_call_snapshots(left_id, right_id)
    assert report["fingerprints_match"] is False
    assert any("request.messages[0].content" in diff for diff in report["request_differences"])
    assert "request:" in replay_module.format_call_diff(report)


def test_replay_call_snapshot_uses_new_trace_and_preserves_original_record(monkeypatch) -> None:
    snapshot = replay_module.build_call_snapshot(
        public_api="call_llm_structured",
        call_kind="structured",
        requested_model="gpt-5",
        messages=[{"role": "user", "content": "hi"}],
        prompt_ref="prompt@1",
        timeout=60,
        num_retries=1,
        reasoning_effort=None,
        api_base=None,
        base_delay=1.0,
        max_delay=30.0,
        retry_on=None,
        fallback_models=None,
        public_kwargs={"temperature": 0},
        response_model=ReplayItem,
    )
    call_id = _insert_call(snapshot)

    captured: dict[str, object] = {}

    def fake_structured(model: str, messages: list[dict[str, object]], response_model: type[object], **kwargs: object) -> tuple[dict[str, str], str]:
        captured["model"] = model
        captured["messages"] = messages
        captured["response_model"] = response_model
        captured["kwargs"] = kwargs
        return {"value": "ok"}, "result"

    monkeypatch.setattr(replay_module, "_call_structured_for_replay", fake_structured)

    result = replay_module.replay_call_snapshot(
        call_id,
        trace_id="trace.replay",
        task="task.replay",
        max_budget=0.0,
        project="project.replay",
    )

    assert result["source_call_id"] == call_id
    assert captured["model"] == "gpt-5"
    assert captured["response_model"] is ReplayItem
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["trace_id"] == "trace.replay"
    assert kwargs["task"] == "task.replay"
    original = replay_module.get_call_record(call_id)
    assert original["trace_id"] == "trace.original"
    assert original["project"] == "test_project"
