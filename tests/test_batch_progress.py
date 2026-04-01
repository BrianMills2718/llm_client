"""Tests for batch progress tracking, stagnation detection, and item timeout (Plan #14)."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_client import io_log
from llm_client.execution.batch_runtime import BatchProgressTracker


# ---------------------------------------------------------------------------
# BatchProgressTracker unit tests
# ---------------------------------------------------------------------------


class TestBatchProgressTracker:
    def test_initial_state(self):
        t = BatchProgressTracker(total=100)
        assert t.pending == 100
        assert t.completed == 0
        assert t.errored == 0
        assert t.avg_latency_s is None
        assert t.completion_rate == 0.0

    def test_record_completion(self):
        t = BatchProgressTracker(total=10)
        t.record_completion(1.5)
        t.record_completion(2.5)
        assert t.completed == 2
        assert t.pending == 8
        assert t.avg_latency_s == 2.0
        assert t.completion_rate == 0.2

    def test_record_error(self):
        t = BatchProgressTracker(total=10)
        t.record_error(ValueError("oops"))
        assert t.errored == 1
        assert t.pending == 9

    def test_stagnation_detected(self):
        t = BatchProgressTracker(total=100, _stagnation_window=3)
        same_error = ValueError("connection refused")
        assert t.record_error(same_error) is False
        assert t.record_error(same_error) is False
        assert t.record_error(same_error) is True  # 3rd identical = stagnation

    def test_stagnation_cleared_by_success(self):
        t = BatchProgressTracker(total=100, _stagnation_window=3)
        same_error = ValueError("connection refused")
        t.record_error(same_error)
        t.record_error(same_error)
        t.record_completion(1.0)  # clears error hash window
        assert t.record_error(same_error) is False  # reset, need 3 more

    def test_stagnation_not_triggered_by_different_errors(self):
        t = BatchProgressTracker(total=100, _stagnation_window=3)
        t.record_error(ValueError("error A"))
        t.record_error(ValueError("error B"))
        assert t.record_error(ValueError("error C")) is False

    def test_summary(self):
        t = BatchProgressTracker(total=10)
        t.record_completion(1.0)
        t.record_error(ValueError("x"))
        s = t.summary()
        assert s["total"] == 10
        assert s["completed"] == 1
        assert s["errored"] == 1
        assert s["pending"] == 8

    def test_emit_run_progress_persists_snapshot(self, tmp_path):
        old_enabled = io_log._enabled
        old_root = io_log._data_root
        old_project = io_log._project
        old_db_path = io_log._db_path
        old_db_conn = io_log._db_conn
        old_run_timers = dict(io_log._run_timers)
        active_token = io_log._active_experiment_run_id.set(None)
        old_enforcement_mode = os.environ.get("LLM_CLIENT_EXPERIMENT_ENFORCEMENT")
        old_task_patterns = os.environ.get("LLM_CLIENT_EXPERIMENT_TASK_PATTERNS")

        io_log._enabled = True
        io_log._data_root = tmp_path
        io_log._project = "batch_test_project"
        io_log._db_path = tmp_path / "test.db"
        io_log._db_conn = None
        io_log._run_timers.clear()

        try:
            run_id = io_log.start_run(dataset="batch_case", model="gpt-5", task="batch.progress")
            tracker = BatchProgressTracker(total=10)
            tracker.record_completion(1.0)
            tracker.record_error(ValueError("x"))

            tracker.emit_run_progress(
                run_id=run_id,
                stage="batch_phase",
                progress_unit="items",
                checkpoint_ref="checkpoint.json",
                metadata={"source": "test"},
            )

            active = io_log.get_active_run_progress()
            assert len(active) == 1
            assert active[0]["run_id"] == run_id
            assert active[0]["stage"] == "batch_phase"
            assert active[0]["total"] == 10
            assert active[0]["completed"] == 1
            assert active[0]["failed"] == 1
            assert active[0]["progress_unit"] == "items"
            assert active[0]["checkpoint_ref"] == "checkpoint.json"
        finally:
            io_log._enabled = old_enabled
            io_log._data_root = old_root
            io_log._project = old_project
            io_log._db_path = old_db_path
            if io_log._db_conn is not None:
                io_log._db_conn.close()
            io_log._db_conn = old_db_conn
            io_log._run_timers.clear()
            io_log._run_timers.update(old_run_timers)
            io_log._active_experiment_run_id.reset(active_token)
            if old_enforcement_mode is None:
                os.environ.pop("LLM_CLIENT_EXPERIMENT_ENFORCEMENT", None)
            else:
                os.environ["LLM_CLIENT_EXPERIMENT_ENFORCEMENT"] = old_enforcement_mode
            if old_task_patterns is None:
                os.environ.pop("LLM_CLIENT_EXPERIMENT_TASK_PATTERNS", None)
            else:
                os.environ["LLM_CLIENT_EXPERIMENT_TASK_PATTERNS"] = old_task_patterns


# ---------------------------------------------------------------------------
# Integration tests (mocked LLM calls)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_progress_callback_fires():
    """on_batch_progress should fire at progress_interval and on completion."""
    # mock-ok: Testing batch progress mechanics, not LLM calls
    mock_result = MagicMock()
    mock_result.cost = 0.001

    progress_snapshots: list[dict] = []

    def on_progress(tracker: BatchProgressTracker) -> None:
        progress_snapshots.append(tracker.summary())

    with patch("llm_client.core.client.acall_llm", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_result

        from llm_client.execution.batch_runtime import acall_llm_batch_impl

        messages_list = [[{"role": "user", "content": f"q{i}"}] for i in range(5)]
        await acall_llm_batch_impl(
            "test-model",
            messages_list,
            progress_interval=2,
            on_batch_progress=on_progress,
            task="test",
            trace_id="test",
            max_budget=1.0,
        )

    # Should get at least the final summary callback
    assert len(progress_snapshots) >= 1
    final = progress_snapshots[-1]
    assert final["completed"] == 5
    assert final["errored"] == 0


@pytest.mark.asyncio
async def test_stagnation_aborts_batch():
    """abort_on_stagnation=True should cancel remaining items after detection."""
    # mock-ok: Testing stagnation abort, not LLM calls
    call_count = 0

    async def _failing_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise ValueError("identical error every time")

    with patch("llm_client.core.client.acall_llm", side_effect=_failing_call):
        from llm_client.execution.batch_runtime import acall_llm_batch_impl

        messages_list = [[{"role": "user", "content": f"q{i}"}] for i in range(20)]
        results = await acall_llm_batch_impl(
            "test-model",
            messages_list,
            return_exceptions=True,
            stagnation_window=3,
            abort_on_stagnation=True,
            max_concurrent=1,  # sequential to guarantee ordering
            task="test",
            trace_id="test",
            max_budget=1.0,
        )

    # Should have errors — some from stagnation abort, some from the actual failures
    errors = [r for r in results if isinstance(r, Exception)]
    assert len(errors) == 20  # all items are errors
    # But stagnation should have fired after 3 identical errors
    stagnation_aborts = [e for e in errors if "stagnation" in str(e).lower()]
    assert len(stagnation_aborts) > 0  # some items were skipped due to abort


@pytest.mark.asyncio
async def test_item_timeout():
    """item_timeout_s should cancel slow items without blocking the batch."""
    # mock-ok: Testing timeout mechanics, not LLM calls
    async def _slow_call(*args, **kwargs):
        await asyncio.sleep(10)  # way longer than timeout
        return MagicMock()

    with patch("llm_client.core.client.acall_llm", side_effect=_slow_call):
        from llm_client.execution.batch_runtime import acall_llm_batch_impl

        messages_list = [[{"role": "user", "content": "q"}]]
        results = await acall_llm_batch_impl(
            "test-model",
            messages_list,
            return_exceptions=True,
            item_timeout_s=0.1,  # 100ms timeout
            task="test",
            trace_id="test",
            max_budget=1.0,
        )

    assert len(results) == 1
    assert isinstance(results[0], (TimeoutError, asyncio.TimeoutError))


def test_tracker_can_emit_run_progress_snapshots(tmp_path):
    """Tracker snapshots can persist through the shared run-progress contract."""
    old_enabled = io_log._enabled
    old_root = io_log._data_root
    old_project = io_log._project
    old_db_path = io_log._db_path
    old_db_conn = io_log._db_conn
    old_run_timers = dict(io_log._run_timers)
    active_token = io_log._active_experiment_run_id.set(None)
    old_enforcement_mode = os.environ.get("LLM_CLIENT_EXPERIMENT_ENFORCEMENT")
    old_task_patterns = os.environ.get("LLM_CLIENT_EXPERIMENT_TASK_PATTERNS")

    io_log._enabled = True
    io_log._data_root = tmp_path
    io_log._project = "batch_progress_test"
    io_log._db_path = tmp_path / "test.db"
    io_log._db_conn = None
    io_log._run_timers.clear()

    try:
        run_id = io_log.start_run(dataset="batch", model="gpt-5", task="llm_client.batch")
        tracker = BatchProgressTracker(total=8)
        tracker.record_completion(1.0)
        tracker.record_completion(2.0)
        tracker.record_error(ValueError("oops"))
        tracker.emit_run_progress(
            run_id=run_id,
            stage="batch_execute",
            progress_unit="items",
            metadata={"source": "tracker"},
        )

        active = io_log.get_active_run_progress(project="batch_progress_test")
        assert len(active) == 1
        assert active[0]["run_id"] == run_id
        assert active[0]["stage"] == "batch_execute"
        assert active[0]["total"] == 8
        assert active[0]["completed"] == 2
        assert active[0]["failed"] == 1
        assert active[0]["progress_unit"] == "items"
        assert active[0]["metadata"] == {"source": "tracker"}
    finally:
        io_log._enabled = old_enabled
        io_log._data_root = old_root
        io_log._project = old_project
        io_log._db_path = old_db_path
        if io_log._db_conn is not None:
            io_log._db_conn.close()
        io_log._db_conn = old_db_conn
        io_log._run_timers.clear()
        io_log._run_timers.update(old_run_timers)
        io_log._active_experiment_run_id.reset(active_token)
        if old_enforcement_mode is None:
            os.environ.pop("LLM_CLIENT_EXPERIMENT_ENFORCEMENT", None)
        else:
            os.environ["LLM_CLIENT_EXPERIMENT_ENFORCEMENT"] = old_enforcement_mode
        if old_task_patterns is None:
            os.environ.pop("LLM_CLIENT_EXPERIMENT_TASK_PATTERNS", None)
        else:
            os.environ["LLM_CLIENT_EXPERIMENT_TASK_PATTERNS"] = old_task_patterns
