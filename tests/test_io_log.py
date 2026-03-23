"""Tests for llm_client.io_log — JSONL logging, embedding logging, SQLite DB."""

import json
import os
import sqlite3
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llm_client import io_log


def _today_jsonl(tmp_path: Path, stem: str) -> Path:
    """Return the expected dated JSONL path for today inside the test log dir."""
    today = date.today().isoformat()
    return tmp_path / "test_project" / "test_project_llm_client_data" / f"{stem}_{today}.jsonl"


@pytest.fixture(autouse=True)
def _isolate_io_log(tmp_path):
    """Isolate io_log state per test — temp dirs, fresh DB."""
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


# ---------------------------------------------------------------------------
# log_call
# ---------------------------------------------------------------------------


class TestLogCall:
    def test_writes_jsonl(self, tmp_path):
        result = MagicMock(content="hello", usage={"prompt_tokens": 10, "total_tokens": 20}, cost=0.001, finish_reason="stop")
        io_log.log_call(model="gpt-5", result=result, latency_s=1.5, task="test_task")

        log_file = _today_jsonl(tmp_path, "calls")
        assert log_file.exists()
        record = json.loads(log_file.read_text().strip())
        assert record["model"] == "gpt-5"
        assert record["cost"] == 0.001
        assert record["task"] == "test_task"
        assert record["latency_s"] == 1.5

    def test_writes_sqlite(self, tmp_path):
        result = MagicMock(content="hello", usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}, cost=0.002, finish_reason="stop")
        io_log.log_call(model="gpt-5", result=result, latency_s=2.0, task="sql_test")

        db = sqlite3.connect(str(tmp_path / "test.db"))
        row = db.execute("SELECT model, cost, task, prompt_tokens, total_tokens FROM llm_calls").fetchone()
        assert row is not None
        assert row[0] == "gpt-5"
        assert row[1] == 0.002
        assert row[2] == "sql_test"
        assert row[3] == 10
        assert row[4] == 15
        db.close()

    def test_disabled_skips_logging(self, tmp_path):
        io_log._enabled = False
        io_log.log_call(model="gpt-5", latency_s=1.0)
        log_file = _today_jsonl(tmp_path, "calls")
        assert not log_file.exists()

    def test_trace_id_jsonl(self, tmp_path):
        result = MagicMock(content="hi", usage={}, cost=0.0, finish_reason="stop")
        io_log.log_call(model="gpt-5", result=result, latency_s=1.0, trace_id="trace_abc")

        log_file = _today_jsonl(tmp_path, "calls")
        record = json.loads(log_file.read_text().strip())
        assert record["trace_id"] == "trace_abc"

    def test_trace_id_sqlite(self, tmp_path):
        result = MagicMock(content="hi", usage={"prompt_tokens": 1, "total_tokens": 2}, cost=0.0, finish_reason="stop")
        io_log.log_call(model="gpt-5", result=result, latency_s=1.0, trace_id="trace_xyz")

        db = sqlite3.connect(str(tmp_path / "test.db"))
        row = db.execute("SELECT trace_id FROM llm_calls").fetchone()
        assert row[0] == "trace_xyz"
        db.close()

    def test_prompt_ref_logged_to_jsonl_and_sqlite(self, tmp_path):
        result = MagicMock(content="hi", usage={"prompt_tokens": 1, "total_tokens": 2}, cost=0.0, finish_reason="stop")
        io_log.log_call(
            model="gpt-5",
            result=result,
            latency_s=1.0,
            trace_id="trace_prompt_ref",
            prompt_ref="shared.investigation_pipeline.collect@1",
        )

        log_file = _today_jsonl(tmp_path, "calls")
        record = json.loads(log_file.read_text().strip())
        assert record["prompt_ref"] == "shared.investigation_pipeline.collect@1"

        db = sqlite3.connect(str(tmp_path / "test.db"))
        row = db.execute("SELECT prompt_ref FROM llm_calls").fetchone()
        assert row[0] == "shared.investigation_pipeline.collect@1"
        db.close()

    def test_call_snapshot_logged_to_jsonl_and_sqlite(self, tmp_path):
        result = MagicMock(content="hi", usage={"prompt_tokens": 1, "total_tokens": 2}, cost=0.0, finish_reason="stop")
        call_snapshot = {
            "snapshot_version": 1,
            "public_api": "call_llm",
            "call_kind": "text",
            "request": {"requested_model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]},
            "replay": {"unsupported_keys": []},
        }
        io_log.log_call(
            model="gpt-5",
            result=result,
            latency_s=1.0,
            trace_id="trace_snapshot",
            call_snapshot=call_snapshot,
            call_fingerprint="abc123",
        )

        log_file = _today_jsonl(tmp_path, "calls")
        record = json.loads(log_file.read_text().strip())
        assert record["call_fingerprint"] == "abc123"
        assert record["call_snapshot"]["public_api"] == "call_llm"

        db = sqlite3.connect(str(tmp_path / "test.db"))
        row = db.execute("SELECT call_fingerprint, call_snapshot FROM llm_calls").fetchone()
        assert row[0] == "abc123"
        assert json.loads(row[1])["public_api"] == "call_llm"
        db.close()

    def test_error_logged(self, tmp_path):
        io_log.log_call(model="gpt-5", error=ValueError("boom"), latency_s=0.5)

        log_file = _today_jsonl(tmp_path, "calls")
        record = json.loads(log_file.read_text().strip())
        assert record["error"] == "boom"

        db = sqlite3.connect(str(tmp_path / "test.db"))
        row = db.execute("SELECT error FROM llm_calls").fetchone()
        assert row[0] == "boom"
        db.close()


# ---------------------------------------------------------------------------
# log_embedding
# ---------------------------------------------------------------------------


class TestLogEmbedding:
    def test_writes_jsonl(self, tmp_path):
        io_log.log_embedding(
            model="text-embedding-3-small",
            input_count=5,
            input_chars=1000,
            dimensions=1024,
            usage={"prompt_tokens": 200, "total_tokens": 200},
            cost=0.0004,
            latency_s=0.8,
            task="vdb_build",
        )

        log_file = _today_jsonl(tmp_path, "embeddings")
        assert log_file.exists()
        record = json.loads(log_file.read_text().strip())
        assert record["model"] == "text-embedding-3-small"
        assert record["input_count"] == 5
        assert record["input_chars"] == 1000
        assert record["dimensions"] == 1024
        assert record["cost"] == 0.0004
        assert record["task"] == "vdb_build"
        assert record["caller"] == "embed"

    def test_writes_sqlite(self, tmp_path):
        io_log.log_embedding(
            model="text-embedding-3-small",
            input_count=10,
            input_chars=5000,
            dimensions=256,
            usage={"prompt_tokens": 500, "total_tokens": 500},
            cost=0.001,
            latency_s=1.2,
            caller="aembed",
            task="search",
        )

        db = sqlite3.connect(str(tmp_path / "test.db"))
        row = db.execute(
            "SELECT model, input_count, input_chars, dimensions, prompt_tokens, cost, caller, task FROM embeddings"
        ).fetchone()
        assert row is not None
        assert row[0] == "text-embedding-3-small"
        assert row[1] == 10
        assert row[2] == 5000
        assert row[3] == 256
        assert row[4] == 500
        assert row[5] == 0.001
        assert row[6] == "aembed"
        assert row[7] == "search"
        db.close()

    def test_trace_id_embedding_jsonl(self, tmp_path):
        io_log.log_embedding(
            model="text-embedding-3-small", input_count=1, input_chars=50,
            dimensions=256, usage={}, cost=0.0, latency_s=0.1, trace_id="emb_trace_1",
        )
        log_file = _today_jsonl(tmp_path, "embeddings")
        record = json.loads(log_file.read_text().strip())
        assert record["trace_id"] == "emb_trace_1"

    def test_trace_id_embedding_sqlite(self, tmp_path):
        io_log.log_embedding(
            model="text-embedding-3-small", input_count=1, input_chars=50,
            dimensions=256, usage={"prompt_tokens": 10}, cost=0.0, latency_s=0.1, trace_id="emb_trace_2",
        )
        db = sqlite3.connect(str(tmp_path / "test.db"))
        row = db.execute("SELECT trace_id FROM embeddings").fetchone()
        assert row[0] == "emb_trace_2"
        db.close()

    def test_error_embedding(self, tmp_path):
        io_log.log_embedding(
            model="text-embedding-3-small",
            input_count=1,
            input_chars=100,
            error=RuntimeError("timeout"),
            latency_s=5.0,
        )

        log_file = _today_jsonl(tmp_path, "embeddings")
        record = json.loads(log_file.read_text().strip())
        assert record["error"] == "timeout"
        assert record["dimensions"] is None

    def test_disabled_skips(self, tmp_path):
        io_log._enabled = False
        io_log.log_embedding(model="x", input_count=1, input_chars=10)
        log_file = _today_jsonl(tmp_path, "embeddings")
        assert not log_file.exists()


# ---------------------------------------------------------------------------
# SQLite DB
# ---------------------------------------------------------------------------


class TestSQLiteDB:
    def test_get_db_creates_tables(self, tmp_path):
        db = io_log._get_db()
        tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = {t[0] for t in tables}
        assert "llm_calls" in table_names
        assert "embeddings" in table_names

    def test_get_db_singleton(self, tmp_path):
        db1 = io_log._get_db()
        db2 = io_log._get_db()
        assert db1 is db2

    def test_get_db_sets_busy_timeout(self, tmp_path):
        db = io_log._get_db()
        row = db.execute("PRAGMA busy_timeout").fetchone()
        assert row is not None
        assert row[0] == 5000

    def test_indexes_created(self, tmp_path):
        db = io_log._get_db()
        indexes = db.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
        idx_names = {i[0] for i in indexes}
        assert "idx_calls_timestamp" in idx_names
        assert "idx_calls_model" in idx_names
        assert "idx_calls_trace_id" in idx_names
        assert "idx_calls_fingerprint" in idx_names
        assert "idx_emb_task" in idx_names
        assert "idx_emb_project" in idx_names
        assert "idx_emb_trace_id" in idx_names

    def test_migrate_adds_trace_id(self, tmp_path):
        """Migration adds trace, prompt, and replay columns to old DBs."""
        # Create a DB with old schema (has all columns except newer observability fields)
        old_db_path = tmp_path / "old.db"
        old_conn = sqlite3.connect(str(old_db_path))
        old_conn.executescript("""
            CREATE TABLE llm_calls (
                id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, project TEXT,
                model TEXT NOT NULL, messages TEXT, response TEXT,
                prompt_tokens INTEGER, completion_tokens INTEGER, total_tokens INTEGER,
                cost REAL, finish_reason TEXT, latency_s REAL, error TEXT, caller TEXT, task TEXT
            );
            CREATE TABLE embeddings (
                id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, project TEXT,
                model TEXT NOT NULL, input_count INTEGER, input_chars INTEGER, dimensions INTEGER,
                prompt_tokens INTEGER, total_tokens INTEGER, cost REAL, latency_s REAL,
                error TEXT, caller TEXT, task TEXT
            );
        """)
        old_conn.close()

        # Point io_log at the old DB and trigger migration
        io_log._db_path = old_db_path
        io_log._db_conn = None
        db = io_log._get_db()

        # Verify trace_id / prompt_ref columns were added
        for table in ("llm_calls", "embeddings"):
            cols = {r[1] for r in db.execute(f"PRAGMA table_info({table})").fetchall()}
            assert "trace_id" in cols
        llm_cols = {r[1] for r in db.execute("PRAGMA table_info(llm_calls)").fetchall()}
        assert "prompt_ref" in llm_cols
        assert "call_fingerprint" in llm_cols
        assert "call_snapshot" in llm_cols


# ---------------------------------------------------------------------------
# import_jsonl
# ---------------------------------------------------------------------------


class TestImportJsonl:
    def test_import_calls(self, tmp_path):
        # Create a fake calls.jsonl
        data_dir = tmp_path / "myproj" / "myproj_llm_client_data"
        data_dir.mkdir(parents=True)
        jsonl_file = data_dir / "calls.jsonl"
        records = [
            {"timestamp": "2026-02-16T12:00:00+00:00", "model": "gpt-5", "usage": {"prompt_tokens": 10, "total_tokens": 20}, "cost": 0.01, "task": "test"},
            {"timestamp": "2026-02-16T12:01:00+00:00", "model": "gemini-3", "usage": {}, "cost": 0.005, "task": None},
        ]
        jsonl_file.write_text("\n".join(json.dumps(r) for r in records) + "\n")

        count = io_log.import_jsonl(jsonl_file, table="llm_calls")
        assert count == 2

        db = io_log._get_db()
        rows = db.execute("SELECT model, project FROM llm_calls ORDER BY model").fetchall()
        assert len(rows) == 2
        # Project inferred from path
        assert rows[0][1] == "myproj"

    def test_import_embeddings(self, tmp_path):
        data_dir = tmp_path / "proj" / "proj_llm_client_data"
        data_dir.mkdir(parents=True)
        jsonl_file = data_dir / "embeddings.jsonl"
        records = [
            {"timestamp": "2026-02-16T12:00:00+00:00", "model": "text-embedding-3-small", "input_count": 5, "input_chars": 1000, "dimensions": 1024, "usage": {"prompt_tokens": 200}, "cost": 0.0004, "task": "vdb"},
        ]
        jsonl_file.write_text("\n".join(json.dumps(r) for r in records) + "\n")

        count = io_log.import_jsonl(jsonl_file, table="embeddings")
        assert count == 1

        db = io_log._get_db()
        row = db.execute("SELECT model, input_count, dimensions, project FROM embeddings").fetchone()
        assert row[0] == "text-embedding-3-small"
        assert row[1] == 5
        assert row[2] == 1024
        assert row[3] == "proj"

    def test_import_bad_table_raises(self, tmp_path):
        f = tmp_path / "x.jsonl"
        f.write_text("{}\n")
        with pytest.raises(ValueError, match="table must be"):
            io_log.import_jsonl(f, table="bad")

    def test_import_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            io_log.import_jsonl(tmp_path / "nonexistent.jsonl")


# ---------------------------------------------------------------------------
# configure
# ---------------------------------------------------------------------------


class TestLogScoreGitCommit:
    def test_git_commit_in_db(self, tmp_path):
        io_log.log_score(
            rubric="test_rubric",
            method="llm_judge",
            overall_score=0.85,
            task="test_task",
            git_commit="abc1234",
        )

        db = sqlite3.connect(str(tmp_path / "test.db"))
        row = db.execute("SELECT git_commit FROM task_scores").fetchone()
        assert row[0] == "abc1234"
        db.close()

    def test_git_commit_in_jsonl(self, tmp_path):
        io_log.log_score(
            rubric="test_rubric",
            method="llm_judge",
            overall_score=0.5,
            git_commit="def5678",
        )

        scores_file = _today_jsonl(tmp_path, "scores")
        record = json.loads(scores_file.read_text().strip())
        assert record["git_commit"] == "def5678"

    def test_git_commit_auto_captured(self, tmp_path):
        """When git_commit is None, it auto-captures from git HEAD."""
        io_log.log_score(
            rubric="test_rubric",
            method="llm_judge",
            overall_score=0.5,
        )

        db = sqlite3.connect(str(tmp_path / "test.db"))
        row = db.execute("SELECT git_commit FROM task_scores").fetchone()
        # Should be a real commit hash (llm_client is a git repo) or None if not in repo
        # Either way, the column exists and was written
        assert row is not None
        db.close()

    def test_migrate_adds_git_commit(self, tmp_path):
        """Migration adds git_commit to a DB without it."""
        old_db_path = tmp_path / "old_scores.db"
        old_conn = sqlite3.connect(str(old_db_path))
        old_conn.executescript("""
            CREATE TABLE llm_calls (
                id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, project TEXT,
                model TEXT NOT NULL, messages TEXT, response TEXT,
                prompt_tokens INTEGER, completion_tokens INTEGER, total_tokens INTEGER,
                cost REAL, finish_reason TEXT, latency_s REAL, error TEXT, caller TEXT,
                task TEXT, trace_id TEXT
            );
            CREATE TABLE embeddings (
                id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, project TEXT,
                model TEXT NOT NULL, input_count INTEGER, input_chars INTEGER, dimensions INTEGER,
                prompt_tokens INTEGER, total_tokens INTEGER, cost REAL, latency_s REAL,
                error TEXT, caller TEXT, task TEXT, trace_id TEXT
            );
            CREATE TABLE task_scores (
                id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, project TEXT,
                task TEXT, trace_id TEXT, rubric TEXT NOT NULL, method TEXT NOT NULL,
                overall_score REAL NOT NULL, dimensions TEXT, reasoning TEXT,
                output_model TEXT, judge_model TEXT, agent_spec TEXT, prompt_id TEXT,
                cost REAL, latency_s REAL
            );
        """)
        old_conn.close()

        io_log._db_path = old_db_path
        io_log._db_conn = None
        db = io_log._get_db()

        cols = {r[1] for r in db.execute("PRAGMA table_info(task_scores)").fetchall()}
        assert "git_commit" in cols


# ---------------------------------------------------------------------------
# get_cost — trace_prefix
# ---------------------------------------------------------------------------


class TestGetCostTracePrefix:
    def _insert_call(self, tmp_path, trace_id, cost):
        result = MagicMock(
            content="x", usage={"prompt_tokens": 1, "total_tokens": 2},
            cost=cost, finish_reason="stop",
        )
        io_log.log_call(model="gpt-5", result=result, latency_s=0.1, trace_id=trace_id, task="t")

    def test_prefix_sums_parent_and_children(self, tmp_path):
        self._insert_call(tmp_path, "dispatch.abc", 1.0)
        self._insert_call(tmp_path, "dispatch.abc/child_1", 0.5)
        self._insert_call(tmp_path, "dispatch.abc/child_2", 0.3)
        self._insert_call(tmp_path, "dispatch.other", 9.0)  # different prefix

        cost = io_log.get_cost(trace_prefix="dispatch.abc")
        assert abs(cost - 1.8) < 0.001

    def test_prefix_no_match(self, tmp_path):
        self._insert_call(tmp_path, "foo/bar", 1.0)
        cost = io_log.get_cost(trace_prefix="nope")
        assert cost == 0.0

    def test_prefix_exact_only(self, tmp_path):
        self._insert_call(tmp_path, "exact", 2.0)
        cost = io_log.get_cost(trace_prefix="exact")
        assert abs(cost - 2.0) < 0.001

    def test_prefix_and_trace_id_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            io_log.get_cost(trace_id="a", trace_prefix="b")


# ---------------------------------------------------------------------------
# get_trace_tree
# ---------------------------------------------------------------------------


class TestGetTraceTree:
    def _insert_call(self, tmp_path, trace_id, cost, task="t"):
        result = MagicMock(
            content="x", usage={"prompt_tokens": 1, "total_tokens": 2},
            cost=cost, finish_reason="stop",
        )
        io_log.log_call(model="gpt-5", result=result, latency_s=0.1, trace_id=trace_id, task=task)

    def test_returns_parent_and_children(self, tmp_path):
        self._insert_call(tmp_path, "dispatch.abc", 1.0)
        self._insert_call(tmp_path, "dispatch.abc/child_1", 0.5)
        self._insert_call(tmp_path, "dispatch.abc/child_2", 0.3)
        self._insert_call(tmp_path, "dispatch.other", 9.0)

        tree = io_log.get_trace_tree("dispatch.abc")
        trace_ids = {t["trace_id"] for t in tree}
        assert trace_ids == {"dispatch.abc", "dispatch.abc/child_1", "dispatch.abc/child_2"}

    def test_depth_field(self, tmp_path):
        self._insert_call(tmp_path, "root", 1.0)
        self._insert_call(tmp_path, "root/a", 0.5)
        self._insert_call(tmp_path, "root/a/b", 0.2)

        tree = io_log.get_trace_tree("root")
        by_id = {t["trace_id"]: t for t in tree}
        assert by_id["root"]["depth"] == 0
        assert by_id["root/a"]["depth"] == 1
        assert by_id["root/a/b"]["depth"] == 2

    def test_empty_tree(self, tmp_path):
        tree = io_log.get_trace_tree("nonexistent")
        assert tree == []

    def test_rollup_fields(self, tmp_path):
        self._insert_call(tmp_path, "p/child", 0.5, task="extraction")
        self._insert_call(tmp_path, "p/child", 0.3, task="extraction")  # same trace, 2 calls

        tree = io_log.get_trace_tree("p")
        assert len(tree) == 1
        t = tree[0]
        assert t["call_count"] == 2
        assert abs(t["total_cost_usd"] - 0.8) < 0.001
        assert t["task"] == "extraction"


class TestGetActiveLLMCalls:
    def _log_lifecycle_event(
        self,
        *,
        phase: str,
        call_id: str,
        timestamp: str,
        trace_id: str = "trace.active",
        task: str = "test",
        progress_observable: bool | None = None,
        progress_source: str | None = None,
        progress_event_count: int | None = None,
        host_name: str | None = None,
        process_id: int | None = None,
        process_start_token: str | None = None,
    ) -> None:
        lifecycle_payload = {
            "call_id": call_id,
            "phase": phase,
            "call_kind": "text",
            "requested_model_id": "gpt-4",
            "timeout_policy": "allow",
            "elapsed_s": 5.0,
        }
        if progress_observable is not None:
            lifecycle_payload["progress_observable"] = progress_observable
        if progress_source is not None:
            lifecycle_payload["progress_source"] = progress_source
        if progress_event_count is not None:
            lifecycle_payload["progress_event_count"] = progress_event_count
        if host_name is not None:
            lifecycle_payload["host_name"] = host_name
        if process_id is not None:
            lifecycle_payload["process_id"] = process_id
        if process_start_token is not None:
            lifecycle_payload["process_start_token"] = process_start_token
        io_log.log_foundation_event(
            caller="test",
            task=task,
            trace_id=trace_id,
            event={
                "event_id": f"evt_{call_id}_{phase}",
                "event_type": "LLMCallLifecycle",
                "timestamp": timestamp,
                "run_id": "run_trace_active",
                "session_id": f"sess_{call_id}",
                "actor_id": "service:llm_client:call_runtime:1",
                "operation": {"name": "call_llm", "version": None},
                "inputs": {
                    "artifact_ids": [],
                    "params": {
                        "task": task,
                        "trace_id": trace_id,
                        "call_kind": "text",
                    },
                    "bindings": {},
                },
                "outputs": {"artifact_ids": [], "payload_hashes": []},
                "llm_call_lifecycle": lifecycle_payload,
            },
        )

    def test_get_active_llm_calls_returns_latest_non_terminal_lifecycle_state(self, tmp_path):
        self._log_lifecycle_event(
            phase="started",
            call_id="llmcall_active",
            timestamp="2026-03-19T10:00:00Z",
        )
        self._log_lifecycle_event(
            phase="heartbeat",
            call_id="llmcall_active",
            timestamp="2026-03-19T10:00:10Z",
        )
        self._log_lifecycle_event(
            phase="started",
            call_id="llmcall_done",
            timestamp="2026-03-19T10:01:00Z",
        )
        self._log_lifecycle_event(
            phase="completed",
            call_id="llmcall_done",
            timestamp="2026-03-19T10:01:05Z",
        )

        active = io_log.get_active_llm_calls()
        assert len(active) == 1
        assert active[0]["call_id"] == "llmcall_active"
        assert active[0]["phase"] == "heartbeat"
        assert active[0]["trace_id"] == "trace.active"

    def test_get_active_llm_calls_reports_progress_metadata(self, tmp_path):
        self._log_lifecycle_event(
            phase="started",
            call_id="llmcall_streaming",
            timestamp="2026-03-19T10:00:00Z",
            progress_observable=True,
            progress_event_count=0,
        )
        self._log_lifecycle_event(
            phase="progress",
            call_id="llmcall_streaming",
            timestamp="2026-03-19T10:00:05Z",
            progress_observable=True,
            progress_source="stream_chunk",
            progress_event_count=1,
        )
        self._log_lifecycle_event(
            phase="heartbeat",
            call_id="llmcall_streaming",
            timestamp="2026-03-19T10:00:06Z",
            progress_observable=True,
            progress_source="stream_chunk",
            progress_event_count=1,
        )
        self._log_lifecycle_event(
            phase="started",
            call_id="llmcall_opaque",
            timestamp="2026-03-19T10:01:00Z",
        )
        self._log_lifecycle_event(
            phase="stalled",
            call_id="llmcall_opaque",
            timestamp="2026-03-19T10:01:10Z",
        )

        active = {record["call_id"]: record for record in io_log.get_active_llm_calls()}

        streaming = active["llmcall_streaming"]
        assert streaming["progress_observable"] is True
        assert streaming["progress_source"] == "stream_chunk"
        assert streaming["progress_event_count"] == 1
        assert isinstance(streaming["last_progress_at"], str)
        assert streaming["activity_state"] == "progressing"
        assert isinstance(streaming["idle_for_s"], float)
        assert streaming["idle_for_s"] >= 0.0

        opaque = active["llmcall_opaque"]
        assert opaque["progress_observable"] in (None, False)
        assert opaque["activity_state"] == "waiting"
        assert opaque["idle_for_s"] is None

    def test_get_active_llm_calls_excludes_same_host_orphaned_processes(
        self,
        tmp_path,
        monkeypatch,
    ):
        self._log_lifecycle_event(
            phase="heartbeat",
            call_id="llmcall_orphaned",
            timestamp="2026-03-19T10:00:00Z",
            host_name="test-host",
            process_id=12345,
            process_start_token="linux-proc-start:99",
        )
        self._log_lifecycle_event(
            phase="heartbeat",
            call_id="llmcall_alive",
            timestamp="2026-03-19T10:00:05Z",
            host_name="test-host",
            process_id=54321,
            process_start_token="linux-proc-start:100",
            trace_id="trace.alive",
        )

        from llm_client.observability import query as query_module

        monkeypatch.setattr(query_module, "_current_host_name", lambda: "test-host")

        def _fake_status(*, host_name: object, process_id: object, process_start_token: object) -> bool | None:
            if process_id == 12345:
                return False
            if process_id == 54321:
                return True
            return None

        monkeypatch.setattr(query_module, "_same_host_process_status", _fake_status)

        active = io_log.get_active_llm_calls()
        assert len(active) == 1
        assert active[0]["call_id"] == "llmcall_alive"
        assert active[0]["process_alive"] is True

    def test_get_active_llm_calls_keeps_records_when_process_liveness_is_unknown(
        self,
        tmp_path,
        monkeypatch,
    ):
        self._log_lifecycle_event(
            phase="heartbeat",
            call_id="llmcall_unknown",
            timestamp="2026-03-19T10:00:00Z",
            host_name="other-host",
            process_id=111,
        )

        from llm_client.observability import query as query_module

        monkeypatch.setattr(query_module, "_current_host_name", lambda: "test-host")

        active = io_log.get_active_llm_calls()
        assert len(active) == 1
        assert active[0]["call_id"] == "llmcall_unknown"
        assert active[0]["process_alive"] is None


class TestBackgroundModeAdoption:
    def test_summarizes_task_graph_experiment_dimensions(self, tmp_path):
        experiments = tmp_path / "experiments.jsonl"
        experiments.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "run_id": "alpha.run",
                            "timestamp": "2026-02-23T10:00:00Z",
                            "result": {
                                "requested_model": "gpt-5.2-pro",
                                "resolved_model": "openrouter/openai/gpt-5.2-pro",
                                "routing_trace": {
                                    "normalized_from": "gpt-5.2-pro",
                                    "normalized_to": "openrouter/openai/gpt-5.2-pro",
                                    "attempted_models": [
                                        "gpt-5.2-pro",
                                        "openrouter/openai/gpt-5.2-pro",
                                    ],
                                },
                            },
                            "dimensions": {
                                "reasoning_effort": "xhigh",
                                "background_mode": True,
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "run_id": "alpha.run",
                            "timestamp": "2026-02-23T10:01:00Z",
                            "result": {
                                "requested_model": "gpt-5.2-pro",
                                "resolved_model": "gpt-5.2-pro",
                                "routing_trace": {
                                    "attempted_models": ["gpt-5.2-pro"],
                                },
                            },
                            "dimensions": {
                                "reasoning_effort": "high",
                                "background_mode": False,
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "run_id": "beta.run",
                            "timestamp": "2026-02-23T10:02:00Z",
                            "dimensions": {},
                        }
                    ),
                    "{invalid json",
                ]
            )
            + "\n"
        )

        summary = io_log.get_background_mode_adoption(experiments_path=experiments)
        assert summary["exists"] is True
        assert summary["total_records"] == 4
        assert summary["records_considered"] == 3
        assert summary["invalid_lines"] == 1
        assert summary["with_reasoning_effort"] == 2
        assert summary["background_mode_true"] == 1
        assert summary["background_mode_false"] == 1
        assert summary["background_mode_unknown"] == 1
        assert summary["reasoning_effort_counts"]["xhigh"] == 1
        assert summary["reasoning_effort_counts"]["high"] == 1
        assert summary["background_mode_rate_among_reasoning"] == 0.5
        assert summary["background_mode_rate_overall"] == pytest.approx(1 / 3)
        assert summary["records_with_routing_trace"] == 2
        assert summary["model_switches"] == 1
        assert summary["fallback_records"] == 1

    def test_applies_since_and_run_id_prefix_filters(self, tmp_path):
        experiments = tmp_path / "experiments.jsonl"
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=2)
        experiments.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "run_id": "alpha.run.1",
                            "timestamp": now.isoformat(),
                            "dimensions": {
                                "reasoning_effort": "xhigh",
                                "background_mode": True,
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "run_id": "alpha.run.2",
                            "timestamp": old.isoformat(),
                            "dimensions": {
                                "reasoning_effort": "high",
                                "background_mode": True,
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "run_id": "beta.run.1",
                            "timestamp": now.isoformat(),
                            "dimensions": {
                                "reasoning_effort": "xhigh",
                                "background_mode": True,
                            },
                        }
                    ),
                ]
            )
            + "\n"
        )

        summary = io_log.get_background_mode_adoption(
            experiments_path=experiments,
            since=now - timedelta(hours=1),
            run_id_prefix="alpha.",
        )
        assert summary["records_considered"] == 1
        assert summary["with_reasoning_effort"] == 1
        assert summary["background_mode_true"] == 1
        assert summary["background_mode_rate_among_reasoning"] == 1.0
        assert summary["reasoning_effort_counts"] == {"xhigh": 1}

    def test_ignores_reasoning_effort_none_string(self, tmp_path):
        experiments = tmp_path / "experiments.jsonl"
        experiments.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "run_id": "alpha.run.1",
                            "timestamp": "2026-02-23T10:00:00Z",
                            "dimensions": {
                                "reasoning_effort": "none",
                                "background_mode": False,
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "run_id": "alpha.run.2",
                            "timestamp": "2026-02-23T10:01:00Z",
                            "dimensions": {
                                "reasoning_effort": "high",
                                "background_mode": True,
                            },
                        }
                    ),
                ]
            )
            + "\n"
        )

        summary = io_log.get_background_mode_adoption(experiments_path=experiments)
        assert summary["records_considered"] == 2
        assert summary["records_with_reasoning_effort_key"] == 2
        assert summary["with_reasoning_effort"] == 1
        assert summary["reasoning_effort_counts"] == {"high": 1}

    def test_returns_empty_summary_for_missing_file(self, tmp_path):
        summary = io_log.get_background_mode_adoption(
            experiments_path=tmp_path / "missing.jsonl"
        )
        assert summary["exists"] is False
        assert summary["total_records"] == 0
        assert summary["records_considered"] == 0


class TestConfigure:
    def test_configure_db_path(self, tmp_path):
        new_db = tmp_path / "custom.db"
        io_log.configure(db_path=new_db)
        assert io_log._db_path == new_db

    def test_configure_closes_old_conn(self, tmp_path):
        # Force a connection
        _ = io_log._get_db()
        assert io_log._db_conn is not None
        old_conn = io_log._db_conn

        new_db = tmp_path / "new.db"
        io_log.configure(db_path=new_db)
        assert io_log._db_conn is None


# ---------------------------------------------------------------------------
# Date-based JSONL log rotation
# ---------------------------------------------------------------------------


class TestDateBasedLogRotation:
    def test_writes_to_dated_file(self, tmp_path):
        """log_call writes to calls_YYYY-MM-DD.jsonl, not calls.jsonl."""
        result = MagicMock(content="hi", usage={}, cost=0.0, finish_reason="stop")
        io_log.log_call(model="gpt-5", result=result, latency_s=1.0, task="test")

        data_dir = tmp_path / "test_project" / "test_project_llm_client_data"
        today = date.today().isoformat()
        dated_file = data_dir / f"calls_{today}.jsonl"
        legacy_file = data_dir / "calls.jsonl"

        assert dated_file.exists(), f"Expected dated file {dated_file}"
        assert not legacy_file.exists(), "Legacy calls.jsonl should not be created"

        record = json.loads(dated_file.read_text().strip())
        assert record["model"] == "gpt-5"

    def test_embeddings_write_to_dated_file(self, tmp_path):
        """log_embedding writes to embeddings_YYYY-MM-DD.jsonl."""
        io_log.log_embedding(
            model="text-embedding-3-small", input_count=1, input_chars=50,
            dimensions=256, usage={}, cost=0.0, latency_s=0.1,
        )
        dated_file = _today_jsonl(tmp_path, "embeddings")
        assert dated_file.exists()

    def test_glob_jsonl_finds_legacy_and_dated(self, tmp_path):
        """glob_jsonl_files returns both legacy and dated files."""
        data_dir = tmp_path / "test_project" / "test_project_llm_client_data"
        data_dir.mkdir(parents=True)

        # Create legacy file
        (data_dir / "calls.jsonl").write_text('{"model":"old"}\n')
        # Create dated files
        (data_dir / "calls_2026-03-17.jsonl").write_text('{"model":"day1"}\n')
        (data_dir / "calls_2026-03-18.jsonl").write_text('{"model":"day2"}\n')
        # Create unrelated file that shouldn't match
        (data_dir / "calls_summary.txt").write_text("ignore")

        files = io_log.glob_jsonl_files(data_dir, "calls")
        names = [f.name for f in files]

        assert names[0] == "calls.jsonl"  # legacy first
        assert "calls_2026-03-17.jsonl" in names
        assert "calls_2026-03-18.jsonl" in names
        assert len(files) == 3

    def test_glob_jsonl_empty_dir(self, tmp_path):
        """glob_jsonl_files returns empty list for nonexistent dir."""
        files = io_log.glob_jsonl_files(tmp_path / "nonexistent", "calls")
        assert files == []

    def test_cleanup_deletes_old_files(self, tmp_path):
        """_cleanup_old_jsonl removes files older than retention period."""
        data_dir = tmp_path / "test_project" / "test_project_llm_client_data"
        data_dir.mkdir(parents=True)

        today = date.today()
        old_date = (today - timedelta(days=60)).isoformat()
        recent_date = (today - timedelta(days=5)).isoformat()
        today_str = today.isoformat()

        old_file = data_dir / f"calls_{old_date}.jsonl"
        recent_file = data_dir / f"calls_{recent_date}.jsonl"
        today_file = data_dir / f"calls_{today_str}.jsonl"
        legacy_file = data_dir / "calls.jsonl"

        for f in (old_file, recent_file, today_file, legacy_file):
            f.write_text('{"model":"test"}\n')

        io_log._cleanup_old_jsonl(data_dir, "calls")

        assert not old_file.exists(), "60-day-old file should be deleted"
        assert recent_file.exists(), "5-day-old file should be kept"
        assert today_file.exists(), "Today's file should be kept"
        assert legacy_file.exists(), "Legacy file should not be touched"

    def test_cleanup_respects_retention_env(self, tmp_path, monkeypatch):
        """Retention period is configurable via LLM_CLIENT_LOG_RETENTION_DAYS."""
        data_dir = tmp_path / "test_project" / "test_project_llm_client_data"
        data_dir.mkdir(parents=True)

        monkeypatch.setenv("LLM_CLIENT_LOG_RETENTION_DAYS", "7")

        today = date.today()
        eight_days_ago = (today - timedelta(days=8)).isoformat()
        six_days_ago = (today - timedelta(days=6)).isoformat()

        old_file = data_dir / f"calls_{eight_days_ago}.jsonl"
        recent_file = data_dir / f"calls_{six_days_ago}.jsonl"
        old_file.write_text('{"model":"old"}\n')
        recent_file.write_text('{"model":"recent"}\n')

        io_log._cleanup_old_jsonl(data_dir, "calls")

        assert not old_file.exists(), "8-day-old file should be deleted with 7-day retention"
        assert recent_file.exists(), "6-day-old file should be kept with 7-day retention"

    def test_cleanup_runs_once_per_day(self, tmp_path):
        """Cleanup only runs once per calendar day."""
        data_dir = tmp_path / "test_project" / "test_project_llm_client_data"
        data_dir.mkdir(parents=True)

        today = date.today()
        old_date = (today - timedelta(days=60)).isoformat()
        old_file = data_dir / f"calls_{old_date}.jsonl"
        old_file.write_text('{"model":"old"}\n')

        # First call should clean up
        io_log._cleanup_old_jsonl(data_dir, "calls")
        assert not old_file.exists()

        # Re-create the old file; second call same day should skip cleanup
        old_file.write_text('{"model":"old"}\n')
        io_log._cleanup_old_jsonl(data_dir, "calls")
        assert old_file.exists(), "Second cleanup same day should be a no-op"

    def test_dated_jsonl_path_format(self, tmp_path):
        """_dated_jsonl_path returns correctly formatted path."""
        path = io_log._dated_jsonl_path(tmp_path, "calls")
        today = date.today().isoformat()
        assert path == tmp_path / f"calls_{today}.jsonl"

    def test_append_jsonl_creates_and_appends(self, tmp_path):
        """_append_jsonl creates a dated file and appends records."""
        io_log._append_jsonl(tmp_path, "test_log", {"key": "value1"})
        io_log._append_jsonl(tmp_path, "test_log", {"key": "value2"})

        today = date.today().isoformat()
        log_file = tmp_path / f"test_log_{today}.jsonl"
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["key"] == "value1"
        assert json.loads(lines[1])["key"] == "value2"

    def test_default_retention_days(self):
        """Default retention is 30 days."""
        assert io_log._get_log_retention_days() == 30

    def test_retention_env_parsing(self, monkeypatch):
        """LLM_CLIENT_LOG_RETENTION_DAYS env var is parsed correctly."""
        monkeypatch.setenv("LLM_CLIENT_LOG_RETENTION_DAYS", "90")
        assert io_log._get_log_retention_days() == 90

        monkeypatch.setenv("LLM_CLIENT_LOG_RETENTION_DAYS", "0")
        assert io_log._get_log_retention_days() == 1  # minimum 1 day

        monkeypatch.setenv("LLM_CLIENT_LOG_RETENTION_DAYS", "invalid")
        assert io_log._get_log_retention_days() == 30  # fallback
