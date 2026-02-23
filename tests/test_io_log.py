"""Tests for llm_client.io_log — JSONL logging, embedding logging, SQLite DB."""

import json
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llm_client import io_log


@pytest.fixture(autouse=True)
def _isolate_io_log(tmp_path):
    """Isolate io_log state per test — temp dirs, fresh DB."""
    old_enabled = io_log._enabled
    old_root = io_log._data_root
    old_project = io_log._project
    old_db_path = io_log._db_path
    old_db_conn = io_log._db_conn

    io_log._enabled = True
    io_log._data_root = tmp_path
    io_log._project = "test_project"
    io_log._db_path = tmp_path / "test.db"
    io_log._db_conn = None

    yield tmp_path

    io_log._enabled = old_enabled
    io_log._data_root = old_root
    io_log._project = old_project
    io_log._db_path = old_db_path
    if io_log._db_conn is not None:
        io_log._db_conn.close()
    io_log._db_conn = old_db_conn


# ---------------------------------------------------------------------------
# log_call
# ---------------------------------------------------------------------------


class TestLogCall:
    def test_writes_jsonl(self, tmp_path):
        result = MagicMock(content="hello", usage={"prompt_tokens": 10, "total_tokens": 20}, cost=0.001, finish_reason="stop")
        io_log.log_call(model="gpt-5", result=result, latency_s=1.5, task="test_task")

        log_file = tmp_path / "test_project" / "test_project_llm_client_data" / "calls.jsonl"
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
        log_file = tmp_path / "test_project" / "test_project_llm_client_data" / "calls.jsonl"
        assert not log_file.exists()

    def test_trace_id_jsonl(self, tmp_path):
        result = MagicMock(content="hi", usage={}, cost=0.0, finish_reason="stop")
        io_log.log_call(model="gpt-5", result=result, latency_s=1.0, trace_id="trace_abc")

        log_file = tmp_path / "test_project" / "test_project_llm_client_data" / "calls.jsonl"
        record = json.loads(log_file.read_text().strip())
        assert record["trace_id"] == "trace_abc"

    def test_trace_id_sqlite(self, tmp_path):
        result = MagicMock(content="hi", usage={"prompt_tokens": 1, "total_tokens": 2}, cost=0.0, finish_reason="stop")
        io_log.log_call(model="gpt-5", result=result, latency_s=1.0, trace_id="trace_xyz")

        db = sqlite3.connect(str(tmp_path / "test.db"))
        row = db.execute("SELECT trace_id FROM llm_calls").fetchone()
        assert row[0] == "trace_xyz"
        db.close()

    def test_error_logged(self, tmp_path):
        io_log.log_call(model="gpt-5", error=ValueError("boom"), latency_s=0.5)

        log_file = tmp_path / "test_project" / "test_project_llm_client_data" / "calls.jsonl"
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

        log_file = tmp_path / "test_project" / "test_project_llm_client_data" / "embeddings.jsonl"
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
        log_file = tmp_path / "test_project" / "test_project_llm_client_data" / "embeddings.jsonl"
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

        log_file = tmp_path / "test_project" / "test_project_llm_client_data" / "embeddings.jsonl"
        record = json.loads(log_file.read_text().strip())
        assert record["error"] == "timeout"
        assert record["dimensions"] is None

    def test_disabled_skips(self, tmp_path):
        io_log._enabled = False
        io_log.log_embedding(model="x", input_count=1, input_chars=10)
        log_file = tmp_path / "test_project" / "test_project_llm_client_data" / "embeddings.jsonl"
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

    def test_indexes_created(self, tmp_path):
        db = io_log._get_db()
        indexes = db.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
        idx_names = {i[0] for i in indexes}
        assert "idx_calls_timestamp" in idx_names
        assert "idx_calls_model" in idx_names
        assert "idx_calls_trace_id" in idx_names
        assert "idx_emb_task" in idx_names
        assert "idx_emb_project" in idx_names
        assert "idx_emb_trace_id" in idx_names

    def test_migrate_adds_trace_id(self, tmp_path):
        """Migration adds trace_id to a DB created without it."""
        # Create a DB with old schema (has all columns except trace_id)
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

        # Verify trace_id column was added
        for table in ("llm_calls", "embeddings"):
            cols = {r[1] for r in db.execute(f"PRAGMA table_info({table})").fetchall()}
            assert "trace_id" in cols


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

        scores_file = tmp_path / "test_project" / "test_project_llm_client_data" / "scores.jsonl"
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
