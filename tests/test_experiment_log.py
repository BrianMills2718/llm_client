"""Tests for experiment logging: start_run, log_item, finish_run, queries."""

import json
import sqlite3

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
# Tables created
# ---------------------------------------------------------------------------


class TestExperimentTables:
    def test_tables_created(self, tmp_path):
        db = io_log._get_db()
        tables = {r[0] for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "experiment_runs" in tables
        assert "experiment_items" in tables

    def test_indexes_created(self, tmp_path):
        db = io_log._get_db()
        indexes = {r[0] for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()}
        assert "idx_expr_dataset" in indexes
        assert "idx_expr_model" in indexes
        assert "idx_expri_run_id" in indexes


# ---------------------------------------------------------------------------
# start_run
# ---------------------------------------------------------------------------


class TestStartRun:
    def test_returns_run_id(self, tmp_path):
        rid = io_log.start_run(dataset="HotpotQA", model="gpt-5")
        assert isinstance(rid, str)
        assert len(rid) == 12

    def test_explicit_run_id(self, tmp_path):
        rid = io_log.start_run(dataset="HotpotQA", model="gpt-5", run_id="my_custom_id")
        assert rid == "my_custom_id"

    def test_writes_sqlite(self, tmp_path):
        rid = io_log.start_run(
            dataset="MuSiQue", model="o4-mini",
            config={"backend": "direct", "timeout": 120},
            metrics_schema=["em", "f1", "llm_em"],
        )
        db = sqlite3.connect(str(tmp_path / "test.db"))
        row = db.execute(
            "SELECT dataset, model, config, metrics_schema, status FROM experiment_runs WHERE run_id = ?",
            (rid,),
        ).fetchone()
        assert row is not None
        assert row[0] == "MuSiQue"
        assert row[1] == "o4-mini"
        assert json.loads(row[2]) == {"backend": "direct", "timeout": 120}
        assert json.loads(row[3]) == ["em", "f1", "llm_em"]
        assert row[4] == "running"
        db.close()

    def test_writes_jsonl(self, tmp_path):
        io_log.start_run(dataset="HotpotQA", model="gpt-5", run_id="jsonl_test")
        jsonl = tmp_path / "test_project" / "test_project_llm_client_data" / "experiments.jsonl"
        assert jsonl.exists()
        record = json.loads(jsonl.read_text().strip())
        assert record["type"] == "run_start"
        assert record["run_id"] == "jsonl_test"
        assert record["dataset"] == "HotpotQA"

    def test_git_commit_explicit(self, tmp_path):
        io_log.start_run(dataset="X", model="Y", run_id="gc_test", git_commit="abc123")
        db = sqlite3.connect(str(tmp_path / "test.db"))
        row = db.execute("SELECT git_commit FROM experiment_runs WHERE run_id = 'gc_test'").fetchone()
        assert row[0] == "abc123"
        db.close()


# ---------------------------------------------------------------------------
# log_item
# ---------------------------------------------------------------------------


class TestLogItem:
    def test_writes_sqlite(self, tmp_path):
        rid = io_log.start_run(dataset="X", model="Y")
        io_log.log_item(
            run_id=rid, item_id="q1",
            metrics={"em": 1, "f1": 0.85},
            predicted="Alice", gold="Alice",
            latency_s=5.2, cost=0.01, n_tool_calls=3,
        )

        db = sqlite3.connect(str(tmp_path / "test.db"))
        row = db.execute(
            "SELECT item_id, metrics, predicted, gold, latency_s, cost, n_tool_calls FROM experiment_items WHERE run_id = ?",
            (rid,),
        ).fetchone()
        assert row[0] == "q1"
        assert json.loads(row[1]) == {"em": 1, "f1": 0.85}
        assert row[2] == "Alice"
        assert row[3] == "Alice"
        assert row[4] == 5.2
        assert row[5] == 0.01
        assert row[6] == 3
        db.close()

    def test_writes_jsonl(self, tmp_path):
        rid = io_log.start_run(dataset="X", model="Y")
        io_log.log_item(
            run_id=rid, item_id="q2",
            metrics={"em": 0}, error="timeout",
        )
        jsonl = tmp_path / "test_project" / "test_project_llm_client_data" / "experiments.jsonl"
        lines = jsonl.read_text().strip().split("\n")
        # First line is run_start, second is item
        item_record = json.loads(lines[1])
        assert item_record["type"] == "item"
        assert item_record["item_id"] == "q2"
        assert item_record["error"] == "timeout"

    def test_upsert_replaces(self, tmp_path):
        """Logging the same item_id twice replaces the first."""
        rid = io_log.start_run(dataset="X", model="Y")
        io_log.log_item(run_id=rid, item_id="q1", metrics={"em": 0})
        io_log.log_item(run_id=rid, item_id="q1", metrics={"em": 1})

        db = sqlite3.connect(str(tmp_path / "test.db"))
        rows = db.execute(
            "SELECT metrics FROM experiment_items WHERE run_id = ? AND item_id = 'q1'", (rid,)
        ).fetchall()
        assert len(rows) == 1
        assert json.loads(rows[0][0])["em"] == 1
        db.close()

    def test_extra_field(self, tmp_path):
        rid = io_log.start_run(dataset="X", model="Y")
        io_log.log_item(
            run_id=rid, item_id="q1",
            metrics={"em": 1},
            extra={"tool_calls": ["entity_vdb_search", "relationship_onehop"]},
        )
        db = sqlite3.connect(str(tmp_path / "test.db"))
        row = db.execute("SELECT extra FROM experiment_items WHERE item_id = 'q1'").fetchone()
        extra = json.loads(row[0])
        assert extra["tool_calls"] == ["entity_vdb_search", "relationship_onehop"]
        db.close()


# ---------------------------------------------------------------------------
# finish_run
# ---------------------------------------------------------------------------


class TestFinishRun:
    def test_auto_aggregates_metrics(self, tmp_path):
        rid = io_log.start_run(
            dataset="HotpotQA", model="gpt-5",
            metrics_schema=["em", "f1"],
        )
        io_log.log_item(run_id=rid, item_id="q1", metrics={"em": 1, "f1": 0.8}, cost=0.01)
        io_log.log_item(run_id=rid, item_id="q2", metrics={"em": 0, "f1": 0.5}, cost=0.02)
        io_log.log_item(run_id=rid, item_id="q3", metrics={"em": 1, "f1": 1.0}, cost=0.015)

        result = io_log.finish_run(run_id=rid, wall_time_s=30.0)

        assert result["n_items"] == 3
        assert result["n_completed"] == 3
        assert result["n_errors"] == 0
        assert result["status"] == "completed"
        assert result["wall_time_s"] == 30.0

        sm = result["summary_metrics"]
        # avg_em = 100 * (1+0+1)/3 = 66.67
        assert abs(sm["avg_em"] - 66.67) < 0.1
        # avg_f1 = 100 * (0.8+0.5+1.0)/3 = 76.67
        assert abs(sm["avg_f1"] - 76.67) < 0.1

        assert abs(result["total_cost"] - 0.045) < 0.001

    def test_counts_errors(self, tmp_path):
        rid = io_log.start_run(dataset="X", model="Y", metrics_schema=["em"])
        io_log.log_item(run_id=rid, item_id="q1", metrics={"em": 1}, cost=0.01)
        io_log.log_item(run_id=rid, item_id="q2", metrics={"em": 0}, error="timeout", cost=0.0)

        result = io_log.finish_run(run_id=rid)
        assert result["n_items"] == 2
        assert result["n_completed"] == 1
        assert result["n_errors"] == 1

    def test_explicit_summary_metrics(self, tmp_path):
        rid = io_log.start_run(dataset="X", model="Y")
        io_log.log_item(run_id=rid, item_id="q1", metrics={"em": 1})

        result = io_log.finish_run(
            run_id=rid,
            summary_metrics={"custom_metric": 99.9},
        )
        assert result["summary_metrics"]["custom_metric"] == 99.9

    def test_interrupted_status(self, tmp_path):
        rid = io_log.start_run(dataset="X", model="Y")
        result = io_log.finish_run(run_id=rid, status="interrupted")
        assert result["status"] == "interrupted"

    def test_writes_jsonl(self, tmp_path):
        rid = io_log.start_run(dataset="X", model="Y")
        io_log.finish_run(run_id=rid, wall_time_s=10.0)

        jsonl = tmp_path / "test_project" / "test_project_llm_client_data" / "experiments.jsonl"
        lines = jsonl.read_text().strip().split("\n")
        finish = json.loads(lines[-1])
        assert finish["type"] == "run_finish"
        assert finish["run_id"] == rid
        assert finish["status"] == "completed"


# ---------------------------------------------------------------------------
# get_runs
# ---------------------------------------------------------------------------


class TestGetRuns:
    def test_returns_newest_first(self, tmp_path):
        io_log.start_run(dataset="A", model="m1", run_id="r1")
        io_log.start_run(dataset="B", model="m2", run_id="r2")

        runs = io_log.get_runs()
        assert len(runs) == 2
        assert runs[0]["run_id"] == "r2"  # newest first
        assert runs[1]["run_id"] == "r1"

    def test_filter_dataset(self, tmp_path):
        io_log.start_run(dataset="HotpotQA", model="m1", run_id="r1")
        io_log.start_run(dataset="MuSiQue", model="m2", run_id="r2")

        runs = io_log.get_runs(dataset="MuSiQue")
        assert len(runs) == 1
        assert runs[0]["dataset"] == "MuSiQue"

    def test_filter_model(self, tmp_path):
        io_log.start_run(dataset="X", model="gpt-5", run_id="r1")
        io_log.start_run(dataset="X", model="o4-mini", run_id="r2")

        runs = io_log.get_runs(model="o4-mini")
        assert len(runs) == 1
        assert runs[0]["model"] == "o4-mini"

    def test_limit(self, tmp_path):
        for i in range(5):
            io_log.start_run(dataset="X", model="Y", run_id=f"r{i}")

        runs = io_log.get_runs(limit=3)
        assert len(runs) == 3

    def test_empty(self, tmp_path):
        runs = io_log.get_runs()
        assert runs == []


# ---------------------------------------------------------------------------
# get_run_items
# ---------------------------------------------------------------------------


class TestGetRunItems:
    def test_returns_items(self, tmp_path):
        rid = io_log.start_run(dataset="X", model="Y")
        io_log.log_item(run_id=rid, item_id="q1", metrics={"em": 1}, predicted="A", gold="A")
        io_log.log_item(run_id=rid, item_id="q2", metrics={"em": 0}, predicted="B", gold="C")

        items = io_log.get_run_items(rid)
        assert len(items) == 2
        assert items[0]["item_id"] == "q1"
        assert items[0]["metrics"]["em"] == 1
        assert items[1]["predicted"] == "B"

    def test_empty_run(self, tmp_path):
        rid = io_log.start_run(dataset="X", model="Y")
        items = io_log.get_run_items(rid)
        assert items == []


# ---------------------------------------------------------------------------
# compare_runs
# ---------------------------------------------------------------------------


class TestCompareRuns:
    def test_compare_two_runs(self, tmp_path):
        r1 = io_log.start_run(dataset="X", model="m1", metrics_schema=["em", "f1"])
        io_log.log_item(run_id=r1, item_id="q1", metrics={"em": 1, "f1": 0.8})
        io_log.log_item(run_id=r1, item_id="q2", metrics={"em": 0, "f1": 0.5})
        io_log.finish_run(run_id=r1)

        r2 = io_log.start_run(dataset="X", model="m2", metrics_schema=["em", "f1"])
        io_log.log_item(run_id=r2, item_id="q1", metrics={"em": 1, "f1": 1.0})
        io_log.log_item(run_id=r2, item_id="q2", metrics={"em": 1, "f1": 0.9})
        io_log.finish_run(run_id=r2)

        result = io_log.compare_runs([r1, r2])
        assert len(result["runs"]) == 2
        assert len(result["deltas_from_first"]) == 1

        # r1: avg_em=50, avg_f1=65
        # r2: avg_em=100, avg_f1=95
        delta = result["deltas_from_first"][0]
        assert delta["avg_em"] == 50.0  # 100 - 50
        assert delta["avg_f1"] == 30.0  # 95 - 65

    def test_compare_requires_two(self, tmp_path):
        r1 = io_log.start_run(dataset="X", model="Y")
        with pytest.raises(ValueError, match="at least 2"):
            io_log.compare_runs([r1])

    def test_compare_missing_run(self, tmp_path):
        r1 = io_log.start_run(dataset="X", model="Y")
        with pytest.raises(ValueError, match="Run not found"):
            io_log.compare_runs([r1, "nonexistent"])


# ---------------------------------------------------------------------------
# Full lifecycle
# ---------------------------------------------------------------------------


class TestFullLifecycle:
    def test_lifecycle(self, tmp_path):
        """Full lifecycle: start → log items → finish → query."""
        rid = io_log.start_run(
            dataset="HotpotQA",
            model="gemini-3-flash",
            config={"backend": "direct", "timeout": 120},
            metrics_schema=["em", "f1", "llm_em"],
            git_commit="abc123",
        )

        # Log 3 items
        io_log.log_item(
            run_id=rid, item_id="q1",
            metrics={"em": 1, "f1": 0.85, "llm_em": 1},
            predicted="Alice", gold="Alice",
            latency_s=5.2, cost=0.01, n_tool_calls=3,
        )
        io_log.log_item(
            run_id=rid, item_id="q2",
            metrics={"em": 0, "f1": 0.5, "llm_em": 1},
            predicted="Bob", gold="Charlie",
            latency_s=8.0, cost=0.02, n_tool_calls=5,
        )
        io_log.log_item(
            run_id=rid, item_id="q3",
            metrics={"em": 1, "f1": 1.0, "llm_em": 1},
            predicted="Dave", gold="Dave",
            latency_s=3.5, cost=0.015, n_tool_calls=2,
            error=None,
            extra={"warnings": ["fallback used"]},
        )

        # Finish
        result = io_log.finish_run(run_id=rid, wall_time_s=45.0)

        assert result["dataset"] == "HotpotQA"
        assert result["model"] == "gemini-3-flash"
        assert result["n_items"] == 3
        assert result["n_completed"] == 3
        assert result["n_errors"] == 0
        assert result["git_commit"] == "abc123"
        assert abs(result["total_cost"] - 0.045) < 0.001

        sm = result["summary_metrics"]
        assert "avg_em" in sm
        assert "avg_f1" in sm
        assert "avg_llm_em" in sm

        # Query
        runs = io_log.get_runs(dataset="HotpotQA")
        assert len(runs) == 1
        assert runs[0]["run_id"] == rid

        items = io_log.get_run_items(rid)
        assert len(items) == 3
