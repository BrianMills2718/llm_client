"""Tests for experiment logging: start_run, log_item, finish_run, queries."""

import json
import os
import sqlite3
import time

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
    old_run_timers = dict(io_log._run_timers)
    active_token = io_log._active_experiment_run_id.set(None)
    feature_profile_token = io_log._active_feature_profile.set(None)
    old_enforcement_mode = os.environ.get("LLM_CLIENT_EXPERIMENT_ENFORCEMENT")
    old_task_patterns = os.environ.get("LLM_CLIENT_EXPERIMENT_TASK_PATTERNS")
    old_feature_profile = os.environ.get("LLM_CLIENT_FEATURE_PROFILE")
    old_feature_enforcement_mode = os.environ.get("LLM_CLIENT_FEATURE_PROFILE_ENFORCEMENT")
    old_feature_task_patterns = os.environ.get("LLM_CLIENT_FEATURE_PROFILE_TASK_PATTERNS")
    old_agent_spec_enforcement_mode = os.environ.get("LLM_CLIENT_AGENT_SPEC_ENFORCEMENT")
    old_agent_spec_task_patterns = os.environ.get("LLM_CLIENT_AGENT_SPEC_TASK_PATTERNS")

    io_log._enabled = True
    io_log._data_root = tmp_path
    io_log._project = "test_project"
    io_log._db_path = tmp_path / "test.db"
    io_log._db_conn = None
    io_log._run_timers.clear()

    yield tmp_path

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
    io_log._active_feature_profile.reset(feature_profile_token)

    if old_enforcement_mode is None:
        os.environ.pop("LLM_CLIENT_EXPERIMENT_ENFORCEMENT", None)
    else:
        os.environ["LLM_CLIENT_EXPERIMENT_ENFORCEMENT"] = old_enforcement_mode
    if old_task_patterns is None:
        os.environ.pop("LLM_CLIENT_EXPERIMENT_TASK_PATTERNS", None)
    else:
        os.environ["LLM_CLIENT_EXPERIMENT_TASK_PATTERNS"] = old_task_patterns
    if old_feature_profile is None:
        os.environ.pop("LLM_CLIENT_FEATURE_PROFILE", None)
    else:
        os.environ["LLM_CLIENT_FEATURE_PROFILE"] = old_feature_profile
    if old_feature_enforcement_mode is None:
        os.environ.pop("LLM_CLIENT_FEATURE_PROFILE_ENFORCEMENT", None)
    else:
        os.environ["LLM_CLIENT_FEATURE_PROFILE_ENFORCEMENT"] = old_feature_enforcement_mode
    if old_feature_task_patterns is None:
        os.environ.pop("LLM_CLIENT_FEATURE_PROFILE_TASK_PATTERNS", None)
    else:
        os.environ["LLM_CLIENT_FEATURE_PROFILE_TASK_PATTERNS"] = old_feature_task_patterns
    if old_agent_spec_enforcement_mode is None:
        os.environ.pop("LLM_CLIENT_AGENT_SPEC_ENFORCEMENT", None)
    else:
        os.environ["LLM_CLIENT_AGENT_SPEC_ENFORCEMENT"] = old_agent_spec_enforcement_mode
    if old_agent_spec_task_patterns is None:
        os.environ.pop("LLM_CLIENT_AGENT_SPEC_TASK_PATTERNS", None)
    else:
        os.environ["LLM_CLIENT_AGENT_SPEC_TASK_PATTERNS"] = old_agent_spec_task_patterns


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
        assert "idx_expr_condition_id" in indexes
        assert "idx_expr_seed" in indexes
        assert "idx_expr_scenario_id" in indexes
        assert "idx_expr_phase" in indexes
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

    def test_writes_cohort_fields(self, tmp_path):
        rid = io_log.start_run(
            dataset="MuSiQue",
            model="o4-mini",
            condition_id="forced_off",
            seed=42,
            replicate=3,
            scenario_id="phase1_matrix",
            phase="phase1",
        )
        db = sqlite3.connect(str(tmp_path / "test.db"))
        row = db.execute(
            "SELECT condition_id, seed, replicate, scenario_id, phase FROM experiment_runs WHERE run_id = ?",
            (rid,),
        ).fetchone()
        assert row == ("forced_off", 42, 3, "phase1_matrix", "phase1")
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

    def test_provenance_explicit(self, tmp_path):
        io_log.start_run(
            dataset="X",
            model="Y",
            run_id="prov_test",
            git_commit="abc123",
            provenance={"prompt_template_sha256": "deadbeef"},
        )
        db = sqlite3.connect(str(tmp_path / "test.db"))
        row = db.execute(
            "SELECT provenance FROM experiment_runs WHERE run_id = 'prov_test'"
        ).fetchone()
        assert row is not None
        prov = json.loads(row[0]) if row[0] else {}
        assert prov["prompt_template_sha256"] == "deadbeef"
        assert prov["git_commit"] == "abc123"
        assert "git_dirty" in prov
        db.close()

    def test_feature_profile_in_provenance(self, tmp_path):
        io_log.start_run(
            dataset="X",
            model="Y",
            run_id="feature_profile_test",
            feature_profile={"name": "benchmark_strict", "features": {"experiment_context": True}},
        )
        db = sqlite3.connect(str(tmp_path / "test.db"))
        row = db.execute(
            "SELECT provenance FROM experiment_runs WHERE run_id = 'feature_profile_test'"
        ).fetchone()
        assert row is not None
        prov = json.loads(row[0]) if row[0] else {}
        assert prov["feature_profile"]["name"] == "benchmark_strict"
        assert prov["feature_profile"]["features"]["experiment_context"] is True
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

    def test_trace_id_persisted(self, tmp_path):
        rid = io_log.start_run(dataset="X", model="Y")
        io_log.log_item(
            run_id=rid,
            item_id="q1",
            metrics={"em": 1},
            trace_id="trace.example.q1",
        )
        db = sqlite3.connect(str(tmp_path / "test.db"))
        row = db.execute(
            "SELECT trace_id FROM experiment_items WHERE run_id = ? AND item_id = 'q1'",
            (rid,),
        ).fetchone()
        assert row[0] == "trace.example.q1"
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

    def test_cpu_fields(self, tmp_path):
        rid = io_log.start_run(dataset="X", model="Y")
        io_log.log_item(run_id=rid, item_id="q1", metrics={"em": 1}, cost=0.01)
        result = io_log.finish_run(
            run_id=rid,
            wall_time_s=10.0,
            cpu_time_s=3.25,
            cpu_user_s=2.5,
            cpu_system_s=0.75,
        )
        assert result["cpu_time_s"] == 3.25
        assert result["cpu_user_s"] == 2.5
        assert result["cpu_system_s"] == 0.75

    def test_auto_captures_timing_when_not_provided(self, tmp_path):
        rid = io_log.start_run(dataset="X", model="Y")
        io_log.log_item(run_id=rid, item_id="q1", metrics={"em": 1}, cost=0.01)
        time.sleep(0.12)
        result = io_log.finish_run(run_id=rid)
        assert result["wall_time_s"] is not None
        assert result["wall_time_s"] >= 0.1
        assert result["cpu_time_s"] is not None
        assert result["cpu_user_s"] is not None
        assert result["cpu_system_s"] is not None
        assert rid not in io_log._run_timers


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

    def test_filter_condition_scenario_phase_seed(self, tmp_path):
        io_log.start_run(
            dataset="X",
            model="m1",
            run_id="r1",
            condition_id="baseline",
            scenario_id="s1",
            phase="phase1",
            seed=10,
        )
        io_log.start_run(
            dataset="X",
            model="m1",
            run_id="r2",
            condition_id="off",
            scenario_id="s1",
            phase="phase1",
            seed=11,
        )

        runs = io_log.get_runs(condition_id="off")
        assert len(runs) == 1
        assert runs[0]["run_id"] == "r2"
        runs = io_log.get_runs(scenario_id="s1", phase="phase1", seed=10)
        assert len(runs) == 1
        assert runs[0]["run_id"] == "r1"

    def test_limit(self, tmp_path):
        for i in range(5):
            io_log.start_run(dataset="X", model="Y", run_id=f"r{i}")

        runs = io_log.get_runs(limit=3)
        assert len(runs) == 3

    def test_empty(self, tmp_path):
        runs = io_log.get_runs()
        assert runs == []


class TestGetRun:
    def test_get_run_by_id(self, tmp_path):
        rid = io_log.start_run(
            dataset="X",
            model="Y",
            run_id="run_123",
            condition_id="baseline",
            seed=101,
            scenario_id="phase1",
            phase="p1",
        )
        row = io_log.get_run(rid)
        assert row is not None
        assert row["run_id"] == "run_123"
        assert row["dataset"] == "X"
        assert row["model"] == "Y"
        assert row["condition_id"] == "baseline"
        assert row["seed"] == 101
        assert row["scenario_id"] == "phase1"
        assert row["phase"] == "p1"

    def test_get_run_missing(self, tmp_path):
        assert io_log.get_run("missing") is None


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
        assert "trace_id" in items[0]

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

    def test_compare_item_deltas(self, tmp_path):
        r1 = io_log.start_run(dataset="X", model="m1", metrics_schema=["em", "llm_em"])
        io_log.log_item(run_id=r1, item_id="q1", metrics={"em": 0, "llm_em": 0})
        io_log.log_item(run_id=r1, item_id="q2", metrics={"em": 1, "llm_em": 1})
        io_log.finish_run(run_id=r1)

        r2 = io_log.start_run(dataset="X", model="m2", metrics_schema=["em", "llm_em"])
        io_log.log_item(run_id=r2, item_id="q1", metrics={"em": 1, "llm_em": 1})
        io_log.log_item(run_id=r2, item_id="q2", metrics={"em": 0, "llm_em": 0})
        io_log.finish_run(run_id=r2)

        result = io_log.compare_runs([r1, r2])
        assert "item_deltas_from_first" in result
        deltas = result["item_deltas_from_first"][0]
        assert deltas["shared_items"] == 2
        assert deltas["improved"]["em"] == ["q1"]
        assert deltas["regressed"]["em"] == ["q2"]


class TestCompareCohorts:
    def test_compare_cohorts_aggregates_and_matches_seed(self, tmp_path):
        # baseline seed 1
        r1 = io_log.start_run(
            dataset="X",
            model="m",
            run_id="base_s1",
            condition_id="baseline",
            scenario_id="scen",
            phase="phase1",
            seed=1,
        )
        io_log.log_item(run_id=r1, item_id="q1", metrics={"em": 1.0})
        io_log.finish_run(run_id=r1, summary_metrics={"score": 10.0})

        # candidate seed 1 (matched with baseline)
        r2 = io_log.start_run(
            dataset="X",
            model="m",
            run_id="off_s1",
            condition_id="off",
            scenario_id="scen",
            phase="phase1",
            seed=1,
        )
        io_log.log_item(run_id=r2, item_id="q1", metrics={"em": 1.0})
        io_log.finish_run(run_id=r2, summary_metrics={"score": 12.0})

        # candidate seed 2 (not matched with baseline)
        r3 = io_log.start_run(
            dataset="X",
            model="m",
            run_id="off_s2",
            condition_id="off",
            scenario_id="scen",
            phase="phase1",
            seed=2,
        )
        io_log.log_item(run_id=r3, item_id="q1", metrics={"em": 1.0})
        io_log.finish_run(run_id=r3, summary_metrics={"score": 9.0})

        result = io_log.compare_cohorts(
            dataset="X",
            scenario_id="scen",
            phase="phase1",
            baseline_condition_id="baseline",
            limit=20,
        )

        assert result["baseline_condition_id"] == "baseline"
        cohorts = {row["condition_id"]: row for row in result["cohorts"]}
        assert "baseline" in cohorts
        assert "off" in cohorts
        assert cohorts["baseline"]["n_runs"] == 1
        assert cohorts["off"]["n_runs"] == 2
        assert "score" in cohorts["off"]["metrics"]
        assert cohorts["off"]["metrics"]["score"]["n"] == 2.0

        deltas = {row["condition_id"]: row for row in result["matched_seed_deltas_from_baseline"]}
        assert deltas["off"]["n_matched_pairs"] == 1
        assert deltas["off"]["metric_deltas"]["score"]["mean"] == 2.0

    def test_compare_cohorts_rejects_missing_baseline(self, tmp_path):
        io_log.start_run(dataset="X", model="m", run_id="r1", condition_id="baseline")
        with pytest.raises(ValueError, match="baseline_condition_id not found"):
            io_log.compare_cohorts(
                dataset="X",
                condition_ids=["baseline"],
                baseline_condition_id="missing",
            )


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


class TestManagedExperimentContext:
    def test_activate_feature_profile_sets_context_temporarily(self, tmp_path):
        profile = {"name": "benchmark_strict", "features": {"experiment_context": True}}
        assert io_log.get_active_feature_profile() is None
        with io_log.activate_feature_profile(profile):
            active = io_log.get_active_feature_profile()
            assert active is not None
            assert active["name"] == "benchmark_strict"
            assert active["features"]["experiment_context"] is True
        assert io_log.get_active_feature_profile() is None

    def test_activate_experiment_run_sets_context_temporarily(self, tmp_path):
        rid = io_log.start_run(dataset="X", model="Y", run_id="ctx_run")
        assert io_log.get_active_experiment_run_id() is None
        with io_log.activate_experiment_run(rid):
            assert io_log.get_active_experiment_run_id() == rid
        assert io_log.get_active_experiment_run_id() is None

    def test_experiment_run_context_auto_finishes(self, tmp_path):
        with io_log.experiment_run(dataset="X", model="Y", run_id="managed_run") as run:
            assert io_log.get_active_experiment_run_id() == "managed_run"
            run.log_item(item_id="q1", metrics={"em": 1}, cost=0.01)

        row = io_log.get_run("managed_run")
        assert row is not None
        assert row["status"] == "completed"
        assert row["n_items"] == 1
        assert io_log.get_active_experiment_run_id() is None


class TestExperimentEnforcement:
    def test_enforcement_warns_without_active_context(self, tmp_path, caplog):
        io_log.configure_experiment_enforcement(mode="warn", task_patterns=["benchmark"])
        with caplog.at_level("WARNING"):
            io_log.enforce_experiment_context("digimon.benchmark", caller="test")
        assert "no active experiment context was found" in caplog.text

    def test_enforcement_errors_without_active_context(self, tmp_path):
        io_log.configure_experiment_enforcement(mode="error", task_patterns=["benchmark"])
        with pytest.raises(ValueError, match="no active experiment context was found"):
            io_log.enforce_experiment_context("digimon.benchmark", caller="test")

    def test_enforcement_allows_when_active_context_present(self, tmp_path):
        io_log.configure_experiment_enforcement(mode="error", task_patterns=["benchmark"])
        rid = io_log.start_run(dataset="X", model="Y")
        with io_log.activate_experiment_run(rid):
            io_log.enforce_experiment_context("digimon.benchmark", caller="test")

    def test_enforcement_ignores_non_matching_task(self, tmp_path, caplog):
        io_log.configure_experiment_enforcement(mode="error", task_patterns=["benchmark"])
        with caplog.at_level("WARNING"):
            io_log.enforce_experiment_context("daily.chat", caller="test")
        assert "no active experiment context" not in caplog.text


class TestFeatureProfileEnforcement:
    def test_profile_enforcement_warns_when_profile_missing(self, tmp_path, caplog):
        io_log.configure_feature_profile(mode="warn", task_patterns=["benchmark"])
        with caplog.at_level("WARNING"):
            io_log.enforce_feature_profile("digimon.benchmark", caller="test")
        assert "no explicit feature profile was declared" in caplog.text

    def test_profile_enforcement_errors_when_profile_missing(self, tmp_path):
        io_log.configure_feature_profile(mode="error", task_patterns=["benchmark"])
        with pytest.raises(ValueError, match="no explicit feature profile was declared"):
            io_log.enforce_feature_profile("digimon.benchmark", caller="test")

    def test_profile_enforcement_allows_when_profile_present(self, tmp_path):
        io_log.configure_feature_profile(mode="error", task_patterns=["benchmark"])
        with io_log.activate_feature_profile("benchmark_strict"):
            with io_log.activate_experiment_run(io_log.start_run(dataset="X", model="Y")):
                io_log.enforce_feature_profile("digimon.benchmark", caller="test")

    def test_profile_experiment_context_requirement(self, tmp_path):
        io_log.configure_feature_profile(mode="error", task_patterns=["benchmark"])
        with io_log.activate_feature_profile({"name": "strict", "features": {"experiment_context": True}}):
            with pytest.raises(ValueError, match="requires experiment_context"):
                io_log.enforce_feature_profile("digimon.benchmark", caller="test")


def _write_agent_spec(path):
    spec = {
        "prompts": {"benchmark": {"path": "prompts/agent.yaml"}},
        "tools": [{"name": "chunk_text_search"}, {"name": "submit_answer"}],
        "artifact_contracts": {
            "chunk_text_search": {"requires_all": ["QUERY_TEXT"], "produces": ["CHUNK_SET"]},
            "submit_answer": {"is_control": True},
        },
        "answer_schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
        "error_taxonomy": {"canonical": ["tool_unavailable", "tool_runtime_error"]},
        "observability": {"required_fields": ["trace_id", "tool_calls"]},
        "evaluation": {"metrics": ["em", "f1", "llm_em"]},
        "gates": {"fail_if": {"tool_unavailable_gt": 0}},
    }
    path.write_text(json.dumps(spec))
    return path


class TestAgentSpecEnforcement:
    def test_agent_spec_enforcement_errors_when_missing(self, tmp_path):
        io_log.configure_agent_spec_enforcement(mode="error", task_patterns=["benchmark"])
        with pytest.raises(ValueError, match="no AgentSpec was declared"):
            io_log.start_run(dataset="X", model="Y", task="digimon.benchmark")

    def test_agent_spec_enforcement_allows_opt_out_with_reason(self, tmp_path):
        io_log.configure_agent_spec_enforcement(mode="error", task_patterns=["benchmark"])
        run_id = io_log.start_run(
            dataset="X",
            model="Y",
            task="digimon.benchmark",
            allow_missing_agent_spec=True,
            missing_agent_spec_reason="legacy project migration in progress",
        )
        row = io_log.get_run(run_id)
        assert row is not None
        prov = row["provenance"]
        assert prov["agent_spec_opt_out"]["enabled"] is True
        assert "migration" in prov["agent_spec_opt_out"]["reason"]

    def test_agent_spec_opt_out_requires_reason(self, tmp_path):
        io_log.configure_agent_spec_enforcement(mode="error", task_patterns=["benchmark"])
        with pytest.raises(ValueError, match="no missing_agent_spec_reason was provided"):
            io_log.start_run(
                dataset="X",
                model="Y",
                task="digimon.benchmark",
                allow_missing_agent_spec=True,
            )

    def test_agent_spec_summary_stored_in_provenance(self, tmp_path):
        io_log.configure_agent_spec_enforcement(mode="error", task_patterns=["benchmark"])
        spec_path = _write_agent_spec(tmp_path / "agent_spec.json")
        run_id = io_log.start_run(
            dataset="X",
            model="Y",
            task="digimon.benchmark",
            agent_spec=spec_path,
        )
        row = io_log.get_run(run_id)
        assert row is not None
        prov = row["provenance"]
        assert prov["agent_spec"]["summary"]["source"].endswith("agent_spec.json")
        assert prov["agent_spec"]["summary"]["tool_count"] == 2
        assert prov["agent_spec"]["summary"]["contract_count"] == 2
