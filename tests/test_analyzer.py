"""Tests for llm_client.analyzer."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_client.analyzer import (
    AnalysisReport,
    IssueCategory,
    Proposal,
    _check_data_quality,
    _check_measurement_error,
    _check_model_overkill,
    _check_model_underkill,
    _check_prompt_drift,
    _check_stuck_loop,
    _check_validation_noise,
    _load_experiments,
    _update_floors,
    analyze_history,
    analyze_scores,
    check_scorer_reliability,
)
from llm_client.task_graph import ExperimentRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_record(
    task_id: str = "test_task",
    difficulty: int = 2,
    outcome: str = "confirmed",
    run_id: str = "run_001",
    model: str = "gemini/gemini-2.5-flash",
    cost: float = 0.01,
    timestamp: str = "2026-02-16T00:00:00Z",
) -> ExperimentRecord:
    return ExperimentRecord(
        run_id=run_id,
        task_id=task_id,
        wave=0,
        timestamp=timestamp,
        hypothesis=f"test at tier {difficulty}",
        difficulty=difficulty,
        model_selected=model,
        agent="codex",
        result={
            "status": "success" if outcome == "confirmed" else "failure",
            "cost_usd": cost,
            "duration_s": 10.0,
            "validation_results": [],
        },
        outcome=outcome,
    )


# ---------------------------------------------------------------------------
# Experiment loading
# ---------------------------------------------------------------------------


def test_load_experiments_empty(tmp_path: Path):
    records = _load_experiments(tmp_path / "nope.jsonl")
    assert records == []


def test_load_experiments(tmp_path: Path):
    log = tmp_path / "experiments.jsonl"
    r = _make_record()
    log.write_text(r.model_dump_json() + "\n")
    records = _load_experiments(log)
    assert len(records) == 1
    assert records[0].task_id == "test_task"


def test_load_experiments_skips_malformed(tmp_path: Path):
    log = tmp_path / "experiments.jsonl"
    r = _make_record()
    log.write_text(r.model_dump_json() + "\n" + "not json\n" + r.model_dump_json() + "\n")
    records = _load_experiments(log)
    assert len(records) == 2


# ---------------------------------------------------------------------------
# Model overkill detection
# ---------------------------------------------------------------------------


def test_overkill_detected():
    """5+ consecutive successes at same tier → propose downgrade."""
    runs = [_make_record(difficulty=2, outcome="confirmed", timestamp=f"2026-02-{i+10}T00:00:00Z") for i in range(6)]
    # mock-ok: we need get_model_for_difficulty to return something for tier 1
    with patch("llm_client.analyzer.get_model_for_difficulty", return_value="deepseek/deepseek-chat"):
        proposal = _check_model_overkill("test_task", runs)
    assert proposal is not None
    assert proposal.category == IssueCategory.MODEL_OVERKILL
    assert proposal.risk == "low"
    assert proposal.auto_apply is True
    assert proposal.evidence["proposed_tier"] == 1


def test_overkill_not_enough_runs():
    runs = [_make_record(difficulty=2, outcome="confirmed") for _ in range(3)]
    proposal = _check_model_overkill("test_task", runs)
    assert proposal is None


def test_overkill_not_all_success():
    runs = [_make_record(difficulty=2, outcome="confirmed") for _ in range(4)]
    runs.append(_make_record(difficulty=2, outcome="hypothesis_rejected"))
    proposal = _check_model_overkill("test_task", runs)
    assert proposal is None


def test_overkill_tier_0_skipped():
    runs = [_make_record(difficulty=0, outcome="confirmed") for _ in range(6)]
    proposal = _check_model_overkill("test_task", runs)
    assert proposal is None


# ---------------------------------------------------------------------------
# Model underkill detection
# ---------------------------------------------------------------------------


def test_underkill_detected():
    runs = [_make_record(difficulty=2, outcome="hypothesis_rejected")]
    with patch("llm_client.analyzer.get_model_for_difficulty", return_value="anthropic/claude-sonnet-4-5-20250929"):
        proposal = _check_model_underkill("test_task", runs)
    assert proposal is not None
    assert proposal.category == IssueCategory.MODEL_UNDERKILL
    assert proposal.risk == "high"
    assert proposal.auto_apply is False
    assert proposal.evidence["proposed_tier"] == 3


def test_underkill_not_on_success():
    runs = [_make_record(difficulty=2, outcome="confirmed")]
    proposal = _check_model_underkill("test_task", runs)
    assert proposal is None


def test_underkill_max_tier():
    runs = [_make_record(difficulty=4, outcome="hypothesis_rejected")]
    proposal = _check_model_underkill("test_task", runs)
    assert proposal is None


# ---------------------------------------------------------------------------
# Stuck loop detection
# ---------------------------------------------------------------------------


def test_stuck_loop_detected():
    runs = [
        _make_record(outcome="confirmed", timestamp="2026-02-10T00:00:00Z"),
        _make_record(outcome="error", timestamp="2026-02-11T00:00:00Z"),
        _make_record(outcome="error", timestamp="2026-02-12T00:00:00Z"),
    ]
    proposal = _check_stuck_loop("test_task", runs)
    assert proposal is not None
    assert proposal.category == IssueCategory.STUCK_LOOP
    assert proposal.evidence["consecutive_errors"] == 2


def test_stuck_loop_single_error_ignored():
    runs = [
        _make_record(outcome="confirmed"),
        _make_record(outcome="error"),
    ]
    proposal = _check_stuck_loop("test_task", runs)
    assert proposal is None


# ---------------------------------------------------------------------------
# Validation noise detection
# ---------------------------------------------------------------------------


def test_validation_noise_detected():
    runs = [
        _make_record(outcome="confirmed", timestamp="2026-02-10T00:00:00Z"),
        _make_record(outcome="hypothesis_rejected", timestamp="2026-02-11T00:00:00Z"),
        _make_record(outcome="confirmed", timestamp="2026-02-12T00:00:00Z"),
        _make_record(outcome="hypothesis_rejected", timestamp="2026-02-13T00:00:00Z"),
    ]
    proposal = _check_validation_noise("test_task", runs)
    assert proposal is not None
    assert proposal.category == IssueCategory.VALIDATION_NOISE


def test_validation_noise_not_enough_runs():
    runs = [_make_record(outcome="confirmed") for _ in range(2)]
    proposal = _check_validation_noise("test_task", runs)
    assert proposal is None


def test_validation_noise_consistent_pass():
    runs = [_make_record(outcome="confirmed", timestamp=f"2026-02-{i+10}T00:00:00Z") for i in range(5)]
    proposal = _check_validation_noise("test_task", runs)
    assert proposal is None


# ---------------------------------------------------------------------------
# Scorer reliability
# ---------------------------------------------------------------------------


def test_scorer_reliability_stable(tmp_path: Path):
    f = tmp_path / "test.txt"
    f.write_text("content")
    result = check_scorer_reliability("test_task", [
        {"type": "file_exists", "path": str(f)},
    ])
    assert result["stable"] is True
    assert result["discrepancies"] == []


def test_scorer_reliability_no_validators():
    result = check_scorer_reliability("test_task", [])
    assert result["stable"] is True


# ---------------------------------------------------------------------------
# Model floors
# ---------------------------------------------------------------------------


def test_update_floors_new_task(tmp_path: Path):
    floors_path = tmp_path / "floors.json"
    experiments = {
        "new_task": [_make_record(task_id="new_task", difficulty=2, outcome="confirmed")],
    }
    floors = _update_floors(experiments, floors_path)
    assert "new_task" in floors
    assert floors["new_task"]["floor"] == 2
    assert floors_path.exists()


def test_update_floors_lowers_on_success(tmp_path: Path):
    floors_path = tmp_path / "floors.json"
    # Pre-existing floor at 3
    floors_path.write_text(json.dumps({"task_a": {"floor": 3, "ceiling": 3, "runs": 5, "last_tested": "2026-02-10"}}))
    experiments = {
        "task_a": [_make_record(task_id="task_a", difficulty=2, outcome="confirmed")],
    }
    floors = _update_floors(experiments, floors_path)
    assert floors["task_a"]["floor"] == 2  # Lowered from 3


def test_update_floors_raises_on_failure(tmp_path: Path):
    floors_path = tmp_path / "floors.json"
    floors_path.write_text(json.dumps({"task_a": {"floor": 1, "ceiling": 3, "runs": 5, "last_tested": "2026-02-10"}}))
    experiments = {
        "task_a": [_make_record(task_id="task_a", difficulty=1, outcome="error")],
    }
    floors = _update_floors(experiments, floors_path)
    assert floors["task_a"]["floor"] == 2  # Raised from 1


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------


def test_analyze_history_empty(tmp_path: Path):
    report = analyze_history(
        experiment_log=tmp_path / "experiments.jsonl",
        proposals_log=tmp_path / "proposals.jsonl",
        floors_path=tmp_path / "floors.json",
    )
    assert report.experiments_analyzed == 0
    assert report.proposals == []


def test_analyze_history_finds_overkill(tmp_path: Path):
    """End-to-end: 6 successes → MODEL_OVERKILL proposal."""
    log = tmp_path / "experiments.jsonl"
    with open(log, "w") as f:
        for i in range(6):
            r = _make_record(difficulty=2, outcome="confirmed", timestamp=f"2026-02-{i+10}T00:00:00Z")
            f.write(r.model_dump_json() + "\n")

    with patch("llm_client.analyzer.get_model_for_difficulty", return_value="deepseek/deepseek-chat"):
        report = analyze_history(
            experiment_log=log,
            proposals_log=tmp_path / "proposals.jsonl",
            floors_path=tmp_path / "floors.json",
        )

    assert report.experiments_analyzed == 6
    assert len(report.proposals) == 1
    assert report.proposals[0].category == IssueCategory.MODEL_OVERKILL

    # Check proposals file was written
    proposals_file = tmp_path / "proposals.jsonl"
    assert proposals_file.exists()
    lines = proposals_file.read_text().strip().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["category"] == "MODEL_OVERKILL"


def test_analyze_history_multiple_proposals(tmp_path: Path):
    """Multiple tasks with different issues."""
    log = tmp_path / "experiments.jsonl"
    with open(log, "w") as f:
        # task_a: 6 successes → overkill
        for i in range(6):
            r = _make_record(task_id="task_a", difficulty=2, outcome="confirmed", timestamp=f"2026-02-{i+10}T00:00:00Z")
            f.write(r.model_dump_json() + "\n")
        # task_b: failed → underkill
        r = _make_record(task_id="task_b", difficulty=2, outcome="hypothesis_rejected")
        f.write(r.model_dump_json() + "\n")

    with patch("llm_client.analyzer.get_model_for_difficulty") as mock_gm:
        mock_gm.return_value = "deepseek/deepseek-chat"
        report = analyze_history(
            experiment_log=log,
            proposals_log=tmp_path / "proposals.jsonl",
            floors_path=tmp_path / "floors.json",
        )

    categories = {p.category for p in report.proposals}
    assert IssueCategory.MODEL_OVERKILL in categories
    assert IssueCategory.MODEL_UNDERKILL in categories


# ---------------------------------------------------------------------------
# Score-based classifiers
# ---------------------------------------------------------------------------


def _make_score(
    task: str = "test_task",
    rubric: str = "research_quality",
    overall_score: float = 0.75,
    git_commit: str | None = "abc1234",
    dimensions: str | None = None,
    timestamp: str = "2026-02-16T00:00:00Z",
) -> dict:
    return {
        "task": task,
        "rubric": rubric,
        "overall_score": overall_score,
        "git_commit": git_commit,
        "dimensions": dimensions or json.dumps({"completeness": 4, "accuracy": 3}),
        "timestamp": timestamp,
    }


# ---------------------------------------------------------------------------
# Prompt drift
# ---------------------------------------------------------------------------


def test_prompt_drift_detected():
    """Score drop + prompt file change → PROMPT_DRIFT."""
    scores = [
        _make_score(overall_score=0.85, git_commit="aaa1111", timestamp=f"2026-02-{i+10}T00:00:00Z")
        for i in range(3)
    ] + [
        _make_score(overall_score=0.60, git_commit="bbb2222", timestamp=f"2026-02-{i+13}T00:00:00Z")
        for i in range(3)
    ]
    # mock-ok: testing classifier logic, not git subprocess
    with patch("llm_client.analyzer.get_diff_files", return_value=["prompts/extract.yaml", "llm_client/client.py"]):
        proposal = _check_prompt_drift("test_task", scores)
    assert proposal is not None
    assert proposal.category == IssueCategory.PROMPT_DRIFT
    assert proposal.action == "review_prompt_change"
    assert proposal.evidence["score_drop"] == 0.25


def test_prompt_drift_no_prompt_change():
    """Score drop but no prompt files in diff → no proposal."""
    scores = [
        _make_score(overall_score=0.85, git_commit="aaa1111", timestamp=f"2026-02-{i+10}T00:00:00Z")
        for i in range(3)
    ] + [
        _make_score(overall_score=0.60, git_commit="bbb2222", timestamp=f"2026-02-{i+13}T00:00:00Z")
        for i in range(3)
    ]
    with patch("llm_client.analyzer.get_diff_files", return_value=["llm_client/client.py"]):
        proposal = _check_prompt_drift("test_task", scores)
    assert proposal is None


def test_prompt_drift_insufficient_scores():
    """Only 1 score per commit → insufficient, no proposal."""
    scores = [
        _make_score(overall_score=0.85, git_commit="aaa1111"),
        _make_score(overall_score=0.60, git_commit="bbb2222"),
    ]
    proposal = _check_prompt_drift("test_task", scores)
    assert proposal is None


# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------


def test_data_quality_low_dimension():
    """3+ consecutive low scores on a dimension → DATA_QUALITY."""
    scores = [
        _make_score(dimensions=json.dumps({"completeness": 2, "accuracy": 4}), timestamp=f"2026-02-{i+10}T00:00:00Z")
        for i in range(4)
    ]
    proposal = _check_data_quality("test_task", scores)
    assert proposal is not None
    assert proposal.category == IssueCategory.DATA_QUALITY
    assert "completeness" in proposal.evidence["weak_dimensions"]


def test_data_quality_above_threshold():
    """All scores above 2 → no proposal."""
    scores = [
        _make_score(dimensions=json.dumps({"completeness": 3, "accuracy": 4}), timestamp=f"2026-02-{i+10}T00:00:00Z")
        for i in range(4)
    ]
    proposal = _check_data_quality("test_task", scores)
    assert proposal is None


# ---------------------------------------------------------------------------
# Measurement error
# ---------------------------------------------------------------------------


def test_measurement_error_high_variance():
    """High variance within same rubric+commit → MEASUREMENT_ERROR."""
    scores = [
        _make_score(overall_score=1.0, git_commit="abc", rubric="r1"),
        _make_score(overall_score=0.0, git_commit="abc", rubric="r1"),
        _make_score(overall_score=1.0, git_commit="abc", rubric="r1"),
    ]
    proposal = _check_measurement_error("test_task", scores)
    assert proposal is not None
    assert proposal.category == IssueCategory.MEASUREMENT_ERROR
    assert proposal.action == "recalibrate_rubric"
    assert proposal.evidence["variance"] > 0.3


def test_measurement_error_stable():
    """Low variance → no proposal."""
    scores = [
        _make_score(overall_score=0.75, git_commit="abc", rubric="r1"),
        _make_score(overall_score=0.76, git_commit="abc", rubric="r1"),
        _make_score(overall_score=0.74, git_commit="abc", rubric="r1"),
    ]
    proposal = _check_measurement_error("test_task", scores)
    assert proposal is None


# ---------------------------------------------------------------------------
# analyze_scores end-to-end
# ---------------------------------------------------------------------------


def _create_scores_db(db_path: Path, scores: list[dict]) -> None:
    """Create a minimal task_scores table and insert rows."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS task_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            project TEXT,
            task TEXT,
            trace_id TEXT,
            rubric TEXT NOT NULL,
            method TEXT NOT NULL,
            overall_score REAL NOT NULL,
            dimensions TEXT,
            reasoning TEXT,
            output_model TEXT,
            judge_model TEXT,
            agent_spec TEXT,
            prompt_id TEXT,
            cost REAL,
            latency_s REAL,
            git_commit TEXT
        )
    """)
    for s in scores:
        conn.execute(
            """INSERT INTO task_scores
               (timestamp, project, task, rubric, method, overall_score, dimensions, git_commit)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                s.get("timestamp", "2026-02-16T00:00:00Z"),
                s.get("project", "test"),
                s.get("task", "test_task"),
                s.get("rubric", "r1"),
                s.get("method", "llm_judge"),
                s["overall_score"],
                s.get("dimensions"),
                s.get("git_commit"),
            ),
        )
    conn.commit()
    conn.close()


def test_analyze_scores_e2e(tmp_path: Path):
    """End-to-end: scores with high variance → MEASUREMENT_ERROR proposal."""
    db = tmp_path / "test.db"
    _create_scores_db(db, [
        {"overall_score": 1.0, "git_commit": "abc", "rubric": "r1", "task": "t1"},
        {"overall_score": 0.0, "git_commit": "abc", "rubric": "r1", "task": "t1"},
        {"overall_score": 1.0, "git_commit": "abc", "rubric": "r1", "task": "t1"},
    ])

    proposals = analyze_scores(
        db_path=db,
        proposals_log=tmp_path / "proposals.jsonl",
    )
    assert len(proposals) >= 1
    assert any(p.category == IssueCategory.MEASUREMENT_ERROR for p in proposals)


def test_analyze_scores_no_db(tmp_path: Path):
    """Non-existent DB → empty list."""
    proposals = analyze_scores(
        db_path=tmp_path / "nonexistent.db",
        proposals_log=tmp_path / "proposals.jsonl",
    )
    assert proposals == []


def test_analyze_history_with_db_path(tmp_path: Path):
    """analyze_history with db_path merges experiment + score proposals."""
    # Create experiment log with overkill data
    exp_log = tmp_path / "experiments.jsonl"
    with open(exp_log, "w") as f:
        for i in range(6):
            r = _make_record(difficulty=2, outcome="confirmed", timestamp=f"2026-02-{i+10}T00:00:00Z")
            f.write(r.model_dump_json() + "\n")

    # Create scores DB with high variance
    db = tmp_path / "test.db"
    _create_scores_db(db, [
        {"overall_score": 1.0, "git_commit": "abc", "rubric": "r1", "task": "score_task"},
        {"overall_score": 0.0, "git_commit": "abc", "rubric": "r1", "task": "score_task"},
        {"overall_score": 1.0, "git_commit": "abc", "rubric": "r1", "task": "score_task"},
    ])

    with patch("llm_client.analyzer.get_model_for_difficulty", return_value="deepseek/deepseek-chat"):
        report = analyze_history(
            experiment_log=exp_log,
            proposals_log=tmp_path / "proposals.jsonl",
            floors_path=tmp_path / "floors.json",
            db_path=db,
        )

    categories = {p.category for p in report.proposals}
    assert IssueCategory.MODEL_OVERKILL in categories
    assert IssueCategory.MEASUREMENT_ERROR in categories
    assert report.score_proposals_count >= 1
