"""Tests for llm_client.scoring — rubric loading, scoring, DB integration."""

import json
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_client import io_log
from llm_client.scoring import (
    CriterionScore,
    Rubric,
    RubricCriterion,
    ScoreResult,
    _JudgeOutput,
    ascore_output,
    list_rubrics,
    load_rubric,
)


@pytest.fixture(autouse=True)
def _isolate_io_log(tmp_path):
    """Isolate io_log state per test."""
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
# Rubric loading
# ---------------------------------------------------------------------------


class TestLoadRubric:
    def test_load_builtin_research_quality(self):
        rubric = load_rubric("research_quality")
        assert rubric.name == "research_quality"
        assert len(rubric.dimensions) == 5
        assert rubric.dimensions[0].name == "completeness"
        assert rubric.dimensions[0].weight == 0.3
        assert rubric.dimensions[0].scale == 5

    def test_load_builtin_extraction_quality(self):
        rubric = load_rubric("extraction_quality")
        assert rubric.name == "extraction_quality"
        assert len(rubric.dimensions) == 4

    def test_load_builtin_with_yaml_extension(self):
        rubric = load_rubric("research_quality.yaml")
        assert rubric.name == "research_quality"

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_rubric("nonexistent_rubric")

    def test_load_from_direct_path(self, tmp_path):
        rubric_file = tmp_path / "custom.yaml"
        rubric_file.write_text(
            "name: custom\nversion: 1\ndimensions:\n"
            "  - name: quality\n    weight: 1.0\n    description: overall\n    scale: 5\n"
        )
        rubric = load_rubric(str(rubric_file))
        assert rubric.name == "custom"
        assert len(rubric.dimensions) == 1

    def test_total_weight(self):
        rubric = load_rubric("research_quality")
        assert abs(rubric.total_weight - 1.0) < 0.01

    def test_all_builtin_rubrics_load(self):
        names = list_rubrics()
        assert len(names) >= 5
        for name in names:
            rubric = load_rubric(name)
            assert rubric.name == name
            assert len(rubric.dimensions) > 0
            for d in rubric.dimensions:
                assert d.weight > 0
                assert d.scale > 0


class TestListRubrics:
    def test_lists_builtin(self):
        names = list_rubrics()
        assert "research_quality" in names
        assert "extraction_quality" in names
        assert "summarization_quality" in names
        assert "analysis_quality" in names
        assert "code_quality" in names


# ---------------------------------------------------------------------------
# Rubric models
# ---------------------------------------------------------------------------


class TestModels:
    def test_rubric_criterion(self):
        c = RubricCriterion(name="test", weight=0.5, description="a test", scale=5)
        assert c.name == "test"
        assert c.scale == 5

    def test_score_result(self):
        sr = ScoreResult(
            rubric="test",
            overall_score=0.75,
            dimensions={"a": 4, "b": 3},
            reasoning={"a": "good", "b": "ok"},
            judge_model="gpt-5-nano",
        )
        assert sr.overall_score == 0.75
        assert sr.dimensions["a"] == 4


# ---------------------------------------------------------------------------
# Scoring (mocked LLM)
# ---------------------------------------------------------------------------


class TestAscoreOutput:
    @pytest.mark.asyncio
    async def test_scores_with_mocked_llm(self, tmp_path):
        """Test full scoring flow with mocked LLM judge call."""
        # mock-ok: Testing scoring logic, not the LLM call itself
        judge_output = _JudgeOutput(
            scores=[
                CriterionScore(criterion="completeness", score=4, reasoning="good coverage"),
                CriterionScore(criterion="accuracy", score=3, reasoning="some unsupported claims"),
                CriterionScore(criterion="source_diversity", score=5, reasoning="many sources"),
                CriterionScore(criterion="actionability", score=3, reasoning="somewhat vague"),
                CriterionScore(criterion="coherence", score=4, reasoning="well organized"),
            ]
        )
        mock_meta = MagicMock(cost=0.001)

        with patch("llm_client.client.acall_llm_structured", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = (judge_output, mock_meta)

            result = await ascore_output(
                output="This is a research report about federal contracts.",
                rubric="research_quality",
                context={"query": "federal contracts"},
                task="test_research",
                trace_id="test_trace_123",
                judge_model="gpt-5-nano",
                output_model="gpt-5-mini",
            )

        assert isinstance(result, ScoreResult)
        assert result.rubric == "research_quality"
        assert 0.0 <= result.overall_score <= 1.0
        assert result.dimensions["completeness"] == 4
        assert result.dimensions["accuracy"] == 3
        assert "completeness" in result.reasoning
        assert result.judge_model == "gpt-5-nano"
        assert result.cost == 0.001

    @pytest.mark.asyncio
    async def test_score_written_to_db(self, tmp_path):
        """Test that scoring writes to the observability DB."""
        # mock-ok: Testing DB write, not the LLM
        judge_output = _JudgeOutput(
            scores=[
                CriterionScore(criterion="completeness", score=4, reasoning="ok"),
                CriterionScore(criterion="accuracy", score=4, reasoning="ok"),
                CriterionScore(criterion="source_diversity", score=4, reasoning="ok"),
                CriterionScore(criterion="actionability", score=4, reasoning="ok"),
                CriterionScore(criterion="coherence", score=4, reasoning="ok"),
            ]
        )
        mock_meta = MagicMock(cost=0.0005)

        with patch("llm_client.client.acall_llm_structured", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = (judge_output, mock_meta)

            await ascore_output(
                output="Test output",
                rubric="research_quality",
                task="db_test",
                trace_id="trace_db",
                judge_model="gpt-5-nano",
            )

        # Check DB
        db = sqlite3.connect(str(tmp_path / "test.db"))
        rows = db.execute("SELECT * FROM task_scores").fetchall()
        assert len(rows) == 1
        row = rows[0]
        # Columns: id, timestamp, project, task, trace_id, rubric, method,
        #          overall_score, dimensions, reasoning, output_model, judge_model,
        #          agent_spec, prompt_id, cost, latency_s
        assert row[3] == "db_test"  # task
        assert row[4] == "trace_db"  # trace_id
        assert row[5] == "research_quality"  # rubric
        assert row[6] == "llm_judge"  # method
        assert row[7] > 0  # overall_score
        db.close()

    @pytest.mark.asyncio
    async def test_score_written_to_jsonl(self, tmp_path):
        """Test that scoring writes to JSONL."""
        # mock-ok: Testing JSONL write, not the LLM
        judge_output = _JudgeOutput(
            scores=[
                CriterionScore(criterion="completeness", score=3, reasoning="ok"),
                CriterionScore(criterion="accuracy", score=3, reasoning="ok"),
                CriterionScore(criterion="source_diversity", score=3, reasoning="ok"),
                CriterionScore(criterion="actionability", score=3, reasoning="ok"),
                CriterionScore(criterion="coherence", score=3, reasoning="ok"),
            ]
        )
        mock_meta = MagicMock(cost=0.0002)

        with patch("llm_client.client.acall_llm_structured", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = (judge_output, mock_meta)

            await ascore_output(
                output="Test",
                rubric="research_quality",
                task="jsonl_test",
                judge_model="gpt-5-nano",
            )

        scores_file = tmp_path / "test_project" / "test_project_llm_client_data" / "scores.jsonl"
        assert scores_file.exists()
        record = json.loads(scores_file.read_text().strip())
        assert record["rubric"] == "research_quality"
        assert record["task"] == "jsonl_test"
        assert record["overall_score"] > 0

    @pytest.mark.asyncio
    async def test_weighted_score_computation(self, tmp_path):
        """Test that overall_score is correctly weighted."""
        # All 5s on all criteria → should be 1.0
        # mock-ok: Testing score computation logic
        judge_output = _JudgeOutput(
            scores=[
                CriterionScore(criterion="completeness", score=5, reasoning="perfect"),
                CriterionScore(criterion="accuracy", score=5, reasoning="perfect"),
                CriterionScore(criterion="source_diversity", score=5, reasoning="perfect"),
                CriterionScore(criterion="actionability", score=5, reasoning="perfect"),
                CriterionScore(criterion="coherence", score=5, reasoning="perfect"),
            ]
        )
        mock_meta = MagicMock(cost=0.0)

        with patch("llm_client.client.acall_llm_structured", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = (judge_output, mock_meta)
            result = await ascore_output("test", "research_quality", judge_model="gpt-5-nano")

        assert result.overall_score == 1.0

    @pytest.mark.asyncio
    async def test_minimum_score(self, tmp_path):
        """All 1s → overall_score should be 0.0."""
        # mock-ok: Testing score computation logic
        judge_output = _JudgeOutput(
            scores=[
                CriterionScore(criterion="completeness", score=1, reasoning="bad"),
                CriterionScore(criterion="accuracy", score=1, reasoning="bad"),
                CriterionScore(criterion="source_diversity", score=1, reasoning="bad"),
                CriterionScore(criterion="actionability", score=1, reasoning="bad"),
                CriterionScore(criterion="coherence", score=1, reasoning="bad"),
            ]
        )
        mock_meta = MagicMock(cost=0.0)

        with patch("llm_client.client.acall_llm_structured", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = (judge_output, mock_meta)
            result = await ascore_output("test", "research_quality", judge_model="gpt-5-nano")

        assert result.overall_score == 0.0

    @pytest.mark.asyncio
    async def test_accepts_rubric_object(self, tmp_path):
        """Test passing a Rubric object directly instead of a name."""
        # mock-ok: Testing API flexibility
        rubric = Rubric(
            name="custom",
            dimensions=[RubricCriterion(name="quality", weight=1.0, description="overall", scale=5)],
        )
        judge_output = _JudgeOutput(
            scores=[CriterionScore(criterion="quality", score=4, reasoning="good")]
        )
        mock_meta = MagicMock(cost=0.0)

        with patch("llm_client.client.acall_llm_structured", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = (judge_output, mock_meta)
            result = await ascore_output("test", rubric, judge_model="gpt-5-nano")

        assert result.rubric == "custom"
        assert result.dimensions["quality"] == 4
        assert result.overall_score == 0.75  # (4-1)/(5-1) = 0.75


# ---------------------------------------------------------------------------
# io_log.log_score
# ---------------------------------------------------------------------------


class TestLogScore:
    def test_writes_to_db(self, tmp_path):
        io_log.log_score(
            rubric="test_rubric",
            method="llm_judge",
            overall_score=0.85,
            dimensions={"a": 4, "b": 5},
            reasoning="test reasoning",
            output_model="gpt-5-mini",
            judge_model="gpt-5-nano",
            task="test_task",
            trace_id="test_trace",
        )

        db = sqlite3.connect(str(tmp_path / "test.db"))
        rows = db.execute("SELECT rubric, overall_score, task FROM task_scores").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "test_rubric"
        assert rows[0][1] == 0.85
        assert rows[0][2] == "test_task"
        db.close()

    def test_writes_jsonl(self, tmp_path):
        io_log.log_score(
            rubric="test_rubric",
            method="llm_judge",
            overall_score=0.5,
            task="jsonl_score_test",
        )

        scores_file = tmp_path / "test_project" / "test_project_llm_client_data" / "scores.jsonl"
        assert scores_file.exists()
        record = json.loads(scores_file.read_text().strip())
        assert record["rubric"] == "test_rubric"
        assert record["overall_score"] == 0.5

    def test_never_raises(self, tmp_path):
        """log_score should never raise, even with bad data."""
        io_log._db_path = Path("/nonexistent/path/db.sqlite")
        io_log._db_conn = None
        # Should not raise
        io_log.log_score(rubric="x", method="y", overall_score=0.5)

    def test_disabled_skips_write(self, tmp_path):
        io_log._enabled = False
        io_log.log_score(rubric="x", method="y", overall_score=0.5, task="skip")

        scores_file = tmp_path / "test_project" / "test_project_llm_client_data" / "scores.jsonl"
        assert not scores_file.exists()


class TestScoreResultGitCommit:
    def test_git_commit_field(self):
        sr = ScoreResult(
            rubric="test",
            overall_score=0.75,
            dimensions={"a": 4},
            git_commit="abc1234",
        )
        assert sr.git_commit == "abc1234"

    def test_git_commit_default_none(self):
        sr = ScoreResult(
            rubric="test",
            overall_score=0.75,
            dimensions={"a": 4},
        )
        assert sr.git_commit is None

    @pytest.mark.asyncio
    async def test_git_commit_passed_through(self, tmp_path):
        """git_commit flows from ascore_output to ScoreResult and DB."""
        # mock-ok: Testing pass-through, not the LLM
        from llm_client.scoring import _JudgeOutput, CriterionScore

        judge_output = _JudgeOutput(
            scores=[
                CriterionScore(criterion="completeness", score=4, reasoning="ok"),
                CriterionScore(criterion="accuracy", score=4, reasoning="ok"),
                CriterionScore(criterion="source_diversity", score=4, reasoning="ok"),
                CriterionScore(criterion="actionability", score=4, reasoning="ok"),
                CriterionScore(criterion="coherence", score=4, reasoning="ok"),
            ]
        )
        mock_meta = MagicMock(cost=0.0)

        with patch("llm_client.client.acall_llm_structured", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = (judge_output, mock_meta)
            result = await ascore_output(
                "test",
                "research_quality",
                judge_model="gpt-5-nano",
                git_commit="xyz7890",
            )

        assert result.git_commit == "xyz7890"

        # Check DB
        db = sqlite3.connect(str(tmp_path / "test.db"))
        row = db.execute("SELECT git_commit FROM task_scores").fetchone()
        assert row[0] == "xyz7890"
        db.close()
