"""Tests for governed-repo friction telemetry import and query helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from llm_client import io_log
from llm_client.observability.experiments import finish_run, start_run
from llm_client.observability.query import (
    get_governed_repo_friction_summary,
    get_governed_repo_top_missing_reads,
    get_governed_repo_variant_comparison,
    import_governed_repo_hook_log,
)


@pytest.fixture(autouse=True)
def _isolate_io_log(tmp_path: Path) -> Path:
    """Isolate shared observability state so governed-repo imports stay deterministic."""
    old_enabled = io_log._enabled
    old_root = io_log._data_root
    old_project = io_log._project
    old_db_path = io_log._db_path
    old_db_conn = io_log._db_conn
    old_last_cleanup = io_log._last_cleanup_date

    io_log._enabled = True
    io_log._data_root = tmp_path / "data"
    io_log._project = "llm_client_test"
    io_log._db_path = tmp_path / "observability.db"
    io_log._db_conn = None
    io_log._last_cleanup_date = None

    yield tmp_path

    if io_log._db_conn is not None:
        io_log._db_conn.close()
    io_log._enabled = old_enabled
    io_log._data_root = old_root
    io_log._project = old_project
    io_log._db_path = old_db_path
    io_log._db_conn = old_db_conn
    io_log._last_cleanup_date = old_last_cleanup


def _write_hook_log(repo_root: Path, rows: list[dict[str, object]]) -> Path:
    """Write one canonical hook log for a synthetic governed repo."""
    log_path = repo_root / ".claude" / "hook_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return log_path


def _timestamp(offset_seconds: int = 0) -> str:
    """Return a stable current UTC timestamp shifted by the given seconds."""
    now = datetime.now(timezone.utc).replace(microsecond=0)
    shifted = now.timestamp() + float(offset_seconds)
    return datetime.fromtimestamp(shifted, tz=timezone.utc).isoformat()


def test_import_hook_log_preserves_repo_file_and_missing_reads(tmp_path: Path) -> None:
    repo_root = tmp_path / "sample_repo"
    log_path = _write_hook_log(
        repo_root,
        [
            {
                "schema_version": 1,
                "timestamp": _timestamp(),
                "hook": "gate-edit",
                "tool_name": "Edit",
                "file_path": "src/sample_repo/cli.py",
                "decision": "block",
                "decision_reason": "missing required reads",
                "reads_file": ".claude/session_reads.txt",
                "required_reads": ["CLAUDE.md", "docs/plans/02.md"],
                "reads_completed": ["CLAUDE.md"],
                "missing_reads": ["docs/plans/02.md"],
                "coupled_docs": ["docs/plans/02.md"],
                "context_emitted": False,
                "context_bytes": 0,
            },
            {
                "schema_version": 1,
                "timestamp": _timestamp(1),
                "hook": "track-reads",
                "tool_name": "Read",
                "file_path": "CLAUDE.md",
                "decision": "recorded",
                "decision_reason": "read observed",
                "reads_file": ".claude/session_reads.txt",
            },
        ],
    )

    assert import_governed_repo_hook_log(log_path) == 2
    assert import_governed_repo_hook_log(log_path) == 0

    row = io_log._get_db().execute(
        "SELECT project, payload FROM foundation_events ORDER BY timestamp ASC LIMIT 1"
    ).fetchone()
    assert row is not None
    assert row[0] == "sample_repo"
    payload = json.loads(row[1])
    hook_payload = payload["governed_repo_hook"]
    assert hook_payload["file_path"] == "src/sample_repo/cli.py"
    assert hook_payload["missing_reads"] == ["docs/plans/02.md"]
    assert hook_payload["repo_name"] == "sample_repo"
    assert hook_payload["reads_file"] == ".claude/session_reads.txt"
    assert hook_payload["session_quality"] == "stable"
    assert hook_payload["session_source"] == "reads_file:.claude/session_reads.txt"


def test_import_hook_log_fails_loud_on_malformed_rows(tmp_path: Path) -> None:
    repo_root = tmp_path / "broken_repo"
    log_path = repo_root / ".claude" / "hook_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "timestamp": _timestamp(),
                "hook": "gate-edit",
                "file_path": "src/broken_repo/cli.py",
                "decision": "block",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid hook row"):
        import_governed_repo_hook_log(log_path)


def test_query_governed_repo_friction_summary_reports_block_allow_error_counts(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "friction_repo"
    log_path = _write_hook_log(
        repo_root,
        [
            {
                "schema_version": 1,
                "timestamp": _timestamp(),
                "hook": "gate-edit",
                "tool_name": "Edit",
                "file_path": "src/friction_repo/a.py",
                "decision": "block",
                "decision_reason": "missing required reads",
                "required_reads": ["CLAUDE.md"],
                "reads_completed": [],
                "missing_reads": ["CLAUDE.md"],
                "coupled_docs": [],
                "context_emitted": False,
                "context_bytes": 0,
            },
            {
                "schema_version": 1,
                "timestamp": _timestamp(1),
                "hook": "gate-edit",
                "tool_name": "Edit",
                "file_path": "src/friction_repo/a.py",
                "decision": "allow",
                "decision_reason": "requirements satisfied",
                "required_reads": ["CLAUDE.md"],
                "reads_completed": ["CLAUDE.md"],
                "missing_reads": [],
                "coupled_docs": [],
                "context_emitted": True,
                "context_bytes": 256,
            },
            {
                "schema_version": 1,
                "timestamp": _timestamp(2),
                "hook": "gate-edit",
                "tool_name": "Write",
                "file_path": "src/friction_repo/b.py",
                "decision": "error",
                "decision_reason": "hook failure",
                "required_reads": ["CLAUDE.md"],
                "reads_completed": [],
                "missing_reads": ["CLAUDE.md"],
                "coupled_docs": [],
                "context_emitted": False,
                "context_bytes": 0,
            },
            {
                "schema_version": 1,
                "timestamp": _timestamp(3),
                "hook": "track-reads",
                "tool_name": "Read",
                "file_path": "CLAUDE.md",
                "decision": "recorded",
                "decision_reason": "read observed",
                "reads_file": ".claude/session_reads.txt",
            },
        ],
    )
    assert import_governed_repo_hook_log(log_path) == 4

    summary = get_governed_repo_friction_summary(repo_name="friction_repo", days=30, limit=10)
    assert summary["total_events"] == 4
    assert summary["decision_counts"] == {
        "allow": 1,
        "block": 1,
        "error": 1,
        "recorded": 1,
    }
    assert summary["hook_counts"] == {"gate-edit": 3, "track-reads": 1}
    assert summary["session_quality_counts"] == {"degraded": 3, "stable": 1}
    assert summary["context_emitted_events"] == 1
    assert summary["context_emitted_rate"] == 1.0 / 3.0
    assert summary["context_bytes_total"] == 256
    assert summary["top_friction_sessions"] == []
    assert summary["top_friction_files"][0] == {"path": "src/friction_repo/a.py", "count": 1} or summary[
        "top_friction_files"
    ][0] == {"path": "src/friction_repo/b.py", "count": 1}
    assert {item["path"] for item in summary["top_friction_files"]} == {
        "src/friction_repo/a.py",
        "src/friction_repo/b.py",
    }


def test_query_governed_repo_top_missing_reads_ranks_repeated_gaps(tmp_path: Path) -> None:
    repo_root = tmp_path / "gap_repo"
    log_path = _write_hook_log(
        repo_root,
        [
            {
                "schema_version": 1,
                "timestamp": _timestamp(),
                "hook": "gate-edit",
                "tool_name": "Edit",
                "file_path": "src/gap_repo/a.py",
                "decision": "block",
                "decision_reason": "missing required reads",
                "required_reads": ["CLAUDE.md", "docs/plans/08.md"],
                "reads_completed": [],
                "missing_reads": ["CLAUDE.md", "docs/plans/08.md"],
                "coupled_docs": [],
            },
            {
                "schema_version": 1,
                "timestamp": _timestamp(1),
                "hook": "gate-edit",
                "tool_name": "Edit",
                "file_path": "src/gap_repo/b.py",
                "decision": "block",
                "decision_reason": "missing required reads",
                "required_reads": ["CLAUDE.md"],
                "reads_completed": [],
                "missing_reads": ["CLAUDE.md"],
                "coupled_docs": [],
            },
        ],
    )
    assert import_governed_repo_hook_log(log_path) == 2

    ranked = get_governed_repo_top_missing_reads(repo_name="gap_repo", days=30, limit=10)
    assert ranked[0] == {"path": "CLAUDE.md", "count": 2}
    assert ranked[1] == {"path": "docs/plans/08.md", "count": 1}


def test_import_hook_log_records_experiment_identity_and_variant(tmp_path: Path) -> None:
    repo_root = tmp_path / "experiment_repo"
    log_path = _write_hook_log(
        repo_root,
        [
            {
                "schema_version": 1,
                "timestamp": _timestamp(),
                "hook": "gate-edit",
                "tool_name": "Edit",
                "file_path": "src/experiment_repo/cli.py",
                "decision": "block",
                "decision_reason": "missing required reads",
                "required_reads": ["CLAUDE.md"],
                "reads_completed": [],
                "missing_reads": ["CLAUDE.md"],
                "coupled_docs": [],
            }
        ],
    )

    assert (
        import_governed_repo_hook_log(
            log_path,
            experiment_id="ctx-exp-1",
            variant_id="rich-context",
            downstream_run_id="run_ctx_eval_1",
        )
        == 1
    )

    row = io_log._get_db().execute(
        "SELECT trace_id, payload FROM foundation_events ORDER BY timestamp ASC LIMIT 1"
    ).fetchone()
    assert row is not None
    assert row[0] == "governed_repo/experiment_repo/ctx-exp-1/rich-context"
    payload = json.loads(row[1])
    hook_payload = payload["governed_repo_hook"]
    assert hook_payload["experiment_id"] == "ctx-exp-1"
    assert hook_payload["variant_id"] == "rich-context"
    assert hook_payload["downstream_run_id"] == "run_ctx_eval_1"


def test_import_hook_log_preserves_source_stamped_experiment_metadata(tmp_path: Path) -> None:
    repo_root = tmp_path / "source_tagged_repo"
    log_path = _write_hook_log(
        repo_root,
        [
            {
                "schema_version": 1,
                "timestamp": _timestamp(),
                "hook": "gate-edit",
                "tool_name": "Edit",
                "file_path": "src/source_tagged_repo/cli.py",
                "decision": "block",
                "decision_reason": "missing required reads",
                "reads_file": ".claude/session_reads.txt",
                "required_reads": ["CLAUDE.md"],
                "reads_completed": [],
                "missing_reads": ["CLAUDE.md"],
                "coupled_docs": [],
                "context_emitted": False,
                "context_bytes": 0,
                "experiment_id": "ctx-exp-source",
                "variant_id": "short-context",
                "downstream_run_id": "run_ctx_source_1",
            }
        ],
    )

    assert import_governed_repo_hook_log(log_path) == 1

    row = io_log._get_db().execute(
        "SELECT trace_id, payload FROM foundation_events ORDER BY timestamp ASC LIMIT 1"
    ).fetchone()
    assert row is not None
    assert row[0] == "governed_repo/source_tagged_repo/ctx-exp-source/short-context"
    payload = json.loads(row[1])
    hook_payload = payload["governed_repo_hook"]
    assert hook_payload["experiment_id"] == "ctx-exp-source"
    assert hook_payload["variant_id"] == "short-context"
    assert hook_payload["downstream_run_id"] == "run_ctx_source_1"


def test_query_governed_repo_variant_comparison_summarizes_friction_metrics(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "variant_repo"
    short_log_path = _write_hook_log(
        repo_root,
        [
            {
                "schema_version": 1,
                "timestamp": _timestamp(),
                "hook": "gate-edit",
                "tool_name": "Edit",
                "file_path": "src/variant_repo/a.py",
                "decision": "block",
                "decision_reason": "missing required reads",
                "reads_file": ".claude/session_short.txt",
                "required_reads": ["CLAUDE.md"],
                "reads_completed": [],
                "missing_reads": ["CLAUDE.md"],
                "coupled_docs": [],
                "context_emitted": False,
                "context_bytes": 0,
            },
            {
                "schema_version": 1,
                "timestamp": _timestamp(1),
                "hook": "gate-edit",
                "tool_name": "Edit",
                "file_path": "src/variant_repo/a.py",
                "decision": "block",
                "decision_reason": "missing required reads",
                "reads_file": ".claude/session_short.txt",
                "required_reads": ["CLAUDE.md"],
                "reads_completed": [],
                "missing_reads": ["CLAUDE.md"],
                "coupled_docs": [],
                "context_emitted": False,
                "context_bytes": 0,
            },
        ],
    )
    rich_log_path = _write_hook_log(
        tmp_path / "variant_repo_rich",
        [
            {
                "schema_version": 1,
                "timestamp": _timestamp(2),
                "hook": "gate-edit",
                "tool_name": "Edit",
                "file_path": "src/variant_repo/b.py",
                "decision": "allow",
                "decision_reason": "requirements satisfied",
                "reads_file": ".claude/session_rich.txt",
                "required_reads": ["CLAUDE.md"],
                "reads_completed": ["CLAUDE.md"],
                "missing_reads": [],
                "coupled_docs": [],
                "context_emitted": True,
                "context_bytes": 128,
            },
            {
                "schema_version": 1,
                "timestamp": _timestamp(3),
                "hook": "gate-edit",
                "tool_name": "Write",
                "file_path": "src/variant_repo/c.py",
                "decision": "error",
                "decision_reason": "hook failure",
                "reads_file": ".claude/session_rich.txt",
                "required_reads": ["CLAUDE.md"],
                "reads_completed": [],
                "missing_reads": ["docs/plans/08.md"],
                "coupled_docs": [],
                "context_emitted": False,
                "context_bytes": 0,
            },
        ],
    )

    assert import_governed_repo_hook_log(
        short_log_path,
        repo_name="variant_repo",
        experiment_id="ctx-exp-compare",
        variant_id="short-context",
    ) == 2
    assert import_governed_repo_hook_log(
        rich_log_path,
        repo_name="variant_repo",
        experiment_id="ctx-exp-compare",
        variant_id="rich-context",
    ) == 2

    comparison = get_governed_repo_variant_comparison(
        experiment_id="ctx-exp-compare",
        repo_name="variant_repo",
        days=30,
    )
    by_variant = {entry["variant_id"]: entry for entry in comparison["variants"]}

    short_summary = by_variant["short-context"]
    assert short_summary["block_events"] == 2
    assert short_summary["repeated_block_events"] == 1
    assert short_summary["repeated_block_rate"] == 0.5
    assert short_summary["repeated_missing_read_events"] == 1
    assert short_summary["repeated_missing_read_rate"] == 0.5
    assert short_summary["hook_error_rate"] == 0.0
    assert short_summary["session_quality_counts"] == {"stable": 2}
    assert short_summary["context_emitted_rate"] == 0.0

    rich_summary = by_variant["rich-context"]
    assert rich_summary["block_events"] == 0
    assert rich_summary["hook_error_events"] == 1
    assert rich_summary["hook_error_rate"] == 0.5
    assert rich_summary["repeated_missing_read_rate"] == 0.0
    assert rich_summary["session_quality_counts"] == {"stable": 2}
    assert rich_summary["context_emitted_events"] == 1
    assert rich_summary["context_emitted_rate"] == 0.5
    assert rich_summary["context_bytes_total"] == 128


def test_query_governed_repo_variant_comparison_joins_downstream_run_outcomes(
    tmp_path: Path,
) -> None:
    completed_run_id = start_run(
        dataset="ctx-exp",
        model="gpt-test",
        task="experiment",
        allow_missing_agent_spec=True,
        missing_agent_spec_reason="synthetic Plan 11 observability test",
    )
    finish_run(run_id=completed_run_id, status="completed")
    failed_run_id = start_run(
        dataset="ctx-exp",
        model="gpt-test",
        task="experiment",
        allow_missing_agent_spec=True,
        missing_agent_spec_reason="synthetic Plan 11 observability test",
    )
    finish_run(run_id=failed_run_id, status="failed")

    repo_root = tmp_path / "outcome_repo"
    control_log_path = _write_hook_log(
        repo_root,
        [
            {
                "schema_version": 1,
                "timestamp": _timestamp(),
                "hook": "gate-edit",
                "tool_name": "Edit",
                "file_path": "src/outcome_repo/a.py",
                "decision": "allow",
                "decision_reason": "requirements satisfied",
                "required_reads": ["CLAUDE.md"],
                "reads_completed": ["CLAUDE.md"],
                "missing_reads": [],
                "coupled_docs": [],
            }
        ],
    )
    variant_log_path = _write_hook_log(
        tmp_path / "outcome_repo_variant",
        [
            {
                "schema_version": 1,
                "timestamp": _timestamp(1),
                "hook": "gate-edit",
                "tool_name": "Edit",
                "file_path": "src/outcome_repo/b.py",
                "decision": "block",
                "decision_reason": "missing required reads",
                "required_reads": ["CLAUDE.md"],
                "reads_completed": [],
                "missing_reads": ["CLAUDE.md"],
                "coupled_docs": [],
            }
        ],
    )

    assert import_governed_repo_hook_log(
        control_log_path,
        repo_name="outcome_repo",
        experiment_id="ctx-exp-outcomes",
        variant_id="control",
        downstream_run_id=completed_run_id,
    ) == 1
    assert import_governed_repo_hook_log(
        variant_log_path,
        repo_name="outcome_repo",
        experiment_id="ctx-exp-outcomes",
        variant_id="variant",
        downstream_run_id=failed_run_id,
    ) == 1

    comparison = get_governed_repo_variant_comparison(
        experiment_id="ctx-exp-outcomes",
        repo_name="outcome_repo",
        days=30,
    )
    by_variant = {entry["variant_id"]: entry for entry in comparison["variants"]}

    assert by_variant["control"]["downstream_runs"]["completed_runs"] == 1
    assert by_variant["control"]["downstream_runs"]["terminal_runs"] == 1
    assert by_variant["control"]["downstream_runs"]["status_counts"] == {"completed": 1}

    assert by_variant["variant"]["downstream_runs"]["completed_runs"] == 0
    assert by_variant["variant"]["downstream_runs"]["terminal_runs"] == 1
    assert by_variant["variant"]["downstream_runs"]["status_counts"] == {"failed": 1}
