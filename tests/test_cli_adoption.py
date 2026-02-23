from __future__ import annotations

import argparse
import json

import llm_client
import pytest
from llm_client.cli.adoption import cmd_adoption, register_parser


def test_cmd_adoption_json_output(monkeypatch, capsys) -> None:
    def fake_summary(**kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["run_id_prefix"] == "nightly_"
        return {
            "experiments_path": "/tmp/experiments.jsonl",
            "exists": True,
            "total_records": 10,
            "records_considered": 8,
            "invalid_lines": 1,
            "with_reasoning_effort": 4,
            "background_mode_true": 3,
            "background_mode_false": 1,
            "background_mode_unknown": 4,
            "background_mode_rate_among_reasoning": 0.75,
            "background_mode_rate_overall": 0.375,
            "reasoning_effort_counts": {"high": 2, "xhigh": 2},
            "run_id_prefix": "nightly_",
            "since": None,
        }

    monkeypatch.setattr(llm_client, "get_background_mode_adoption", fake_summary)
    args = argparse.Namespace(
        experiments_path=None,
        since=None,
        run_id_prefix="nightly_",
        format="json",
        min_rate=None,
        metric="among_reasoning",
        min_samples=1,
        warn_only=False,
        gate_fail_exit_code=2,
    )
    cmd_adoption(args)

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["background_mode_true"] == 3
    assert payload["reasoning_effort_counts"]["xhigh"] == 2


def test_cmd_adoption_table_output(monkeypatch, capsys) -> None:
    def fake_summary(**kwargs):  # type: ignore[no-untyped-def]
        return {
            "experiments_path": "/tmp/experiments.jsonl",
            "exists": True,
            "total_records": 5,
            "records_considered": 5,
            "invalid_lines": 0,
            "with_reasoning_effort": 2,
            "background_mode_true": 2,
            "background_mode_false": 0,
            "background_mode_unknown": 3,
            "background_mode_rate_among_reasoning": 1.0,
            "background_mode_rate_overall": 0.4,
            "reasoning_effort_counts": {"xhigh": 2},
            "run_id_prefix": None,
            "since": None,
        }

    monkeypatch.setattr(llm_client, "get_background_mode_adoption", fake_summary)
    args = argparse.Namespace(
        experiments_path=None,
        since=None,
        run_id_prefix=None,
        format="table",
        min_rate=None,
        metric="among_reasoning",
        min_samples=1,
        warn_only=False,
        gate_fail_exit_code=2,
    )
    cmd_adoption(args)

    out = capsys.readouterr().out
    assert "Long-Thinking Adoption:" in out
    assert "background=true:" in out
    assert "100.0%" in out
    assert "xhigh: 2" in out


def test_register_parser_sets_handler() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    register_parser(subparsers)
    args = parser.parse_args(["adoption", "--format", "json"])
    assert args.command == "adoption"
    assert callable(args.handler)
    assert args.metric == "among_reasoning"
    assert args.min_samples == 1


def test_cmd_adoption_real_jsonl_since_and_prefix_filters(tmp_path, capsys) -> None:
    experiments = tmp_path / "experiments.jsonl"
    experiments.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "run_id": "nightly_001",
                        "timestamp": "2026-02-23T10:00:00Z",
                        "dimensions": {"reasoning_effort": "xhigh", "background_mode": True},
                    }
                ),
                json.dumps(
                    {
                        "run_id": "nightly_000",
                        "timestamp": "2026-02-22T10:00:00Z",
                        "dimensions": {"reasoning_effort": "high", "background_mode": False},
                    }
                ),
                json.dumps(
                    {
                        "run_id": "other_001",
                        "timestamp": "2026-02-23T10:00:00Z",
                        "dimensions": {"reasoning_effort": "xhigh", "background_mode": True},
                    }
                ),
                "{invalid json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        experiments_path=str(experiments),
        since="2026-02-23T00:00:00Z",
        run_id_prefix="nightly_",
        format="json",
        min_rate=None,
        metric="among_reasoning",
        min_samples=1,
        warn_only=False,
        gate_fail_exit_code=2,
    )
    cmd_adoption(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload["exists"] is True
    assert payload["total_records"] == 4
    assert payload["invalid_lines"] == 1
    assert payload["records_considered"] == 1
    assert payload["with_reasoning_effort"] == 1
    assert payload["background_mode_true"] == 1
    assert payload["background_mode_false"] == 0
    assert payload["background_mode_rate_among_reasoning"] == 1.0
    assert payload["reasoning_effort_counts"] == {"xhigh": 1}
    assert payload["since"] == "2026-02-23T00:00:00+00:00"


def test_cmd_adoption_gate_fail_exits_nonzero(monkeypatch, capsys) -> None:
    def fake_summary(**kwargs):  # type: ignore[no-untyped-def]
        return {
            "experiments_path": "/tmp/experiments.jsonl",
            "exists": True,
            "total_records": 10,
            "records_considered": 10,
            "invalid_lines": 0,
            "with_reasoning_effort": 10,
            "background_mode_true": 4,
            "background_mode_false": 6,
            "background_mode_unknown": 0,
            "background_mode_rate_among_reasoning": 0.4,
            "background_mode_rate_overall": 0.4,
            "reasoning_effort_counts": {"xhigh": 10},
            "run_id_prefix": None,
            "since": None,
        }

    monkeypatch.setattr(llm_client, "get_background_mode_adoption", fake_summary)
    args = argparse.Namespace(
        experiments_path=None,
        since=None,
        run_id_prefix=None,
        format="json",
        min_rate=0.95,
        metric="among_reasoning",
        min_samples=5,
        warn_only=False,
        gate_fail_exit_code=7,
    )

    with pytest.raises(SystemExit) as excinfo:
        cmd_adoption(args)
    assert int(excinfo.value.code) == 7
    payload = json.loads(capsys.readouterr().out)
    assert payload["gate"]["passed"] is False
    assert payload["gate"]["reason"] == "rate_below_threshold"


def test_cmd_adoption_gate_warn_only_does_not_exit(monkeypatch, capsys) -> None:
    def fake_summary(**kwargs):  # type: ignore[no-untyped-def]
        return {
            "experiments_path": "/tmp/experiments.jsonl",
            "exists": True,
            "total_records": 1,
            "records_considered": 1,
            "invalid_lines": 0,
            "with_reasoning_effort": 1,
            "background_mode_true": 0,
            "background_mode_false": 1,
            "background_mode_unknown": 0,
            "background_mode_rate_among_reasoning": 0.0,
            "background_mode_rate_overall": 0.0,
            "reasoning_effort_counts": {"high": 1},
            "run_id_prefix": None,
            "since": None,
        }

    monkeypatch.setattr(llm_client, "get_background_mode_adoption", fake_summary)
    args = argparse.Namespace(
        experiments_path=None,
        since=None,
        run_id_prefix=None,
        format="table",
        min_rate=0.5,
        metric="among_reasoning",
        min_samples=1,
        warn_only=True,
        gate_fail_exit_code=2,
    )
    cmd_adoption(args)

    out = capsys.readouterr().out
    assert "Gate:" in out
    assert "Verdict:          FAIL" in out
