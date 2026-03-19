from __future__ import annotations

import argparse
import json

import llm_client
import pytest
from llm_client.cli.experiments import cmd_experiments


def _detail_args(**overrides) -> argparse.Namespace:
    base = {
        "dataset": None,
        "model": None,
        "project": None,
        "condition_id": None,
        "scenario_id": None,
        "phase": None,
        "seed": None,
        "since": None,
        "limit": 50,
        "compare_cohorts": None,
        "baseline_condition_id": None,
        "compare_diff": None,
        "compare": None,
        "detail": "run_123",
        "include_triage": True,
        "det_checks": "none",
        "review_rubric": None,
        "review_model": None,
        "review_max_items": 0,
        "gate_policy": None,
        "gate_fail_exit_code": False,
        "require_adoption_profile": None,
        "require_adoption_satisfied": False,
        "adoption_gate_fail_exit_code": 4,
        "format": "json",
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_cmd_experiments_detail_includes_adoption_summary(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        llm_client.io_log,
        "get_run",
        lambda run_id: {"run_id": run_id, "dataset": "MuSiQue", "model": "deepseek", "status": "completed"},
    )
    monkeypatch.setattr(
        llm_client.io_log,
        "get_run_items",
        lambda run_id: [
            {
                "item_id": "q1",
                "metrics": {"em": 1.0},
                "predicted": "Alice",
                "gold": "Alice",
                "extra": {
                    "agent": {
                        "submit_completion_mode": "grounded_submit",
                        "adoption_profile_effective": "strict",
                        "adoption_profile_satisfied": True,
                        "adoption_profile_violations": [],
                    }
                },
            },
            {
                "item_id": "q2",
                "metrics": {"em": 0.0},
                "predicted": "Bob",
                "gold": "Carol",
                "extra": {
                    "agent": {
                        "submit_completion_mode": "forced_terminal_accept",
                        "adoption_profile_effective": "strict",
                        "adoption_profile_satisfied": True,
                        "adoption_profile_violations": [],
                    }
                },
            },
        ],
    )

    cmd_experiments(_detail_args())
    payload = json.loads(capsys.readouterr().out)
    assert payload["adoption"]["effective_profile_counts"] == {"strict": 2}
    assert payload["adoption"]["satisfied_count"] == 2


def test_cmd_experiments_detail_adoption_gate_exits(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        llm_client.io_log,
        "get_run",
        lambda run_id: {"run_id": run_id, "dataset": "MuSiQue", "model": "deepseek", "status": "completed"},
    )
    monkeypatch.setattr(
        llm_client.io_log,
        "get_run_items",
        lambda run_id: [
            {
                "item_id": "q1",
                "metrics": {"em": 0.0},
                "predicted": "Alice",
                "gold": "Alice",
                "extra": {
                    "agent": {
                        "submit_completion_mode": "grounded_submit",
                        "adoption_profile_effective": "standard",
                        "adoption_profile_satisfied": False,
                        "adoption_profile_violations": ["require_tool_reasoning must be enabled"],
                    }
                },
            }
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        cmd_experiments(
            _detail_args(
                require_adoption_profile="strict",
                require_adoption_satisfied=True,
                adoption_gate_fail_exit_code=9,
            )
        )
    assert int(excinfo.value.code) == 9
    payload = json.loads(capsys.readouterr().out)
    assert payload["adoption"]["effective_profile_counts"] == {"standard": 1}
