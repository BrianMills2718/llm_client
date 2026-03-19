"""Tests for llm_client.experiment_eval."""

from __future__ import annotations

import json

from llm_client.experiment_eval import (
    build_gate_signals,
    evaluate_gate_policy,
    extract_adoption_profile,
    extract_agent_outcome,
    load_gate_policy,
    review_items_with_rubric,
    run_deterministic_checks_for_item,
    run_deterministic_checks_for_items,
    summarize_adoption_profiles,
    summarize_agent_outcomes,
    triage_items,
)


def _sample_items() -> list[dict]:
    return [
        {
            "item_id": "q1",
            "metrics": {"em": 1, "f1": 1.0, "llm_em": 1},
            "predicted": "Alice",
            "gold": "Alice",
            "error": None,
            "trace_id": "trace.q1",
            "submit_completion_mode": "grounded_submit",
            "submit_validator_accepted": True,
            "extra": {"composability": {"n_errors": 0, "error_categories": {}}},
        },
        {
            "item_id": "q2",
            "metrics": {"em": 0, "f1": 0.0, "llm_em": 0},
            "predicted": "",
            "gold": "Bob",
            "error": "timeout",
            "trace_id": "",
            "submit_completion_mode": "missing_required_submit",
            "required_submit_missing": True,
            "extra": {
                "composability": {
                    "n_errors": 3,
                    "n_control_loop_suppressed": 1,
                    "error_categories": {
                        "tool_interface_mismatch": 2,
                        "tool_unavailable": 1,
                    },
                }
            },
        },
    ]


def test_run_deterministic_checks_for_item_flags_missing_prediction():
    item = _sample_items()[1]
    results = run_deterministic_checks_for_item(item, checks=["prediction_present", "trace_id_present"])
    assert len(results) == 2
    assert results[0]["name"] == "prediction_present"
    assert results[0]["passed"] is False
    assert results[1]["name"] == "trace_id_present"
    assert results[1]["passed"] is False


def test_run_deterministic_checks_for_items_summary():
    report = run_deterministic_checks_for_items(_sample_items(), checks="prediction_present,no_item_error")
    assert report["n_items"] == 2
    assert report["total_checks"] == 4
    assert report["n_failed_items"] == 1
    assert report["pass_rate"] == 0.5


def test_load_gate_policy_inline_and_file(tmp_path):
    inline = load_gate_policy('{"pass_if": {"avg_llm_em_gte": 80}}')
    assert inline["pass_if"]["avg_llm_em_gte"] == 80

    path = tmp_path / "gate.json"
    path.write_text(json.dumps({"fail_if": {"n_errors_gt": 0}}))
    from_file = load_gate_policy(str(path))
    assert from_file["fail_if"]["n_errors_gt"] == 0


def test_gate_policy_evaluation_pass_and_fail():
    signals = {"avg_llm_em": 82.0, "n_errors": 0.0}
    policy = {"pass_if": {"avg_llm_em_gte": 80, "n_errors_eq": 0}}
    report = evaluate_gate_policy(policy=policy, signals=signals)
    assert report["passed"] is True

    bad = {"fail_if": {"avg_llm_em_lt": 90}}
    failed = evaluate_gate_policy(policy=bad, signals=signals)
    assert failed["passed"] is False
    assert len(failed["triggered_fail_if"]) == 1


def test_build_gate_signals_includes_summary_and_item_error_buckets():
    run_info = {
        "n_items": 2,
        "n_errors": 1,
        "summary_metrics": {"avg_em": 50.0, "avg_llm_em": 50.0},
    }
    signals = build_gate_signals(run_info=run_info, items=_sample_items())
    assert signals["n_items"] == 2.0
    assert signals["avg_llm_em"] == 50.0
    assert signals["item_error_count"] == 1.0
    assert signals["tool_interface_error_total"] == 2.0
    assert signals["tool_unavailable_error_total"] == 1.0
    assert signals["submit_mode_grounded_submit_count"] == 1.0
    assert signals["submit_mode_missing_required_submit_count"] == 1.0
    assert signals["grounded_completed_count"] == 1.0
    assert signals["required_submit_missing_count"] == 1.0
    assert signals["primary_failure_unknown_count"] == 2.0


def test_review_items_with_rubric_aggregates(monkeypatch):
    class _FakeScore:
        def __init__(self, score: float):
            self.overall_score = score
            self.dimensions = {"quality": 4}
            self.reasoning = {"quality": "ok"}
            self.cost = 0.001
            self.latency_s = 0.2

    def _fake_score_output(*args, **kwargs):
        item_task = kwargs.get("task", "")
        return _FakeScore(0.9 if item_task.endswith(".q1") else 0.4)

    monkeypatch.setattr("llm_client.experiment_eval.score_output", _fake_score_output)
    report = review_items_with_rubric(_sample_items(), rubric="research_quality")
    assert report["n_items_considered"] == 2
    assert report["n_scored"] == 2
    assert report["n_failed"] == 0
    assert report["avg_overall_score"] == 0.65
    assert report["min_overall_score"] == 0.4
    assert report["max_overall_score"] == 0.9


def test_triage_items_detects_multiple_error_classes():
    report = triage_items(_sample_items())
    assert report["n_items"] == 2
    cats = report["category_counts"]
    assert cats["runtime_error"] == 1
    assert cats["tool_interface"] == 1
    assert cats["tool_unavailable"] == 1
    assert cats["control_loop"] == 1
    assert cats["grounded_completion"] == 1
    assert cats["required_submit_missing"] == 1


def test_extract_agent_outcome_derives_fields_from_submit_mode_and_prediction():
    item = {
        "id": "q3",
        "predicted": "Alice",
        "submit_completion_mode": "forced_terminal_accept",
        "primary_failure_class": "control_churn",
        "first_terminal_failure_event_code": "SUBMIT_FORCED_ACCEPT_TURN_EXHAUSTION",
    }
    outcome = extract_agent_outcome(item)
    assert outcome["answer_present"] is True
    assert outcome["forced_terminal_accepted"] is True
    assert outcome["grounded_completed"] is False
    assert outcome["reliability_completed"] is True
    assert outcome["primary_failure_class"] == "control_churn"
    assert outcome["first_terminal_failure_event_code"] == "SUBMIT_FORCED_ACCEPT_TURN_EXHAUSTION"


def test_summarize_agent_outcomes_counts_completion_modes_and_failure_classes():
    items = [
        {
            "item_id": "q1",
            "predicted": "Alice",
            "submit_completion_mode": "grounded_submit",
            "submit_validator_accepted": True,
            "primary_failure_class": "none",
        },
        {
            "item_id": "q2",
            "predicted": "Bob",
            "submit_completion_mode": "forced_terminal_accept",
            "primary_failure_class": "control_churn",
        },
        {
            "item_id": "q3",
            "predicted": "",
            "submit_completion_mode": "missing_required_submit",
            "required_submit_missing": True,
            "primary_failure_class": "policy",
        },
    ]
    summary = summarize_agent_outcomes(items)
    assert summary["n_items"] == 3
    assert summary["answer_present_count"] == 2
    assert summary["grounded_completed_count"] == 1
    assert summary["forced_terminal_accepted_count"] == 1
    assert summary["required_submit_missing_count"] == 1
    assert summary["submit_completion_mode_counts"]["grounded_submit"] == 1
    assert summary["submit_completion_mode_counts"]["forced_terminal_accept"] == 1
    assert summary["primary_failure_class_counts"]["control_churn"] == 1
    assert summary["first_terminal_failure_event_code_counts"]["none"] == 3


def test_extract_adoption_profile_reads_nested_agent_metadata():
    item = {
        "item_id": "q4",
        "extra": {
            "agent": {
                "adoption_profile_requested": "strict",
                "adoption_profile_effective": "strict",
                "adoption_profile_satisfied": True,
                "adoption_profile_violations": [],
            }
        },
    }
    adoption = extract_adoption_profile(item)
    assert adoption["requested_profile"] == "strict"
    assert adoption["effective_profile"] == "strict"
    assert adoption["satisfied"] is True
    assert adoption["has_metadata"] is True


def test_summarize_adoption_profiles_counts_profiles_and_coverage():
    items = [
        {
            "item_id": "q1",
            "extra": {
                "agent": {
                    "adoption_profile_effective": "strict",
                    "adoption_profile_satisfied": True,
                    "adoption_profile_violations": [],
                }
            },
        },
        {
            "item_id": "q2",
            "extra": {
                "agent": {
                    "adoption_profile_effective": "standard",
                    "adoption_profile_satisfied": False,
                    "adoption_profile_violations": ["require_tool_reasoning must be enabled"],
                }
            },
        },
        {"item_id": "q3", "extra": {}},
    ]
    summary = summarize_adoption_profiles(items)
    assert summary["n_items"] == 3
    assert summary["n_items_with_metadata"] == 2
    assert summary["satisfied_count"] == 1
    assert summary["effective_profile_counts"] == {"standard": 1, "strict": 1}
    assert summary["violation_counts"]["require_tool_reasoning must be enabled"] == 1
    assert summary["metadata_coverage_rate"] == 0.6667
