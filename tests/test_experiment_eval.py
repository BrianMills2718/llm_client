"""Tests for llm_client.experiment_eval."""

from __future__ import annotations

import json

from llm_client.experiment_eval import (
    build_gate_signals,
    evaluate_gate_policy,
    load_gate_policy,
    review_items_with_rubric,
    run_deterministic_checks_for_item,
    run_deterministic_checks_for_items,
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
            "extra": {"composability": {"n_errors": 0, "error_categories": {}}},
        },
        {
            "item_id": "q2",
            "metrics": {"em": 0, "f1": 0.0, "llm_em": 0},
            "predicted": "",
            "gold": "Bob",
            "error": "timeout",
            "trace_id": "",
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
