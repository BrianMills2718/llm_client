"""Tests for shared experiment item summary helpers."""

from __future__ import annotations

from llm_client.experiment_summary import (
    extract_adoption_profile,
    extract_agent_outcome,
    summarize_adoption_profiles,
    summarize_agent_outcomes,
)


def test_extract_agent_outcome_derives_fields_from_submit_mode_and_prediction() -> None:
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


def test_summarize_agent_outcomes_counts_completion_modes_and_failure_classes() -> None:
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


def test_extract_adoption_profile_reads_nested_agent_metadata() -> None:
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


def test_summarize_adoption_profiles_counts_profiles_and_coverage() -> None:
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
