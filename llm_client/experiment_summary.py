"""Shared experiment-item outcome and adoption summary helpers.

This module extracts and summarizes agent-outcome and adoption-profile metadata
from experiment item records. It exists so shared observability and CLI paths
can compute run summaries without depending on the broader eval/review module.
"""

from __future__ import annotations

import re
from typing import Any


def _to_bool(value: Any) -> bool | None:
    """Normalize common JSON-ish truthy/falsey values into booleans."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y", "on"}:
            return True
        if text in {"false", "0", "no", "n", "off"}:
            return False
    return None


def _lookup_item_value(item: dict[str, Any], key: str) -> Any:
    """Read an item field from top-level, nested extra, or nested metadata."""
    if key in item:
        return item.get(key)
    extra = item.get("extra")
    if isinstance(extra, dict):
        if key in extra:
            return extra.get(key)
        agent = extra.get("agent")
        if isinstance(agent, dict) and key in agent:
            return agent.get(key)
    metadata = item.get("metadata")
    if isinstance(metadata, dict) and key in metadata:
        return metadata.get(key)
    return None


def _slug_token(value: str) -> str:
    """Normalize free-form labels into stable summary tokens."""
    token = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return token or "unknown"


def extract_agent_outcome(item: dict[str, Any]) -> dict[str, Any]:
    """Extract normalized agent-outcome metadata from one experiment item."""
    submit_completion_mode = str(_lookup_item_value(item, "submit_completion_mode") or "").strip()
    if not submit_completion_mode:
        submit_validator_accepted = _to_bool(_lookup_item_value(item, "submit_validator_accepted"))
        required_submit_missing = _to_bool(_lookup_item_value(item, "required_submit_missing"))
        requires_submit_answer = _to_bool(_lookup_item_value(item, "requires_submit_answer"))
        forced_accept = _to_bool(_lookup_item_value(item, "submit_forced_accept_on_budget_exhaustion"))
        if submit_validator_accepted:
            submit_completion_mode = "grounded_submit"
        elif forced_accept:
            submit_completion_mode = "forced_terminal_accept"
        elif required_submit_missing:
            submit_completion_mode = "missing_required_submit"
        elif requires_submit_answer is False:
            submit_completion_mode = "no_submit_required"
        else:
            submit_completion_mode = "unknown"

    predicted = str(item.get("predicted") or item.get("prediction") or item.get("content") or "").strip()
    answer_present = _to_bool(_lookup_item_value(item, "answer_present"))
    if answer_present is None:
        answer_present = bool(predicted)

    submit_validator_accepted = _to_bool(_lookup_item_value(item, "submit_validator_accepted"))
    if submit_validator_accepted is None:
        submit_validator_accepted = submit_completion_mode == "grounded_submit"

    submit_answer_succeeded = _to_bool(_lookup_item_value(item, "submit_answer_succeeded"))
    required_submit_missing = _to_bool(_lookup_item_value(item, "required_submit_missing"))
    if required_submit_missing is None:
        required_submit_missing = submit_completion_mode == "missing_required_submit"

    required_submit_satisfied = _to_bool(_lookup_item_value(item, "required_submit_satisfied"))
    if required_submit_satisfied is None:
        required_submit_satisfied = not required_submit_missing

    forced_terminal_accepted = _to_bool(_lookup_item_value(item, "forced_terminal_accepted"))
    if forced_terminal_accepted is None:
        forced_terminal_accepted = (
            _to_bool(_lookup_item_value(item, "submit_forced_accept_on_budget_exhaustion")) is True
            or submit_completion_mode == "forced_terminal_accept"
        )

    grounded_completed = _to_bool(_lookup_item_value(item, "grounded_completed"))
    if grounded_completed is None:
        grounded_completed = bool(answer_present and submit_validator_accepted)

    reliability_completed = _to_bool(_lookup_item_value(item, "reliability_completed"))
    if reliability_completed is None:
        reliability_completed = bool(
            answer_present
            and (
                submit_answer_succeeded
                or submit_completion_mode in {"grounded_submit", "forced_terminal_accept", "no_submit_required"}
            )
        )

    primary_failure_class = str(_lookup_item_value(item, "primary_failure_class") or "").strip() or None
    first_terminal_failure_event_code = (
        str(_lookup_item_value(item, "first_terminal_failure_event_code") or "").strip() or None
    )

    return {
        "submit_completion_mode": submit_completion_mode,
        "answer_present": bool(answer_present),
        "submit_validator_accepted": bool(submit_validator_accepted),
        "required_submit_missing": bool(required_submit_missing),
        "required_submit_satisfied": bool(required_submit_satisfied),
        "grounded_completed": bool(grounded_completed),
        "forced_terminal_accepted": bool(forced_terminal_accepted),
        "reliability_completed": bool(reliability_completed),
        "primary_failure_class": primary_failure_class,
        "first_terminal_failure_event_code": first_terminal_failure_event_code,
    }


def extract_adoption_profile(item: dict[str, Any]) -> dict[str, Any]:
    """Extract normalized adoption-profile metadata from one experiment item."""
    requested_profile = str(_lookup_item_value(item, "adoption_profile_requested") or "").strip() or None
    effective_profile = str(_lookup_item_value(item, "adoption_profile_effective") or "").strip() or None
    satisfied = _to_bool(_lookup_item_value(item, "adoption_profile_satisfied"))
    violations = _lookup_item_value(item, "adoption_profile_violations")
    normalized_violations: list[str] = []
    if isinstance(violations, list):
        normalized_violations = [str(v).strip() for v in violations if str(v).strip()]
    has_metadata = bool(
        requested_profile
        or effective_profile
        or satisfied is not None
        or normalized_violations
    )
    return {
        "requested_profile": requested_profile,
        "effective_profile": effective_profile,
        "satisfied": satisfied,
        "violations": normalized_violations,
        "has_metadata": has_metadata,
    }


def summarize_agent_outcomes(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize outcome/completion metadata across experiment items."""
    n_items = len(items)
    summary: dict[str, Any] = {
        "n_items": n_items,
        "answer_present_count": 0,
        "grounded_completed_count": 0,
        "forced_terminal_accepted_count": 0,
        "reliability_completed_count": 0,
        "required_submit_missing_count": 0,
        "submit_validator_accepted_count": 0,
        "submit_completion_mode_counts": {},
        "primary_failure_class_counts": {},
        "first_terminal_failure_event_code_counts": {},
        "items": [],
    }

    mode_counts: dict[str, int] = {}
    failure_counts: dict[str, int] = {}
    terminal_counts: dict[str, int] = {}

    for item in items:
        item_id = str(item.get("item_id") or item.get("id") or "")
        outcome = extract_agent_outcome(item)
        summary["answer_present_count"] += int(outcome["answer_present"])
        summary["grounded_completed_count"] += int(outcome["grounded_completed"])
        summary["forced_terminal_accepted_count"] += int(outcome["forced_terminal_accepted"])
        summary["reliability_completed_count"] += int(outcome["reliability_completed"])
        summary["required_submit_missing_count"] += int(outcome["required_submit_missing"])
        summary["submit_validator_accepted_count"] += int(outcome["submit_validator_accepted"])

        mode = str(outcome.get("submit_completion_mode") or "unknown").strip() or "unknown"
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

        primary_failure = str(outcome.get("primary_failure_class") or "unknown").strip() or "unknown"
        failure_counts[primary_failure] = failure_counts.get(primary_failure, 0) + 1

        terminal_code = str(outcome.get("first_terminal_failure_event_code") or "none").strip() or "none"
        terminal_counts[terminal_code] = terminal_counts.get(terminal_code, 0) + 1

        summary["items"].append({"item_id": item_id, **outcome})

    summary["submit_completion_mode_counts"] = dict(sorted(mode_counts.items()))
    summary["primary_failure_class_counts"] = dict(sorted(failure_counts.items()))
    summary["first_terminal_failure_event_code_counts"] = dict(sorted(terminal_counts.items()))

    for key in (
        "answer_present",
        "grounded_completed",
        "forced_terminal_accepted",
        "reliability_completed",
        "required_submit_missing",
        "submit_validator_accepted",
    ):
        count = int(summary[f"{key}_count"])
        summary[f"{key}_rate"] = round((count / n_items), 4) if n_items else None

    return summary


def summarize_adoption_profiles(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize adoption-profile metadata across experiment items."""
    n_items = len(items)
    summary: dict[str, Any] = {
        "n_items": n_items,
        "n_items_with_metadata": 0,
        "satisfied_count": 0,
        "requested_profile_counts": {},
        "effective_profile_counts": {},
        "violation_counts": {},
        "items": [],
    }

    requested_counts: dict[str, int] = {}
    effective_counts: dict[str, int] = {}
    violation_counts: dict[str, int] = {}

    for item in items:
        item_id = str(item.get("item_id") or item.get("id") or "")
        adoption = extract_adoption_profile(item)
        if adoption["has_metadata"]:
            summary["n_items_with_metadata"] += 1
        if adoption["satisfied"] is True:
            summary["satisfied_count"] += 1
        requested_profile = adoption["requested_profile"]
        if requested_profile:
            requested_counts[requested_profile] = requested_counts.get(requested_profile, 0) + 1
        effective_profile = adoption["effective_profile"]
        if effective_profile:
            effective_counts[effective_profile] = effective_counts.get(effective_profile, 0) + 1
        for violation in adoption["violations"]:
            violation_counts[violation] = violation_counts.get(violation, 0) + 1

        summary["items"].append({"item_id": item_id, **adoption})

    summary["requested_profile_counts"] = dict(sorted(requested_counts.items()))
    summary["effective_profile_counts"] = dict(sorted(effective_counts.items()))
    summary["violation_counts"] = dict(sorted(violation_counts.items()))
    summary["metadata_coverage_rate"] = round((summary["n_items_with_metadata"] / n_items), 4) if n_items else None
    summary["satisfied_rate"] = round((summary["satisfied_count"] / n_items), 4) if n_items else None
    return summary


__all__ = [
    "extract_adoption_profile",
    "extract_agent_outcome",
    "summarize_adoption_profiles",
    "summarize_agent_outcomes",
]
