"""Tests for the Gemini strict-schema study harness."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.study_gemini_schema_behavior import (
    build_case_registry,
    build_study_call_kwargs,
    summarize_observability_rows,
    write_summary_json,
)


def test_build_case_registry_returns_named_cases() -> None:
    """The study should expose a fixed, explicit case registry."""

    cases = build_case_registry()
    assert [case.name for case in cases] == [
        "flat_required",
        "nested_object",
        "optional_field",
        "list_of_objects",
        "decision_like",
    ]


def test_summarize_observability_rows_classifies_execution_paths() -> None:
    """Execution-path summaries should come from the final observability row."""

    rows = [
        {
            "execution_path": "native_schema",
            "response_format_type": "json_schema",
            "error": None,
            "error_type": None,
            "validation_errors": None,
            "retry_count": 0,
        },
        {
            "execution_path": "instructor",
            "response_format_type": "instructor",
            "error": None,
            "error_type": None,
            "validation_errors": None,
            "retry_count": 1,
        },
    ]

    summary = summarize_observability_rows(rows)

    assert summary["status"] == "success"
    assert summary["execution_path"] == "instructor"
    assert summary["response_format_type"] == "instructor"
    assert summary["retry_count"] == 1
    assert summary["row_count"] == 2


def test_build_study_call_kwargs_adds_direct_gemini_thinking_budget() -> None:
    """Direct Gemini runs can opt into a positive thinking budget for the study."""

    kwargs = build_study_call_kwargs(
        model="gemini/gemini-2.5-pro",
        max_budget=1.0,
        direct_gemini_thinking_budget=256,
    )

    assert kwargs["max_budget"] == 1.0
    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 256}


def test_build_study_call_kwargs_does_not_add_thinking_for_openrouter_gemini() -> None:
    """OpenRouter Gemini models should not inherit direct-Gemini thinking kwargs."""

    kwargs = build_study_call_kwargs(
        model="openrouter/google/gemini-3.1-pro-preview",
        max_budget=1.0,
        direct_gemini_thinking_budget=256,
    )

    assert kwargs == {"max_budget": 1.0}


def test_write_summary_json_emits_expected_shape(tmp_path: Path) -> None:
    """Summary writing should produce stable machine-readable JSON."""

    out = tmp_path / "summary.json"
    payload = {"models": ["gemini/gemini-2.5-pro"], "results": [{"case": "flat_required"}]}

    write_summary_json(out, payload)

    loaded = json.loads(out.read_text())
    assert loaded == payload
