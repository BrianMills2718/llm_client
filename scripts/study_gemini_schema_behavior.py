#!/usr/bin/env python3
"""Run a small Gemini structured-output study on Tyler-like schemas.

This script exists to answer one question with evidence instead of guesswork:
for representative structured-output cases, does Gemini succeed via native JSON
schema, instructor fallback, or an error path?
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_client import acall_llm_structured, io_log


class FlatCaseModel(BaseModel):
    """Simple required-field schema."""

    answer: str = Field(description="Direct answer to the question.")
    confidence: str = Field(description="Short confidence label such as high, medium, or low.")


class NestedCaseSource(BaseModel):
    """Nested source row for structured-output stress testing."""

    title: str = Field(description="Short source title.")
    url: str = Field(description="Source URL.")


class NestedCaseModel(BaseModel):
    """Nested object schema similar to Tyler stage outputs."""

    summary: str = Field(description="One-sentence summary of the topic.")
    source: NestedCaseSource = Field(description="Primary source supporting the summary.")


class OptionalCaseModel(BaseModel):
    """Schema with an optional follow-up field."""

    claim: str = Field(description="Main claim.")
    caveat: str | None = Field(description="Optional caveat if one exists, otherwise null.")


class ListItemModel(BaseModel):
    """One list item in a structured set."""

    label: str = Field(description="Short item label.")
    rationale: str = Field(description="Short reason this item belongs in the list.")


class ListCaseModel(BaseModel):
    """Schema with a list of typed objects."""

    items: list[ListItemModel] = Field(description="Two or three typed items.")


class DecisionCaseModel(BaseModel):
    """Enum-heavy decision schema similar to recommendation outputs."""

    recommendation: str = Field(description="Recommended course of action.")
    urgency: str = Field(description="One of immediate, near_term, or watch.")
    tradeoff: str = Field(description="Main tradeoff behind the recommendation.")


@dataclass(frozen=True)
class StudyCase:
    """One fixed schema-behavior study case."""

    name: str
    prompt: str
    response_model: type[BaseModel]


def build_study_call_kwargs(
    *,
    model: str,
    max_budget: float,
    direct_gemini_thinking_budget: int | None,
) -> dict[str, Any]:
    """Build per-call kwargs for the live study without hiding provider requirements."""

    kwargs: dict[str, Any] = {"max_budget": max_budget}
    if model.lower().startswith("gemini/") and direct_gemini_thinking_budget is not None:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": direct_gemini_thinking_budget}
    return kwargs


def build_case_registry() -> list[StudyCase]:
    """Return the fixed Tyler-like schema cases used by this study."""

    return [
        StudyCase(
            name="flat_required",
            prompt="Answer briefly: what is a prudent first step when evaluating a new policy claim?",
            response_model=FlatCaseModel,
        ),
        StudyCase(
            name="nested_object",
            prompt="Provide a one-sentence summary of why source provenance matters and cite one illustrative source.",
            response_model=NestedCaseModel,
        ),
        StudyCase(
            name="optional_field",
            prompt="State one claim about evidence quality and include a caveat only if needed.",
            response_model=OptionalCaseModel,
        ),
        StudyCase(
            name="list_of_objects",
            prompt="List two reasons structured-output validation matters for production systems.",
            response_model=ListCaseModel,
        ),
        StudyCase(
            name="decision_like",
            prompt="Recommend a validation posture for a new structured-output provider path.",
            response_model=DecisionCaseModel,
        ),
    ]


def fetch_observability_rows(db_path: Path, trace_id: str) -> list[dict[str, Any]]:
    """Load llm_client observability rows for one study trace ID."""

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """
            SELECT task, model, error, error_type, execution_path, schema_hash,
                   response_format_type, validation_errors, retry_count
            FROM llm_calls
            WHERE trace_id = ?
            ORDER BY id
            """,
            (trace_id,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        con.close()


def summarize_observability_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize execution-path evidence for one structured call."""

    if not rows:
        return {
            "status": "missing_observability",
            "execution_path": None,
            "response_format_type": None,
            "error_type": None,
            "validation_errors": None,
            "retry_count": None,
            "row_count": 0,
        }
    final = rows[-1]
    return {
        "status": "error" if final.get("error") else "success",
        "execution_path": final.get("execution_path"),
        "response_format_type": final.get("response_format_type"),
        "error_type": final.get("error_type"),
        "validation_errors": final.get("validation_errors"),
        "retry_count": final.get("retry_count"),
        "row_count": len(rows),
    }


def write_summary_json(path: Path, payload: dict[str, Any]) -> None:
    """Write the study summary as stable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


async def run_case(
    model: str,
    case: StudyCase,
    db_path: Path,
    max_budget: float,
    direct_gemini_thinking_budget: int | None,
) -> dict[str, Any]:
    """Execute one live Gemini/schema case and attach observability evidence."""

    trace_id = f"llm_client/gemini_schema_study/{case.name}/{uuid.uuid4().hex[:12]}"
    prompt = (
        "Return valid JSON matching the response schema exactly. "
        "Do not add extra keys.\n\n"
        f"{case.prompt}"
    )
    result_summary: dict[str, Any] = {
        "case": case.name,
        "model": model,
        "trace_id": trace_id,
    }
    try:
        call_kwargs = build_study_call_kwargs(
            model=model,
            max_budget=max_budget,
            direct_gemini_thinking_budget=direct_gemini_thinking_budget,
        )
        parsed, _meta = await acall_llm_structured(
            model,
            [{"role": "user", "content": prompt}],
            case.response_model,
            task="gemini_schema_behavior_study",
            trace_id=trace_id,
            **call_kwargs,
        )
        result_summary["parsed"] = parsed.model_dump(mode="json")
    except Exception as exc:  # noqa: BLE001 - study must record the provider/runtime failure
        result_summary["error"] = f"{type(exc).__name__}: {exc}"
    rows = fetch_observability_rows(db_path, trace_id)
    result_summary["observability"] = summarize_observability_rows(rows)
    return result_summary


async def run_study(
    models: list[str],
    db_path: Path,
    output_json: Path,
    max_budget: float,
    direct_gemini_thinking_budget: int | None,
) -> dict[str, Any]:
    """Run the full model x case matrix and write one JSON summary."""

    io_log.configure(project="llm_client", db_path=db_path)
    results: list[dict[str, Any]] = []
    for model in models:
        for case in build_case_registry():
            results.append(
                await run_case(
                    model=model,
                    case=case,
                    db_path=db_path,
                    max_budget=max_budget,
                    direct_gemini_thinking_budget=direct_gemini_thinking_budget,
                )
            )
    payload = {
        "models": models,
        "db_path": str(db_path),
        "direct_gemini_thinking_budget": direct_gemini_thinking_budget,
        "results": results,
    }
    write_summary_json(output_json, payload)
    return payload


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the live study harness."""

    parser = argparse.ArgumentParser(description="Study Gemini structured-output behavior on Tyler-like schemas.")
    parser.add_argument("--model", action="append", dest="models", required=True, help="Gemini model id to evaluate. Repeat for multiple models.")
    parser.add_argument("--db-path", default="tmp/gemini_schema_behavior_study/llm_observability.db", help="Dedicated observability SQLite path.")
    parser.add_argument("--output-json", default="tmp/gemini_schema_behavior_study/summary.json", help="Summary JSON output path.")
    parser.add_argument("--max-budget", type=float, default=1.0, help="Per-call max budget passed to llm_client.")
    parser.add_argument(
        "--direct-gemini-thinking-budget",
        type=int,
        default=None,
        help="Optional positive thinking budget for direct gemini/* models when the provider rejects budget_tokens=0.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the live study harness."""

    args = parse_args()
    db_path = Path(os.path.expanduser(args.db_path))
    output_json = Path(os.path.expanduser(args.output_json))
    payload = asyncio.run(
        run_study(
            models=args.models,
            db_path=db_path,
            output_json=output_json,
            max_budget=args.max_budget,
            direct_gemini_thinking_budget=args.direct_gemini_thinking_budget,
        )
    )
    print(json.dumps({"output_json": str(output_json), "cases": len(payload["results"])}, indent=2))


if __name__ == "__main__":
    main()
