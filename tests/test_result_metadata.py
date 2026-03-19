"""Focused tests for additive result metadata helpers.

# mock-ok: exercises pure metadata normalization without provider transport
"""

from __future__ import annotations

from llm_client.client import LLMCallResult
from llm_client.result_metadata import (
    annotate_result_identity,
    build_routing_trace,
    merge_warning_records,
    warning_record,
    warning_record_from_message,
)


def test_build_routing_trace_tracks_normalization_and_api_base_injection() -> None:
    """Routing traces should preserve normalization and injected api-base facts."""
    trace = build_routing_trace(
        requested_model="gpt-4",
        attempted_models=["openrouter/openai/gpt-4", "openrouter/openai/gpt-3.5-turbo"],
        selected_model="openrouter/openai/gpt-3.5-turbo",
        effective_api_base="https://openrouter.ai/api/v1",
        sticky_fallback=True,
        background_mode=True,
        routing_policy="openrouter_on",
    )

    assert trace["routing_policy"] == "openrouter_on"
    assert trace["normalized_from"] == "gpt-4"
    assert trace["normalized_to"] == "openrouter/openai/gpt-4"
    assert trace["selected_model"] == "openrouter/openai/gpt-3.5-turbo"
    assert trace["api_base_injected"] is True
    assert trace["sticky_fallback"] is True
    assert trace["background_mode"] is True


def test_warning_record_from_message_maps_known_warning() -> None:
    """Known warning prefixes should map to stable warning codes."""
    record = warning_record_from_message("AUTO_TAG: task=adhoc")

    assert record is not None
    assert record["code"] == "LLMC_WARN_AUTO_TAG"
    assert record["category"] == "UserWarning"


def test_merge_warning_records_deduplicates_warning_messages_and_extra_records() -> None:
    """Equivalent warning code/message pairs should appear only once."""
    existing = [
        warning_record(
            code="LLMC_WARN_AUTO_TAG",
            category="UserWarning",
            message="AUTO_TAG: task=adhoc",
        )
    ]
    merged = merge_warning_records(
        existing=existing,
        warnings=["AUTO_TAG: task=adhoc", "FALLBACK: primary -> backup"],
        extra_records=[
            warning_record(
                code="LLMC_WARN_FALLBACK",
                category="UserWarning",
                message="FALLBACK: primary -> backup",
            ),
            warning_record(
                code="LLMC_WARN_MODEL_OUTCLASSED",
                category="UserWarning",
                message="Model gpt-4o is outclassed but still allowed.",
            ),
        ],
    )

    assert len(merged) == 3
    assert [record["code"] for record in merged] == [
        "LLMC_WARN_AUTO_TAG",
        "LLMC_WARN_FALLBACK",
        "LLMC_WARN_MODEL_OUTCLASSED",
    ]


def test_annotate_result_identity_sets_identity_and_merges_warning_records() -> None:
    """Result identity stamping should preserve existing warnings and metadata."""
    result = LLMCallResult(
        content="hi",
        usage={"total_tokens": 1},
        cost=0.0,
        model="fallback-model",
        warnings=["FALLBACK: requested-model -> fallback-model"],
    )

    annotated = annotate_result_identity(
        result,
        requested_model="requested-model",
        resolved_model="fallback-model",
        routing_trace={"routing_policy": "openrouter_off", "attempted_models": ["requested-model", "fallback-model"]},
        warning_records=[
            warning_record(
                code="LLMC_WARN_MODEL_OUTCLASSED",
                category="UserWarning",
                message="Model requested-model is outclassed but still allowed.",
            )
        ],
    )

    assert annotated.requested_model == "requested-model"
    assert annotated.resolved_model == "fallback-model"
    assert annotated.execution_model == "fallback-model"
    assert annotated.model == "fallback-model"
    assert annotated.routing_trace == {
        "routing_policy": "openrouter_off",
        "attempted_models": ["requested-model", "fallback-model"],
    }
    assert [record["code"] for record in annotated.warning_records] == [
        "LLMC_WARN_FALLBACK",
        "LLMC_WARN_MODEL_OUTCLASSED",
    ]
