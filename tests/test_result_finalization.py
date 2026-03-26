"""Focused tests for post-dispatch result finalization helpers.

# mock-ok: exercises pure result finalization without provider transport
"""

from __future__ import annotations

from llm_client.core.client import LLMCallResult
from llm_client.result_finalization import cache_hit_view, finalize_result
from llm_client.result_metadata import warning_record


def test_cache_hit_view_returns_copy_without_mutating_source() -> None:
    """Cache-hit views should not mutate the cached source object."""
    result = LLMCallResult(
        content="hi",
        usage={"total_tokens": 3},
        cost=1.25,
        model="gpt-4",
    )

    cached_result = cache_hit_view(result)

    assert cached_result is not result
    assert result.cache_hit is False
    assert result.cost_source == "unspecified"
    assert result.marginal_cost == 1.25
    assert cached_result.cache_hit is True
    assert cached_result.cost_source == "cache_hit"
    assert cached_result.marginal_cost == 0.0


def test_finalize_result_applies_cache_hit_and_identity_metadata() -> None:
    """Finalization should combine cache-hit bookkeeping with identity metadata."""
    result = LLMCallResult(
        content="cached",
        usage={"total_tokens": 3},
        cost=1.25,
        model="fallback-model",
        warnings=["FALLBACK: requested-model -> fallback-model"],
    )

    finalized = finalize_result(
        result,
        requested_model="requested-model",
        resolved_model="fallback-model",
        routing_trace={
            "routing_policy": "openrouter_off",
            "attempted_models": ["requested-model", "fallback-model"],
        },
        warning_records=[
            warning_record(
                code="LLMC_WARN_MODEL_OUTCLASSED",
                category="UserWarning",
                message="Model requested-model is outclassed but still allowed.",
            )
        ],
        cache_hit=True,
    )

    assert finalized is not result
    assert result.cache_hit is False
    assert finalized.cache_hit is True
    assert finalized.cost_source == "cache_hit"
    assert finalized.marginal_cost == 0.0
    assert finalized.requested_model == "requested-model"
    assert finalized.resolved_model == "fallback-model"
    assert finalized.execution_model == "fallback-model"
    assert finalized.routing_trace == {
        "routing_policy": "openrouter_off",
        "attempted_models": ["requested-model", "fallback-model"],
    }
    assert [record["code"] for record in finalized.warning_records] == [
        "LLMC_WARN_FALLBACK",
        "LLMC_WARN_MODEL_OUTCLASSED",
    ]
