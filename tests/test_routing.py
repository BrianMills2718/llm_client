"""Unit tests for pure routing/config resolution."""

from llm_client.core.config import ClientConfig
from llm_client.core.client_dispatch import _resolve_call_plan
from llm_client.core.model_availability import clear_model_unavailability, record_model_unavailability
from llm_client.core.routing import CallRequest, resolve_api_base_for_model, resolve_call


def test_resolve_call_openrouter_normalizes_and_deduplicates() -> None:
    cfg = ClientConfig(routing_policy="openrouter")
    plan = resolve_call(
        CallRequest(
            model="gpt-5",
            fallback_models=["gpt-5", "openrouter/openai/gpt-4", "gpt-4"],
        ),
        cfg,
    )

    assert plan.primary_model == "openrouter/openai/gpt-5"
    assert plan.models == ["openrouter/openai/gpt-5", "openrouter/openai/gpt-4"]
    assert plan.fallback_models == ["openrouter/openai/gpt-4"]
    assert plan.routing_trace["routing_policy"] == "openrouter_on"
    assert plan.routing_trace["normalized_from"] == "gpt-5"
    assert plan.routing_trace["normalized_to"] == "openrouter/openai/gpt-5"


def test_resolve_call_direct_preserves_model_ids() -> None:
    cfg = ClientConfig(routing_policy="direct")
    plan = resolve_call(
        CallRequest(model="gpt-5", fallback_models=["gpt-4"]),
        cfg,
    )

    assert plan.primary_model == "gpt-5"
    assert plan.models == ["gpt-5", "gpt-4"]
    assert plan.routing_trace["routing_policy"] == "openrouter_off"
    assert "normalized_from" not in plan.routing_trace
    assert "normalization_events" not in plan.routing_trace


def test_resolve_api_base_prefers_explicit_and_injects_openrouter_default() -> None:
    cfg = ClientConfig(
        routing_policy="openrouter",
        openrouter_api_base="https://router.example/api/v1",
    )

    assert (
        resolve_api_base_for_model("openrouter/openai/gpt-4", None, cfg)
        == "https://router.example/api/v1"
    )
    assert resolve_api_base_for_model("gpt-4", None, cfg) is None
    assert (
        resolve_api_base_for_model("openrouter/openai/gpt-4", "https://override", cfg)
        == "https://override"
    )


def test_resolve_call_plan_skips_temporarily_unavailable_models() -> None:
    cfg = ClientConfig(routing_policy="direct")

    class ExhaustedError(Exception):
        pass

    clear_model_unavailability()
    record_model_unavailability(
        "gemini/gemini-2.5-flash",
        ExhaustedError(
            "Rate limit exceeded for GenerateContentRequestsPerDayPerProjectPerModel-FreeTier. "
            "Please try again tomorrow."
        ),
    )

    plan = _resolve_call_plan(
        model="gemini/gemini-2.5-flash",
        fallback_models=["gemini/gemini-2.5-flash-lite", "openrouter/openai/gpt-5.4-mini"],
        api_base=None,
        config=cfg,
    )
    clear_model_unavailability()

    assert plan.primary_model == "gemini/gemini-2.5-flash-lite"
    assert plan.models == ["gemini/gemini-2.5-flash-lite", "openrouter/openai/gpt-5.4-mini"]
    suppressed = plan.routing_trace["suppressed_models"]
    assert suppressed[0]["model"] == "gemini/gemini-2.5-flash"
    assert suppressed[0]["reason"] == "provider_daily_quota_exhausted"


def test_record_model_unavailability_uses_provider_retry_hint_for_daily_quota() -> None:
    class ExhaustedError(Exception):
        pass

    clear_model_unavailability()
    record = record_model_unavailability(
        "gemini/gemini-2.5-flash",
        ExhaustedError(
            "Quota exceeded for metric: generativelanguage.googleapis.com/generate_requests_per_model_per_day. "
            '{"details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"34820s"}]}'
        ),
        now_monotonic=0.0,
    )
    clear_model_unavailability()

    assert record is not None
    assert record["reason"] == "provider_daily_quota_exhausted"
    assert record["cooldown_s"] == 34820.0
