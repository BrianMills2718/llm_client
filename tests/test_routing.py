"""Unit tests for pure routing/config resolution."""

from llm_client.core.config import ClientConfig
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


def test_resolve_call_normalizes_google_gemini_alias_to_openrouter() -> None:
    """OpenRouter policy should canonicalize google/gemini provider aliases."""
    cfg = ClientConfig(routing_policy="openrouter")

    plan = resolve_call(
        CallRequest(model="google/gemini-2.0-flash-001"),
        cfg,
    )

    assert plan.primary_model == "openrouter/google/gemini-2.0-flash-001"
    assert plan.routing_trace["normalized_from"] == "google/gemini-2.0-flash-001"
    assert plan.routing_trace["normalized_to"] == "openrouter/google/gemini-2.0-flash-001"


def test_resolve_call_normalizes_bare_gemini_model_id() -> None:
    """Bare Gemini ids should canonicalize before provider routing kicks in."""
    cfg = ClientConfig(routing_policy="openrouter")

    plan = resolve_call(
        CallRequest(model="gemini-2.5-flash"),
        cfg,
    )

    assert plan.primary_model == "gemini/gemini-2.5-flash"
    assert plan.routing_trace["normalized_from"] == "gemini-2.5-flash"
    assert plan.routing_trace["normalized_to"] == "gemini/gemini-2.5-flash"


def test_resolve_call_direct_still_canonicalizes_bare_gemini_model_id() -> None:
    """Direct policy still needs a canonical provider id for bare Gemini models."""
    cfg = ClientConfig(routing_policy="direct")

    plan = resolve_call(
        CallRequest(model="gemini-3-flash-preview"),
        cfg,
    )

    assert plan.primary_model == "gemini/gemini-3-flash-preview"
    assert plan.models == ["gemini/gemini-3-flash-preview"]
    assert plan.routing_trace["routing_policy"] == "openrouter_off"
    assert plan.routing_trace["normalized_from"] == "gemini-3-flash-preview"
    assert plan.routing_trace["normalized_to"] == "gemini/gemini-3-flash-preview"
