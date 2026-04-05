"""Tests for typed provider-governance policy."""

from pydantic import ValidationError

from llm_client.core.provider_policy import (
    BlockRule,
    ProviderRuntimePolicy,
    blocked_model_reason,
    canonicalize_model_for_policy,
    get_provider_governance_policy,
    get_provider_runtime_policy,
)


def test_canonicalizes_exact_aliases_before_route_selection() -> None:
    assert canonicalize_model_for_policy("gpt-5.4", "openrouter") == "codex/gpt-5.4"
    assert canonicalize_model_for_policy("openrouter/openai/gpt-5.4", "openrouter") == "codex/gpt-5.4"


def test_policy_declares_forced_reroute_and_block_rules() -> None:
    policy = get_provider_governance_policy()

    assert policy.exact_aliases["gpt-5.4"].canonical_model == "codex/gpt-5.4"
    assert get_provider_runtime_policy("google").shared_limit == 4
    assert get_provider_runtime_policy("google").cooldown_floor_s == 15.0

    blocked_policy = policy.model_copy(
        update={
            "blocked_exact_aliases": {
                "forbidden-model": BlockRule(reason="Forbidden for governance reasons."),
            },
        }
    )
    assert blocked_model_reason("forbidden-model", policy=blocked_policy) == "Forbidden for governance reasons."


def test_provider_runtime_policy_defaults_to_zero_caps_for_unknown_provider() -> None:
    runtime_policy = get_provider_runtime_policy("unknown-provider")

    assert runtime_policy == ProviderRuntimePolicy(route_class="direct_provider")


def test_policy_model_rejects_negative_runtime_defaults() -> None:
    try:
        ProviderRuntimePolicy(route_class="direct_provider", shared_limit=-1)
    except ValidationError:
        return
    raise AssertionError("ProviderRuntimePolicy should reject negative shared_limit")


def test_policy_model_requires_template_placeholder() -> None:
    policy = get_provider_governance_policy()
    gemini_rule = policy.direct_prefix_templates[0]
    assert "{model}" in gemini_rule.template


def test_blocked_model_reason_ignores_empty_model() -> None:
    assert blocked_model_reason("") is None
    assert blocked_model_reason("  ") is None
