"""Typed provider-governance policy for routing and shared runtime defaults."""

from __future__ import annotations

from functools import lru_cache
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from llm_client.core.config import RoutingPolicy

ProviderRouteClass = Literal["agent_sdk", "direct_provider", "openrouter"]
_CODEX_AGENT_ALIASES: frozenset[str] = frozenset({"codex-mini-latest", "gpt-5.4"})
_CODEX_FAMILY_RE = re.compile(r"-codex(?:-|$)", re.IGNORECASE)


class ProviderRuntimePolicy(BaseModel):
    """Default shared runtime policy for one provider."""

    model_config = ConfigDict(frozen=True)

    route_class: ProviderRouteClass = Field(
        description="Default execution lane for this provider family.",
    )
    shared_limit: int = Field(
        default=0,
        ge=0,
        description="Cross-process concurrency cap for the provider. Zero disables shared leasing.",
    )
    cooldown_floor_s: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum shared cooldown applied after provider rate-limit signals.",
    )


class ExactAliasRule(BaseModel):
    """Canonical route for one exact model alias."""

    model_config = ConfigDict(frozen=True)

    canonical_model: str = Field(
        description="Authoritative model route emitted by the runtime.",
    )
    route_class: ProviderRouteClass = Field(
        description="Execution lane used after canonicalization.",
    )
    reason: str = Field(
        description="Why this exact alias is canonicalized.",
    )


class BlockRule(BaseModel):
    """Hard-block policy for one exact model alias."""

    model_config = ConfigDict(frozen=True)

    reason: str = Field(
        description="Why the runtime should reject this model alias.",
    )


class PrefixTemplateRule(BaseModel):
    """Canonicalization rule for a bare-model prefix family."""

    model_config = ConfigDict(frozen=True)

    prefixes: tuple[str, ...] = Field(
        description="Lower-case prefixes that trigger this rule.",
    )
    template: str = Field(
        description="Format string used to build the canonical route. Must include {model}.",
    )
    route_class: ProviderRouteClass = Field(
        description="Execution lane used after canonicalization.",
    )
    reason: str = Field(
        description="Why this family is canonicalized through this rule.",
    )


class ExplicitProviderRouteRule(BaseModel):
    """Routing rule for already provider-qualified model ids."""

    model_config = ConfigDict(frozen=True)

    providers: tuple[str, ...] = Field(
        description="Provider prefixes eligible for rerouting.",
    )
    target_prefix: str = Field(
        description="Prefix prepended to the provider-qualified model id.",
    )
    route_class: ProviderRouteClass = Field(
        description="Execution lane used after canonicalization.",
    )
    reason: str = Field(
        description="Why these provider-qualified ids are rerouted.",
    )


class ProviderGovernancePolicy(BaseModel):
    """Typed source of truth for provider-governance routing behavior."""

    model_config = ConfigDict(frozen=True)

    passthrough_prefixes: tuple[str, ...] = Field(
        description="Prefixes that already represent canonical routes for the current runtime.",
    )
    blocked_exact_aliases: dict[str, BlockRule] = Field(
        description="Lower-cased exact model aliases that should fail closed before routing.",
    )
    exact_aliases: dict[str, ExactAliasRule] = Field(
        description="Lower-cased exact model aliases that canonicalize before provider selection.",
    )
    direct_prefix_templates: tuple[PrefixTemplateRule, ...] = Field(
        description="Prefix-template rules that canonicalize regardless of routing policy.",
    )
    explicit_provider_routes: tuple[ExplicitProviderRouteRule, ...] = Field(
        description="Rules for provider-qualified ids that should reroute through a shared gateway.",
    )
    bare_model_routes: tuple[PrefixTemplateRule, ...] = Field(
        description="Rules for bare model ids that should route through an explicit provider gateway.",
    )
    provider_defaults: dict[str, ProviderRuntimePolicy] = Field(
        description="Provider-scoped shared runtime defaults.",
    )


def _base_model_name(model: str) -> str:
    """Return the final model component in lower case for family checks."""

    return model.lower().rsplit("/", 1)[-1]


def _is_codex_alias_model(model: str) -> bool:
    """Return whether *model* is a named Codex SDK alias."""

    return _base_model_name(model) in _CODEX_AGENT_ALIASES


def _is_codex_family_model(model: str) -> bool:
    """Return whether *model* belongs to the Codex family by naming pattern."""

    return bool(_CODEX_FAMILY_RE.search(_base_model_name(model)))


def _is_image_generation_model(model: str) -> bool:
    """Return whether *model* is an image-generation family we leave untouched."""

    base = _base_model_name(model)
    hints = (
        "gpt-image",
        "dall-e",
        "imagen",
        "stable-diffusion",
        "sdxl",
        "flux",
    )
    return any(hint in base for hint in hints)


@lru_cache(maxsize=1)
def get_provider_governance_policy() -> ProviderGovernancePolicy:
    """Return the canonical provider-governance policy for the current runtime."""

    return ProviderGovernancePolicy(
        passthrough_prefixes=(
            "openrouter/",
            "gemini/",
            "codex",
            "claude-code",
            "openai-agents",
        ),
        blocked_exact_aliases={},
        exact_aliases={
            "gpt-5.4": ExactAliasRule(
                canonical_model="codex/gpt-5.4",
                route_class="agent_sdk",
                reason="Exact gpt-5.4 aliases must use the Codex SDK lane.",
            ),
        },
        direct_prefix_templates=(
            PrefixTemplateRule(
                prefixes=("gemini-",),
                template="gemini/{model}",
                route_class="direct_provider",
                reason="Bare Gemini ids are not stable provider identities and must be canonicalized.",
            ),
        ),
        explicit_provider_routes=(
            ExplicitProviderRouteRule(
                providers=("openai", "anthropic", "deepseek", "x-ai", "xai", "mistral", "mistralai", "google"),
                target_prefix="openrouter",
                route_class="openrouter",
                reason="These provider-qualified ids route through OpenRouter under openrouter policy.",
            ),
        ),
        bare_model_routes=(
            PrefixTemplateRule(
                prefixes=("gpt-", "o1", "o3", "o4", "chatgpt", "text-embedding-", "text-moderation-"),
                template="openrouter/openai/{model}",
                route_class="openrouter",
                reason="Bare OpenAI-family ids route through OpenRouter under openrouter policy.",
            ),
            PrefixTemplateRule(
                prefixes=("claude",),
                template="openrouter/anthropic/{model}",
                route_class="openrouter",
                reason="Bare Claude ids route through OpenRouter under openrouter policy.",
            ),
            PrefixTemplateRule(
                prefixes=("deepseek",),
                template="openrouter/deepseek/{model}",
                route_class="openrouter",
                reason="Bare DeepSeek ids route through OpenRouter under openrouter policy.",
            ),
            PrefixTemplateRule(
                prefixes=("grok",),
                template="openrouter/x-ai/{model}",
                route_class="openrouter",
                reason="Bare Grok ids route through OpenRouter under openrouter policy.",
            ),
            PrefixTemplateRule(
                prefixes=("mistral",),
                template="openrouter/mistralai/{model}",
                route_class="openrouter",
                reason="Bare Mistral ids route through OpenRouter under openrouter policy.",
            ),
        ),
        provider_defaults={
            "google": ProviderRuntimePolicy(
                route_class="direct_provider",
                shared_limit=4,
                cooldown_floor_s=15.0,
            ),
        },
    )


def canonicalize_model_for_policy(
    model: str,
    routing_policy: RoutingPolicy,
    *,
    policy: ProviderGovernancePolicy | None = None,
) -> str:
    """Return the canonical model id for the active routing policy."""

    raw = str(model or "").strip()
    if not raw:
        return raw

    active_policy = policy or get_provider_governance_policy()
    lower = raw.lower()

    if lower in active_policy.blocked_exact_aliases:
        return raw

    exact_alias = active_policy.exact_aliases.get(lower)
    if exact_alias is not None:
        return exact_alias.canonical_model

    if lower == "codex" or lower.startswith("codex/"):
        return raw
    if _is_codex_alias_model(raw) or _is_codex_family_model(raw):
        return f"codex/{raw.rsplit('/', 1)[-1]}"

    for rule in active_policy.direct_prefix_templates:
        if any(lower.startswith(prefix) for prefix in rule.prefixes):
            return rule.template.format(model=raw)

    if routing_policy == "direct":
        return raw

    if lower.startswith(active_policy.passthrough_prefixes):
        return raw
    if _is_image_generation_model(raw):
        return raw

    if "/" in raw:
        provider = lower.split("/", 1)[0]
        for rule in active_policy.explicit_provider_routes:
            if provider in rule.providers:
                return f"{rule.target_prefix}/{raw}"
        return raw

    for rule in active_policy.bare_model_routes:
        if any(lower.startswith(prefix) for prefix in rule.prefixes):
            return rule.template.format(model=raw)

    return raw


def describe_model_governance(
    model: str,
    routing_policy: RoutingPolicy,
    *,
    policy: ProviderGovernancePolicy | None = None,
) -> dict[str, str] | None:
    """Describe the governance rule that applies to *model*, if any."""

    raw = str(model or "").strip()
    if not raw:
        return None

    active_policy = policy or get_provider_governance_policy()
    lower = raw.lower()

    blocked = active_policy.blocked_exact_aliases.get(lower)
    if blocked is not None:
        return {
            "event": "model_blocked",
            "reason": blocked.reason,
        }

    exact_alias = active_policy.exact_aliases.get(lower)
    if exact_alias is not None:
        return {
            "event": "model_canonicalized",
            "reason": exact_alias.reason,
            "route_class": exact_alias.route_class,
            "canonical_model": exact_alias.canonical_model,
        }

    if lower == "codex" or lower.startswith("codex/"):
        return None
    if _is_codex_alias_model(raw) or _is_codex_family_model(raw):
        canonical = f"codex/{raw.rsplit('/', 1)[-1]}"
        return {
            "event": "model_canonicalized",
            "reason": "Codex-family models must route through the Codex SDK lane.",
            "route_class": "agent_sdk",
            "canonical_model": canonical,
        }

    for rule in active_policy.direct_prefix_templates:
        if any(lower.startswith(prefix) for prefix in rule.prefixes):
            return {
                "event": "model_canonicalized",
                "reason": rule.reason,
                "route_class": rule.route_class,
                "canonical_model": rule.template.format(model=raw),
            }

    if routing_policy == "direct":
        return None

    if lower.startswith(active_policy.passthrough_prefixes) or _is_image_generation_model(raw):
        return None

    if "/" in raw:
        provider = lower.split("/", 1)[0]
        for rule in active_policy.explicit_provider_routes:
            if provider in rule.providers:
                return {
                    "event": "model_canonicalized",
                    "reason": rule.reason,
                    "route_class": rule.route_class,
                    "canonical_model": f"{rule.target_prefix}/{raw}",
                }
        return None

    for rule in active_policy.bare_model_routes:
        if any(lower.startswith(prefix) for prefix in rule.prefixes):
            return {
                "event": "model_canonicalized",
                "reason": rule.reason,
                "route_class": rule.route_class,
                "canonical_model": rule.template.format(model=raw),
            }

    return None


def get_provider_runtime_policy(
    provider: str,
    *,
    policy: ProviderGovernancePolicy | None = None,
) -> ProviderRuntimePolicy:
    """Return shared runtime defaults for *provider*."""

    active_policy = policy or get_provider_governance_policy()
    return active_policy.provider_defaults.get(
        provider,
        ProviderRuntimePolicy(route_class="direct_provider"),
    )


def blocked_model_reason(
    model: str,
    *,
    policy: ProviderGovernancePolicy | None = None,
) -> str | None:
    """Return the configured block reason for *model*, if any."""

    raw = str(model or "").strip().lower()
    if not raw:
        return None
    active_policy = policy or get_provider_governance_policy()
    block_rule = active_policy.blocked_exact_aliases.get(raw)
    if block_rule is None:
        return None
    return block_rule.reason
