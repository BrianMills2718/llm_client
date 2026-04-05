"""Pure routing/normalization resolver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm_client.core.config import ClientConfig, RoutingPolicy
from llm_client.core.provider_policy import (
    canonicalize_model_for_policy,
    describe_model_governance,
)


@dataclass(frozen=True)
class CallRequest:
    """Input contract for routing resolution."""

    model: str
    fallback_models: list[str] | None = None
    api_base: str | None = None


@dataclass(frozen=True)
class ResolvedCallPlan:
    """Resolved execution plan produced by pure routing logic."""

    requested_model: str
    models: list[str]
    primary_model: str
    fallback_models: list[str] = field(default_factory=list)
    routing_policy: RoutingPolicy = "openrouter"
    requested_api_base: str | None = None
    routing_trace: dict[str, Any] = field(default_factory=dict)


def routing_policy_label(policy: RoutingPolicy) -> str:
    return "openrouter_on" if policy == "openrouter" else "openrouter_off"


def normalize_model_for_policy(model: str, policy: RoutingPolicy) -> str:
    """Normalize model IDs according to explicit routing policy."""
    return canonicalize_model_for_policy(model, policy)


def resolve_api_base_for_model(
    model: str,
    requested_api_base: str | None,
    config: ClientConfig,
) -> str | None:
    """Resolve effective api_base for a model under typed config."""
    if requested_api_base is not None:
        return requested_api_base
    if str(model or "").strip().lower().startswith("openrouter/"):
        return config.openrouter_api_base
    return None


def resolve_call(request: CallRequest, config: ClientConfig) -> ResolvedCallPlan:
    """Pure routing resolver with deterministic output from inputs only."""
    requested_model = str(request.model or "").strip()
    candidates = [requested_model] + list(request.fallback_models or [])

    models: list[str] = []
    seen: set[str] = set()
    normalized_events: list[dict[str, str]] = []
    provider_governance_events: list[dict[str, str]] = []

    for candidate in candidates:
        raw = str(candidate or "").strip()
        if not raw:
            continue
        normalized = normalize_model_for_policy(raw, config.routing_policy)
        if normalized != raw:
            normalized_events.append({"from": raw, "to": normalized})
            governance_event = describe_model_governance(raw, config.routing_policy)
            if governance_event is not None:
                provider_governance_events.append(
                    {
                        **governance_event,
                        "from": raw,
                        "to": normalized,
                    }
                )
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        models.append(normalized)

    if not models:
        normalized = normalize_model_for_policy(requested_model, config.routing_policy)
        models = [normalized]
        if normalized != requested_model:
            normalized_events.append({"from": requested_model, "to": normalized})

    primary_model = models[0]
    fallback_models = models[1:]

    trace: dict[str, Any] = {
        "routing_policy": routing_policy_label(config.routing_policy),
        "attempted_models": list(models),
    }
    if normalized_events:
        trace["normalization_events"] = normalized_events
    if provider_governance_events:
        trace["provider_governance_events"] = provider_governance_events
    if requested_model != primary_model:
        trace["normalized_from"] = requested_model
        trace["normalized_to"] = primary_model

    return ResolvedCallPlan(
        requested_model=requested_model,
        models=models,
        primary_model=primary_model,
        fallback_models=fallback_models,
        routing_policy=config.routing_policy,
        requested_api_base=request.api_base,
        routing_trace=trace,
    )
