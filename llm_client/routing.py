"""Pure routing/normalization resolver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm_client.config import ClientConfig, RoutingPolicy


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


def _base_model_name(model: str) -> str:
    return model.lower().rsplit("/", 1)[-1]


def _is_image_generation_model(model: str) -> bool:
    base = _base_model_name(model)
    hints = (
        "gpt-image",
        "dall-e",
        "imagen",
        "stable-diffusion",
        "sdxl",
        "flux",
    )
    return any(h in base for h in hints)


def normalize_model_for_policy(model: str, policy: RoutingPolicy) -> str:
    """Normalize model IDs according to explicit routing policy."""
    raw = str(model or "").strip()
    if not raw:
        return raw
    if policy == "direct":
        return raw

    lower = raw.lower()
    if lower.startswith(("openrouter/", "gemini/")):
        return raw
    if lower.startswith(("codex", "claude-code", "openai-agents")):
        return raw
    if _is_image_generation_model(raw):
        return raw

    # Explicit provider/model IDs.
    if "/" in raw:
        provider = lower.split("/", 1)[0]
        if provider in {"openai", "anthropic", "deepseek", "x-ai", "xai", "mistral", "mistralai"}:
            return f"openrouter/{raw}"
        return raw

    # Bare model IDs.
    if lower.startswith(("gpt-", "o1", "o3", "o4", "chatgpt", "text-embedding-", "text-moderation-")):
        return f"openrouter/openai/{raw}"
    if lower.startswith("claude"):
        return f"openrouter/anthropic/{raw}"
    if lower.startswith("deepseek"):
        return f"openrouter/deepseek/{raw}"
    if lower.startswith("grok"):
        return f"openrouter/x-ai/{raw}"
    if lower.startswith("mistral"):
        return f"openrouter/mistralai/{raw}"
    return raw


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

    for candidate in candidates:
        raw = str(candidate or "").strip()
        if not raw:
            continue
        normalized = normalize_model_for_policy(raw, config.routing_policy)
        if normalized != raw:
            normalized_events.append({"from": raw, "to": normalized})
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
