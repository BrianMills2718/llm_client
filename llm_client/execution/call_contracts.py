"""Pre-call contract helpers for ``llm_client``.

This module centralizes the invariants that must hold before any provider or
agent SDK dispatch happens:

1. every call has a resolved ``task``,
2. every call has a resolved ``trace_id``,
3. every call has a pre-flight budget check,
4. prompt asset provenance is validated before it enters observability,
5. retry-safety policy is derived consistently for agent SDK calls,
6. execution-mode and model/kwargs capability validation,
7. unsupported-param coercion, agent-only kwargs filtering,
8. model deprecation warnings and empty-response error classification.

These checks belong to the runtime substrate itself, not to any one transport
backend. Keeping them in one module makes the boundary easier to reason about
and easier to test without dragging the full client runtime with it.
"""

from __future__ import annotations

import json as _json
import logging
import os
import uuid
from typing import Any, Literal, NoReturn

import litellm

from llm_client import io_log as _io_log
from llm_client.core.errors import (
    LLMBudgetExceededError,
    LLMCapabilityError,
    LLMEmptyResponseError,
    LLMModelNotFoundError,
)
from llm_client.core.model_detection import (
    _base_model_name,
    _is_responses_api_model,
)
from llm_client.prompt_assets import parse_prompt_ref

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tag / budget / retry-safety contracts (original surface)
# ---------------------------------------------------------------------------

REQUIRE_TAGS_ENV = "LLM_CLIENT_REQUIRE_TAGS"
AGENT_RETRY_SAFE_ENV = "LLM_CLIENT_AGENT_RETRY_SAFE"


def truthy_env(value: Any) -> bool:
    """Parse common truthy env-style values."""
    if isinstance(value, bool):
        return value
    raw = str(value or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def tags_strict_mode(task: str | None) -> bool:
    """Whether missing task/trace/budget tags should raise instead of defaulting."""
    if truthy_env(os.environ.get(REQUIRE_TAGS_ENV)):
        return True
    if truthy_env(os.environ.get("CI")):
        return True
    normalized_task = str(task or "").strip().lower()
    return normalized_task.startswith(("benchmark", "bench", "eval", "ci"))


def normalize_prompt_ref(prompt_ref: str | None) -> str | None:
    """Validate prompt asset identity before it enters observability."""
    if prompt_ref is None:
        return None
    normalized = str(prompt_ref).strip()
    if not normalized:
        raise ValueError("prompt_ref must not be empty when provided.")
    return parse_prompt_ref(normalized).prompt_ref


def require_tags(
    task: str | None,
    trace_id: str | None,
    max_budget: float | None,
    *,
    caller: str,
) -> tuple[str, str, float, list[str]]:
    """Resolve observability tags and enforce shared guardrails.

    In strict mode, missing values fail loudly. Outside strict mode, the
    substrate fills in conservative defaults and emits warnings so the call is
    still observable and queryable.
    """
    missing: list[str] = []
    if not task:
        missing.append("task")
    if not trace_id:
        missing.append("trace_id")
    if max_budget is None:
        missing.append("max_budget")

    if tags_strict_mode(task) and missing:
        raise ValueError(
            f"Missing required kwargs: {', '.join(missing)}. "
            "Strict tag enforcement is enabled "
            f"(set {REQUIRE_TAGS_ENV}=0 to disable outside CI/benchmark)."
        )

    resolved_task = str(task).strip() if task else "adhoc"
    resolved_trace_id = (
        str(trace_id).strip() if trace_id else f"auto/{caller}/{uuid.uuid4().hex[:12]}"
    )
    if max_budget is None:
        resolved_max_budget = 0.0
    else:
        try:
            resolved_max_budget = float(max_budget)
        except (TypeError, ValueError):
            raise ValueError(f"max_budget must be numeric, got {max_budget!r}") from None

    auto_warnings: list[str] = []
    if not task:
        auto_warnings.append("AUTO_TAG: task=adhoc")
    if not trace_id:
        auto_warnings.append(f"AUTO_TAG: trace_id={resolved_trace_id}")
    if max_budget is None:
        auto_warnings.append("AUTO_TAG: max_budget=0 (unlimited)")

    _io_log.enforce_feature_profile(resolved_task, caller="llm_client.client")
    _io_log.enforce_experiment_context(resolved_task, caller="llm_client.client")
    return resolved_task, resolved_trace_id, resolved_max_budget, auto_warnings


def check_budget(trace_id: str, max_budget: float) -> None:
    """Raise before dispatch if a trace has already spent its budget."""
    if max_budget <= 0:
        return
    spent = _io_log.get_cost(trace_id=trace_id)
    if spent >= max_budget:
        raise LLMBudgetExceededError(
            f"Budget exceeded for trace {trace_id}: "
            f"${spent:.4f} spent >= ${max_budget:.4f} limit"
        )


def agent_retry_safe_enabled(explicit: Any | None) -> bool:
    """Whether retries on agent SDK calls are allowed."""
    if explicit is not None:
        return truthy_env(explicit)
    return truthy_env(os.environ.get(AGENT_RETRY_SAFE_ENV))


# ---------------------------------------------------------------------------
# Empty-response classification and schema-error detection
# ---------------------------------------------------------------------------


def _compact_diagnostics(diagnostics: dict[str, Any], *, max_len: int = 600) -> str:
    """Render diagnostics dict into a bounded JSON string for errors/logging."""
    try:
        rendered = _json.dumps(diagnostics, sort_keys=True, ensure_ascii=True, default=str)
    except Exception:
        rendered = str(diagnostics)
    if len(rendered) <= max_len:
        return rendered
    return rendered[:max_len] + "...(truncated)"


def _raise_empty_response(
    *,
    provider: str,
    classification: str,
    retryable: bool,
    diagnostics: dict[str, Any],
) -> NoReturn:
    """Raise typed empty-response error with structured diagnostics."""
    payload = dict(diagnostics)
    payload["provider"] = provider
    payload["classification"] = classification
    payload["retryable"] = retryable
    message = (
        f"Empty content from LLM [{provider}:{classification} retryable={retryable}] "
        f"diagnostics={_compact_diagnostics(payload)}"
    )
    raise LLMEmptyResponseError(
        message,
        retryable=retryable,
        classification=classification,
        diagnostics=payload,
    )


class _NativeSchemaFallback(Exception):
    """Signal native-schema rejection and trigger instructor fallback."""


# Patterns indicating the provider rejected the JSON schema itself (not a
# transient error).  When detected in the native JSON-schema path, the call
# falls back to the instructor path which prompts for JSON instead of
# enforcing via API-level schema constraints.
_SCHEMA_ERROR_PATTERNS: list[str] = [
    "nesting depth",
    "schema is invalid",
    "schema exceeds",
    "invalid schema",
    "unsupported schema",
    "schema too complex",
    "schema validation",
    "not a valid json schema",
    "response_format",
]


def _is_schema_error(error: Exception) -> bool:
    """Check if an error indicates the provider rejected the response schema."""
    error_str = str(error).lower()
    # Must be a 400-class error (BadRequest), not a transient/server error
    error_type = type(error).__name__.lower()
    is_bad_request = "badrequest" in error_type or "invalid_argument" in error_str or "400" in error_str
    if not is_bad_request:
        return False
    return any(p in error_str for p in _SCHEMA_ERROR_PATTERNS)


# ---------------------------------------------------------------------------
# GPT-5 / Responses API sampling and param-policy constants
# ---------------------------------------------------------------------------

_GPT5_ALWAYS_STRIP_SAMPLING = {"gpt-5", "gpt-5-mini", "gpt-5-nano"}
_GPT5_REASONING_GATED_SAMPLING = {
    "gpt-5.1",
    "gpt-5.2",
    "gpt-5.2-pro",
    "gpt-5.1-chat-latest",
    "gpt-5.2-chat-latest",
}
# Models that support long-thinking (5-10 min) and need background polling
_LONG_THINKING_MODELS = {"gpt-5.2-pro"}
_LONG_THINKING_REASONING_EFFORTS = {"high", "xhigh"}
_GPT5_SAMPLING_PARAMS = ("temperature", "top_p", "logprobs", "top_logprobs")
_UNSUPPORTED_PARAM_POLICY_ENV = "LLM_CLIENT_UNSUPPORTED_PARAM_POLICY"
_UNSUPPORTED_PARAM_POLICIES = frozenset({"coerce_and_warn", "coerce_silent", "error"})
_UNSUPPORTED_PARAM_POLICY_ALIASES = {
    "warn": "coerce_and_warn",
    "coerce": "coerce_and_warn",
    "silent": "coerce_silent",
    "strict": "error",
    "raise": "error",
    "error_only": "error",
}


# ---------------------------------------------------------------------------
# Param coercion and stripping
# ---------------------------------------------------------------------------


def _strip_incompatible_sampling_params(model: str, call_kwargs: dict[str, Any]) -> list[str]:
    """Drop sampling params that are unsupported for GPT-5 family variants.

    GPT-5 legacy models reject sampling controls entirely in many reasoning
    configurations. Keeping this normalization at the client layer avoids
    provider-specific 400s and silent retries when callers pass generic kwargs.
    """
    base = _base_model_name(model)
    reasoning_effort = str(call_kwargs.get("reasoning_effort", "")).strip().lower()

    should_strip = False
    if base in _GPT5_ALWAYS_STRIP_SAMPLING:
        should_strip = True
    elif base in _GPT5_REASONING_GATED_SAMPLING and reasoning_effort and reasoning_effort != "none":
        should_strip = True

    if not should_strip:
        return []

    removed: list[str] = []
    for key in _GPT5_SAMPLING_PARAMS:
        if key in call_kwargs:
            call_kwargs.pop(key, None)
            removed.append(key)
    return removed


def _resolve_unsupported_param_policy(explicit_policy: Any) -> str:
    """Resolve the unsupported-param policy from explicit arg or env."""
    raw = explicit_policy
    if raw is None:
        raw = os.environ.get(_UNSUPPORTED_PARAM_POLICY_ENV, "coerce_and_warn")
    policy = str(raw).strip().lower()
    policy = _UNSUPPORTED_PARAM_POLICY_ALIASES.get(policy, policy)
    if policy not in _UNSUPPORTED_PARAM_POLICIES:
        allowed = ", ".join(sorted(_UNSUPPORTED_PARAM_POLICIES))
        raise ValueError(
            f"Invalid unsupported_param_policy={raw!r}. "
            f"Allowed: {allowed} (or aliases: {', '.join(sorted(_UNSUPPORTED_PARAM_POLICY_ALIASES))})"
        )
    return policy


def _coerce_model_incompatible_params(
    *,
    model: str,
    kwargs: dict[str, Any],
    policy: str,
    warning_sink: list[str] | None = None,
) -> list[str]:
    """Normalize unsupported params and emit loud diagnostics."""
    removed: list[str] = []

    # Bare GPT-5 models route via responses API and reject temperature.
    if _is_responses_api_model(model) and "temperature" in kwargs:
        kwargs.pop("temperature", None)
        removed.append("temperature")

    # GPT-5 family sampling incompatibilities across providers/completions.
    removed.extend(_strip_incompatible_sampling_params(model, kwargs))

    if not removed:
        return []

    removed_unique = sorted(set(removed))
    detail = (
        f"COERCE_PARAMS model={model} policy={policy} "
        f"removed={','.join(removed_unique)} "
        f"rule=gpt5_sampling_compatibility"
    )
    if policy == "error":
        raise LLMCapabilityError(
            f"Unsupported params for model {model}: {', '.join(removed_unique)}. "
            "Use unsupported_param_policy='coerce_and_warn' to auto-coerce."
        )
    if policy == "coerce_and_warn":
        logger.warning(detail)
        if warning_sink is not None:
            warning_sink.append(detail)
    else:
        logger.info(detail)
    return removed_unique


def _strip_llm_internal_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop llm_client-internal keys from a kwargs mapping.

    Internal keys are conventionally prefixed with ``_`` and used for private
    control flow (for example ``_lifecycle_monitor``). They are not part of the
    provider contract and are intentionally excluded from cache keying and
    provider payloads.
    """
    return {k: v for k, v in kwargs.items() if not k.startswith("_")}


def _apply_max_tokens(model: str, call_kwargs: dict[str, Any]) -> None:
    """Clamp explicit output-token caps to the model maximum when present.

    The client does not invent output-token ceilings when callers omit them.
    Defaulting to the provider maximum turns routine calls into accidental
    high-cost requests, especially for structured-output workloads where a
    large generated cap is not the same as a useful response. When callers do
    supply an explicit cap, this helper only prevents provider-side
    ``max_tokens > model_max`` validation errors.

    Silently skips if model info lookup fails (unknown/custom models).
    """
    try:
        info = litellm.get_model_info(model)
    except Exception:
        return  # Unknown model — pass through unchanged

    model_max = info.get("max_output_tokens")
    if not model_max:
        return

    # Determine which key the caller used (if any)
    token_key = None
    for key in ("max_completion_tokens", "max_tokens"):
        if key in call_kwargs:
            token_key = key
            break

    if token_key:
        # Clamp to model's max
        if call_kwargs[token_key] > model_max:
            logger.debug(
                "Clamping %s from %d to %d for %s",
                token_key, call_kwargs[token_key], model_max, model,
            )
            call_kwargs[token_key] = model_max
    else:
        return


# ---------------------------------------------------------------------------
# Agent model detection and execution-mode contracts
# ---------------------------------------------------------------------------

_CODEX_AGENT_ALIASES: frozenset[str] = frozenset({"codex-mini-latest"})


def _is_agent_model(model: str) -> bool:
    """Check if model routes to an agent SDK instead of litellm.

    Agent models like "claude-code" or "claude-code/opus" use the Claude
    Agent SDK. "openai-agents/*" is reserved for future OpenAI Agents SDK.
    """
    lower = model.lower()
    for prefix in ("claude-code", "codex", "openai-agents"):
        if lower == prefix or lower.startswith(prefix + "/"):
            return True
    # Support selected Codex aliases that map to Codex agent SDK models.
    if lower in _CODEX_AGENT_ALIASES:
        return True
    return False


ExecutionMode = Literal["text", "structured", "workspace_agent", "workspace_tools"]
_VALID_EXECUTION_MODES: frozenset[str] = frozenset(
    {"text", "structured", "workspace_agent", "workspace_tools"}
)
_AGENT_ONLY_KWARGS: frozenset[str] = frozenset(
    {
        "allowed_tools",
        "agent_idle_timeout",
        "cwd",
        "max_turns",
        "max_tool_calls",
        "permission_mode",
        "max_budget_usd",
        "sandbox_mode",
        "working_directory",
        "approval_policy",
        "model_reasoning_effort",
        "network_access_enabled",
        "web_search_enabled",
        "additional_directories",
        "skip_git_repo_check",
        "yolo_mode",
        "codex_home",
    }
)


def _validate_execution_contract(
    *,
    models: list[str],
    execution_mode: str,
    kwargs: dict[str, Any],
    caller: str,
) -> None:
    """Validate model/kwargs capability compatibility before dispatch."""
    if execution_mode not in _VALID_EXECUTION_MODES:
        valid = ", ".join(sorted(_VALID_EXECUTION_MODES))
        raise ValueError(f"Invalid execution_mode={execution_mode!r}. Valid values: {valid}")

    if execution_mode == "workspace_agent":
        non_agent = [m for m in models if not _is_agent_model(m)]
        if non_agent:
            raise LLMCapabilityError(
                f"{caller}: execution_mode='workspace_agent' requires agent models "
                f"(codex/claude-code/openai-agents). Incompatible models: {non_agent}"
            )

    if execution_mode == "workspace_tools":
        agent_models = [m for m in models if _is_agent_model(m)]
        if agent_models:
            raise LLMCapabilityError(
                f"{caller}: execution_mode='workspace_tools' requires non-agent models. "
                f"Incompatible models: {agent_models}"
            )
        if not any(k in kwargs for k in ("python_tools", "mcp_servers", "mcp_sessions")):
            raise LLMCapabilityError(
                f"{caller}: execution_mode='workspace_tools' requires python_tools "
                "or mcp_servers/mcp_sessions."
            )

    # max_turns/max_tool_calls are valid for non-agent models when using MCP/python_tools
    has_tool_loop = any(k in kwargs for k in ("mcp_servers", "mcp_sessions", "python_tools"))
    check_set = _AGENT_ONLY_KWARGS - {"max_turns", "max_tool_calls"} if has_tool_loop else _AGENT_ONLY_KWARGS
    agent_only = sorted(k for k in kwargs if k in check_set)
    if agent_only:
        non_agent = [m for m in models if not _is_agent_model(m)]
        agent_models = [m for m in models if _is_agent_model(m)]
        if non_agent and not agent_models:
            raise LLMCapabilityError(
                f"{caller}: agent-only kwargs {agent_only} are incompatible with "
                f"non-agent model(s) {non_agent}. Use codex/claude-code or remove "
                "agent-only kwargs."
            )
        if non_agent and agent_models:
            logger.warning(
                "%s: mixed agent/non-agent fallback chain detected; agent-only kwargs %s "
                "will be ignored on non-agent fallback legs.",
                caller,
                agent_only,
            )


def _coerce_model_kwargs_for_execution(
    *,
    current_model: str,
    kwargs: dict[str, Any],
    warning_sink: list[str] | None,
) -> dict[str, Any]:
    """Strip kwargs unsupported for the current execution leg.

    This enables mixed agent/non-agent fallback chains by removing agent-only
    kwargs when executing non-agent models.
    """
    # Drop llm_client internal runtime kwargs from all execution paths.
    # They are injected for orchestration/observability and should never be
    # hashed into cache keys or sent into provider/SDK payloads.
    internal_removed = sorted(k for k in kwargs if k.startswith("_"))

    if _is_agent_model(current_model):
        return {k: v for k, v in kwargs.items() if k not in internal_removed}

    removed = sorted(k for k in kwargs if k in _AGENT_ONLY_KWARGS)
    all_removed = sorted(set([*internal_removed, *removed]))
    if not all_removed:
        return kwargs

    model_kwargs = dict(kwargs)
    for key in all_removed:
        model_kwargs.pop(key, None)

    if not removed:
        return model_kwargs

    detail = (
        f"COERCE_PARAMS model={current_model} policy=coerce_and_warn "
        f"removed={','.join(removed)} "
        "rule=agent_fallback_compatibility"
    )
    logger.warning(detail)
    if warning_sink is not None:
        warning_sink.append(detail)
    return model_kwargs


# ---------------------------------------------------------------------------
# Model deprecation warnings
# ---------------------------------------------------------------------------

# Models that are outclassed on both price and quality by newer alternatives.
# Key: model substring (matched case-insensitively against the model string).
# Value: (replacement suggestion, reason).
# Checked at every call_llm / stream_llm entry point.
_DEPRECATED_MODELS: dict[str, tuple[str, str]] = {
    "gpt-4o-mini": (
        "deepseek/deepseek-chat OR gemini/gemini-2.5-flash",
        "GPT-4o-mini (intel 30, $0.15/$0.60) is outclassed by DeepSeek V3.2 "
        "(intel 42, $0.28/$0.42) and MiMo-V2-Flash (intel 41, $0.15 blended). "
        "Both are smarter AND cheaper.",
    ),
    # gpt-4o moved to _WARNED_MODELS (warn-only, never banned)
    "o1-mini": (
        "o3-mini",
        "o1-mini is deprecated. Use o3-mini for reasoning tasks.",
    ),
    "o4-mini": (
        "o3-mini",
        "o4-mini was retired by OpenAI on Feb 16, 2026. Use o3-mini "
        "for reasoning tasks or gpt-5-mini for general tasks.",
    ),
    "o1-pro": (
        "o3",
        "o1-pro ($150/$600) is superseded by o3 ($2/$8) which is better at "
        "reasoning at a fraction of the cost.",
    ),
    "gemini-1.5": (
        "gemini/gemini-2.5-flash OR gemini/gemini-2.5-pro",
        "All Gemini 1.5 models are superseded by 2.5+ equivalents at the "
        "same price with better quality. Use gemini-2.5-flash or gemini-2.5-pro.",
    ),
    "gemini-2.0-flash": (
        "gemini/gemini-2.5-flash",
        "Gemini 2.0 Flash is superseded by 2.5 Flash at the same price with "
        "significantly better quality.",
    ),
    "claude-3-5": (
        "anthropic/claude-sonnet-4-5-20250929 OR anthropic/claude-haiku-4-5-20251001",
        "Claude 3.5 models are superseded by 4.5 equivalents at the same price "
        "with better quality.",
    ),
    "claude-3-opus": (
        "anthropic/claude-opus-4-6",
        "Claude 3 Opus is superseded by Opus 4.5/4.6 at a lower price with "
        "dramatically better quality.",
    ),
    "claude-3-sonnet": (
        "anthropic/claude-sonnet-4-5-20250929",
        "Claude 3 Sonnet is superseded by Sonnet 4.5 at the same price with "
        "much better quality.",
    ),
    "claude-3-haiku": (
        "anthropic/claude-haiku-4-5-20251001",
        "Claude 3 Haiku is superseded by Haiku 4.5 at the same price with "
        "much better quality.",
    ),
    "mistral-large": (
        "deepseek/deepseek-chat OR gemini/gemini-2.5-flash",
        "Mistral Large (intel ~27, $2.75 blended) is dramatically overpriced "
        "for its quality. DeepSeek V3.2 (intel 42, $0.32) is 8x cheaper and smarter.",
    ),
}

# Models that are outclassed but still usable — warn loudly, never ban.
# Same format as _DEPRECATED_MODELS. Useful for benchmarking against baselines.
_WARNED_MODELS: dict[str, tuple[str, str]] = {
    "gpt-4o": (
        "gpt-5",
        "GPT-4o ($2.50/$10) is outclassed by GPT-5 ($1.25/$10) — "
        "GPT-5 is cheaper and smarter. Consider switching.",
    ),
}

# Models that match a deprecated pattern but should NOT be flagged
_DEPRECATED_MODEL_EXCEPTIONS: set[str] = {
    "gpt-4o-mini",  # has its own entry — prevent double-match from gpt-4o
    "gemini-2.0-flash-lite",  # NOT deprecated — cheapest Google model, no 2.5 equivalent
}


def _check_model_deprecation(model: str) -> None:
    """Emit a loud warning (or raise) if the model is outclassed.

    The warning text is intentionally aggressive because it may be read by
    an LLM agent that needs to STOP and ask the user before proceeding.

    Set ``LLM_CLIENT_STRICT_MODELS=1`` to raise ``LLMModelNotFoundError``
    instead of warning. This is recommended for benchmarks and CI.
    """
    lower = model.lower()
    strict = os.environ.get("LLM_CLIENT_STRICT_MODELS", "").strip() == "1"
    for pattern, (replacement, reason) in _DEPRECATED_MODELS.items():
        if pattern in lower:
            # Check exceptions (e.g., don't flag gpt-4o-mini under gpt-4o)
            if any(exc in lower and exc != pattern for exc in _DEPRECATED_MODEL_EXCEPTIONS):
                continue
            if strict:
                raise LLMModelNotFoundError(
                    f"BANNED MODEL: {model}. {reason} Use instead: {replacement}. "
                    f"(LLM_CLIENT_STRICT_MODELS=1 is set — deprecated models are blocked.)"
                )
            warning_msg = (
                f"\n{'=' * 72}\n"
                f"WARNING: DEPRECATED MODEL DETECTED: {model}\n"
                f"{'=' * 72}\n"
                f"\n"
                f"STOP. DO NOT USE THIS MODEL UNLESS THE USER HAS SPECIFICALLY "
                f"TOLD YOU TO AND HAS CONFIRMED THEY UNDERSTAND WHY NOT TO USE IT. "
                f"YOU MUST GET USER PERMISSION. STOP NOW AND ASK IF YOU DON'T HAVE IT.\n"
                f"\n"
                f"Reason: {reason}\n"
                f"\n"
                f"Use instead: {replacement}\n"
                f"{'=' * 72}\n"
            )
            logger.warning(warning_msg)
            import warnings
            warnings.warn(warning_msg, DeprecationWarning, stacklevel=3)
            return
    # Warned models: loud warning but never banned, even in strict mode
    for pattern, (replacement, reason) in _WARNED_MODELS.items():
        if pattern in lower:
            if any(exc in lower and exc != pattern for exc in _DEPRECATED_MODEL_EXCEPTIONS):
                continue
            warning_msg = (
                f"\n{'=' * 72}\n"
                f"WARNING: OUTCLASSED MODEL: {model}\n"
                f"{'=' * 72}\n"
                f"Reason: {reason}\n"
                f"Use instead: {replacement}\n"
                f"{'=' * 72}\n"
            )
            logger.warning(warning_msg)
            import warnings
            warnings.warn(warning_msg, UserWarning, stacklevel=3)
            return
