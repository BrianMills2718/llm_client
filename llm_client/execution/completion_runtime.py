"""Completion-path helper utilities for ``llm_client``.

This module owns the helper cluster used by the chat-completions path:
provider-kwargs preparation, provider-hint extraction, first-choice
normalization, and completion-result finalization. Policy helpers are
imported directly from ``call_contracts`` and ``model_detection``.
"""

from __future__ import annotations

import logging
from typing import Any

from llm_client.execution.call_contracts import (
    _apply_max_tokens,
    _coerce_model_incompatible_params,
    _raise_empty_response,
    _resolve_unsupported_param_policy,
    _strip_llm_internal_kwargs,
)
from llm_client.utils.cost_utils import _compute_cost, _extract_tool_calls, _extract_usage, _parse_cost_result
from llm_client.core.data_types import LLMCallResult
from llm_client.core.model_detection import _is_claude_model, _is_responses_api_model, _is_thinking_model
from llm_client.execution.retry import _EMPTY_POLICY_FINISH_REASONS, _EMPTY_TOOL_PROTOCOL_FINISH_REASONS

logger = logging.getLogger(__name__)


def _prepare_call_kwargs(
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout: int,
    num_retries: int = 0,
    reasoning_effort: str | None,
    api_base: str | None,
    kwargs: dict[str, Any],
    warning_sink: list[str] | None = None,
) -> dict[str, Any]:
    """Build kwargs dict shared by sync and async completions calls.

    ``num_retries`` is accepted for call-site compatibility but not used
    here — retry policy is enforced by the execution kernel.
    """

    provider_kwargs = _strip_llm_internal_kwargs(kwargs)
    policy = _resolve_unsupported_param_policy(
        provider_kwargs.pop("unsupported_param_policy", None)
    )
    call_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        **provider_kwargs,
    }
    if timeout > 0:
        call_kwargs["timeout"] = timeout
    if api_base is not None:
        call_kwargs["api_base"] = api_base

    if reasoning_effort and _is_claude_model(model):
        call_kwargs["reasoning_effort"] = reasoning_effort
    elif reasoning_effort:
        logger.debug(
            "reasoning_effort=%s ignored for non-Claude model %s",
            reasoning_effort,
            model,
        )

    if _is_thinking_model(model) and "thinking" not in provider_kwargs:
        is_openrouter = model.lower().startswith("openrouter/")
        if not is_openrouter:
            try:
                from litellm import get_supported_openai_params

                provider = model.split("/")[0] if "/" in model else ""
                supported = get_supported_openai_params(
                    model=model,
                    custom_llm_provider=provider,
                ) or []
                if "thinking" in supported:
                    call_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 0}
            except Exception:
                logger.debug("Thinking probe failed for model=%s", model, exc_info=True)

    _coerce_model_incompatible_params(
        model=model,
        kwargs=call_kwargs,
        policy=policy,
        warning_sink=warning_sink,
    )

    if not _is_responses_api_model(model):
        _apply_max_tokens(model, call_kwargs)

    return call_kwargs


def _provider_hint_from_response(response: Any) -> str | None:
    """Best-effort provider hint from LiteLLM response metadata."""

    hidden = getattr(response, "_hidden_params", None)
    if isinstance(hidden, dict):
        for key in ("custom_llm_provider", "provider", "litellm_provider"):
            value = hidden.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for attr in ("provider", "llm_provider"):
        value = getattr(response, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _first_choice_or_empty_error(
    response: Any,
    *,
    model: str,
    provider: str,
) -> Any:
    """Return first completion choice or raise a typed empty-response error."""

    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        _raise_empty_response(
            provider=provider,
            classification="provider_empty_candidates",
            retryable=True,
            diagnostics={
                "model": model,
                "provider_hint": _provider_hint_from_response(response),
                "has_choices": isinstance(choices, list),
                "choice_count": len(choices) if isinstance(choices, list) else 0,
            },
        )
    return choices[0]


def _build_result_from_response(
    response: Any,
    model: str,
    *,
    warnings: list[str] | None = None,
) -> LLMCallResult:
    """Extract a completion-path response into ``LLMCallResult``."""

    first_choice = _first_choice_or_empty_error(
        response,
        model=model,
        provider="litellm_completion",
    )
    content: str = first_choice.message.content or ""
    finish_reason: str = first_choice.finish_reason or ""
    tool_calls = _extract_tool_calls(first_choice.message)
    usage = _extract_usage(response)
    cost, cost_source = _parse_cost_result(_compute_cost(response))

    if finish_reason == "length":
        raise RuntimeError(
            f"LLM response truncated ({len(content)} chars). Increase max_tokens or simplify the prompt."
        )

    if not content.strip() and not tool_calls:
        finish_norm = str(finish_reason).strip().lower()
        diagnostics = {
            "model": model,
            "provider_hint": _provider_hint_from_response(response),
            "finish_reason": finish_reason or None,
            "has_tool_calls": bool(tool_calls),
        }
        if finish_norm in _EMPTY_POLICY_FINISH_REASONS:
            _raise_empty_response(
                provider="litellm_completion",
                classification="provider_policy_block",
                retryable=False,
                diagnostics=diagnostics,
            )
        if finish_norm in _EMPTY_TOOL_PROTOCOL_FINISH_REASONS:
            _raise_empty_response(
                provider="litellm_completion",
                classification="provider_tool_protocol",
                retryable=False,
                diagnostics=diagnostics,
            )
        _raise_empty_response(
            provider="litellm_completion",
            classification="provider_empty_unknown",
            retryable=True,
            diagnostics=diagnostics,
        )

    logger.debug(
        "LLM call: model=%s tokens=%d cost=$%.6f finish=%s",
        model,
        usage["total_tokens"],
        cost,
        finish_reason,
    )

    return LLMCallResult(
        content=content,
        usage=usage,
        cost=cost,
        model=model,
        resolved_model=model,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        raw_response=response,
        warnings=warnings or [],
        cost_source=cost_source,
    )
