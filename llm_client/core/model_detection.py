"""Model detection and classification utilities.

Pure functions for identifying model families, API routing requirements,
and provider-specific behavior. Extracted from client.py for concern
separation; client.py re-exports everything for backward compatibility.

This module depends on config and routing but must not import from client.py.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RESPONSES_API_MODELS = {"gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5.2-pro"}


# ---------------------------------------------------------------------------
# Model family detection
# ---------------------------------------------------------------------------


def _is_claude_model(model: str) -> bool:
    """Check if model string refers to a Claude model."""
    return "claude" in model.lower() or "anthropic" in model.lower()


def _is_thinking_model(model: str) -> bool:
    """Check if model needs thinking budget configuration.

    Gemini 2.5+ thinking models allocate reasoning tokens by default,
    consuming output token budget. Setting budget_tokens=0 disables
    this so all tokens go to the actual response.
    """
    lower = model.lower()
    # Gemini 2.5-flash, 2.5-pro, 2.5-flash-lite, 3.x, 4.x are all thinking models
    return "gemini-2.5" in lower or "gemini-3" in lower or "gemini-4" in lower


def _is_responses_api_model(model: str) -> bool:
    """Check if model requires litellm.responses() instead of completion().

    GPT-5 models use OpenAI's Responses API which has different parameters
    and response format than the Chat Completions API. This function
    detects them so call_llm/acall_llm can route automatically.

    Only bare OpenAI model names match. Provider-prefixed models
    (openrouter/openai/gpt-5, azure/gpt-5, etc.) use Chat Completions API.
    """
    lower = model.lower()
    # Any provider prefix means proxied -> Chat Completions, not Responses API
    if "/" in lower:
        return False
    return lower in _RESPONSES_API_MODELS


def _base_model_name(model: str) -> str:
    """Return the provider-agnostic lowercase model name."""
    return model.lower().rsplit("/", 1)[-1]


def _is_image_generation_model(model: str) -> bool:
    """Best-effort detection for image-generation model families."""
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


# ---------------------------------------------------------------------------
# Gemini detection
# ---------------------------------------------------------------------------


def _is_gemini_model(model: str) -> bool:
    """Check if model targets Google's Gemini API namespace."""
    return model.lower().startswith("gemini/")


# ---------------------------------------------------------------------------
# Model routing resolution
# ---------------------------------------------------------------------------


def _normalize_model_for_routing(model: str) -> str:
    """Route non-Gemini, non-image model IDs through OpenRouter by default.

    Note: with default routing enabled, bare ``gpt-5*`` model IDs are normalized
    to ``openrouter/openai/gpt-5*`` and therefore use completion routing instead
    of OpenAI Responses API. Disable routing via
    ``LLM_CLIENT_OPENROUTER_ROUTING=off`` to keep bare OpenAI IDs.
    """
    from llm_client.core.config import ClientConfig
    from llm_client.core.routing import CallRequest, resolve_call

    cfg = ClientConfig.from_env()
    req = CallRequest(model=model)
    return resolve_call(req, cfg).primary_model


def _resolve_api_base_for_model(
    model: str,
    api_base: str | None,
    config: Any = None,
) -> str | None:
    """Resolve provider API base after model normalization."""
    from llm_client.core.config import ClientConfig
    from llm_client.core.routing import resolve_api_base_for_model

    cfg = config or ClientConfig.from_env()
    return resolve_api_base_for_model(model, api_base, cfg)
