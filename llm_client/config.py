"""Typed runtime configuration for llm_client."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

OPENROUTER_ROUTING_ENV = "LLM_CLIENT_OPENROUTER_ROUTING"
OPENROUTER_API_BASE_ENV = "OPENROUTER_API_BASE"
OPENROUTER_DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
RESULT_MODEL_SEMANTICS_ENV = "LLM_CLIENT_RESULT_MODEL_SEMANTICS"

RoutingPolicy = Literal["openrouter", "direct"]
ResultModelSemantics = Literal["legacy", "requested", "resolved"]


@dataclass(frozen=True)
class ClientConfig:
    """Runtime policy/config resolved once and passed explicitly through calls."""

    routing_policy: RoutingPolicy = "openrouter"
    openrouter_api_base: str = OPENROUTER_DEFAULT_API_BASE
    result_model_semantics: ResultModelSemantics = "legacy"

    @classmethod
    def from_env(cls) -> "ClientConfig":
        """Build typed config from environment variables (compat mode)."""
        routing_raw = os.environ.get(OPENROUTER_ROUTING_ENV, "on").strip().lower()
        if routing_raw in {"0", "false", "no", "off"}:
            routing_policy: RoutingPolicy = "direct"
        elif routing_raw in {"1", "true", "yes", "on", ""}:
            routing_policy = "openrouter"
        else:
            logger.warning(
                "Invalid %s=%r; expected on/off boolean. Defaulting to on.",
                OPENROUTER_ROUTING_ENV,
                routing_raw,
            )
            routing_policy = "openrouter"

        semantics_raw = os.environ.get(RESULT_MODEL_SEMANTICS_ENV, "legacy").strip().lower()
        if semantics_raw in {"legacy", "requested", "resolved"}:
            result_model_semantics: ResultModelSemantics = semantics_raw  # type: ignore[assignment]
        else:
            logger.warning(
                "Invalid %s=%r; expected legacy/requested/resolved. Defaulting to legacy.",
                RESULT_MODEL_SEMANTICS_ENV,
                semantics_raw,
            )
            result_model_semantics = "legacy"

        return cls(
            routing_policy=routing_policy,
            openrouter_api_base=os.environ.get(OPENROUTER_API_BASE_ENV, OPENROUTER_DEFAULT_API_BASE),
            result_model_semantics=result_model_semantics,
        )
