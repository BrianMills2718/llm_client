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

RoutingPolicy = Literal["openrouter", "direct"]


@dataclass(frozen=True)
class ClientConfig:
    """Runtime policy/config resolved once and passed explicitly through calls.

    Default values for call parameters (timeout, retries, etc.) are set here
    so they can be tuned centrally instead of hardcoded in every function
    signature. See Plan #15.
    """

    routing_policy: RoutingPolicy = "openrouter"
    openrouter_api_base: str = OPENROUTER_DEFAULT_API_BASE

    # Call defaults (Plan #15) — these are used when the caller passes None
    default_timeout: int = 60
    default_num_retries: int = 2
    default_base_delay: float = 1.0
    default_max_delay: float = 30.0
    default_max_concurrent: int = 5

    def resolve_timeout(self, timeout: int | None) -> int:
        """Resolve timeout: explicit kwarg wins, else config default."""
        return timeout if timeout is not None else self.default_timeout

    def resolve_num_retries(self, num_retries: int | None) -> int:
        """Resolve num_retries: explicit kwarg wins, else config default."""
        return num_retries if num_retries is not None else self.default_num_retries

    def resolve_base_delay(self, base_delay: float | None) -> float:
        """Resolve base_delay: explicit kwarg wins, else config default."""
        return base_delay if base_delay is not None else self.default_base_delay

    def resolve_max_delay(self, max_delay: float | None) -> float:
        """Resolve max_delay: explicit kwarg wins, else config default."""
        return max_delay if max_delay is not None else self.default_max_delay

    def resolve_max_concurrent(self, max_concurrent: int | None) -> int:
        """Resolve max_concurrent: explicit kwarg wins, else config default."""
        return max_concurrent if max_concurrent is not None else self.default_max_concurrent

    def resolve_defaults(
        self,
        *,
        timeout: int | None = None,
        num_retries: int | None = None,
        base_delay: float | None = None,
        max_delay: float | None = None,
        max_concurrent: int | None = None,
    ) -> dict[str, int | float]:
        """Resolve all call defaults in one shot. Returns a dict of resolved values."""
        return {
            "timeout": timeout if timeout is not None else self.default_timeout,
            "num_retries": num_retries if num_retries is not None else self.default_num_retries,
            "base_delay": base_delay if base_delay is not None else self.default_base_delay,
            "max_delay": max_delay if max_delay is not None else self.default_max_delay,
            "max_concurrent": max_concurrent if max_concurrent is not None else self.default_max_concurrent,
        }

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

        return cls(
            routing_policy=routing_policy,
            openrouter_api_base=os.environ.get(OPENROUTER_API_BASE_ENV, OPENROUTER_DEFAULT_API_BASE),
        )
