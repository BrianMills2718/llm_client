"""Pydantic schema representations of core dataclass types.

LLMCallResult and EmbeddingResult are dataclasses for performance and
simplicity in the hot path.  This module provides parallel Pydantic models
so the @boundary decorator can register proper JSON schemas in the
contract registry without changing the runtime types.

Usage:
    from llm_client.schemas import LLMCallResultSchema
    schema = LLMCallResultSchema.model_json_schema()
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LLMCallResultSchema(BaseModel):
    """Pydantic schema representation of LLMCallResult for contract registry.

    Mirrors every field on the dataclass so the boundary registry has a
    complete JSON schema.  Not used at runtime for validation — the
    dataclass remains the canonical return type.
    """

    content: str = Field(description="LLM response text")
    usage: dict[str, Any] = Field(
        description="Token counts: prompt_tokens, completion_tokens, total_tokens"
    )
    cost: float = Field(description="Cost in USD for this call")
    model: str = Field(description="Model string that was used")
    requested_model: str | None = Field(
        default=None,
        description="Raw model string provided at the public API boundary",
    )
    resolved_model: str | None = Field(
        default=None,
        description="Best-effort model string used for the successful terminal attempt",
    )
    execution_model: str | None = Field(
        default=None,
        description="Alias for resolved terminal model, kept additive for migration clarity",
    )
    routing_trace: dict[str, Any] | None = Field(
        default=None,
        description="Optional routing/fallback trace for contract characterization and debugging",
    )
    tool_calls: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Tool calls if the model invoked tools, else empty list",
    )
    finish_reason: str = Field(
        default="",
        description='Why the model stopped: "stop", "length", "tool_calls", "content_filter", etc.',
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Diagnostic warnings accumulated during retry/fallback/routing",
    )
    warning_records: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Machine-readable warning records (code/category/message/remediation)",
    )
    full_text: str | None = Field(
        default=None,
        description="For agent SDKs: full conversation text (all assistant messages). None for non-agent calls.",
    )
    cost_source: str = Field(
        default="unspecified",
        description="How cost was determined: provider_reported, computed, fallback_estimate, cache_hit, etc.",
    )
    billing_mode: str = Field(
        default="api_metered",
        description="Billing mode: api_metered, subscription_included, or unknown",
    )
    marginal_cost: float | None = Field(
        default=None,
        description="Incremental cost attributed to this call; defaults to cost when omitted",
    )
    cache_hit: bool = Field(
        default=False,
        description="Whether this result came from cache instead of a model call",
    )

    model_config = {"extra": "allow"}


class EmbeddingResultSchema(BaseModel):
    """Pydantic schema representation of EmbeddingResult for contract registry."""

    embeddings: list[list[float]] = Field(
        description="List of embedding vectors (one per input text)"
    )
    usage: dict[str, Any] = Field(
        description="Token counts: prompt_tokens, total_tokens"
    )
    cost: float = Field(description="Cost in USD for this call")
    model: str = Field(description="Model string that was used")

    model_config = {"extra": "allow"}
