"""Process-local model availability tracking for recent provider exhaustion.

This module intentionally keeps only lightweight in-memory state. It exists to
prevent long-running jobs from repeatedly probing models that just failed with a
known quota/spend-cap exhaustion signal.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any

from llm_client.core.errors import LLMQuotaExhaustedError, _QUOTA_PATTERNS, classify_error


@dataclass
class ModelUnavailabilityRecord:
    """Temporary suppression record for a model that recently exhausted."""

    model: str
    reason: str
    expires_at_monotonic: float
    detail: str


_MODEL_UNAVAILABILITY: dict[str, ModelUnavailabilityRecord] = {}


def _cooldown_seconds(env_name: str, default: float) -> float:
    """Return a positive cooldown duration from environment or default."""
    raw = os.environ.get(env_name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return default if value <= 0 else value


def _retry_hint_cap_seconds() -> float:
    """Return the max provider retry hint to convert into cooldown."""
    return _cooldown_seconds("LLM_CLIENT_PROVIDER_RETRY_HINT_COOLDOWN_CAP_S", 43200.0)


def _provider_retry_hint_seconds(detail: str) -> float | None:
    """Extract an unbounded provider retry-delay hint from raw error detail."""
    patterns = (
        r'"retryDelay"\s*:\s*"([0-9]+(?:\.[0-9]+)?)(ms|s|m|h)?"',
        r'retry(?:ing)?\s+(?:in|after)\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hour|hours)?',
    )
    for pattern in patterns:
        match = re.search(pattern, detail, flags=re.IGNORECASE)
        if not match:
            continue
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        unit = (match.group(2) or "s").strip().lower()
        if unit == "ms":
            value /= 1000.0
        elif unit in {"m", "min", "mins", "minute", "minutes"}:
            value *= 60.0
        elif unit in {"h", "hour", "hours"}:
            value *= 3600.0
        if value > 0:
            return value
    return None


def _apply_provider_retry_hint(default_cooldown_s: float, detail: str) -> float:
    """Honor provider retry hints when they exceed the default cooldown."""
    hint_s = _provider_retry_hint_seconds(detail)
    if hint_s is None:
        return default_cooldown_s
    return max(default_cooldown_s, min(hint_s, _retry_hint_cap_seconds()))


def _unavailability_reason(detail: str) -> tuple[str, float] | None:
    """Classify quota exhaustion detail into a cooldown reason and duration."""
    lower = detail.lower()
    if not any(pattern in lower for pattern in _QUOTA_PATTERNS):
        return None
    if any(
        pattern in lower
        for pattern in (
            "requests per day",
            "per day per project",
            "per day per model",
            "generate_requests_per_model_per_day",
            "generaterequestsperdayperprojectpermodel",
            "generatecontentrequestsperday",
            "try again tomorrow",
        )
    ):
        return (
            "provider_daily_quota_exhausted",
            _apply_provider_retry_hint(
                _cooldown_seconds("LLM_CLIENT_DAILY_QUOTA_COOLDOWN_S", 3600.0),
                detail,
            ),
        )
    if any(pattern in lower for pattern in ("monthly spending cap", "monthly spend cap", "spending cap", "spend cap")):
        return (
            "provider_spend_cap_exhausted",
            _apply_provider_retry_hint(
                _cooldown_seconds("LLM_CLIENT_SPEND_CAP_COOLDOWN_S", 300.0),
                detail,
            ),
        )
    return (
        "provider_quota_exhausted",
        _apply_provider_retry_hint(
            _cooldown_seconds("LLM_CLIENT_QUOTA_COOLDOWN_S", 900.0),
            detail,
        ),
    )


def record_model_unavailability(
    model: str,
    error: Exception,
    *,
    now_monotonic: float | None = None,
) -> dict[str, Any] | None:
    """Record temporary model unavailability for quota/spend-cap exhaustion.

    Returns a summary dict when a record is created or extended, otherwise
    returns ``None``.
    """

    if classify_error(error) is not LLMQuotaExhaustedError:
        return None

    detail = str(error).strip() or type(error).__name__
    reason_payload = _unavailability_reason(detail)
    if reason_payload is None:
        return None
    reason, cooldown_s = reason_payload
    now_value = time.monotonic() if now_monotonic is None else now_monotonic
    expires_at = now_value + cooldown_s
    key = str(model).strip().lower()
    existing = _MODEL_UNAVAILABILITY.get(key)
    if existing is not None and existing.expires_at_monotonic >= expires_at:
        expires_at = existing.expires_at_monotonic

    record = ModelUnavailabilityRecord(
        model=str(model).strip(),
        reason=reason,
        expires_at_monotonic=expires_at,
        detail=detail,
    )
    _MODEL_UNAVAILABILITY[key] = record
    return {
        "model": record.model,
        "reason": record.reason,
        "cooldown_s": max(0.0, round(record.expires_at_monotonic - now_value, 3)),
    }


def filter_available_models(
    models: list[str],
    *,
    now_monotonic: float | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Return the available subset of *models* and suppression metadata."""

    now_value = time.monotonic() if now_monotonic is None else now_monotonic
    available: list[str] = []
    suppressed: list[dict[str, Any]] = []

    expired = [
        key for key, record in _MODEL_UNAVAILABILITY.items()
        if record.expires_at_monotonic <= now_value
    ]
    for key in expired:
        _MODEL_UNAVAILABILITY.pop(key, None)

    for model in models:
        key = str(model).strip().lower()
        record = _MODEL_UNAVAILABILITY.get(key)
        if record is None:
            available.append(model)
            continue
        suppressed.append(
            {
                "model": record.model,
                "reason": record.reason,
                "retry_after_s": max(0.0, round(record.expires_at_monotonic - now_value, 3)),
            }
        )
    return available, suppressed


def clear_model_unavailability(model: str | None = None) -> None:
    """Clear temporary unavailability state for one model or all models."""

    if model is None:
        _MODEL_UNAVAILABILITY.clear()
        return
    _MODEL_UNAVAILABILITY.pop(str(model).strip().lower(), None)
