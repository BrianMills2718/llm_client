"""OpenRouter key rotation, routing, and detection utilities.

Manages the OpenRouter API key ring, key rotation on quota exhaustion,
and routing detection. These are extracted from client.py for concern
separation; client.py re-exports everything for backward compatibility.

This module depends on retry._error_status_code for status code extraction.
It must not import from client.py directly.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any, Callable

from llm_client.execution.retry import _error_status_code

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (also defined in client.py — canonical home is here)
# ---------------------------------------------------------------------------

OPENROUTER_ROUTING_ENV = "LLM_CLIENT_OPENROUTER_ROUTING"
OPENROUTER_DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_API_BASE_ENV = "OPENROUTER_API_BASE"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_API_KEYS_ENV = "OPENROUTER_API_KEYS"

# ---------------------------------------------------------------------------
# Module-level state for key rotation
# ---------------------------------------------------------------------------

_OPENROUTER_KEY_ROTATION_LOCK = threading.Lock()
_OPENROUTER_KEY_RING: tuple[str, ...] = ()
_OPENROUTER_KEY_RING_INDEX: int = 0


# ---------------------------------------------------------------------------
# Key normalization and splitting
# ---------------------------------------------------------------------------


def _normalize_api_key_value(value: Any) -> str:
    """Normalize API key env/input values."""
    return str(value or "").strip().strip("\"'")


def _split_api_keys(raw: str) -> list[str]:
    """Split comma/semicolon/newline-delimited key lists."""
    normalized: list[str] = []
    for part in re.split(r"[,\n;]", raw):
        value = _normalize_api_key_value(part)
        if value:
            normalized.append(value)
    return normalized


# ---------------------------------------------------------------------------
# Key discovery
# ---------------------------------------------------------------------------


def _openrouter_key_candidates_from_env() -> tuple[str, ...]:
    """Collect OpenRouter keys from supported env vars in stable order."""
    candidates: list[str] = []

    raw_multi = _normalize_api_key_value(os.environ.get(OPENROUTER_API_KEYS_ENV))
    if raw_multi:
        candidates.extend(_split_api_keys(raw_multi))

    primary = _normalize_api_key_value(os.environ.get(OPENROUTER_API_KEY_ENV))
    if primary:
        candidates.append(primary)

    numbered_re = re.compile(rf"^{re.escape(OPENROUTER_API_KEY_ENV)}_(\d+)$")
    numbered: list[tuple[int, str]] = []
    for env_name, env_value in os.environ.items():
        match = numbered_re.match(env_name)
        if not match:
            continue
        normalized = _normalize_api_key_value(env_value)
        if not normalized:
            continue
        numbered.append((int(match.group(1)), normalized))
    numbered.sort(key=lambda item: item[0])
    candidates.extend(value for _, value in numbered)

    deduped: list[str] = []
    seen: set[str] = set()
    for key in candidates:
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return tuple(deduped)


# ---------------------------------------------------------------------------
# Key masking and detection
# ---------------------------------------------------------------------------


def _mask_api_key(key: str | None) -> str:
    """Return a safe, short key fingerprint for logs/warnings."""
    normalized = _normalize_api_key_value(key)
    if not normalized:
        return "<empty>"
    return f"...{normalized[-4:]}"


def _is_openrouter_call(model: str, api_base: str | None) -> bool:
    """Best-effort OpenRouter call detection."""
    model_lower = str(model or "").strip().lower()
    if model_lower.startswith("openrouter/"):
        return True
    base_lower = str(api_base or "").strip().lower()
    return "openrouter.ai" in base_lower


# ---------------------------------------------------------------------------
# Key limit error detection
# ---------------------------------------------------------------------------


def _is_openrouter_key_limit_error(error: Exception) -> bool:
    """Whether an error is OpenRouter key/quota exhaustion suitable for key rotation."""
    text = str(error or "").lower()
    status = _error_status_code(error)

    key_limit = ("key limit exceeded" in text) or ("key limit reached" in text)
    insufficient_credits = (
        ("insufficient credits" in text)
        or ("insufficient quota" in text)
        or (status == 402)
    )
    if not (key_limit or insufficient_credits):
        return False

    if status not in {None, 402, 403}:
        return False

    provider = str(getattr(error, "llm_provider", "") or "").lower()
    model = str(getattr(error, "model", "") or "").lower()
    if "openrouter" in provider or model.startswith("openrouter/") or "openrouter" in text:
        return True
    if key_limit:
        return status in {None, 403}
    return status == 402


# ---------------------------------------------------------------------------
# Key rotation
# ---------------------------------------------------------------------------


def _reset_openrouter_key_rotation_state() -> None:
    """Test helper: reset OpenRouter key-ring cache/index."""
    global _OPENROUTER_KEY_RING, _OPENROUTER_KEY_RING_INDEX  # noqa: PLW0603
    with _OPENROUTER_KEY_ROTATION_LOCK:
        _OPENROUTER_KEY_RING = ()
        _OPENROUTER_KEY_RING_INDEX = 0


def _rotate_openrouter_api_key() -> tuple[str, str, int] | None:
    """Rotate OPENROUTER_API_KEY to the next configured key, if available."""
    global _OPENROUTER_KEY_RING, _OPENROUTER_KEY_RING_INDEX  # noqa: PLW0603

    with _OPENROUTER_KEY_ROTATION_LOCK:
        ring = _openrouter_key_candidates_from_env()
        if not ring:
            return None

        if ring != _OPENROUTER_KEY_RING:
            _OPENROUTER_KEY_RING = ring
            current_env_key = _normalize_api_key_value(os.environ.get(OPENROUTER_API_KEY_ENV))
            if current_env_key and current_env_key in ring:
                _OPENROUTER_KEY_RING_INDEX = ring.index(current_env_key)
            elif _OPENROUTER_KEY_RING_INDEX >= len(ring):
                _OPENROUTER_KEY_RING_INDEX = 0

        if len(ring) < 2:
            return None

        current_env_key = _normalize_api_key_value(os.environ.get(OPENROUTER_API_KEY_ENV))
        if current_env_key and current_env_key in ring:
            current_idx = ring.index(current_env_key)
        else:
            current_idx = _OPENROUTER_KEY_RING_INDEX

        next_idx = (current_idx + 1) % len(ring)
        if next_idx == current_idx:
            return None

        old_key = ring[current_idx]
        new_key = ring[next_idx]
        os.environ[OPENROUTER_API_KEY_ENV] = new_key
        _OPENROUTER_KEY_RING_INDEX = next_idx
        return old_key, new_key, len(ring)


# ---------------------------------------------------------------------------
# Retry with key rotation
# ---------------------------------------------------------------------------


def _maybe_retry_with_openrouter_key_rotation(
    *,
    error: Exception,
    attempt: int,
    max_retries: int,
    current_model: str,
    current_api_base: str | None,
    user_kwargs: dict[str, Any],
    warning_sink: list[str] | None,
    on_retry: Callable[[int, Exception, float], None] | None,
    caller: str,
) -> bool:
    """Rotate OpenRouter key on key/quota exhaustion and trigger immediate retry."""
    explicit_api_key = bool(_normalize_api_key_value(user_kwargs.get("api_key")))
    if explicit_api_key:
        return False
    if not _is_openrouter_call(current_model, current_api_base):
        return False
    if not _is_openrouter_key_limit_error(error):
        return False

    rotated = _rotate_openrouter_api_key()
    if rotated is None:
        msg = (
            "OPENROUTER_KEY_ROTATION_UNAVAILABLE: received OpenRouter key/quota "
            "exhaustion but no backup keys are configured."
        )
        if warning_sink is not None:
            warning_sink.append(msg)
        logger.warning("%s %s", caller, msg)
        return False

    old_key, new_key, pool_size = rotated
    rotation_msg = (
        "OPENROUTER_KEY_ROTATED: "
        f"{_mask_api_key(old_key)} -> {_mask_api_key(new_key)} "
        f"(pool={pool_size})"
    )
    if warning_sink is not None:
        warning_sink.append(rotation_msg)
    logger.warning("%s %s", caller, rotation_msg)

    if attempt >= max_retries:
        return False

    retry_delay_source = "openrouter_key_rotation"
    delay = 0.0
    if on_retry is not None:
        on_retry(attempt, error, delay)
    if warning_sink is not None:
        warning_sink.append(
            f"RETRY {attempt + 1}/{max_retries + 1}: "
            f"{current_model} ({type(error).__name__}: {error}) "
            f"[retry_delay_source={retry_delay_source}]"
        )
    logger.warning(
        "%s attempt %d/%d failed (retrying immediately, source=%s): %s",
        caller,
        attempt + 1,
        max_retries + 1,
        retry_delay_source,
        error,
    )
    return True


# ---------------------------------------------------------------------------
# Routing enablement
# ---------------------------------------------------------------------------


def _openrouter_routing_enabled() -> bool:
    """Whether automatic OpenRouter model normalization is enabled."""
    raw = os.environ.get(OPENROUTER_ROUTING_ENV, "on").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on", ""}:
        return True
    logger.warning(
        "Invalid %s=%r; expected on/off boolean. Defaulting to on.",
        OPENROUTER_ROUTING_ENV,
        raw,
    )
    return True
