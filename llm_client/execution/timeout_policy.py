"""Shared timeout-policy helpers for ``llm_client`` runtimes.

Timeout handling is a cross-cutting runtime policy rather than a transport
detail. Both provider-backed calls and agent SDK calls need the same core
normalization rules:

1. parse timeout values conservatively,
2. clamp invalid or negative values to zero,
3. honor the global timeout disable switch loudly,
4. optionally expose warnings back to the public result surface.
"""

from __future__ import annotations

import logging
import os
from typing import Any

TIMEOUT_POLICY_ENV = "LLM_CLIENT_TIMEOUT_POLICY"
SAFETY_TIMEOUT_ENV = "LLM_CLIENT_SAFETY_TIMEOUT"

# Safety ceiling: maximum time any single LLM call can take, regardless of
# timeout policy. This is NOT a request timeout (which controls expected
# response time) — it is a dead-connection detector. No legitimate LLM call
# takes 5 minutes without producing output. Override via LLM_CLIENT_SAFETY_TIMEOUT.
DEFAULT_SAFETY_TIMEOUT_S = 300  # 5 minutes

_logger = logging.getLogger(__name__)
_TIMEOUT_POLICY_LOGGED = False


def timeouts_disabled() -> bool:
    """Whether timeout arguments should be ignored globally."""
    raw = str(os.environ.get(TIMEOUT_POLICY_ENV, "") or "").strip().lower()
    if not raw:
        return False
    if raw in {"allow", "allowed", "enable", "enabled", "on", "true", "yes", "1"}:
        return False
    if raw in {"ban", "disable", "disabled", "off", "none", "false", "no", "0"}:
        return True
    return False


def timeout_policy_label() -> str:
    """Return the stable label for the current process timeout policy."""
    return "ban" if timeouts_disabled() else "allow"


def log_timeout_policy_once(
    *,
    caller: str,
    logger: logging.Logger | None = None,
) -> None:
    """Emit a one-time process-level timeout policy log."""
    global _TIMEOUT_POLICY_LOGGED  # noqa: PLW0603
    if _TIMEOUT_POLICY_LOGGED:
        return
    active_logger = logger or _logger
    active_logger.warning(
        "LLM_CLIENT_TIMEOUT_POLICY=%s (first observed in %s)",
        timeout_policy_label(),
        caller,
    )
    _TIMEOUT_POLICY_LOGGED = True


def normalize_timeout(
    timeout: Any,
    *,
    caller: str,
    warning_sink: list[str] | None = None,
    logger: logging.Logger | None = None,
    log_policy_once_enabled: bool = False,
) -> int:
    """Normalize timeout value and enforce optional global disable policy."""
    active_logger = logger or _logger
    if log_policy_once_enabled:
        log_timeout_policy_once(caller=caller, logger=active_logger)
    try:
        parsed = int(timeout)
    except (TypeError, ValueError):
        parsed = 0
    if parsed < 0:
        parsed = 0
    if parsed > 0 and timeouts_disabled():
        msg = (
            f"TIMEOUT_DISABLED[{caller}]: timeout={parsed}s ignored "
            f"(set {TIMEOUT_POLICY_ENV}=allow to re-enable)."
        )
        active_logger.warning(msg)
        if warning_sink is not None and msg not in warning_sink:
            warning_sink.append(msg)
        return 0
    return parsed


def safety_timeout_s() -> int:
    """Return the safety ceiling timeout in seconds.

    This timeout applies even when TIMEOUT_POLICY=ban. It prevents
    infinite hangs on dead connections. Not a request timeout —
    a dead-connection detector.

    Override via LLM_CLIENT_SAFETY_TIMEOUT env var. Set to 0 to disable
    (not recommended).
    """
    raw = os.environ.get(SAFETY_TIMEOUT_ENV, "")
    if raw:
        try:
            val = int(raw)
            return max(val, 0)
        except (TypeError, ValueError):
            pass
    return DEFAULT_SAFETY_TIMEOUT_S


# ---------------------------------------------------------------------------
# IPv4 forcing for providers with IPv6 routing issues (Gemini on WSL2)
# ---------------------------------------------------------------------------

_IPV4_FORCED = False


def _is_wsl2() -> bool:
    """Detect WSL2 environment via /proc/version."""
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except (OSError, IOError):
        return False

IPV4_FORCE_ENV = "LLM_CLIENT_FORCE_IPV4"


def force_ipv4_if_configured() -> bool:
    """Force IPv4-only DNS resolution if LLM_CLIENT_FORCE_IPV4=1 or on WSL2.

    Gemini API endpoints resolve to IPv6 addresses that experience routing
    problems on WSL2, causing socket-level hangs that bypass all Python-level
    timeouts. This patches socket.getaddrinfo to filter out IPv6 results.

    Auto-enables on WSL2 (detected via /proc/version) unless explicitly
    disabled with LLM_CLIENT_FORCE_IPV4=0.

    Safe to call multiple times — only patches once.
    Returns True if IPv4 forcing was activated.
    """
    global _IPV4_FORCED  # noqa: PLW0603
    if _IPV4_FORCED:
        return True

    raw = os.environ.get(IPV4_FORCE_ENV, "").strip().lower()

    # Explicit opt-out
    if raw in {"0", "false", "no", "off"}:
        return False

    # Explicit opt-in
    if raw in {"1", "true", "yes", "on"}:
        pass  # Fall through to activation
    elif _is_wsl2():
        _logger.info("WSL2 detected — auto-enabling IPv4-only DNS (set %s=0 to disable)", IPV4_FORCE_ENV)
    else:
        return False

    import socket
    _original_getaddrinfo = socket.getaddrinfo

    def _ipv4_only(*args: Any, **kwargs: Any) -> list:
        results = _original_getaddrinfo(*args, **kwargs)
        ipv4 = [r for r in results if r[0] == socket.AF_INET]
        return ipv4 if ipv4 else results  # Fall back to original if no IPv4

    socket.getaddrinfo = _ipv4_only  # type: ignore[assignment]
    _IPV4_FORCED = True
    _logger.info("IPv4-only DNS resolution enabled (LLM_CLIENT_FORCE_IPV4=1)")
    return True
