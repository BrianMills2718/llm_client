"""Structured error types for llm_client.

Callers can catch specific error types instead of parsing raw litellm exceptions:

    from llm_client.errors import LLMRateLimitError, LLMQuotaExhaustedError

    try:
        result = await acall_llm("gpt-4o", messages)
    except LLMQuotaExhaustedError:
        # Switch provider or abort — retrying won't help
        ...
    except LLMRateLimitError:
        # Transient — llm_client already retried, but caller may want to wait longer
        ...
"""

from __future__ import annotations


class LLMError(Exception):
    """Base for all llm_client errors."""

    def __init__(self, message: str, original: Exception | None = None) -> None:
        super().__init__(message)
        self.original = original


class LLMRateLimitError(LLMError):
    """Transient rate limit (429) — retry with backoff."""


class LLMQuotaExhaustedError(LLMError):
    """Permanent quota/billing exhaustion — don't retry, try fallback or abort."""


class LLMAuthError(LLMError):
    """Authentication failed (401/403) — API key invalid or forbidden."""


class LLMContentFilterError(LLMError):
    """Content policy violation — request was blocked."""


class LLMTransientError(LLMError):
    """Server error (500/502/503), timeout, connection — retry."""


class LLMModelNotFoundError(LLMError):
    """Model doesn't exist (404)."""


class LLMBudgetExceededError(LLMError):
    """Trace has exceeded its max_budget — no more calls allowed."""


# Patterns that indicate permanent quota exhaustion (not transient rate limit).
_QUOTA_PATTERNS = [
    "quota",
    "billing",
    "insufficient",
    "exceeded your current",
    "plan and billing",
    "account deactivated",
    "account suspended",
]


def classify_error(error: Exception) -> type[LLMError]:
    """Classify any exception into an LLMError subtype.

    Uses litellm exception types when available, falls back to string matching.
    """
    try:
        import litellm as _lt

        if isinstance(error, (_lt.AuthenticationError, _lt.PermissionDeniedError)):
            return LLMAuthError

        if isinstance(error, _lt.NotFoundError):
            return LLMModelNotFoundError

        if isinstance(error, _lt.ContentPolicyViolationError):
            return LLMContentFilterError

        if isinstance(error, _lt.BudgetExceededError):
            return LLMQuotaExhaustedError

        if isinstance(error, _lt.RateLimitError):
            error_str = str(error).lower()
            if any(p in error_str for p in _QUOTA_PATTERNS):
                return LLMQuotaExhaustedError
            return LLMRateLimitError

        if isinstance(error, (
            _lt.InternalServerError,
            _lt.ServiceUnavailableError,
            _lt.APIConnectionError,
            _lt.BadGatewayError,
        )):
            return LLMTransientError
    except ImportError:
        pass

    # Fallback: string pattern matching
    error_str = str(error).lower()

    if any(p in error_str for p in _QUOTA_PATTERNS):
        return LLMQuotaExhaustedError
    if "401" in error_str or "authentication" in error_str or "unauthorized" in error_str:
        return LLMAuthError
    if "403" in error_str or "forbidden" in error_str or "permission" in error_str:
        return LLMAuthError
    if "404" in error_str or "not found" in error_str or "does not exist" in error_str:
        return LLMModelNotFoundError
    if "content" in error_str and ("policy" in error_str or "filter" in error_str):
        return LLMContentFilterError
    if "rate" in error_str and "limit" in error_str:
        return LLMRateLimitError
    if any(p in error_str for p in ("timeout", "timed out", "connection", "500", "502", "503", "server error")):
        return LLMTransientError

    return LLMError


def wrap_error(error: Exception) -> LLMError:
    """Wrap an exception in the appropriate LLMError subclass.

    If the error is already an LLMError, returns it unchanged.
    """
    if isinstance(error, LLMError):
        return error
    cls = classify_error(error)
    return cls(str(error), original=error)
