"""Tests for llm_client.errors — error classification and wrapping."""

from __future__ import annotations

import pytest
import litellm

from llm_client.errors import (
    LLMAuthError,
    LLMConfigurationError,
    LLMContentFilterError,
    LLMError,
    LLMModelNotFoundError,
    LLMQuotaExhaustedError,
    LLMRateLimitError,
    LLMTransientError,
    classify_error,
    wrap_error,
)


# ---------------------------------------------------------------------------
# classify_error — litellm exception types
# ---------------------------------------------------------------------------


class TestClassifyLitellmTypes:
    def test_auth_error(self):
        err = litellm.AuthenticationError(
            message="Invalid API key", model="gpt-4o", llm_provider="openai"
        )
        assert classify_error(err) is LLMAuthError

    def test_permission_denied(self):
        from unittest.mock import MagicMock
        mock_response = MagicMock()
        err = litellm.PermissionDeniedError(
            message="Forbidden", model="gpt-4o", llm_provider="openai", response=mock_response
        )
        assert classify_error(err) is LLMAuthError

    def test_not_found(self):
        err = litellm.NotFoundError(
            message="Model not found", model="gpt-99", llm_provider="openai"
        )
        assert classify_error(err) is LLMModelNotFoundError

    def test_content_policy(self):
        err = litellm.ContentPolicyViolationError(
            message="Content blocked", model="gpt-4o", llm_provider="openai"
        )
        assert classify_error(err) is LLMContentFilterError

    def test_budget_exceeded(self):
        err = litellm.BudgetExceededError(
            message="Budget exceeded", current_cost=10.0, max_budget=5.0
        )
        assert classify_error(err) is LLMQuotaExhaustedError

    def test_rate_limit_transient(self):
        err = litellm.RateLimitError(
            message="Rate limit exceeded, please retry after 1s",
            model="gpt-4o",
            llm_provider="openai",
        )
        assert classify_error(err) is LLMRateLimitError

    def test_rate_limit_quota(self):
        err = litellm.RateLimitError(
            message="You exceeded your current quota, check billing",
            model="gpt-4o",
            llm_provider="openai",
        )
        assert classify_error(err) is LLMQuotaExhaustedError

    def test_internal_server_error(self):
        err = litellm.InternalServerError(
            message="Internal server error", model="gpt-4o", llm_provider="openai"
        )
        assert classify_error(err) is LLMTransientError

    def test_service_unavailable(self):
        err = litellm.ServiceUnavailableError(
            message="Service unavailable", model="gpt-4o", llm_provider="openai"
        )
        assert classify_error(err) is LLMTransientError

    def test_api_connection_error(self):
        err = litellm.APIConnectionError(
            message="Connection reset", model="gpt-4o", llm_provider="openai"
        )
        assert classify_error(err) is LLMTransientError


# ---------------------------------------------------------------------------
# classify_error — string fallback
# ---------------------------------------------------------------------------


class TestClassifyStringFallback:
    def test_quota_string(self):
        err = Exception("exceeded your current quota")
        assert classify_error(err) is LLMQuotaExhaustedError

    def test_billing_string(self):
        err = Exception("check your billing details")
        assert classify_error(err) is LLMQuotaExhaustedError

    def test_auth_401(self):
        err = Exception("Error 401: unauthorized")
        assert classify_error(err) is LLMAuthError

    def test_not_found_404(self):
        err = Exception("Error 404: model does not exist")
        assert classify_error(err) is LLMModelNotFoundError

    def test_content_filter(self):
        err = Exception("content policy violation")
        assert classify_error(err) is LLMContentFilterError

    def test_rate_limit_string(self):
        err = Exception("rate limit exceeded")
        assert classify_error(err) is LLMRateLimitError

    def test_timeout_string(self):
        err = Exception("Request timed out after 60s")
        assert classify_error(err) is LLMTransientError

    def test_connection_string(self):
        err = Exception("connection reset by peer")
        assert classify_error(err) is LLMTransientError

    def test_unknown(self):
        err = Exception("something completely unexpected")
        assert classify_error(err) is LLMError


# ---------------------------------------------------------------------------
# wrap_error
# ---------------------------------------------------------------------------


class TestWrapError:
    def test_wrap_generic_exception(self):
        original = litellm.AuthenticationError(
            message="Invalid API key", model="gpt-4o", llm_provider="openai"
        )
        wrapped = wrap_error(original)
        assert isinstance(wrapped, LLMAuthError)
        assert wrapped.original is original
        assert "Invalid API key" in str(wrapped)

    def test_wrap_preserves_llm_error(self):
        original = LLMRateLimitError("already wrapped")
        wrapped = wrap_error(original)
        assert wrapped is original

    def test_wrap_quota_error(self):
        original = litellm.RateLimitError(
            message="You exceeded your current quota",
            model="gpt-4o",
            llm_provider="openai",
        )
        wrapped = wrap_error(original)
        assert isinstance(wrapped, LLMQuotaExhaustedError)
        assert wrapped.original is original

    def test_wrap_transient_rate_limit(self):
        original = litellm.RateLimitError(
            message="Rate limit reached, retry after 2s",
            model="gpt-4o",
            llm_provider="openai",
        )
        wrapped = wrap_error(original)
        assert isinstance(wrapped, LLMRateLimitError)

    def test_wrap_blank_timeout_message(self):
        original = TimeoutError()
        wrapped = wrap_error(original)
        assert isinstance(wrapped, LLMTransientError)
        assert str(wrapped) == "TimeoutError"


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class TestErrorHierarchy:
    def test_all_subclass_llm_error(self):
        for cls in [
            LLMRateLimitError,
            LLMQuotaExhaustedError,
            LLMAuthError,
            LLMContentFilterError,
            LLMTransientError,
            LLMModelNotFoundError,
            LLMConfigurationError,
        ]:
            assert issubclass(cls, LLMError)
            assert issubclass(cls, Exception)

    def test_catch_specific(self):
        err = LLMQuotaExhaustedError("quota gone")
        with pytest.raises(LLMQuotaExhaustedError):
            raise err

    def test_catch_base(self):
        err = LLMQuotaExhaustedError("quota gone")
        with pytest.raises(LLMError):
            raise err

    def test_original_attribute(self):
        original = ValueError("raw error")
        err = LLMTransientError("wrapped", original=original)
        assert err.original is original

    def test_configuration_error_fields(self):
        err = LLMConfigurationError(
            "bad config",
            error_code="LLMC_ERR_BACKGROUND_ENDPOINT_UNSUPPORTED",
            details={"api_base": "https://openrouter.ai/api/v1"},
        )
        assert err.error_code == "LLMC_ERR_BACKGROUND_ENDPOINT_UNSUPPORTED"
        assert err.details["api_base"] == "https://openrouter.ai/api/v1"
