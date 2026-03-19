"""Tests for Langfuse callback integration.

Verifies that Langfuse callbacks activate only when configured via env vars
and the langfuse package is available. Tests metadata injection into litellm
kwargs for callback propagation.
"""

from __future__ import annotations

import importlib
import os
from unittest.mock import patch

import litellm
import pytest

from llm_client.langfuse_callbacks import (
    _is_active,
    configure_langfuse_callbacks,
    inject_metadata,
)


@pytest.fixture(autouse=True)
def _reset_langfuse_state() -> None:  # type: ignore[misc]
    """Reset module state and litellm callbacks between tests."""
    import llm_client.langfuse_callbacks as mod

    mod._initialized = False
    # Remove langfuse from callbacks if present
    if "langfuse" in litellm.success_callback:
        litellm.success_callback.remove("langfuse")
    if "langfuse" in litellm.failure_callback:
        litellm.failure_callback.remove("langfuse")


class TestConfigureLangfuseCallbacks:
    """Tests for configure_langfuse_callbacks()."""

    def test_no_env_var_does_nothing(self) -> None:
        """When LITELLM_CALLBACKS is not set, no callbacks are registered."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LITELLM_CALLBACKS", None)
            result = configure_langfuse_callbacks()
        assert result is False
        assert "langfuse" not in litellm.success_callback
        assert "langfuse" not in litellm.failure_callback

    def test_env_var_without_langfuse_does_nothing(self) -> None:
        """When LITELLM_CALLBACKS is set but doesn't include langfuse, skip."""
        with patch.dict(os.environ, {"LITELLM_CALLBACKS": "prometheus,datadog"}):
            result = configure_langfuse_callbacks()
        assert result is False
        assert "langfuse" not in litellm.success_callback

    def test_env_var_with_langfuse_but_not_installed(self) -> None:
        """When langfuse is requested but not importable, warn and skip."""
        with (
            patch.dict(os.environ, {"LITELLM_CALLBACKS": "langfuse"}),
            patch.dict("sys.modules", {"langfuse": None}),
            patch("builtins.__import__", side_effect=ImportError("no langfuse")),
        ):
            # mock-ok: testing behavior when optional dependency is missing
            result = configure_langfuse_callbacks()
        assert result is False
        assert "langfuse" not in litellm.success_callback

    def test_env_var_with_langfuse_installed_activates(self) -> None:
        """When langfuse is requested and importable, register callbacks."""
        # mock-ok: simulating langfuse availability without installing it
        import types

        fake_langfuse = types.ModuleType("langfuse")
        with (
            patch.dict(os.environ, {"LITELLM_CALLBACKS": "langfuse"}),
            patch.dict("sys.modules", {"langfuse": fake_langfuse}),
        ):
            result = configure_langfuse_callbacks()
        assert result is True
        assert "langfuse" in litellm.success_callback
        assert "langfuse" in litellm.failure_callback

    def test_idempotent(self) -> None:
        """Calling configure twice doesn't double-register callbacks."""
        import types

        fake_langfuse = types.ModuleType("langfuse")
        with (
            patch.dict(os.environ, {"LITELLM_CALLBACKS": "langfuse"}),
            patch.dict("sys.modules", {"langfuse": fake_langfuse}),
        ):
            configure_langfuse_callbacks()
            configure_langfuse_callbacks()
        assert litellm.success_callback.count("langfuse") == 1
        assert litellm.failure_callback.count("langfuse") == 1

    def test_comma_separated_env(self) -> None:
        """LITELLM_CALLBACKS with multiple values, including langfuse."""
        import types

        fake_langfuse = types.ModuleType("langfuse")
        with (
            patch.dict(os.environ, {"LITELLM_CALLBACKS": "prometheus, langfuse, datadog"}),
            patch.dict("sys.modules", {"langfuse": fake_langfuse}),
        ):
            result = configure_langfuse_callbacks()
        assert result is True
        assert "langfuse" in litellm.success_callback


class TestIsActive:
    """Tests for _is_active()."""

    def test_inactive_by_default(self) -> None:
        """Without configuration, Langfuse is not active."""
        assert _is_active() is False

    def test_active_after_configuration(self) -> None:
        """After successful configuration, reports active."""
        litellm.success_callback.append("langfuse")
        assert _is_active() is True


class TestInjectMetadata:
    """Tests for inject_metadata()."""

    def test_injects_task_and_trace_id(self) -> None:
        """Both task and trace_id are added to metadata dict."""
        kwargs: dict[str, object] = {}
        inject_metadata(kwargs, task="my_task", trace_id="trace-123")
        assert kwargs["metadata"] == {"task": "my_task", "trace_id": "trace-123"}

    def test_noop_when_both_none(self) -> None:
        """When both values are None, kwargs is not modified."""
        kwargs: dict[str, object] = {}
        inject_metadata(kwargs, task=None, trace_id=None)
        assert "metadata" not in kwargs

    def test_partial_injection(self) -> None:
        """Only non-None values are injected."""
        kwargs: dict[str, object] = {}
        inject_metadata(kwargs, task="only_task", trace_id=None)
        assert kwargs["metadata"] == {"task": "only_task"}

    def test_preserves_existing_metadata(self) -> None:
        """Existing metadata keys are preserved, new ones merged."""
        kwargs: dict[str, object] = {"metadata": {"existing_key": "value"}}
        inject_metadata(kwargs, task="my_task", trace_id="trace-456")
        meta = kwargs["metadata"]
        assert isinstance(meta, dict)
        assert meta["existing_key"] == "value"
        assert meta["task"] == "my_task"
        assert meta["trace_id"] == "trace-456"

    def test_overrides_conflicting_metadata(self) -> None:
        """If metadata already has task/trace_id, our values win."""
        kwargs: dict[str, object] = {"metadata": {"task": "old_task"}}
        inject_metadata(kwargs, task="new_task", trace_id=None)
        meta = kwargs["metadata"]
        assert isinstance(meta, dict)
        assert meta["task"] == "new_task"
