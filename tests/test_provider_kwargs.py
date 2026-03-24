"""Focused tests for provider-kwargs preparation at the public client boundary.

These tests keep the transport contract explicit: llm_client may attach
underscore-prefixed runtime control objects internally, but provider-facing
kwargs must stay JSON-serializable and must not receive those private values.
"""

from __future__ import annotations

from llm_client.client import _prepare_call_kwargs


def test_prepare_call_kwargs_strips_internal_runtime_objects() -> None:
    """Provider kwargs should exclude underscore-prefixed internal control objects."""

    monitor = object()
    call_kwargs = _prepare_call_kwargs(
        "deepseek/deepseek-chat",
        [{"role": "user", "content": "hello"}],
        timeout=0,
        num_retries=0,
        reasoning_effort=None,
        api_base=None,
        kwargs={
            "_lifecycle_monitor": monitor,
            "metadata": {"scope": "test"},
        },
    )

    assert "_lifecycle_monitor" not in call_kwargs
    assert call_kwargs["metadata"] == {"scope": "test"}
