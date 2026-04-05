from __future__ import annotations

import asyncio
import logging
from unittest.mock import patch

import pytest

from llm_client.execution.execution_kernel import (
    _maybe_register_provider_cooldown,
    run_async_with_fallback,
    run_async_with_retry,
    run_sync_with_fallback,
    run_sync_with_retry,
)


def test_run_sync_with_retry_retries_and_succeeds() -> None:
    attempts: list[int] = []
    warnings: list[str] = []

    def invoke(attempt: int) -> str:
        attempts.append(attempt)
        if attempt < 2:
            raise ValueError("transient")
        return "ok"

    result = run_sync_with_retry(
        caller="test",
        model="m",
        max_retries=3,
        invoke=invoke,
        should_retry=lambda exc: isinstance(exc, ValueError),
        compute_delay=lambda attempt, exc: (0.0, "none"),
        warning_sink=warnings,
        logger=logging.getLogger("test_execution_kernel"),
    )

    assert result == "ok"
    assert attempts == [0, 1, 2]
    assert len([w for w in warnings if w.startswith("RETRY")]) == 2


@pytest.mark.asyncio
async def test_run_async_with_retry_retries_and_succeeds() -> None:
    attempts: list[int] = []
    warnings: list[str] = []

    async def invoke(attempt: int) -> str:
        attempts.append(attempt)
        if attempt < 1:
            raise RuntimeError("transient")
        return "ok"

    result = await run_async_with_retry(
        caller="test",
        model="m",
        max_retries=2,
        invoke=invoke,
        should_retry=lambda exc: isinstance(exc, RuntimeError),
        compute_delay=lambda attempt, exc: (0.0, "none"),
        warning_sink=warnings,
        logger=logging.getLogger("test_execution_kernel"),
    )

    assert result == "ok"
    assert attempts == [0, 1]
    assert len([w for w in warnings if w.startswith("RETRY")]) == 1


def test_run_sync_with_fallback_uses_next_model() -> None:
    warnings: list[str] = []
    seen: list[str] = []

    def execute_model(model_idx: int, model_name: str) -> str:
        seen.append(model_name)
        if model_name == "primary":
            raise ValueError("boom")
        return "ok"

    result = run_sync_with_fallback(
        models=["primary", "fallback"],
        execute_model=execute_model,
        warning_sink=warnings,
        logger=logging.getLogger("test_execution_kernel"),
    )

    assert result == "ok"
    assert seen == ["primary", "fallback"]
    assert any("FALLBACK: primary -> fallback" in w for w in warnings)


@pytest.mark.asyncio
async def test_run_async_with_fallback_uses_next_model() -> None:
    warnings: list[str] = []
    seen: list[str] = []

    async def execute_model(model_idx: int, model_name: str) -> str:
        seen.append(model_name)
        await asyncio.sleep(0)
        if model_name == "primary":
            raise ValueError("boom")
        return "ok"

    result = await run_async_with_fallback(
        models=["primary", "fallback"],
        execute_model=execute_model,
        warning_sink=warnings,
        logger=logging.getLogger("test_execution_kernel"),
    )

    assert result == "ok"
    assert seen == ["primary", "fallback"]
    assert any("FALLBACK: primary -> fallback" in w for w in warnings)


def test_register_provider_cooldown_emits_provider_governance_warning() -> None:
    warnings: list[str] = []

    with patch("llm_client.execution.retry._is_rate_limit_error", return_value=True), patch(
        "llm_client.execution.retry._retry_delay_hint",
        return_value=(1.5, "provider-hint"),
    ), patch(
        "llm_client.utils.rate_limit.register_rate_limit_cooldown",
        return_value=1.5,
    ):
        applied = _maybe_register_provider_cooldown(
            model="gemini/gemini-2.5-flash",
            exc=RuntimeError("rate limit"),
            warning_sink=warnings,
            logger=logging.getLogger("test_execution_kernel"),
        )

    assert applied == 1.5
    assert warnings == [
        "PROVIDER_GOVERNANCE_EVENT[cooldown_registered]: provider=google delay_s=1.5 source=provider-hint"
    ]
