from __future__ import annotations

import asyncio
import logging

import pytest

from llm_client.core.model_availability import clear_model_unavailability, filter_available_models
from llm_client.execution.execution_kernel import (
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


@pytest.mark.asyncio
async def test_run_async_with_fallback_records_exhausted_model_for_future_calls() -> None:
    warnings: list[str] = []
    clear_model_unavailability()

    class ExhaustedError(Exception):
        pass

    async def execute_model(model_idx: int, model_name: str) -> str:
        del model_idx
        await asyncio.sleep(0)
        if model_name == "gemini/gemini-2.5-flash":
            raise ExhaustedError(
                "Your project has exceeded its monthly spending cap. "
                "Please go to AI Studio at https://ai.studio/spend to manage your project spend cap."
            )
        return "ok"

    result = await run_async_with_fallback(
        models=["gemini/gemini-2.5-flash", "openrouter/openai/gpt-5.4-mini"],
        execute_model=execute_model,
        warning_sink=warnings,
        logger=logging.getLogger("test_execution_kernel"),
    )

    available, suppressed = filter_available_models(
        ["gemini/gemini-2.5-flash", "openrouter/openai/gpt-5.4-mini"]
    )
    clear_model_unavailability()

    assert result == "ok"
    assert available == ["openrouter/openai/gpt-5.4-mini"]
    assert suppressed[0]["model"] == "gemini/gemini-2.5-flash"
    assert suppressed[0]["reason"] == "provider_spend_cap_exhausted"
    assert any("MODEL_UNAVAILABLE: gemini/gemini-2.5-flash" in w for w in warnings)
