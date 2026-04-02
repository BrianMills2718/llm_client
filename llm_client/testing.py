"""LLM-judged testing for non-deterministic code.

Shared infrastructure for testing code that produces LLM-generated output.
Instead of asserting on exact values, tests define semantic criteria and
an LLM judge evaluates pass/fail.

Usage::

    from llm_client.testing import assert_llm_output, LLMTestResult

    def test_extraction():
        result = my_extraction_function("some text")
        assert_llm_output(
            output=result,
            criteria=[
                "Contains at least 2 named entities",
                "Entities have correct types (person, org, place)",
                "No hallucinated entities not in the source text",
            ],
        )
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from llm_client.core.client import call_llm_structured
from llm_client.prompts import render_prompt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

_DEFAULT_JUDGE_MODEL = "gemini/gemini-2.5-flash-lite"
_PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "llm_test_judge.yaml"


class CriterionVerdict(BaseModel):
    """Verdict for a single criterion evaluated by the LLM judge."""

    criterion: str = Field(description="The criterion text that was evaluated.")
    passed: bool = Field(description="Whether the criterion was met. True = pass, False = fail.")
    reasoning: str = Field(description="One-sentence explanation of why the criterion passed or failed.")


class JudgeResponse(BaseModel):
    """Structured output schema for the LLM judge.

    The LLM returns this model directly via call_llm_structured.
    """

    verdicts: list[CriterionVerdict] = Field(
        description="One verdict per criterion, in the same order as the input criteria."
    )


class LLMTestResult(BaseModel):
    """Aggregate result from an LLM-judged test evaluation."""

    verdicts: list[CriterionVerdict] = Field(
        description="Per-criterion verdicts from the judge."
    )
    passed: bool = Field(
        description="Overall pass/fail. True only if pass_rate >= threshold used."
    )
    pass_rate: float = Field(
        description="Fraction of criteria that passed (0.0 to 1.0)."
    )
    model: str = Field(
        description="Model used for judging."
    )
    cost_usd: float = Field(
        description="Cost in USD for the judge call."
    )
    latency_s: float = Field(
        description="Wall-clock seconds for the judge call."
    )


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def _resolve_judge_model(model: str | None) -> str:
    """Resolve the judge model from explicit arg, env var, or default."""
    if model is not None:
        return model
    return os.environ.get("LLM_TEST_JUDGE_MODEL", _DEFAULT_JUDGE_MODEL)


def judge_output(
    output: str,
    criteria: list[str],
    *,
    context: str | None = None,
    model: str | None = None,
    task: str | None = None,
    trace_id: str | None = None,
) -> LLMTestResult:
    """Evaluate an LLM-generated output against semantic criteria.

    Calls an LLM judge to assess each criterion as pass/fail with reasoning.

    Args:
        output: The text output to evaluate.
        criteria: List of human-readable criteria strings.
        context: Optional context about the task that produced the output
            (e.g., the original input text). Helps the judge make informed
            decisions.
        model: Judge model override. Defaults to env var
            ``LLM_TEST_JUDGE_MODEL`` or ``gemini/gemini-2.5-flash-lite``.
        task: Observability task tag. Defaults to ``"llm_test.judge"``.
        trace_id: Observability trace ID. Auto-generated if not provided.

    Returns:
        LLMTestResult with per-criterion verdicts and aggregate metrics.
    """
    resolved_model = _resolve_judge_model(model)
    resolved_task = task or "llm_test.judge"
    resolved_trace_id = trace_id or f"llm_test/{uuid.uuid4().hex[:12]}"

    messages = render_prompt(
        template_path=str(_PROMPT_TEMPLATE_PATH),
        output=output,
        criteria=criteria,
        context=context or "",
    )

    t0 = time.monotonic()
    parsed, call_result = call_llm_structured(
        resolved_model,
        messages,
        response_model=JudgeResponse,
        task=resolved_task,
        trace_id=resolved_trace_id,
        max_budget=0.50,
    )
    latency = time.monotonic() - t0

    verdicts = parsed.verdicts
    num_passed = sum(1 for v in verdicts if v.passed)
    total = len(verdicts) if verdicts else 1
    pass_rate = num_passed / total

    result = LLMTestResult(
        verdicts=verdicts,
        passed=pass_rate >= 1.0,
        pass_rate=pass_rate,
        model=resolved_model,
        cost_usd=call_result.cost,
        latency_s=round(latency, 3),
    )

    logger.info(
        "LLM judge: %d/%d criteria passed (%.0f%%) model=%s cost=$%.4f latency=%.1fs",
        num_passed,
        len(verdicts),
        pass_rate * 100,
        resolved_model,
        call_result.cost,
        latency,
    )

    return result


def assert_llm_output(
    output: str,
    criteria: list[str],
    *,
    threshold: float = 1.0,
    context: str | None = None,
    model: str | None = None,
    task: str | None = None,
    trace_id: str | None = None,
) -> LLMTestResult | None:
    """Assert that an LLM output meets semantic criteria, judged by an LLM.

    Calls :func:`judge_output` and then asserts that the fraction of passing
    criteria meets or exceeds ``threshold``.

    Args:
        output: The text output to evaluate.
        criteria: List of human-readable criteria strings.
        threshold: Minimum fraction of criteria that must pass (0.0 to 1.0).
            Defaults to 1.0 (all must pass).
        context: Optional context about the task that produced the output.
        model: Judge model override.
        task: Observability task tag.
        trace_id: Observability trace ID.

    Returns:
        The LLMTestResult on success, or None if skipped.

    Raises:
        AssertionError: If the pass rate is below the threshold. The error
            message includes the judge's reasoning for each failed criterion.
        pytest.skip: If ``LLM_TEST_SKIP=1`` is set in the environment.
    """
    if os.environ.get("LLM_TEST_SKIP", "").strip() == "1":
        import pytest

        pytest.skip("LLM_TEST_SKIP=1: skipping LLM-judged assertion")
        return None  # unreachable, but satisfies type checker

    result = judge_output(
        output=output,
        criteria=criteria,
        context=context,
        model=model,
        task=task,
        trace_id=trace_id,
    )

    if result.pass_rate < threshold:
        failed = [v for v in result.verdicts if not v.passed]
        lines = [
            f"LLM judge: {result.pass_rate:.0%} criteria passed, "
            f"threshold is {threshold:.0%} "
            f"(model={result.model}, cost=${result.cost_usd:.4f})",
        ]
        for v in failed:
            lines.append(f"  FAIL: {v.criterion}")
            lines.append(f"        Reason: {v.reasoning}")
        raise AssertionError("\n".join(lines))

    return result
