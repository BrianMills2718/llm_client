"""Tests for llm_client.testing — LLM-judged test infrastructure.

All tests are offline-safe by default (mock the LLM call). Tests that
actually call an LLM are marked with @pytest.mark.llm_test.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest

from llm_client.testing import (
    CriterionVerdict,
    JudgeResponse,
    LLMTestResult,
    assert_llm_output,
    judge_output,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeLLMCallResult:
    """Minimal stand-in for LLMCallResult (dataclass, not importable without side effects)."""

    content: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    cost: float = 0.001
    model: str = "fake-model"
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"


def _make_judge_response(verdicts: list[tuple[str, bool, str]]) -> JudgeResponse:
    """Build a JudgeResponse from (criterion, passed, reasoning) tuples."""
    return JudgeResponse(
        verdicts=[
            CriterionVerdict(criterion=c, passed=p, reasoning=r)
            for c, p, r in verdicts
        ]
    )


# ---------------------------------------------------------------------------
# Unit tests — models
# ---------------------------------------------------------------------------


class TestCriterionVerdict:
    def test_accepts_valid_data(self) -> None:
        v = CriterionVerdict(
            criterion="Output is JSON",
            passed=True,
            reasoning="The output is valid JSON.",
        )
        assert v.criterion == "Output is JSON"
        assert v.passed is True
        assert v.reasoning == "The output is valid JSON."

    def test_rejects_missing_fields(self) -> None:
        with pytest.raises(Exception):
            CriterionVerdict(criterion="x")  # type: ignore[call-arg]


class TestLLMTestResult:
    def test_pass_rate_calculation(self) -> None:
        """Verify pass_rate reflects the fraction of passed criteria."""
        verdicts = [
            CriterionVerdict(criterion="A", passed=True, reasoning="ok"),
            CriterionVerdict(criterion="B", passed=False, reasoning="nope"),
            CriterionVerdict(criterion="C", passed=True, reasoning="ok"),
            CriterionVerdict(criterion="D", passed=True, reasoning="ok"),
        ]
        result = LLMTestResult(
            verdicts=verdicts,
            passed=False,
            pass_rate=0.75,
            model="test",
            cost_usd=0.0,
            latency_s=0.0,
        )
        assert result.pass_rate == 0.75
        assert len(result.verdicts) == 4
        assert sum(1 for v in result.verdicts if v.passed) == 3

    def test_all_pass(self) -> None:
        verdicts = [
            CriterionVerdict(criterion="A", passed=True, reasoning="ok"),
        ]
        result = LLMTestResult(
            verdicts=verdicts,
            passed=True,
            pass_rate=1.0,
            model="test",
            cost_usd=0.0,
            latency_s=0.0,
        )
        assert result.passed is True
        assert result.pass_rate == 1.0


# ---------------------------------------------------------------------------
# Unit tests — judge_output (mocked LLM)
# ---------------------------------------------------------------------------


class TestJudgeOutputOffline:
    """Verify judge_output processes LLM results correctly without real API calls."""

    def test_processes_results_correctly(self) -> None:
        """Mock the LLM call and verify the function assembles the result."""
        fake_response = _make_judge_response([
            ("Contains entities", True, "Found 3 entities"),
            ("No hallucinations", False, "Entity 'ACME' not in source"),
        ])
        fake_call_result = FakeLLMCallResult(cost=0.002)

        with (
            patch("llm_client.testing.render_prompt", return_value=[{"role": "user", "content": "test"}]),
            patch(
                "llm_client.testing.call_llm_structured",
                return_value=(fake_response, fake_call_result),
            ),
        ):
            result = judge_output(
                output="Some extracted text",
                criteria=["Contains entities", "No hallucinations"],
            )

        assert isinstance(result, LLMTestResult)
        assert len(result.verdicts) == 2
        assert result.verdicts[0].passed is True
        assert result.verdicts[1].passed is False
        assert result.pass_rate == 0.5
        assert result.passed is False
        assert result.cost_usd == 0.002
        assert result.model == "gemini/gemini-2.5-flash-lite"

    def test_custom_model_override(self) -> None:
        """Explicit model= param takes precedence."""
        fake_response = _make_judge_response([
            ("criterion", True, "ok"),
        ])
        fake_call_result = FakeLLMCallResult(cost=0.01)

        with (
            patch("llm_client.testing.render_prompt", return_value=[{"role": "user", "content": "test"}]),
            patch(
                "llm_client.testing.call_llm_structured",
                return_value=(fake_response, fake_call_result),
            ),
        ):
            result = judge_output(
                output="test",
                criteria=["criterion"],
                model="gpt-4o",
            )

        assert result.model == "gpt-4o"

    def test_env_var_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LLM_TEST_JUDGE_MODEL env var is respected when no explicit model."""
        monkeypatch.setenv("LLM_TEST_JUDGE_MODEL", "claude-sonnet")
        fake_response = _make_judge_response([
            ("criterion", True, "ok"),
        ])
        fake_call_result = FakeLLMCallResult(cost=0.0)

        with (
            patch("llm_client.testing.render_prompt", return_value=[{"role": "user", "content": "test"}]),
            patch(
                "llm_client.testing.call_llm_structured",
                return_value=(fake_response, fake_call_result),
            ),
        ):
            result = judge_output(
                output="test",
                criteria=["criterion"],
            )

        assert result.model == "claude-sonnet"


# ---------------------------------------------------------------------------
# Unit tests — assert_llm_output (mocked LLM)
# ---------------------------------------------------------------------------


class TestAssertLLMOutput:
    def _mock_judge(self, verdicts: list[tuple[str, bool, str]]):
        """Return context managers that mock render_prompt and call_llm_structured."""
        fake_response = _make_judge_response(verdicts)
        fake_call_result = FakeLLMCallResult(cost=0.001)
        return (
            patch("llm_client.testing.render_prompt", return_value=[{"role": "user", "content": "test"}]),
            patch(
                "llm_client.testing.call_llm_structured",
                return_value=(fake_response, fake_call_result),
            ),
        )

    def test_threshold_passes_when_met(self) -> None:
        """3/4 criteria pass with threshold=0.75 -> passes."""
        verdicts = [
            ("A", True, "ok"),
            ("B", True, "ok"),
            ("C", True, "ok"),
            ("D", False, "nope"),
        ]
        mock_render, mock_call = self._mock_judge(verdicts)
        with mock_render, mock_call:
            result = assert_llm_output(
                output="test output",
                criteria=["A", "B", "C", "D"],
                threshold=0.75,
            )
        assert result is not None
        assert result.pass_rate == 0.75

    def test_threshold_fails_below(self) -> None:
        """1/4 criteria pass with threshold=0.5 -> AssertionError."""
        verdicts = [
            ("A", True, "ok"),
            ("B", False, "wrong"),
            ("C", False, "missing"),
            ("D", False, "absent"),
        ]
        mock_render, mock_call = self._mock_judge(verdicts)
        with mock_render, mock_call:
            with pytest.raises(AssertionError, match="25% criteria passed.*threshold is 50%"):
                assert_llm_output(
                    output="test output",
                    criteria=["A", "B", "C", "D"],
                    threshold=0.5,
                )

    def test_failure_message_includes_reasoning(self) -> None:
        """Failed criteria reasoning appears in the AssertionError message."""
        verdicts = [
            ("Must have entities", False, "No entities found in output"),
        ]
        mock_render, mock_call = self._mock_judge(verdicts)
        with mock_render, mock_call:
            with pytest.raises(AssertionError, match="No entities found in output"):
                assert_llm_output(
                    output="test",
                    criteria=["Must have entities"],
                )

    def test_all_pass_default_threshold(self) -> None:
        """All criteria pass with default threshold=1.0 -> success."""
        verdicts = [
            ("A", True, "ok"),
            ("B", True, "ok"),
        ]
        mock_render, mock_call = self._mock_judge(verdicts)
        with mock_render, mock_call:
            result = assert_llm_output(
                output="test",
                criteria=["A", "B"],
            )
        assert result is not None
        assert result.passed is True


class TestSkipWhenEnvSet:
    def test_skip_when_env_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With LLM_TEST_SKIP=1, assert_llm_output is a no-op (pytest.skip)."""
        monkeypatch.setenv("LLM_TEST_SKIP", "1")
        with pytest.raises(pytest.skip.Exception):
            assert_llm_output(
                output="anything",
                criteria=["anything"],
            )

    def test_no_skip_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without LLM_TEST_SKIP, the function runs normally (with mocked LLM)."""
        monkeypatch.delenv("LLM_TEST_SKIP", raising=False)
        fake_response = _make_judge_response([("A", True, "ok")])
        fake_call_result = FakeLLMCallResult(cost=0.0)
        with (
            patch("llm_client.testing.render_prompt", return_value=[{"role": "user", "content": "test"}]),
            patch(
                "llm_client.testing.call_llm_structured",
                return_value=(fake_response, fake_call_result),
            ),
        ):
            result = assert_llm_output(
                output="test",
                criteria=["A"],
            )
        assert result is not None
