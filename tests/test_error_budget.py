"""Tests for agent error budget (Plan #18).

Verifies that AgentErrorBudget and ErrorBudgetState correctly:
- Track turns and errors
- Classify errors as recoverable/non-recoverable
- Stop on budget exhaustion
- Stop on non-recoverable errors immediately
- Stop on consecutive model errors
- Report summary for observability
"""

import pytest

from llm_client.agent.agent_contracts import (
    AgentErrorBudget,
    ErrorBudgetState,
    classify_error,
)


class TestClassifyError:
    """Error classification into recoverable/non-recoverable/unknown."""

    def test_non_recoverable_quota(self) -> None:
        assert classify_error("quota exceeded") == "non_recoverable"

    def test_non_recoverable_auth(self) -> None:
        assert classify_error("invalid api key") == "non_recoverable"

    def test_non_recoverable_billing(self) -> None:
        assert classify_error("plan and billing details") == "non_recoverable"

    def test_non_recoverable_model_not_found(self) -> None:
        assert classify_error("model not found") == "non_recoverable"

    def test_recoverable_rate_limit(self) -> None:
        assert classify_error("rate limit exceeded") == "recoverable"

    def test_recoverable_timeout(self) -> None:
        assert classify_error("connection timed out") == "recoverable"

    def test_recoverable_server_error(self) -> None:
        assert classify_error("http 503 service unavailable") == "recoverable"

    def test_unknown_error(self) -> None:
        assert classify_error("something unexpected happened") == "unknown"

    def test_exception_object(self) -> None:
        assert classify_error(TimeoutError("timed out")) == "recoverable"


class TestErrorBudgetState:
    """ErrorBudgetState tracks turns, errors, and enforces limits."""

    def test_fresh_state(self) -> None:
        budget = AgentErrorBudget()
        state = ErrorBudgetState(budget=budget)
        assert state.total_turns == 0
        assert state.total_errors == 0
        stop, reason = state.should_stop()
        assert not stop
        assert reason == ""

    def test_record_success_increments_turns(self) -> None:
        state = ErrorBudgetState(budget=AgentErrorBudget())
        state.record_success("gpt-5-mini")
        assert state.total_turns == 1
        assert state.total_errors == 0

    def test_record_error_increments_both(self) -> None:
        state = ErrorBudgetState(budget=AgentErrorBudget())
        classification = state.record_error("gpt-5-mini", "rate limit")
        assert state.total_turns == 1
        assert state.total_errors == 1
        assert classification == "recoverable"

    def test_success_resets_consecutive_errors(self) -> None:
        state = ErrorBudgetState(budget=AgentErrorBudget())
        state.record_error("gpt-5-mini", "rate limit")
        state.record_error("gpt-5-mini", "rate limit")
        assert state.consecutive_errors_by_model["gpt-5-mini"] == 2
        state.record_success("gpt-5-mini")
        assert state.consecutive_errors_by_model["gpt-5-mini"] == 0

    def test_max_total_errors_stops(self) -> None:
        budget = AgentErrorBudget(max_total_errors=3)
        state = ErrorBudgetState(budget=budget)
        for _ in range(3):
            state.record_error("gpt-5-mini", "rate limit")
        stop, reason = state.should_stop()
        assert stop
        assert "max_total_errors" in reason

    def test_max_agent_turns_stops(self) -> None:
        budget = AgentErrorBudget(max_agent_turns=5)
        state = ErrorBudgetState(budget=budget)
        for _ in range(5):
            state.record_success("gpt-5-mini")
        stop, reason = state.should_stop()
        assert stop
        assert "max_agent_turns" in reason

    def test_non_recoverable_stops_immediately(self) -> None:
        state = ErrorBudgetState(budget=AgentErrorBudget())
        state.record_error("gpt-5-mini", "invalid api key")
        stop, reason = state.should_stop()
        assert stop
        assert "non_recoverable" in reason

    def test_consecutive_model_errors_stops(self) -> None:
        budget = AgentErrorBudget(max_consecutive_errors_per_model=3)
        state = ErrorBudgetState(budget=budget)
        for _ in range(3):
            state.record_error("gpt-5-mini", "rate limit")
        stop, reason = state.should_stop()
        assert stop
        assert "max_consecutive_errors_per_model" in reason
        assert "gpt-5-mini" in reason

    def test_different_models_dont_share_consecutive(self) -> None:
        budget = AgentErrorBudget(max_consecutive_errors_per_model=3)
        state = ErrorBudgetState(budget=budget)
        state.record_error("gpt-5-mini", "rate limit")
        state.record_error("gpt-5-mini", "rate limit")
        state.record_error("gemini-flash", "rate limit")  # different model
        assert not state.should_skip_model("gpt-5-mini")  # only 2
        assert not state.should_skip_model("gemini-flash")  # only 1

    def test_should_skip_model(self) -> None:
        budget = AgentErrorBudget(max_consecutive_errors_per_model=2)
        state = ErrorBudgetState(budget=budget)
        state.record_error("gpt-5-mini", "rate limit")
        assert not state.should_skip_model("gpt-5-mini")
        state.record_error("gpt-5-mini", "rate limit")
        assert state.should_skip_model("gpt-5-mini")

    def test_summary_includes_all_fields(self) -> None:
        state = ErrorBudgetState(budget=AgentErrorBudget())
        state.record_error("gpt-5-mini", "rate limit")
        state.record_success("gpt-5-mini")
        summary = state.summary()
        assert summary["total_turns"] == 2
        assert summary["total_errors"] == 1
        assert "budget_max_turns" in summary
        assert "budget_max_errors" in summary
        assert "budget_max_consecutive" in summary
        assert "consecutive_errors_by_model" in summary
        assert "last_model" in summary
        assert "last_classification" in summary

    def test_generous_defaults_dont_fire_for_normal_runs(self) -> None:
        """Default budget is generous enough for normal agent runs (5-20 turns)."""
        budget = AgentErrorBudget()  # defaults
        state = ErrorBudgetState(budget=budget)
        # Simulate a normal 15-turn run with 2 errors
        for i in range(15):
            if i in (3, 7):
                state.record_error("gpt-5-mini", "rate limit")
            else:
                state.record_success("gpt-5-mini")
        stop, _ = state.should_stop()
        assert not stop, "Default budget should not fire for a normal 15-turn run"

    def test_pathological_318_calls_capped(self) -> None:
        """The 318-call pathological case would be capped by default budget."""
        budget = AgentErrorBudget()  # max_agent_turns=200
        state = ErrorBudgetState(budget=budget)
        stopped_at = None
        for i in range(318):
            state.record_error("gpt-5-mini", "rate limit")
            stop, _ = state.should_stop()
            if stop:
                stopped_at = i + 1
                break
        assert stopped_at is not None, "Budget should have stopped before 318 calls"
        assert stopped_at <= 100, f"Should stop well before 318 calls, stopped at {stopped_at}"
