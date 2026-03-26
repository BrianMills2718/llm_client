from __future__ import annotations

from llm_client.agent.agent_outcomes import (
    EVENT_CODE_PROVIDER_EMPTY,
    EVENT_CODE_REQUIRED_SUBMIT_NOT_ACCEPTED,
    _classify_failure_signals,
    _first_terminal_failure_event_code,
    _summarize_failure_events,
    _summarize_finalization_attempts,
)


def test_classify_failure_signals_prefers_provider() -> None:
    primary, secondary = _classify_failure_signals(
        failure_event_codes=[EVENT_CODE_PROVIDER_EMPTY, "RETRIEVAL_NO_HITS"],
        retrieval_no_hits_count=1,
        control_loop_suppressed_calls=0,
        force_final_reason=None,
        submit_answer_succeeded=False,
    )
    assert primary == "provider"
    assert "retrieval" in secondary


def test_first_terminal_failure_event_code_uses_earliest_terminal() -> None:
    assert _first_terminal_failure_event_code(
        ["CONTROL_CHURN_SUPPRESSED", EVENT_CODE_REQUIRED_SUBMIT_NOT_ACCEPTED, EVENT_CODE_PROVIDER_EMPTY]
    ) == EVENT_CODE_REQUIRED_SUBMIT_NOT_ACCEPTED


def test_summarize_failure_events_counts_provider_incompletion() -> None:
    summary = _summarize_failure_events(
        failure_event_codes=[EVENT_CODE_PROVIDER_EMPTY, EVENT_CODE_PROVIDER_EMPTY],
        retrieval_no_hits_count=0,
        control_loop_suppressed_calls=0,
        force_final_reason=None,
        run_completed=False,
    )
    assert summary.primary_failure_class == "provider"
    assert summary.provider_failure_event_total == 2
    assert summary.provider_caused_incompletion is True


def test_summarize_finalization_attempts_counts_fallback_and_breaker() -> None:
    summary = _summarize_finalization_attempts(
        finalization_fallback_attempts=[
            {"model": "model-a", "status": "error", "error_code": EVENT_CODE_PROVIDER_EMPTY},
            {"model": "model-b", "status": "success"},
        ],
        finalization_primary_model="model-a",
        forced_final_attempts=2,
        forced_final_circuit_breaker_opened=True,
    )
    assert summary.finalization_fallback_attempt_count == 1
    assert summary.provider_empty_attempt_counts_by_model == {"model-a": 1}
    assert summary.finalization_breaker_open_rate == 1.0
