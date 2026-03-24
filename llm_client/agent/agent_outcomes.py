"""Failure/finalization outcome helpers for agent runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

EVENT_CODE_PROVIDER_EMPTY = "PROVIDER_EMPTY_CANDIDATES"
EVENT_CODE_PROVIDER_CREDITS_EXHAUSTED = "PROVIDER_CREDITS_EXHAUSTED"
EVENT_CODE_FINALIZATION_CIRCUIT_BREAKER_OPEN = "FINALIZATION_CIRCUIT_BREAKER_OPEN"
EVENT_CODE_FINALIZATION_TOOL_CALL_DISALLOWED = "FINALIZATION_TOOL_CALL_DISALLOWED"
EVENT_CODE_RETRIEVAL_STAGNATION = "RETRIEVAL_STAGNATION"
EVENT_CODE_CONTROL_CHURN_THRESHOLD = "CONTROL_CHURN_THRESHOLD_EXCEEDED"
EVENT_CODE_REQUIRED_SUBMIT_NOT_ATTEMPTED = "REQUIRED_SUBMIT_NOT_ATTEMPTED"
EVENT_CODE_REQUIRED_SUBMIT_NOT_ACCEPTED = "REQUIRED_SUBMIT_NOT_ACCEPTED"
EVENT_CODE_SUBMIT_FORCED_ACCEPT_BUDGET_EXHAUSTION = "SUBMIT_FORCED_ACCEPT_BUDGET_EXHAUSTION"
EVENT_CODE_SUBMIT_FORCED_ACCEPT_TURN_EXHAUSTION = "SUBMIT_FORCED_ACCEPT_TURN_EXHAUSTION"
EVENT_CODE_SUBMIT_FORCED_ACCEPT_FORCED_FINAL = "SUBMIT_FORCED_ACCEPT_FORCED_FINAL"
EVENT_CODE_NO_LEGAL_NONCONTROL_TOOLS = "NO_LEGAL_NONCONTROL_TOOLS"

_PRIMARY_FAILURE_PRIORITY: tuple[str, ...] = (
    "provider",
    "composability",
    "policy",
    "retrieval",
    "control_churn",
    "reasoning",
)

_TERMINAL_FAILURE_EVENT_CODES: frozenset[str] = frozenset({
    EVENT_CODE_PROVIDER_EMPTY,
    EVENT_CODE_PROVIDER_CREDITS_EXHAUSTED,
    EVENT_CODE_FINALIZATION_CIRCUIT_BREAKER_OPEN,
    EVENT_CODE_RETRIEVAL_STAGNATION,
    EVENT_CODE_CONTROL_CHURN_THRESHOLD,
    EVENT_CODE_REQUIRED_SUBMIT_NOT_ATTEMPTED,
    EVENT_CODE_REQUIRED_SUBMIT_NOT_ACCEPTED,
})


@dataclass(frozen=True)
class FailureSummary:
    """Aggregated failure counters/classification for agent-loop metadata."""

    primary_failure_class: str
    secondary_failure_classes: list[str]
    first_terminal_failure_event_code: str | None
    failure_event_code_counts: dict[str, int]
    failure_event_class_counts: dict[str, int]
    provider_failure_event_code_counts: dict[str, int]
    provider_failure_event_total: int
    provider_caused_incompletion: bool


@dataclass(frozen=True)
class FinalizationSummary:
    """Aggregated finalization counters for agent-loop metadata."""

    finalization_attempt_counts_by_model: dict[str, int]
    finalization_failure_counts_by_model: dict[str, int]
    finalization_success_counts_by_model: dict[str, int]
    finalization_failure_code_counts: dict[str, int]
    provider_empty_attempt_counts_by_model: dict[str, int]
    finalization_fallback_attempt_count: int
    finalization_fallback_usage_rate: float
    finalization_breaker_open_rate: float
    finalization_breaker_open_by_model: dict[str, int]


@dataclass(frozen=True)
class ForcedFinalizationResult:
    """Outcome and metric deltas from the forced-final execution path."""

    final_content: str
    final_finish_reason: str
    finalization_primary_model: str
    forced_final_attempts: int
    forced_final_circuit_breaker_opened: bool
    finalization_fallback_used: bool
    finalization_fallback_succeeded: bool
    finalization_events: list[str]
    finalization_fallback_attempts: list[dict[str, Any]]
    failure_event_codes: list[str]
    context_tool_result_clearings_delta: int
    context_tool_results_cleared_delta: int
    context_tool_result_cleared_chars_delta: int
    context_compactions_delta: int
    context_compacted_messages_delta: int
    context_compacted_chars_delta: int
    total_cost_delta: float
    total_input_tokens_delta: int
    total_output_tokens_delta: int
    total_cached_tokens_delta: int
    total_cache_creation_tokens_delta: int
    turns_delta: int


def _first_terminal_failure_event_code(failure_event_codes: list[str]) -> str | None:
    """Return earliest terminal blocker event code in event order, if present."""
    for code in failure_event_codes:
        if code in _TERMINAL_FAILURE_EVENT_CODES:
            return code
    return None


def _failure_class_for_event_code(code: str) -> str | None:
    """Map event code to one stable failure class for rollups/classification."""
    if code in {
        EVENT_CODE_SUBMIT_FORCED_ACCEPT_BUDGET_EXHAUSTION,
        EVENT_CODE_SUBMIT_FORCED_ACCEPT_TURN_EXHAUSTION,
        EVENT_CODE_SUBMIT_FORCED_ACCEPT_FORCED_FINAL,
    }:
        return None
    if code == EVENT_CODE_FINALIZATION_TOOL_CALL_DISALLOWED:
        return "policy"
    if code.startswith("REQUIRED_SUBMIT_"):
        return "policy"
    if code.startswith("PROVIDER_"):
        return "provider"
    if code.startswith("FINALIZATION_"):
        return "provider"
    if code.startswith("TOOL_VALIDATION_REJECTED_") or code == EVENT_CODE_NO_LEGAL_NONCONTROL_TOOLS:
        return "composability"
    if code.startswith("RETRIEVAL_"):
        return "retrieval"
    if code.startswith("CONTROL_CHURN_"):
        return "control_churn"
    if code.startswith("REASONING_"):
        return "reasoning"
    return None


def _classify_failure_signals(
    *,
    failure_event_codes: list[str],
    retrieval_no_hits_count: int,
    control_loop_suppressed_calls: int,
    force_final_reason: str | None,
    submit_answer_succeeded: bool,
) -> tuple[str, list[str]]:
    """Deterministic primary/secondary failure classes from event codes + counters."""
    classes: set[str] = set()

    for code in failure_event_codes:
        cls = _failure_class_for_event_code(code)
        if cls is not None:
            classes.add(cls)

    if retrieval_no_hits_count > 0:
        classes.add("retrieval")
    if control_loop_suppressed_calls > 0:
        classes.add("control_churn")
    if force_final_reason is not None and not classes and not submit_answer_succeeded:
        classes.add("reasoning")

    ordered = [c for c in _PRIMARY_FAILURE_PRIORITY if c in classes]
    if not ordered:
        return "none", []
    return ordered[0], ordered[1:]


def _summarize_failure_events(
    *,
    failure_event_codes: list[str],
    retrieval_no_hits_count: int,
    control_loop_suppressed_calls: int,
    force_final_reason: str | None,
    run_completed: bool,
) -> FailureSummary:
    primary_failure_class, secondary_failure_classes = _classify_failure_signals(
        failure_event_codes=failure_event_codes,
        retrieval_no_hits_count=retrieval_no_hits_count,
        control_loop_suppressed_calls=control_loop_suppressed_calls,
        force_final_reason=force_final_reason,
        submit_answer_succeeded=run_completed,
    )
    first_terminal_code = _first_terminal_failure_event_code(failure_event_codes)
    failure_event_code_counts: dict[str, int] = {}
    for code in failure_event_codes:
        failure_event_code_counts[code] = failure_event_code_counts.get(code, 0) + 1
    failure_event_class_counts: dict[str, int] = {}
    provider_failure_event_code_counts: dict[str, int] = {}
    for code, count in failure_event_code_counts.items():
        cls = _failure_class_for_event_code(code)
        if cls is None:
            continue
        failure_event_class_counts[cls] = failure_event_class_counts.get(cls, 0) + count
        if cls == "provider":
            provider_failure_event_code_counts[code] = count
    provider_failure_event_total = sum(provider_failure_event_code_counts.values())
    provider_caused_incompletion = (not run_completed) and primary_failure_class == "provider"
    return FailureSummary(
        primary_failure_class=primary_failure_class,
        secondary_failure_classes=secondary_failure_classes,
        first_terminal_failure_event_code=first_terminal_code,
        failure_event_code_counts=failure_event_code_counts,
        failure_event_class_counts=failure_event_class_counts,
        provider_failure_event_code_counts=provider_failure_event_code_counts,
        provider_failure_event_total=provider_failure_event_total,
        provider_caused_incompletion=provider_caused_incompletion,
    )


def _summarize_finalization_attempts(
    *,
    finalization_fallback_attempts: list[dict[str, Any]],
    finalization_primary_model: str | None,
    forced_final_attempts: int,
    forced_final_circuit_breaker_opened: bool,
) -> FinalizationSummary:
    finalization_attempt_counts_by_model: dict[str, int] = {}
    finalization_failure_counts_by_model: dict[str, int] = {}
    finalization_success_counts_by_model: dict[str, int] = {}
    finalization_failure_code_counts: dict[str, int] = {}
    provider_empty_attempt_counts_by_model: dict[str, int] = {}
    finalization_fallback_attempt_count = 0
    for attempt in finalization_fallback_attempts:
        model_name = str(attempt.get("model", "")).strip() or "<unknown>"
        status = str(attempt.get("status", "")).strip().lower()
        error_code = attempt.get("error_code")
        finalization_attempt_counts_by_model[model_name] = (
            finalization_attempt_counts_by_model.get(model_name, 0) + 1
        )
        if (
            isinstance(finalization_primary_model, str)
            and finalization_primary_model.strip()
            and model_name.lower() != finalization_primary_model.lower()
        ):
            finalization_fallback_attempt_count += 1
        if status == "success":
            finalization_success_counts_by_model[model_name] = (
                finalization_success_counts_by_model.get(model_name, 0) + 1
            )
            continue
        finalization_failure_counts_by_model[model_name] = (
            finalization_failure_counts_by_model.get(model_name, 0) + 1
        )
        if isinstance(error_code, str) and error_code:
            finalization_failure_code_counts[error_code] = (
                finalization_failure_code_counts.get(error_code, 0) + 1
            )
            if error_code == EVENT_CODE_PROVIDER_EMPTY:
                provider_empty_attempt_counts_by_model[model_name] = (
                    provider_empty_attempt_counts_by_model.get(model_name, 0) + 1
                )
    finalization_fallback_usage_rate = (
        (finalization_fallback_attempt_count / forced_final_attempts)
        if forced_final_attempts > 0
        else 0.0
    )
    finalization_breaker_open_rate = 1.0 if forced_final_circuit_breaker_opened else 0.0
    finalization_breaker_open_by_model: dict[str, int] = {}
    if forced_final_circuit_breaker_opened and finalization_fallback_attempts:
        breaker_model = str(finalization_fallback_attempts[-1].get("model", "")).strip() or "<unknown>"
        finalization_breaker_open_by_model[breaker_model] = 1
    return FinalizationSummary(
        finalization_attempt_counts_by_model=finalization_attempt_counts_by_model,
        finalization_failure_counts_by_model=finalization_failure_counts_by_model,
        finalization_success_counts_by_model=finalization_success_counts_by_model,
        finalization_failure_code_counts=finalization_failure_code_counts,
        provider_empty_attempt_counts_by_model=provider_empty_attempt_counts_by_model,
        finalization_fallback_attempt_count=finalization_fallback_attempt_count,
        finalization_fallback_usage_rate=finalization_fallback_usage_rate,
        finalization_breaker_open_rate=finalization_breaker_open_rate,
        finalization_breaker_open_by_model=finalization_breaker_open_by_model,
    )
