"""Forced-finalization handoff and submit-completion policy for MCP turns.

This module owns the post-loop completion path after the turn orchestrator has
either finished naturally or chosen a forced-final reason. It keeps the async
forced-finalization handoff, submit exhaustion policy, and required-submit
failure handling out of ``mcp_turn_execution`` so the main loop can focus on
turn sequencing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from llm_client.foundation import new_event_id, now_iso
from llm_client.mcp_finalization import (
    _execute_forced_finalization,
    _normalize_forced_final_answer,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentTurnCompletionResult:
    """Resolved run-completion state after the MCP turn loop exits."""

    final_content: str
    final_finish_reason: str
    finalization_primary_model: str | None
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
    required_submit_missing: bool
    submit_forced_retry_on_budget_exhaustion: bool
    submit_forced_accept_on_budget_exhaustion: bool
    submit_answer_call_count_delta: int
    submit_answer_succeeded: bool
    submitted_answer_value: str | None
    warnings: list[str]


async def _resolve_turn_completion(
    *,
    force_final_reason: str | None,
    max_turns: int,
    max_message_chars: int,
    tool_result_keep_recent: int,
    tool_result_context_preview_chars: int,
    messages: list[dict[str, Any]],
    tool_result_metadata_by_id: dict[str, dict[str, Any]],
    artifact_context_message_index: int | None,
    kwargs: dict[str, Any],
    timeout: float | None,
    effective_model: str,
    attempted_models: list[str],
    finalization_fallback_models: list[str],
    forced_final_max_attempts: int,
    forced_final_circuit_breaker_threshold: int,
    available_artifacts: set[str],
    available_capabilities: dict[str, set[tuple[str | None, str | None, str | None]]],
    available_bindings: dict[str, Any],
    active_artifact_context_enabled: bool,
    active_artifact_context_max_handles: int,
    active_artifact_context_max_chars: int,
    foundation_run_id: str,
    foundation_session_id: str,
    foundation_actor_id: str,
    agent_result: Any,
    emit_foundation_event: Any,
    final_content: str,
    final_finish_reason: str,
    finalization_primary_model: str | None,
    forced_final_attempts: int,
    forced_final_circuit_breaker_opened: bool,
    finalization_fallback_used: bool,
    finalization_fallback_succeeded: bool,
    finalization_events: list[str],
    finalization_fallback_attempts: list[dict[str, Any]],
    requires_submit_answer: bool,
    submit_answer_call_count: int,
    submit_answer_succeeded: bool,
    submitted_answer_value: str | None,
    force_submit_retry_on_max_tool_calls: bool,
    accept_forced_answer_on_max_tool_calls: bool,
    fallback_submit_guess_value: str | None,
    event_code_submit_forced_accept_budget_exhaustion: str,
    event_code_submit_forced_accept_turn_exhaustion: str,
    event_code_submit_forced_accept_forced_final: str,
    event_code_required_submit_not_attempted: str,
    event_code_required_submit_not_accepted: str,
) -> AgentTurnCompletionResult:
    """Resolve forced-finalization adoption and required-submit completion."""

    failure_event_codes: list[str] = []
    warnings: list[str] = []
    context_tool_result_clearings_delta = 0
    context_tool_results_cleared_delta = 0
    context_tool_result_cleared_chars_delta = 0
    context_compactions_delta = 0
    context_compacted_messages_delta = 0
    context_compacted_chars_delta = 0
    total_cost_delta = 0.0
    total_input_tokens_delta = 0
    total_output_tokens_delta = 0
    total_cached_tokens_delta = 0
    total_cache_creation_tokens_delta = 0
    turns_delta = 0
    submit_forced_retry_on_budget_exhaustion = False
    submit_forced_accept_on_budget_exhaustion = False
    submit_answer_call_count_delta = 0

    if force_final_reason is not None and not submit_answer_succeeded:
        forced_final_result = await _execute_forced_finalization(
            force_final_reason=force_final_reason,
            max_turns=max_turns,
            max_message_chars=max_message_chars,
            tool_result_keep_recent=tool_result_keep_recent,
            tool_result_context_preview_chars=tool_result_context_preview_chars,
            messages=messages,
            tool_result_metadata_by_id=tool_result_metadata_by_id,
            artifact_context_message_index=artifact_context_message_index,
            kwargs=kwargs,
            timeout=timeout,
            effective_model=effective_model,
            attempted_models=attempted_models,
            finalization_fallback_models=finalization_fallback_models,
            forced_final_max_attempts=forced_final_max_attempts,
            forced_final_circuit_breaker_threshold=forced_final_circuit_breaker_threshold,
            available_artifacts=available_artifacts,
            available_capabilities=available_capabilities,
            available_bindings=available_bindings,
            active_artifact_context_enabled=active_artifact_context_enabled,
            active_artifact_context_max_handles=active_artifact_context_max_handles,
            active_artifact_context_max_chars=active_artifact_context_max_chars,
            foundation_run_id=foundation_run_id,
            foundation_session_id=foundation_session_id,
            foundation_actor_id=foundation_actor_id,
            agent_result=agent_result,
            emit_foundation_event=emit_foundation_event,
        )
        final_content = forced_final_result.final_content
        final_finish_reason = forced_final_result.final_finish_reason
        finalization_primary_model = forced_final_result.finalization_primary_model
        forced_final_attempts = forced_final_result.forced_final_attempts
        forced_final_circuit_breaker_opened = (
            forced_final_result.forced_final_circuit_breaker_opened
        )
        finalization_fallback_used = forced_final_result.finalization_fallback_used
        finalization_fallback_succeeded = (
            forced_final_result.finalization_fallback_succeeded
        )
        finalization_events = list(forced_final_result.finalization_events)
        finalization_fallback_attempts = list(
            forced_final_result.finalization_fallback_attempts
        )
        failure_event_codes.extend(forced_final_result.failure_event_codes)
        context_tool_result_clearings_delta += (
            forced_final_result.context_tool_result_clearings_delta
        )
        context_tool_results_cleared_delta += (
            forced_final_result.context_tool_results_cleared_delta
        )
        context_tool_result_cleared_chars_delta += (
            forced_final_result.context_tool_result_cleared_chars_delta
        )
        context_compactions_delta += forced_final_result.context_compactions_delta
        context_compacted_messages_delta += (
            forced_final_result.context_compacted_messages_delta
        )
        context_compacted_chars_delta += (
            forced_final_result.context_compacted_chars_delta
        )
        total_cost_delta += forced_final_result.total_cost_delta
        total_input_tokens_delta += forced_final_result.total_input_tokens_delta
        total_output_tokens_delta += forced_final_result.total_output_tokens_delta
        total_cached_tokens_delta += forced_final_result.total_cached_tokens_delta
        total_cache_creation_tokens_delta += (
            forced_final_result.total_cache_creation_tokens_delta
        )
        turns_delta += forced_final_result.turns_delta

    required_submit_missing = requires_submit_answer and not submit_answer_succeeded
    forced_exhaustion_reason: str | None = None
    if force_final_reason == "max_tool_calls":
        forced_exhaustion_reason = "budget"
    elif force_final_reason == "max_turns":
        forced_exhaustion_reason = "turns"
    elif force_final_reason is not None:
        forced_exhaustion_reason = force_final_reason

    if (
        forced_exhaustion_reason is not None
        and requires_submit_answer
        and not submit_answer_succeeded
    ):
        if force_submit_retry_on_max_tool_calls:
            submit_forced_retry_on_budget_exhaustion = True
            submit_answer_call_count_delta += 1
            if forced_exhaustion_reason == "budget":
                warning = (
                    "SUBMIT_FORCED_RETRY_BUDGET_EXHAUSTION: counted one forced submit retry "
                    "attempt after tool budget exhaustion."
                )
            elif forced_exhaustion_reason == "turns":
                warning = (
                    "SUBMIT_FORCED_RETRY_TURN_EXHAUSTION: counted one forced submit retry "
                    "attempt after max-turn exhaustion."
                )
            else:
                warning = (
                    "SUBMIT_FORCED_RETRY_FORCED_FINAL: counted one forced submit retry "
                    f"attempt after forced-final reason '{forced_exhaustion_reason}'."
                )
            warnings.append(warning)
            logger.warning(warning)

        has_fallback_guess = bool(
            isinstance(fallback_submit_guess_value, str)
            and fallback_submit_guess_value.strip()
        )
        forced_final_succeeded = final_finish_reason != "error"
        has_final_content = forced_final_succeeded and bool(final_content.strip())
        if accept_forced_answer_on_max_tool_calls and (
            has_final_content or has_fallback_guess
        ):
            submit_forced_accept_on_budget_exhaustion = True
            submit_answer_succeeded = True
            if has_fallback_guess and fallback_submit_guess_value is not None:
                normalized_forced_answer = fallback_submit_guess_value.strip()
            elif has_final_content:
                normalized_forced_answer = _normalize_forced_final_answer(final_content)
            else:
                normalized_forced_answer = ""
            submitted_answer_value = (
                submitted_answer_value
                or normalized_forced_answer
                or (final_content.strip() if forced_final_succeeded else "")
            )
            required_submit_missing = False
            if forced_exhaustion_reason == "budget":
                warning = (
                    "SUBMIT_FORCED_ACCEPT_BUDGET_EXHAUSTION: accepted forced-final answer "
                    "without grounding validation because tool budget was exhausted."
                )
                failure_event_codes.append(
                    event_code_submit_forced_accept_budget_exhaustion
                )
            elif forced_exhaustion_reason == "turns":
                warning = (
                    "SUBMIT_FORCED_ACCEPT_TURN_EXHAUSTION: accepted forced-final answer "
                    "without grounding validation because max turns were exhausted."
                )
                failure_event_codes.append(
                    event_code_submit_forced_accept_turn_exhaustion
                )
            else:
                warning = (
                    "SUBMIT_FORCED_ACCEPT_FORCED_FINAL: accepted forced-final answer "
                    "without grounding validation after forced-final termination "
                    f"('{forced_exhaustion_reason}')."
                )
                failure_event_codes.append(
                    event_code_submit_forced_accept_forced_final
                )
            warnings.append(warning)
            logger.warning(warning)

    if required_submit_missing:
        submit_failure_code = (
            event_code_required_submit_not_attempted
            if submit_answer_call_count + submit_answer_call_count_delta == 0
            else event_code_required_submit_not_accepted
        )
        if not failure_event_codes:
            failure_event_codes.append(submit_failure_code)
        warning = (
            "REQUIRED_SUBMIT: submit_answer tool is available but no accepted submit was recorded. "
            f"submit_answer_call_count={submit_answer_call_count + submit_answer_call_count_delta}."
        )
        warnings.append(warning)
        logger.warning(warning)
        emit_foundation_event(
            {
                "event_id": new_event_id(),
                "event_type": "ToolFailed",
                "timestamp": now_iso(),
                "run_id": foundation_run_id,
                "session_id": foundation_session_id,
                "actor_id": foundation_actor_id,
                "operation": {"name": "submit_answer", "version": None},
                "inputs": {
                    "artifact_ids": sorted(available_artifacts),
                    "params": {
                        "submit_answer_call_count": (
                            submit_answer_call_count + submit_answer_call_count_delta
                        ),
                        "submit_answer_succeeded": submit_answer_succeeded,
                        "reason": "required_submit_not_satisfied",
                    },
                    "bindings": dict(available_bindings),
                },
                "outputs": {"artifact_ids": [], "payload_hashes": []},
                "failure": {
                    "error_code": submit_failure_code,
                    "category": "policy",
                    "phase": "post_validation",
                    "retryable": False,
                    "tool_name": "submit_answer",
                    "user_message": warning,
                    "debug_ref": None,
                },
            }
        )

    return AgentTurnCompletionResult(
        final_content=final_content,
        final_finish_reason=final_finish_reason,
        finalization_primary_model=finalization_primary_model,
        forced_final_attempts=forced_final_attempts,
        forced_final_circuit_breaker_opened=forced_final_circuit_breaker_opened,
        finalization_fallback_used=finalization_fallback_used,
        finalization_fallback_succeeded=finalization_fallback_succeeded,
        finalization_events=list(finalization_events),
        finalization_fallback_attempts=list(finalization_fallback_attempts),
        failure_event_codes=failure_event_codes,
        context_tool_result_clearings_delta=context_tool_result_clearings_delta,
        context_tool_results_cleared_delta=context_tool_results_cleared_delta,
        context_tool_result_cleared_chars_delta=(
            context_tool_result_cleared_chars_delta
        ),
        context_compactions_delta=context_compactions_delta,
        context_compacted_messages_delta=context_compacted_messages_delta,
        context_compacted_chars_delta=context_compacted_chars_delta,
        total_cost_delta=total_cost_delta,
        total_input_tokens_delta=total_input_tokens_delta,
        total_output_tokens_delta=total_output_tokens_delta,
        total_cached_tokens_delta=total_cached_tokens_delta,
        total_cache_creation_tokens_delta=total_cache_creation_tokens_delta,
        turns_delta=turns_delta,
        required_submit_missing=required_submit_missing,
        submit_forced_retry_on_budget_exhaustion=(
            submit_forced_retry_on_budget_exhaustion
        ),
        submit_forced_accept_on_budget_exhaustion=(
            submit_forced_accept_on_budget_exhaustion
        ),
        submit_answer_call_count_delta=submit_answer_call_count_delta,
        submit_answer_succeeded=submit_answer_succeeded,
        submitted_answer_value=submitted_answer_value,
        warnings=warnings,
    )
