"""Post-tool outcome handling for MCP agent turns.

This module owns the turn-local policy and bookkeeping that happens after tool
execution finishes: argument-coercion warnings, evidence-digest updates,
retrieval-stagnation policy, submit-answer bookkeeping, TODO-state injection,
and control-churn threshold handling. Keeping that block out of the main turn
orchestrator makes the orchestration loop easier to read without changing the
governed MCP-agent behavior.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Pattern

from llm_client.foundation import new_event_id, now_iso
from llm_client.agent.mcp_evidence import (
    _evidence_digest,
    _is_evidence_tool_name,
    _tool_evidence_pointer_labels,
)
from llm_client.agent.mcp_finalization import _FORCED_REFUSAL_RE
from llm_client.agent.mcp_tools import _parse_record_result_json
from llm_client.tools.tool_runtime_common import MCPToolCallRecord

logger = logging.getLogger(__name__)


@dataclass
class AgentTurnOutcomeResult:
    """Updated turn-local state after post-tool outcome processing."""

    tool_arg_coercions_delta: int
    tool_arg_coercion_calls_delta: int
    tool_arg_validation_rejections_delta: int
    submit_answer_call_count_delta: int
    submit_answer_succeeded_now: bool
    submitted_answer_value: str | None
    fallback_submit_guess_value: str | None
    submit_requires_new_evidence: bool
    submit_evidence_digest_at_last_failure: str | None
    evidence_pointer_count: int
    evidence_digest_change_count_delta: int
    evidence_turns_total_delta: int
    evidence_turns_with_new_evidence_delta: int
    evidence_turns_without_new_evidence_delta: int
    retrieval_stagnation_streak: int
    retrieval_stagnation_streak_max: int
    retrieval_stagnation_alerted_for_current_streak: bool
    retrieval_stagnation_triggered: bool
    retrieval_stagnation_turn: int | None
    prev_turn_had_evidence_tools: bool
    prev_turn_deficit_digest: str | None
    zero_exec_tool_turn_streak: int
    max_zero_exec_tool_turn_streak: int
    last_todo_status_line: str | None
    emitted_messages: list[dict[str, Any]]
    warnings: list[str]
    failure_event_codes: list[str]
    force_final_reason: str | None
    stop_agent_loop: bool


def _process_turn_outcomes(
    *,
    turn: int,
    max_turns: int,
    records: list[MCPToolCallRecord],
    executed_records: list[MCPToolCallRecord],
    tool_calls_this_turn: list[dict[str, Any]],
    tool_calls_to_execute: list[dict[str, Any]],
    evidence_pointer_labels: set[str],
    submit_requires_new_evidence: bool,
    submit_evidence_digest_at_last_failure: str | None,
    current_turn_deficit_digest: str | None,
    retrieval_stagnation_streak: int,
    retrieval_stagnation_streak_max: int,
    retrieval_stagnation_alerted_for_current_streak: bool,
    retrieval_stagnation_turns: int,
    retrieval_stagnation_action: str,
    retrieval_stagnation_turn: int | None,
    zero_exec_tool_turn_streak: int,
    max_zero_exec_tool_turn_streak: int,
    last_todo_status_line: str | None,
    submit_validation_reason_counts: dict[str, int],
    contract_rejected_record_count: int,
    suppressed_record_count: int,
    available_artifacts: set[str],
    available_bindings: dict[str, Any],
    foundation_run_id: str,
    foundation_session_id: str,
    foundation_actor_id: str,
    emit_foundation_event: Any,
    event_code_retrieval_stagnation: str,
    event_code_retrieval_stagnation_observed: str,
    event_code_control_churn_threshold: str,
    control_churn_turn_threshold: int,
    forced_refusal_pattern: Pattern[str] = _FORCED_REFUSAL_RE,
) -> AgentTurnOutcomeResult:
    """Update post-tool turn state and emit policy messages/events.

    The orchestrator has already executed the tool calls for the turn. This
    helper consolidates the follow-on bookkeeping and policy checks that depend
    on those results, while leaving higher-level loop sequencing in
    ``mcp_turn_execution``.
    """

    emitted_messages: list[dict[str, Any]] = []
    warnings: list[str] = []
    failure_event_codes: list[str] = []
    force_final_reason: str | None = None
    stop_agent_loop = False

    tool_arg_coercions_delta = 0
    tool_arg_coercion_calls_delta = 0
    tool_arg_validation_rejections_delta = 0
    for record in records:
        coercions = getattr(record, "arg_coercions", None) or []
        if coercions:
            tool_arg_coercions_delta += len(coercions)
            tool_arg_coercion_calls_delta += 1
        if record.error and "validation error:" in record.error.lower():
            tool_arg_validation_rejections_delta += 1

    if tool_arg_coercions_delta:
        warning = (
            "TOOL_ARG_COERCION: turn "
            f"{turn + 1}/{max_turns} applied {tool_arg_coercions_delta} coercion(s) "
            f"across {tool_arg_coercion_calls_delta} call(s)"
        )
        warnings.append(warning)
        logger.warning(warning)
    if tool_arg_validation_rejections_delta:
        warning = (
            "TOOL_ARG_VALIDATION: turn "
            f"{turn + 1}/{max_turns} rejected {tool_arg_validation_rejections_delta} "
            "tool call(s) due to argument/schema mismatch"
        )
        warnings.append(warning)
        logger.warning(warning)

    submit_answer_call_count_delta = 0
    submit_answer_succeeded_now = False
    submitted_answer_value: str | None = None
    fallback_submit_guess_value: str | None = None
    submit_error_this_turn = False
    submit_errors: list[str] = []
    submit_needs_new_evidence_signal = False

    evidence_digest_before_turn = _evidence_digest(evidence_pointer_labels)
    for record in executed_records:
        evidence_pointer_labels.update(_tool_evidence_pointer_labels(record))
    evidence_pointer_count = len(evidence_pointer_labels)
    evidence_digest_after_turn = _evidence_digest(evidence_pointer_labels)
    new_evidence_this_turn = evidence_digest_after_turn != evidence_digest_before_turn
    evidence_digest_change_count_delta = int(new_evidence_this_turn)
    if (
        new_evidence_this_turn
        and submit_requires_new_evidence
        and submit_evidence_digest_at_last_failure is not None
        and evidence_digest_after_turn != submit_evidence_digest_at_last_failure
    ):
        submit_requires_new_evidence = False
        submit_evidence_digest_at_last_failure = None

    evidence_tools_executed = [
        rec.tool for rec in executed_records if _is_evidence_tool_name(rec.tool) and not rec.error
    ]
    evidence_turns_total_delta = 0
    evidence_turns_with_new_evidence_delta = 0
    evidence_turns_without_new_evidence_delta = 0
    if evidence_tools_executed:
        evidence_turns_total_delta = 1
        if new_evidence_this_turn:
            evidence_turns_with_new_evidence_delta = 1
            retrieval_stagnation_streak = 0
            retrieval_stagnation_alerted_for_current_streak = False
        else:
            evidence_turns_without_new_evidence_delta = 1
            retrieval_stagnation_streak += 1
    else:
        retrieval_stagnation_streak = 0
        retrieval_stagnation_alerted_for_current_streak = False
    if retrieval_stagnation_streak > retrieval_stagnation_streak_max:
        retrieval_stagnation_streak_max = retrieval_stagnation_streak

    prev_turn_had_evidence_tools = bool(evidence_tools_executed)
    prev_turn_deficit_digest = current_turn_deficit_digest
    retrieval_stagnation_triggered = False

    if (
        retrieval_stagnation_streak >= retrieval_stagnation_turns
        and not retrieval_stagnation_alerted_for_current_streak
    ):
        retrieval_stagnation_triggered = True
        if retrieval_stagnation_turn is None:
            retrieval_stagnation_turn = turn + 1
        stagnation_event_code = (
            event_code_retrieval_stagnation
            if retrieval_stagnation_action == "force_final"
            else event_code_retrieval_stagnation_observed
        )
        failure_event_codes.append(stagnation_event_code)
        warning = (
            "RETRIEVAL_STAGNATION: "
            f"{retrieval_stagnation_streak} consecutive evidence turns produced no new evidence refs."
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
                "operation": {"name": "__retrieval_stagnation__", "version": None},
                "inputs": {
                    "artifact_ids": sorted(available_artifacts),
                    "params": {
                        "streak": retrieval_stagnation_streak,
                        "turn": turn + 1,
                        "evidence_tools": evidence_tools_executed,
                    },
                    "bindings": dict(available_bindings),
                },
                "outputs": {"artifact_ids": [], "payload_hashes": []},
                "failure": {
                    "error_code": stagnation_event_code,
                    "category": "policy",
                    "phase": "post_validation",
                    "retryable": False,
                    "tool_name": "__retrieval_stagnation__",
                    "user_message": warning,
                    "debug_ref": None,
                },
            }
        )
        if retrieval_stagnation_action == "force_final":
            emitted_messages.append(
                {
                    "role": "user",
                    "content": (
                        "[SYSTEM: Evidence has stagnated across consecutive retrieval turns. "
                        "Stop repeating equivalent searches. Verify a different bridge entity, "
                        "run a conversion tool, or submit your best answer now.]"
                    ),
                }
            )
            retrieval_stagnation_alerted_for_current_streak = True
            force_final_reason = "retrieval_stagnation"
            stop_agent_loop = True
            return AgentTurnOutcomeResult(
                tool_arg_coercions_delta=tool_arg_coercions_delta,
                tool_arg_coercion_calls_delta=tool_arg_coercion_calls_delta,
                tool_arg_validation_rejections_delta=tool_arg_validation_rejections_delta,
                submit_answer_call_count_delta=submit_answer_call_count_delta,
                submit_answer_succeeded_now=submit_answer_succeeded_now,
                submitted_answer_value=submitted_answer_value,
                fallback_submit_guess_value=fallback_submit_guess_value,
                submit_requires_new_evidence=submit_requires_new_evidence,
                submit_evidence_digest_at_last_failure=submit_evidence_digest_at_last_failure,
                evidence_pointer_count=evidence_pointer_count,
                evidence_digest_change_count_delta=evidence_digest_change_count_delta,
                evidence_turns_total_delta=evidence_turns_total_delta,
                evidence_turns_with_new_evidence_delta=evidence_turns_with_new_evidence_delta,
                evidence_turns_without_new_evidence_delta=evidence_turns_without_new_evidence_delta,
                retrieval_stagnation_streak=retrieval_stagnation_streak,
                retrieval_stagnation_streak_max=retrieval_stagnation_streak_max,
                retrieval_stagnation_alerted_for_current_streak=(
                    retrieval_stagnation_alerted_for_current_streak
                ),
                retrieval_stagnation_triggered=retrieval_stagnation_triggered,
                retrieval_stagnation_turn=retrieval_stagnation_turn,
                prev_turn_had_evidence_tools=prev_turn_had_evidence_tools,
                prev_turn_deficit_digest=prev_turn_deficit_digest,
                zero_exec_tool_turn_streak=zero_exec_tool_turn_streak,
                max_zero_exec_tool_turn_streak=max_zero_exec_tool_turn_streak,
                last_todo_status_line=last_todo_status_line,
                emitted_messages=emitted_messages,
                warnings=warnings,
                failure_event_codes=failure_event_codes,
                force_final_reason=force_final_reason,
                stop_agent_loop=stop_agent_loop,
            )

        emitted_messages.append(
            {
                "role": "user",
                "content": (
                    "[SYSTEM: Evidence stagnation observed. Do NOT repeat equivalent retrieval. "
                    "Pivot strategy now (different bridge, conversion, or graph path) and continue.]"
                ),
            }
        )
        retrieval_stagnation_alerted_for_current_streak = True

    for record in records:
        if record.tool != "submit_answer":
            continue
        submit_answer_call_count_delta += 1
        arg_answer = record.arguments.get("answer") if isinstance(record.arguments, dict) else None
        if (
            isinstance(arg_answer, str)
            and arg_answer.strip()
            and not forced_refusal_pattern.match(arg_answer.strip())
        ):
            fallback_submit_guess_value = arg_answer.strip()
        if record.error:
            submit_error_this_turn = True
            submit_errors.append(str(record.error))
            continue

        parsed = _parse_record_result_json(record)
        if parsed is not None:
            status = str(parsed.get("status", "")).strip().lower()
            if status and status not in {"submitted", "submitted_with_warning", "ok", "success"}:
                submit_error_this_turn = True
                validation_payload = parsed.get("validation_error")
                reason_code = ""
                detail = ""
                recovery_policy = parsed.get("recovery_policy")
                if (
                    isinstance(recovery_policy, dict)
                    and bool(recovery_policy.get("new_evidence_required_before_retry"))
                ):
                    submit_needs_new_evidence_signal = True
                if isinstance(validation_payload, dict):
                    reason_code = str(validation_payload.get("reason_code", "")).strip()
                    detail = str(validation_payload.get("message", "")).strip()
                if reason_code:
                    submit_validation_reason_counts[reason_code] = (
                        submit_validation_reason_counts.get(reason_code, 0) + 1
                    )
                err_parts = [f"submit_answer not accepted (status={status})"]
                if reason_code:
                    err_parts.append(f"reason_code={reason_code}")
                if detail:
                    err_parts.append(detail)
                submit_errors.append(" | ".join(err_parts))
                continue

        submit_answer_succeeded_now = True
        if isinstance(arg_answer, str) and arg_answer.strip():
            submitted_answer_value = arg_answer.strip()
            continue
        if isinstance(parsed, dict):
            parsed_answer = parsed.get("answer")
            if isinstance(parsed_answer, str) and parsed_answer.strip():
                submitted_answer_value = parsed_answer.strip()

    if submit_error_this_turn:
        refusal_blocked = any(
            ("refusal-style" in err.lower()) or ("not acceptable" in err.lower())
            for err in submit_errors
        )
        if submit_needs_new_evidence_signal and not new_evidence_this_turn:
            submit_requires_new_evidence = True
            submit_evidence_digest_at_last_failure = evidence_digest_after_turn

        # Generic repeated-rejection suppression: after 3+ consecutive identical
        # rejections for any reason_code, force evidence gate even if the validator
        # didn't explicitly request it. This breaks submit -> rejected -> submit
        # churn loops that waste turns without progress.
        if not submit_requires_new_evidence:
            max_reason_count = max(submit_validation_reason_counts.values(), default=0) if submit_validation_reason_counts else 0
            # Also count total submit failures (including ValueError-style rejections
            # like refusal text, empty answer) by checking the overall call count
            # vs success ratio: if submit_answer_call_count_delta > 0 and none
            # succeeded, and we already had 3+ failed reasons, activate the gate.
            if max_reason_count >= 3:
                submit_requires_new_evidence = True
                submit_evidence_digest_at_last_failure = evidence_digest_after_turn
                logger.info(
                    "Repeated submit rejection detected (max_count=%d) — "
                    "activating evidence gate to break churn loop",
                    max_reason_count,
                )

        evidence_fix_hint = (
            " Validator requires NEW evidence refs before retry. "
            "Run at least one non-control evidence tool call that yields new "
            "chunk/entity-backed evidence before submit."
            if submit_requires_new_evidence
            else ""
        )
        emitted_messages.append(
            {
                "role": "user",
                "content": (
                    "[SYSTEM: submit_answer failed. "
                    + evidence_fix_hint
                    + (
                        "Do NOT use refusal text (cannot, unknown, insufficient, no such, not found). "
                        "Submit the single best factual guess from retrieved evidence. "
                        if refusal_blocked
                        else ""
                    )
                    + "Call submit_answer again with answer as a short fact only "
                    "(name/date/number/yes/no, <=8 words).]"
                ),
            }
        )

    todo_write_called_this_turn = False
    updated_last_todo_status_line = last_todo_status_line
    for record in records:
        if record.tool == "todo_write" and not record.error:
            todo_write_called_this_turn = True
            parsed = _parse_record_result_json(record)
            if isinstance(parsed, dict):
                status_line = parsed.get("status_line")
                if isinstance(status_line, str) and status_line.strip():
                    updated_last_todo_status_line = status_line.strip()
    if not todo_write_called_this_turn and updated_last_todo_status_line:
        emitted_messages.append(
            {
                "role": "user",
                "content": f"[TODO_STATE] {updated_last_todo_status_line}",
            }
        )

    if tool_calls_this_turn and not tool_calls_to_execute:
        zero_exec_tool_turn_streak += 1
    else:
        zero_exec_tool_turn_streak = 0
    if zero_exec_tool_turn_streak > max_zero_exec_tool_turn_streak:
        max_zero_exec_tool_turn_streak = zero_exec_tool_turn_streak

    if zero_exec_tool_turn_streak >= control_churn_turn_threshold:
        blocked_tools = sorted(
            {
                str(tc.get("function", {}).get("name", "")).strip() or "<unknown>"
                for tc in tool_calls_this_turn
                if isinstance(tc, dict)
            }
        )
        warning = (
            "CONTROL_CHURN: consecutive turns with tool calls but no executable calls "
            f"({zero_exec_tool_turn_streak} turns). Forcing final answer."
        )
        warnings.append(warning)
        logger.warning(warning)
        failure_event_codes.append(event_code_control_churn_threshold)
        emitted_messages.append(
            {
                "role": "user",
                "content": (
                    "[SYSTEM: Repeated non-executable tool-call loop detected "
                    f"({zero_exec_tool_turn_streak} turns). Stop calling tools and "
                    "submit your best answer now from existing evidence.]"
                ),
            }
        )
        emit_foundation_event(
            {
                "event_id": new_event_id(),
                "event_type": "ToolFailed",
                "timestamp": now_iso(),
                "run_id": foundation_run_id,
                "session_id": foundation_session_id,
                "actor_id": foundation_actor_id,
                "operation": {"name": "__control_churn__", "version": None},
                "inputs": {
                    "artifact_ids": sorted(available_artifacts),
                    "params": {
                        "blocked_tools": blocked_tools,
                        "streak": zero_exec_tool_turn_streak,
                        "contract_rejections": contract_rejected_record_count,
                        "suppressed_calls": suppressed_record_count,
                    },
                    "bindings": dict(available_bindings),
                },
                "outputs": {"artifact_ids": [], "payload_hashes": []},
                "failure": {
                    "error_code": event_code_control_churn_threshold,
                    "category": "policy",
                    "phase": "post_validation",
                    "retryable": False,
                    "tool_name": "__control_churn__",
                    "user_message": warning,
                    "debug_ref": None,
                },
            }
        )
        force_final_reason = "control_churn"
        stop_agent_loop = True

    return AgentTurnOutcomeResult(
        tool_arg_coercions_delta=tool_arg_coercions_delta,
        tool_arg_coercion_calls_delta=tool_arg_coercion_calls_delta,
        tool_arg_validation_rejections_delta=tool_arg_validation_rejections_delta,
        submit_answer_call_count_delta=submit_answer_call_count_delta,
        submit_answer_succeeded_now=submit_answer_succeeded_now,
        submitted_answer_value=submitted_answer_value,
        fallback_submit_guess_value=fallback_submit_guess_value,
        submit_requires_new_evidence=submit_requires_new_evidence,
        submit_evidence_digest_at_last_failure=submit_evidence_digest_at_last_failure,
        evidence_pointer_count=evidence_pointer_count,
        evidence_digest_change_count_delta=evidence_digest_change_count_delta,
        evidence_turns_total_delta=evidence_turns_total_delta,
        evidence_turns_with_new_evidence_delta=evidence_turns_with_new_evidence_delta,
        evidence_turns_without_new_evidence_delta=evidence_turns_without_new_evidence_delta,
        retrieval_stagnation_streak=retrieval_stagnation_streak,
        retrieval_stagnation_streak_max=retrieval_stagnation_streak_max,
        retrieval_stagnation_alerted_for_current_streak=(
            retrieval_stagnation_alerted_for_current_streak
        ),
        retrieval_stagnation_triggered=retrieval_stagnation_triggered,
        retrieval_stagnation_turn=retrieval_stagnation_turn,
        prev_turn_had_evidence_tools=prev_turn_had_evidence_tools,
        prev_turn_deficit_digest=prev_turn_deficit_digest,
        zero_exec_tool_turn_streak=zero_exec_tool_turn_streak,
        max_zero_exec_tool_turn_streak=max_zero_exec_tool_turn_streak,
        last_todo_status_line=updated_last_todo_status_line,
        emitted_messages=emitted_messages,
        warnings=warnings,
        failure_event_codes=failure_event_codes,
        force_final_reason=force_final_reason,
        stop_agent_loop=stop_agent_loop,
    )
