"""End-of-run summary and metadata writeout for MCP agent loops.

The MCP turn orchestrator accumulates a large amount of per-run state. This
module turns that state into durable result metadata and summary warnings at
the end of the run so the main loop can stay focused on turn-by-turn control
flow instead of the final bookkeeping block.
"""

from __future__ import annotations

import logging
from typing import Any

from llm_client.agent.agent_adoption import AdoptionProfileAssessment
from llm_client.agent.agent_outcomes import (
    _PRIMARY_FAILURE_PRIORITY,
    _TERMINAL_FAILURE_EVENT_CODES,
    _summarize_failure_events,
    _summarize_finalization_attempts,
)
from llm_client.agent.mcp_contracts import (
    _capability_state_snapshot,
    _full_bindings_spec,
    _full_bindings_state_hash,
    _hard_bindings_spec,
    _hard_bindings_state_hash,
)
from llm_client.agent.mcp_evidence import _evidence_digest
from llm_client.agent.mcp_tools import BUDGET_EXEMPT_TOOL_NAMES, _count_budgeted_records
from llm_client.tools.tool_runtime_common import MCPAgentResult

logger = logging.getLogger(__name__)


def _apply_agent_loop_summary(
    *,
    agent_result: MCPAgentResult,
    model: str,
    effective_model: str,
    attempted_models: list[str],
    sticky_fallback: bool,
    total_cost: float,
    max_turns: int,
    max_tool_calls: int | None,
    forced_final_attempts: int,
    forced_final_max_attempts: int,
    forced_final_circuit_breaker_threshold: int,
    forced_final_breaker_effective: bool,
    force_submit_retry_on_max_tool_calls: bool,
    accept_forced_answer_on_max_tool_calls: bool,
    forced_final_circuit_breaker_opened: bool,
    finalization_primary_model: str | None,
    finalization_fallback_models: list[str],
    finalization_fallback_used: bool,
    finalization_fallback_succeeded: bool,
    finalization_fallback_attempts: list[dict[str, Any]],
    finalization_events: list[str],
    final_content: str,
    final_finish_reason: str,
    requires_submit_answer: bool,
    submit_answer_call_count: int,
    submit_answer_succeeded: bool,
    required_submit_missing: bool,
    submitted_answer_value: str | None,
    submit_forced_retry_on_budget_exhaustion: bool,
    submit_forced_accept_on_budget_exhaustion: bool,
    require_tool_reasoning: bool,
    rejected_missing_reasoning_calls: int,
    control_loop_suppressed_calls: int,
    submit_validation_reason_counts: dict[str, int],
    tool_call_turns_total: int,
    tool_call_empty_text_turns: int,
    responses_tool_call_empty_text_turns: int,
    tool_arg_coercions: int,
    tool_arg_coercion_calls: int,
    tool_arg_validation_rejections: int,
    tool_result_keep_recent: int,
    tool_result_context_preview_chars: int,
    active_artifact_context_enabled: bool,
    active_artifact_context_max_handles: int,
    active_artifact_context_max_chars: int,
    artifact_context_updates: int,
    artifact_context_chars: int,
    context_tool_result_clearings: int,
    context_tool_results_cleared: int,
    context_tool_result_cleared_chars: int,
    tool_result_metadata_by_id: dict[str, dict[str, Any]],
    context_compactions: int,
    context_compacted_messages: int,
    context_compacted_chars: int,
    enforce_tool_contracts: bool,
    progressive_tool_disclosure: bool,
    suppress_control_loop_calls: bool,
    gate_rejected_calls: int,
    gate_violation_events: list[dict[str, Any]],
    contract_rejected_calls: int,
    contract_violation_events: list[dict[str, Any]],
    normalized_tool_contracts: dict[str, dict[str, Any]],
    runtime_artifact_tool_name: str,
    runtime_artifact_registry_by_id: dict[str, dict[str, Any]],
    handle_input_resolution_count: int,
    handle_input_resolved_artifact_count: int,
    initial_artifact_snapshot: list[str],
    available_artifacts: set[str],
    initial_capability_snapshot: list[dict[str, Any]],
    available_capabilities: dict[str, set[tuple[str | None, str | None, str | None]]],
    artifact_timeline: list[dict[str, Any]],
    initial_binding_snapshot: dict[str, Any],
    available_bindings: dict[str, Any],
    run_config_spec: dict[str, Any],
    run_config_hash: str,
    adoption_assessment: AdoptionProfileAssessment,
    tool_disclosure_turns: int,
    tool_disclosure_hidden_total: int,
    tool_disclosure_unavailable_msgs: int,
    tool_disclosure_unavailable_reason_chars: int,
    tool_disclosure_unavailable_reason_tokens_est: int,
    tool_disclosure_repair_suggestions: int,
    lane_closure_analysis: dict[str, Any],
    no_legal_noncontrol_turns: int,
    max_deficit_no_progress_streak: int,
    deficit_no_progress_nudges: int,
    max_zero_exec_tool_turn_streak: int,
    retrieval_no_hits_count: int,
    retrieval_stagnation_turns: int,
    retrieval_stagnation_action: str,
    retrieval_stagnation_streak: int,
    retrieval_stagnation_streak_max: int,
    retrieval_stagnation_triggered: bool,
    retrieval_stagnation_turn: int | None,
    evidence_digest_change_count: int,
    evidence_turns_total: int,
    evidence_turns_with_new_evidence: int,
    evidence_turns_without_new_evidence: int,
    evidence_pointer_count: int,
    failure_event_codes: list[str],
    submit_evidence_digest_at_last_failure: str | None,
    foundation_events: list[dict[str, Any]],
    foundation_event_types: dict[str, int],
    foundation_event_validation_errors: int,
    foundation_events_logged: int,
    evidence_pointer_labels: set[str],
    force_final_reason: str | None,
) -> None:
    """Write the final MCP-agent run summary into ``agent_result.metadata``."""

    submit_validator_accepted = bool(
        submit_answer_succeeded and not submit_forced_accept_on_budget_exhaustion
    )
    submit_completion_mode = (
        "grounded_submit"
        if submit_validator_accepted
        else "forced_terminal_accept"
        if submit_forced_accept_on_budget_exhaustion
        else "missing_required_submit"
        if required_submit_missing
        else "no_submit_required"
        if not requires_submit_answer
        else "unknown"
    )

    answer_present = bool(
        isinstance((submitted_answer_value or final_content), str)
        and (submitted_answer_value or final_content).strip()
    )
    required_submit_satisfied = not required_submit_missing
    grounded_completed = bool(answer_present and submit_validator_accepted)
    forced_terminal_accepted = bool(submit_forced_accept_on_budget_exhaustion)
    reliability_completed = bool(
        answer_present
        and (submit_answer_succeeded or final_finish_reason != "error")
    )

    run_completed = required_submit_satisfied and (
        submit_answer_succeeded or final_finish_reason != "error"
    )
    failure_summary = _summarize_failure_events(
        failure_event_codes=failure_event_codes,
        retrieval_no_hits_count=retrieval_no_hits_count,
        control_loop_suppressed_calls=control_loop_suppressed_calls,
        force_final_reason=force_final_reason,
        run_completed=run_completed,
    )
    finalization_summary = _summarize_finalization_attempts(
        finalization_fallback_attempts=finalization_fallback_attempts,
        finalization_primary_model=finalization_primary_model,
        forced_final_attempts=forced_final_attempts,
        forced_final_circuit_breaker_opened=forced_final_circuit_breaker_opened,
    )

    hard_bindings_spec = _hard_bindings_spec(available_bindings)
    full_bindings_spec = _full_bindings_spec(available_bindings)
    hard_bindings_hash = _hard_bindings_state_hash(available_bindings)
    full_bindings_hash = _full_bindings_state_hash(available_bindings)
    initial_hard_bindings_spec = _hard_bindings_spec(initial_binding_snapshot)
    initial_full_bindings_spec = _full_bindings_spec(initial_binding_snapshot)
    initial_hard_bindings_hash = _hard_bindings_state_hash(initial_binding_snapshot)
    initial_full_bindings_hash = _full_bindings_state_hash(initial_binding_snapshot)
    final_evidence_digest = _evidence_digest(evidence_pointer_labels)

    agent_result.metadata["total_cost"] = total_cost
    agent_result.metadata["requested_model"] = model
    agent_result.metadata["resolved_model"] = effective_model
    agent_result.metadata["attempted_models"] = attempted_models
    agent_result.metadata["sticky_fallback"] = sticky_fallback
    agent_result.metadata["max_turns"] = max_turns
    agent_result.metadata["max_tool_calls"] = max_tool_calls
    agent_result.metadata["forced_final_attempts"] = forced_final_attempts
    agent_result.metadata["forced_final_max_attempts"] = forced_final_max_attempts
    agent_result.metadata["forced_final_circuit_breaker_threshold"] = (
        forced_final_circuit_breaker_threshold
    )
    agent_result.metadata["forced_final_breaker_effective"] = (
        forced_final_breaker_effective
    )
    agent_result.metadata["force_submit_retry_on_max_tool_calls"] = (
        force_submit_retry_on_max_tool_calls
    )
    agent_result.metadata["accept_forced_answer_on_max_tool_calls"] = (
        accept_forced_answer_on_max_tool_calls
    )
    agent_result.metadata["forced_final_circuit_breaker_opened"] = (
        forced_final_circuit_breaker_opened
    )
    agent_result.metadata["finalization_primary_model"] = finalization_primary_model
    agent_result.metadata["finalization_fallback_models"] = list(
        finalization_fallback_models
    )
    agent_result.metadata["finalization_fallback_used"] = finalization_fallback_used
    agent_result.metadata["finalization_fallback_attempt_count"] = (
        finalization_summary.finalization_fallback_attempt_count
    )
    agent_result.metadata["finalization_fallback_usage_rate"] = (
        finalization_summary.finalization_fallback_usage_rate
    )
    agent_result.metadata["finalization_fallback_succeeded"] = (
        finalization_fallback_succeeded
    )
    agent_result.metadata["finalization_fallback_attempts"] = list(
        finalization_fallback_attempts
    )
    agent_result.metadata["finalization_attempt_counts_by_model"] = (
        finalization_summary.finalization_attempt_counts_by_model
    )
    agent_result.metadata["finalization_failure_counts_by_model"] = (
        finalization_summary.finalization_failure_counts_by_model
    )
    agent_result.metadata["finalization_success_counts_by_model"] = (
        finalization_summary.finalization_success_counts_by_model
    )
    agent_result.metadata["finalization_failure_code_counts"] = (
        finalization_summary.finalization_failure_code_counts
    )
    agent_result.metadata["provider_empty_attempt_counts_by_model"] = (
        finalization_summary.provider_empty_attempt_counts_by_model
    )
    agent_result.metadata["finalization_breaker_open_rate"] = (
        finalization_summary.finalization_breaker_open_rate
    )
    agent_result.metadata["finalization_breaker_open_by_model"] = (
        finalization_summary.finalization_breaker_open_by_model
    )
    agent_result.metadata["finalization_events"] = list(finalization_events)
    agent_result.metadata["tool_calls_used"] = len(agent_result.tool_calls)
    agent_result.metadata["budgeted_tool_calls_used"] = _count_budgeted_records(
        agent_result.tool_calls
    )
    agent_result.metadata["budget_exempt_tools"] = sorted(BUDGET_EXEMPT_TOOL_NAMES)
    agent_result.metadata["requires_submit_answer"] = requires_submit_answer
    agent_result.metadata["submit_answer_call_count"] = submit_answer_call_count
    agent_result.metadata["submit_answer_attempted"] = submit_answer_call_count > 0
    agent_result.metadata["submit_answer_succeeded"] = submit_answer_succeeded
    agent_result.metadata["submit_validator_accepted"] = submit_validator_accepted
    agent_result.metadata["required_submit_missing"] = required_submit_missing
    agent_result.metadata["required_submit_satisfied"] = required_submit_satisfied
    agent_result.metadata["submitted_answer_value"] = submitted_answer_value
    agent_result.metadata["submit_forced_retry_on_budget_exhaustion"] = (
        submit_forced_retry_on_budget_exhaustion
    )
    agent_result.metadata["submit_forced_accept_on_budget_exhaustion"] = (
        submit_forced_accept_on_budget_exhaustion
    )
    agent_result.metadata["submit_completion_mode"] = submit_completion_mode
    agent_result.metadata["answer_present"] = answer_present
    agent_result.metadata["grounded_completed"] = grounded_completed
    agent_result.metadata["forced_terminal_accepted"] = forced_terminal_accepted
    agent_result.metadata["reliability_completed"] = reliability_completed
    agent_result.metadata["run_completed"] = run_completed
    agent_result.metadata["require_tool_reasoning"] = require_tool_reasoning
    agent_result.metadata["rejected_missing_reasoning_calls"] = (
        rejected_missing_reasoning_calls
    )
    agent_result.metadata["control_loop_suppressed_calls"] = (
        control_loop_suppressed_calls
    )
    agent_result.metadata["submit_validation_reason_counts"] = dict(
        sorted(submit_validation_reason_counts.items())
    )
    agent_result.metadata["tool_call_turns_total"] = tool_call_turns_total
    agent_result.metadata["tool_call_empty_text_turns"] = tool_call_empty_text_turns
    agent_result.metadata["responses_tool_call_empty_text_turns"] = (
        responses_tool_call_empty_text_turns
    )
    agent_result.metadata["tool_call_empty_text_turn_ratio"] = (
        (tool_call_empty_text_turns / tool_call_turns_total)
        if tool_call_turns_total
        else 0.0
    )
    agent_result.metadata["tool_arg_coercions"] = tool_arg_coercions
    agent_result.metadata["tool_arg_coercion_calls"] = tool_arg_coercion_calls
    agent_result.metadata["tool_arg_validation_rejections"] = (
        tool_arg_validation_rejections
    )
    agent_result.metadata["tool_result_keep_recent"] = tool_result_keep_recent
    agent_result.metadata["tool_result_context_preview_chars"] = (
        tool_result_context_preview_chars
    )
    agent_result.metadata["active_artifact_context_enabled"] = (
        active_artifact_context_enabled
    )
    agent_result.metadata["active_artifact_context_max_handles"] = (
        active_artifact_context_max_handles
    )
    agent_result.metadata["active_artifact_context_max_chars"] = (
        active_artifact_context_max_chars
    )
    agent_result.metadata["active_artifact_context_updates"] = artifact_context_updates
    agent_result.metadata["active_artifact_context_chars"] = artifact_context_chars
    agent_result.metadata["context_tool_result_clearings"] = (
        context_tool_result_clearings
    )
    agent_result.metadata["context_tool_results_cleared"] = (
        context_tool_results_cleared
    )
    agent_result.metadata["context_tool_result_cleared_chars"] = (
        context_tool_result_cleared_chars
    )
    agent_result.metadata["tool_result_metadata_tracked"] = len(tool_result_metadata_by_id)
    agent_result.metadata["context_compactions"] = context_compactions
    agent_result.metadata["context_compacted_messages"] = context_compacted_messages
    agent_result.metadata["context_compacted_chars"] = context_compacted_chars
    agent_result.metadata["enforce_tool_contracts"] = enforce_tool_contracts
    agent_result.metadata["progressive_tool_disclosure"] = (
        progressive_tool_disclosure
    )
    agent_result.metadata["suppress_control_loop_calls"] = suppress_control_loop_calls
    agent_result.metadata["tool_gate_rejections"] = gate_rejected_calls
    agent_result.metadata["tool_gate_violation_events"] = gate_violation_events
    agent_result.metadata["tool_contract_rejections"] = contract_rejected_calls
    agent_result.metadata["tool_contract_violation_events"] = (
        contract_violation_events
    )
    agent_result.metadata["tool_contracts_declared"] = sorted(
        normalized_tool_contracts.keys()
    )
    agent_result.metadata["runtime_tool_names"] = [runtime_artifact_tool_name]
    agent_result.metadata["runtime_artifact_registry_size"] = len(
        runtime_artifact_registry_by_id
    )
    agent_result.metadata["runtime_artifact_registry_ids"] = sorted(
        runtime_artifact_registry_by_id.keys()
    )
    agent_result.metadata["handle_input_resolution_count"] = (
        handle_input_resolution_count
    )
    agent_result.metadata["handle_input_resolved_artifact_count"] = (
        handle_input_resolved_artifact_count
    )
    agent_result.metadata["initial_artifacts"] = initial_artifact_snapshot
    agent_result.metadata["available_artifacts_final"] = sorted(available_artifacts)
    agent_result.metadata["initial_capabilities"] = initial_capability_snapshot
    agent_result.metadata["available_capabilities_final"] = _capability_state_snapshot(
        available_capabilities
    )
    agent_result.metadata["artifact_timeline"] = artifact_timeline
    agent_result.metadata["initial_bindings"] = initial_binding_snapshot
    agent_result.metadata["available_bindings_final"] = dict(available_bindings)
    agent_result.metadata["initial_hard_bindings_spec"] = initial_hard_bindings_spec
    agent_result.metadata["initial_full_bindings_spec"] = initial_full_bindings_spec
    agent_result.metadata["initial_hard_bindings_hash"] = initial_hard_bindings_hash
    agent_result.metadata["initial_full_bindings_hash"] = initial_full_bindings_hash
    agent_result.metadata["hard_bindings_spec"] = hard_bindings_spec
    agent_result.metadata["full_bindings_spec"] = full_bindings_spec
    agent_result.metadata["hard_bindings_hash"] = hard_bindings_hash
    agent_result.metadata["full_bindings_hash"] = full_bindings_hash
    agent_result.metadata["run_config_spec"] = run_config_spec
    agent_result.metadata["run_config_hash"] = run_config_hash
    agent_result.metadata["adoption_profile_requested"] = (
        adoption_assessment.requested_profile
    )
    agent_result.metadata["adoption_profile_effective"] = (
        adoption_assessment.effective_profile
    )
    agent_result.metadata["adoption_profile_enforce"] = adoption_assessment.enforce
    agent_result.metadata["adoption_profile_satisfied"] = (
        adoption_assessment.satisfied
    )
    agent_result.metadata["adoption_profile_violations"] = list(
        adoption_assessment.violations
    )
    agent_result.metadata["tool_disclosure_turns"] = tool_disclosure_turns
    agent_result.metadata["tool_disclosure_hidden_total"] = (
        tool_disclosure_hidden_total
    )
    agent_result.metadata["tool_disclosure_unavailable_msgs"] = (
        tool_disclosure_unavailable_msgs
    )
    agent_result.metadata["tool_disclosure_unavailable_reason_chars"] = (
        tool_disclosure_unavailable_reason_chars
    )
    agent_result.metadata["tool_disclosure_unavailable_reason_tokens_est"] = (
        tool_disclosure_unavailable_reason_tokens_est
    )
    agent_result.metadata["tool_disclosure_repair_suggestions"] = (
        tool_disclosure_repair_suggestions
    )
    agent_result.metadata["lane_closure_analysis"] = lane_closure_analysis
    agent_result.metadata["no_legal_noncontrol_turns"] = no_legal_noncontrol_turns
    agent_result.metadata["deficit_no_progress_streak_max"] = (
        max_deficit_no_progress_streak
    )
    agent_result.metadata["deficit_no_progress_nudges"] = deficit_no_progress_nudges
    agent_result.metadata["max_zero_exec_tool_turn_streak"] = (
        max_zero_exec_tool_turn_streak
    )
    agent_result.metadata["retrieval_no_hits_count"] = retrieval_no_hits_count
    agent_result.metadata["retrieval_stagnation_turns"] = retrieval_stagnation_turns
    agent_result.metadata["retrieval_stagnation_action"] = (
        retrieval_stagnation_action
    )
    agent_result.metadata["retrieval_stagnation_streak"] = (
        retrieval_stagnation_streak
    )
    agent_result.metadata["retrieval_stagnation_streak_max"] = (
        retrieval_stagnation_streak_max
    )
    agent_result.metadata["retrieval_stagnation_triggered"] = (
        retrieval_stagnation_triggered
    )
    agent_result.metadata["retrieval_stagnation_turn"] = retrieval_stagnation_turn
    agent_result.metadata["evidence_digest_change_count"] = (
        evidence_digest_change_count
    )
    agent_result.metadata["evidence_turns_total"] = evidence_turns_total
    agent_result.metadata["evidence_turns_with_new_evidence"] = (
        evidence_turns_with_new_evidence
    )
    agent_result.metadata["evidence_turns_without_new_evidence"] = (
        evidence_turns_without_new_evidence
    )
    agent_result.metadata["evidence_pointer_count"] = evidence_pointer_count
    agent_result.metadata["evidence_digest_basis"] = "canonical_evidence_pointers"
    agent_result.metadata["failure_event_codes"] = list(failure_event_codes)
    agent_result.metadata["failure_event_code_counts"] = (
        failure_summary.failure_event_code_counts
    )
    agent_result.metadata["failure_event_class_counts"] = (
        failure_summary.failure_event_class_counts
    )
    agent_result.metadata["provider_failure_event_code_counts"] = (
        failure_summary.provider_failure_event_code_counts
    )
    agent_result.metadata["provider_failure_event_total"] = (
        failure_summary.provider_failure_event_total
    )
    agent_result.metadata["provider_caused_incompletion"] = (
        failure_summary.provider_caused_incompletion
    )
    agent_result.metadata["primary_failure_class"] = (
        failure_summary.primary_failure_class
    )
    agent_result.metadata["secondary_failure_classes"] = (
        failure_summary.secondary_failure_classes
    )
    agent_result.metadata["first_terminal_failure_event_code"] = (
        failure_summary.first_terminal_failure_event_code
    )
    agent_result.metadata["failure_priority_order"] = list(_PRIMARY_FAILURE_PRIORITY)
    agent_result.metadata["terminal_failure_event_codes"] = sorted(
        _TERMINAL_FAILURE_EVENT_CODES
    )
    agent_result.metadata["evidence_digest"] = final_evidence_digest
    agent_result.metadata["submit_evidence_digest_at_last_failure"] = (
        submit_evidence_digest_at_last_failure
    )
    agent_result.metadata["foundation_event_count"] = len(foundation_events)
    agent_result.metadata["foundation_event_types"] = dict(foundation_event_types)
    agent_result.metadata["foundation_event_validation_errors"] = (
        foundation_event_validation_errors
    )
    agent_result.metadata["foundation_events_logged"] = foundation_events_logged
    agent_result.metadata["foundation_events"] = foundation_events

    if tool_call_turns_total > 0:
        agent_result.warnings.append(
            "METRIC: tool_call_empty_text_turns="
            f"{tool_call_empty_text_turns}/{tool_call_turns_total}, "
            f"responses_tool_call_empty_text_turns={responses_tool_call_empty_text_turns}"
        )
    if tool_arg_coercions or tool_arg_validation_rejections:
        agent_result.warnings.append(
            "METRIC: tool_arg_coercions="
            f"{tool_arg_coercions} across {tool_arg_coercion_calls} calls, "
            f"tool_arg_validation_rejections={tool_arg_validation_rejections}"
        )
    if evidence_turns_total > 0:
        agent_result.warnings.append(
            "METRIC: evidence_turns="
            f"{evidence_turns_total}, new_evidence_turns={evidence_turns_with_new_evidence}, "
            f"stagnant_evidence_turns={evidence_turns_without_new_evidence}, "
            f"retrieval_stagnation_streak_max={retrieval_stagnation_streak_max}, "
            f"evidence_pointer_count={evidence_pointer_count}"
        )
    logger.info(
        "Agent loop metrics: tool_call_turns=%d empty_text_tool_call_turns=%d responses_empty_text_tool_call_turns=%d "
        "tool_arg_coercions=%d tool_arg_validation_rejections=%d",
        tool_call_turns_total,
        tool_call_empty_text_turns,
        responses_tool_call_empty_text_turns,
        tool_arg_coercions,
        tool_arg_validation_rejections,
    )
