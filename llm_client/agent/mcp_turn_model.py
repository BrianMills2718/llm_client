"""Pre-call model-stage helpers for MCP agent turns.

This module owns the dense middle of each MCP turn before tool execution:
context compaction, progressive disclosure filtering, the model call itself,
and the response handling that decides whether the turn should continue,
terminate, or proceed into tool execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from llm_client.foundation import new_event_id, now_iso, sha256_json, sha256_text
from llm_client.agent.mcp_context import (
    _clear_old_tool_results_for_context,
    _compact_tool_history_for_context,
)
from llm_client.agent.mcp_contracts import (
    _capability_state_snapshot,
    _deficit_labels_from_hidden_entries,
    _disclosure_message,
    _filter_tools_for_disclosure,
    _is_control_tool_name,
    _upsert_active_artifact_context_message,
)
from llm_client.agent.mcp_finalization import _provider_failure_classification
from llm_client.agent.mcp_tools import (
    BUDGET_EXEMPT_TOOL_NAMES,
    _autofill_tool_reasoning,
    _count_budgeted_records,
    _count_budgeted_tool_calls,
    _extract_tool_call_args,
    _is_budget_exempt_tool,
    _normalize_tool_call_name_inplace,
    _trim_tool_calls_to_budget,
)
from llm_client.tools.tool_runtime_common import MCPAgentResult, TOOL_REASONING_FIELD, extract_usage_counts

logger = logging.getLogger(__name__)


@dataclass
class AgentTurnModelStageResult:
    """Outcome of the per-turn model stage before tool execution."""

    artifact_context_message_index: int | None
    context_tool_result_clearings_delta: int
    context_tool_results_cleared_delta: int
    context_tool_result_cleared_chars_delta: int
    context_compactions_delta: int
    context_compacted_messages_delta: int
    context_compacted_chars_delta: int
    artifact_context_updates_delta: int
    artifact_context_chars_delta: int
    tool_disclosure_turns_delta: int
    tool_disclosure_hidden_total_delta: int
    tool_disclosure_unavailable_msgs_delta: int
    tool_disclosure_unavailable_reason_chars_delta: int
    tool_disclosure_unavailable_reason_tokens_est_delta: int
    tool_disclosure_repair_suggestions_delta: int
    no_legal_noncontrol_turns_delta: int
    deficit_no_progress_streak: int
    max_deficit_no_progress_streak: int
    deficit_no_progress_nudges_delta: int
    deficit_no_progress_last_nudged: int
    current_turn_deficit_digest: str | None
    disclosure_repair_hints: list[str]
    effective_model: str
    sticky_fallback: bool
    total_cost_delta: float
    total_input_tokens_delta: int
    total_output_tokens_delta: int
    total_cached_tokens_delta: int
    total_cache_creation_tokens_delta: int
    final_content: str
    final_finish_reason: str
    failure_event_codes: list[str]
    plain_text_no_tool_turn_streak: int
    tool_calls_this_turn: list[dict[str, Any]]
    tool_call_turns_total_delta: int
    tool_call_empty_text_turns_delta: int
    responses_tool_call_empty_text_turns_delta: int
    autofilled_tool_reasoning_calls_delta: int
    autofilled_tool_reasoning_by_tool_delta: dict[str, int]
    should_continue_turn: bool
    should_break_loop: bool


async def _run_turn_model_stage(
    *,
    turn: int,
    model: str,
    effective_model: str,
    attempted_models: list[str],
    sticky_fallback: bool,
    messages: list[dict[str, Any]],
    openai_tools: list[dict[str, Any]],
    runtime_artifact_registry_by_id: dict[str, dict[str, Any]],
    runtime_artifact_read_tool: dict[str, Any],
    tool_result_metadata_by_id: dict[str, dict[str, Any]],
    agent_result: MCPAgentResult,
    max_turns: int,
    max_tool_calls: int | None,
    max_message_chars: int,
    tool_result_keep_recent: int,
    tool_result_context_preview_chars: int,
    progressive_tool_disclosure: bool,
    normalized_tool_contracts: dict[str, dict[str, Any]],
    available_artifacts: set[str],
    available_capabilities: dict[str, set[tuple[str | None, str | None, str | None]]],
    available_bindings: dict[str, Any],
    prev_turn_deficit_digest: str | None,
    prev_turn_had_evidence_tools: bool,
    deficit_no_progress_streak: int,
    max_deficit_no_progress_streak: int,
    deficit_no_progress_last_nudged: int,
    plain_text_no_tool_turn_streak: int,
    require_tool_reasoning: bool,
    timeout: float | None,
    kwargs: dict[str, Any],
    foundation_run_id: str,
    foundation_session_id: str,
    foundation_actor_id: str,
    emit_foundation_event: Any,
    inner_acall_llm: Any,
    is_responses_api_raw_response: Any,
    artifact_context_message_index: int | None,
    active_artifact_context_enabled: bool,
    active_artifact_context_max_handles: int,
    active_artifact_context_max_chars: int,
    requires_submit_answer: bool,
    submit_answer_succeeded: bool,
    current_tool_call_count: int,
    turn_warning_threshold: int,
    disclosure_token_chars: int,
    event_code_no_legal_noncontrol_tools: str,
    event_code_provider_empty: str,
) -> AgentTurnModelStageResult:
    """Prepare the current tool surface, dispatch the model call, and normalize its response."""

    context_tool_result_clearings_delta = 0
    context_tool_results_cleared_delta = 0
    context_tool_result_cleared_chars_delta = 0
    context_compactions_delta = 0
    context_compacted_messages_delta = 0
    context_compacted_chars_delta = 0
    artifact_context_updates_delta = 0
    artifact_context_chars_delta = 0
    tool_disclosure_turns_delta = 0
    tool_disclosure_hidden_total_delta = 0
    tool_disclosure_unavailable_msgs_delta = 0
    tool_disclosure_unavailable_reason_chars_delta = 0
    tool_disclosure_unavailable_reason_tokens_est_delta = 0
    tool_disclosure_repair_suggestions_delta = 0
    no_legal_noncontrol_turns_delta = 0
    deficit_no_progress_nudges_delta = 0
    total_cost_delta = 0.0
    total_input_tokens_delta = 0
    total_output_tokens_delta = 0
    total_cached_tokens_delta = 0
    total_cache_creation_tokens_delta = 0
    final_content = ""
    final_finish_reason = "stop"
    failure_event_codes: list[str] = []
    disclosure_repair_hints: list[str] = []
    current_turn_deficit_digest: str | None = None
    tool_calls_this_turn: list[dict[str, Any]] = []
    tool_call_turns_total_delta = 0
    tool_call_empty_text_turns_delta = 0
    responses_tool_call_empty_text_turns_delta = 0
    autofilled_tool_reasoning_calls_delta = 0
    autofilled_tool_reasoning_by_tool_delta: dict[str, int] = {}

    (
        artifact_context_message_index,
        artifact_context_content,
        artifact_context_changed,
    ) = _upsert_active_artifact_context_message(
        messages,
        available_artifacts=available_artifacts,
        available_capabilities=available_capabilities,
        tool_result_metadata_by_id=tool_result_metadata_by_id,
        enabled=active_artifact_context_enabled,
        max_handles=active_artifact_context_max_handles,
        max_chars=active_artifact_context_max_chars,
        existing_index=artifact_context_message_index,
    )
    # The caller pre-computes runtime policy knobs; this helper only mirrors the existing
    # turn behavior and therefore accepts the same values via the artifact-context message.
    if artifact_context_changed and artifact_context_content:
        artifact_context_updates_delta += 1
        artifact_context_chars_delta += len(artifact_context_content)
        agent_result.conversation_trace.append(
            {
                "role": "user",
                "content": artifact_context_content,
                "synthetic": "active_artifact_context",
            }
        )

    cleared_count, cleared_chars = _clear_old_tool_results_for_context(
        messages,
        keep_recent=tool_result_keep_recent,
        preview_chars=tool_result_context_preview_chars,
        tool_result_metadata_by_id=tool_result_metadata_by_id,
    )
    if cleared_count:
        context_tool_result_clearings_delta = 1
        context_tool_results_cleared_delta = cleared_count
        context_tool_result_cleared_chars_delta = cleared_chars
        warning = (
            "CONTEXT_TOOL_RESULT_CLEARING: replaced "
            f"{cleared_count} older tool payload(s) with compact stubs "
            f"(saved ~{cleared_chars} chars; keep_recent={tool_result_keep_recent})."
        )
        agent_result.warnings.append(warning)
        logger.warning(warning)

    compacted_count, compacted_chars, current_chars = _compact_tool_history_for_context(
        messages,
        max_message_chars,
    )
    if compacted_count:
        context_compactions_delta = 1
        context_compacted_messages_delta = compacted_count
        context_compacted_chars_delta = compacted_chars
        warning = (
            "CONTEXT_COMPACTION: compacted "
            f"{compacted_count} tool message(s), saved ~{compacted_chars} chars "
            f"(history ~{current_chars} chars)"
        )
        agent_result.warnings.append(warning)
        logger.warning(warning)

    current_tool_surface = list(openai_tools)
    if runtime_artifact_registry_by_id:
        current_tool_surface.append(runtime_artifact_read_tool)

    disclosed_tools = current_tool_surface
    hidden_disclosure: list[dict[str, Any]] = []
    hidden_disclosure_total = 0
    current_turn_deficit_labels: list[str] = []
    if progressive_tool_disclosure and normalized_tool_contracts:
        disclosed_tools, hidden_disclosure, hidden_disclosure_total = (
            _filter_tools_for_disclosure(
                openai_tools=current_tool_surface,
                normalized_tool_contracts=normalized_tool_contracts,
                available_artifacts=available_artifacts,
                available_capabilities=available_capabilities,
                available_bindings=available_bindings,
            )
        )
        if hidden_disclosure:
            tool_disclosure_turns_delta = 1
            tool_disclosure_hidden_total_delta = hidden_disclosure_total
            tool_disclosure_repair_suggestions_delta = sum(
                len(entry.get("repair_tools") or [])
                for entry in hidden_disclosure
                if isinstance(entry, dict)
            )
            seen_repair_hints: set[str] = set()
            for entry in hidden_disclosure:
                if not isinstance(entry, dict):
                    continue
                for candidate in (entry.get("repair_tools") or []):
                    name = str(candidate or "").strip()
                    if not name or name in seen_repair_hints:
                        continue
                    seen_repair_hints.add(name)
                    disclosure_repair_hints.append(name)
            current_turn_deficit_labels = _deficit_labels_from_hidden_entries(
                hidden_disclosure
            )
            if current_turn_deficit_labels:
                current_turn_deficit_digest = sha256_json(
                    current_turn_deficit_labels
                ).replace("sha256:", "")
            hidden_names = [str(h.get("tool", "")) for h in hidden_disclosure]
            disclosure_reason = _disclosure_message(hidden_disclosure)
            disclosure_msg = {
                "role": "user",
                "content": (
                    "[SYSTEM: Currently unavailable tools with missing requirements: "
                    f"{disclosure_reason}]"
                ),
            }
            disclosure_chars = len(str(disclosure_msg.get("content", "")))
            tool_disclosure_unavailable_msgs_delta = 1
            tool_disclosure_unavailable_reason_chars_delta = disclosure_chars
            tool_disclosure_unavailable_reason_tokens_est_delta = max(
                1,
                disclosure_chars // disclosure_token_chars,
            )
            messages.append(disclosure_msg)
            agent_result.conversation_trace.append(disclosure_msg)
            logger.info(
                "TOOL_DISCLOSURE turn=%d exposed=%d/%d hidden=%s available_artifacts=%s available_capabilities=%s",
                turn + 1,
                len(disclosed_tools),
                len(current_tool_surface),
                hidden_names,
                sorted(available_artifacts),
                _capability_state_snapshot(available_capabilities),
            )
            agent_result.warnings.append(
                "TOOL_DISCLOSURE: hidden currently incompatible tools on turn "
                f"{turn + 1}: {', '.join(hidden_names)}"
            )

    if prev_turn_had_evidence_tools and current_turn_deficit_digest:
        if prev_turn_deficit_digest == current_turn_deficit_digest:
            deficit_no_progress_streak += 1
        else:
            deficit_no_progress_streak = 0
    elif not prev_turn_had_evidence_tools:
        deficit_no_progress_streak = 0

    if deficit_no_progress_streak > max_deficit_no_progress_streak:
        max_deficit_no_progress_streak = deficit_no_progress_streak

    if (
        deficit_no_progress_streak >= 2
        and deficit_no_progress_streak > deficit_no_progress_last_nudged
        and current_turn_deficit_labels
    ):
        deficit_no_progress_nudges_delta = 1
        deficit_no_progress_last_nudged = deficit_no_progress_streak
        suggested = [
            name
            for name in disclosure_repair_hints
            if name and not _is_budget_exempt_tool(name)
        ][:3]
        suggested_hint = f" Try: {', '.join(suggested)}." if suggested else ""
        deficit_msg = {
            "role": "user",
            "content": (
                "[SYSTEM: Recent evidence calls did not reduce unresolved artifacts. "
                "Still missing: "
                + ", ".join(current_turn_deficit_labels[:4])
                + ". Choose the next call to close one missing requirement."
                + suggested_hint
                + "]"
            ),
        }
        messages.append(deficit_msg)
        agent_result.conversation_trace.append(deficit_msg)

    if progressive_tool_disclosure and normalized_tool_contracts:
        noncontrol_disclosed = [
            tool
            for tool in disclosed_tools
            if isinstance(tool, dict)
            and not _is_control_tool_name(
                str(tool.get("function", {}).get("name", "")).strip(),
                normalized_tool_contracts,
            )
        ]
        if not noncontrol_disclosed:
            no_legal_noncontrol_turns_delta = 1
            warning = (
                "TOOL_DISCLOSURE: no legal non-control tools available this turn. "
                "Use conversion/planning/control tools to unlock capabilities."
            )
            agent_result.warnings.append(warning)
            logger.warning(warning)
            failure_event_codes.append(event_code_no_legal_noncontrol_tools)
            emit_foundation_event(
                {
                    "event_id": new_event_id(),
                    "event_type": "ToolFailed",
                    "timestamp": now_iso(),
                    "run_id": foundation_run_id,
                    "session_id": foundation_session_id,
                    "actor_id": foundation_actor_id,
                    "operation": {"name": "__disclosure__", "version": None},
                    "inputs": {
                        "artifact_ids": sorted(available_artifacts),
                        "params": {
                            "hidden_tools": [h.get("tool") for h in hidden_disclosure],
                            "hidden_total": hidden_disclosure_total,
                        },
                        "bindings": dict(available_bindings),
                    },
                    "outputs": {"artifact_ids": [], "payload_hashes": []},
                    "failure": {
                        "error_code": event_code_no_legal_noncontrol_tools,
                        "category": "validation",
                        "phase": "input_validation",
                        "retryable": True,
                        "tool_name": "__disclosure__",
                        "user_message": warning,
                        "debug_ref": None,
                    },
                }
            )

    try:
        result = await inner_acall_llm(
            effective_model,
            messages,
            timeout=timeout,
            tools=disclosed_tools,
            **kwargs,
        )
    except Exception as exc:
        error_text = str(exc).strip() or f"{type(exc).__name__}"
        (
            is_provider_failure,
            event_code,
            provider_classification,
            retryable_provider,
        ) = _provider_failure_classification(exc, error_text)
        failure_event_codes.append(event_code or "TOOL_EXECUTION_RUNTIME_ERROR")
        warning = (
            "AGENT_LLM_CALL_FAILED: turn="
            f"{turn + 1} model={effective_model} error={error_text}"
        )
        agent_result.warnings.append(warning)
        logger.warning(warning)
        emit_foundation_event(
            {
                "event_id": new_event_id(),
                "event_type": "ToolFailed",
                "timestamp": now_iso(),
                "run_id": foundation_run_id,
                "session_id": foundation_session_id,
                "actor_id": foundation_actor_id,
                "operation": {"name": "_inner_acall_llm", "version": None},
                "inputs": {
                    "artifact_ids": sorted(available_artifacts),
                    "params": {
                        "turn": turn + 1,
                        "error": error_text,
                        "provider_classification": provider_classification or "",
                        "provider_subevent": (
                            "first_turn" if is_provider_failure and turn == 0 else "turn"
                            if is_provider_failure
                            else ""
                        ),
                    },
                    "bindings": dict(available_bindings),
                },
                "outputs": {"artifact_ids": [], "payload_hashes": []},
                "failure": {
                    "error_code": event_code or "TOOL_EXECUTION_RUNTIME_ERROR",
                    "category": "provider" if is_provider_failure else "execution",
                    "phase": "execution",
                    "retryable": bool(retryable_provider),
                    "tool_name": "_inner_acall_llm",
                    "user_message": error_text,
                    "debug_ref": None,
                },
            }
        )
        return AgentTurnModelStageResult(
            artifact_context_message_index=artifact_context_message_index,
            context_tool_result_clearings_delta=context_tool_result_clearings_delta,
            context_tool_results_cleared_delta=context_tool_results_cleared_delta,
            context_tool_result_cleared_chars_delta=context_tool_result_cleared_chars_delta,
            context_compactions_delta=context_compactions_delta,
            context_compacted_messages_delta=context_compacted_messages_delta,
            context_compacted_chars_delta=context_compacted_chars_delta,
            artifact_context_updates_delta=artifact_context_updates_delta,
            artifact_context_chars_delta=artifact_context_chars_delta,
            tool_disclosure_turns_delta=tool_disclosure_turns_delta,
            tool_disclosure_hidden_total_delta=tool_disclosure_hidden_total_delta,
            tool_disclosure_unavailable_msgs_delta=tool_disclosure_unavailable_msgs_delta,
            tool_disclosure_unavailable_reason_chars_delta=tool_disclosure_unavailable_reason_chars_delta,
            tool_disclosure_unavailable_reason_tokens_est_delta=tool_disclosure_unavailable_reason_tokens_est_delta,
            tool_disclosure_repair_suggestions_delta=tool_disclosure_repair_suggestions_delta,
            no_legal_noncontrol_turns_delta=no_legal_noncontrol_turns_delta,
            deficit_no_progress_streak=deficit_no_progress_streak,
            max_deficit_no_progress_streak=max_deficit_no_progress_streak,
            deficit_no_progress_nudges_delta=deficit_no_progress_nudges_delta,
            deficit_no_progress_last_nudged=deficit_no_progress_last_nudged,
            current_turn_deficit_digest=current_turn_deficit_digest,
            disclosure_repair_hints=disclosure_repair_hints,
            effective_model=effective_model,
            sticky_fallback=sticky_fallback,
            total_cost_delta=total_cost_delta,
            total_input_tokens_delta=total_input_tokens_delta,
            total_output_tokens_delta=total_output_tokens_delta,
            total_cached_tokens_delta=total_cached_tokens_delta,
            total_cache_creation_tokens_delta=total_cache_creation_tokens_delta,
            final_content=error_text,
            final_finish_reason="error",
            failure_event_codes=failure_event_codes,
            plain_text_no_tool_turn_streak=plain_text_no_tool_turn_streak,
            tool_calls_this_turn=tool_calls_this_turn,
            tool_call_turns_total_delta=tool_call_turns_total_delta,
            tool_call_empty_text_turns_delta=tool_call_empty_text_turns_delta,
            responses_tool_call_empty_text_turns_delta=responses_tool_call_empty_text_turns_delta,
            autofilled_tool_reasoning_calls_delta=autofilled_tool_reasoning_calls_delta,
            autofilled_tool_reasoning_by_tool_delta=autofilled_tool_reasoning_by_tool_delta,
            should_continue_turn=False,
            should_break_loop=True,
        )

    result_model = str(result.model).strip() or effective_model
    agent_result.models_used.add(result_model)
    if result_model and result_model not in attempted_models:
        attempted_models.append(result_model)
    if result.warnings:
        agent_result.warnings.extend(result.warnings)
    if result_model != effective_model:
        agent_result.warnings.append(
            f"STICKY_FALLBACK: {effective_model} failed, "
            f"using {result_model} for remaining turns"
        )
        effective_model = result_model
        sticky_fallback = True

    emit_foundation_event(
        {
            "event_id": new_event_id(),
            "event_type": "LLMCalled",
            "timestamp": now_iso(),
            "run_id": foundation_run_id,
            "session_id": foundation_session_id,
            "actor_id": foundation_actor_id,
            "operation": {"name": "_inner_acall_llm", "version": None},
            "inputs": {
                "artifact_ids": sorted(available_artifacts),
                "params": {
                    "turn": turn + 1,
                    "model": effective_model,
                    "capabilities_sha256": sha256_json(
                        _capability_state_snapshot(available_capabilities)
                    ),
                },
                "bindings": dict(available_bindings),
            },
            "outputs": {
                "artifact_ids": [],
                "payload_hashes": [sha256_text(result.content or "")],
            },
            "llm": {
                "model_id": result_model,
                "content_persisted": "hash_only",
                "prompt_sha256": sha256_json(messages),
                "response_sha256": sha256_text(result.content or ""),
                "token_usage": dict(result.usage or {}),
                "cost_usd": float(result.cost or 0.0),
            },
        }
    )

    inp, out, cached, cache_create = extract_usage_counts(result.usage or {})
    total_input_tokens_delta = inp
    total_output_tokens_delta = out
    total_cached_tokens_delta = cached
    total_cache_creation_tokens_delta = cache_create
    total_cost_delta = float(result.cost or 0.0)

    if not result.tool_calls:
        remaining_turns = max_turns - (turn + 1)
        if requires_submit_answer and not submit_answer_succeeded and remaining_turns > 0:
            if result.content:
                draft_msg = {"role": "assistant", "content": result.content}
                messages.append(draft_msg)
                agent_result.conversation_trace.append(draft_msg)

            near_end = remaining_turns <= turn_warning_threshold
            if plain_text_no_tool_turn_streak == 0 and not near_end:
                continue_nudge = {
                    "role": "user",
                    "content": (
                        "[SYSTEM: Do NOT finalize yet. Continue with tools to resolve remaining TODO atoms. "
                        "Review your TODO status, gather missing evidence, then submit.]"
                    ),
                }
                messages.append(continue_nudge)
                agent_result.conversation_trace.append(continue_nudge)
                plain_text_no_tool_turn_streak += 1
                logger.warning(
                    "Agent loop: model returned plain text without tool calls on turn %d/%d; nudging continued tool use before submission.",
                    turn + 1,
                    max_turns,
                )
                return AgentTurnModelStageResult(
                    artifact_context_message_index=artifact_context_message_index,
                    context_tool_result_clearings_delta=context_tool_result_clearings_delta,
                    context_tool_results_cleared_delta=context_tool_results_cleared_delta,
                    context_tool_result_cleared_chars_delta=context_tool_result_cleared_chars_delta,
                    context_compactions_delta=context_compactions_delta,
                    context_compacted_messages_delta=context_compacted_messages_delta,
                    context_compacted_chars_delta=context_compacted_chars_delta,
                    artifact_context_updates_delta=artifact_context_updates_delta,
                    artifact_context_chars_delta=artifact_context_chars_delta,
                    tool_disclosure_turns_delta=tool_disclosure_turns_delta,
                    tool_disclosure_hidden_total_delta=tool_disclosure_hidden_total_delta,
                    tool_disclosure_unavailable_msgs_delta=tool_disclosure_unavailable_msgs_delta,
                    tool_disclosure_unavailable_reason_chars_delta=tool_disclosure_unavailable_reason_chars_delta,
                    tool_disclosure_unavailable_reason_tokens_est_delta=tool_disclosure_unavailable_reason_tokens_est_delta,
                    tool_disclosure_repair_suggestions_delta=tool_disclosure_repair_suggestions_delta,
                    no_legal_noncontrol_turns_delta=no_legal_noncontrol_turns_delta,
                    deficit_no_progress_streak=deficit_no_progress_streak,
                    max_deficit_no_progress_streak=max_deficit_no_progress_streak,
                    deficit_no_progress_nudges_delta=deficit_no_progress_nudges_delta,
                    deficit_no_progress_last_nudged=deficit_no_progress_last_nudged,
                    current_turn_deficit_digest=current_turn_deficit_digest,
                    disclosure_repair_hints=disclosure_repair_hints,
                    effective_model=effective_model,
                    sticky_fallback=sticky_fallback,
                    total_cost_delta=total_cost_delta,
                    total_input_tokens_delta=total_input_tokens_delta,
                    total_output_tokens_delta=total_output_tokens_delta,
                    total_cached_tokens_delta=total_cached_tokens_delta,
                    total_cache_creation_tokens_delta=total_cache_creation_tokens_delta,
                    final_content=final_content,
                    final_finish_reason=final_finish_reason,
                    failure_event_codes=failure_event_codes,
                    plain_text_no_tool_turn_streak=plain_text_no_tool_turn_streak,
                    tool_calls_this_turn=tool_calls_this_turn,
                    tool_call_turns_total_delta=tool_call_turns_total_delta,
                    tool_call_empty_text_turns_delta=tool_call_empty_text_turns_delta,
                    responses_tool_call_empty_text_turns_delta=responses_tool_call_empty_text_turns_delta,
                    autofilled_tool_reasoning_calls_delta=autofilled_tool_reasoning_calls_delta,
                    autofilled_tool_reasoning_by_tool_delta=autofilled_tool_reasoning_by_tool_delta,
                    should_continue_turn=True,
                    should_break_loop=False,
                )

            submit_nudge = {
                "role": "user",
                "content": (
                    "[SYSTEM: Do NOT answer in plain text. You MUST call "
                    "submit_answer(reasoning, answer) now. Use a short factual "
                    "answer (<=8 words). No additional searches.]"
                ),
            }
            messages.append(submit_nudge)
            agent_result.conversation_trace.append(submit_nudge)
            plain_text_no_tool_turn_streak += 1
            logger.warning(
                "Agent loop: model returned plain text without submit_answer on turn %d/%d; nudging explicit submission.",
                turn + 1,
                max_turns,
            )
            return AgentTurnModelStageResult(
                artifact_context_message_index=artifact_context_message_index,
                context_tool_result_clearings_delta=context_tool_result_clearings_delta,
                context_tool_results_cleared_delta=context_tool_results_cleared_delta,
                context_tool_result_cleared_chars_delta=context_tool_result_cleared_chars_delta,
                context_compactions_delta=context_compactions_delta,
                context_compacted_messages_delta=context_compacted_messages_delta,
                context_compacted_chars_delta=context_compacted_chars_delta,
                artifact_context_updates_delta=artifact_context_updates_delta,
                artifact_context_chars_delta=artifact_context_chars_delta,
                tool_disclosure_turns_delta=tool_disclosure_turns_delta,
                tool_disclosure_hidden_total_delta=tool_disclosure_hidden_total_delta,
                tool_disclosure_unavailable_msgs_delta=tool_disclosure_unavailable_msgs_delta,
                tool_disclosure_unavailable_reason_chars_delta=tool_disclosure_unavailable_reason_chars_delta,
                tool_disclosure_unavailable_reason_tokens_est_delta=tool_disclosure_unavailable_reason_tokens_est_delta,
                tool_disclosure_repair_suggestions_delta=tool_disclosure_repair_suggestions_delta,
                no_legal_noncontrol_turns_delta=no_legal_noncontrol_turns_delta,
                deficit_no_progress_streak=deficit_no_progress_streak,
                max_deficit_no_progress_streak=max_deficit_no_progress_streak,
                deficit_no_progress_nudges_delta=deficit_no_progress_nudges_delta,
                deficit_no_progress_last_nudged=deficit_no_progress_last_nudged,
                current_turn_deficit_digest=current_turn_deficit_digest,
                disclosure_repair_hints=disclosure_repair_hints,
                effective_model=effective_model,
                sticky_fallback=sticky_fallback,
                total_cost_delta=total_cost_delta,
                total_input_tokens_delta=total_input_tokens_delta,
                total_output_tokens_delta=total_output_tokens_delta,
                total_cached_tokens_delta=total_cached_tokens_delta,
                total_cache_creation_tokens_delta=total_cache_creation_tokens_delta,
                final_content=final_content,
                final_finish_reason=final_finish_reason,
                failure_event_codes=failure_event_codes,
                plain_text_no_tool_turn_streak=plain_text_no_tool_turn_streak,
                tool_calls_this_turn=tool_calls_this_turn,
                tool_call_turns_total_delta=tool_call_turns_total_delta,
                tool_call_empty_text_turns_delta=tool_call_empty_text_turns_delta,
                responses_tool_call_empty_text_turns_delta=responses_tool_call_empty_text_turns_delta,
                autofilled_tool_reasoning_calls_delta=autofilled_tool_reasoning_calls_delta,
                autofilled_tool_reasoning_by_tool_delta=autofilled_tool_reasoning_by_tool_delta,
                should_continue_turn=True,
                should_break_loop=False,
            )

        final_content = result.content
        final_finish_reason = result.finish_reason
        if not result.content:
            failure_event_codes.append(event_code_provider_empty)
            emit_foundation_event(
                {
                    "event_id": new_event_id(),
                    "event_type": "ToolFailed",
                    "timestamp": now_iso(),
                    "run_id": foundation_run_id,
                    "session_id": foundation_session_id,
                    "actor_id": foundation_actor_id,
                    "operation": {"name": "_inner_acall_llm", "version": None},
                    "inputs": {
                        "artifact_ids": sorted(available_artifacts),
                        "params": {
                            "turn": turn + 1,
                            "finish_reason": result.finish_reason,
                            "provider_classification": "provider_empty_candidates",
                            "provider_subevent": "first_turn" if turn == 0 else "turn",
                        },
                        "bindings": dict(available_bindings),
                    },
                    "outputs": {"artifact_ids": [], "payload_hashes": []},
                    "failure": {
                        "error_code": event_code_provider_empty,
                        "category": "provider",
                        "phase": "execution",
                        "retryable": True,
                        "tool_name": "_inner_acall_llm",
                        "user_message": (
                            "Provider returned empty content with no tool calls on first turn."
                            if turn == 0
                            else "Provider returned empty content with no tool calls."
                        ),
                        "debug_ref": None,
                    },
                }
            )
            if turn == 0:
                logger.error(
                    "Agent loop: model=%s returned empty content with 0 tool calls on turn 1 "
                    "(finish_reason=%s). All %d retries + fallback exhausted at the per-turn level.",
                    model,
                    result.finish_reason,
                    kwargs.get("num_retries", 2),
                )
            else:
                logger.warning(
                    "Agent loop: model=%s returned empty content on turn %d/%d "
                    "(finish_reason=%s, %d tool calls so far).",
                    model,
                    turn + 1,
                    max_turns,
                    result.finish_reason,
                    current_tool_call_count,
                )
        if result.content:
            agent_result.conversation_trace.append(
                {"role": "assistant", "content": result.content}
            )
        return AgentTurnModelStageResult(
            artifact_context_message_index=artifact_context_message_index,
            context_tool_result_clearings_delta=context_tool_result_clearings_delta,
            context_tool_results_cleared_delta=context_tool_results_cleared_delta,
            context_tool_result_cleared_chars_delta=context_tool_result_cleared_chars_delta,
            context_compactions_delta=context_compactions_delta,
            context_compacted_messages_delta=context_compacted_messages_delta,
            context_compacted_chars_delta=context_compacted_chars_delta,
            artifact_context_updates_delta=artifact_context_updates_delta,
            artifact_context_chars_delta=artifact_context_chars_delta,
            tool_disclosure_turns_delta=tool_disclosure_turns_delta,
            tool_disclosure_hidden_total_delta=tool_disclosure_hidden_total_delta,
            tool_disclosure_unavailable_msgs_delta=tool_disclosure_unavailable_msgs_delta,
            tool_disclosure_unavailable_reason_chars_delta=tool_disclosure_unavailable_reason_chars_delta,
            tool_disclosure_unavailable_reason_tokens_est_delta=tool_disclosure_unavailable_reason_tokens_est_delta,
            tool_disclosure_repair_suggestions_delta=tool_disclosure_repair_suggestions_delta,
            no_legal_noncontrol_turns_delta=no_legal_noncontrol_turns_delta,
            deficit_no_progress_streak=deficit_no_progress_streak,
            max_deficit_no_progress_streak=max_deficit_no_progress_streak,
            deficit_no_progress_nudges_delta=deficit_no_progress_nudges_delta,
            deficit_no_progress_last_nudged=deficit_no_progress_last_nudged,
            current_turn_deficit_digest=current_turn_deficit_digest,
            disclosure_repair_hints=disclosure_repair_hints,
            effective_model=effective_model,
            sticky_fallback=sticky_fallback,
            total_cost_delta=total_cost_delta,
            total_input_tokens_delta=total_input_tokens_delta,
            total_output_tokens_delta=total_output_tokens_delta,
            total_cached_tokens_delta=total_cached_tokens_delta,
            total_cache_creation_tokens_delta=total_cache_creation_tokens_delta,
            final_content=final_content,
            final_finish_reason=final_finish_reason,
            failure_event_codes=failure_event_codes,
            plain_text_no_tool_turn_streak=plain_text_no_tool_turn_streak,
            tool_calls_this_turn=tool_calls_this_turn,
            tool_call_turns_total_delta=tool_call_turns_total_delta,
            tool_call_empty_text_turns_delta=tool_call_empty_text_turns_delta,
            responses_tool_call_empty_text_turns_delta=responses_tool_call_empty_text_turns_delta,
            autofilled_tool_reasoning_calls_delta=autofilled_tool_reasoning_calls_delta,
            autofilled_tool_reasoning_by_tool_delta=autofilled_tool_reasoning_by_tool_delta,
            should_continue_turn=False,
            should_break_loop=True,
        )

    plain_text_no_tool_turn_streak = 0
    tool_calls_this_turn = list(result.tool_calls)
    patched_calls: list[dict[str, Any]] = []
    autofilled_reasoning_tools: list[str] = []
    for tool_call in tool_calls_this_turn:
        patched, changed = _autofill_tool_reasoning(tool_call)
        normalized_name = _normalize_tool_call_name_inplace(patched)
        if changed:
            if normalized_name:
                autofilled_reasoning_tools.append(normalized_name)
                autofilled_tool_reasoning_by_tool_delta[normalized_name] = (
                    autofilled_tool_reasoning_by_tool_delta.get(normalized_name, 0) + 1
                )
            autofilled_tool_reasoning_calls_delta += 1
        patched_calls.append(patched)
    tool_calls_this_turn = patched_calls
    if autofilled_reasoning_tools:
        agent_result.warnings.append(
            "OBSERVABILITY: auto-filled missing tool_reasoning on tools: "
            + ", ".join(autofilled_reasoning_tools)
        )
    tool_call_turns_total_delta = 1
    if not (result.content or "").strip():
        tool_call_empty_text_turns_delta = 1
        if is_responses_api_raw_response(result.raw_response):
            responses_tool_call_empty_text_turns_delta = 1
        logger.info(
            "Agent loop metric: turn %d/%d model=%s returned %d tool call(s) with empty assistant text%s",
            turn + 1,
            max_turns,
            result.model,
            len(tool_calls_this_turn),
            " [responses-api]" if responses_tool_call_empty_text_turns_delta else "",
        )

    if max_tool_calls is not None:
        budgeted_used = _count_budgeted_records(agent_result.tool_calls)
        remaining_tool_calls = max_tool_calls - budgeted_used
        budgeted_requested = _count_budgeted_tool_calls(tool_calls_this_turn)
        if budgeted_requested > remaining_tool_calls:
            tool_calls_this_turn, dropped = _trim_tool_calls_to_budget(
                tool_calls_this_turn,
                max(remaining_tool_calls, 0),
            )
            trim_msg = {
                "role": "user",
                "content": (
                    f"[SYSTEM: Retrieval-tool budget allows only {remaining_tool_calls} more budgeted call(s). "
                    f"Ignored {dropped} over-budget retrieval call(s). "
                    f"Budget-exempt tools ({', '.join(sorted(BUDGET_EXEMPT_TOOL_NAMES))}) are still allowed.]"
                ),
            }
            messages.append(trim_msg)
            agent_result.conversation_trace.append(trim_msg)
            logger.warning(
                "Agent loop trimmed %d over-budget retrieval call(s) on turn %d; budget remaining=%d",
                dropped,
                turn + 1,
                remaining_tool_calls,
            )

    missing_reasoning_tools: list[str] = []
    for tool_call in tool_calls_this_turn:
        function_payload = tool_call.get("function", {})
        tool_name = function_payload.get("name", "")
        tool_args = _extract_tool_call_args(tool_call)
        if not isinstance(tool_args, dict):
            missing_reasoning_tools.append(tool_name or "<unknown>")
            continue
        tool_reasoning = tool_args.get(TOOL_REASONING_FIELD)
        if not isinstance(tool_reasoning, str) or not tool_reasoning.strip():
            missing_reasoning_tools.append(tool_name or "<unknown>")

    if missing_reasoning_tools:
        reasoning_nudge = {
            "role": "user",
            "content": (
                "[SYSTEM: Observability requirement: every tool call must include "
                f"'{TOOL_REASONING_FIELD}' with one concise sentence explaining why this call is needed."
                + (" Calls without it are rejected." if require_tool_reasoning else "")
                + "]"
            ),
        }
        messages.append(reasoning_nudge)
        agent_result.conversation_trace.append(reasoning_nudge)
        agent_result.warnings.append(
            "OBSERVABILITY: missing tool_reasoning on tools: "
            + ", ".join(missing_reasoning_tools)
        )

    assistant_msg = {
        "role": "assistant",
        "content": result.content or None,
        "tool_calls": tool_calls_this_turn,
    }
    messages.append(assistant_msg)
    agent_result.conversation_trace.append(
        {
            "role": "assistant",
            "content": result.content or "",
            "tool_calls": [
                {
                    "id": tool_call.get("id", ""),
                    "name": tool_call.get("function", {}).get("name", ""),
                    "arguments": tool_call.get("function", {}).get("arguments", ""),
                }
                for tool_call in tool_calls_this_turn
            ],
        }
    )

    return AgentTurnModelStageResult(
        artifact_context_message_index=artifact_context_message_index,
        context_tool_result_clearings_delta=context_tool_result_clearings_delta,
        context_tool_results_cleared_delta=context_tool_results_cleared_delta,
        context_tool_result_cleared_chars_delta=context_tool_result_cleared_chars_delta,
        context_compactions_delta=context_compactions_delta,
        context_compacted_messages_delta=context_compacted_messages_delta,
        context_compacted_chars_delta=context_compacted_chars_delta,
        artifact_context_updates_delta=artifact_context_updates_delta,
        artifact_context_chars_delta=artifact_context_chars_delta,
        tool_disclosure_turns_delta=tool_disclosure_turns_delta,
        tool_disclosure_hidden_total_delta=tool_disclosure_hidden_total_delta,
        tool_disclosure_unavailable_msgs_delta=tool_disclosure_unavailable_msgs_delta,
        tool_disclosure_unavailable_reason_chars_delta=tool_disclosure_unavailable_reason_chars_delta,
        tool_disclosure_unavailable_reason_tokens_est_delta=tool_disclosure_unavailable_reason_tokens_est_delta,
        tool_disclosure_repair_suggestions_delta=tool_disclosure_repair_suggestions_delta,
        no_legal_noncontrol_turns_delta=no_legal_noncontrol_turns_delta,
        deficit_no_progress_streak=deficit_no_progress_streak,
        max_deficit_no_progress_streak=max_deficit_no_progress_streak,
        deficit_no_progress_nudges_delta=deficit_no_progress_nudges_delta,
        deficit_no_progress_last_nudged=deficit_no_progress_last_nudged,
        current_turn_deficit_digest=current_turn_deficit_digest,
        disclosure_repair_hints=disclosure_repair_hints,
        effective_model=effective_model,
        sticky_fallback=sticky_fallback,
        total_cost_delta=total_cost_delta,
        total_input_tokens_delta=total_input_tokens_delta,
        total_output_tokens_delta=total_output_tokens_delta,
        total_cached_tokens_delta=total_cached_tokens_delta,
        total_cache_creation_tokens_delta=total_cache_creation_tokens_delta,
        final_content=final_content,
        final_finish_reason=final_finish_reason,
        failure_event_codes=failure_event_codes,
        plain_text_no_tool_turn_streak=plain_text_no_tool_turn_streak,
        tool_calls_this_turn=tool_calls_this_turn,
        tool_call_turns_total_delta=tool_call_turns_total_delta,
        tool_call_empty_text_turns_delta=tool_call_empty_text_turns_delta,
        responses_tool_call_empty_text_turns_delta=responses_tool_call_empty_text_turns_delta,
        autofilled_tool_reasoning_calls_delta=autofilled_tool_reasoning_calls_delta,
        autofilled_tool_reasoning_by_tool_delta=autofilled_tool_reasoning_by_tool_delta,
        should_continue_turn=False,
        should_break_loop=False,
    )
