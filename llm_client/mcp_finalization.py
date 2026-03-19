"""Forced finalization and circuit breaker for the MCP agent loop.

When the agent loop exhausts its turns, tool budget, or detects stagnation,
this module runs a no-tools forced-final LLM call to extract the best
answer from accumulated evidence.  Includes circuit-breaker logic for
repeated same-class failures and fallback model chains.
"""

from __future__ import annotations

import json as _json
import logging
import re
from typing import Any

from llm_client.agent_outcomes import ForcedFinalizationResult
from llm_client.foundation import new_event_id, now_iso
from llm_client.mcp_context import (
    _clear_old_tool_results_for_context,
    _compact_tool_history_for_context,
)
from llm_client.tool_runtime_common import (
    MCPAgentResult,
    extract_usage_counts as _extract_usage,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_FORCED_FINAL_MAX_ATTEMPTS: int = 1
"""Maximum forced-final LLM attempts before terminating."""

DEFAULT_FORCED_FINAL_CIRCUIT_BREAKER_THRESHOLD: int = 2
"""Open forced-final circuit breaker after this many same-class failures in a row."""

DEFAULT_FORCE_SUBMIT_RETRY_ON_MAX_TOOL_CALLS: bool = True
"""When tool budget is exhausted, count one forced submit retry attempt for observability."""

DEFAULT_ACCEPT_FORCED_ANSWER_ON_MAX_TOOL_CALLS: bool = True
"""When tool budget is exhausted, allow forced-final plain answer to satisfy required submit."""

EVENT_CODE_TOOL_RUNTIME_ERROR = "TOOL_EXECUTION_RUNTIME_ERROR"
EVENT_CODE_PROVIDER_EMPTY = "PROVIDER_EMPTY_CANDIDATES"
EVENT_CODE_FINALIZATION_CIRCUIT_BREAKER_OPEN = "FINALIZATION_CIRCUIT_BREAKER_OPEN"
EVENT_CODE_FINALIZATION_TOOL_CALL_DISALLOWED = "FINALIZATION_TOOL_CALL_DISALLOWED"


# ---------------------------------------------------------------------------
# Answer normalization regexes
# ---------------------------------------------------------------------------

_FINAL_ANSWER_LINE_RE = re.compile(
    r"(?im)^\s*(?:\*\*+\s*)?(?:final\s+answer|answer)\s*[:\-]\s*(.+?)\s*$"
)
_FORCED_REFUSAL_RE = re.compile(
    r"^\s*(?:unknown|cannot\s+determine|can't\s+determine|insufficient(?:\s+evidence)?|not\s+found)\.?\s*$",
    flags=re.IGNORECASE,
)
_DATE_PHRASE_RE = re.compile(
    r"\b\d{1,2}\s+(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|"
    r"jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\s+\d{3,4}\b",
    flags=re.IGNORECASE,
)
_YEAR_TOKEN_RE = re.compile(r"\b\d{3,4}\b")


# ---------------------------------------------------------------------------
# Forced-final answer normalization
# ---------------------------------------------------------------------------

def _normalize_forced_final_answer(content: str) -> str:
    """Extract a short factual answer span from forced-final free text."""
    text = (content or "").strip()
    if not text:
        return ""

    def _clean(candidate: str) -> str:
        c = (candidate or "").strip()
        c = c.strip("`*_ \t\r\n")
        c = c.rstrip(".")
        c = " ".join(c.split())
        return c

    answer_match = _FINAL_ANSWER_LINE_RE.search(text)
    if answer_match:
        candidate = _clean(answer_match.group(1))
        if candidate and not _FORCED_REFUSAL_RE.match(candidate):
            return " ".join(candidate.split()[:8])

    date_hits = list(_DATE_PHRASE_RE.finditer(text))
    if date_hits:
        return _clean(date_hits[-1].group(0))

    year_hits = list(_YEAR_TOKEN_RE.finditer(text))
    if year_hits:
        return _clean(year_hits[-1].group(0))

    if answer_match:
        # If explicit answer line exists but is refusal-like, keep it as a last resort.
        candidate = _clean(answer_match.group(1))
        if candidate:
            return " ".join(candidate.split()[:8])

    first_line = _clean(text.splitlines()[0] if text.splitlines() else text)
    return " ".join(first_line.split()[:8])


# ---------------------------------------------------------------------------
# Provider failure classification
# ---------------------------------------------------------------------------

def _provider_failure_classification(exc: Exception, error_text: str) -> tuple[bool, str, str, bool]:
    """Best-effort provider failure classification for taxonomy + retry semantics."""
    classification = getattr(exc, "classification", None)
    if isinstance(classification, str):
        normalized = classification.strip().lower()
        if normalized.startswith("provider_empty"):
            return True, EVENT_CODE_PROVIDER_EMPTY, normalized, True
        if normalized in {
            "provider_credits_exhausted",
            "provider_insufficient_credits",
            "provider_insufficient_quota",
        }:
            from llm_client.mcp_agent import EVENT_CODE_PROVIDER_CREDITS_EXHAUSTED
            return True, EVENT_CODE_PROVIDER_CREDITS_EXHAUSTED, normalized, False

    lower = (error_text or "").strip().lower()
    if "provider_empty_candidates" in lower or "empty content from llm" in lower:
        return True, EVENT_CODE_PROVIDER_EMPTY, "provider_empty_candidates", True
    if (
        "insufficient credits" in lower
        or "insufficient quota" in lower
        or "\"code\":402" in lower
    ):
        from llm_client.mcp_agent import EVENT_CODE_PROVIDER_CREDITS_EXHAUSTED
        return (
            True,
            EVENT_CODE_PROVIDER_CREDITS_EXHAUSTED,
            "provider_credits_exhausted",
            False,
        )
    return False, "", "", False


# ---------------------------------------------------------------------------
# Forced finalization execution
# ---------------------------------------------------------------------------

async def _execute_forced_finalization(
    *,
    force_final_reason: str,
    max_turns: int,
    max_message_chars: int,
    tool_result_keep_recent: int,
    tool_result_context_preview_chars: int,
    messages: list[dict[str, Any]],
    tool_result_metadata_by_id: dict[str, dict[str, Any]],
    artifact_context_message_index: int | None,
    kwargs: dict[str, Any],
    timeout: int,
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
    agent_result: MCPAgentResult,
    emit_foundation_event: Any,
) -> ForcedFinalizationResult:
    """Run the forced-final no-tools lane and return aggregate deltas/outcomes."""
    # Lazy imports to avoid circular dependencies
    from llm_client.mcp_agent import _inner_acall_llm, _upsert_active_artifact_context_message

    _no_tools_suffix = (
        " Tools are now DISABLED — do NOT call any functions. "
        "Respond with ONLY a short text answer (name, date, number, or yes/no)."
    )
    if force_final_reason == "max_tool_calls":
        force_msg = {
            "role": "user",
            "content": (
                "[SYSTEM: Tool-call budget exhausted. You MUST give your best answer now "
                "using evidence already retrieved. Extract the best name/date/number."
                + _no_tools_suffix + "]"
            ),
        }
    elif force_final_reason == "control_churn":
        force_msg = {
            "role": "user",
            "content": (
                "[SYSTEM: Tool-call loop stalled with repeated non-executable calls. "
                "You MUST submit your best factual answer now from current evidence."
                + _no_tools_suffix + "]"
            ),
        }
    elif force_final_reason == "retrieval_stagnation":
        force_msg = {
            "role": "user",
            "content": (
                "[SYSTEM: Retrieval stagnated with no new evidence. "
                "Submit your best factual answer now based only on retrieved evidence."
                + _no_tools_suffix + "]"
            ),
        }
    else:
        logger.warning(
            "Agent loop exhausted max_turns=%d, forcing final answer",
            max_turns,
        )
        force_msg = {
            "role": "user",
            "content": (
                "[SYSTEM: All turns exhausted. You MUST give your best answer now "
                "based on the evidence you found. Extract any name, date, or number "
                "from the tool results. 'I don't know' is NOT acceptable — guess."
                + _no_tools_suffix + "]"
            ),
        }

    messages.append(force_msg)
    agent_result.conversation_trace.append(force_msg)

    _artifact_context_idx, artifact_context_content, artifact_context_changed = (
        _upsert_active_artifact_context_message(
            messages,
            available_artifacts=available_artifacts,
            available_capabilities=available_capabilities,
            tool_result_metadata_by_id=tool_result_metadata_by_id,
            enabled=active_artifact_context_enabled,
            max_handles=active_artifact_context_max_handles,
            max_chars=active_artifact_context_max_chars,
            existing_index=artifact_context_message_index,
        )
    )
    if artifact_context_changed and artifact_context_content:
        agent_result.conversation_trace.append(
            {
                "role": "user",
                "content": artifact_context_content,
                "synthetic": "active_artifact_context",
            }
        )

    context_tool_result_clearings_delta = 0
    context_tool_results_cleared_delta = 0
    context_tool_result_cleared_chars_delta = 0
    context_compactions_delta = 0
    context_compacted_messages_delta = 0
    context_compacted_chars_delta = 0
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
            f"(saved ~{cleared_chars} chars; keep_recent={tool_result_keep_recent}) before forced-final call"
        )
        agent_result.warnings.append(warning)
        logger.warning(warning)

    compacted_count, compacted_chars, current_chars = _compact_tool_history_for_context(
        messages, max_message_chars,
    )
    if compacted_count:
        context_compactions_delta = 1
        context_compacted_messages_delta = compacted_count
        context_compacted_chars_delta = compacted_chars
        warning = (
            "CONTEXT_COMPACTION: compacted "
            f"{compacted_count} tool message(s), saved ~{compacted_chars} chars "
            f"(history ~{current_chars} chars) before forced-final call"
        )
        agent_result.warnings.append(warning)
        logger.warning(warning)

    # Finalization fallback lane:
    # - Primary forced-final call uses the current effective model.
    # - Optional fallback models are attempted only for finalization.
    # - No tools are passed, and per-turn fallback chains are disabled.
    forced_kwargs = dict(kwargs)
    forced_kwargs.pop("fallback_models", None)
    # Forced-final is a terminal rescue path; keep retries bounded so final
    # submission does not spend multiple full timeout windows before returning.
    forced_retry_value = forced_kwargs.get("num_retries", 1)
    try:
        forced_retry_value = int(forced_retry_value)
    except Exception:
        forced_retry_value = 1
    forced_kwargs["num_retries"] = max(0, min(forced_retry_value, 1))
    finalization_primary_model = effective_model
    base_model_chain: list[str] = [effective_model]
    for fb_model in finalization_fallback_models:
        if fb_model.lower() == effective_model.lower():
            continue
        base_model_chain.append(fb_model)

    def _forced_final_model_for_attempt(attempt_idx: int) -> str:
        if attempt_idx < len(base_model_chain):
            return base_model_chain[attempt_idx]
        return base_model_chain[-1]

    total_cost_delta = 0.0
    total_input_tokens_delta = 0
    total_output_tokens_delta = 0
    total_cached_tokens_delta = 0
    total_cache_creation_tokens_delta = 0
    turns_delta = 0

    forced_final_attempts = 0
    forced_final_circuit_breaker_opened = False
    forced_final_error_streak = 0
    last_forced_final_error_class: str | None = None
    finalization_fallback_used = False
    finalization_fallback_succeeded = False
    finalization_events: list[str] = []
    finalization_fallback_attempts: list[dict[str, Any]] = []
    final_result = None
    terminal_error_text: str | None = None
    forced_final_failure_codes: list[str] = []

    def _record_forced_final_error(
        *,
        attempt_idx: int,
        final_model: str,
        event_code: str,
        error_text: str,
        error_class: str,
        is_provider_failure: bool,
    ) -> bool:
        nonlocal forced_final_error_streak, last_forced_final_error_class
        nonlocal terminal_error_text, forced_final_circuit_breaker_opened
        forced_final_failure_codes.append(event_code)
        if final_model.lower() == effective_model.lower():
            finalization_events.append("FINALIZATION_PRIMARY_FAILED")
        else:
            finalization_events.append("FINALIZATION_FALLBACK_FAILED")
        finalization_fallback_attempts.append(
            {
                "attempt_index": attempt_idx + 1,
                "model": final_model,
                "status": "error",
                "error_code": event_code,
                "error_text": error_text,
                "error_class": error_class,
            }
        )
        if error_class == last_forced_final_error_class:
            forced_final_error_streak += 1
        else:
            forced_final_error_streak = 1
            last_forced_final_error_class = error_class
        terminal_error_text = error_text
        if forced_final_error_streak < forced_final_circuit_breaker_threshold:
            return False
        forced_final_circuit_breaker_opened = True
        finalization_events.append("FINALIZATION_CIRCUIT_BREAKER_OPEN")
        forced_final_failure_codes.append(EVENT_CODE_FINALIZATION_CIRCUIT_BREAKER_OPEN)
        emit_foundation_event(
            {
                "event_id": new_event_id(),
                "event_type": "ToolFailed",
                "timestamp": now_iso(),
                "run_id": foundation_run_id,
                "session_id": foundation_session_id,
                "actor_id": foundation_actor_id,
                "operation": {"name": "__finalization__", "version": None},
                "inputs": {
                    "artifact_ids": sorted(available_artifacts),
                    "params": {
                        "attempts": forced_final_attempts,
                        "threshold": forced_final_circuit_breaker_threshold,
                        "error_class": error_class,
                    },
                    "bindings": dict(available_bindings),
                },
                "outputs": {"artifact_ids": [], "payload_hashes": []},
                "failure": {
                    "error_code": EVENT_CODE_FINALIZATION_CIRCUIT_BREAKER_OPEN,
                    "category": "provider" if is_provider_failure else "execution",
                    "phase": "execution",
                    "retryable": False,
                    "tool_name": "__finalization__",
                    "user_message": (
                        "Finalization circuit breaker opened after repeated same-class failures."
                    ),
                    "debug_ref": None,
                },
            }
        )
        return True

    for attempt_idx in range(forced_final_max_attempts):
        if forced_final_circuit_breaker_opened:
            break
        final_model = _forced_final_model_for_attempt(attempt_idx)
        forced_final_attempts += 1
        is_fallback_attempt = final_model.lower() != effective_model.lower()
        if is_fallback_attempt and not finalization_fallback_used:
            finalization_fallback_used = True
            finalization_events.append("FINALIZATION_FALLBACK_USED")

        try:
            candidate_result = await _inner_acall_llm(
                final_model,
                messages,
                timeout=timeout,
                **forced_kwargs,
            )
        except Exception as exc:
            error_text = str(exc).strip() or f"{type(exc).__name__}"
            (
                is_provider_failure,
                event_code,
                provider_classification,
                retryable_provider,
            ) = _provider_failure_classification(
                exc,
                error_text,
            )
            if not event_code:
                event_code = EVENT_CODE_TOOL_RUNTIME_ERROR
            error_class = provider_classification or type(exc).__name__
            warning = (
                "AGENT_LLM_CALL_FAILED: forced_final attempt="
                f"{attempt_idx + 1}/{forced_final_max_attempts} model={final_model} error={error_text}"
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
                            "turn": agent_result.turns + 1,
                            "forced_final": True,
                            "attempt": attempt_idx + 1,
                            "model": final_model,
                            "error": error_text,
                            "provider_classification": provider_classification or "",
                            "provider_subevent": "finalization" if is_provider_failure else "",
                        },
                        "bindings": dict(available_bindings),
                    },
                    "outputs": {"artifact_ids": [], "payload_hashes": []},
                    "failure": {
                        "error_code": event_code,
                        "category": "provider" if is_provider_failure else "execution",
                        "phase": "execution",
                        "retryable": bool(retryable_provider),
                        "tool_name": "_inner_acall_llm",
                        "user_message": error_text,
                        "debug_ref": None,
                    },
                }
            )
            _record_forced_final_error(
                attempt_idx=attempt_idx,
                final_model=final_model,
                event_code=event_code,
                error_text=error_text,
                error_class=error_class,
                is_provider_failure=is_provider_failure,
            )
            continue

        final_result_model = str(candidate_result.model).strip() or final_model
        agent_result.models_used.add(final_result_model)
        if final_result_model and final_result_model not in attempted_models:
            attempted_models.append(final_result_model)
        if candidate_result.warnings:
            agent_result.warnings.extend(candidate_result.warnings)

        if candidate_result.tool_calls:
            # Model returned tool calls from conversation history even though
            # tools weren't passed.  Instead of rejecting the response, salvage
            # the answer: check text content first, then look for submit_answer
            # in the tool calls themselves.
            salvaged_content = (candidate_result.content or "").strip()
            if not salvaged_content:
                for tc in candidate_result.tool_calls:
                    fn = getattr(tc, "function", tc) if not isinstance(tc, dict) else tc
                    tc_name = (
                        getattr(fn, "name", None)
                        or (fn.get("function", {}).get("name") if isinstance(fn, dict) else None)
                        or ""
                    )
                    if "submit_answer" in tc_name.lower():
                        tc_args = getattr(fn, "arguments", None)
                        if isinstance(tc_args, str):
                            try:
                                tc_args = _json.loads(tc_args)
                            except Exception:
                                pass
                        if isinstance(tc_args, dict):
                            salvaged_content = str(tc_args.get("answer", "")).strip()
                            if salvaged_content:
                                break
            if salvaged_content:
                # Use the salvaged content as if the model returned text.
                candidate_result = type(candidate_result)(
                    content=salvaged_content,
                    usage=candidate_result.usage,
                    cost=candidate_result.cost,
                    model=candidate_result.model,
                    tool_calls=[],
                    finish_reason="stop",
                    raw_response=candidate_result.raw_response,
                    warnings=list(candidate_result.warnings or []) + [
                        "FINALIZATION_TOOL_CALLS_SALVAGED: extracted answer from "
                        "tool_calls returned in no-tools forced-final lane."
                    ],
                )
                agent_result.warnings.append(
                    "FINALIZATION_TOOL_CALLS_SALVAGED: model returned tool_calls in "
                    "forced-final lane; salvaged answer from tool call arguments."
                )
            else:
                disallowed_msg = (
                    "Forced-final returned tool calls while tools are disallowed."
                )
                _record_forced_final_error(
                    attempt_idx=attempt_idx,
                    final_model=final_model,
                    event_code=EVENT_CODE_FINALIZATION_TOOL_CALL_DISALLOWED,
                    error_text=disallowed_msg,
                    error_class="finalization_tool_call_disallowed",
                    is_provider_failure=False,
                )
                continue

        if not candidate_result.content:
            empty_msg = "Provider returned empty content on forced-final call."
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
                            "turn": agent_result.turns + 1,
                            "forced_final": True,
                            "attempt": attempt_idx + 1,
                            "model": final_model,
                            "provider_classification": "provider_empty_candidates",
                            "provider_subevent": "finalization",
                        },
                        "bindings": dict(available_bindings),
                    },
                    "outputs": {"artifact_ids": [], "payload_hashes": []},
                    "failure": {
                        "error_code": EVENT_CODE_PROVIDER_EMPTY,
                        "category": "provider",
                        "phase": "execution",
                        "retryable": True,
                        "tool_name": "_inner_acall_llm",
                        "user_message": empty_msg,
                        "debug_ref": None,
                    },
                }
            )
            _record_forced_final_error(
                attempt_idx=attempt_idx,
                final_model=final_model,
                event_code=EVENT_CODE_PROVIDER_EMPTY,
                error_text=empty_msg,
                error_class="provider_empty_candidates",
                is_provider_failure=True,
            )
            continue

        forced_final_error_streak = 0
        last_forced_final_error_class = None
        final_result = candidate_result
        final_content = candidate_result.content
        final_finish_reason = candidate_result.finish_reason
        finalization_fallback_attempts.append(
            {
                "attempt_index": attempt_idx + 1,
                "model": final_model,
                "status": "success",
                "error_code": None,
                "error_text": "",
                "error_class": None,
            }
        )
        if is_fallback_attempt:
            finalization_fallback_succeeded = True
            finalization_events.append("FINALIZATION_FALLBACK_SUCCEEDED")
        else:
            finalization_events.append("FINALIZATION_PRIMARY_SUCCEEDED")
        total_cost_delta += candidate_result.cost
        inp, out, cached, cache_create = _extract_usage(candidate_result.usage or {})
        total_input_tokens_delta += inp
        total_output_tokens_delta += out
        total_cached_tokens_delta += cached
        total_cache_creation_tokens_delta += cache_create
        turns_delta += 1
        agent_result.conversation_trace.append({
            "role": "assistant",
            "content": final_content,
        })
        break

    if final_result is None:
        if forced_final_circuit_breaker_opened:
            warning = (
                "FINALIZATION_CIRCUIT_BREAKER_OPEN: halted forced-final retries after "
                f"{forced_final_attempts} attempt(s)."
            )
            agent_result.warnings.append(warning)
            logger.warning(warning)
        final_content = terminal_error_text or "Forced-final failed with no provider content."
        final_finish_reason = "error"
        failure_event_codes = list(forced_final_failure_codes)
    else:
        failure_event_codes = []

    return ForcedFinalizationResult(
        final_content=final_content,
        final_finish_reason=final_finish_reason,
        finalization_primary_model=finalization_primary_model,
        forced_final_attempts=forced_final_attempts,
        forced_final_circuit_breaker_opened=forced_final_circuit_breaker_opened,
        finalization_fallback_used=finalization_fallback_used,
        finalization_fallback_succeeded=finalization_fallback_succeeded,
        finalization_events=finalization_events,
        finalization_fallback_attempts=finalization_fallback_attempts,
        failure_event_codes=failure_event_codes,
        context_tool_result_clearings_delta=context_tool_result_clearings_delta,
        context_tool_results_cleared_delta=context_tool_results_cleared_delta,
        context_tool_result_cleared_chars_delta=context_tool_result_cleared_chars_delta,
        context_compactions_delta=context_compactions_delta,
        context_compacted_messages_delta=context_compacted_messages_delta,
        context_compacted_chars_delta=context_compacted_chars_delta,
        total_cost_delta=total_cost_delta,
        total_input_tokens_delta=total_input_tokens_delta,
        total_output_tokens_delta=total_output_tokens_delta,
        total_cached_tokens_delta=total_cached_tokens_delta,
        total_cache_creation_tokens_delta=total_cache_creation_tokens_delta,
        turns_delta=turns_delta,
    )
