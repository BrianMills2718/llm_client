"""MCP agent loop turn-execution helpers for llm_client.

Enables any litellm model to act as a tool-calling agent by connecting
to MCP servers, discovering their tools, and running an autonomous
tool-calling loop.

Usage (via call_llm/acall_llm — routing is automatic):
    result = await acall_llm(
        "gemini/gemini-3-flash-preview",
        messages,
        mcp_servers={
            "my-server": {
                "command": "python",
                "args": ["-u", "server.py"],
                "env": {"KEY": "value"},
            }
        },
        max_turns=20,
    )

This module holds the turn-execution implementation that the facade re-exports.

The loop:
    1. Start MCP server subprocesses (stdio transport)
    2. Discover tools via session.list_tools()
    3. Convert MCP tool schemas to OpenAI function-calling format
    4. Call LLM with tools → if tool_calls → execute via MCP → repeat
    5. Return LLMCallResult with accumulated usage/cost
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import os
import time
from contextlib import AsyncExitStack
from typing import Any

from llm_client.agent.agent_artifacts import (
    _artifact_handle_summaries as _agent_artifact_handle_summaries,
    _build_active_artifact_context_content as _agent_build_active_artifact_context_content,
    _collect_recent_artifact_handles as _agent_collect_recent_artifact_handles,
    _upsert_active_artifact_context_message as _agent_upsert_active_artifact_context_message,
)
from llm_client.agent.agent_contracts import (
    CapabilityRequirement,
    ToolCallValidation,
    _apply_handle_input_injections as _agent_apply_handle_input_injections,
    _artifact_output_state_from_payload as _agent_artifact_output_state_from_payload,
    _capability_state_add as _agent_capability_state_add,
    _capability_state_snapshot as _agent_capability_state_snapshot,
    _contract_output_capabilities as _agent_contract_output_capabilities,
    _contract_outputs as _agent_contract_outputs,
    _full_bindings_spec as _agent_full_bindings_spec,
    _full_bindings_state_hash as _agent_full_bindings_state_hash,
    _hard_bindings_spec as _agent_hard_bindings_spec,
    _hard_bindings_state_hash as _agent_hard_bindings_state_hash,
    _infer_output_capabilities as _agent_infer_output_capabilities,
    _is_control_tool_name as _agent_is_control_tool_name,
)
from llm_client.agent.agent_disclosure import (
    _deficit_labels_from_hidden_entries as _agent_deficit_labels_from_hidden_entries,
    _disclosure_message as _agent_disclosure_message,
    _disclosure_reason_from_entry as _agent_disclosure_reason_from_entry,
    _filter_tools_for_disclosure as _agent_filter_tools_for_disclosure,
)
from llm_client.agent.agent_adoption import (
    DEFAULT_ADOPTION_PROFILE,
    AdoptionProfileAssessment,
    assess_adoption_profile,
    normalize_adoption_profile,
)
from llm_client.agent.agent_outcomes import (
    ForcedFinalizationResult,
    _PRIMARY_FAILURE_PRIORITY,
    _TERMINAL_FAILURE_EVENT_CODES,
    _classify_failure_signals,
    _failure_class_for_event_code,
    _first_terminal_failure_event_code,
    _summarize_failure_events,
    _summarize_finalization_attempts,
)
from llm_client.core.client import LLMCallResult
from llm_client.agent.compliance_gate import (
    build_tool_parameter_index,
    validate_tool_call_inputs,
)
from llm_client.foundation import (
    BINDING_KEYS,
    HARD_BINDING_KEYS,
    check_binding_conflicts,
    coerce_run_id,
    evidence_pointer_label,
    extract_artifact_envelopes,
    empty_bindings,
    extract_bindings_from_tool_args,
    merge_binding_state,
    new_event_id,
    new_session_id,
    normalize_bindings,
    now_iso,
    sha256_json,
    sha256_text,
    validate_foundation_event,
)
# --- Decomposed sub-modules (Phase 1B) ---
from llm_client.agent.mcp_tools import (  # noqa: F401  re-exported
    AUTO_REASONING_TOOL_DEFAULTS as AUTO_REASONING_TOOL_DEFAULTS,
    BUDGET_EXEMPT_TOOL_NAMES as BUDGET_EXEMPT_TOOL_NAMES,
    RUNTIME_ARTIFACT_READ_TOOL_NAME as RUNTIME_ARTIFACT_READ_TOOL_NAME,
    _autofill_tool_reasoning as _autofill_tool_reasoning,
    _count_budgeted_records as _count_budgeted_records,
    _count_budgeted_tool_calls as _count_budgeted_tool_calls,
    _extract_tool_call_args as _extract_tool_call_args,
    _is_budget_exempt_tool as _is_budget_exempt_tool,
    _normalize_tool_call_name_inplace as _normalize_tool_call_name_inplace,
    _normalized_tool_name as _normalized_tool_name,
    _parse_record_result_json as _parse_record_result_json,
    _parse_record_result_json_value as _parse_record_result_json_value,
    _runtime_artifact_read_contract as _runtime_artifact_read_contract,
    _runtime_artifact_read_result as _runtime_artifact_read_result,
    _runtime_artifact_read_tool_def as _runtime_artifact_read_tool_def,
    _set_tool_call_args as _set_tool_call_args,
    _tool_error_signature as _tool_error_signature,
    _trim_tool_calls_to_budget as _trim_tool_calls_to_budget,
)
from llm_client.agent.mcp_context import (  # noqa: F401  re-exported
    DEFAULT_MAX_MESSAGE_CHARS as DEFAULT_MAX_MESSAGE_CHARS,
    DEFAULT_TOOL_RESULT_CONTEXT_PREVIEW_CHARS as DEFAULT_TOOL_RESULT_CONTEXT_PREVIEW_CHARS,
    DEFAULT_TOOL_RESULT_KEEP_RECENT as DEFAULT_TOOL_RESULT_KEEP_RECENT,
    _clear_old_tool_results_for_context as _clear_old_tool_results_for_context,
    _compact_tool_history_for_context as _compact_tool_history_for_context,
    _message_char_length as _message_char_length,
    _trim_text as _trim_text,
)
from llm_client.agent.mcp_turn_completion import _resolve_turn_completion
from llm_client.agent.mcp_evidence import (  # noqa: F401  re-exported
    DEFAULT_RETRIEVAL_STAGNATION_ACTION as DEFAULT_RETRIEVAL_STAGNATION_ACTION,
    DEFAULT_RETRIEVAL_STAGNATION_TURNS as DEFAULT_RETRIEVAL_STAGNATION_TURNS,
    RETRIEVAL_STAGNATION_ACTIONS as RETRIEVAL_STAGNATION_ACTIONS,
    _collect_evidence_pointer_labels as _collect_evidence_pointer_labels,
    _evidence_digest as _evidence_digest,
    _is_evidence_tool_name as _is_evidence_tool_name,
    _tool_evidence_pointer_labels as _tool_evidence_pointer_labels,
)
from llm_client.agent.mcp_finalization import (  # noqa: F401  re-exported
    DEFAULT_ACCEPT_FORCED_ANSWER_ON_MAX_TOOL_CALLS as DEFAULT_ACCEPT_FORCED_ANSWER_ON_MAX_TOOL_CALLS,
    DEFAULT_FORCE_SUBMIT_RETRY_ON_MAX_TOOL_CALLS as DEFAULT_FORCE_SUBMIT_RETRY_ON_MAX_TOOL_CALLS,
    DEFAULT_FORCED_FINAL_CIRCUIT_BREAKER_THRESHOLD as DEFAULT_FORCED_FINAL_CIRCUIT_BREAKER_THRESHOLD,
    DEFAULT_FORCED_FINAL_MAX_ATTEMPTS as DEFAULT_FORCED_FINAL_MAX_ATTEMPTS,
    _FORCED_REFUSAL_RE as _FORCED_REFUSAL_RE,
    _execute_forced_finalization as _execute_forced_finalization,
    _normalize_forced_final_answer as _normalize_forced_final_answer,
    _provider_failure_classification as _provider_failure_classification,
)
from llm_client.agent.mcp_loop_summary import _apply_agent_loop_summary
from llm_client.agent.mcp_turn_model import _run_turn_model_stage
from llm_client.agent.mcp_turn_outcomes import _process_turn_outcomes
from llm_client.agent.agent_contracts import AgentErrorBudget, ErrorBudgetState
from llm_client.agent.mcp_turn_tools import _process_tool_calls_turn
from llm_client.agent.mcp_state import (  # noqa: F401  re-exported
    ADOPTION_PROFILE_ENFORCE_ENV as ADOPTION_PROFILE_ENFORCE_ENV,
    ADOPTION_PROFILE_ENV as ADOPTION_PROFILE_ENV,
    AgentLoopRuntimePolicy as AgentLoopRuntimePolicy,
    AgentLoopToolState as AgentLoopToolState,
    DEFAULT_ACTIVE_ARTIFACT_CONTEXT_ENABLED as DEFAULT_ACTIVE_ARTIFACT_CONTEXT_ENABLED,
    DEFAULT_ACTIVE_ARTIFACT_CONTEXT_MAX_CHARS as DEFAULT_ACTIVE_ARTIFACT_CONTEXT_MAX_CHARS,
    DEFAULT_ACTIVE_ARTIFACT_CONTEXT_MAX_HANDLES as DEFAULT_ACTIVE_ARTIFACT_CONTEXT_MAX_HANDLES,
    DEFAULT_INITIAL_ARTIFACTS as DEFAULT_INITIAL_ARTIFACTS,
    _coerce_bool as _coerce_bool,
    _initialize_agent_tool_state as _initialize_agent_tool_state,
    _normalize_model_chain as _normalize_model_chain,
    _resolve_agent_runtime_policy as _resolve_agent_runtime_policy,
)
from llm_client.agent.mcp_contracts import (  # noqa: F401  re-exported
    _analyze_lane_closure as _analyze_lane_closure,
    _apply_handle_input_injections as _apply_handle_input_injections,
    _artifact_handle_summaries as _artifact_handle_summaries,
    _artifact_output_state_from_record as _artifact_output_state_from_record,
    _build_active_artifact_context_content as _build_active_artifact_context_content,
    _capability_requirement_from_raw as _capability_requirement_from_raw,
    _capability_state_add as _capability_state_add,
    _capability_state_has as _capability_state_has,
    _capability_state_snapshot as _capability_state_snapshot,
    _collect_recent_artifact_handles as _collect_recent_artifact_handles,
    _contract_declares_no_artifact_prereqs as _contract_declares_no_artifact_prereqs,
    _contract_output_capabilities as _contract_output_capabilities,
    _contract_outputs as _contract_outputs,
    _deficit_labels_from_hidden_entries as _deficit_labels_from_hidden_entries,
    _disclosure_message as _disclosure_message,
    _disclosure_reason_from_entry as _disclosure_reason_from_entry,
    _effective_contract_requirements as _effective_contract_requirements,
    _filter_tools_for_disclosure as _filter_tools_for_disclosure,
    _find_repair_tools_for_missing_requirements as _find_repair_tools_for_missing_requirements,
    _full_bindings_spec as _full_bindings_spec,
    _full_bindings_state_hash as _full_bindings_state_hash,
    _hard_bindings_spec as _hard_bindings_spec,
    _hard_bindings_state_hash as _hard_bindings_state_hash,
    _infer_output_capabilities as _infer_output_capabilities,
    _is_control_tool_name as _is_control_tool_name,
    _normalize_artifact_kind as _normalize_artifact_kind,
    _normalize_tool_contracts as _normalize_tool_contracts,
    _short_requirement as _short_requirement,
    _tool_declares_no_artifact_prereqs as _tool_declares_no_artifact_prereqs,
    _upsert_active_artifact_context_message as _upsert_active_artifact_context_message,
    _validate_tool_contract_call as _validate_tool_contract_call,
)
from llm_client.tools.tool_runtime_common import (
    DEFAULT_TOOL_INPUT_EXAMPLE_MAX_CHARS as DEFAULT_TOOL_INPUT_EXAMPLE_MAX_CHARS,
    DEFAULT_TOOL_INPUT_EXAMPLES_MAX_ITEMS as DEFAULT_TOOL_INPUT_EXAMPLES_MAX_ITEMS,
    MCPAgentResult as MCPAgentResult,
    MCPToolCallRecord as MCPToolCallRecord,
    TOOL_REASONING_FIELD as TOOL_REASONING_FIELD,
    append_input_examples_to_description as _append_input_examples_to_description,
    extract_usage_counts as _extract_usage,
    normalize_tool_contracts as _shared_normalize_tool_contracts,
    normalize_tool_input_examples as _normalize_tool_input_examples,
    truncate_text as _truncate,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configurable defaults — all overridable via kwargs to call_llm/acall_llm
# ---------------------------------------------------------------------------

DEFAULT_MAX_TURNS: int = 20
"""Maximum tool-calling loop iterations before forcing a final answer."""

DEFAULT_MAX_TOOL_CALLS: int | None = 20
"""Maximum tool calls before forcing a final answer. None disables tool budgeting."""

DEFAULT_REQUIRE_TOOL_REASONING: bool = False
"""Hard-fail tool calls missing tool_reasoning when enabled."""

TURN_WARNING_THRESHOLD: int = 3
"""Inject a 'wrap up' system message this many turns before max_turns."""

DEFAULT_MCP_INIT_TIMEOUT: float = 30.0
"""Seconds to wait for each MCP server subprocess to initialize."""

DEFAULT_TOOL_RESULT_MAX_LENGTH: int = 50_000
"""Maximum character length for a single tool result. Longer results are truncated."""

DEFAULT_ENFORCE_TOOL_CONTRACTS: bool = False
"""When enabled, reject tool calls that violate declared composability contracts."""

DEFAULT_PROGRESSIVE_TOOL_DISCLOSURE: bool = True
"""When enabled with contracts, expose only currently composable tools per turn."""

DEFAULT_SUPPRESS_CONTROL_LOOP_CALLS: bool = False
"""When enabled, suppress repeated submit/todo control calls after validation failures."""

DEFAULT_TOOL_DISCLOSURE_MAX_UNAVAILABLE: int = 10
"""Maximum unavailable tools reported per turn in disclosure guidance."""

DEFAULT_TOOL_DISCLOSURE_MAX_MISSING_PER_TOOL: int = 2
"""Maximum missing requirements listed per unavailable tool."""

DEFAULT_TOOL_DISCLOSURE_MAX_REPAIR_TOOLS: int = 2
"""Maximum conversion/repair tool hints surfaced per unavailable tool."""

DEFAULT_TOOL_DISCLOSURE_REASON_MAX_CHARS: int = 220
"""Maximum chars allocated to each unavailable-tool reason snippet."""

DEFAULT_TOOL_DISCLOSURE_TOKEN_CHARS: int = 4
"""Approximate chars/token conversion used for prompt-overhead accounting."""

DEFAULT_TOOL_CALL_STALL_TURNS: int = 3
"""Force final answer after this many consecutive tool-call turns with zero executable calls."""

FOUNDATION_SCHEMA_STRICT_ENV: str = "FOUNDATION_SCHEMA_STRICT"
"""When true, invalid foundation events raise instead of only warning."""

EVENT_CODE_TOOL_VALIDATION_MISSING_PREREQUISITE = "TOOL_VALIDATION_REJECTED_MISSING_PREREQUISITE"
EVENT_CODE_TOOL_VALIDATION_MISSING_CAPABILITY = "TOOL_VALIDATION_REJECTED_MISSING_CAPABILITY"
EVENT_CODE_TOOL_VALIDATION_BINDING_CONFLICT = "TOOL_VALIDATION_REJECTED_BINDING_CONFLICT"
EVENT_CODE_TOOL_VALIDATION_MISSING_TOOL_REASONING = "TOOL_VALIDATION_REJECTED_MISSING_TOOL_REASONING"
EVENT_CODE_TOOL_VALIDATION_SCHEMA = "TOOL_VALIDATION_REJECTED_SCHEMA"
EVENT_CODE_CONTROL_LOOP_SUPPRESSED = "CONTROL_CHURN_SUPPRESSED"
EVENT_CODE_CONTROL_CHURN_THRESHOLD = "CONTROL_CHURN_THRESHOLD_EXCEEDED"
EVENT_CODE_TOOL_RUNTIME_ERROR = "TOOL_EXECUTION_RUNTIME_ERROR"
EVENT_CODE_PROVIDER_EMPTY = "PROVIDER_EMPTY_CANDIDATES"
EVENT_CODE_PROVIDER_CREDITS_EXHAUSTED = "PROVIDER_CREDITS_EXHAUSTED"
# Compatibility aliases retained for external imports; taxonomy now uses one canonical code.
EVENT_CODE_PROVIDER_EMPTY_FIRST_TURN = EVENT_CODE_PROVIDER_EMPTY
EVENT_CODE_PROVIDER_EMPTY_RESPONSE = EVENT_CODE_PROVIDER_EMPTY
EVENT_CODE_FINALIZATION_CIRCUIT_BREAKER_OPEN = "FINALIZATION_CIRCUIT_BREAKER_OPEN"
EVENT_CODE_FINALIZATION_TOOL_CALL_DISALLOWED = "FINALIZATION_TOOL_CALL_DISALLOWED"
EVENT_CODE_RETRIEVAL_STAGNATION = "RETRIEVAL_STAGNATION"
EVENT_CODE_RETRIEVAL_STAGNATION_OBSERVED = "RETRIEVAL_STAGNATION_OBSERVED"
EVENT_CODE_NO_LEGAL_NONCONTROL_TOOLS = "NO_LEGAL_NONCONTROL_TOOLS"
EVENT_CODE_REQUIRED_SUBMIT_NOT_ATTEMPTED = "REQUIRED_SUBMIT_NOT_ATTEMPTED"
EVENT_CODE_REQUIRED_SUBMIT_NOT_ACCEPTED = "REQUIRED_SUBMIT_NOT_ACCEPTED"
EVENT_CODE_SUBMIT_FORCED_ACCEPT_BUDGET_EXHAUSTION = "SUBMIT_FORCED_ACCEPT_BUDGET_EXHAUSTION"
EVENT_CODE_SUBMIT_FORCED_ACCEPT_TURN_EXHAUSTION = "SUBMIT_FORCED_ACCEPT_TURN_EXHAUSTION"
EVENT_CODE_SUBMIT_FORCED_ACCEPT_FORCED_FINAL = "SUBMIT_FORCED_ACCEPT_FORCED_FINAL"

# Kwargs consumed by the MCP agent loop (popped before passing to inner acall_llm)
MCP_LOOP_KWARGS = frozenset({
    "mcp_servers",
    "mcp_sessions",
    "max_turns",
    "max_tool_calls",
    "require_tool_reasoning",
    "mcp_init_timeout",
    "tool_result_max_length",
    "max_message_chars",
    "tool_result_keep_recent",
    "tool_result_context_preview_chars",
    "active_artifact_context_enabled",
    "active_artifact_context_max_handles",
    "active_artifact_context_max_chars",
    "enforce_tool_contracts",
    "progressive_tool_disclosure",
    "suppress_control_loop_calls",
    "tool_contracts",
    "initial_artifacts",
    "initial_bindings",
    "forced_final_max_attempts",
    "forced_final_circuit_breaker_threshold",
    "force_submit_retry_on_max_tool_calls",
    "accept_forced_answer_on_max_tool_calls",
    "finalization_fallback_models",
    "retrieval_stagnation_turns",
    "retrieval_stagnation_action",
    "adoption_profile",
    "adoption_profile_enforce",
})

# Kwargs consumed by the direct tool loop
TOOL_LOOP_KWARGS = frozenset({
    "python_tools",
    "max_turns",
    "max_tool_calls",
    "require_tool_reasoning",
    "tool_result_max_length",
    "max_message_chars",
    "tool_result_keep_recent",
    "tool_result_context_preview_chars",
    "active_artifact_context_enabled",
    "active_artifact_context_max_handles",
    "active_artifact_context_max_chars",
    "enforce_tool_contracts",
    "progressive_tool_disclosure",
    "suppress_control_loop_calls",
    "tool_contracts",
    "initial_artifacts",
    "initial_bindings",
    "forced_final_max_attempts",
    "forced_final_circuit_breaker_threshold",
    "force_submit_retry_on_max_tool_calls",
    "accept_forced_answer_on_max_tool_calls",
    "finalization_fallback_models",
    "retrieval_stagnation_turns",
    "retrieval_stagnation_action",
    "adoption_profile",
    "adoption_profile_enforce",
})


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _is_retrieval_no_hits_result(
    *,
    tool_name: str,
    result_text: str | None,
) -> bool:
    """Best-effort retrieval no-hit detector for taxonomy accounting."""
    if not result_text or _is_budget_exempt_tool(tool_name):
        return False

    lower = result_text.lower()
    if "no results" in lower or "not found" in lower or "\"status\": \"no_results\"" in lower:
        return True

    try:
        payload = _json.loads(result_text)
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False

    for key in ("results", "entities", "chunks", "relationships", "neighbors", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return len(value) == 0

    for key in ("count", "total", "hits"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return int(value) == 0
    return False


def _is_responses_api_raw_response(raw_response: Any) -> bool:
    """Best-effort check for Responses API objects (vs chat completions)."""
    if raw_response is None:
        return False
    has_output = hasattr(raw_response, "output")
    has_choices = hasattr(raw_response, "choices")
    return bool(has_output and not has_choices)


async def _agent_loop(
    model: str,
    messages: list[dict[str, Any]],
    openai_tools: list[dict[str, Any]],
    agent_result: MCPAgentResult,
    executor: Any,  # Callable[[list, int], Awaitable[tuple[list[MCPToolCallRecord], list[dict]]]]
    max_turns: int,
    max_tool_calls: int | None,
    require_tool_reasoning: bool,
    tool_result_max_length: int,
    max_message_chars: int = DEFAULT_MAX_MESSAGE_CHARS,
    enforce_tool_contracts: bool = DEFAULT_ENFORCE_TOOL_CONTRACTS,
    progressive_tool_disclosure: bool = DEFAULT_PROGRESSIVE_TOOL_DISCLOSURE,
    suppress_control_loop_calls: bool = DEFAULT_SUPPRESS_CONTROL_LOOP_CALLS,
    tool_contracts: dict[str, dict[str, Any]] | None = None,
    initial_artifacts: list[str] | tuple[str, ...] | None = DEFAULT_INITIAL_ARTIFACTS,
    initial_bindings: dict[str, Any] | None = None,
    timeout: int = 60,
    kwargs: dict[str, Any] | None = None,
    error_budget: "AgentErrorBudget | None" = None,
) -> tuple[str, str]:
    """Core agent loop shared by MCP, direct-tool, and session-pool paths.

    Args:
        executor: async callable (tool_calls, max_result_length) -> (records, tool_messages).
            For MCP: wraps _execute_tool_calls with bound sessions.
            For direct tools: wraps execute_direct_tool_calls with bound tool_map.
        error_budget: Aggregate error budget for this agent invocation.
            Controls total retry effort across all models. Consumers should
            declare explicitly. When None, a generous default is used.

    Returns (final_content, final_finish_reason).
    """
    from llm_client.agent import mcp_agent as _mcp_agent

    _build_agent_usage = _mcp_agent._build_agent_usage
    _foundation_schema_strict_enabled = _mcp_agent._foundation_schema_strict_enabled
    _inner_acall_llm = _mcp_agent._inner_acall_llm
    _is_responses_api_raw_response = _mcp_agent._is_responses_api_raw_response
    validate_foundation_event = _mcp_agent.validate_foundation_event

    kwargs = dict(kwargs or {})

    total_cost = 0.0
    final_content = ""
    final_finish_reason = "stop"
    effective_model = model
    attempted_models: list[str] = []
    sticky_fallback = False
    tool_call_counts: dict[str, int] = {}
    tool_loop_nudges: dict[str, int] = {}
    tool_error_counts: dict[tuple[str, str], int] = {}
    tool_error_nudges: dict[tuple[str, str], int] = {}
    submit_requires_new_evidence = False
    control_loop_suppressed_calls = 0
    last_budget_remaining: int | None = None
    rejected_missing_reasoning_calls = 0
    tool_call_turns_total = 0
    tool_call_empty_text_turns = 0
    responses_tool_call_empty_text_turns = 0
    tool_arg_coercions = 0
    tool_arg_coercion_calls = 0
    tool_arg_validation_rejections = 0
    tool_disclosure_turns = 0
    tool_disclosure_hidden_total = 0
    tool_disclosure_unavailable_msgs = 0
    tool_disclosure_unavailable_reason_chars = 0
    tool_disclosure_unavailable_reason_tokens_est = 0
    tool_disclosure_repair_suggestions = 0
    no_legal_noncontrol_turns = 0
    deficit_no_progress_streak = 0
    max_deficit_no_progress_streak = 0
    deficit_no_progress_nudges = 0
    deficit_no_progress_last_nudged = 0
    prev_turn_deficit_digest: str | None = None
    prev_turn_had_evidence_tools = False
    zero_exec_tool_turn_streak = 0
    max_zero_exec_tool_turn_streak = 0
    plain_text_no_tool_turn_streak = 0
    context_compactions = 0
    context_compacted_messages = 0
    context_compacted_chars = 0
    context_tool_result_clearings = 0
    context_tool_results_cleared = 0
    context_tool_result_cleared_chars = 0
    tool_result_metadata_by_id: dict[str, dict[str, Any]] = {}
    artifact_context_message_index: int | None = None
    artifact_context_updates = 0
    artifact_context_chars = 0
    gate_rejected_calls = 0
    gate_violation_events: list[dict[str, Any]] = []
    contract_rejected_calls = 0
    contract_violation_events: list[dict[str, Any]] = []
    failure_event_codes: list[str] = []

    # Error budget tracking
    _error_budget = error_budget if error_budget is not None else AgentErrorBudget()
    _budget_state = ErrorBudgetState(budget=_error_budget)
    handle_input_resolution_count = 0
    handle_input_resolved_artifact_count = 0
    autofilled_tool_reasoning_calls = 0
    autofilled_tool_reasoning_by_tool: dict[str, int] = {}
    submit_validation_reason_counts: dict[str, int] = {}
    evidence_pointer_labels: set[str] = set()
    submit_evidence_digest_at_last_failure: str | None = None
    evidence_digest_change_count = 0
    evidence_turns_total = 0
    evidence_turns_with_new_evidence = 0
    evidence_turns_without_new_evidence = 0
    evidence_pointer_count = 0
    retrieval_no_hits_count = 0
    tool_state = _initialize_agent_tool_state(
        openai_tools=openai_tools,
        tool_contracts=tool_contracts,
        initial_artifacts=initial_artifacts,
        initial_bindings=initial_bindings,
        kwargs=kwargs,
        enforce_tool_contracts=enforce_tool_contracts,
        warning_sink=agent_result.warnings,
    )
    normalized_tool_contracts = tool_state.normalized_tool_contracts
    tool_parameter_index = tool_state.tool_parameter_index
    available_artifacts = tool_state.available_artifacts
    initial_artifact_snapshot = tool_state.initial_artifact_snapshot
    available_capabilities = tool_state.available_capabilities
    initial_capability_snapshot = tool_state.initial_capability_snapshot
    available_bindings = tool_state.available_bindings
    initial_binding_snapshot = tool_state.initial_binding_snapshot
    lane_closure_analysis = tool_state.lane_closure_analysis
    artifact_timeline = tool_state.artifact_timeline
    requires_submit_answer = tool_state.requires_submit_answer
    runtime_artifact_registry_by_id: dict[str, dict[str, Any]] = {}
    runtime_artifact_read_tool = _runtime_artifact_read_tool_def()
    runtime_artifact_tool_name = str(
        runtime_artifact_read_tool.get("function", {}).get("name", "")
    ).strip() or RUNTIME_ARTIFACT_READ_TOOL_NAME
    if any(
        isinstance(tool_def, dict)
        and str(tool_def.get("function", {}).get("name", "")).strip() == runtime_artifact_tool_name
        for tool_def in openai_tools
    ):
        raise ValueError(
            f"Reserved runtime tool name collision: {runtime_artifact_tool_name}"
        )
    tool_parameter_index.update(build_tool_parameter_index([runtime_artifact_read_tool]))
    normalized_tool_contracts = dict(normalized_tool_contracts)
    normalized_tool_contracts.update(
        _normalize_tool_contracts(
            {runtime_artifact_tool_name: _runtime_artifact_read_contract()}
        )
    )
    submit_answer_succeeded = False
    submit_answer_call_count = 0
    submitted_answer_value: str | None = None
    fallback_submit_guess_value: str | None = None
    _last_todo_status_line: str | None = None
    force_final_reason: str | None = None


    trace_id = str(kwargs.get("trace_id", "")).strip() or None
    task = str(kwargs.get("task", "")).strip() or None
    foundation_session_id = new_session_id()
    foundation_actor_id = "agent:mcp_loop:default:1"
    foundation_events: list[dict[str, Any]] = []
    foundation_event_types: dict[str, int] = {}
    foundation_event_validation_errors = 0
    foundation_events_logged = 0
    _io_log: Any | None = None
    try:
        import llm_client.io_log as _io_log_module  # local import to avoid module-cycle hazards
        _io_log = _io_log_module
        active_run_id = _io_log_module.get_active_experiment_run_id()
    except Exception:
        active_run_id = None
    foundation_run_id = coerce_run_id(
        active_run_id if isinstance(active_run_id, str) else None,
        trace_id,
    )
    runtime_policy = _resolve_agent_runtime_policy(
        model=model,
        max_turns=max_turns,
        max_tool_calls=max_tool_calls,
        require_tool_reasoning=require_tool_reasoning,
        enforce_tool_contracts=enforce_tool_contracts,
        progressive_tool_disclosure=progressive_tool_disclosure,
        suppress_control_loop_calls=suppress_control_loop_calls,
        tool_result_max_length=tool_result_max_length,
        max_message_chars=max_message_chars,
        kwargs=kwargs,
        warning_sink=agent_result.warnings,
    )
    run_config_spec = runtime_policy.run_config_spec
    run_config_hash = runtime_policy.run_config_hash
    forced_final_max_attempts = runtime_policy.forced_final_max_attempts
    forced_final_circuit_breaker_threshold = runtime_policy.forced_final_circuit_breaker_threshold
    forced_final_breaker_effective = runtime_policy.forced_final_breaker_effective
    force_submit_retry_on_max_tool_calls = runtime_policy.force_submit_retry_on_max_tool_calls
    accept_forced_answer_on_max_tool_calls = runtime_policy.accept_forced_answer_on_max_tool_calls
    finalization_fallback_models = runtime_policy.finalization_fallback_models
    retrieval_stagnation_turns = runtime_policy.retrieval_stagnation_turns
    retrieval_stagnation_action = runtime_policy.retrieval_stagnation_action
    tool_result_keep_recent = runtime_policy.tool_result_keep_recent
    tool_result_context_preview_chars = runtime_policy.tool_result_context_preview_chars
    active_artifact_context_enabled = runtime_policy.active_artifact_context_enabled
    active_artifact_context_max_handles = runtime_policy.active_artifact_context_max_handles
    active_artifact_context_max_chars = runtime_policy.active_artifact_context_max_chars
    adoption_profile = runtime_policy.adoption_profile
    adoption_profile_enforce = runtime_policy.adoption_profile_enforce

    adoption_assessment = assess_adoption_profile(
        requested_profile=adoption_profile,
        enforce=adoption_profile_enforce,
        openai_tools=openai_tools,
        normalized_tool_contracts=normalized_tool_contracts,
        require_tool_reasoning=require_tool_reasoning,
        enforce_tool_contracts=enforce_tool_contracts,
        progressive_tool_disclosure=progressive_tool_disclosure,
        lane_closure_analysis=lane_closure_analysis,
        tool_reasoning_field=TOOL_REASONING_FIELD,
    )
    if adoption_assessment.violations:
        violation_msg = (
            f"ADOPTION_PROFILE_VIOLATION[{adoption_assessment.effective_profile}]: "
            + "; ".join(adoption_assessment.violations)
        )
        if adoption_assessment.enforce:
            raise ValueError(violation_msg)
        agent_result.warnings.append(violation_msg)
        logger.warning(violation_msg)

    forced_final_attempts = 0
    forced_final_circuit_breaker_opened = False
    retrieval_stagnation_streak = 0
    retrieval_stagnation_streak_max = 0
    retrieval_stagnation_alerted_for_current_streak = False
    retrieval_stagnation_triggered = False
    retrieval_stagnation_turn: int | None = None
    finalization_events: list[str] = []
    finalization_fallback_used = False
    finalization_fallback_succeeded = False
    finalization_fallback_attempts: list[dict[str, Any]] = []
    finalization_primary_model: str | None = None

    def _emit_foundation_event(payload: dict[str, Any]) -> None:
        nonlocal foundation_event_validation_errors, foundation_events_logged
        strict_foundation_schema = _foundation_schema_strict_enabled()
        try:
            validated = validate_foundation_event(payload)
        except Exception as exc:
            foundation_event_validation_errors += 1
            warning = f"FOUNDATION_EVENT_INVALID: {type(exc).__name__}: {exc}"
            agent_result.warnings.append(warning)
            logger.warning(warning)
            if strict_foundation_schema:
                raise RuntimeError(warning) from exc
            return

        foundation_events.append(validated)
        event_type = str(validated.get("event_type", "unknown"))
        foundation_event_types[event_type] = foundation_event_types.get(event_type, 0) + 1

        if _io_log is None:
            return
        try:
            _io_log.log_foundation_event(
                event=validated,
                caller="llm_client.agent.mcp_agent",
                task=task,
                trace_id=trace_id,
            )
            foundation_events_logged += 1
        except Exception as exc:
            warning = f"FOUNDATION_EVENT_LOG_FAILED: {type(exc).__name__}: {exc}"
            agent_result.warnings.append(warning)
            logger.warning(warning)

    if (
        "max_tokens" not in kwargs
        and "max_completion_tokens" not in kwargs
    ):
        # Keep a higher default cap for long tool-calling traces to avoid
        # non-retryable finish_reason='length' truncation on provider lanes.
        # Override via env when needed per deployment/benchmark.
        max_completion_default = 8192
        try:
            max_completion_default = int(
                os.getenv("LLM_CLIENT_MCP_MAX_COMPLETION_TOKENS", str(max_completion_default))
            )
        except Exception:
            max_completion_default = 8192
        kwargs = dict(kwargs)
        kwargs["max_completion_tokens"] = max(1024, max_completion_default)

    for turn in range(max_turns):
        if max_tool_calls is not None:
            budgeted_calls_used = _count_budgeted_records(agent_result.tool_calls)
            remaining_tool_calls = max_tool_calls - budgeted_calls_used
            if remaining_tool_calls <= 0:
                force_final_reason = "max_tool_calls"
                logger.warning(
                    "Agent loop exhausted max_tool_calls=%d after %d turns (%d budgeted, %d total tool calls); forcing final answer",
                    max_tool_calls,
                    turn,
                    budgeted_calls_used,
                    len(agent_result.tool_calls),
                )
                break
            if remaining_tool_calls != last_budget_remaining:
                budget_msg = {
                    "role": "user",
                    "content": (
                        f"[SYSTEM: Retrieval-tool budget: {budgeted_calls_used}/{max_tool_calls} used, "
                        f"{remaining_tool_calls} remaining. Plan carefully. "
                        f"Budget-exempt tools: {', '.join(sorted(BUDGET_EXEMPT_TOOL_NAMES))}. "
                        "If you already have enough evidence, submit your answer now.]"
                    ),
                }
                messages.append(budget_msg)
                agent_result.conversation_trace.append(budget_msg)
                last_budget_remaining = remaining_tool_calls

        agent_result.turns = turn + 1
        turn_model = await _run_turn_model_stage(
            turn=turn,
            model=model,
            effective_model=effective_model,
            attempted_models=attempted_models,
            sticky_fallback=sticky_fallback,
            messages=messages,
            openai_tools=openai_tools,
            runtime_artifact_registry_by_id=runtime_artifact_registry_by_id,
            runtime_artifact_read_tool=runtime_artifact_read_tool,
            tool_result_metadata_by_id=tool_result_metadata_by_id,
            agent_result=agent_result,
            max_turns=max_turns,
            max_tool_calls=max_tool_calls,
            max_message_chars=max_message_chars,
            tool_result_keep_recent=tool_result_keep_recent,
            tool_result_context_preview_chars=tool_result_context_preview_chars,
            progressive_tool_disclosure=progressive_tool_disclosure,
            normalized_tool_contracts=normalized_tool_contracts,
            available_artifacts=available_artifacts,
            available_capabilities=available_capabilities,
            available_bindings=available_bindings,
            prev_turn_deficit_digest=prev_turn_deficit_digest,
            prev_turn_had_evidence_tools=prev_turn_had_evidence_tools,
            deficit_no_progress_streak=deficit_no_progress_streak,
            max_deficit_no_progress_streak=max_deficit_no_progress_streak,
            deficit_no_progress_last_nudged=deficit_no_progress_last_nudged,
            plain_text_no_tool_turn_streak=plain_text_no_tool_turn_streak,
            require_tool_reasoning=require_tool_reasoning,
            timeout=timeout,
            kwargs=kwargs,
            foundation_run_id=foundation_run_id,
            foundation_session_id=foundation_session_id,
            foundation_actor_id=foundation_actor_id,
            emit_foundation_event=_emit_foundation_event,
            inner_acall_llm=_inner_acall_llm,
            is_responses_api_raw_response=_is_responses_api_raw_response,
            artifact_context_message_index=artifact_context_message_index,
            active_artifact_context_enabled=active_artifact_context_enabled,
            active_artifact_context_max_handles=active_artifact_context_max_handles,
            active_artifact_context_max_chars=active_artifact_context_max_chars,
            requires_submit_answer=requires_submit_answer,
            submit_answer_succeeded=submit_answer_succeeded,
            current_tool_call_count=len(agent_result.tool_calls),
            turn_warning_threshold=TURN_WARNING_THRESHOLD,
            disclosure_token_chars=DEFAULT_TOOL_DISCLOSURE_TOKEN_CHARS,
            event_code_no_legal_noncontrol_tools=EVENT_CODE_NO_LEGAL_NONCONTROL_TOOLS,
            event_code_provider_empty=EVENT_CODE_PROVIDER_EMPTY,
        )
        artifact_context_message_index = turn_model.artifact_context_message_index
        context_tool_result_clearings += (
            turn_model.context_tool_result_clearings_delta
        )
        context_tool_results_cleared += (
            turn_model.context_tool_results_cleared_delta
        )
        context_tool_result_cleared_chars += (
            turn_model.context_tool_result_cleared_chars_delta
        )
        context_compactions += turn_model.context_compactions_delta
        context_compacted_messages += (
            turn_model.context_compacted_messages_delta
        )
        context_compacted_chars += turn_model.context_compacted_chars_delta
        artifact_context_updates += turn_model.artifact_context_updates_delta
        artifact_context_chars += turn_model.artifact_context_chars_delta
        tool_disclosure_turns += turn_model.tool_disclosure_turns_delta
        tool_disclosure_hidden_total += (
            turn_model.tool_disclosure_hidden_total_delta
        )
        tool_disclosure_unavailable_msgs += (
            turn_model.tool_disclosure_unavailable_msgs_delta
        )
        tool_disclosure_unavailable_reason_chars += (
            turn_model.tool_disclosure_unavailable_reason_chars_delta
        )
        tool_disclosure_unavailable_reason_tokens_est += (
            turn_model.tool_disclosure_unavailable_reason_tokens_est_delta
        )
        tool_disclosure_repair_suggestions += (
            turn_model.tool_disclosure_repair_suggestions_delta
        )
        no_legal_noncontrol_turns += turn_model.no_legal_noncontrol_turns_delta
        deficit_no_progress_streak = turn_model.deficit_no_progress_streak
        max_deficit_no_progress_streak = (
            turn_model.max_deficit_no_progress_streak
        )
        deficit_no_progress_nudges += (
            turn_model.deficit_no_progress_nudges_delta
        )
        deficit_no_progress_last_nudged = (
            turn_model.deficit_no_progress_last_nudged
        )
        current_turn_deficit_digest = turn_model.current_turn_deficit_digest
        disclosure_repair_hints = turn_model.disclosure_repair_hints
        effective_model = turn_model.effective_model
        sticky_fallback = turn_model.sticky_fallback
        total_cost += turn_model.total_cost_delta
        agent_result.total_input_tokens += turn_model.total_input_tokens_delta
        agent_result.total_output_tokens += turn_model.total_output_tokens_delta
        agent_result.total_cached_tokens += turn_model.total_cached_tokens_delta
        agent_result.total_cache_creation_tokens += (
            turn_model.total_cache_creation_tokens_delta
        )
        final_content = turn_model.final_content
        final_finish_reason = turn_model.final_finish_reason
        failure_event_codes.extend(turn_model.failure_event_codes)
        plain_text_no_tool_turn_streak = (
            turn_model.plain_text_no_tool_turn_streak
        )
        tool_calls_this_turn = turn_model.tool_calls_this_turn
        tool_call_turns_total += turn_model.tool_call_turns_total_delta
        tool_call_empty_text_turns += (
            turn_model.tool_call_empty_text_turns_delta
        )
        responses_tool_call_empty_text_turns += (
            turn_model.responses_tool_call_empty_text_turns_delta
        )
        autofilled_tool_reasoning_calls += (
            turn_model.autofilled_tool_reasoning_calls_delta
        )
        for tool_name, delta in (
            turn_model.autofilled_tool_reasoning_by_tool_delta.items()
        ):
            autofilled_tool_reasoning_by_tool[tool_name] = (
                autofilled_tool_reasoning_by_tool.get(tool_name, 0) + delta
            )
        if turn_model.should_continue_turn:
            continue
        if turn_model.should_break_loop:
            break

        tool_processing = await _process_tool_calls_turn(
            turn=turn,
            tool_calls_this_turn=tool_calls_this_turn,
            executor=executor,
            tool_result_max_length=tool_result_max_length,
            require_tool_reasoning=require_tool_reasoning,
            enforce_tool_contracts=enforce_tool_contracts,
            suppress_control_loop_calls=suppress_control_loop_calls,
            normalized_tool_contracts=normalized_tool_contracts,
            tool_parameter_index=tool_parameter_index,
            available_artifacts=available_artifacts,
            available_capabilities=available_capabilities,
            available_bindings=available_bindings,
            runtime_artifact_registry_by_id=runtime_artifact_registry_by_id,
            runtime_artifact_tool_name=runtime_artifact_tool_name,
            tool_result_metadata_by_id=tool_result_metadata_by_id,
            artifact_timeline=artifact_timeline,
            retrieval_no_hits_detector=_is_retrieval_no_hits_result,
            submit_requires_new_evidence=submit_requires_new_evidence,
            submit_evidence_digest_at_last_failure=submit_evidence_digest_at_last_failure,
            evidence_pointer_labels=evidence_pointer_labels,
            foundation_run_id=foundation_run_id,
            foundation_session_id=foundation_session_id,
            foundation_actor_id=foundation_actor_id,
            emit_foundation_event=_emit_foundation_event,
        )
        tool_calls_to_execute = tool_processing.tool_calls_to_execute
        records = tool_processing.records
        tool_messages = tool_processing.tool_messages
        executed_records = tool_processing.executed_records
        gate_rejected_calls += tool_processing.gate_rejected_calls_delta
        rejected_missing_reasoning_calls += (
            tool_processing.rejected_missing_reasoning_calls_delta
        )
        gate_violation_events.extend(tool_processing.gate_violation_events)
        contract_rejected_calls += tool_processing.contract_rejected_calls_delta
        contract_violation_events.extend(tool_processing.contract_violation_events)
        control_loop_suppressed_calls += (
            tool_processing.control_loop_suppressed_calls_delta
        )
        failure_event_codes.extend(tool_processing.failure_event_codes)
        handle_input_resolution_count += (
            tool_processing.handle_input_resolution_count_delta
        )
        handle_input_resolved_artifact_count += (
            tool_processing.handle_input_resolved_artifact_count_delta
        )
        retrieval_no_hits_count += tool_processing.retrieval_no_hits_count_delta
        submit_requires_new_evidence = tool_processing.submit_requires_new_evidence
        submit_evidence_digest_at_last_failure = (
            tool_processing.submit_evidence_digest_at_last_failure
        )
        contract_rejected_record_count = (
            tool_processing.contract_rejected_record_count
        )
        suppressed_record_count = tool_processing.suppressed_record_count
        agent_result.tool_calls.extend(records)
        messages.extend(tool_messages)
        if tool_processing.pending_gate_msg is not None:
            messages.append(tool_processing.pending_gate_msg)
            agent_result.conversation_trace.append(tool_processing.pending_gate_msg)
        if tool_processing.pending_contract_msg is not None:
            messages.append(tool_processing.pending_contract_msg)
            agent_result.conversation_trace.append(tool_processing.pending_contract_msg)
        if tool_processing.pending_control_loop_msg is not None:
            messages.append(tool_processing.pending_control_loop_msg)
            agent_result.conversation_trace.append(
                tool_processing.pending_control_loop_msg
            )

        turn_outcome = _process_turn_outcomes(
            turn=turn,
            max_turns=max_turns,
            records=records,
            executed_records=executed_records,
            tool_calls_this_turn=tool_calls_this_turn,
            tool_calls_to_execute=tool_calls_to_execute,
            evidence_pointer_labels=evidence_pointer_labels,
            submit_requires_new_evidence=submit_requires_new_evidence,
            submit_evidence_digest_at_last_failure=submit_evidence_digest_at_last_failure,
            current_turn_deficit_digest=current_turn_deficit_digest,
            retrieval_stagnation_streak=retrieval_stagnation_streak,
            retrieval_stagnation_streak_max=retrieval_stagnation_streak_max,
            retrieval_stagnation_alerted_for_current_streak=(
                retrieval_stagnation_alerted_for_current_streak
            ),
            retrieval_stagnation_turns=retrieval_stagnation_turns,
            retrieval_stagnation_action=retrieval_stagnation_action,
            retrieval_stagnation_turn=retrieval_stagnation_turn,
            zero_exec_tool_turn_streak=zero_exec_tool_turn_streak,
            max_zero_exec_tool_turn_streak=max_zero_exec_tool_turn_streak,
            last_todo_status_line=_last_todo_status_line,
            submit_validation_reason_counts=submit_validation_reason_counts,
            contract_rejected_record_count=contract_rejected_record_count,
            suppressed_record_count=suppressed_record_count,
            available_artifacts=available_artifacts,
            available_bindings=available_bindings,
            foundation_run_id=foundation_run_id,
            foundation_session_id=foundation_session_id,
            foundation_actor_id=foundation_actor_id,
            emit_foundation_event=_emit_foundation_event,
            event_code_retrieval_stagnation=EVENT_CODE_RETRIEVAL_STAGNATION,
            event_code_retrieval_stagnation_observed=(
                EVENT_CODE_RETRIEVAL_STAGNATION_OBSERVED
            ),
            event_code_control_churn_threshold=EVENT_CODE_CONTROL_CHURN_THRESHOLD,
            control_churn_turn_threshold=DEFAULT_TOOL_CALL_STALL_TURNS,
        )
        tool_arg_coercions += turn_outcome.tool_arg_coercions_delta
        tool_arg_coercion_calls += turn_outcome.tool_arg_coercion_calls_delta
        tool_arg_validation_rejections += (
            turn_outcome.tool_arg_validation_rejections_delta
        )
        submit_answer_call_count += turn_outcome.submit_answer_call_count_delta
        if turn_outcome.submit_answer_succeeded_now:
            submit_answer_succeeded = True
        if turn_outcome.submitted_answer_value is not None:
            submitted_answer_value = turn_outcome.submitted_answer_value
        if turn_outcome.fallback_submit_guess_value is not None:
            fallback_submit_guess_value = turn_outcome.fallback_submit_guess_value
        submit_requires_new_evidence = turn_outcome.submit_requires_new_evidence
        submit_evidence_digest_at_last_failure = (
            turn_outcome.submit_evidence_digest_at_last_failure
        )
        evidence_pointer_count = turn_outcome.evidence_pointer_count
        evidence_digest_change_count += (
            turn_outcome.evidence_digest_change_count_delta
        )
        evidence_turns_total += turn_outcome.evidence_turns_total_delta
        evidence_turns_with_new_evidence += (
            turn_outcome.evidence_turns_with_new_evidence_delta
        )
        evidence_turns_without_new_evidence += (
            turn_outcome.evidence_turns_without_new_evidence_delta
        )
        retrieval_stagnation_streak = turn_outcome.retrieval_stagnation_streak
        retrieval_stagnation_streak_max = (
            turn_outcome.retrieval_stagnation_streak_max
        )
        retrieval_stagnation_alerted_for_current_streak = (
            turn_outcome.retrieval_stagnation_alerted_for_current_streak
        )
        retrieval_stagnation_triggered = (
            retrieval_stagnation_triggered or turn_outcome.retrieval_stagnation_triggered
        )
        retrieval_stagnation_turn = turn_outcome.retrieval_stagnation_turn
        prev_turn_had_evidence_tools = turn_outcome.prev_turn_had_evidence_tools
        prev_turn_deficit_digest = turn_outcome.prev_turn_deficit_digest
        zero_exec_tool_turn_streak = turn_outcome.zero_exec_tool_turn_streak
        max_zero_exec_tool_turn_streak = (
            turn_outcome.max_zero_exec_tool_turn_streak
        )
        _last_todo_status_line = turn_outcome.last_todo_status_line
        agent_result.warnings.extend(turn_outcome.warnings)
        failure_event_codes.extend(turn_outcome.failure_event_codes)
        for emitted_message in turn_outcome.emitted_messages:
            messages.append(emitted_message)
            agent_result.conversation_trace.append(emitted_message)
        if turn_outcome.force_final_reason is not None:
            force_final_reason = turn_outcome.force_final_reason
        if turn_outcome.stop_agent_loop:
            break

        # Repeated tool-error detection: nudge strategy change instead of
        # repeatedly retrying near-identical invalid calls.
        for record in records:
            if not record.error:
                continue
            sig = _tool_error_signature(str(record.error))
            error_key = (record.tool, sig)
            count = tool_error_counts.get(error_key, 0) + 1
            tool_error_counts[error_key] = count

            last_nudged = tool_error_nudges.get(error_key, 0)
            if count >= 2 and count > last_nudged:
                err_msg = {
                    "role": "user",
                    "content": (
                        f"[SYSTEM: Repeated tool error from '{record.tool}': {sig}. "
                        "STOP retrying near-identical calls. Change strategy "
                        "(new evidence, alternate hypothesis, or upstream repair) before continuing.]"
                    ),
                }
                messages.append(err_msg)
                agent_result.conversation_trace.append(err_msg)
                tool_error_nudges[error_key] = count
                logger.warning(
                    "Loop detection: repeated tool error from '%s' (count=%d): %s",
                    record.tool, count, sig,
                )

        # Track tool call frequency for loop detection
        for tc in tool_calls_to_execute:
            tc_name = tc.get("function", {}).get("name", "")
            if tc_name and not _is_budget_exempt_tool(tc_name):
                tool_call_counts[tc_name] = tool_call_counts.get(tc_name, 0) + 1

        # Loop detection: if same tool called 3+ times, nudge model to switch
        for tool_name, count in tool_call_counts.items():
            last_nudged = tool_loop_nudges.get(tool_name, 0)
            if count >= 3 and count % 3 == 0 and count > last_nudged:
                loop_hint = ""
                if disclosure_repair_hints:
                    suggested = [
                        name
                        for name in disclosure_repair_hints
                        if name != tool_name and not _is_budget_exempt_tool(name)
                    ][:3]
                    if suggested:
                        loop_hint = "Try these alternatives now: " + ", ".join(suggested) + ". "
                loop_msg = {
                    "role": "user",
                    "content": (
                        f"[SYSTEM: You have called '{tool_name}' {count} times. "
                        "STOP repeating the same searches. Read what you already have. "
                        + loop_hint
                        + "If still blocked, submit your answer now.]"
                    ),
                }
                messages.append(loop_msg)
                agent_result.conversation_trace.append(loop_msg)
                tool_loop_nudges[tool_name] = count
                logger.warning(
                    "Loop detection: tool '%s' called %d times on turn %d",
                    tool_name, count, turn + 1,
                )

        # Capture tool results in trace
        for tmsg in tool_messages:
            agent_result.conversation_trace.append({
                "role": "tool",
                "tool_call_id": tmsg.get("tool_call_id", ""),
                "content": tmsg.get("content", ""),
            })

        if submit_answer_succeeded:
            final_content = submitted_answer_value or (result.content or "").strip() or "submitted"
            final_finish_reason = "submitted"
            logger.info(
                "Agent loop: submit_answer succeeded on turn %d/%d",
                turn + 1, max_turns,
            )
            break

        # --- Error budget tracking ---
        turn_had_errors = any(r.error for r in records)
        if turn_had_errors:
            for r in records:
                if r.error:
                    _budget_state.record_error(effective_model, r.error)
        else:
            _budget_state.record_success(effective_model)

        budget_stop, budget_reason = _budget_state.should_stop()
        if budget_stop:
            logger.warning(
                "Agent error budget exhausted: %s (turns=%d, errors=%d)",
                budget_reason, _budget_state.total_turns, _budget_state.total_errors,
            )
            agent_result.warnings.append(f"error_budget_exhausted: {budget_reason}")
            agent_result.metadata["error_budget"] = _budget_state.summary()
            break

        # Always include budget summary in metadata for observability
        agent_result.metadata["error_budget"] = _budget_state.summary()

        logger.debug(
            "Agent turn %d/%d: %d tool calls (budget: %d/%d turns, %d/%d errors)",
            turn + 1, max_turns, len(tool_calls_this_turn),
            _budget_state.total_turns, _error_budget.max_agent_turns,
            _budget_state.total_errors, _error_budget.max_total_errors,
        )

        # Turn countdown: warn agent when it's running low on turns
        remaining = max_turns - (turn + 1)
        if remaining == TURN_WARNING_THRESHOLD:
            countdown_msg = {
                "role": "user",
                "content": (
                    f"[SYSTEM: You have {remaining} turns remaining. "
                    "Submit your best answer now based on the evidence you have. "
                    "Do not search for more information.]"
                ),
            }
            messages.append(countdown_msg)
            agent_result.conversation_trace.append(countdown_msg)
    else:
        force_final_reason = "max_turns"

    turn_completion = await _resolve_turn_completion(
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
        emit_foundation_event=_emit_foundation_event,
        final_content=final_content,
        final_finish_reason=final_finish_reason,
        finalization_primary_model=finalization_primary_model,
        forced_final_attempts=forced_final_attempts,
        forced_final_circuit_breaker_opened=forced_final_circuit_breaker_opened,
        finalization_fallback_used=finalization_fallback_used,
        finalization_fallback_succeeded=finalization_fallback_succeeded,
        finalization_events=finalization_events,
        finalization_fallback_attempts=finalization_fallback_attempts,
        requires_submit_answer=requires_submit_answer,
        submit_answer_call_count=submit_answer_call_count,
        submit_answer_succeeded=submit_answer_succeeded,
        submitted_answer_value=submitted_answer_value,
        force_submit_retry_on_max_tool_calls=force_submit_retry_on_max_tool_calls,
        accept_forced_answer_on_max_tool_calls=accept_forced_answer_on_max_tool_calls,
        fallback_submit_guess_value=fallback_submit_guess_value,
        event_code_submit_forced_accept_budget_exhaustion=(
            EVENT_CODE_SUBMIT_FORCED_ACCEPT_BUDGET_EXHAUSTION
        ),
        event_code_submit_forced_accept_turn_exhaustion=(
            EVENT_CODE_SUBMIT_FORCED_ACCEPT_TURN_EXHAUSTION
        ),
        event_code_submit_forced_accept_forced_final=(
            EVENT_CODE_SUBMIT_FORCED_ACCEPT_FORCED_FINAL
        ),
        event_code_required_submit_not_attempted=(
            EVENT_CODE_REQUIRED_SUBMIT_NOT_ATTEMPTED
        ),
        event_code_required_submit_not_accepted=(
            EVENT_CODE_REQUIRED_SUBMIT_NOT_ACCEPTED
        ),
    )
    final_content = turn_completion.final_content
    final_finish_reason = turn_completion.final_finish_reason
    finalization_primary_model = turn_completion.finalization_primary_model
    forced_final_attempts = turn_completion.forced_final_attempts
    forced_final_circuit_breaker_opened = (
        turn_completion.forced_final_circuit_breaker_opened
    )
    finalization_fallback_used = turn_completion.finalization_fallback_used
    finalization_fallback_succeeded = (
        turn_completion.finalization_fallback_succeeded
    )
    finalization_events = list(turn_completion.finalization_events)
    finalization_fallback_attempts = list(
        turn_completion.finalization_fallback_attempts
    )
    failure_event_codes.extend(turn_completion.failure_event_codes)
    context_tool_result_clearings += (
        turn_completion.context_tool_result_clearings_delta
    )
    context_tool_results_cleared += (
        turn_completion.context_tool_results_cleared_delta
    )
    context_tool_result_cleared_chars += (
        turn_completion.context_tool_result_cleared_chars_delta
    )
    context_compactions += turn_completion.context_compactions_delta
    context_compacted_messages += (
        turn_completion.context_compacted_messages_delta
    )
    context_compacted_chars += turn_completion.context_compacted_chars_delta
    total_cost += turn_completion.total_cost_delta
    agent_result.total_input_tokens += turn_completion.total_input_tokens_delta
    agent_result.total_output_tokens += turn_completion.total_output_tokens_delta
    agent_result.total_cached_tokens += turn_completion.total_cached_tokens_delta
    agent_result.total_cache_creation_tokens += (
        turn_completion.total_cache_creation_tokens_delta
    )
    agent_result.turns += turn_completion.turns_delta
    required_submit_missing = turn_completion.required_submit_missing
    submit_forced_retry_on_budget_exhaustion = (
        turn_completion.submit_forced_retry_on_budget_exhaustion
    )
    submit_forced_accept_on_budget_exhaustion = (
        turn_completion.submit_forced_accept_on_budget_exhaustion
    )
    submit_answer_call_count += turn_completion.submit_answer_call_count_delta
    submit_answer_succeeded = turn_completion.submit_answer_succeeded
    submitted_answer_value = turn_completion.submitted_answer_value
    agent_result.warnings.extend(turn_completion.warnings)

    _apply_agent_loop_summary(
        agent_result=agent_result,
        model=model,
        effective_model=effective_model,
        attempted_models=attempted_models,
        sticky_fallback=sticky_fallback,
        total_cost=total_cost,
        max_turns=max_turns,
        max_tool_calls=max_tool_calls,
        forced_final_attempts=forced_final_attempts,
        forced_final_max_attempts=forced_final_max_attempts,
        forced_final_circuit_breaker_threshold=forced_final_circuit_breaker_threshold,
        forced_final_breaker_effective=forced_final_breaker_effective,
        force_submit_retry_on_max_tool_calls=force_submit_retry_on_max_tool_calls,
        accept_forced_answer_on_max_tool_calls=accept_forced_answer_on_max_tool_calls,
        forced_final_circuit_breaker_opened=forced_final_circuit_breaker_opened,
        finalization_primary_model=finalization_primary_model,
        finalization_fallback_models=finalization_fallback_models,
        finalization_fallback_used=finalization_fallback_used,
        finalization_fallback_succeeded=finalization_fallback_succeeded,
        finalization_fallback_attempts=finalization_fallback_attempts,
        finalization_events=finalization_events,
        final_content=final_content,
        final_finish_reason=final_finish_reason,
        requires_submit_answer=requires_submit_answer,
        submit_answer_call_count=submit_answer_call_count,
        submit_answer_succeeded=submit_answer_succeeded,
        required_submit_missing=required_submit_missing,
        submitted_answer_value=submitted_answer_value,
        submit_forced_retry_on_budget_exhaustion=submit_forced_retry_on_budget_exhaustion,
        submit_forced_accept_on_budget_exhaustion=submit_forced_accept_on_budget_exhaustion,
        require_tool_reasoning=require_tool_reasoning,
        rejected_missing_reasoning_calls=rejected_missing_reasoning_calls,
        control_loop_suppressed_calls=control_loop_suppressed_calls,
        submit_validation_reason_counts=submit_validation_reason_counts,
        tool_call_turns_total=tool_call_turns_total,
        tool_call_empty_text_turns=tool_call_empty_text_turns,
        responses_tool_call_empty_text_turns=responses_tool_call_empty_text_turns,
        tool_arg_coercions=tool_arg_coercions,
        tool_arg_coercion_calls=tool_arg_coercion_calls,
        tool_arg_validation_rejections=tool_arg_validation_rejections,
        tool_result_keep_recent=tool_result_keep_recent,
        tool_result_context_preview_chars=tool_result_context_preview_chars,
        active_artifact_context_enabled=active_artifact_context_enabled,
        active_artifact_context_max_handles=active_artifact_context_max_handles,
        active_artifact_context_max_chars=active_artifact_context_max_chars,
        artifact_context_updates=artifact_context_updates,
        artifact_context_chars=artifact_context_chars,
        context_tool_result_clearings=context_tool_result_clearings,
        context_tool_results_cleared=context_tool_results_cleared,
        context_tool_result_cleared_chars=context_tool_result_cleared_chars,
        tool_result_metadata_by_id=tool_result_metadata_by_id,
        context_compactions=context_compactions,
        context_compacted_messages=context_compacted_messages,
        context_compacted_chars=context_compacted_chars,
        enforce_tool_contracts=enforce_tool_contracts,
        progressive_tool_disclosure=progressive_tool_disclosure,
        suppress_control_loop_calls=suppress_control_loop_calls,
        gate_rejected_calls=gate_rejected_calls,
        gate_violation_events=gate_violation_events,
        contract_rejected_calls=contract_rejected_calls,
        contract_violation_events=contract_violation_events,
        normalized_tool_contracts=normalized_tool_contracts,
        runtime_artifact_tool_name=runtime_artifact_tool_name,
        runtime_artifact_registry_by_id=runtime_artifact_registry_by_id,
        handle_input_resolution_count=handle_input_resolution_count,
        handle_input_resolved_artifact_count=handle_input_resolved_artifact_count,
        initial_artifact_snapshot=initial_artifact_snapshot,
        available_artifacts=available_artifacts,
        initial_capability_snapshot=initial_capability_snapshot,
        available_capabilities=available_capabilities,
        artifact_timeline=artifact_timeline,
        initial_binding_snapshot=initial_binding_snapshot,
        available_bindings=available_bindings,
        run_config_spec=run_config_spec,
        run_config_hash=run_config_hash,
        adoption_assessment=adoption_assessment,
        tool_disclosure_turns=tool_disclosure_turns,
        tool_disclosure_hidden_total=tool_disclosure_hidden_total,
        tool_disclosure_unavailable_msgs=tool_disclosure_unavailable_msgs,
        tool_disclosure_unavailable_reason_chars=tool_disclosure_unavailable_reason_chars,
        tool_disclosure_unavailable_reason_tokens_est=tool_disclosure_unavailable_reason_tokens_est,
        tool_disclosure_repair_suggestions=tool_disclosure_repair_suggestions,
        lane_closure_analysis=lane_closure_analysis,
        no_legal_noncontrol_turns=no_legal_noncontrol_turns,
        max_deficit_no_progress_streak=max_deficit_no_progress_streak,
        deficit_no_progress_nudges=deficit_no_progress_nudges,
        max_zero_exec_tool_turn_streak=max_zero_exec_tool_turn_streak,
        retrieval_no_hits_count=retrieval_no_hits_count,
        retrieval_stagnation_turns=retrieval_stagnation_turns,
        retrieval_stagnation_action=retrieval_stagnation_action,
        retrieval_stagnation_streak=retrieval_stagnation_streak,
        retrieval_stagnation_streak_max=retrieval_stagnation_streak_max,
        retrieval_stagnation_triggered=retrieval_stagnation_triggered,
        retrieval_stagnation_turn=retrieval_stagnation_turn,
        evidence_digest_change_count=evidence_digest_change_count,
        evidence_turns_total=evidence_turns_total,
        evidence_turns_with_new_evidence=evidence_turns_with_new_evidence,
        evidence_turns_without_new_evidence=evidence_turns_without_new_evidence,
        evidence_pointer_count=evidence_pointer_count,
        failure_event_codes=failure_event_codes,
        submit_evidence_digest_at_last_failure=submit_evidence_digest_at_last_failure,
        foundation_events=foundation_events,
        foundation_event_types=foundation_event_types,
        foundation_event_validation_errors=foundation_event_validation_errors,
        foundation_events_logged=foundation_events_logged,
        evidence_pointer_labels=evidence_pointer_labels,
        force_final_reason=force_final_reason,
    )
    return final_content, final_finish_reason
