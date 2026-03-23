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

from llm_client.agent_artifacts import (
    _artifact_handle_summaries as _agent_artifact_handle_summaries,
    _build_active_artifact_context_content as _agent_build_active_artifact_context_content,
    _collect_recent_artifact_handles as _agent_collect_recent_artifact_handles,
    _upsert_active_artifact_context_message as _agent_upsert_active_artifact_context_message,
)
from llm_client.agent_contracts import (
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
from llm_client.agent_disclosure import (
    _deficit_labels_from_hidden_entries as _agent_deficit_labels_from_hidden_entries,
    _disclosure_message as _agent_disclosure_message,
    _disclosure_reason_from_entry as _agent_disclosure_reason_from_entry,
    _filter_tools_for_disclosure as _agent_filter_tools_for_disclosure,
)
from llm_client.agent_adoption import (
    DEFAULT_ADOPTION_PROFILE,
    AdoptionProfileAssessment,
    assess_adoption_profile,
    normalize_adoption_profile,
)
from llm_client.agent_outcomes import (
    ForcedFinalizationResult,
    _PRIMARY_FAILURE_PRIORITY,
    _TERMINAL_FAILURE_EVENT_CODES,
    _classify_failure_signals,
    _failure_class_for_event_code,
    _first_terminal_failure_event_code,
    _summarize_failure_events,
    _summarize_finalization_attempts,
)
from llm_client.client import LLMCallResult
from llm_client.compliance_gate import (
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
from llm_client.mcp_tools import (  # noqa: F401  re-exported
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
from llm_client.mcp_context import (  # noqa: F401  re-exported
    DEFAULT_MAX_MESSAGE_CHARS as DEFAULT_MAX_MESSAGE_CHARS,
    DEFAULT_TOOL_RESULT_CONTEXT_PREVIEW_CHARS as DEFAULT_TOOL_RESULT_CONTEXT_PREVIEW_CHARS,
    DEFAULT_TOOL_RESULT_KEEP_RECENT as DEFAULT_TOOL_RESULT_KEEP_RECENT,
    _clear_old_tool_results_for_context as _clear_old_tool_results_for_context,
    _compact_tool_history_for_context as _compact_tool_history_for_context,
    _message_char_length as _message_char_length,
    _trim_text as _trim_text,
)
from llm_client.mcp_evidence import (  # noqa: F401  re-exported
    DEFAULT_RETRIEVAL_STAGNATION_ACTION as DEFAULT_RETRIEVAL_STAGNATION_ACTION,
    DEFAULT_RETRIEVAL_STAGNATION_TURNS as DEFAULT_RETRIEVAL_STAGNATION_TURNS,
    RETRIEVAL_STAGNATION_ACTIONS as RETRIEVAL_STAGNATION_ACTIONS,
    _collect_evidence_pointer_labels as _collect_evidence_pointer_labels,
    _evidence_digest as _evidence_digest,
    _is_evidence_tool_name as _is_evidence_tool_name,
    _tool_evidence_pointer_labels as _tool_evidence_pointer_labels,
)
from llm_client.mcp_finalization import (  # noqa: F401  re-exported
    DEFAULT_ACCEPT_FORCED_ANSWER_ON_MAX_TOOL_CALLS as DEFAULT_ACCEPT_FORCED_ANSWER_ON_MAX_TOOL_CALLS,
    DEFAULT_FORCE_SUBMIT_RETRY_ON_MAX_TOOL_CALLS as DEFAULT_FORCE_SUBMIT_RETRY_ON_MAX_TOOL_CALLS,
    DEFAULT_FORCED_FINAL_CIRCUIT_BREAKER_THRESHOLD as DEFAULT_FORCED_FINAL_CIRCUIT_BREAKER_THRESHOLD,
    DEFAULT_FORCED_FINAL_MAX_ATTEMPTS as DEFAULT_FORCED_FINAL_MAX_ATTEMPTS,
    _FORCED_REFUSAL_RE as _FORCED_REFUSAL_RE,
    _execute_forced_finalization as _execute_forced_finalization,
    _normalize_forced_final_answer as _normalize_forced_final_answer,
    _provider_failure_classification as _provider_failure_classification,
)
from llm_client.mcp_state import (  # noqa: F401  re-exported
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
from llm_client.mcp_contracts import (  # noqa: F401  re-exported
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
from llm_client.tool_runtime_common import (
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


# ---------------------------------------------------------------------------
# Session Pool — reuse MCP servers across multiple calls
# ---------------------------------------------------------------------------


class MCPSessionPool:
    """Persistent MCP server connections for reuse across multiple acall_llm() calls.

    Usage:
        async with MCPSessionPool(mcp_servers) as pool:
            for question in questions:
                result = await acall_llm(model, msgs, mcp_sessions=pool)
    """

    def __init__(
        self,
        mcp_servers: dict[str, dict[str, Any]],
        init_timeout: float = DEFAULT_MCP_INIT_TIMEOUT,
    ):
        self.mcp_servers = mcp_servers
        self.init_timeout = init_timeout
        self._stack: AsyncExitStack | None = None
        self.sessions: dict[str, Any] = {}
        self.tool_to_server: dict[str, str] = {}
        self.openai_tools: list[dict[str, Any]] = []

    async def __aenter__(self) -> "MCPSessionPool":
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        self.sessions, self.tool_to_server, self.openai_tools = await _start_servers(
            self.mcp_servers, self._stack, self.init_timeout,
        )
        logger.info(
            "MCPSessionPool: started %d servers, %d tools",
            len(self.sessions), len(self.openai_tools),
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._stack:
            await self._stack.__aexit__(*exc)
            self._stack = None
            self.sessions = {}
            self.tool_to_server = {}
            self.openai_tools = []


# ---------------------------------------------------------------------------
# Schema conversion
# ---------------------------------------------------------------------------
def _mcp_tool_to_openai(tool: Any) -> dict[str, Any]:
    """Convert an MCP Tool object to OpenAI function-calling format.

    MCP: {"name": "foo", "description": "...", "inputSchema": {...}}
    OpenAI: {"type": "function", "function": {"name": "foo", "description": "...", "parameters": {...}}}
    """
    parameters = dict(tool.inputSchema or {"type": "object", "properties": {}})
    if not isinstance(parameters, dict):
        parameters = {"type": "object", "properties": {}}
    parameters.setdefault("type", "object")
    properties = parameters.setdefault("properties", {})
    if not isinstance(properties, dict):
        properties = {}
        parameters["properties"] = properties
    properties.setdefault(
        TOOL_REASONING_FIELD,
        {
            "type": "string",
            "description": "Why this specific tool call is needed right now.",
        },
    )
    required = parameters.get("required")
    if not isinstance(required, list):
        required = []
    if TOOL_REASONING_FIELD not in required:
        required.append(TOOL_REASONING_FIELD)
    parameters["required"] = required

    raw_examples = None
    input_examples_attr = getattr(tool, "inputExamples", None)
    if isinstance(input_examples_attr, (str, list, tuple, dict)):
        raw_examples = input_examples_attr
    elif isinstance(parameters.get("examples"), list):
        raw_examples = parameters.get("examples")
    elif isinstance(getattr(tool, "metadata", None), dict):
        raw_examples = tool.metadata.get("input_examples")

    description = _append_input_examples_to_description(
        str(tool.description or ""),
        _normalize_tool_input_examples(raw_examples),
    )

    # Clean schema for Gemini compatibility (safe for all providers).
    # Strips additionalProperties, strict, title; converts anyOf+null to nullable.
    from llm_client.client import _clean_schema_for_gemini
    parameters = _clean_schema_for_gemini(parameters)

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": description,
            "parameters": parameters,
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_agent_usage(agent_result: MCPAgentResult) -> dict[str, int]:
    """Build consistent usage/loop counters for agent-mode calls."""
    usage: dict[str, int] = {
        "input_tokens": int(agent_result.total_input_tokens),
        "output_tokens": int(agent_result.total_output_tokens),
        "total_tokens": int(agent_result.total_input_tokens + agent_result.total_output_tokens),
        "num_turns": int(agent_result.turns or 0),
        "n_tool_calls": int(len(agent_result.tool_calls)),
        "n_budgeted_tool_calls": int(_count_budgeted_records(agent_result.tool_calls)),
    }
    if agent_result.total_cached_tokens:
        usage["cached_tokens"] = int(agent_result.total_cached_tokens)
    if agent_result.total_cache_creation_tokens:
        usage["cache_creation_tokens"] = int(agent_result.total_cache_creation_tokens)
    return usage

def _foundation_schema_strict_enabled() -> bool:
    """Whether invalid foundation events should fail fast."""
    raw = os.environ.get(FOUNDATION_SCHEMA_STRICT_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}



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


# ---------------------------------------------------------------------------
# MCP Agent Loop
# ---------------------------------------------------------------------------


def _import_mcp() -> tuple[Any, ...]:
    """Lazily import mcp client components.

    Returns:
        (stdio_client, StdioServerParameters, ClientSession)
    """
    try:
        from mcp.client.stdio import (
            StdioServerParameters,
            stdio_client,
        )
        from mcp import ClientSession
    except ImportError:
        raise ImportError(
            "mcp package is required for MCP agent loop. "
            "Install with: pip install llm_client[mcp]"
        ) from None
    return stdio_client, StdioServerParameters, ClientSession


async def _start_servers(
    mcp_servers: dict[str, dict[str, Any]],
    stack: AsyncExitStack,
    init_timeout: float,
) -> tuple[dict[str, Any], dict[str, str], list[dict[str, Any]]]:
    """Start MCP servers and discover tools.

    Returns:
        (sessions, tool_to_server, openai_tools)
    """
    stdio_client, StdioServerParameters, ClientSession = _import_mcp()

    sessions: dict[str, Any] = {}
    tool_to_server: dict[str, str] = {}
    openai_tools: list[dict[str, Any]] = []

    for server_name, server_cfg in mcp_servers.items():
        params = StdioServerParameters(
            command=server_cfg["command"],
            args=server_cfg.get("args", []),
            env=server_cfg.get("env"),
            cwd=server_cfg.get("cwd"),
        )

        read_stream, write_stream = await stack.enter_async_context(
            stdio_client(params)
        )
        session = await stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await asyncio.wait_for(session.initialize(), timeout=init_timeout)
        sessions[server_name] = session

        # Discover tools
        tools_result = await session.list_tools()
        for tool in tools_result.tools:
            if tool.name in tool_to_server:
                logger.warning(
                    "Duplicate tool %r from server %r (already from %r)",
                    tool.name, server_name, tool_to_server[tool.name],
                )
                continue
            tool_to_server[tool.name] = server_name
            openai_tools.append(_mcp_tool_to_openai(tool))

    logger.info(
        "MCP agent loop: %d tools from %d servers",
        len(openai_tools), len(sessions),
    )
    return sessions, tool_to_server, openai_tools


async def _execute_tool_calls(
    tool_calls: list[dict[str, Any]],
    sessions: dict[str, Any],
    tool_to_server: dict[str, str],
    max_result_length: int,
    require_tool_reasoning: bool = DEFAULT_REQUIRE_TOOL_REASONING,
) -> tuple[list[MCPToolCallRecord], list[dict[str, Any]]]:
    """Execute tool calls against MCP servers.

    Returns:
        (records, tool_messages) — records for tracking, messages to append
    """
    records: list[MCPToolCallRecord] = []
    tool_messages: list[dict[str, Any]] = []

    for tc in tool_calls:
        fn = tc.get("function", {})
        tool_name = fn.get("name", "")
        arguments_str = fn.get("arguments", "{}")
        tc_id = tc.get("id", "")

        arguments = _extract_tool_call_args(tc)
        if arguments is None:
            logger.error("Failed to parse tool call arguments for %s: %s", tool_name, str(arguments_str)[:200])
            record = MCPToolCallRecord(
                server=tool_to_server.get(tool_name) or "unknown",
                tool=tool_name,
                arguments={},
                tool_call_id=tc_id,
                error="JSON parse error: arguments must decode to a JSON object",
            )
            tool_messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": "ERROR: Invalid JSON arguments: arguments must decode to a JSON object",
            })
            records.append(record)
            continue

        server_name = tool_to_server.get(tool_name)

        tool_reasoning_raw = arguments.pop(TOOL_REASONING_FIELD, None)
        tool_reasoning = None
        if isinstance(tool_reasoning_raw, str):
            stripped = tool_reasoning_raw.strip()
            if stripped:
                tool_reasoning = stripped

        record = MCPToolCallRecord(
            server=server_name or "unknown",
            tool=tool_name,
            arguments=arguments,
            tool_call_id=tc_id,
            tool_reasoning=tool_reasoning,
        )

        if require_tool_reasoning and not tool_reasoning:
            record.error = f"Missing required argument: {TOOL_REASONING_FIELD}"
            tool_messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": _json.dumps({"error": record.error}),
            })
            records.append(record)
            continue

        t0 = time.monotonic()
        if server_name is None:
            record.error = f"Unknown tool: {tool_name}"
            tool_content = _json.dumps({"error": record.error})
        else:
            try:
                session = sessions[server_name]
                mcp_result = await session.call_tool(tool_name, arguments)

                parts: list[str] = []
                for content_item in mcp_result.content or []:
                    if hasattr(content_item, "text"):
                        parts.append(content_item.text)
                    else:
                        parts.append(str(content_item))
                tool_content = "\n".join(parts)
                tool_content = _truncate(tool_content, max_result_length)

                if mcp_result.isError:
                    record.error = tool_content
                else:
                    record.result = tool_content
            except Exception as e:
                record.error = str(e)
                tool_content = _json.dumps({"error": str(e)})

        record.latency_s = round(time.monotonic() - t0, 3)
        records.append(record)

        tool_messages.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "content": tool_content,
        })

    return records, tool_messages


async def _inner_acall_llm(
    model: str,
    messages: list[dict[str, Any]],
    **kwargs: Any,
) -> LLMCallResult:
    """Call acall_llm from client module. Separate function for testability."""
    from llm_client.client import acall_llm
    return await acall_llm(model, messages, **kwargs)


async def _acall_with_mcp(
    model: str,
    messages: list[dict[str, Any]],
    mcp_servers: dict[str, dict[str, Any]] | None = None,
    *,
    mcp_sessions: MCPSessionPool | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_tool_calls: int | None = DEFAULT_MAX_TOOL_CALLS,
    require_tool_reasoning: bool = DEFAULT_REQUIRE_TOOL_REASONING,
    mcp_init_timeout: float = DEFAULT_MCP_INIT_TIMEOUT,
    tool_result_max_length: int = DEFAULT_TOOL_RESULT_MAX_LENGTH,
    max_message_chars: int = DEFAULT_MAX_MESSAGE_CHARS,
    enforce_tool_contracts: bool = DEFAULT_ENFORCE_TOOL_CONTRACTS,
    progressive_tool_disclosure: bool = DEFAULT_PROGRESSIVE_TOOL_DISCLOSURE,
    suppress_control_loop_calls: bool = DEFAULT_SUPPRESS_CONTROL_LOOP_CALLS,
    tool_contracts: dict[str, dict[str, Any]] | None = None,
    initial_artifacts: list[str] | tuple[str, ...] | None = DEFAULT_INITIAL_ARTIFACTS,
    initial_bindings: dict[str, Any] | None = None,
    timeout: int = 60,
    **kwargs: Any,
) -> LLMCallResult:
    """Run an MCP tool-calling agent loop with any litellm model.

    Starts MCP server subprocesses, discovers tools, then loops:
    1. Call LLM with tool definitions
    2. If LLM returns tool_calls → execute via MCP → append results
    3. Repeat until LLM returns text (no tool calls) or max_turns

    Args:
        model: Any litellm model string (NOT an agent model)
        messages: Initial messages (system + user)
        mcp_servers: Dict of server_name -> {command, args?, env?, cwd?}
            (ignored if mcp_sessions is provided)
        mcp_sessions: Pre-started MCPSessionPool for server reuse across calls.
            When provided, servers are NOT started/stopped per call.
        max_turns: Maximum loop iterations
        max_tool_calls: Maximum tool calls before final forced answer. None disables.
        require_tool_reasoning: If True, reject tool calls missing tool_reasoning.
        mcp_init_timeout: Seconds to wait for server startup
        tool_result_max_length: Max chars per tool result (truncated if longer)
        enforce_tool_contracts: If True, reject tool calls violating tool contracts.
        progressive_tool_disclosure: If True, hide tools whose contracts are not currently satisfiable.
        suppress_control_loop_calls: If True, suppress repeated submit/todo control calls.
        tool_contracts: Optional per-tool composability contracts.
        initial_artifacts: Artifact kinds available before any tool call.
        initial_bindings: Binding state available before any tool call.
        timeout: Per-turn LLM call timeout
        **kwargs: Passed through to acall_llm (retry, hooks, etc.)
    """
    agent_result = MCPAgentResult()
    messages = list(messages)  # don't mutate caller's list
    total_cost = 0.0
    final_content = ""
    final_finish_reason = "stop"

    if mcp_sessions is not None:
        # Reuse existing sessions — no server spawn/teardown
        sessions = mcp_sessions.sessions
        tool_to_server = mcp_sessions.tool_to_server
        openai_tools = mcp_sessions.openai_tools

        if not openai_tools:
            raise ValueError("MCPSessionPool has no tools — was it entered as context manager?")

        async def _mcp_executor(
            tool_calls: list[dict[str, Any]], max_len: int,
        ) -> tuple[list[MCPToolCallRecord], list[dict[str, Any]]]:
            return await _execute_tool_calls(
                tool_calls,
                sessions,
                tool_to_server,
                max_len,
                require_tool_reasoning=require_tool_reasoning,
            )

        final_content, final_finish_reason = await _agent_loop(
            model, messages, openai_tools,
            agent_result,
            _mcp_executor,
            max_turns,
            max_tool_calls,
            require_tool_reasoning,
            tool_result_max_length,
            max_message_chars,
            enforce_tool_contracts,
            progressive_tool_disclosure,
            suppress_control_loop_calls,
            tool_contracts,
            initial_artifacts,
            initial_bindings,
            timeout,
            kwargs,
        )
        total_cost = sum(r.latency_s for r in agent_result.tool_calls)  # placeholder; real cost tracked below

    elif mcp_servers is not None:
        async with AsyncExitStack() as stack:
            # Start servers and discover tools
            sessions, tool_to_server, openai_tools = await _start_servers(
                mcp_servers, stack, mcp_init_timeout,
            )

            if not openai_tools:
                raise ValueError(
                    f"No tools discovered from MCP servers: {list(mcp_servers.keys())}"
                )

            async def _mcp_executor(
                tool_calls: list[dict[str, Any]], max_len: int,
            ) -> tuple[list[MCPToolCallRecord], list[dict[str, Any]]]:
                return await _execute_tool_calls(
                    tool_calls,
                    sessions,
                    tool_to_server,
                    max_len,
                    require_tool_reasoning=require_tool_reasoning,
                )

            final_content, final_finish_reason = await _agent_loop(
                model, messages, openai_tools,
                agent_result,
                _mcp_executor,
                max_turns,
                max_tool_calls,
                require_tool_reasoning,
                tool_result_max_length,
                max_message_chars,
                enforce_tool_contracts,
                progressive_tool_disclosure,
                suppress_control_loop_calls,
                tool_contracts,
                initial_artifacts,
                initial_bindings,
                timeout,
                kwargs,
            )
    else:
        raise ValueError("Either mcp_servers or mcp_sessions must be provided")

    # Cost is accumulated during _agent_loop via agent_result metadata.
    usage = _build_agent_usage(agent_result)
    resolved_model = agent_result.metadata.get("resolved_model")
    if not isinstance(resolved_model, str) or not resolved_model.strip():
        resolved_model = None
    attempted_models = agent_result.metadata.get("attempted_models")
    if not isinstance(attempted_models, list):
        attempted_models = None
    sticky_fallback = agent_result.metadata.get("sticky_fallback")
    if not isinstance(sticky_fallback, bool):
        sticky_fallback = None
    return LLMCallResult(
        content=final_content,
        usage=usage,
        cost=agent_result.metadata.get("total_cost", 0.0),
        model=resolved_model or model,
        requested_model=model,
        resolved_model=resolved_model,
        routing_trace={
            "attempted_models": attempted_models,
            "sticky_fallback": sticky_fallback,
        },
        finish_reason=final_finish_reason,
        raw_response=agent_result,
        warnings=agent_result.warnings,
    )


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
) -> tuple[str, str]:
    """Core agent loop shared by MCP, direct-tool, and session-pool paths.

    Args:
        executor: async callable (tool_calls, max_result_length) -> (records, tool_messages).
            For MCP: wraps _execute_tool_calls with bound sessions.
            For direct tools: wraps execute_direct_tool_calls with bound tool_map.

    Returns (final_content, final_finish_reason).
    """
    from llm_client import mcp_agent as _mcp_agent

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
        from llm_client import io_log as _io_log_module  # local import to avoid module-cycle hazards
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
                caller="llm_client.mcp_agent",
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
        if artifact_context_changed and artifact_context_content:
            artifact_context_updates += 1
            artifact_context_chars += len(artifact_context_content)
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
            context_tool_result_clearings += 1
            context_tool_results_cleared += cleared_count
            context_tool_result_cleared_chars += cleared_chars
            warning = (
                "CONTEXT_TOOL_RESULT_CLEARING: replaced "
                f"{cleared_count} older tool payload(s) with compact stubs "
                f"(saved ~{cleared_chars} chars; keep_recent={tool_result_keep_recent})."
            )
            agent_result.warnings.append(warning)
            logger.warning(warning)

        compacted_count, compacted_chars, current_chars = _compact_tool_history_for_context(
            messages, max_message_chars,
        )
        if compacted_count:
            context_compactions += 1
            context_compacted_messages += compacted_count
            context_compacted_chars += compacted_chars
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
        disclosure_repair_hints: list[str] = []
        current_turn_deficit_labels: list[str] = []
        current_turn_deficit_digest: str | None = None
        if progressive_tool_disclosure and normalized_tool_contracts:
            disclosed_tools, hidden_disclosure, hidden_disclosure_total = _filter_tools_for_disclosure(
                openai_tools=current_tool_surface,
                normalized_tool_contracts=normalized_tool_contracts,
                available_artifacts=available_artifacts,
                available_capabilities=available_capabilities,
                available_bindings=available_bindings,
            )
            if hidden_disclosure:
                tool_disclosure_turns += 1
                tool_disclosure_hidden_total += hidden_disclosure_total
                tool_disclosure_repair_suggestions += sum(
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
                current_turn_deficit_labels = _deficit_labels_from_hidden_entries(hidden_disclosure)
                if current_turn_deficit_labels:
                    current_turn_deficit_digest = sha256_json(current_turn_deficit_labels).replace("sha256:", "")
                hidden_names = [h["tool"] for h in hidden_disclosure]
                disclosure_reason = _disclosure_message(hidden_disclosure)
                disclosure_msg = {
                    "role": "user",
                    "content": (
                        "[SYSTEM: Currently unavailable tools with missing requirements: "
                        f"{disclosure_reason}]"
                    ),
                }
                disclosure_chars = len(str(disclosure_msg.get("content", "")))
                tool_disclosure_unavailable_msgs += 1
                tool_disclosure_unavailable_reason_chars += disclosure_chars
                tool_disclosure_unavailable_reason_tokens_est += max(
                    1, disclosure_chars // DEFAULT_TOOL_DISCLOSURE_TOKEN_CHARS,
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
            deficit_no_progress_nudges += 1
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
                t for t in disclosed_tools
                if isinstance(t, dict)
                and not _is_control_tool_name(
                    str(t.get("function", {}).get("name", "")).strip(),
                    normalized_tool_contracts,
                )
            ]
            if not noncontrol_disclosed:
                no_legal_noncontrol_turns += 1
                warning = (
                    "TOOL_DISCLOSURE: no legal non-control tools available this turn. "
                    "Use conversion/planning/control tools to unlock capabilities."
                )
                agent_result.warnings.append(warning)
                logger.warning(warning)
                failure_event_codes.append(EVENT_CODE_NO_LEGAL_NONCONTROL_TOOLS)
                _emit_foundation_event(
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
                            "error_code": EVENT_CODE_NO_LEGAL_NONCONTROL_TOOLS,
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
            result = await _inner_acall_llm(
                effective_model, messages, timeout=timeout, tools=disclosed_tools, **kwargs,
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
            provider_subevent = "first_turn" if turn == 0 else "turn"
            failure_event_codes.append(event_code)
            warning = (
                "AGENT_LLM_CALL_FAILED: turn="
                f"{turn + 1} model={effective_model} error={error_text}"
            )
            agent_result.warnings.append(warning)
            logger.warning(warning)
            _emit_foundation_event(
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
                            "provider_subevent": provider_subevent if is_provider_failure else "",
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
            final_content = error_text
            final_finish_reason = "error"
            break

        # Track per-turn diagnostics
        result_model = str(result.model).strip() or effective_model
        agent_result.models_used.add(result_model)
        if result_model and result_model not in attempted_models:
            attempted_models.append(result_model)
        if result.warnings:
            agent_result.warnings.extend(result.warnings)

        # Sticky fallback: if inner call fell back to a different model,
        # use that model for remaining turns (avoids re-hitting dead primary).
        if result_model != effective_model:
            agent_result.warnings.append(
                f"STICKY_FALLBACK: {effective_model} failed, "
                f"using {result_model} for remaining turns"
            )
            effective_model = result_model
            sticky_fallback = True

        _emit_foundation_event(
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
                        "capabilities_sha256": sha256_json(_capability_state_snapshot(available_capabilities)),
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

        # Accumulate usage
        inp, out, cached, cache_create = _extract_usage(result.usage or {})
        agent_result.total_input_tokens += inp
        agent_result.total_output_tokens += out
        agent_result.total_cached_tokens += cached
        agent_result.total_cache_creation_tokens += cache_create
        total_cost += result.cost

        # No tool calls → done
        if not result.tool_calls:
            remaining_turns = max_turns - (turn + 1)
            if requires_submit_answer and not submit_answer_succeeded and remaining_turns > 0:
                # Benchmark loops require explicit submit_answer() so scorers can
                # reliably parse the final answer from tool calls.
                if result.content:
                    draft_msg = {
                        "role": "assistant",
                        "content": result.content,
                    }
                    messages.append(draft_msg)
                    agent_result.conversation_trace.append(draft_msg)

                # Two-stage recovery for plain-text replies:
                # 1) first no-tool turn -> continue tool-based solving (avoid premature submit loops)
                # 2) repeated no-tool turn (or near end) -> require explicit submit_answer
                near_end = remaining_turns <= TURN_WARNING_THRESHOLD
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
                        turn + 1, max_turns,
                    )
                    continue

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
                    turn + 1, max_turns,
                )
                continue

            final_content = result.content
            final_finish_reason = result.finish_reason
            # Log visibility: empty content on first turn is almost always a model failure
            if not result.content and turn == 0:
                failure_event_codes.append(EVENT_CODE_PROVIDER_EMPTY)
                _emit_foundation_event(
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
                                "provider_subevent": "first_turn",
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
                            "user_message": "Provider returned empty content with no tool calls on first turn.",
                            "debug_ref": None,
                        },
                    }
                )
                logger.error(
                    "Agent loop: model=%s returned empty content with 0 tool calls on turn 1 "
                    "(finish_reason=%s). All %d retries + fallback exhausted at the per-turn level.",
                    model, result.finish_reason, kwargs.get("num_retries", 2),
                )
            elif not result.content:
                failure_event_codes.append(EVENT_CODE_PROVIDER_EMPTY)
                _emit_foundation_event(
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
                                "provider_subevent": "turn",
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
                            "user_message": "Provider returned empty content with no tool calls.",
                            "debug_ref": None,
                        },
                    }
                )
                logger.warning(
                    "Agent loop: model=%s returned empty content on turn %d/%d "
                    "(finish_reason=%s, %d tool calls so far).",
                    model, turn + 1, max_turns, result.finish_reason,
                    len(agent_result.tool_calls),
                )
            # Capture final assistant message in trace
            if result.content:
                agent_result.conversation_trace.append({
                    "role": "assistant",
                    "content": result.content,
                })
            break

        # Reset plain-text streak once the model returns at least one tool call.
        plain_text_no_tool_turn_streak = 0

        # Append assistant message with tool calls
        tool_calls_this_turn = list(result.tool_calls)
        autofilled_reasoning_tools: list[str] = []
        patched_calls: list[dict[str, Any]] = []
        for tc in tool_calls_this_turn:
            patched, changed = _autofill_tool_reasoning(tc)
            normalized_name = _normalize_tool_call_name_inplace(patched)
            if changed:
                name = normalized_name
                if name:
                    autofilled_reasoning_tools.append(name)
                    autofilled_tool_reasoning_by_tool[name] = (
                        autofilled_tool_reasoning_by_tool.get(name, 0) + 1
                    )
                autofilled_tool_reasoning_calls += 1
            patched_calls.append(patched)
        tool_calls_this_turn = patched_calls
        if autofilled_reasoning_tools:
            agent_result.warnings.append(
                "OBSERVABILITY: auto-filled missing tool_reasoning on tools: "
                + ", ".join(autofilled_reasoning_tools)
            )
        tool_call_turns_total += 1
        if not (result.content or "").strip():
            tool_call_empty_text_turns += 1
            is_responses_turn = _is_responses_api_raw_response(result.raw_response)
            if is_responses_turn:
                responses_tool_call_empty_text_turns += 1
            logger.info(
                "Agent loop metric: turn %d/%d model=%s returned %d tool call(s) with empty assistant text%s",
                turn + 1,
                max_turns,
                result.model,
                len(tool_calls_this_turn),
                " [responses-api]" if is_responses_turn else "",
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
        for tc in tool_calls_this_turn:
            fn = tc.get("function", {})
            tc_name = fn.get("name", "")
            tc_args = _extract_tool_call_args(tc)
            if not isinstance(tc_args, dict):
                missing_reasoning_tools.append(tc_name or "<unknown>")
                continue
            tc_reasoning = tc_args.get(TOOL_REASONING_FIELD)
            if not isinstance(tc_reasoning, str) or not tc_reasoning.strip():
                missing_reasoning_tools.append(tc_name or "<unknown>")

        if missing_reasoning_tools:
            reasoning_nudge = {
                "role": "user",
                "content": (
                    "[SYSTEM: Observability requirement: every tool call must include "
                    f"'{TOOL_REASONING_FIELD}' with one concise sentence explaining why this call is needed."
                    + (
                        " Calls without it are rejected."
                        if require_tool_reasoning else
                        ""
                    )
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

        # Capture assistant message in trace (with tool call names for readability)
        agent_result.conversation_trace.append({
            "role": "assistant",
            "content": result.content or "",
            "tool_calls": [
                {
                    "id": tc.get("id", ""),
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": tc.get("function", {}).get("arguments", ""),
                }
                for tc in tool_calls_this_turn
            ],
        })

        tool_calls_to_execute = tool_calls_this_turn
        gate_rejected_records: list[MCPToolCallRecord] = []
        gate_rejected_messages: list[dict[str, Any]] = []
        pending_gate_msg: dict[str, Any] | None = None

        filtered_gate_calls: list[dict[str, Any]] = []
        for tc in tool_calls_to_execute:
            tool_name = str(tc.get("function", {}).get("name", "")).strip()
            tc_id = tc.get("id", "")
            parsed_args = _extract_tool_call_args(tc)
            gate_validation = validate_tool_call_inputs(
                tool_name=tool_name or "<unknown>",
                parsed_args=parsed_args,
                tool_parameters=tool_parameter_index.get(tool_name),
                require_tool_reasoning=require_tool_reasoning,
                tool_reasoning_field=TOOL_REASONING_FIELD,
                available_bindings=available_bindings,
                error_code_schema=EVENT_CODE_TOOL_VALIDATION_SCHEMA,
                error_code_missing_reasoning=EVENT_CODE_TOOL_VALIDATION_MISSING_TOOL_REASONING,
                error_code_binding_conflict=EVENT_CODE_TOOL_VALIDATION_BINDING_CONFLICT,
            )
            if gate_validation.is_valid:
                filtered_gate_calls.append(tc)
                continue

            gate_rejected_calls += 1
            if gate_validation.error_code == EVENT_CODE_TOOL_VALIDATION_MISSING_TOOL_REASONING:
                rejected_missing_reasoning_calls += 1
            gate_reason = gate_validation.reason or "Compliance gate rejected tool call."
            gate_error = f"Validation error: {gate_reason}"
            gate_error_code = gate_validation.error_code or EVENT_CODE_TOOL_VALIDATION_SCHEMA
            failure_event_codes.append(gate_error_code)
            gate_violation_events.append(
                {
                    "turn": turn + 1,
                    "tool": tool_name or "<unknown>",
                    "reason": gate_reason,
                    "error_code": gate_error_code,
                    "failure_phase": gate_validation.failure_phase or "input_validation",
                    "available_artifacts": sorted(available_artifacts),
                    "available_capabilities": _capability_state_snapshot(available_capabilities),
                    "available_bindings": dict(available_bindings),
                    "call_bindings": dict(gate_validation.call_bindings),
                    "arg_keys": sorted(parsed_args.keys()) if isinstance(parsed_args, dict) else [],
                    "missing_requirements": list(gate_validation.missing_requirements or []),
                }
            )
            gate_rejected_records.append(
                MCPToolCallRecord(
                    server="__compliance__",
                    tool=tool_name or "<unknown>",
                    arguments=parsed_args if isinstance(parsed_args, dict) else {},
                    error=gate_error,
                )
            )
            gate_rejected_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": _json.dumps(
                        {
                            "error": gate_error,
                            "error_code": gate_error_code,
                            "failure_phase": gate_validation.failure_phase or "input_validation",
                            "call_bindings": gate_validation.call_bindings,
                            "missing_requirements": gate_validation.missing_requirements,
                        }
                    ),
                }
            )
            _emit_foundation_event(
                {
                    "event_id": new_event_id(),
                    "event_type": "ToolFailed",
                    "timestamp": now_iso(),
                    "run_id": foundation_run_id,
                    "session_id": foundation_session_id,
                    "actor_id": foundation_actor_id,
                    "operation": {"name": tool_name or "<unknown>", "version": None},
                    "inputs": {
                        "artifact_ids": sorted(available_artifacts),
                        "params": parsed_args if isinstance(parsed_args, dict) else {},
                        "bindings": dict(available_bindings),
                    },
                    "outputs": {"artifact_ids": [], "payload_hashes": []},
                    "failure": {
                        "error_code": gate_error_code,
                        "category": "validation",
                        "phase": gate_validation.failure_phase or "input_validation",
                        "retryable": True,
                        "tool_name": tool_name or "<unknown>",
                        "user_message": gate_error,
                        "debug_ref": None,
                    },
                }
            )

        tool_calls_to_execute = filtered_gate_calls
        if gate_rejected_records:
            pending_gate_msg = {
                "role": "user",
                "content": (
                    "[SYSTEM: One or more tool calls were rejected by compliance checks "
                    "(schema/binding/observability). Re-issue corrected calls only if still needed.]"
                ),
            }

        contract_rejected_records: list[MCPToolCallRecord] = []
        contract_rejected_messages: list[dict[str, Any]] = []
        pending_contract_msg: dict[str, Any] | None = None
        if enforce_tool_contracts and normalized_tool_contracts:
            filtered_contract_calls: list[dict[str, Any]] = []
            for tc in tool_calls_to_execute:
                tool_name = tc.get("function", {}).get("name", "")
                tc_id = tc.get("id", "")
                parsed_args = _extract_tool_call_args(tc)
                contract = normalized_tool_contracts.get(tool_name)
                if not isinstance(contract, dict):
                    filtered_contract_calls.append(tc)
                    continue

                contract_validation = _validate_tool_contract_call(
                    tool_name=tool_name,
                    contract=contract,
                    parsed_args=parsed_args,
                    available_artifacts=available_artifacts,
                    available_capabilities=available_capabilities,
                    available_bindings=available_bindings,
                    artifact_registry_by_id=runtime_artifact_registry_by_id,
                )
                if contract_validation.is_valid:
                    filtered_contract_calls.append(tc)
                    continue

                contract_rejected_calls += 1
                err = f"Tool contract violation: {contract_validation.reason}"
                contract_violation_events.append({
                    "turn": turn + 1,
                    "tool": tool_name or "<unknown>",
                    "contract_mode": contract_validation.contract_mode,
                    "reason": contract_validation.reason,
                    "error_code": contract_validation.error_code or EVENT_CODE_TOOL_VALIDATION_MISSING_PREREQUISITE,
                    "failure_phase": contract_validation.failure_phase or "input_validation",
                    "available_artifacts": sorted(available_artifacts),
                    "available_capabilities": _capability_state_snapshot(available_capabilities),
                    "available_bindings": dict(available_bindings),
                    "call_bindings": dict(contract_validation.call_bindings),
                    "arg_keys": sorted(parsed_args.keys()) if isinstance(parsed_args, dict) else [],
                    "missing_requirements": list(contract_validation.missing_requirements or []),
                })
                failure_event_codes.append(
                    contract_validation.error_code or EVENT_CODE_TOOL_VALIDATION_MISSING_PREREQUISITE
                )
                contract_rejected_records.append(
                    MCPToolCallRecord(
                        server="__contract__",
                        tool=tool_name or "<unknown>",
                        arguments=parsed_args if isinstance(parsed_args, dict) else {},
                        error=err,
                    )
                )
                contract_rejected_messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": _json.dumps(
                        {
                            "error": err,
                            "error_code": contract_validation.error_code or EVENT_CODE_TOOL_VALIDATION_MISSING_PREREQUISITE,
                            "failure_phase": contract_validation.failure_phase or "input_validation",
                            "contract_mode": contract_validation.contract_mode,
                            "call_bindings": contract_validation.call_bindings,
                            "missing_requirements": contract_validation.missing_requirements,
                        }
                    ),
                })
                _emit_foundation_event(
                    {
                        "event_id": new_event_id(),
                        "event_type": "ToolFailed",
                        "timestamp": now_iso(),
                        "run_id": foundation_run_id,
                        "session_id": foundation_session_id,
                        "actor_id": foundation_actor_id,
                        "operation": {"name": tool_name or "<unknown>", "version": None},
                        "inputs": {
                            "artifact_ids": sorted(available_artifacts),
                            "params": parsed_args if isinstance(parsed_args, dict) else {},
                            "bindings": dict(available_bindings),
                        },
                        "outputs": {"artifact_ids": [], "payload_hashes": []},
                        "failure": {
                            "error_code": contract_validation.error_code or EVENT_CODE_TOOL_VALIDATION_MISSING_PREREQUISITE,
                            "category": "validation",
                            "phase": contract_validation.failure_phase or "input_validation",
                            "retryable": False,
                            "tool_name": tool_name or "<unknown>",
                            "user_message": err,
                            "debug_ref": None,
                        },
                    }
                )

            tool_calls_to_execute = filtered_contract_calls
            if contract_rejected_records:
                pending_contract_msg = {
                    "role": "user",
                    "content": (
                        "[SYSTEM: One or more tool calls were rejected by tool composability contracts. "
                        f"Use tools compatible with available artifacts: {sorted(available_artifacts)}. "
                        f"Capabilities: {_capability_state_snapshot(available_capabilities)}. "
                        "Adjust plan and try again.]"
                    ),
                }

        # Control-loop suppression: block repeated submit_answer when
        # evidence hasn't changed since last rejected submission.
        current_evidence_digest = _evidence_digest(evidence_pointer_labels)
        suppressed_records: list[MCPToolCallRecord] = []
        suppressed_tool_messages: list[dict[str, Any]] = []
        filtered_tool_calls: list[dict[str, Any]] = []
        if suppress_control_loop_calls:
            for tc in tool_calls_to_execute:
                tool_name = tc.get("function", {}).get("name", "")
                tc_id = tc.get("id", "")
                parsed_args = _extract_tool_call_args(tc) or {}

                if tool_name == "submit_answer" and submit_requires_new_evidence:
                    if (
                        submit_evidence_digest_at_last_failure is not None
                        and current_evidence_digest != submit_evidence_digest_at_last_failure
                    ):
                        submit_requires_new_evidence = False
                        submit_evidence_digest_at_last_failure = None
                    else:
                        err = (
                            "submit_answer suppressed: validator requires NEW evidence before retry. "
                            "Evidence digest has not changed since last rejected submit. "
                            "Run at least one non-control evidence tool call "
                            "(entity_*, chunk_*, relationship_*, subgraph_*) first, then retry submit."
                        )
                        suppressed_records.append(
                            MCPToolCallRecord(
                                server="__agent__",
                                tool=tool_name,
                                arguments=parsed_args,
                                error=err,
                            )
                        )
                        suppressed_tool_messages.append({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": _json.dumps(
                                {
                                    "error": err,
                                    "error_code": EVENT_CODE_CONTROL_LOOP_SUPPRESSED,
                                    "evidence_digest": current_evidence_digest,
                                    "required_evidence_digest_change_from": submit_evidence_digest_at_last_failure,
                                }
                            ),
                        })
                        continue

                filtered_tool_calls.append(tc)
        else:
            filtered_tool_calls = list(tool_calls_to_execute)

        tool_calls_to_execute = filtered_tool_calls
        pending_control_loop_msg: dict[str, Any] | None = None
        if suppressed_records:
            control_loop_suppressed_calls += len(suppressed_records)
            pending_control_loop_msg = {
                "role": "user",
                "content": (
                    "[SYSTEM: Repeated control-call loop detected. Some tool calls were suppressed. "
                    "Update TODO state or change hypotheses before retrying submit/completion.]"
                ),
            }
            for rec in suppressed_records:
                _emit_foundation_event(
                    {
                        "event_id": new_event_id(),
                        "event_type": "ToolFailed",
                        "timestamp": now_iso(),
                        "run_id": foundation_run_id,
                        "session_id": foundation_session_id,
                        "actor_id": foundation_actor_id,
                        "operation": {"name": rec.tool or "<unknown>", "version": None},
                        "inputs": {
                            "artifact_ids": sorted(available_artifacts),
                            "params": rec.arguments if isinstance(rec.arguments, dict) else {},
                            "bindings": dict(available_bindings),
                        },
                        "outputs": {"artifact_ids": [], "payload_hashes": []},
                        "failure": {
                            "error_code": EVENT_CODE_CONTROL_LOOP_SUPPRESSED,
                            "category": "policy",
                            "phase": "post_validation",
                            "retryable": False,
                            "tool_name": rec.tool or "<unknown>",
                            "user_message": rec.error or "suppressed",
                            "debug_ref": None,
                        },
                    }
                )
                failure_event_codes.append(EVENT_CODE_CONTROL_LOOP_SUPPRESSED)

        for tc in tool_calls_to_execute:
            tool_name = tc.get("function", {}).get("name", "") or "<unknown>"
            parsed_args = _extract_tool_call_args(tc)
            _emit_foundation_event(
                {
                    "event_id": new_event_id(),
                    "event_type": "ToolCalled",
                    "timestamp": now_iso(),
                    "run_id": foundation_run_id,
                    "session_id": foundation_session_id,
                    "actor_id": foundation_actor_id,
                    "operation": {"name": tool_name, "version": None},
                    "inputs": {
                        "artifact_ids": sorted(available_artifacts),
                        "params": parsed_args if isinstance(parsed_args, dict) else {},
                        "bindings": dict(available_bindings),
                    },
                    "outputs": {"artifact_ids": [], "payload_hashes": []},
                }
            )

        # Execute valid tool calls via runtime helpers + external executor.
        external_tool_calls: list[dict[str, Any]] = []
        runtime_tool_call_indexes: dict[int, tuple[MCPToolCallRecord, dict[str, Any]]] = {}
        for idx, tc in enumerate(tool_calls_to_execute):
            tool_name = str(tc.get("function", {}).get("name", "")).strip()
            if tool_name == runtime_artifact_tool_name:
                runtime_tool_call_indexes[idx] = _runtime_artifact_read_result(
                    artifact_registry_by_id=runtime_artifact_registry_by_id,
                    tc=tc,
                    max_result_length=tool_result_max_length,
                    require_tool_reasoning=require_tool_reasoning,
                )
            else:
                patched_tc, handle_injections = _apply_handle_input_injections(
                    tc=tc,
                    normalized_tool_contracts=normalized_tool_contracts,
                    artifact_registry_by_id=runtime_artifact_registry_by_id,
                )
                if handle_injections:
                    handle_input_resolution_count += len(handle_injections)
                    handle_input_resolved_artifact_count += sum(
                        int(item.get("resolved_count") or 0)
                        for item in handle_injections
                        if isinstance(item, dict)
                    )
                external_tool_calls.append(patched_tc)

        if external_tool_calls:
            external_records, external_tool_messages = await executor(
                external_tool_calls, tool_result_max_length,
            )
        else:
            external_records, external_tool_messages = [], []

        executed_records = []
        executed_tool_messages = []
        external_record_iter = iter(external_records)
        external_message_iter = iter(external_tool_messages)
        for idx, _tc in enumerate(tool_calls_to_execute):
            runtime_pair = runtime_tool_call_indexes.get(idx)
            if runtime_pair is not None:
                record, tool_message = runtime_pair
            else:
                record = next(external_record_iter)
                tool_message = next(external_message_iter)
            executed_records.append(record)
            executed_tool_messages.append(tool_message)

        for record in executed_records:
            if record.error:
                _emit_foundation_event(
                    {
                        "event_id": new_event_id(),
                        "event_type": "ToolFailed",
                        "timestamp": now_iso(),
                        "run_id": foundation_run_id,
                        "session_id": foundation_session_id,
                        "actor_id": foundation_actor_id,
                        "operation": {"name": record.tool or "<unknown>", "version": None},
                        "inputs": {
                            "artifact_ids": sorted(available_artifacts),
                            "params": record.arguments if isinstance(record.arguments, dict) else {},
                            "bindings": dict(available_bindings),
                        },
                        "outputs": {"artifact_ids": [], "payload_hashes": []},
                        "failure": {
                            "error_code": EVENT_CODE_TOOL_RUNTIME_ERROR,
                            "category": "execution",
                            "phase": "execution",
                            "retryable": False,
                            "tool_name": record.tool or "<unknown>",
                            "user_message": record.error,
                            "debug_ref": None,
                        },
                    }
                )
                failure_event_codes.append(EVENT_CODE_TOOL_RUNTIME_ERROR)
                continue

            observed_bindings = extract_bindings_from_tool_args(record.arguments)
            (
                _typed_output_envelopes,
                typed_output_artifacts,
                typed_output_capabilities,
                typed_output_artifact_ids,
                typed_output_bindings,
            ) = _artifact_output_state_from_record(
                record,
                fallback_bindings=available_bindings,
            )
            for envelope in _typed_output_envelopes:
                artifact_id = str(envelope.get("artifact_id", "")).strip()
                if artifact_id:
                    runtime_artifact_registry_by_id[artifact_id] = dict(envelope)
            record.artifact_ids = sorted(typed_output_artifact_ids)
            record.artifact_handles = _artifact_handle_summaries(_typed_output_envelopes)
            if record.tool_call_id:
                tool_result_metadata_by_id[record.tool_call_id] = {
                    "artifact_ids": list(record.artifact_ids),
                    "artifact_handles": list(record.artifact_handles),
                }
            observed_bindings = merge_binding_state(
                available_bindings=observed_bindings,
                observed_bindings=typed_output_bindings,
            )
            merged_bindings = merge_binding_state(
                available_bindings=available_bindings,
                observed_bindings=observed_bindings,
            )
            if merged_bindings != available_bindings:
                old_bindings = dict(available_bindings)
                available_bindings = merged_bindings
                _emit_foundation_event(
                    {
                        "event_id": new_event_id(),
                        "event_type": "BindingChanged",
                        "timestamp": now_iso(),
                        "run_id": foundation_run_id,
                        "session_id": foundation_session_id,
                        "actor_id": foundation_actor_id,
                        "operation": {"name": record.tool or "<unknown>", "version": None},
                        "inputs": {
                            "artifact_ids": sorted(available_artifacts),
                            "params": record.arguments if isinstance(record.arguments, dict) else {},
                            "bindings": old_bindings,
                        },
                        "outputs": {"artifact_ids": [], "payload_hashes": []},
                        "binding_change": {
                            "old_bindings": old_bindings,
                            "new_bindings": dict(available_bindings),
                            "reason": "adopted_new_binding_from_successful_tool_call",
                        },
                    }
                )

            produced: set[str] = set()
            produced_capabilities: list[CapabilityRequirement] = []
            if enforce_tool_contracts and normalized_tool_contracts:
                contract = normalized_tool_contracts.get(record.tool)
                parsed_record_args = record.arguments if isinstance(record.arguments, dict) else {}
                produced = _contract_outputs(contract, parsed_record_args)
                produced_capabilities = _contract_output_capabilities(contract, parsed_record_args)

            produced.update(typed_output_artifacts)
            if produced:
                available_artifacts.update(produced)

            inferred_caps = _infer_output_capabilities(
                tool_name=record.tool,
                parsed_args=record.arguments if isinstance(record.arguments, dict) else {},
                produced_artifacts=produced,
                available_bindings=available_bindings,
            )

            produced_capability_payload: list[dict[str, str]] = []
            for req in produced_capabilities + typed_output_capabilities + inferred_caps:
                added = _capability_state_add(
                    available_capabilities,
                    kind=req.kind,
                    ref_type=req.ref_type,
                    namespace=req.namespace,
                    bindings_hash=req.bindings_hash,
                )
                if added:
                    produced_capability_payload.append(req.to_dict())

            if produced:
                artifact_timeline.append({
                    "turn": turn + 1,
                    "phase": "tool_success",
                    "tool": record.tool,
                    "produced": sorted(produced),
                    "produced_capabilities": produced_capability_payload,
                    "available_artifacts": sorted(available_artifacts),
                    "available_capabilities": _capability_state_snapshot(available_capabilities),
                })

            if _is_retrieval_no_hits_result(tool_name=record.tool, result_text=record.result):
                retrieval_no_hits_count += 1
                failure_event_codes.append("RETRIEVAL_NO_HITS")

            _emit_foundation_event(
                {
                    "event_id": new_event_id(),
                    "event_type": "ArtifactCreated",
                    "timestamp": now_iso(),
                    "run_id": foundation_run_id,
                    "session_id": foundation_session_id,
                    "actor_id": foundation_actor_id,
                    "operation": {"name": record.tool or "<unknown>", "version": None},
                    "inputs": {
                        "artifact_ids": sorted(available_artifacts),
                        "params": record.arguments if isinstance(record.arguments, dict) else {},
                        "bindings": dict(available_bindings),
                    },
                    "outputs": {
                        "artifact_ids": sorted(typed_output_artifact_ids or produced),
                        "payload_hashes": (
                            [sha256_text(record.result or "")]
                            if record.result is not None else []
                        ),
                    },
                    "artifacts": list(_typed_output_envelopes),
                }
            )

        records = gate_rejected_records + contract_rejected_records + suppressed_records + executed_records
        tool_messages = gate_rejected_messages + contract_rejected_messages + suppressed_tool_messages + executed_tool_messages
        agent_result.tool_calls.extend(records)
        messages.extend(tool_messages)
        if pending_gate_msg is not None:
            messages.append(pending_gate_msg)
            agent_result.conversation_trace.append(pending_gate_msg)
        if pending_contract_msg is not None:
            messages.append(pending_contract_msg)
            agent_result.conversation_trace.append(pending_contract_msg)
        if pending_control_loop_msg is not None:
            messages.append(pending_control_loop_msg)
            agent_result.conversation_trace.append(pending_control_loop_msg)

        turn_arg_coercions = 0
        turn_arg_coercion_calls = 0
        turn_arg_validation_rejections = 0
        for record in records:
            coercions = getattr(record, "arg_coercions", None) or []
            if coercions:
                turn_arg_coercions += len(coercions)
                turn_arg_coercion_calls += 1
            if record.error and "validation error:" in record.error.lower():
                turn_arg_validation_rejections += 1

        if turn_arg_coercions:
            tool_arg_coercions += turn_arg_coercions
            tool_arg_coercion_calls += turn_arg_coercion_calls
            warning = (
                "TOOL_ARG_COERCION: turn "
                f"{turn + 1}/{max_turns} applied {turn_arg_coercions} coercion(s) "
                f"across {turn_arg_coercion_calls} call(s)"
            )
            agent_result.warnings.append(warning)
            logger.warning(warning)
        if turn_arg_validation_rejections:
            tool_arg_validation_rejections += turn_arg_validation_rejections
            warning = (
                "TOOL_ARG_VALIDATION: turn "
                f"{turn + 1}/{max_turns} rejected {turn_arg_validation_rejections} "
                "tool call(s) due to argument/schema mismatch"
            )
            agent_result.warnings.append(warning)
            logger.warning(warning)

        submit_error_this_turn = False
        submit_errors: list[str] = []
        submit_reason_codes_this_turn: list[str] = []
        submit_needs_new_evidence_signal = False
        evidence_digest_before_turn = _evidence_digest(evidence_pointer_labels)
        for record in executed_records:
            evidence_pointer_labels.update(_tool_evidence_pointer_labels(record))
        evidence_pointer_count = len(evidence_pointer_labels)
        evidence_digest_after_turn = _evidence_digest(evidence_pointer_labels)
        new_evidence_this_turn = evidence_digest_after_turn != evidence_digest_before_turn
        if new_evidence_this_turn:
            evidence_digest_change_count += 1
        if (
            new_evidence_this_turn
            and submit_requires_new_evidence
            and submit_evidence_digest_at_last_failure is not None
            and evidence_digest_after_turn != submit_evidence_digest_at_last_failure
        ):
            submit_requires_new_evidence = False
            submit_evidence_digest_at_last_failure = None

        evidence_tools_executed = [
            rec.tool
            for rec in executed_records
            if _is_evidence_tool_name(rec.tool) and not rec.error
        ]
        if evidence_tools_executed:
            evidence_turns_total += 1
            if new_evidence_this_turn:
                evidence_turns_with_new_evidence += 1
                retrieval_stagnation_streak = 0
                retrieval_stagnation_alerted_for_current_streak = False
            else:
                evidence_turns_without_new_evidence += 1
                retrieval_stagnation_streak += 1
        else:
            retrieval_stagnation_streak = 0
            retrieval_stagnation_alerted_for_current_streak = False
        if retrieval_stagnation_streak > retrieval_stagnation_streak_max:
            retrieval_stagnation_streak_max = retrieval_stagnation_streak

        # Persist deficit snapshot for next-turn deficit-progress evaluation.
        prev_turn_had_evidence_tools = bool(evidence_tools_executed)
        prev_turn_deficit_digest = current_turn_deficit_digest

        if (
            retrieval_stagnation_streak >= retrieval_stagnation_turns
            and not retrieval_stagnation_alerted_for_current_streak
        ):
            retrieval_stagnation_triggered = True
            if retrieval_stagnation_turn is None:
                retrieval_stagnation_turn = turn + 1
            stagnation_event_code = (
                EVENT_CODE_RETRIEVAL_STAGNATION
                if retrieval_stagnation_action == "force_final"
                else EVENT_CODE_RETRIEVAL_STAGNATION_OBSERVED
            )
            failure_event_codes.append(stagnation_event_code)
            warning = (
                "RETRIEVAL_STAGNATION: "
                f"{retrieval_stagnation_streak} consecutive evidence turns produced no new evidence refs."
            )
            agent_result.warnings.append(warning)
            logger.warning(warning)
            _emit_foundation_event(
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
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "[SYSTEM: Evidence has stagnated across consecutive retrieval turns. "
                            "Stop repeating equivalent searches. Verify a different bridge entity, "
                            "run a conversion tool, or submit your best answer now.]"
                        ),
                    }
                )
                agent_result.conversation_trace.append(messages[-1])
                force_final_reason = "retrieval_stagnation"
                retrieval_stagnation_alerted_for_current_streak = True
                break
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "[SYSTEM: Evidence stagnation observed. Do NOT repeat equivalent retrieval. "
                        "Pivot strategy now (different bridge, conversion, or graph path) and continue.]"
                    ),
                }
            )
            agent_result.conversation_trace.append(messages[-1])
            retrieval_stagnation_alerted_for_current_streak = True

        for record in records:
            if record.tool != "submit_answer":
                continue
            submit_answer_call_count += 1
            arg_answer = record.arguments.get("answer") if isinstance(record.arguments, dict) else None
            if (
                isinstance(arg_answer, str)
                and arg_answer.strip()
                and not _FORCED_REFUSAL_RE.match(arg_answer.strip())
            ):
                fallback_submit_guess_value = arg_answer.strip()
            if record.error:
                submit_error_this_turn = True
                submit_errors.append(str(record.error))
                continue
            parsed = _parse_record_result_json(record)
            if parsed is not None:
                status = str(parsed.get("status", "")).strip().lower()
                if status and status not in {"submitted", "ok", "success"}:
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
                        submit_reason_codes_this_turn.append(reason_code)
                    err_parts = [f"submit_answer not accepted (status={status})"]
                    if reason_code:
                        err_parts.append(f"reason_code={reason_code}")
                    if detail:
                        err_parts.append(detail)
                    submit_errors.append(" | ".join(err_parts))
                    continue

            submit_answer_succeeded = True
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
            evidence_fix_hint = (
                " Validator requires NEW evidence refs before retry. "
                "Run at least one non-control evidence tool call that yields new "
                "chunk/entity-backed evidence before submit."
                if submit_requires_new_evidence else
                ""
            )
            retry_submit_msg = {
                "role": "user",
                "content": (
                    "[SYSTEM: submit_answer failed. "
                    + evidence_fix_hint
                    + (
                        "Do NOT use refusal text (cannot, unknown, insufficient, no such, not found). "
                        "Submit the single best factual guess from retrieved evidence. "
                        if refusal_blocked else
                        ""
                    )
                    + "Call submit_answer again with answer as a short fact only "
                    "(name/date/number/yes/no, <=8 words).]"
                ),
            }
            messages.append(retry_submit_msg)
            agent_result.conversation_trace.append(retry_submit_msg)

        # --- TODO state injection ---
        # Cache status_line from todo_write results; inject on turns without todo_write.
        todo_write_called_this_turn = False
        for record in records:
            if record.tool == "todo_write" and not record.error:
                todo_write_called_this_turn = True
                parsed = _parse_record_result_json(record)
                if isinstance(parsed, dict):
                    sl = parsed.get("status_line")
                    if isinstance(sl, str) and sl.strip():
                        _last_todo_status_line = sl.strip()
        if not todo_write_called_this_turn and _last_todo_status_line:
            todo_inject_msg = {
                "role": "user",
                "content": f"[TODO_STATE] {_last_todo_status_line}",
            }
            messages.append(todo_inject_msg)
            agent_result.conversation_trace.append(todo_inject_msg)

        if tool_calls_this_turn and not tool_calls_to_execute:
            zero_exec_tool_turn_streak += 1
        else:
            zero_exec_tool_turn_streak = 0
        if zero_exec_tool_turn_streak > max_zero_exec_tool_turn_streak:
            max_zero_exec_tool_turn_streak = zero_exec_tool_turn_streak

        if zero_exec_tool_turn_streak >= DEFAULT_TOOL_CALL_STALL_TURNS:
            blocked_tools = sorted({
                str(tc.get("function", {}).get("name", "")).strip() or "<unknown>"
                for tc in tool_calls_this_turn
                if isinstance(tc, dict)
            })
            warning = (
                "CONTROL_CHURN: consecutive turns with tool calls but no executable calls "
                f"({zero_exec_tool_turn_streak} turns). Forcing final answer."
            )
            agent_result.warnings.append(warning)
            logger.warning(warning)
            failure_event_codes.append(EVENT_CODE_CONTROL_CHURN_THRESHOLD)
            stall_msg = {
                "role": "user",
                "content": (
                    "[SYSTEM: Repeated non-executable tool-call loop detected "
                    f"({zero_exec_tool_turn_streak} turns). Stop calling tools and "
                    "submit your best answer now from existing evidence.]"
                ),
            }
            messages.append(stall_msg)
            agent_result.conversation_trace.append(stall_msg)
            _emit_foundation_event(
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
                            "contract_rejections": len(contract_rejected_records),
                            "suppressed_calls": len(suppressed_records),
                        },
                        "bindings": dict(available_bindings),
                    },
                    "outputs": {"artifact_ids": [], "payload_hashes": []},
                    "failure": {
                        "error_code": EVENT_CODE_CONTROL_CHURN_THRESHOLD,
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

        logger.debug(
            "Agent turn %d/%d: %d tool calls",
            turn + 1, max_turns, len(tool_calls_this_turn),
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
            emit_foundation_event=_emit_foundation_event,
        )
        final_content = forced_final_result.final_content
        final_finish_reason = forced_final_result.final_finish_reason
        finalization_primary_model = forced_final_result.finalization_primary_model
        forced_final_attempts = forced_final_result.forced_final_attempts
        forced_final_circuit_breaker_opened = forced_final_result.forced_final_circuit_breaker_opened
        finalization_fallback_used = forced_final_result.finalization_fallback_used
        finalization_fallback_succeeded = forced_final_result.finalization_fallback_succeeded
        finalization_events = list(forced_final_result.finalization_events)
        finalization_fallback_attempts = list(forced_final_result.finalization_fallback_attempts)
        failure_event_codes.extend(forced_final_result.failure_event_codes)
        context_tool_result_clearings += forced_final_result.context_tool_result_clearings_delta
        context_tool_results_cleared += forced_final_result.context_tool_results_cleared_delta
        context_tool_result_cleared_chars += (
            forced_final_result.context_tool_result_cleared_chars_delta
        )
        context_compactions += forced_final_result.context_compactions_delta
        context_compacted_messages += forced_final_result.context_compacted_messages_delta
        context_compacted_chars += forced_final_result.context_compacted_chars_delta
        total_cost += forced_final_result.total_cost_delta
        agent_result.total_input_tokens += forced_final_result.total_input_tokens_delta
        agent_result.total_output_tokens += forced_final_result.total_output_tokens_delta
        agent_result.total_cached_tokens += forced_final_result.total_cached_tokens_delta
        agent_result.total_cache_creation_tokens += (
            forced_final_result.total_cache_creation_tokens_delta
        )
        agent_result.turns += forced_final_result.turns_delta

    required_submit_missing = requires_submit_answer and not submit_answer_succeeded
    submit_forced_retry_on_budget_exhaustion = False
    submit_forced_accept_on_budget_exhaustion = False
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
            submit_answer_call_count += 1
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
            agent_result.warnings.append(warning)
            logger.warning(warning)
        # Accept the best available answer when budget/turns are exhausted.
        # Sources in priority order:
        #   1. Agent's last explicit submit_answer guess (non-refusal)
        #   2. Forced-final text extraction (only when forced-final succeeded)
        # When forced-final errored (finish_reason="error"), its content is an
        # error message, not an answer — never extract from it.
        _has_fallback_guess = (
            isinstance(fallback_submit_guess_value, str)
            and bool(fallback_submit_guess_value.strip())
        )
        _forced_final_succeeded = final_finish_reason != "error"
        _has_final_content = _forced_final_succeeded and bool(final_content.strip())
        if accept_forced_answer_on_max_tool_calls and (_has_final_content or _has_fallback_guess):
            submit_forced_accept_on_budget_exhaustion = True
            submit_answer_succeeded = True
            if _has_fallback_guess and fallback_submit_guess_value is not None:
                normalized_forced_answer = fallback_submit_guess_value.strip()
            elif _has_final_content:
                normalized_forced_answer = _normalize_forced_final_answer(final_content)
            else:
                normalized_forced_answer = ""
            submitted_answer_value = (
                submitted_answer_value
                or normalized_forced_answer
                or (final_content.strip() if _forced_final_succeeded else "")
            )
            required_submit_missing = False
            if forced_exhaustion_reason == "budget":
                warning = (
                    "SUBMIT_FORCED_ACCEPT_BUDGET_EXHAUSTION: accepted forced-final answer "
                    "without grounding validation because tool budget was exhausted."
                )
                failure_event_codes.append(EVENT_CODE_SUBMIT_FORCED_ACCEPT_BUDGET_EXHAUSTION)
            elif forced_exhaustion_reason == "turns":
                warning = (
                    "SUBMIT_FORCED_ACCEPT_TURN_EXHAUSTION: accepted forced-final answer "
                    "without grounding validation because max turns were exhausted."
                )
                failure_event_codes.append(EVENT_CODE_SUBMIT_FORCED_ACCEPT_TURN_EXHAUSTION)
            else:
                warning = (
                    "SUBMIT_FORCED_ACCEPT_FORCED_FINAL: accepted forced-final answer "
                    "without grounding validation after forced-final termination "
                    f"('{forced_exhaustion_reason}')."
                )
                failure_event_codes.append(EVENT_CODE_SUBMIT_FORCED_ACCEPT_FORCED_FINAL)
            agent_result.warnings.append(warning)
            logger.warning(warning)

    if required_submit_missing:
        submit_failure_code = (
            EVENT_CODE_REQUIRED_SUBMIT_NOT_ATTEMPTED
            if submit_answer_call_count == 0
            else EVENT_CODE_REQUIRED_SUBMIT_NOT_ACCEPTED
        )
        if not failure_event_codes:
            # Classify missing required submit as terminal/policy only when no
            # stronger prior failure signal has already explained the run.
            failure_event_codes.append(submit_failure_code)
        warning = (
            "REQUIRED_SUBMIT: submit_answer tool is available but no accepted submit was recorded. "
            f"submit_answer_call_count={submit_answer_call_count}."
        )
        agent_result.warnings.append(warning)
        logger.warning(warning)
        _emit_foundation_event(
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
                        "submit_answer_call_count": submit_answer_call_count,
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

    submit_validator_accepted = bool(
        submit_answer_succeeded and not submit_forced_accept_on_budget_exhaustion
    )
    submit_completion_mode = (
        "grounded_submit"
        if submit_validator_accepted else
        "forced_terminal_accept"
        if submit_forced_accept_on_budget_exhaustion else
        "missing_required_submit"
        if required_submit_missing else
        "no_submit_required"
        if not requires_submit_answer else
        "unknown"
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
    agent_result.metadata["forced_final_circuit_breaker_threshold"] = forced_final_circuit_breaker_threshold
    agent_result.metadata["forced_final_breaker_effective"] = forced_final_breaker_effective
    agent_result.metadata["force_submit_retry_on_max_tool_calls"] = force_submit_retry_on_max_tool_calls
    agent_result.metadata["accept_forced_answer_on_max_tool_calls"] = accept_forced_answer_on_max_tool_calls
    agent_result.metadata["forced_final_circuit_breaker_opened"] = forced_final_circuit_breaker_opened
    agent_result.metadata["finalization_primary_model"] = finalization_primary_model
    agent_result.metadata["finalization_fallback_models"] = list(finalization_fallback_models)
    agent_result.metadata["finalization_fallback_used"] = finalization_fallback_used
    agent_result.metadata["finalization_fallback_attempt_count"] = finalization_summary.finalization_fallback_attempt_count
    agent_result.metadata["finalization_fallback_usage_rate"] = finalization_summary.finalization_fallback_usage_rate
    agent_result.metadata["finalization_fallback_succeeded"] = finalization_fallback_succeeded
    agent_result.metadata["finalization_fallback_attempts"] = list(finalization_fallback_attempts)
    agent_result.metadata["finalization_attempt_counts_by_model"] = finalization_summary.finalization_attempt_counts_by_model
    agent_result.metadata["finalization_failure_counts_by_model"] = finalization_summary.finalization_failure_counts_by_model
    agent_result.metadata["finalization_success_counts_by_model"] = finalization_summary.finalization_success_counts_by_model
    agent_result.metadata["finalization_failure_code_counts"] = finalization_summary.finalization_failure_code_counts
    agent_result.metadata["provider_empty_attempt_counts_by_model"] = finalization_summary.provider_empty_attempt_counts_by_model
    agent_result.metadata["finalization_breaker_open_rate"] = finalization_summary.finalization_breaker_open_rate
    agent_result.metadata["finalization_breaker_open_by_model"] = finalization_summary.finalization_breaker_open_by_model
    agent_result.metadata["finalization_events"] = list(finalization_events)
    agent_result.metadata["tool_calls_used"] = len(agent_result.tool_calls)
    agent_result.metadata["budgeted_tool_calls_used"] = _count_budgeted_records(agent_result.tool_calls)
    agent_result.metadata["budget_exempt_tools"] = sorted(BUDGET_EXEMPT_TOOL_NAMES)
    agent_result.metadata["requires_submit_answer"] = requires_submit_answer
    agent_result.metadata["submit_answer_call_count"] = submit_answer_call_count
    agent_result.metadata["submit_answer_attempted"] = submit_answer_call_count > 0
    agent_result.metadata["submit_answer_succeeded"] = submit_answer_succeeded
    agent_result.metadata["submit_validator_accepted"] = submit_validator_accepted
    agent_result.metadata["required_submit_missing"] = required_submit_missing
    agent_result.metadata["required_submit_satisfied"] = required_submit_satisfied
    agent_result.metadata["submitted_answer_value"] = submitted_answer_value
    agent_result.metadata["submit_forced_retry_on_budget_exhaustion"] = submit_forced_retry_on_budget_exhaustion
    agent_result.metadata["submit_forced_accept_on_budget_exhaustion"] = submit_forced_accept_on_budget_exhaustion
    agent_result.metadata["submit_completion_mode"] = submit_completion_mode
    agent_result.metadata["answer_present"] = answer_present
    agent_result.metadata["grounded_completed"] = grounded_completed
    agent_result.metadata["forced_terminal_accepted"] = forced_terminal_accepted
    agent_result.metadata["reliability_completed"] = reliability_completed
    agent_result.metadata["run_completed"] = run_completed
    agent_result.metadata["require_tool_reasoning"] = require_tool_reasoning
    agent_result.metadata["rejected_missing_reasoning_calls"] = rejected_missing_reasoning_calls
    agent_result.metadata["control_loop_suppressed_calls"] = control_loop_suppressed_calls
    agent_result.metadata["submit_validation_reason_counts"] = dict(
        sorted(submit_validation_reason_counts.items())
    )
    agent_result.metadata["tool_call_turns_total"] = tool_call_turns_total
    agent_result.metadata["tool_call_empty_text_turns"] = tool_call_empty_text_turns
    agent_result.metadata["responses_tool_call_empty_text_turns"] = responses_tool_call_empty_text_turns
    agent_result.metadata["tool_call_empty_text_turn_ratio"] = (
        (tool_call_empty_text_turns / tool_call_turns_total)
        if tool_call_turns_total else 0.0
    )
    agent_result.metadata["tool_arg_coercions"] = tool_arg_coercions
    agent_result.metadata["tool_arg_coercion_calls"] = tool_arg_coercion_calls
    agent_result.metadata["tool_arg_validation_rejections"] = tool_arg_validation_rejections
    agent_result.metadata["tool_result_keep_recent"] = tool_result_keep_recent
    agent_result.metadata["tool_result_context_preview_chars"] = tool_result_context_preview_chars
    agent_result.metadata["active_artifact_context_enabled"] = active_artifact_context_enabled
    agent_result.metadata["active_artifact_context_max_handles"] = active_artifact_context_max_handles
    agent_result.metadata["active_artifact_context_max_chars"] = active_artifact_context_max_chars
    agent_result.metadata["active_artifact_context_updates"] = artifact_context_updates
    agent_result.metadata["active_artifact_context_chars"] = artifact_context_chars
    agent_result.metadata["context_tool_result_clearings"] = context_tool_result_clearings
    agent_result.metadata["context_tool_results_cleared"] = context_tool_results_cleared
    agent_result.metadata["context_tool_result_cleared_chars"] = context_tool_result_cleared_chars
    agent_result.metadata["tool_result_metadata_tracked"] = len(tool_result_metadata_by_id)
    agent_result.metadata["context_compactions"] = context_compactions
    agent_result.metadata["context_compacted_messages"] = context_compacted_messages
    agent_result.metadata["context_compacted_chars"] = context_compacted_chars
    agent_result.metadata["enforce_tool_contracts"] = enforce_tool_contracts
    agent_result.metadata["progressive_tool_disclosure"] = progressive_tool_disclosure
    agent_result.metadata["suppress_control_loop_calls"] = suppress_control_loop_calls
    agent_result.metadata["tool_gate_rejections"] = gate_rejected_calls
    agent_result.metadata["tool_gate_violation_events"] = gate_violation_events
    agent_result.metadata["tool_contract_rejections"] = contract_rejected_calls
    agent_result.metadata["tool_contract_violation_events"] = contract_violation_events
    agent_result.metadata["tool_contracts_declared"] = sorted(normalized_tool_contracts.keys())
    agent_result.metadata["runtime_tool_names"] = [runtime_artifact_tool_name]
    agent_result.metadata["runtime_artifact_registry_size"] = len(runtime_artifact_registry_by_id)
    agent_result.metadata["runtime_artifact_registry_ids"] = sorted(runtime_artifact_registry_by_id.keys())
    agent_result.metadata["handle_input_resolution_count"] = handle_input_resolution_count
    agent_result.metadata["handle_input_resolved_artifact_count"] = handle_input_resolved_artifact_count
    agent_result.metadata["initial_artifacts"] = initial_artifact_snapshot
    agent_result.metadata["available_artifacts_final"] = sorted(available_artifacts)
    agent_result.metadata["initial_capabilities"] = initial_capability_snapshot
    agent_result.metadata["available_capabilities_final"] = _capability_state_snapshot(available_capabilities)
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
    agent_result.metadata["adoption_profile_requested"] = adoption_assessment.requested_profile
    agent_result.metadata["adoption_profile_effective"] = adoption_assessment.effective_profile
    agent_result.metadata["adoption_profile_enforce"] = adoption_assessment.enforce
    agent_result.metadata["adoption_profile_satisfied"] = adoption_assessment.satisfied
    agent_result.metadata["adoption_profile_violations"] = list(adoption_assessment.violations)
    agent_result.metadata["tool_disclosure_turns"] = tool_disclosure_turns
    agent_result.metadata["tool_disclosure_hidden_total"] = tool_disclosure_hidden_total
    agent_result.metadata["tool_disclosure_unavailable_msgs"] = tool_disclosure_unavailable_msgs
    agent_result.metadata["tool_disclosure_unavailable_reason_chars"] = tool_disclosure_unavailable_reason_chars
    agent_result.metadata["tool_disclosure_unavailable_reason_tokens_est"] = tool_disclosure_unavailable_reason_tokens_est
    agent_result.metadata["tool_disclosure_repair_suggestions"] = tool_disclosure_repair_suggestions
    agent_result.metadata["lane_closure_analysis"] = lane_closure_analysis
    agent_result.metadata["no_legal_noncontrol_turns"] = no_legal_noncontrol_turns
    agent_result.metadata["deficit_no_progress_streak_max"] = max_deficit_no_progress_streak
    agent_result.metadata["deficit_no_progress_nudges"] = deficit_no_progress_nudges
    agent_result.metadata["max_zero_exec_tool_turn_streak"] = max_zero_exec_tool_turn_streak
    agent_result.metadata["retrieval_no_hits_count"] = retrieval_no_hits_count
    agent_result.metadata["retrieval_stagnation_turns"] = retrieval_stagnation_turns
    agent_result.metadata["retrieval_stagnation_action"] = retrieval_stagnation_action
    agent_result.metadata["retrieval_stagnation_streak"] = retrieval_stagnation_streak
    agent_result.metadata["retrieval_stagnation_streak_max"] = retrieval_stagnation_streak_max
    agent_result.metadata["retrieval_stagnation_triggered"] = retrieval_stagnation_triggered
    agent_result.metadata["retrieval_stagnation_turn"] = retrieval_stagnation_turn
    agent_result.metadata["evidence_digest_change_count"] = evidence_digest_change_count
    agent_result.metadata["evidence_turns_total"] = evidence_turns_total
    agent_result.metadata["evidence_turns_with_new_evidence"] = evidence_turns_with_new_evidence
    agent_result.metadata["evidence_turns_without_new_evidence"] = evidence_turns_without_new_evidence
    agent_result.metadata["evidence_pointer_count"] = evidence_pointer_count
    agent_result.metadata["evidence_digest_basis"] = "canonical_evidence_pointers"
    agent_result.metadata["failure_event_codes"] = list(failure_event_codes)
    agent_result.metadata["failure_event_code_counts"] = failure_summary.failure_event_code_counts
    agent_result.metadata["failure_event_class_counts"] = failure_summary.failure_event_class_counts
    agent_result.metadata["provider_failure_event_code_counts"] = failure_summary.provider_failure_event_code_counts
    agent_result.metadata["provider_failure_event_total"] = failure_summary.provider_failure_event_total
    agent_result.metadata["provider_caused_incompletion"] = failure_summary.provider_caused_incompletion
    agent_result.metadata["primary_failure_class"] = failure_summary.primary_failure_class
    agent_result.metadata["secondary_failure_classes"] = failure_summary.secondary_failure_classes
    agent_result.metadata["first_terminal_failure_event_code"] = failure_summary.first_terminal_failure_event_code
    agent_result.metadata["failure_priority_order"] = list(_PRIMARY_FAILURE_PRIORITY)
    agent_result.metadata["terminal_failure_event_codes"] = sorted(_TERMINAL_FAILURE_EVENT_CODES)
    agent_result.metadata["autofilled_tool_reasoning_calls"] = autofilled_tool_reasoning_calls
    agent_result.metadata["autofilled_tool_reasoning_by_tool"] = dict(autofilled_tool_reasoning_by_tool)
    agent_result.metadata["evidence_digest"] = final_evidence_digest
    agent_result.metadata["submit_evidence_digest_at_last_failure"] = submit_evidence_digest_at_last_failure
    agent_result.metadata["foundation_event_count"] = len(foundation_events)
    agent_result.metadata["foundation_event_types"] = dict(foundation_event_types)
    agent_result.metadata["foundation_event_validation_errors"] = foundation_event_validation_errors
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
    return final_content, final_finish_reason
