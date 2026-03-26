"""MCP agent loop for llm_client.

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
    from llm_client.core.client import _clean_schema_for_gemini
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
    from llm_client.core.client import acall_llm
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
    error_budget: "AgentErrorBudget | None" = None,
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

        try:
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

                error_budget=error_budget,
            )
        except asyncio.CancelledError:
            final_content = agent_result.metadata.get("last_content", "")
            final_finish_reason = "cancelled"
            logger.info(
                "Agent loop cancelled — returning partial result "
                "(turns=%d, tool_calls=%d, content_len=%d)",
                agent_result.turns,
                len(agent_result.tool_calls),
                len(final_content),
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

            try:
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

                    error_budget=error_budget,
                )
            except asyncio.CancelledError:
                final_finish_reason = "cancelled"
                logger.info(
                    "Agent loop cancelled — returning partial result "
                    "(turns=%d, tool_calls=%d)",
                    agent_result.turns,
                    len(agent_result.tool_calls),
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


from llm_client.agent.agent_contracts import AgentErrorBudget
from llm_client.agent.mcp_turn_execution import _agent_loop as _agent_loop


# ---------------------------------------------------------------------------
# Direct Python Tool Loop
# ---------------------------------------------------------------------------


async def _acall_with_tools(
    model: str,
    messages: list[dict[str, Any]],
    python_tools: list[Any],
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_tool_calls: int | None = DEFAULT_MAX_TOOL_CALLS,
    require_tool_reasoning: bool = DEFAULT_REQUIRE_TOOL_REASONING,
    tool_result_max_length: int = DEFAULT_TOOL_RESULT_MAX_LENGTH,
    max_message_chars: int = DEFAULT_MAX_MESSAGE_CHARS,
    enforce_tool_contracts: bool = DEFAULT_ENFORCE_TOOL_CONTRACTS,
    progressive_tool_disclosure: bool = DEFAULT_PROGRESSIVE_TOOL_DISCLOSURE,
    suppress_control_loop_calls: bool = DEFAULT_SUPPRESS_CONTROL_LOOP_CALLS,
    tool_contracts: dict[str, dict[str, Any]] | None = None,
    initial_artifacts: list[str] | tuple[str, ...] | None = DEFAULT_INITIAL_ARTIFACTS,
    initial_bindings: dict[str, Any] | None = None,
    timeout: int = 60,
    error_budget: "AgentErrorBudget | None" = None,
    **kwargs: Any,
) -> LLMCallResult:
    """Run a tool-calling agent loop with direct Python functions.

    Same loop as _acall_with_mcp but calls Python functions in-process
    instead of going through MCP subprocess/stdio/JSON-RPC.

    Args:
        model: Any litellm model string (NOT an agent model).
        messages: Initial messages (system + user).
        python_tools: List of typed Python callables (sync or async).
        max_turns: Maximum loop iterations.
        max_tool_calls: Maximum tool calls before final forced answer. None disables.
        require_tool_reasoning: If True, reject tool calls missing tool_reasoning.
        tool_result_max_length: Max chars per tool result.
        progressive_tool_disclosure: If True, hide tools whose contracts are not currently satisfiable.
        suppress_control_loop_calls: If True, suppress repeated submit/todo control calls.
        initial_bindings: Binding state available before any tool call.
        timeout: Per-turn LLM call timeout.
        **kwargs: Passed through to acall_llm.
    """
    from llm_client.tools.tool_utils import execute_direct_tool_calls, prepare_direct_tools

    tool_map, openai_tools = prepare_direct_tools(python_tools)

    if not openai_tools:
        raise ValueError("python_tools list is empty — no tools to call.")

    agent_result = MCPAgentResult()
    messages = list(messages)  # don't mutate caller's list

    async def _direct_executor(
        tool_calls: list[dict[str, Any]], max_len: int,
    ) -> tuple[list[MCPToolCallRecord], list[dict[str, Any]]]:
        return await execute_direct_tool_calls(
            tool_calls,
            tool_map,
            max_len,
            require_tool_reasoning=require_tool_reasoning,
        )

    try:
        final_content, final_finish_reason = await _agent_loop(
            model, messages, openai_tools,
            agent_result,
            _direct_executor,
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

            error_budget=error_budget,
        )
    except asyncio.CancelledError:
        final_content = agent_result.metadata.get("last_content", "")
        final_finish_reason = "cancelled"
        logger.info(
            "Agent loop cancelled — returning partial result "
            "(turns=%d, tool_calls=%d, content_len=%d)",
            agent_result.turns,
            len(agent_result.tool_calls),
            len(final_content),
        )

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


async def acall_with_mcp_runtime(
    model: str,
    messages: list[dict[str, Any]],
    mcp_servers: dict[str, dict[str, Any]] | None = None,
    **kwargs: Any,
) -> LLMCallResult:
    """Run the optional MCP-backed tool runtime through an explicit adapter.

    Core runtime code should call this adapter instead of depending on the
    private ``_acall_with_mcp`` implementation directly. Tests that patch the
    private function remain compatible because this adapter delegates at call
    time.
    """

    return await _acall_with_mcp(model, messages, mcp_servers=mcp_servers, **kwargs)


async def acall_with_python_tools_runtime(
    model: str,
    messages: list[dict[str, Any]],
    python_tools: list[Any],
    **kwargs: Any,
) -> LLMCallResult:
    """Run the optional direct Python-tool runtime through an explicit adapter.

    Core runtime code should call this adapter instead of depending on the
    private ``_acall_with_tools`` implementation directly. Tests that patch the
    private function remain compatible because this adapter delegates at call
    time.
    """

    return await _acall_with_tools(model, messages, python_tools=python_tools, **kwargs)
