"""Capability state, contract validation, and tool disclosure wrappers for the MCP agent loop.

Thin delegation layer that binds the agent_contracts, agent_artifacts,
and agent_disclosure modules to the mcp_agent constants and types.  All
public names are re-exported from mcp_agent.py for backward compatibility.
"""

from __future__ import annotations

from typing import Any

from llm_client.agent.agent_artifacts import (
    _artifact_handle_summaries as _agent_artifact_handle_summaries,
    _build_active_artifact_context_content as _agent_build_active_artifact_context_content,
    _collect_recent_artifact_handles as _agent_collect_recent_artifact_handles,
    _upsert_active_artifact_context_message as _agent_upsert_active_artifact_context_message,
)
from llm_client.agent.agent_contracts import (
    CapabilityRequirement,
    _analyze_lane_closure as _agent_analyze_lane_closure,
    _apply_handle_input_injections as _agent_apply_handle_input_injections,
    _artifact_output_state_from_payload as _agent_artifact_output_state_from_payload,
    _capability_requirement_from_raw as _agent_capability_requirement_from_raw,
    _capability_state_add as _agent_capability_state_add,
    _capability_state_has as _agent_capability_state_has,
    _capability_state_snapshot as _agent_capability_state_snapshot,
    _contract_declares_no_artifact_prereqs as _agent_contract_declares_no_artifact_prereqs,
    _contract_output_capabilities as _agent_contract_output_capabilities,
    _contract_outputs as _agent_contract_outputs,
    _effective_contract_requirements as _agent_effective_contract_requirements,
    _find_repair_tools_for_missing_requirements as _agent_find_repair_tools_for_missing_requirements,
    _full_bindings_spec as _agent_full_bindings_spec,
    _full_bindings_state_hash as _agent_full_bindings_state_hash,
    _hard_bindings_spec as _agent_hard_bindings_spec,
    _hard_bindings_state_hash as _agent_hard_bindings_state_hash,
    _infer_output_capabilities as _agent_infer_output_capabilities,
    _is_control_tool_name as _agent_is_control_tool_name,
    _normalize_artifact_kind as _agent_normalize_artifact_kind,
    _short_requirement as _agent_short_requirement,
    _tool_declares_no_artifact_prereqs as _agent_tool_declares_no_artifact_prereqs,
    _validate_tool_contract_call as _agent_validate_tool_contract_call,
)
from llm_client.agent.agent_disclosure import (
    _deficit_labels_from_hidden_entries as _agent_deficit_labels_from_hidden_entries,
    _disclosure_message as _agent_disclosure_message,
    _disclosure_reason_from_entry as _agent_disclosure_reason_from_entry,
    _filter_tools_for_disclosure as _agent_filter_tools_for_disclosure,
)
from llm_client.agent.mcp_tools import (
    BUDGET_EXEMPT_TOOL_NAMES,
    RUNTIME_ARTIFACT_READ_TOOL_NAME,
    _extract_tool_call_args,
    _is_budget_exempt_tool,
    _parse_record_result_json_value,
    _set_tool_call_args,
)
from llm_client.agent.mcp_context import _trim_text
from llm_client.tools.tool_runtime_common import (
    MCPToolCallRecord,
    TOOL_REASONING_FIELD,
    normalize_tool_contracts as _shared_normalize_tool_contracts,
)

# Import event code constants lazily to avoid circular imports.
# These are string constants defined in mcp_agent.py.
EVENT_CODE_TOOL_VALIDATION_MISSING_PREREQUISITE = "TOOL_VALIDATION_REJECTED_MISSING_PREREQUISITE"
EVENT_CODE_TOOL_VALIDATION_MISSING_CAPABILITY = "TOOL_VALIDATION_REJECTED_MISSING_CAPABILITY"
EVENT_CODE_TOOL_VALIDATION_BINDING_CONFLICT = "TOOL_VALIDATION_REJECTED_BINDING_CONFLICT"

DEFAULT_TOOL_DISCLOSURE_MAX_UNAVAILABLE: int = 10
DEFAULT_TOOL_DISCLOSURE_MAX_MISSING_PER_TOOL: int = 2
DEFAULT_TOOL_DISCLOSURE_MAX_REPAIR_TOOLS: int = 2
DEFAULT_TOOL_DISCLOSURE_REASON_MAX_CHARS: int = 220


# ---------------------------------------------------------------------------
# Capability / artifact state
# ---------------------------------------------------------------------------

def _capability_requirement_from_raw(raw: Any) -> CapabilityRequirement | None:
    """Parse a raw capability requirement value into a typed object."""
    return _agent_capability_requirement_from_raw(raw)


def _normalize_artifact_kind(kind: Any) -> str | None:
    """Normalize an artifact kind value to a canonical string."""
    return _agent_normalize_artifact_kind(kind)


def _capability_state_add(
    state: dict[str, set[tuple[str | None, str | None, str | None]]],
    *,
    kind: str | None,
    ref_type: str | None = None,
    namespace: str | None = None,
    bindings_hash: str | None = None,
) -> bool:
    """Add a capability to the mutable state dict."""
    return _agent_capability_state_add(
        state,
        kind=kind,
        ref_type=ref_type,
        namespace=namespace,
        bindings_hash=bindings_hash,
    )


def _capability_state_has(
    state: dict[str, set[tuple[str | None, str | None, str | None]]],
    req: CapabilityRequirement,
) -> bool:
    """Check whether a capability requirement is satisfied by current state."""
    return _agent_capability_state_has(state, req)


def _capability_state_snapshot(
    state: dict[str, set[tuple[str | None, str | None, str | None]]],
) -> list[dict[str, str]]:
    """Return a JSON-serializable snapshot of current capability state."""
    return _agent_capability_state_snapshot(state)


def _hard_bindings_spec(bindings: dict[str, str | None]) -> dict[str, str | None]:
    """Extract hard-binding keys from the bindings dict."""
    return _agent_hard_bindings_spec(bindings)


def _full_bindings_spec(bindings: dict[str, str | None]) -> dict[str, str | None]:
    """Extract all binding keys from the bindings dict."""
    return _agent_full_bindings_spec(bindings)


def _hard_bindings_state_hash(bindings: dict[str, str | None]) -> str:
    """Hash of hard-binding state for change detection."""
    return _agent_hard_bindings_state_hash(bindings)


def _full_bindings_state_hash(bindings: dict[str, str | None]) -> str:
    """Hash of full binding state for change detection."""
    return _agent_full_bindings_state_hash(bindings)


def _infer_output_capabilities(
    *,
    tool_name: str,
    parsed_args: dict[str, Any] | None,
    produced_artifacts: set[str],
    available_bindings: dict[str, str | None],
) -> list[CapabilityRequirement]:
    """Infer output capabilities from a successful tool call."""
    return _agent_infer_output_capabilities(
        tool_name=tool_name,
        parsed_args=parsed_args,
        produced_artifacts=produced_artifacts,
        available_bindings=available_bindings,
    )


def _artifact_output_state_from_record(
    record: MCPToolCallRecord,
    *,
    fallback_bindings: dict[str, str | None],
) -> tuple[list[dict[str, Any]], set[str], list[CapabilityRequirement], list[str], dict[str, str | None]]:
    """Extract artifact output state from a tool call record."""
    parsed = _parse_record_result_json_value(record)
    return _agent_artifact_output_state_from_payload(
        parsed,
        fallback_bindings=fallback_bindings,
    )


def _artifact_handle_summaries(envelopes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize artifact handles from envelopes."""
    return _agent_artifact_handle_summaries(envelopes)


def _collect_recent_artifact_handles(
    tool_result_metadata_by_id: dict[str, dict[str, Any]],
    *,
    max_handles: int,
) -> list[dict[str, Any]]:
    """Collect recent artifact handles from tool result metadata."""
    return _agent_collect_recent_artifact_handles(
        tool_result_metadata_by_id,
        max_handles=max_handles,
    )


def _build_active_artifact_context_content(
    *,
    available_artifacts: set[str],
    available_capabilities: dict[str, set[tuple[str | None, str | None, str | None]]],
    tool_result_metadata_by_id: dict[str, dict[str, Any]],
    max_handles: int,
    max_chars: int,
) -> str | None:
    """Build active artifact context content for injection into prompt."""
    return _agent_build_active_artifact_context_content(
        available_artifacts=available_artifacts,
        available_capabilities=available_capabilities,
        tool_result_metadata_by_id=tool_result_metadata_by_id,
        max_handles=max_handles,
        max_chars=max_chars,
        runtime_artifact_read_tool_name=RUNTIME_ARTIFACT_READ_TOOL_NAME,
        capability_state_snapshot=_capability_state_snapshot,
    )


def _upsert_active_artifact_context_message(
    messages: list[dict[str, Any]],
    *,
    available_artifacts: set[str],
    available_capabilities: dict[str, set[tuple[str | None, str | None, str | None]]],
    tool_result_metadata_by_id: dict[str, dict[str, Any]],
    enabled: bool,
    max_handles: int,
    max_chars: int,
    existing_index: int | None,
) -> tuple[int | None, str | None, bool]:
    """Upsert the active artifact context system message."""
    return _agent_upsert_active_artifact_context_message(
        messages,
        available_artifacts=available_artifacts,
        available_capabilities=available_capabilities,
        tool_result_metadata_by_id=tool_result_metadata_by_id,
        enabled=enabled,
        max_handles=max_handles,
        max_chars=max_chars,
        existing_index=existing_index,
        runtime_artifact_read_tool_name=RUNTIME_ARTIFACT_READ_TOOL_NAME,
        capability_state_snapshot=_capability_state_snapshot,
    )


# ---------------------------------------------------------------------------
# Contract validation
# ---------------------------------------------------------------------------

def _short_requirement(req: CapabilityRequirement) -> str:
    """Short human-readable label for a capability requirement."""
    return _agent_short_requirement(req)


def _contract_declares_no_artifact_prereqs(contract: dict[str, Any] | None) -> bool:
    """Check whether a contract declares no artifact prerequisites."""
    return _agent_contract_declares_no_artifact_prereqs(contract)


def _apply_handle_input_injections(
    *,
    tc: dict[str, Any],
    normalized_tool_contracts: dict[str, dict[str, Any]],
    artifact_registry_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply handle input injections to a tool call."""
    return _agent_apply_handle_input_injections(
        tc=tc,
        normalized_tool_contracts=normalized_tool_contracts,
        artifact_registry_by_id=artifact_registry_by_id,
        extract_tool_call_args=_extract_tool_call_args,
        set_tool_call_args=_set_tool_call_args,
    )


def _tool_declares_no_artifact_prereqs(
    tool_name: str,
    contract: dict[str, Any] | None,
) -> bool:
    """Check whether a tool declares no artifact prerequisites."""
    return _agent_tool_declares_no_artifact_prereqs(tool_name, contract)


def _normalize_tool_contracts(raw: Any) -> dict[str, dict[str, Any]]:
    """Normalize raw tool contracts to canonical form."""
    return _shared_normalize_tool_contracts(raw)


def _effective_contract_requirements(
    tool_name: str,
    contract: dict[str, Any],
    parsed_args: dict[str, Any] | None,
) -> tuple[set[str], set[str]]:
    """Compute effective contract requirements for a tool call."""
    return _agent_effective_contract_requirements(tool_name, contract, parsed_args)


def _validate_tool_contract_call(
    *,
    tool_name: str,
    contract: dict[str, Any],
    parsed_args: dict[str, Any] | None,
    available_artifacts: set[str],
    available_capabilities: dict[str, set[tuple[str | None, str | None, str | None]]] | None = None,
    available_bindings: dict[str, str | None] | None = None,
    artifact_registry_by_id: dict[str, dict[str, Any]] | None = None,
) -> Any:
    """Validate a tool call against its contract."""
    return _agent_validate_tool_contract_call(
        tool_name=tool_name,
        contract=contract,
        parsed_args=parsed_args,
        available_artifacts=available_artifacts,
        available_capabilities=available_capabilities,
        available_bindings=available_bindings,
        artifact_registry_by_id=artifact_registry_by_id,
        event_code_missing_prerequisite=EVENT_CODE_TOOL_VALIDATION_MISSING_PREREQUISITE,
        event_code_missing_capability=EVENT_CODE_TOOL_VALIDATION_MISSING_CAPABILITY,
        event_code_binding_conflict=EVENT_CODE_TOOL_VALIDATION_BINDING_CONFLICT,
    )


def _find_repair_tools_for_missing_requirements(
    *,
    current_tool_name: str,
    missing_requirements: list[dict[str, Any]],
    normalized_tool_contracts: dict[str, dict[str, Any]],
    available_artifacts: set[str],
    available_capabilities: dict[str, set[tuple[str | None, str | None, str | None]]],
    available_bindings: dict[str, str | None] | None,
    max_repair_tools: int,
) -> list[str]:
    """Find tools that could repair missing requirements."""
    return _agent_find_repair_tools_for_missing_requirements(
        current_tool_name=current_tool_name,
        missing_requirements=missing_requirements,
        normalized_tool_contracts=normalized_tool_contracts,
        available_artifacts=available_artifacts,
        available_capabilities=available_capabilities,
        available_bindings=available_bindings,
        max_repair_tools=max_repair_tools,
        event_code_missing_prerequisite=EVENT_CODE_TOOL_VALIDATION_MISSING_PREREQUISITE,
        event_code_missing_capability=EVENT_CODE_TOOL_VALIDATION_MISSING_CAPABILITY,
        event_code_binding_conflict=EVENT_CODE_TOOL_VALIDATION_BINDING_CONFLICT,
    )


def _filter_tools_for_disclosure(
    *,
    openai_tools: list[dict[str, Any]],
    normalized_tool_contracts: dict[str, dict[str, Any]],
    available_artifacts: set[str],
    available_capabilities: dict[str, set[tuple[str | None, str | None, str | None]]],
    available_bindings: dict[str, str | None] | None = None,
    max_unavailable: int = DEFAULT_TOOL_DISCLOSURE_MAX_UNAVAILABLE,
    max_missing_per_tool: int = DEFAULT_TOOL_DISCLOSURE_MAX_MISSING_PER_TOOL,
    max_repair_tools: int = DEFAULT_TOOL_DISCLOSURE_MAX_REPAIR_TOOLS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Filter tools for progressive disclosure based on current capabilities."""
    return _agent_filter_tools_for_disclosure(
        openai_tools=openai_tools,
        normalized_tool_contracts=normalized_tool_contracts,
        available_artifacts=available_artifacts,
        available_capabilities=available_capabilities,
        available_bindings=available_bindings,
        max_unavailable=max_unavailable,
        max_missing_per_tool=max_missing_per_tool,
        max_repair_tools=max_repair_tools,
        tool_declares_no_artifact_prereqs=_tool_declares_no_artifact_prereqs,
        validate_tool_contract_call=_validate_tool_contract_call,
        find_repair_tools_for_missing_requirements=_find_repair_tools_for_missing_requirements,
    )


def _contract_outputs(
    contract: dict[str, Any] | None,
    parsed_args: dict[str, Any] | None = None,
) -> set[str]:
    """Get the output artifact kinds declared by a contract."""
    return _agent_contract_outputs(contract, parsed_args)


def _contract_output_capabilities(
    contract: dict[str, Any] | None,
    parsed_args: dict[str, Any] | None = None,
) -> list[CapabilityRequirement]:
    """Get the output capabilities declared by a contract."""
    return _agent_contract_output_capabilities(contract, parsed_args)


def _is_control_tool_name(
    tool_name: str,
    normalized_tool_contracts: dict[str, dict[str, Any]],
) -> bool:
    """Check whether a tool is classified as a control tool."""
    return _agent_is_control_tool_name(tool_name, normalized_tool_contracts)


def _disclosure_reason_from_entry(entry: dict[str, Any]) -> str:
    """Build a human-readable disclosure reason from a hidden entry."""
    return _agent_disclosure_reason_from_entry(
        entry,
        capability_requirement_from_raw=_capability_requirement_from_raw,
        short_requirement=_short_requirement,
    )


def _disclosure_message(hidden_entries: list[dict[str, Any]]) -> str:
    """Build a disclosure guidance message from hidden entries."""
    return _agent_disclosure_message(
        hidden_entries,
        disclosure_reason_from_entry=_disclosure_reason_from_entry,
        trim_text=_trim_text,
        max_reason_chars=DEFAULT_TOOL_DISCLOSURE_REASON_MAX_CHARS,
    )


def _deficit_labels_from_hidden_entries(hidden_entries: list[dict[str, Any]]) -> list[str]:
    """Extract deficit labels from hidden disclosure entries."""
    return _agent_deficit_labels_from_hidden_entries(
        hidden_entries,
        capability_requirement_from_raw=_capability_requirement_from_raw,
        short_requirement=_short_requirement,
    )


def _analyze_lane_closure(
    *,
    normalized_tool_contracts: dict[str, dict[str, Any]],
    initial_artifacts: set[str],
    initial_capabilities: dict[str, set[tuple[str | None, str | None, str | None]]],
    available_bindings: dict[str, str | None] | None,
) -> dict[str, Any]:
    """Analyze whether the tool lane can reach closure from initial state."""
    return _agent_analyze_lane_closure(
        normalized_tool_contracts=normalized_tool_contracts,
        initial_artifacts=initial_artifacts,
        initial_capabilities=initial_capabilities,
        available_bindings=available_bindings,
        event_code_missing_prerequisite=EVENT_CODE_TOOL_VALIDATION_MISSING_PREREQUISITE,
        event_code_missing_capability=EVENT_CODE_TOOL_VALIDATION_MISSING_CAPABILITY,
        event_code_binding_conflict=EVENT_CODE_TOOL_VALIDATION_BINDING_CONFLICT,
    )
