"""Per-turn tool-processing helpers for the MCP agent loop.

This module owns the dense middle of the extracted turn-execution runtime:
validating tool calls, enforcing contracts, suppressing repeated control-loop
calls, executing runtime/external tools, and reconciling artifact/capability
state after each successful tool result.
"""

from __future__ import annotations

import json as _json
from dataclasses import dataclass
from typing import Any

from llm_client.agent.agent_contracts import CapabilityRequirement
from llm_client.agent.compliance_gate import validate_tool_call_inputs
from llm_client.foundation import (
    extract_bindings_from_tool_args,
    merge_binding_state,
    new_event_id,
    now_iso,
    sha256_text,
)
from llm_client.agent.mcp_contracts import (
    _apply_handle_input_injections,
    _artifact_handle_summaries,
    _artifact_output_state_from_record,
    _capability_state_add,
    _capability_state_snapshot,
    _contract_output_capabilities,
    _contract_outputs,
    _infer_output_capabilities,
    _validate_tool_contract_call,
)
from llm_client.agent.mcp_evidence import _evidence_digest, _tool_evidence_pointer_labels
from llm_client.agent.mcp_tools import _extract_tool_call_args, _runtime_artifact_read_result
from llm_client.agent.agent_planning import PlanState, execute_plan_tool
from llm_client.tools.tool_runtime_common import MCPToolCallRecord, TOOL_REASONING_FIELD

EVENT_CODE_TOOL_VALIDATION_SCHEMA = "TOOL_VALIDATION_REJECTED_SCHEMA"
EVENT_CODE_TOOL_VALIDATION_MISSING_TOOL_REASONING = (
    "TOOL_VALIDATION_REJECTED_MISSING_TOOL_REASONING"
)
EVENT_CODE_TOOL_VALIDATION_BINDING_CONFLICT = (
    "TOOL_VALIDATION_REJECTED_BINDING_CONFLICT"
)
EVENT_CODE_TOOL_VALIDATION_MISSING_PREREQUISITE = (
    "TOOL_VALIDATION_REJECTED_MISSING_PREREQUISITE"
)
EVENT_CODE_CONTROL_LOOP_SUPPRESSED = "CONTROL_CHURN_SUPPRESSED"
EVENT_CODE_TOOL_RUNTIME_ERROR = "TOOL_EXECUTION_RUNTIME_ERROR"


@dataclass
class AgentTurnToolProcessingResult:
    """Outcome of validating and executing one turn's tool calls."""

    tool_calls_to_execute: list[dict[str, Any]]
    records: list[MCPToolCallRecord]
    tool_messages: list[dict[str, Any]]
    executed_records: list[MCPToolCallRecord]
    pending_gate_msg: dict[str, Any] | None
    pending_contract_msg: dict[str, Any] | None
    pending_control_loop_msg: dict[str, Any] | None
    gate_rejected_calls_delta: int
    rejected_missing_reasoning_calls_delta: int
    gate_violation_events: list[dict[str, Any]]
    contract_rejected_calls_delta: int
    contract_violation_events: list[dict[str, Any]]
    control_loop_suppressed_calls_delta: int
    failure_event_codes: list[str]
    handle_input_resolution_count_delta: int
    handle_input_resolved_artifact_count_delta: int
    retrieval_no_hits_count_delta: int
    submit_requires_new_evidence: bool
    submit_evidence_digest_at_last_failure: str | None
    submit_requires_todo_progress: bool
    submit_todo_status_at_last_failure: str | None
    evidence_pointer_labels: set[str]
    contract_rejected_record_count: int
    suppressed_record_count: int


async def _process_tool_calls_turn(
    *,
    turn: int,
    tool_calls_this_turn: list[dict[str, Any]],
    executor: Any,
    tool_result_max_length: int,
    require_tool_reasoning: bool,
    enforce_tool_contracts: bool,
    suppress_control_loop_calls: bool,
    normalized_tool_contracts: dict[str, dict[str, Any]],
    tool_parameter_index: dict[str, dict[str, Any]],
    available_artifacts: set[str],
    available_capabilities: dict[str, set[tuple[str | None, str | None, str | None]]],
    available_bindings: dict[str, Any],
    runtime_artifact_registry_by_id: dict[str, dict[str, Any]],
    runtime_artifact_tool_name: str,
    plan_state: PlanState,
    plan_tool_names: set[str],
    planning_question: str,
    tool_result_metadata_by_id: dict[str, dict[str, Any]],
    artifact_timeline: list[dict[str, Any]],
    retrieval_no_hits_detector: Any,
    submit_requires_new_evidence: bool,
    submit_evidence_digest_at_last_failure: str | None,
    submit_requires_todo_progress: bool,
    submit_todo_status_at_last_failure: str | None,
    last_todo_status_line: str | None,
    evidence_pointer_labels: set[str],
    foundation_run_id: str,
    foundation_session_id: str,
    foundation_actor_id: str,
    emit_foundation_event: Any,
) -> AgentTurnToolProcessingResult:
    """Validate, execute, and reconcile one turn's tool calls."""

    tool_calls_to_execute = list(tool_calls_this_turn)
    gate_rejected_records: list[MCPToolCallRecord] = []
    gate_rejected_messages: list[dict[str, Any]] = []
    pending_gate_msg: dict[str, Any] | None = None
    gate_violation_events: list[dict[str, Any]] = []
    gate_rejected_calls_delta = 0
    rejected_missing_reasoning_calls_delta = 0
    failure_event_codes: list[str] = []

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

        gate_rejected_calls_delta += 1
        if gate_validation.error_code == EVENT_CODE_TOOL_VALIDATION_MISSING_TOOL_REASONING:
            rejected_missing_reasoning_calls_delta += 1
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
        emit_foundation_event(
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
    contract_violation_events: list[dict[str, Any]] = []
    contract_rejected_calls_delta = 0
    if enforce_tool_contracts and normalized_tool_contracts:
        filtered_contract_calls: list[dict[str, Any]] = []
        for tc in tool_calls_to_execute:
            tool_name = str(tc.get("function", {}).get("name", "")).strip()
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

            contract_rejected_calls_delta += 1
            err = f"Tool contract violation: {contract_validation.reason}"
            contract_error_code = (
                contract_validation.error_code
                or EVENT_CODE_TOOL_VALIDATION_MISSING_PREREQUISITE
            )
            contract_violation_events.append(
                {
                    "turn": turn + 1,
                    "tool": tool_name or "<unknown>",
                    "contract_mode": contract_validation.contract_mode,
                    "reason": contract_validation.reason,
                    "error_code": contract_error_code,
                    "failure_phase": contract_validation.failure_phase or "input_validation",
                    "available_artifacts": sorted(available_artifacts),
                    "available_capabilities": _capability_state_snapshot(available_capabilities),
                    "available_bindings": dict(available_bindings),
                    "call_bindings": dict(contract_validation.call_bindings),
                    "arg_keys": sorted(parsed_args.keys()) if isinstance(parsed_args, dict) else [],
                    "missing_requirements": list(contract_validation.missing_requirements or []),
                }
            )
            failure_event_codes.append(contract_error_code)
            contract_rejected_records.append(
                MCPToolCallRecord(
                    server="__contract__",
                    tool=tool_name or "<unknown>",
                    arguments=parsed_args if isinstance(parsed_args, dict) else {},
                    error=err,
                )
            )
            contract_rejected_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": _json.dumps(
                        {
                            "error": err,
                            "error_code": contract_error_code,
                            "failure_phase": contract_validation.failure_phase
                            or "input_validation",
                            "contract_mode": contract_validation.contract_mode,
                            "call_bindings": contract_validation.call_bindings,
                            "missing_requirements": contract_validation.missing_requirements,
                        }
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
                    "operation": {"name": tool_name or "<unknown>", "version": None},
                    "inputs": {
                        "artifact_ids": sorted(available_artifacts),
                        "params": parsed_args if isinstance(parsed_args, dict) else {},
                        "bindings": dict(available_bindings),
                    },
                    "outputs": {"artifact_ids": [], "payload_hashes": []},
                    "failure": {
                        "error_code": contract_error_code,
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

    current_evidence_digest = _evidence_digest(evidence_pointer_labels)
    suppressed_records: list[MCPToolCallRecord] = []
    suppressed_tool_messages: list[dict[str, Any]] = []
    filtered_tool_calls: list[dict[str, Any]] = []
    if suppress_control_loop_calls:
        for tc in tool_calls_to_execute:
            tool_name = str(tc.get("function", {}).get("name", "")).strip()
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
                    suppressed_tool_messages.append(
                        {
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
                        }
                    )
                    continue

            if tool_name == "submit_answer" and submit_requires_todo_progress:
                if (
                    submit_todo_status_at_last_failure is not None
                    and last_todo_status_line is not None
                    and last_todo_status_line != submit_todo_status_at_last_failure
                ):
                    submit_requires_todo_progress = False
                    submit_todo_status_at_last_failure = None
                else:
                    err = (
                        "submit_answer suppressed: validator requires TODO progress before retry. "
                        "Pending atom state has not changed since the last rejected submit. "
                        "Resolve or complete the pending atom first, then retry submit."
                    )
                    suppressed_records.append(
                        MCPToolCallRecord(
                            server="__agent__",
                            tool=tool_name,
                            arguments=parsed_args,
                            error=err,
                        )
                    )
                    suppressed_tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": _json.dumps(
                                {
                                    "error": err,
                                    "error_code": EVENT_CODE_CONTROL_LOOP_SUPPRESSED,
                                    "todo_status_line": last_todo_status_line,
                                    "required_todo_progress_from": submit_todo_status_at_last_failure,
                                }
                            ),
                        }
                    )
                    continue

            filtered_tool_calls.append(tc)
    else:
        filtered_tool_calls = list(tool_calls_to_execute)

    tool_calls_to_execute = filtered_tool_calls
    pending_control_loop_msg: dict[str, Any] | None = None
    control_loop_suppressed_calls_delta = 0
    if suppressed_records:
        control_loop_suppressed_calls_delta = len(suppressed_records)
        pending_control_loop_msg = {
            "role": "user",
            "content": (
                "[SYSTEM: Repeated control-call loop detected. Some tool calls were suppressed. "
                "Update TODO state or change hypotheses before retrying submit/completion.]"
            ),
        }
        for rec in suppressed_records:
            emit_foundation_event(
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
        tool_name = str(tc.get("function", {}).get("name", "")).strip() or "<unknown>"
        parsed_args = _extract_tool_call_args(tc)
        emit_foundation_event(
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

    external_tool_calls: list[dict[str, Any]] = []
    runtime_tool_call_indexes: dict[int, tuple[MCPToolCallRecord, dict[str, Any]]] = {}
    handle_input_resolution_count_delta = 0
    handle_input_resolved_artifact_count_delta = 0
    for idx, tc in enumerate(tool_calls_to_execute):
        tool_name = str(tc.get("function", {}).get("name", "")).strip()
        if tool_name == runtime_artifact_tool_name:
            runtime_tool_call_indexes[idx] = _runtime_artifact_read_result(
                artifact_registry_by_id=runtime_artifact_registry_by_id,
                tc=tc,
                max_result_length=tool_result_max_length,
                require_tool_reasoning=require_tool_reasoning,
            )
        elif tool_name in plan_tool_names:
            parsed_args = _extract_tool_call_args(tc) or {}
            if isinstance(parsed_args, dict):
                parsed_args = dict(parsed_args)
                parsed_args.pop(TOOL_REASONING_FIELD, None)
            result_text = execute_plan_tool(
                tool_name=tool_name,
                arguments=parsed_args,
                plan_state=plan_state,
                question=planning_question,
                turn=turn,
            )
            tc_id = str(tc.get("id", ""))
            if result_text.startswith("Error") or result_text.startswith("Unknown"):
                runtime_tool_call_indexes[idx] = (
                    MCPToolCallRecord(
                        server="__agent__",
                        tool=tool_name,
                        arguments=parsed_args,
                        tool_call_id=tc_id,
                        error=result_text,
                    ),
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": _json.dumps({"error": result_text}),
                    },
                )
            else:
                runtime_tool_call_indexes[idx] = (
                    MCPToolCallRecord(
                        server="__agent__",
                        tool=tool_name,
                        arguments=parsed_args,
                        tool_call_id=tc_id,
                        result=result_text,
                    ),
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": result_text,
                    },
                )
        else:
            patched_tc, handle_injections = _apply_handle_input_injections(
                tc=tc,
                normalized_tool_contracts=normalized_tool_contracts,
                artifact_registry_by_id=runtime_artifact_registry_by_id,
            )
            if handle_injections:
                handle_input_resolution_count_delta += len(handle_injections)
                handle_input_resolved_artifact_count_delta += sum(
                    int(item.get("resolved_count") or 0)
                    for item in handle_injections
                    if isinstance(item, dict)
                )
            external_tool_calls.append(patched_tc)

    if external_tool_calls:
        external_records, external_tool_messages = await executor(
            external_tool_calls,
            tool_result_max_length,
        )
    else:
        external_records, external_tool_messages = [], []

    executed_records: list[MCPToolCallRecord] = []
    executed_tool_messages: list[dict[str, Any]] = []
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

    retrieval_no_hits_count_delta = 0
    for record in executed_records:
        if record.error:
            emit_foundation_event(
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
            typed_output_envelopes,
            typed_output_artifacts,
            typed_output_capabilities,
            typed_output_artifact_ids,
            typed_output_bindings,
        ) = _artifact_output_state_from_record(
            record,
            fallback_bindings=available_bindings,
        )
        for envelope in typed_output_envelopes:
            artifact_id = str(envelope.get("artifact_id", "")).strip()
            if artifact_id:
                runtime_artifact_registry_by_id[artifact_id] = dict(envelope)
        record.artifact_ids = sorted(typed_output_artifact_ids)
        record.artifact_handles = _artifact_handle_summaries(typed_output_envelopes)
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
            available_bindings.clear()
            available_bindings.update(merged_bindings)
            emit_foundation_event(
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
            produced_capabilities = _contract_output_capabilities(
                contract,
                parsed_record_args,
            )

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
            artifact_timeline.append(
                {
                    "turn": turn + 1,
                    "phase": "tool_success",
                    "tool": record.tool,
                    "produced": sorted(produced),
                    "produced_capabilities": produced_capability_payload,
                    "available_artifacts": sorted(available_artifacts),
                    "available_capabilities": _capability_state_snapshot(available_capabilities),
                }
            )

        if retrieval_no_hits_detector(tool_name=record.tool, result_text=record.result):
            retrieval_no_hits_count_delta += 1
            failure_event_codes.append("RETRIEVAL_NO_HITS")

        emit_foundation_event(
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
                        if record.result is not None
                        else []
                    ),
                },
                "artifacts": list(typed_output_envelopes),
            }
        )

    records = (
        gate_rejected_records
        + contract_rejected_records
        + suppressed_records
        + executed_records
    )
    tool_messages = (
        gate_rejected_messages
        + contract_rejected_messages
        + suppressed_tool_messages
        + executed_tool_messages
    )
    return AgentTurnToolProcessingResult(
        tool_calls_to_execute=tool_calls_to_execute,
        records=records,
        tool_messages=tool_messages,
        executed_records=executed_records,
        pending_gate_msg=pending_gate_msg,
        pending_contract_msg=pending_contract_msg,
        pending_control_loop_msg=pending_control_loop_msg,
        gate_rejected_calls_delta=gate_rejected_calls_delta,
        rejected_missing_reasoning_calls_delta=rejected_missing_reasoning_calls_delta,
        gate_violation_events=gate_violation_events,
        contract_rejected_calls_delta=contract_rejected_calls_delta,
        contract_violation_events=contract_violation_events,
        control_loop_suppressed_calls_delta=control_loop_suppressed_calls_delta,
        failure_event_codes=failure_event_codes,
        handle_input_resolution_count_delta=handle_input_resolution_count_delta,
        handle_input_resolved_artifact_count_delta=handle_input_resolved_artifact_count_delta,
        retrieval_no_hits_count_delta=retrieval_no_hits_count_delta,
        submit_requires_new_evidence=submit_requires_new_evidence,
        submit_evidence_digest_at_last_failure=submit_evidence_digest_at_last_failure,
        submit_requires_todo_progress=submit_requires_todo_progress,
        submit_todo_status_at_last_failure=submit_todo_status_at_last_failure,
        evidence_pointer_labels=set(evidence_pointer_labels),
        contract_rejected_record_count=len(contract_rejected_records),
        suppressed_record_count=len(suppressed_records),
    )
