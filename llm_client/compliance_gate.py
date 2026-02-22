"""Deterministic pre-execution compliance checks for tool calls.

This module is intentionally side-effect free and model-free. It validates
tool-call arguments against declared schema/observability policy and binding
authority rules before execution is attempted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm_client.foundation import check_binding_conflicts, extract_bindings_from_tool_args


@dataclass
class ComplianceGateResult:
    """Normalized pre-execution compliance outcome for one tool call."""

    is_valid: bool
    reason: str = ""
    error_code: str | None = None
    failure_phase: str | None = None
    missing_requirements: list[dict[str, Any]] = field(default_factory=list)
    call_bindings: dict[str, str | None] = field(default_factory=dict)


def build_tool_parameter_index(openai_tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build tool name -> JSON-schema parameters map from OpenAI tool defs."""
    out: dict[str, dict[str, Any]] = {}
    for item in openai_tools:
        if not isinstance(item, dict):
            continue
        fn = item.get("function")
        if not isinstance(fn, dict):
            continue
        name = str(fn.get("name", "")).strip()
        params = fn.get("parameters")
        if not name or not isinstance(params, dict):
            continue
        out[name] = params
    return out


def validate_tool_call_inputs(
    *,
    tool_name: str,
    parsed_args: dict[str, Any] | None,
    tool_parameters: dict[str, Any] | None,
    require_tool_reasoning: bool,
    tool_reasoning_field: str,
    available_bindings: dict[str, Any] | None,
    error_code_schema: str,
    error_code_missing_reasoning: str,
    error_code_binding_conflict: str,
) -> ComplianceGateResult:
    """Validate arguments for schema strictness, reasoning policy, and bindings."""
    if not isinstance(parsed_args, dict):
        return ComplianceGateResult(
            is_valid=False,
            reason=f"{tool_name} arguments must be a JSON object",
            error_code=error_code_schema,
            failure_phase="input_validation",
            missing_requirements=[],
            call_bindings={},
        )

    call_bindings = extract_bindings_from_tool_args(parsed_args)
    if call_bindings:
        bindings_ok, bindings_reason, _conflicts, normalized_call_bindings = check_binding_conflicts(
            available_bindings=available_bindings,
            proposed_bindings=call_bindings,
        )
        if not bindings_ok:
            return ComplianceGateResult(
                is_valid=False,
                reason=bindings_reason,
                error_code=error_code_binding_conflict,
                failure_phase="binding_validation",
                missing_requirements=[],
                call_bindings=normalized_call_bindings,
            )
        call_bindings = normalized_call_bindings

    if require_tool_reasoning:
        raw_reasoning = parsed_args.get(tool_reasoning_field)
        if not isinstance(raw_reasoning, str) or not raw_reasoning.strip():
            return ComplianceGateResult(
                is_valid=False,
                reason=f"Missing required argument: {tool_reasoning_field}",
                error_code=error_code_missing_reasoning,
                failure_phase="input_validation",
                missing_requirements=[{"arg": tool_reasoning_field}],
                call_bindings=call_bindings,
            )

    if isinstance(tool_parameters, dict):
        properties_raw = tool_parameters.get("properties")
        if isinstance(properties_raw, dict):
            # Only enforce strict unknown-arg rejection when the schema declares
            # at least one non-observability argument. Empty/minimal schemas are
            # treated as permissive until tool schemas are fully normalized.
            declared_args = {
                key for key in properties_raw.keys()
                if key != tool_reasoning_field
            }
            if declared_args:
                unknown_args = sorted(
                    key for key in parsed_args.keys()
                    if key not in properties_raw
                )
                if unknown_args:
                    return ComplianceGateResult(
                        is_valid=False,
                        reason=f"{tool_name} unsupported args: {', '.join(unknown_args)}",
                        error_code=error_code_schema,
                        failure_phase="input_validation",
                        missing_requirements=[{"arg": arg} for arg in unknown_args],
                        call_bindings=call_bindings,
                    )

        required_raw = tool_parameters.get("required")
        if isinstance(required_raw, list):
            required_args = [str(v) for v in required_raw if isinstance(v, str) and v.strip()]
            if not require_tool_reasoning:
                required_args = [v for v in required_args if v != tool_reasoning_field]
            missing_required = [arg for arg in required_args if arg not in parsed_args]
            if missing_required:
                return ComplianceGateResult(
                    is_valid=False,
                    reason=f"{tool_name} missing required args: {', '.join(missing_required)}",
                    error_code=error_code_schema,
                    failure_phase="input_validation",
                    missing_requirements=[{"arg": arg} for arg in missing_required],
                    call_bindings=call_bindings,
                )

    return ComplianceGateResult(
        is_valid=True,
        call_bindings=call_bindings,
    )
