"""Normalize additive metadata attached to call results.

This module owns the pure bookkeeping that keeps ``LLMCallResult`` identity
stable across multiple runtime entrypoints:

1. build routing traces from decisions that were already made elsewhere,
2. normalize free-form warning strings into machine-readable records,
3. merge additive warning records without duplicate inflation,
4. stamp requested/resolved/execution identity on result objects.

The module does not decide routing policy and does not encode model-governance
rules. Those policy choices stay in the runtime boundary that calls into this
module.
"""

from __future__ import annotations

from typing import Any, Protocol, TypeVar


class ResultIdentityTarget(Protocol):
    """Structural contract for results that carry additive identity metadata."""

    model: str
    requested_model: str | None
    resolved_model: str | None
    execution_model: str | None
    routing_trace: dict[str, Any] | None
    warnings: list[str]
    warning_records: list[dict[str, Any]]


ResultT = TypeVar("ResultT", bound=ResultIdentityTarget)


def build_routing_trace(
    *,
    requested_model: str,
    routing_policy: str,
    attempted_models: list[str] | None = None,
    selected_model: str | None = None,
    requested_api_base: str | None = None,
    effective_api_base: str | None = None,
    sticky_fallback: bool | None = None,
    background_mode: bool | None = None,
) -> dict[str, Any]:
    """Build a minimal routing trace from an already-resolved routing decision."""
    trace: dict[str, Any] = {"routing_policy": routing_policy}
    attempts = [model for model in (attempted_models or []) if isinstance(model, str) and model.strip()]
    if attempts:
        trace["attempted_models"] = attempts
        if requested_model != attempts[0]:
            trace["normalized_from"] = requested_model
            trace["normalized_to"] = attempts[0]
    if selected_model:
        trace["selected_model"] = selected_model
    if sticky_fallback is not None:
        trace["sticky_fallback"] = bool(sticky_fallback)
    if background_mode is not None:
        trace["background_mode"] = bool(background_mode)
    if requested_api_base is None and effective_api_base is not None:
        trace["api_base_injected"] = True
    elif requested_api_base is not None:
        trace["api_base_injected"] = False
    return trace


def warning_record(
    *,
    code: str,
    category: str,
    message: str,
    field_path: str | None = None,
    remediation: str | None = None,
) -> dict[str, Any]:
    """Build one machine-readable warning record."""
    record: dict[str, Any] = {
        "code": code,
        "category": category,
        "message": message,
    }
    if field_path:
        record["field_path"] = field_path
    if remediation:
        record["remediation"] = remediation
    return record


def warning_record_from_message(message: str) -> dict[str, Any] | None:
    """Translate one stable warning string into a machine-readable record."""
    text = str(message or "")
    if text.startswith("RETRY "):
        return warning_record(
            code="LLMC_WARN_RETRY",
            category="RuntimeWarning",
            message=text,
            remediation="Inspect transient provider/network conditions.",
        )
    if text.startswith("FALLBACK:"):
        return warning_record(
            code="LLMC_WARN_FALLBACK",
            category="UserWarning",
            message=text,
            remediation="Check primary model/provider health and fallback policy.",
        )
    if text.startswith("STICKY_FALLBACK:"):
        return warning_record(
            code="LLMC_WARN_STICKY_FALLBACK",
            category="UserWarning",
            message=text,
            remediation="Investigate persistent failures on the requested primary model.",
        )
    if text.startswith("AUTO_TAG:"):
        return warning_record(
            code="LLMC_WARN_AUTO_TAG",
            category="UserWarning",
            message=text,
            remediation="Pass explicit task/trace_id/max_budget for deterministic observability.",
        )
    if text.startswith("AGENT_RETRY_DISABLED:"):
        return warning_record(
            code="LLMC_WARN_AGENT_RETRY_DISABLED",
            category="UserWarning",
            message=text,
            remediation="Enable agent_retry_safe only for read-only/idempotent agent runs.",
        )
    if text.startswith("TOOL_DISCLOSURE:"):
        return warning_record(
            code="LLMC_WARN_TOOL_DISCLOSURE",
            category="UserWarning",
            message=text,
        )
    return None


def merge_warning_records(
    *,
    existing: list[dict[str, Any]] | None,
    warnings: list[str] | None,
    extra_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Merge warning records without duplicating equivalent code/message pairs."""
    merged: list[dict[str, Any]] = [dict(record) for record in (existing or []) if isinstance(record, dict)]
    seen: set[tuple[str, str]] = {
        (str(record.get("code", "")), str(record.get("message", "")))
        for record in merged
    }

    for message in warnings or []:
        derived_record = warning_record_from_message(message)
        if derived_record is None:
            continue
        key = (str(derived_record.get("code", "")), str(derived_record.get("message", "")))
        if key not in seen:
            merged.append(derived_record)
            seen.add(key)

    for record in extra_records or []:
        if not isinstance(record, dict):
            continue
        key = (str(record.get("code", "")), str(record.get("message", "")))
        if key not in seen:
            merged.append(dict(record))
            seen.add(key)
    return merged


def annotate_result_identity(
    result: ResultT,
    *,
    requested_model: str,
    resolved_model: str | None = None,
    routing_trace: dict[str, Any] | None = None,
    warning_records: list[dict[str, Any]] | None = None,
) -> ResultT:
    """Attach stable identity metadata to a result object in place."""
    if result.requested_model is None:
        result.requested_model = requested_model
    if resolved_model is not None and result.resolved_model is None:
        result.resolved_model = resolved_model
    if result.resolved_model is None and isinstance(result.model, str) and result.model.strip():
        result.resolved_model = result.model
    if result.execution_model is None and result.resolved_model is not None:
        result.execution_model = result.resolved_model

    if routing_trace:
        existing = result.routing_trace if isinstance(result.routing_trace, dict) else {}
        merged_routing_trace = dict(existing)
        merged_routing_trace.update(routing_trace)
        result.routing_trace = merged_routing_trace

    resolved_identity = result.resolved_model or resolved_model
    if resolved_identity:
        result.model = resolved_identity

    result.warning_records = merge_warning_records(
        existing=result.warning_records,
        warnings=result.warnings,
        extra_records=warning_records,
    )
    return result
