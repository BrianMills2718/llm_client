"""Artifact/evidence/context helpers for agent runtimes."""

from __future__ import annotations

import json as _json
from typing import Any, Callable, Protocol

from llm_client.foundation import (
    evidence_pointer_label,
    extract_artifact_envelopes,
    sha256_json,
)


class ToolCallRecordLike(Protocol):
    """Minimal tool-call record surface needed for evidence extraction."""

    error: str | None
    tool: str
    result: str | None


def _runtime_artifact_read_tool_def(
    *,
    tool_name: str,
    tool_reasoning_field: str,
) -> dict[str, Any]:
    """Declarative runtime tool for reopening prior typed artifacts by handle."""
    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": (
                "Read previously produced typed artifacts by artifact_id from the runtime artifact registry. "
                "Use this when older tool payloads were cleared from active context but artifact handles remain visible."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "artifact_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Stable artifact_id handles to reopen from runtime state.",
                    },
                    "include_payload": {
                        "type": "boolean",
                        "description": "Whether to include the artifact payload field. Default true.",
                    },
                    "include_provenance": {
                        "type": "boolean",
                        "description": "Whether to include provenance/evidence_refs. Default true.",
                    },
                    tool_reasoning_field: {
                        "type": "string",
                        "description": "Short explanation of why this runtime artifact read is needed.",
                    },
                },
                "required": ["artifact_ids", tool_reasoning_field],
            },
        },
    }


def _runtime_artifact_read_contract() -> dict[str, Any]:
    """Declarative contract for the runtime artifact read helper."""
    return {
        "is_control": True,
        "artifact_prereqs": "none",
    }


def _runtime_artifact_read_result(
    *,
    artifact_registry_by_id: dict[str, dict[str, Any]],
    tc: dict[str, Any],
    max_result_length: int,
    require_tool_reasoning: bool,
    tool_name: str,
    tool_reasoning_field: str,
    record_factory: Callable[..., Any],
    extract_tool_call_args: Callable[[dict[str, Any]], dict[str, Any] | None],
    truncate_text: Callable[[str, int], str],
) -> tuple[Any, dict[str, Any]]:
    """Execute one runtime artifact read without delegating to an external executor."""
    fn = tc.get("function", {})
    resolved_tool_name = str(fn.get("name", "")).strip() or tool_name
    tc_id = tc.get("id", "")
    arguments = extract_tool_call_args(tc)

    if arguments is None:
        record = record_factory(
            server="__runtime__",
            tool=resolved_tool_name,
            arguments={},
            tool_call_id=tc_id,
            error="JSON parse error: arguments must decode to a JSON object",
        )
        return record, {
            "role": "tool",
            "tool_call_id": tc_id,
            "content": "ERROR: Invalid JSON arguments: arguments must decode to a JSON object",
        }

    runtime_args = dict(arguments)
    tool_reasoning_raw = runtime_args.pop(tool_reasoning_field, None)
    tool_reasoning = None
    if isinstance(tool_reasoning_raw, str):
        stripped = tool_reasoning_raw.strip()
        if stripped:
            tool_reasoning = stripped

    record = record_factory(
        server="__runtime__",
        tool=resolved_tool_name,
        arguments=runtime_args,
        tool_call_id=tc_id,
        tool_reasoning=tool_reasoning,
    )

    if require_tool_reasoning and not tool_reasoning:
        record.error = f"Missing required argument: {tool_reasoning_field}"
        return record, {
            "role": "tool",
            "tool_call_id": tc_id,
            "content": _json.dumps({"error": record.error}),
        }

    requested_ids_raw = runtime_args.get("artifact_ids")
    requested_ids: list[str] = []
    if isinstance(requested_ids_raw, str):
        requested_ids = [requested_ids_raw.strip()] if requested_ids_raw.strip() else []
    elif isinstance(requested_ids_raw, list):
        requested_ids = [
            str(item).strip()
            for item in requested_ids_raw
            if str(item).strip()
        ]
    include_payload = runtime_args.get("include_payload")
    include_payload = True if include_payload is None else bool(include_payload)
    include_provenance = runtime_args.get("include_provenance")
    include_provenance = True if include_provenance is None else bool(include_provenance)

    found_artifacts: list[dict[str, Any]] = []
    missing_artifact_ids: list[str] = []
    for artifact_id in requested_ids:
        stored = artifact_registry_by_id.get(artifact_id)
        if not isinstance(stored, dict):
            missing_artifact_ids.append(artifact_id)
            continue
        artifact_payload = _json.loads(_json.dumps(stored))
        if not include_payload:
            artifact_payload.pop("payload", None)
        if not include_provenance:
            artifact_payload.pop("provenance", None)
        found_artifacts.append(artifact_payload)

    if not found_artifacts:
        record.error = (
            "Unknown artifact_id(s): "
            + ", ".join(missing_artifact_ids or requested_ids or ["<none>"])
        )
        return record, {
            "role": "tool",
            "tool_call_id": tc_id,
            "content": _json.dumps(
                {
                    "error": record.error,
                    "error_code": "RUNTIME_ARTIFACT_NOT_FOUND",
                    "requested_artifact_ids": requested_ids,
                    "missing_artifact_ids": missing_artifact_ids or requested_ids,
                }
            ),
        }

    payload = {
        "artifacts": found_artifacts,
        "artifact_ids": [str(item.get("artifact_id", "")).strip() for item in found_artifacts],
        "missing_artifact_ids": missing_artifact_ids,
        "count": len(found_artifacts),
    }
    content = truncate_text(_json.dumps(payload), max_result_length)
    record.result = content
    return record, {
        "role": "tool",
        "tool_call_id": tc_id,
        "content": content,
    }


def _normalize_evidence_pointer_label(raw: Any) -> str | None:
    """Canonicalize one evidence reference into a stable pointer label."""
    return evidence_pointer_label(raw)


def _collect_evidence_pointer_labels(payload: Any, out: set[str]) -> None:
    """Recursively extract stable evidence pointers from JSON tool payloads."""
    if isinstance(payload, dict):
        for envelope in extract_artifact_envelopes(payload):
            provenance = envelope.get("provenance")
            if isinstance(provenance, dict):
                raw_refs = provenance.get("evidence_refs")
                if isinstance(raw_refs, list):
                    for item in raw_refs:
                        normalized = _normalize_evidence_pointer_label(item)
                        if normalized:
                            out.add(normalized)

        explicit_ref_added = False

        raw_ref = payload.get("evidence_ref")
        normalized = _normalize_evidence_pointer_label(raw_ref)
        if normalized:
            out.add(normalized)
            explicit_ref_added = True

        raw_refs = payload.get("evidence_refs")
        if isinstance(raw_refs, list):
            for item in raw_refs:
                normalized = _normalize_evidence_pointer_label(item)
                if normalized:
                    out.add(normalized)
                    explicit_ref_added = True

        chunk_id = payload.get("chunk_id")
        if (
            not explicit_ref_added
            and isinstance(chunk_id, str)
            and chunk_id.strip().startswith("chunk_")
        ):
            out.add(f"chunk:{chunk_id.strip()}")

        for value in payload.values():
            if isinstance(value, (dict, list)):
                _collect_evidence_pointer_labels(value, out)
        return

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, (dict, list)):
                _collect_evidence_pointer_labels(item, out)


def _parse_record_result_json_value(record: ToolCallRecordLike) -> Any | None:
    """Best-effort parse of JSON tool result payloads preserving the top-level type."""
    if not record.result or not isinstance(record.result, str):
        return None
    try:
        parsed = _json.loads(record.result)
    except Exception:
        return None
    return parsed


def _extract_evidence_pointers_from_text(text: str) -> set[str]:
    """Fallback evidence pointer extraction from plain-text tool results.

    When tool results are linearized into natural language (not JSON), the
    standard JSON-based extraction produces zero pointers, causing the
    stagnation detector to fire incorrectly. This fallback finds chunk IDs
    and entity-like patterns in the text so stagnation tracking works.
    """
    import re
    labels: set[str] = set()
    # Chunk IDs: [chunk_123], chunk_123, chunk-123
    for m in re.finditer(r'\b(chunk[_-]\d+)\b', text, re.IGNORECASE):
        labels.add(f"chunk:{m.group(1)}")
    # Quoted entity names from linearized entity_search output: 'entity_name'
    for m in re.finditer(r"'([^']{2,60})'", text):
        name = m.group(1).strip()
        if name and not name.startswith(("http", "results/", "/")):
            labels.add(f"entity:{name}")
    return labels


def _tool_evidence_pointer_labels(
    record: ToolCallRecordLike,
    *,
    budget_exempt_tool_names: frozenset[str],
) -> set[str]:
    """Extract canonical evidence pointers from one successful evidence tool call."""
    if record.error or record.tool in budget_exempt_tool_names:
        return set()
    parsed = _parse_record_result_json_value(record)
    if parsed is not None and isinstance(parsed, (dict, list)):
        labels: set[str] = set()
        _collect_evidence_pointer_labels(parsed, labels)
        redundant_labels = {
            label.split("#", 1)[0]
            for label in labels
            if label.startswith("chunk:") and "#" in label
        }
        labels.difference_update(redundant_labels)
        if labels:
            return labels
    # Fallback: extract from plain text (linearized tool results)
    if record.result and isinstance(record.result, str) and len(record.result) > 10:
        return _extract_evidence_pointers_from_text(record.result)
    return set()


def _evidence_digest(evidence_labels: set[str]) -> str:
    """Deterministic digest of accumulated canonical evidence pointers."""
    return sha256_json(sorted(evidence_labels)).replace("sha256:", "")


def _artifact_handle_summaries(envelopes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    handles: list[dict[str, Any]] = []
    for envelope in envelopes:
        if not isinstance(envelope, dict):
            continue
        artifact_id = str(envelope.get("artifact_id", "")).strip()
        if not artifact_id:
            continue
        handle: dict[str, Any] = {
            "artifact_id": artifact_id,
            "artifact_type": str(envelope.get("artifact_type", "")).strip().upper(),
        }
        capabilities = envelope.get("capabilities")
        if isinstance(capabilities, list) and capabilities:
            first_cap = capabilities[0]
            if isinstance(first_cap, dict):
                kind = str(first_cap.get("kind", "")).strip().upper()
                ref_type = str(first_cap.get("ref_type", "")).strip()
                namespace = str(first_cap.get("namespace", "")).strip()
                if kind:
                    handle["kind"] = kind
                if ref_type:
                    handle["ref_type"] = ref_type
                if namespace:
                    handle["namespace"] = namespace
        handles.append(handle)
    return handles


def _collect_recent_artifact_handles(
    tool_result_metadata_by_id: dict[str, dict[str, Any]],
    *,
    max_handles: int,
) -> list[dict[str, Any]]:
    handles: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    if max_handles <= 0:
        return handles
    for metadata in reversed(list(tool_result_metadata_by_id.values())):
        if not isinstance(metadata, dict):
            continue
        raw_handles = metadata.get("artifact_handles")
        if not isinstance(raw_handles, list):
            continue
        for handle in raw_handles:
            if not isinstance(handle, dict):
                continue
            artifact_id = str(handle.get("artifact_id", "")).strip()
            if not artifact_id or artifact_id in seen_ids:
                continue
            seen_ids.add(artifact_id)
            handles.append(dict(handle))
            if len(handles) >= max_handles:
                return handles
    return handles


def _trim_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def _build_active_artifact_context_content(
    *,
    available_artifacts: set[str],
    available_capabilities: list[dict[str, str]] | dict[str, set[tuple[str | None, str | None, str | None]]],
    tool_result_metadata_by_id: dict[str, dict[str, Any]],
    max_handles: int,
    max_chars: int,
    runtime_artifact_read_tool_name: str,
    capability_state_snapshot: Callable[
        [dict[str, set[tuple[str | None, str | None, str | None]]]],
        list[dict[str, str]],
    ] | None = None,
) -> str | None:
    recent_handles = _collect_recent_artifact_handles(
        tool_result_metadata_by_id,
        max_handles=max_handles,
    )
    if isinstance(available_capabilities, dict):
        capability_labels = (
            capability_state_snapshot(available_capabilities)
            if capability_state_snapshot is not None
            else []
        )
    else:
        capability_labels = list(available_capabilities)
    artifact_labels = sorted(str(kind) for kind in available_artifacts if str(kind).strip())
    filtered_capability_labels = [
        cap
        for cap in capability_labels
        if not (
            isinstance(cap, dict)
            and cap.get("kind") == "QUERY_TEXT"
            and not cap.get("ref_type")
            and not cap.get("namespace")
            and not cap.get("bindings_hash")
        )
    ]

    if not recent_handles and not filtered_capability_labels and artifact_labels in ([], None, ["QUERY_TEXT"]):
        return None

    handle_parts: list[str] = []
    for handle in recent_handles:
        artifact_id = str(handle.get("artifact_id", "")).strip()
        artifact_type = str(handle.get("artifact_type", "")).strip().upper()
        kind = str(handle.get("kind", "")).strip().upper()
        ref_type = str(handle.get("ref_type", "")).strip()
        namespace = str(handle.get("namespace", "")).strip()
        label_parts = [artifact_id]
        if artifact_type:
            label_parts.append(artifact_type)
        if kind and kind != artifact_type:
            label_parts.append(kind)
        if ref_type:
            label_parts.append(f"ref_type={ref_type}")
        if namespace:
            label_parts.append(f"namespace={namespace}")
        handle_parts.append(" ".join(label_parts))

    content = (
        "[SYSTEM: Active artifact context. "
        + (
            "Recent typed artifacts: " + "; ".join(handle_parts) + ". "
            if handle_parts else
            ""
        )
        + (
            "Available artifact kinds: " + ", ".join(artifact_labels) + ". "
            if artifact_labels else
            ""
        )
        + (
            "Available capabilities: "
            + "; ".join(
                cap.get("kind", "")
                + (
                    "[" + ", ".join(
                        f"{k}={cap[k]}"
                        for k in ("ref_type", "namespace", "bindings_hash")
                        if cap.get(k)
                    ) + "]"
                    if any(cap.get(k) for k in ("ref_type", "namespace", "bindings_hash"))
                    else ""
                )
                for cap in filtered_capability_labels[:max(1, max_handles)]
                if isinstance(cap, dict)
            )
            + ". "
            if filtered_capability_labels else
            ""
        )
        + (
            "Older tool payloads may be cleared from context; "
            f"use {runtime_artifact_read_tool_name} with artifact_ids to reopen typed payloads when needed.]"
        )
    )
    return _trim_text(content, max_chars)


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
    runtime_artifact_read_tool_name: str,
    capability_state_snapshot: Callable[
        [dict[str, set[tuple[str | None, str | None, str | None]]]],
        list[dict[str, str]],
    ],
) -> tuple[int | None, str | None, bool]:
    if not enabled:
        return existing_index, None, False
    content = _build_active_artifact_context_content(
        available_artifacts=available_artifacts,
        available_capabilities=available_capabilities,
        tool_result_metadata_by_id=tool_result_metadata_by_id,
        max_handles=max_handles,
        max_chars=max_chars,
        runtime_artifact_read_tool_name=runtime_artifact_read_tool_name,
        capability_state_snapshot=capability_state_snapshot,
    )
    if not content:
        return existing_index, None, False
    message = {"role": "user", "content": content}
    if (
        existing_index is not None
        and 0 <= existing_index < len(messages)
        and isinstance(messages[existing_index], dict)
    ):
        previous_content = str(messages[existing_index].get("content") or "")
        if previous_content == content:
            return existing_index, content, False
        messages[existing_index] = message
        return existing_index, content, True
    messages.append(message)
    return len(messages) - 1, content, True
