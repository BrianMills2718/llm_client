"""Shared call snapshot, comparison, and replay helpers.

This module captures one durable truth for cross-project observability work:
the normalized call contract at the `llm_client` boundary. That call snapshot
is used for three related jobs:

1. stable request fingerprinting,
2. compact divergence reports between two captured calls,
3. controlled replay of a captured call through the shared runtime.

The goal is not to reconstruct arbitrary project workflow state. The goal is to
make call-level debugging reusable across projects once a workflow has already
reached the shared `llm_client` boundary.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

from llm_client import io_log as _io_log

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
JSONObject = dict[str, JSONValue]

_SNAPSHOT_VERSION = 1
_OBSERVABILITY_ONLY_KWARGS = {
    "task",
    "trace_id",
    "max_budget",
    "prompt_ref",
    "lifecycle_heartbeat_interval_s",
    "lifecycle_stall_after_s",
}


def _qualified_name(value: type[Any]) -> str:
    """Return a stable fully qualified name for one class-like object."""

    return f"{value.__module__}.{value.__qualname__}"


def _normalize_float(value: float) -> JSONValue:
    """Return a JSON-safe float representation without inventing precision."""

    if math.isfinite(value):
        return value
    return repr(value)


def _normalize_json_value(value: Any) -> tuple[JSONValue, bool]:
    """Normalize arbitrary runtime values into JSON-like storage.

    The boolean indicates whether the value remains replay-safe. Primitive JSON,
    `Path`, and recursively JSON-like collections remain replay-safe. Everything
    else is summarized rather than silently dropped, and that summary marks the
    containing key as replay-unsupported.
    """

    if value is None or isinstance(value, (str, int, bool)):
        return value, True
    if isinstance(value, float):
        return _normalize_float(value), math.isfinite(value)
    if isinstance(value, Path):
        return str(value), True
    if isinstance(value, Mapping):
        normalized: JSONObject = {}
        supported = True
        for key in sorted(value.keys(), key=lambda item: str(item)):
            child, child_supported = _normalize_json_value(value[key])
            normalized[str(key)] = child
            supported = supported and child_supported
        return normalized, supported
    if isinstance(value, (list, tuple)):
        normalized_items: list[JSONValue] = []
        supported = True
        for item in value:
            child, child_supported = _normalize_json_value(item)
            normalized_items.append(child)
            supported = supported and child_supported
        return normalized_items, supported
    if isinstance(value, set):
        normalized_items: list[JSONValue] = []
        supported = True
        for item in sorted(value, key=repr):
            child, child_supported = _normalize_json_value(item)
            normalized_items.append(child)
            supported = supported and child_supported
        return normalized_items, supported
    if isinstance(value, type):
        return _qualified_name(value), False

    return {
        "__type__": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
        "__repr__": repr(value),
    }, False


def _normalize_messages(messages: list[dict[str, Any]]) -> list[JSONValue]:
    """Normalize chat messages into a deterministic JSON-like structure."""

    normalized, _ = _normalize_json_value(messages)
    if isinstance(normalized, list):
        return normalized
    raise TypeError("normalized messages must be a list")


def _normalize_public_kwargs(public_kwargs: Mapping[str, Any]) -> tuple[JSONObject, list[str]]:
    """Normalize replay kwargs and collect keys that cannot be replayed exactly."""

    normalized_kwargs: JSONObject = {}
    unsupported_keys: list[str] = []
    for key in sorted(public_kwargs.keys()):
        if key in _OBSERVABILITY_ONLY_KWARGS:
            continue
        value = public_kwargs[key]
        normalized_value, supported = _normalize_json_value(value)
        normalized_kwargs[key] = normalized_value
        if not supported:
            unsupported_keys.append(key)
    return normalized_kwargs, unsupported_keys


def _normalize_response_model_schema(response_model: type[Any] | None) -> JSONValue:
    """Return the full structured-output schema when a Pydantic model is supplied."""

    if response_model is None or not hasattr(response_model, "model_json_schema"):
        return None
    schema = response_model.model_json_schema()
    normalized, _ = _normalize_json_value(schema)
    return normalized


def build_call_snapshot(
    *,
    public_api: str,
    call_kind: str,
    requested_model: str,
    messages: list[dict[str, Any]],
    prompt_ref: str | None,
    timeout: int,
    num_retries: int,
    reasoning_effort: str | None,
    api_base: str | None,
    base_delay: float,
    max_delay: float,
    retry_on: list[str] | None,
    fallback_models: list[str] | None,
    public_kwargs: Mapping[str, Any],
    execution_mode: str | None = None,
    response_model: type[Any] | None = None,
) -> JSONObject:
    """Build the normalized call snapshot used for fingerprinting and replay.

    This captures caller-visible inputs at the `llm_client` boundary and keeps
    observability-only metadata out of the replay identity.
    """

    normalized_kwargs, unsupported_keys = _normalize_public_kwargs(public_kwargs)
    response_model_fqn = _qualified_name(response_model) if response_model is not None else None
    snapshot: JSONObject = {
        "snapshot_version": _SNAPSHOT_VERSION,
        "public_api": public_api,
        "call_kind": call_kind,
        "request": {
            "requested_model": requested_model,
            "messages": _normalize_messages(messages),
            "prompt_ref": prompt_ref,
            "control": {
                "timeout": timeout,
                "num_retries": num_retries,
                "reasoning_effort": reasoning_effort,
                "api_base": api_base,
                "base_delay": base_delay,
                "max_delay": max_delay,
                "retry_on": list(retry_on) if retry_on is not None else None,
                "fallback_models": list(fallback_models) if fallback_models is not None else None,
                "execution_mode": execution_mode,
            },
            "kwargs": normalized_kwargs,
            "response_model_fqn": response_model_fqn,
            "response_model_schema": _normalize_response_model_schema(response_model),
        },
        "replay": {
            "unsupported_keys": unsupported_keys,
        },
    }
    return snapshot


def snapshot_request_identity(snapshot: Mapping[str, Any]) -> JSONObject:
    """Return the canonical request identity used for fingerprinting."""

    request = snapshot.get("request")
    if not isinstance(request, Mapping):
        raise ValueError("snapshot is missing request identity")
    normalized_request, _ = _normalize_json_value(dict(request))
    if not isinstance(normalized_request, dict):
        raise TypeError("normalized request identity must be an object")
    return normalized_request


def snapshot_fingerprint(snapshot: Mapping[str, Any]) -> str:
    """Return a deterministic fingerprint for one normalized call snapshot."""

    payload = json.dumps(
        snapshot_request_identity(snapshot),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _preview_value(value: Any, *, limit: int = 120) -> str:
    """Return a compact human-readable preview for diff output."""

    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=True)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...({len(text)} chars)"


def _diff_json_values(left: Any, right: Any, *, path: str) -> list[str]:
    """Return deterministic compact diffs between two JSON-like values."""

    if left == right:
        return []
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        diffs: list[str] = []
        keys = sorted(set(left.keys()) | set(right.keys()), key=str)
        for key in keys:
            child_path = f"{path}.{key}" if path else str(key)
            if key not in left:
                diffs.append(f"{child_path}: missing on left; right={_preview_value(right[key])}")
                continue
            if key not in right:
                diffs.append(f"{child_path}: left={_preview_value(left[key])}; missing on right")
                continue
            diffs.extend(_diff_json_values(left[key], right[key], path=child_path))
        return diffs
    if isinstance(left, list) and isinstance(right, list):
        diffs: list[str] = []
        max_len = max(len(left), len(right))
        for idx in range(max_len):
            child_path = f"{path}[{idx}]"
            if idx >= len(left):
                diffs.append(f"{child_path}: missing on left; right={_preview_value(right[idx])}")
                continue
            if idx >= len(right):
                diffs.append(f"{child_path}: left={_preview_value(left[idx])}; missing on right")
                continue
            diffs.extend(_diff_json_values(left[idx], right[idx], path=child_path))
        return diffs
    return [f"{path}: left={_preview_value(left)} right={_preview_value(right)}"]


def _decode_json_column(value: str | None) -> JSONValue:
    """Decode one JSON text column if present."""

    if value is None:
        return None
    loaded = json.loads(value)
    normalized, _ = _normalize_json_value(loaded)
    return normalized


def get_call_record(call_id: int) -> dict[str, Any]:
    """Return one persisted call record with decoded snapshot and messages."""

    db = _io_log._get_db()
    row = db.execute(
        """
        SELECT id, timestamp, project, model, messages, response,
               finish_reason, latency_s, error, caller, task, trace_id,
               prompt_ref, call_fingerprint, call_snapshot
        FROM llm_calls
        WHERE id = ?
        """,
        (call_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Call id {call_id} not found.")
    return {
        "id": row[0],
        "timestamp": row[1],
        "project": row[2],
        "model": row[3],
        "messages": _decode_json_column(row[4]),
        "response": row[5],
        "finish_reason": row[6],
        "latency_s": row[7],
        "error": row[8],
        "caller": row[9],
        "task": row[10],
        "trace_id": row[11],
        "prompt_ref": row[12],
        "call_fingerprint": row[13],
        "call_snapshot": _decode_json_column(row[14]),
    }


def get_call_snapshot(call_id: int) -> JSONObject:
    """Return the decoded call snapshot for one call id."""

    snapshot = get_call_record(call_id)["call_snapshot"]
    if not isinstance(snapshot, dict):
        raise ValueError(f"Call id {call_id} does not have a replayable call snapshot.")
    return snapshot


def compare_call_snapshots(left_call_id: int, right_call_id: int) -> dict[str, Any]:
    """Compare two captured calls and return a compact divergence report."""

    left = get_call_record(left_call_id)
    right = get_call_record(right_call_id)
    left_snapshot = left["call_snapshot"]
    right_snapshot = right["call_snapshot"]
    if not isinstance(left_snapshot, dict):
        raise ValueError(f"Call id {left_call_id} does not have a replayable call snapshot.")
    if not isinstance(right_snapshot, dict):
        raise ValueError(f"Call id {right_call_id} does not have a replayable call snapshot.")

    left_request = snapshot_request_identity(left_snapshot)
    right_request = snapshot_request_identity(right_snapshot)
    report = {
        "left_call_id": left_call_id,
        "right_call_id": right_call_id,
        "left_fingerprint": left["call_fingerprint"] or snapshot_fingerprint(left_snapshot),
        "right_fingerprint": right["call_fingerprint"] or snapshot_fingerprint(right_snapshot),
        "fingerprints_match": (left["call_fingerprint"] or snapshot_fingerprint(left_snapshot))
        == (right["call_fingerprint"] or snapshot_fingerprint(right_snapshot)),
        "request_differences": _diff_json_values(left_request, right_request, path="request"),
        "result_differences": _diff_json_values(
            {
                "model": left["model"],
                "finish_reason": left["finish_reason"],
                "error": left["error"],
                "response": left["response"],
            },
            {
                "model": right["model"],
                "finish_reason": right["finish_reason"],
                "error": right["error"],
                "response": right["response"],
            },
            path="result",
        ),
        "left_summary": {
            "project": left["project"],
            "caller": left["caller"],
            "task": left["task"],
            "trace_id": left["trace_id"],
            "model": left["model"],
            "error": left["error"],
        },
        "right_summary": {
            "project": right["project"],
            "caller": right["caller"],
            "task": right["task"],
            "trace_id": right["trace_id"],
            "model": right["model"],
            "error": right["error"],
        },
    }
    return report


def format_call_diff(report: Mapping[str, Any]) -> str:
    """Render a compact human-readable divergence report."""

    left_call_id = report["left_call_id"]
    right_call_id = report["right_call_id"]
    header = [
        f"compare {left_call_id} vs {right_call_id}",
        f"fingerprints_match={report['fingerprints_match']}",
    ]
    request_diffs = list(report.get("request_differences", []))
    result_diffs = list(report.get("result_differences", []))
    lines = header
    lines.append("request:")
    lines.extend(
        [f"  - {diff}" for diff in request_diffs] if request_diffs else ["  - no request differences"]
    )
    lines.append("result:")
    lines.extend(
        [f"  - {diff}" for diff in result_diffs] if result_diffs else ["  - no result differences"]
    )
    return "\n".join(lines)


def _resolve_response_model(model_fqn: str) -> type[Any]:
    """Import and return a structured response model from its fully qualified name."""

    module_name, _, qualname = model_fqn.rpartition(".")
    if not module_name or not qualname:
        raise ValueError(f"Invalid response model path: {model_fqn!r}")
    module = importlib.import_module(module_name)
    current: Any = module
    for part in qualname.split("."):
        current = getattr(current, part)
    if not isinstance(current, type):
        raise TypeError(f"Resolved response model is not a type: {model_fqn!r}")
    return current


def _call_text_for_replay(
    model: str,
    messages: list[dict[str, Any]],
    **kwargs: Any,
) -> Any:
    """Dispatch one text replay through the shared public runtime."""

    from llm_client import call_llm

    return call_llm(model, messages, **kwargs)


def _call_structured_for_replay(
    model: str,
    messages: list[dict[str, Any]],
    response_model: type[Any],
    **kwargs: Any,
) -> Any:
    """Dispatch one structured replay through the shared public runtime."""

    from llm_client import call_llm_structured

    return call_llm_structured(model, messages, response_model, **kwargs)


async def _acall_text_for_replay(
    model: str,
    messages: list[dict[str, Any]],
    **kwargs: Any,
) -> Any:
    """Dispatch one async text replay through the shared public runtime."""

    from llm_client import acall_llm

    return await acall_llm(model, messages, **kwargs)


async def _acall_structured_for_replay(
    model: str,
    messages: list[dict[str, Any]],
    response_model: type[Any],
    **kwargs: Any,
) -> Any:
    """Dispatch one async structured replay through the shared public runtime."""

    from llm_client import acall_llm_structured

    return await acall_llm_structured(model, messages, response_model, **kwargs)


@contextmanager
def _temporary_project_override(project: str | None) -> Any:
    """Temporarily override the active observability project for one replay."""

    if project is None:
        yield
        return
    old_project = _io_log._project
    try:
        _io_log.configure(project=project)
        yield
    finally:
        _io_log.configure(project=old_project)


def replay_call_snapshot(
    call_id: int,
    *,
    trace_id: str,
    task: str | None = None,
    max_budget: float = 0.0,
    project: str | None = None,
) -> dict[str, Any]:
    """Replay one captured call snapshot through the shared runtime.

    Replay is intentionally call-level. If the original call depended on
    workflow state that never reached `llm_client`, the owning project must
    reconstruct that state first and then hand this module a prepared call.
    """

    record = get_call_record(call_id)
    snapshot = record["call_snapshot"]
    if not isinstance(snapshot, dict):
        raise ValueError(f"Call id {call_id} does not have a replayable call snapshot.")

    replay = snapshot.get("replay")
    unsupported_keys = (
        list(replay.get("unsupported_keys", []))
        if isinstance(replay, Mapping) and isinstance(replay.get("unsupported_keys"), list)
        else []
    )
    if unsupported_keys:
        joined = ", ".join(sorted(str(key) for key in unsupported_keys))
        raise ValueError(
            f"Call id {call_id} includes replay-unsupported kwargs: {joined}. "
            "Replay would not be exact, so llm_client refuses it."
        )

    request = snapshot_request_identity(snapshot)
    messages = request.get("messages")
    control = request.get("control")
    public_kwargs = request.get("kwargs")
    if not isinstance(messages, list):
        raise ValueError(f"Call id {call_id} snapshot is missing request.messages.")
    if not isinstance(control, Mapping):
        raise ValueError(f"Call id {call_id} snapshot is missing request.control.")
    if not isinstance(public_kwargs, Mapping):
        raise ValueError(f"Call id {call_id} snapshot is missing request.kwargs.")

    replay_task = task or record["task"] or f"observability.replay.{snapshot.get('public_api', 'call')}"
    replay_project = project if project is not None else record["project"]
    call_kwargs: dict[str, Any] = {
        "timeout": control.get("timeout", 60),
        "num_retries": control.get("num_retries", 0),
        "reasoning_effort": control.get("reasoning_effort"),
        "api_base": control.get("api_base"),
        "base_delay": control.get("base_delay", 1.0),
        "max_delay": control.get("max_delay", 30.0),
        "retry_on": control.get("retry_on"),
        "fallback_models": control.get("fallback_models"),
        "task": replay_task,
        "trace_id": trace_id,
        "max_budget": max_budget,
        "prompt_ref": request.get("prompt_ref"),
        **dict(public_kwargs),
    }

    requested_model = request.get("requested_model")
    if not isinstance(requested_model, str) or not requested_model:
        raise ValueError(f"Call id {call_id} snapshot is missing request.requested_model.")

    public_api = str(snapshot.get("public_api", "call_llm"))
    with _temporary_project_override(replay_project):
        if public_api == "call_llm":
            result = _call_text_for_replay(requested_model, messages, **call_kwargs)
        elif public_api == "acall_llm":
            result = asyncio.run(_acall_text_for_replay(requested_model, messages, **call_kwargs))
        elif public_api == "call_llm_structured":
            model_fqn = request.get("response_model_fqn")
            if not isinstance(model_fqn, str) or not model_fqn:
                raise ValueError(f"Call id {call_id} snapshot is missing request.response_model_fqn.")
            response_model = _resolve_response_model(model_fqn)
            result = _call_structured_for_replay(
                requested_model,
                messages,
                response_model,
                **call_kwargs,
            )
        elif public_api == "acall_llm_structured":
            model_fqn = request.get("response_model_fqn")
            if not isinstance(model_fqn, str) or not model_fqn:
                raise ValueError(f"Call id {call_id} snapshot is missing request.response_model_fqn.")
            response_model = _resolve_response_model(model_fqn)
            result = asyncio.run(
                _acall_structured_for_replay(
                    requested_model,
                    messages,
                    response_model,
                    **call_kwargs,
                )
            )
        else:
            raise ValueError(
                f"Replay is not supported for public_api={public_api!r}. "
                "Only call_llm/acall_llm/call_llm_structured/acall_llm_structured are supported."
            )

    return {
        "source_call_id": call_id,
        "replay_trace_id": trace_id,
        "task": replay_task,
        "project": replay_project,
        "public_api": public_api,
        "result": result,
    }
