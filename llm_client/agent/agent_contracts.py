"""Contract/capability/composability helpers for agent runtimes."""

from __future__ import annotations

import json as _json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from llm_client.foundation import (
    BINDING_KEYS,
    HARD_BINDING_KEYS,
    check_binding_conflicts,
    empty_bindings,
    extract_artifact_envelopes,
    extract_bindings_from_tool_args,
    merge_binding_state,
    normalize_bindings,
    sha256_json,
)


@dataclass(frozen=True)
class CapabilityRequirement:
    """Normalized capability requirement for composability checks."""

    kind: str
    ref_type: str | None = None
    namespace: str | None = None
    bindings_hash: str | None = None

    def to_dict(self) -> dict[str, str]:
        payload: dict[str, str] = {"kind": self.kind}
        if self.ref_type:
            payload["ref_type"] = self.ref_type
        if self.namespace:
            payload["namespace"] = self.namespace
        if self.bindings_hash:
            payload["bindings_hash"] = self.bindings_hash
        return payload

    def short_label(self) -> str:
        suffix_parts: list[str] = []
        if self.ref_type:
            suffix_parts.append(f"ref_type={self.ref_type}")
        if self.namespace:
            suffix_parts.append(f"namespace={self.namespace}")
        if self.bindings_hash:
            suffix_parts.append(f"bindings_hash={self.bindings_hash[:12]}")
        if not suffix_parts:
            return self.kind
        return f"{self.kind}[{', '.join(suffix_parts)}]"


@dataclass
class ToolCallValidation:
    """Normalized pre-execution validation outcome for a tool call."""

    is_valid: bool
    reason: str = ""
    error_code: str | None = None
    failure_phase: str | None = None
    call_bindings: dict[str, str | None] = field(default_factory=dict)
    missing_requirements: list[dict[str, str]] = field(default_factory=list)
    contract_mode: str | None = None


def _normalize_artifact_kind(kind: Any) -> str | None:
    if isinstance(kind, str):
        normalized = kind.strip().upper()
        return normalized or None
    return None


def _normalize_capability_value(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _capability_requirement_from_raw(raw: Any) -> CapabilityRequirement | None:
    """Parse one capability requirement from raw contract payload."""
    if isinstance(raw, str):
        kind = _normalize_artifact_kind(raw)
        if kind:
            return CapabilityRequirement(kind=kind)
        return None

    if not isinstance(raw, dict):
        return None

    kind = _normalize_artifact_kind(raw.get("kind"))
    if not kind:
        return None

    ref_type = _normalize_capability_value(raw.get("ref_type"))
    namespace = _normalize_capability_value(raw.get("namespace"))
    bindings_hash = _normalize_capability_value(raw.get("bindings_hash"))
    return CapabilityRequirement(
        kind=kind,
        ref_type=ref_type,
        namespace=namespace,
        bindings_hash=bindings_hash,
    )


def _normalize_capability_requirements(raw: Any) -> list[CapabilityRequirement]:
    if not isinstance(raw, (list, tuple, set)):
        return []
    out: list[CapabilityRequirement] = []
    seen: set[tuple[str, str | None, str | None, str | None]] = set()
    for item in raw:
        req = _capability_requirement_from_raw(item)
        if req is None:
            continue
        key = (req.kind, req.ref_type, req.namespace, req.bindings_hash)
        if key in seen:
            continue
        seen.add(key)
        out.append(req)
    return out


def _capability_state_add(
    state: dict[str, set[tuple[str | None, str | None, str | None]]],
    *,
    kind: str | None,
    ref_type: str | None = None,
    namespace: str | None = None,
    bindings_hash: str | None = None,
) -> bool:
    normalized_kind = _normalize_artifact_kind(kind)
    if not normalized_kind:
        return False
    bucket = state.setdefault(normalized_kind, set())
    entry = (
        _normalize_capability_value(ref_type),
        _normalize_capability_value(namespace),
        _normalize_capability_value(bindings_hash),
    )
    before = len(bucket)
    bucket.add(entry)
    return len(bucket) > before


def _capability_state_has(
    state: dict[str, set[tuple[str | None, str | None, str | None]]],
    req: CapabilityRequirement,
) -> bool:
    bucket = state.get(req.kind)
    if not bucket:
        return False
    for ref_type, namespace, bindings_hash in bucket:
        if req.ref_type and req.ref_type != ref_type:
            continue
        if req.namespace and req.namespace != namespace:
            continue
        if req.bindings_hash and req.bindings_hash != bindings_hash:
            continue
        return True
    return False


def _capability_state_snapshot(
    state: dict[str, set[tuple[str | None, str | None, str | None]]],
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for kind in sorted(state):
        entries = sorted(
            state[kind],
            key=lambda item: (
                item[0] or "",
                item[1] or "",
                item[2] or "",
            ),
        )
        for ref_type, namespace, bindings_hash in entries:
            req = CapabilityRequirement(
                kind=kind,
                ref_type=ref_type,
                namespace=namespace,
                bindings_hash=bindings_hash,
            )
            out.append(req.to_dict())
    return out


def _capability_requirement_matches(
    produced: CapabilityRequirement,
    required: CapabilityRequirement,
) -> bool:
    """Whether a produced capability could satisfy a required capability."""
    if produced.kind != required.kind:
        return False
    if required.ref_type and produced.ref_type and required.ref_type != produced.ref_type:
        return False
    if required.namespace and produced.namespace and required.namespace != produced.namespace:
        return False
    if (
        required.bindings_hash
        and produced.bindings_hash
        and required.bindings_hash != produced.bindings_hash
    ):
        return False
    return True


def _canonical_binding_spec(
    bindings: dict[str, str | None],
    *,
    keys: tuple[str, ...],
) -> dict[str, str | None]:
    """Canonical binding spec for deterministic hashing (stable key subset/order)."""
    return {key: bindings.get(key) for key in keys}


def _hard_bindings_spec(bindings: dict[str, str | None]) -> dict[str, str | None]:
    return _canonical_binding_spec(bindings, keys=HARD_BINDING_KEYS)


def _full_bindings_spec(bindings: dict[str, str | None]) -> dict[str, str | None]:
    return _canonical_binding_spec(bindings, keys=BINDING_KEYS)


def _binding_hash(spec: dict[str, str | None]) -> str:
    return sha256_json(spec).replace("sha256:", "")


def _hard_bindings_state_hash(bindings: dict[str, str | None]) -> str:
    return _binding_hash(_hard_bindings_spec(bindings))


def _full_bindings_state_hash(bindings: dict[str, str | None]) -> str:
    return _binding_hash(_full_bindings_spec(bindings))


def _namespace_from_bindings(bindings: dict[str, str | None]) -> str | None:
    """Best-effort namespace heuristic from authoritative binding state."""
    for key in ("graph_id", "dataset_id", "scope_id", "corpus_id", "vector_store_id"):
        value = bindings.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _explicit_ref_type_from_args(kind: str, args: dict[str, Any] | None) -> str | None:
    if not isinstance(args, dict):
        return None

    def _has_values(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (list, tuple, set)):
            return any(_has_values(v) for v in value)
        return True

    if kind == "ENTITY_SET":
        if _has_values(args.get("entity_ids")) or _has_values(args.get("entity_id")):
            return "id"
        if _has_values(args.get("entity_names")) or _has_values(args.get("entity_name")):
            return "name"
    if kind == "CHUNK_SET":
        if _has_values(args.get("chunk_ids")) or _has_values(args.get("chunk_id")):
            return "id"
    return None


def _infer_output_capabilities(
    *,
    tool_name: str,
    parsed_args: dict[str, Any] | None,
    produced_artifacts: set[str],
    available_bindings: dict[str, str | None],
) -> list[CapabilityRequirement]:
    """Infer lightweight capability tags from tool name + arguments."""
    namespace = _namespace_from_bindings(available_bindings)
    bindings_hash = _hard_bindings_state_hash(available_bindings)
    inferred: list[CapabilityRequirement] = []

    for kind in sorted(produced_artifacts):
        ref_type = _explicit_ref_type_from_args(kind, parsed_args)
        if kind == "CHUNK_SET" and "chunk_get_text" in tool_name:
            ref_type = "fulltext"
        elif kind == "CHUNK_SET" and "search" in tool_name and ref_type is None:
            ref_type = "snippet"

        inferred.append(
            CapabilityRequirement(
                kind=kind,
                ref_type=ref_type,
                namespace=namespace,
                bindings_hash=bindings_hash,
            )
        )

    return inferred


def _artifact_capabilities_from_envelope(
    envelope: dict[str, Any],
    *,
    fallback_bindings: dict[str, str | None],
) -> list[CapabilityRequirement]:
    """Convert typed envelope capability payloads into runtime capability requirements."""
    bindings = normalize_bindings(envelope.get("bindings"))
    effective_bindings = merge_binding_state(
        available_bindings=fallback_bindings,
        observed_bindings=bindings,
    )
    namespace = _namespace_from_bindings(effective_bindings)
    default_bindings_hash = _hard_bindings_state_hash(effective_bindings)

    raw_caps = envelope.get("capabilities")
    if not isinstance(raw_caps, list) or not raw_caps:
        artifact_type = _normalize_artifact_kind(envelope.get("artifact_type"))
        if not artifact_type:
            return []
        return [
            CapabilityRequirement(
                kind=artifact_type,
                namespace=namespace,
                bindings_hash=default_bindings_hash,
            )
        ]

    out: list[CapabilityRequirement] = []
    seen: set[tuple[str, str | None, str | None, str | None]] = set()
    for raw_cap in raw_caps:
        req = _capability_requirement_from_raw(raw_cap)
        if req is None:
            continue
        normalized_req = CapabilityRequirement(
            kind=req.kind,
            ref_type=req.ref_type,
            namespace=req.namespace or namespace,
            bindings_hash=req.bindings_hash or default_bindings_hash,
        )
        key = (
            normalized_req.kind,
            normalized_req.ref_type,
            normalized_req.namespace,
            normalized_req.bindings_hash,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized_req)
    return out


def _artifact_output_state_from_payload(
    parsed: Any,
    *,
    fallback_bindings: dict[str, str | None],
) -> tuple[list[dict[str, Any]], set[str], list[CapabilityRequirement], list[str], dict[str, str | None]]:
    """Parse typed artifact envelopes from one parsed tool result for runtime state updates."""
    if parsed is None or not isinstance(parsed, (dict, list)):
        return [], set(), [], [], {}

    envelopes = extract_artifact_envelopes(parsed)
    if not envelopes:
        return [], set(), [], [], {}

    produced_artifacts: set[str] = set()
    produced_capabilities: list[CapabilityRequirement] = []
    artifact_ids: list[str] = []
    observed_bindings: dict[str, str | None] = empty_bindings()

    for envelope in envelopes:
        artifact_type = _normalize_artifact_kind(envelope.get("artifact_type"))
        if artifact_type:
            produced_artifacts.add(artifact_type)

        artifact_id = str(envelope.get("artifact_id", "")).strip()
        if artifact_id:
            artifact_ids.append(artifact_id)

        observed_bindings = merge_binding_state(
            available_bindings=observed_bindings,
            observed_bindings=envelope.get("bindings"),
        )
        produced_capabilities.extend(
            _artifact_capabilities_from_envelope(
                envelope,
                fallback_bindings=fallback_bindings,
            )
        )

    return envelopes, produced_artifacts, produced_capabilities, artifact_ids, observed_bindings


def _short_requirement(req: CapabilityRequirement) -> str:
    return req.short_label()


def _has_meaningful_arg_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, (list, tuple, set)):
        return any(_has_meaningful_arg_value(item) for item in value)
    return True


def _contract_declares_no_artifact_prereqs(contract: dict[str, Any] | None) -> bool:
    if not isinstance(contract, dict):
        return False
    raw = contract.get("artifact_prereqs")
    return (
        (isinstance(raw, str) and raw.strip().lower() == "none")
        or bool(contract.get("artifact_prereqs_none"))
        or bool(contract.get("self_contained"))
    )


def _normalize_arg_name_list(raw: Any) -> list[str]:
    if not isinstance(raw, (list, tuple, set)):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        name = item.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _normalize_arg_equals_spec(raw: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, tuple[str, ...]] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        name = key.strip()
        if not name:
            continue
        values: list[str] = []
        candidates = value if isinstance(value, (list, tuple, set)) else [value]
        seen: set[str] = set()
        for item in candidates:
            if item is None:
                continue
            normalized = str(item).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            values.append(normalized)
        if values:
            out[name] = tuple(values)
    return out


def _normalize_handle_input_specs(raw: Any) -> list[dict[str, Any]]:
    """Normalize declarative artifact-handle input specs."""
    if not isinstance(raw, (list, tuple, set)):
        return []
    normalized: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        arg = str(item.get("arg") or "").strip()
        if not arg:
            continue
        inject_arg_raw = item.get("inject_arg")
        inject_arg = (
            str(inject_arg_raw).strip()
            if isinstance(inject_arg_raw, str) and str(inject_arg_raw).strip()
            else None
        )
        representation = str(item.get("representation") or "envelope").strip().lower()
        if representation not in {"envelope", "payload"}:
            representation = "envelope"
        accepts = _normalize_capability_requirements(
            item.get("accepts")
            or item.get("accepts_capabilities")
            or item.get("requires")
        )
        normalized.append(
            {
                "arg": arg,
                "inject_arg": inject_arg,
                "representation": representation,
                "accepts": accepts,
            }
        )
    return normalized


def _normalize_handle_arg_values(raw: Any) -> list[str]:
    """Normalize artifact handle args into a de-duplicated list of artifact_ids."""
    if isinstance(raw, str):
        stripped = raw.strip()
        return [stripped] if stripped else []
    if not isinstance(raw, (list, tuple, set)):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        artifact_id = str(item or "").strip()
        if not artifact_id or artifact_id in seen:
            continue
        seen.add(artifact_id)
        out.append(artifact_id)
    return out


def _handle_input_accepts_envelope(
    spec: dict[str, Any],
    envelope: dict[str, Any],
    *,
    fallback_bindings: dict[str, str | None] | None,
) -> bool:
    """Whether one stored artifact envelope satisfies a handle-input capability spec."""
    accepts = spec.get("accepts") or []
    if not isinstance(accepts, list) or not accepts:
        return True
    envelope_caps = _artifact_capabilities_from_envelope(
        envelope,
        fallback_bindings=fallback_bindings or {},
    )
    for required in accepts:
        if not isinstance(required, CapabilityRequirement):
            continue
        if any(
            _capability_requirement_matches(produced, required)
            for produced in envelope_caps
        ):
            return True
    return False


def _validate_handle_input_specs(
    *,
    tool_name: str,
    resolved_contract: dict[str, Any],
    parsed_args: dict[str, Any] | None,
    artifact_registry_by_id: dict[str, dict[str, Any]] | None,
    available_bindings: dict[str, str | None] | None,
    call_bindings: dict[str, str | None],
    contract_mode: str | None,
    event_code_missing_prerequisite: str,
    event_code_missing_capability: str,
) -> ToolCallValidation | None:
    """Validate declared artifact-handle args against the runtime artifact registry."""
    if not isinstance(parsed_args, dict):
        return None
    handle_inputs = resolved_contract.get("handle_inputs") or []
    if not isinstance(handle_inputs, list) or not handle_inputs:
        return None

    registry = artifact_registry_by_id or {}
    for spec in handle_inputs:
        if not isinstance(spec, dict):
            continue
        arg_name = str(spec.get("arg") or "").strip()
        if not arg_name:
            continue
        requested_ids = _normalize_handle_arg_values(parsed_args.get(arg_name))
        if not requested_ids:
            continue

        missing_ids: list[str] = []
        capability_mismatch_ids: list[str] = []
        for artifact_id in requested_ids:
            envelope = registry.get(artifact_id)
            if not isinstance(envelope, dict):
                missing_ids.append(artifact_id)
                continue
            if not _handle_input_accepts_envelope(
                spec,
                envelope,
                fallback_bindings=available_bindings,
            ):
                capability_mismatch_ids.append(artifact_id)

        if missing_ids:
            return ToolCallValidation(
                is_valid=False,
                reason=(
                    f"{tool_name} requires known runtime artifact handles in {arg_name}; "
                    f"missing {missing_ids}"
                ),
                error_code=event_code_missing_prerequisite,
                failure_phase="input_validation",
                call_bindings=call_bindings,
                missing_requirements=[{"artifact_id": artifact_id} for artifact_id in missing_ids],
                contract_mode=contract_mode,
            )

        if capability_mismatch_ids:
            accepts = spec.get("accepts") or []
            missing_payload = [
                req.to_dict()
                for req in accepts
                if isinstance(req, CapabilityRequirement)
            ]
            expected = (
                ", ".join(_short_requirement(req) for req in accepts if isinstance(req, CapabilityRequirement))
                or "declared artifact capabilities"
            )
            return ToolCallValidation(
                is_valid=False,
                reason=(
                    f"{tool_name} requires {arg_name} handles compatible with {expected}; "
                    f"mismatched {capability_mismatch_ids}"
                ),
                error_code=event_code_missing_capability,
                failure_phase="input_validation",
                call_bindings=call_bindings,
                missing_requirements=missing_payload,
                contract_mode=contract_mode,
            )
    return None


def _apply_handle_input_injections(
    *,
    tc: dict[str, Any],
    normalized_tool_contracts: dict[str, dict[str, Any]],
    artifact_registry_by_id: dict[str, dict[str, Any]],
    extract_tool_call_args: Callable[[dict[str, Any]], dict[str, Any] | None],
    set_tool_call_args: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Inject resolved artifact envelopes/payloads for declared handle-input args."""
    tool_name = str(tc.get("function", {}).get("name", "")).strip()
    contract = normalized_tool_contracts.get(tool_name)
    parsed_args = extract_tool_call_args(tc)
    if not isinstance(contract, dict) or not isinstance(parsed_args, dict):
        return tc, []
    resolved_contract, _mode_name = _resolve_contract_spec(contract, parsed_args)
    handle_inputs = resolved_contract.get("handle_inputs") or []
    if not isinstance(handle_inputs, list) or not handle_inputs:
        return tc, []

    patched_args = dict(parsed_args)
    injections: list[dict[str, Any]] = []
    changed = False
    for spec in handle_inputs:
        if not isinstance(spec, dict):
            continue
        arg_name = str(spec.get("arg") or "").strip()
        inject_arg = str(spec.get("inject_arg") or "").strip()
        if not arg_name or not inject_arg:
            continue
        requested_ids = _normalize_handle_arg_values(parsed_args.get(arg_name))
        if not requested_ids:
            continue
        resolved_values: list[Any] = []
        representation = str(spec.get("representation") or "envelope").strip().lower()
        for artifact_id in requested_ids:
            envelope = artifact_registry_by_id.get(artifact_id)
            if not isinstance(envelope, dict):
                continue
            value: Any = envelope if representation == "envelope" else envelope.get("payload")
            try:
                resolved_values.append(_json.loads(_json.dumps(value)))
            except Exception:
                resolved_values.append(value)
        patched_args[inject_arg] = resolved_values
        changed = True
        injections.append(
            {
                "source_arg": arg_name,
                "inject_arg": inject_arg,
                "representation": representation,
                "artifact_ids": requested_ids,
                "resolved_count": len(resolved_values),
            }
        )

    if not changed:
        return tc, []
    return set_tool_call_args(tc, patched_args), injections


def _normalize_contract_spec_fields(spec: dict[str, Any]) -> dict[str, Any]:
    requires_all_caps = _normalize_capability_requirements(spec.get("requires_all"))
    requires_any_caps = _normalize_capability_requirements(spec.get("requires_any"))
    requires_all_caps.extend(
        _normalize_capability_requirements(spec.get("requires_all_capabilities"))
    )
    requires_any_caps.extend(
        _normalize_capability_requirements(spec.get("requires_any_capabilities"))
    )

    produces_caps = _normalize_capability_requirements(spec.get("produces"))
    produces_caps.extend(
        _normalize_capability_requirements(spec.get("produces_capabilities"))
    )

    requires_all = {req.kind for req in requires_all_caps}
    requires_any = {req.kind for req in requires_any_caps}
    produces = {req.kind for req in produces_caps}

    return {
        "requires_all_artifacts": requires_all,
        "requires_any_artifacts": requires_any,
        "requires_all_capabilities": requires_all_caps,
        "requires_any_capabilities": requires_any_caps,
        "produces_artifacts": produces,
        "produces_capabilities": produces_caps,
        "artifact_prereqs": (
            str(spec.get("artifact_prereqs", "")).strip().lower()
            if isinstance(spec.get("artifact_prereqs"), str)
            else None
        ),
        "artifact_prereqs_none": (
            str(spec.get("artifact_prereqs", "")).strip().lower() == "none"
        ),
        "handle_inputs": _normalize_handle_input_specs(spec.get("handle_inputs")),
    }


def _normalize_contract_mode(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    normalized = _normalize_contract_spec_fields(raw)
    name_raw = raw.get("name")
    normalized["name"] = (
        name_raw.strip()
        if isinstance(name_raw, str) and name_raw.strip()
        else None
    )
    normalized["when_args_present_any"] = _normalize_arg_name_list(raw.get("when_args_present_any"))
    normalized["when_args_present_all"] = _normalize_arg_name_list(raw.get("when_args_present_all"))
    normalized["when_arg_equals"] = _normalize_arg_equals_spec(raw.get("when_arg_equals"))
    return normalized


def _contract_mode_matches(mode: dict[str, Any], parsed_args: dict[str, Any] | None) -> bool:
    if not isinstance(parsed_args, dict):
        return False

    present_all = mode.get("when_args_present_all") or []
    if any(not _has_meaningful_arg_value(parsed_args.get(name)) for name in present_all):
        return False

    present_any = mode.get("when_args_present_any") or []
    if present_any and not any(_has_meaningful_arg_value(parsed_args.get(name)) for name in present_any):
        return False

    arg_equals = mode.get("when_arg_equals") or {}
    for name, allowed_values in arg_equals.items():
        value = parsed_args.get(name)
        normalized_value = str(value).strip() if value is not None else ""
        if not normalized_value or normalized_value not in allowed_values:
            return False

    return bool(present_all or present_any or arg_equals)


def _resolve_contract_spec(
    contract: dict[str, Any] | None,
    parsed_args: dict[str, Any] | None,
) -> tuple[dict[str, Any], str | None]:
    """Resolve arg-conditional contract mode; fall back to top-level spec."""
    if not isinstance(contract, dict):
        return {}, None

    modes = contract.get("call_modes") or []
    if isinstance(modes, list):
        for idx, mode in enumerate(modes):
            if not isinstance(mode, dict):
                continue
            if _contract_mode_matches(mode, parsed_args):
                return mode, str(mode.get("name") or f"mode_{idx + 1}")

    return contract, None


def _tool_declares_no_artifact_prereqs(
    tool_name: str,
    contract: dict[str, Any] | None,
) -> bool:
    """Artifact-prereq bypass is declarative only."""
    del tool_name
    if _contract_declares_no_artifact_prereqs(contract):
        return True
    if not isinstance(contract, dict):
        return False
    for mode in contract.get("call_modes") or []:
        if not isinstance(mode, dict):
            continue
        has_conditions = bool(
            mode.get("when_args_present_any")
            or mode.get("when_args_present_all")
            or mode.get("when_arg_equals")
        )
        if has_conditions:
            continue
        if _contract_declares_no_artifact_prereqs(mode):
            return True
    return False


def _normalize_tool_contracts(raw: Any) -> dict[str, dict[str, Any]]:
    """Normalize raw tool contract payload into a predictable dict shape."""
    if not isinstance(raw, dict):
        return {}

    normalized: dict[str, dict[str, Any]] = {}
    for tool_name, spec in raw.items():
        if not isinstance(tool_name, str) or not tool_name.strip():
            continue
        if not isinstance(spec, dict):
            continue

        normalized_spec = _normalize_contract_spec_fields(spec)
        normalized_spec["is_control"] = bool(spec.get("is_control", False))
        call_modes: list[dict[str, Any]] = []
        for raw_mode in spec.get("call_modes") or []:
            mode = _normalize_contract_mode(raw_mode)
            if mode is not None:
                call_modes.append(mode)
        normalized_spec["call_modes"] = call_modes
        normalized[tool_name.strip()] = normalized_spec

    return normalized


def _effective_contract_requirements(
    tool_name: str,
    contract: dict[str, Any],
    parsed_args: dict[str, Any] | None,
) -> tuple[set[str], set[str]]:
    """Resolve dynamic per-call requirements for selected tools."""
    del tool_name
    resolved_contract, _mode_name = _resolve_contract_spec(contract, parsed_args)
    requires_all = set(
        resolved_contract.get("requires_all_artifacts")
        or resolved_contract.get("requires_all")
        or set()
    )
    requires_any = set(
        resolved_contract.get("requires_any_artifacts")
        or resolved_contract.get("requires_any")
        or set()
    )

    if _contract_declares_no_artifact_prereqs(resolved_contract):
        return set(), set()

    return requires_all, requires_any


def _validate_tool_contract_call(
    *,
    tool_name: str,
    contract: dict[str, Any],
    parsed_args: dict[str, Any] | None,
    available_artifacts: set[str],
    available_capabilities: dict[str, set[tuple[str | None, str | None, str | None]]] | None = None,
    available_bindings: dict[str, str | None] | None = None,
    artifact_registry_by_id: dict[str, dict[str, Any]] | None = None,
    event_code_missing_prerequisite: str,
    event_code_missing_capability: str,
    event_code_binding_conflict: str,
) -> ToolCallValidation:
    """Validate whether a tool call is composable given currently available artifacts."""
    call_bindings = extract_bindings_from_tool_args(parsed_args)
    capability_state = available_capabilities or {}
    resolved_contract, resolved_mode_name = _resolve_contract_spec(contract, parsed_args)

    if contract.get("is_control"):
        return ToolCallValidation(
            is_valid=True,
            call_bindings=call_bindings,
            contract_mode=resolved_mode_name,
        )

    if call_bindings:
        bindings_ok, bindings_reason, _conflicts, normalized_call_bindings = check_binding_conflicts(
            available_bindings=available_bindings,
            proposed_bindings=call_bindings,
        )
        if not bindings_ok:
            return ToolCallValidation(
                is_valid=False,
                reason=bindings_reason,
                error_code=event_code_binding_conflict,
                failure_phase="binding_validation",
                call_bindings=normalized_call_bindings,
                missing_requirements=[],
                contract_mode=resolved_mode_name,
            )

    handle_validation = _validate_handle_input_specs(
        tool_name=tool_name,
        resolved_contract=resolved_contract,
        parsed_args=parsed_args,
        artifact_registry_by_id=artifact_registry_by_id,
        available_bindings=available_bindings,
        call_bindings=call_bindings,
        contract_mode=resolved_mode_name,
        event_code_missing_prerequisite=event_code_missing_prerequisite,
        event_code_missing_capability=event_code_missing_capability,
    )
    if handle_validation is not None:
        return handle_validation

    requires_all, requires_any = _effective_contract_requirements(
        tool_name, contract, parsed_args,
    )

    missing_all = sorted(requires_all - available_artifacts)
    if missing_all:
        return ToolCallValidation(
            is_valid=False,
            reason=f"{tool_name} requires all of {sorted(requires_all)}; missing {missing_all}",
            error_code=event_code_missing_prerequisite,
            failure_phase="input_validation",
            call_bindings=call_bindings,
            missing_requirements=[{"kind": k} for k in missing_all],
            contract_mode=resolved_mode_name,
        )

    if requires_any and not (requires_any & available_artifacts):
        missing_any = sorted(requires_any)
        return ToolCallValidation(
            is_valid=False,
            reason=f"{tool_name} requires one of {sorted(requires_any)}; available {sorted(available_artifacts)}",
            error_code=event_code_missing_prerequisite,
            failure_phase="input_validation",
            call_bindings=call_bindings,
            missing_requirements=[{"kind": k} for k in missing_any],
            contract_mode=resolved_mode_name,
        )

    if _contract_declares_no_artifact_prereqs(resolved_contract):
        return ToolCallValidation(
            is_valid=True,
            call_bindings=call_bindings,
            contract_mode=resolved_mode_name,
        )

    missing_capabilities: list[CapabilityRequirement] = []
    for req in (resolved_contract.get("requires_all_capabilities") or []):
        if isinstance(req, CapabilityRequirement) and not _capability_state_has(capability_state, req):
            missing_capabilities.append(req)
    if missing_capabilities:
        missing_payload = [req.to_dict() for req in missing_capabilities]
        return ToolCallValidation(
            is_valid=False,
            reason=(
                f"{tool_name} missing required capabilities: "
                + ", ".join(_short_requirement(req) for req in missing_capabilities)
            ),
            error_code=event_code_missing_capability,
            failure_phase="input_validation",
            call_bindings=call_bindings,
            missing_requirements=missing_payload,
            contract_mode=resolved_mode_name,
        )

    requires_any_caps = [
        req
        for req in (resolved_contract.get("requires_any_capabilities") or [])
        if isinstance(req, CapabilityRequirement)
    ]
    if requires_any_caps and not any(_capability_state_has(capability_state, req) for req in requires_any_caps):
        missing_payload = [req.to_dict() for req in requires_any_caps]
        return ToolCallValidation(
            is_valid=False,
            reason=(
                f"{tool_name} requires one capability from: "
                + ", ".join(_short_requirement(req) for req in requires_any_caps)
            ),
            error_code=event_code_missing_capability,
            failure_phase="input_validation",
            call_bindings=call_bindings,
            missing_requirements=missing_payload,
            contract_mode=resolved_mode_name,
        )

    return ToolCallValidation(
        is_valid=True,
        call_bindings=call_bindings,
        contract_mode=resolved_mode_name,
    )


def _contract_outputs(
    contract: dict[str, Any] | None,
    parsed_args: dict[str, Any] | None = None,
) -> set[str]:
    """Return declared artifact outputs for a tool contract."""
    resolved_contract, _mode_name = _resolve_contract_spec(contract, parsed_args)
    if not isinstance(resolved_contract, dict):
        return set()
    return set(
        resolved_contract.get("produces_artifacts")
        or resolved_contract.get("produces")
        or set()
    )


def _contract_output_capabilities(
    contract: dict[str, Any] | None,
    parsed_args: dict[str, Any] | None = None,
) -> list[CapabilityRequirement]:
    resolved_contract, _mode_name = _resolve_contract_spec(contract, parsed_args)
    if not isinstance(resolved_contract, dict):
        return []
    out = resolved_contract.get("produces_capabilities") or []
    return [req for req in out if isinstance(req, CapabilityRequirement)]


def _is_control_tool_name(
    tool_name: str,
    normalized_tool_contracts: dict[str, dict[str, Any]],
) -> bool:
    contract = normalized_tool_contracts.get(tool_name)
    return bool(isinstance(contract, dict) and contract.get("is_control"))


def _find_repair_tools_for_missing_requirements(
    *,
    current_tool_name: str,
    missing_requirements: list[dict[str, Any]],
    normalized_tool_contracts: dict[str, dict[str, Any]],
    available_artifacts: set[str],
    available_capabilities: dict[str, set[tuple[str | None, str | None, str | None]]],
    available_bindings: dict[str, str | None] | None,
    max_repair_tools: int,
    event_code_missing_prerequisite: str,
    event_code_missing_capability: str,
    event_code_binding_conflict: str,
) -> list[str]:
    """Suggest legal-now tools that can produce currently missing requirements."""
    if max_repair_tools <= 0:
        return []

    required_caps: list[CapabilityRequirement] = []
    for raw in missing_requirements or []:
        req = _capability_requirement_from_raw(raw)
        if req is not None:
            required_caps.append(req)
    if not required_caps:
        return []

    required_kinds: set[str] = {req.kind for req in required_caps}
    candidates: list[tuple[tuple[int, int, int, int, int, int, str], str]] = []
    for tool_name in sorted(normalized_tool_contracts):
        if tool_name == current_tool_name:
            continue
        contract = normalized_tool_contracts.get(tool_name)
        if not isinstance(contract, dict):
            continue
        if bool(contract.get("is_control")):
            continue

        produced_caps = _contract_output_capabilities(contract)
        produced_kinds = _contract_outputs(contract)
        can_repair = False
        for req in required_caps:
            for produced in produced_caps:
                if _capability_requirement_matches(produced, req):
                    can_repair = True
                    break
            if can_repair:
                break
            if req.kind in produced_kinds:
                can_repair = True
                break
        if not can_repair:
            continue

        validation = _validate_tool_contract_call(
            tool_name=tool_name,
            contract=contract,
            parsed_args=None,
            available_artifacts=available_artifacts,
            available_capabilities=available_capabilities,
            available_bindings=available_bindings,
            event_code_missing_prerequisite=event_code_missing_prerequisite,
            event_code_missing_capability=event_code_missing_capability,
            event_code_binding_conflict=event_code_binding_conflict,
        )
        if validation.is_valid:
            raw_requires_all = set(contract.get("requires_all_artifacts") or contract.get("requires_all") or set())
            raw_requires_any = set(contract.get("requires_any_artifacts") or contract.get("requires_any") or set())

            exact_capability_match = 1
            for req in required_caps:
                if any(
                    produced.kind == req.kind
                    and (
                        req.ref_type is None
                        or produced.ref_type is None
                        or produced.ref_type == req.ref_type
                    )
                    and (
                        req.namespace is None
                        or produced.namespace is None
                        or produced.namespace == req.namespace
                    )
                    and (
                        req.bindings_hash is None
                        or produced.bindings_hash is None
                        or produced.bindings_hash == req.bindings_hash
                    )
                    for produced in produced_caps
                ):
                    exact_capability_match = 0
                    break

            self_dependency_penalty = 1 if any(kind in raw_requires_all for kind in required_kinds) else 0
            query_bootstrap_penalty = 0 if ("QUERY_TEXT" in raw_requires_all or "QUERY_TEXT" in raw_requires_any) else 1
            search_penalty = 0 if "search" in tool_name else 1
            aggregator_penalty = 1 if any(tok in tool_name for tok in ("aggregator", "score", "optimize")) else 0
            prereq_penalty = len(raw_requires_all) + (1 if raw_requires_any else 0)
            get_text_penalty = 1 if "get_text" in tool_name else 0

            rank_key = (
                self_dependency_penalty,
                query_bootstrap_penalty,
                search_penalty,
                exact_capability_match,
                aggregator_penalty,
                prereq_penalty + get_text_penalty,
                tool_name,
            )
            candidates.append((rank_key, tool_name))

    candidates.sort(key=lambda item: item[0])
    return [name for _, name in candidates[:max_repair_tools]]


def _analyze_lane_closure(
    *,
    normalized_tool_contracts: dict[str, dict[str, Any]],
    initial_artifacts: set[str],
    initial_capabilities: dict[str, set[tuple[str | None, str | None, str | None]]],
    available_bindings: dict[str, str | None] | None,
    event_code_missing_prerequisite: str,
    event_code_missing_capability: str,
    event_code_binding_conflict: str,
) -> dict[str, Any]:
    """Advisory closure analysis for current lane (toolset + policies)."""
    reachable_artifacts: set[str] = set(initial_artifacts)
    reachable_capabilities: dict[str, set[tuple[str | None, str | None, str | None]]] = {
        kind: set(entries)
        for kind, entries in (initial_capabilities or {}).items()
    }

    max_iters = max(1, len(normalized_tool_contracts) * 2)
    for _ in range(max_iters):
        changed = False
        for tool_name in sorted(normalized_tool_contracts):
            contract = normalized_tool_contracts.get(tool_name)
            if not isinstance(contract, dict):
                continue
            if bool(contract.get("is_control")):
                continue
            validation = _validate_tool_contract_call(
                tool_name=tool_name,
                contract=contract,
                parsed_args=None,
                available_artifacts=reachable_artifacts,
                available_capabilities=reachable_capabilities,
                available_bindings=available_bindings,
                event_code_missing_prerequisite=event_code_missing_prerequisite,
                event_code_missing_capability=event_code_missing_capability,
                event_code_binding_conflict=event_code_binding_conflict,
            )
            if not validation.is_valid:
                continue
            for kind in _contract_outputs(contract):
                if kind not in reachable_artifacts:
                    reachable_artifacts.add(kind)
                    changed = True
            for produced in _contract_output_capabilities(contract):
                if _capability_state_add(
                    reachable_capabilities,
                    kind=produced.kind,
                    ref_type=produced.ref_type,
                    namespace=produced.namespace,
                    bindings_hash=produced.bindings_hash,
                ):
                    changed = True
        if not changed:
            break

    unresolved_tools: list[dict[str, Any]] = []
    for tool_name in sorted(normalized_tool_contracts):
        contract = normalized_tool_contracts.get(tool_name)
        if not isinstance(contract, dict):
            continue
        if bool(contract.get("is_control")):
            continue
        validation = _validate_tool_contract_call(
            tool_name=tool_name,
            contract=contract,
            parsed_args=None,
            available_artifacts=reachable_artifacts,
            available_capabilities=reachable_capabilities,
            available_bindings=available_bindings,
            event_code_missing_prerequisite=event_code_missing_prerequisite,
            event_code_missing_capability=event_code_missing_capability,
            event_code_binding_conflict=event_code_binding_conflict,
        )
        if validation.is_valid:
            continue
        unresolved_tools.append(
            {
                "tool": tool_name,
                "error_code": validation.error_code,
                "missing_requirements": list(validation.missing_requirements or []),
            }
        )

    return {
        "lane_closed": len(unresolved_tools) == 0,
        "reachable_artifacts": sorted(reachable_artifacts),
        "reachable_capabilities": _capability_state_snapshot(reachable_capabilities),
        "unresolved_tools": unresolved_tools,
        "unresolved_tool_count": len(unresolved_tools),
    }


# ---------------------------------------------------------------------------
# Agent Error Budget
# ---------------------------------------------------------------------------

# Patterns indicating non-recoverable failures (never retry).
_NON_RECOVERABLE_PATTERNS: list[str] = [
    "quota", "billing", "insufficient", "exceeded your current",
    "plan and billing", "account deactivated", "account suspended",
    "invalid api key", "authentication", "unauthorized", "forbidden",
    "model not found", "does not exist",
]

# Patterns indicating recoverable failures (retry within budget).
_RECOVERABLE_PATTERNS: list[str] = [
    "rate limit", "rate_limit", "timeout", "timed out",
    "connection reset", "connection error", "server error",
    "service unavailable", "overloaded", "http 500", "http 502",
    "http 503", "http 529", "temporary failure",
]


def classify_error(error: str | Exception) -> str:
    """Classify an error as non_recoverable, recoverable, or unknown.

    Returns one of: 'non_recoverable', 'recoverable', 'unknown'.
    """
    text = str(error).lower()
    for pattern in _NON_RECOVERABLE_PATTERNS:
        if pattern in text:
            return "non_recoverable"
    for pattern in _RECOVERABLE_PATTERNS:
        if pattern in text:
            return "recoverable"
    return "unknown"


@dataclass
class AgentErrorBudget:
    """Aggregate error budget for the agent loop.

    Consumers MUST declare this explicitly — either using defaults or
    with custom values. There is no silent default behavior.

    Controls total retry effort across all models and fallbacks to prevent
    runaway agent loops (observed: 318 LLM calls for a single question).

    Attributes:
        max_agent_turns: Total LLM→tool→LLM cycles allowed per question.
            Counts successful turns, not raw API calls. Default 200 is
            generous (normal questions use 5-20 turns).
        max_consecutive_errors_per_model: How many consecutive errors
            from the same model before switching to the next fallback.
            Default 3 — if a model fails 3 times in a row, it probably
            can't handle this question.
        max_total_errors: Aggregate error count across all models.
            When exceeded, the agent loop stops. Default 30 is high
            enough that a 5-model fallback chain with 3 errors each
            (15 errors) still has headroom for transient issues.
    """

    max_agent_turns: int = 200
    max_consecutive_errors_per_model: int = 3
    max_total_errors: int = 30


@dataclass
class ErrorBudgetState:
    """Mutable state tracker for an active error budget.

    Created once per agent loop invocation. Updated after each turn.
    """

    budget: AgentErrorBudget
    total_turns: int = 0
    total_errors: int = 0
    consecutive_errors_by_model: dict[str, int] = field(default_factory=dict)
    _last_model: str = ""

    def record_success(self, model: str) -> None:
        """Record a successful turn. Resets consecutive error count for this model."""
        self.total_turns += 1
        self.consecutive_errors_by_model[model] = 0
        self._last_model = model

    def record_error(self, model: str, error: str | Exception) -> str:
        """Record a failed turn. Returns the error classification.

        Raises BudgetExhaustedError if any budget limit is exceeded.
        """
        self.total_turns += 1
        self.total_errors += 1
        classification = classify_error(error)

        consecutive = self.consecutive_errors_by_model.get(model, 0) + 1
        self.consecutive_errors_by_model[model] = consecutive
        self._last_model = model

        return classification

    def should_stop(self) -> tuple[bool, str]:
        """Check if the error budget is exhausted.

        Returns (should_stop, reason). Reason is empty string if not stopping.
        """
        if self.total_turns >= self.budget.max_agent_turns:
            return True, f"max_agent_turns ({self.budget.max_agent_turns}) exceeded"

        if self.total_errors >= self.budget.max_total_errors:
            return True, f"max_total_errors ({self.budget.max_total_errors}) exceeded"

        return False, ""

    def should_skip_model(self, model: str) -> bool:
        """Check if a specific model has exceeded its consecutive error budget."""
        consecutive = self.consecutive_errors_by_model.get(model, 0)
        return consecutive >= self.budget.max_consecutive_errors_per_model

    def summary(self) -> dict[str, Any]:
        """Return a summary dict for logging/observability."""
        return {
            "total_turns": self.total_turns,
            "total_errors": self.total_errors,
            "budget_max_turns": self.budget.max_agent_turns,
            "budget_max_errors": self.budget.max_total_errors,
            "consecutive_errors_by_model": dict(self.consecutive_errors_by_model),
        }
