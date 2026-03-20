"""FOUNDATION v1.1 runtime helpers.

Centralizes:
- binding normalization/conflict checks
- deterministic IDs/hashing helpers
- machine-validated event envelopes (Pydantic)
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator


BINDING_KEYS: tuple[str, ...] = (
    "scope_id",
    "dataset_id",
    "corpus_id",
    "graph_id",
    "vector_store_id",
    "table_id",
    "model_id",
)

HARD_BINDING_KEYS: tuple[str, ...] = (
    "scope_id",
    "dataset_id",
    "corpus_id",
    "graph_id",
    "vector_store_id",
    "table_id",
)

SOFT_BINDING_KEYS: tuple[str, ...] = ("model_id",)

_BINDING_ARG_ALIASES: dict[str, tuple[str, ...]] = {
    "scope_id": ("scope_id",),
    "dataset_id": ("dataset_id", "dataset_name"),
    "corpus_id": ("corpus_id", "document_collection_id"),
    "graph_id": ("graph_id", "graph_reference_id"),
    "vector_store_id": ("vector_store_id", "vdb_reference_id", "vector_reference_id"),
    "table_id": ("table_id", "table_reference_id"),
    "model_id": ("model_id",),
}


def _norm_binding_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value).strip() or None


def _norm_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value).strip() or None


class BindingSet(BaseModel):
    """Normalized binding state used for compatibility checks."""

    model_config = ConfigDict(extra="forbid")

    scope_id: str | None = None
    dataset_id: str | None = None
    corpus_id: str | None = None
    graph_id: str | None = None
    vector_store_id: str | None = None
    table_id: str | None = None
    model_id: str | None = None


class EvidenceRef(BaseModel):
    """Typed evidence pointer consumed by runtime observability helpers."""

    model_config = ConfigDict(extra="ignore")

    backend: str | None = None
    uri: str | None = None
    doc_id: str | None = None
    chunk_id: str | None = None
    artifact_id: str | None = None
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    line_start: int | None = Field(default=None, ge=0)
    line_end: int | None = Field(default=None, ge=0)
    content_sha256: str | None = None
    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_fields(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        payload = dict(data)
        for key in (
            "backend",
            "uri",
            "doc_id",
            "chunk_id",
            "artifact_id",
            "content_sha256",
            "notes",
        ):
            if key in payload:
                payload[key] = _norm_optional_text(payload.get(key))
        return payload

    @model_validator(mode="after")
    def _validate_locator_and_spans(self) -> "EvidenceRef":
        if not any((self.uri, self.doc_id, self.chunk_id, self.artifact_id)):
            raise ValueError(
                "EvidenceRef requires at least one locator: uri, doc_id, chunk_id, or artifact_id"
            )
        if (self.char_start is None) != (self.char_end is None):
            raise ValueError("EvidenceRef char span requires both char_start and char_end")
        if (
            self.char_start is not None
            and self.char_end is not None
            and self.char_end < self.char_start
        ):
            raise ValueError("EvidenceRef char_end must be >= char_start")
        if (self.line_start is None) != (self.line_end is None):
            raise ValueError("EvidenceRef line span requires both line_start and line_end")
        if (
            self.line_start is not None
            and self.line_end is not None
            and self.line_end < self.line_start
        ):
            raise ValueError("EvidenceRef line_end must be >= line_start")
        return self


class ArtifactCapability(BaseModel):
    """Typed capability tags optionally emitted with artifact envelopes."""

    model_config = ConfigDict(extra="ignore")

    kind: str = Field(min_length=1)
    ref_type: str | None = None
    namespace: str | None = None
    bindings_hash: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_fields(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        payload = dict(data)
        kind = payload.get("kind")
        if isinstance(kind, str):
            payload["kind"] = kind.strip().upper()
        for key in ("ref_type", "namespace", "bindings_hash"):
            if key in payload:
                payload[key] = _norm_optional_text(payload.get(key))
        return payload


class ArtifactProducer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tool_name: str = Field(min_length=1)
    tool_version: str | None = None
    tool_instance_id: str | None = None
    code_hash: str | None = None
    registry_digest: str | None = None


class ArtifactProvenance(BaseModel):
    model_config = ConfigDict(extra="ignore")

    event_id: str | None = None
    upstream_event_ids: list[str] = Field(default_factory=list)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    notes: str | None = None


class ArtifactEnvelope(BaseModel):
    """Typed artifact wrapper for output-driven runtime state and provenance."""

    model_config = ConfigDict(extra="ignore")

    artifact_id: str = Field(min_length=1)
    artifact_type: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    created_at: str | None = None
    producer: ArtifactProducer | None = None
    input_artifact_ids: list[str] = Field(default_factory=list)
    bindings: BindingSet = Field(default_factory=BindingSet)
    capabilities: list[ArtifactCapability] = Field(default_factory=list)
    payload: Any = Field(default_factory=dict)
    payload_sha256: str | None = None
    payload_ref: dict[str, Any] | None = None
    provenance: ArtifactProvenance = Field(default_factory=ArtifactProvenance)
    confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="before")
    @classmethod
    def _normalize_fields(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        payload = dict(data)
        artifact_type = payload.get("artifact_type")
        if isinstance(artifact_type, str):
            payload["artifact_type"] = artifact_type.strip().upper()
        for key in (
            "artifact_id",
            "schema_version",
            "created_at",
            "payload_sha256",
        ):
            if key in payload:
                payload[key] = _norm_optional_text(payload.get(key))
        return payload


def empty_bindings() -> dict[str, str | None]:
    return BindingSet().model_dump()


def extract_bindings_from_tool_args(args: Mapping[str, Any] | None) -> dict[str, str | None]:
    """Extract canonical bindings from tool arguments (supports aliases)."""
    if not isinstance(args, Mapping):
        return {}

    out: dict[str, str | None] = {}
    for canonical, aliases in _BINDING_ARG_ALIASES.items():
        for alias in aliases:
            if alias not in args:
                continue
            norm = _norm_binding_value(args.get(alias))
            if norm is not None:
                out[canonical] = norm
                break
    return out


def normalize_bindings(bindings: Mapping[str, Any] | None) -> dict[str, str | None]:
    """Normalize arbitrary binding dict/args into canonical binding shape."""
    if not isinstance(bindings, Mapping):
        return empty_bindings()

    payload: dict[str, Any] = {}
    for key in BINDING_KEYS:
        if key in bindings:
            payload[key] = _norm_binding_value(bindings.get(key))
    payload.update(extract_bindings_from_tool_args(bindings))
    return BindingSet(**payload).model_dump()


def check_binding_conflicts(
    *,
    available_bindings: Mapping[str, Any] | None,
    proposed_bindings: Mapping[str, Any] | None,
) -> tuple[bool, str, dict[str, tuple[str, str]], dict[str, str | None]]:
    """Return whether proposed bindings are compatible with current hard bindings."""
    available = normalize_bindings(available_bindings)
    proposed = normalize_bindings(proposed_bindings)

    conflicts: dict[str, tuple[str, str]] = {}
    for key in HARD_BINDING_KEYS:
        existing = available.get(key)
        incoming = proposed.get(key)
        if existing and incoming and existing != incoming:
            conflicts[key] = (existing, incoming)

    if conflicts:
        details = ", ".join(
            f"{k}(existing={v[0]!r}, proposed={v[1]!r})"
            for k, v in sorted(conflicts.items())
        )
        return (
            False,
            f"binding_conflict: {details}",
            conflicts,
            proposed,
        )

    return True, "", {}, proposed


def merge_binding_state(
    *,
    available_bindings: Mapping[str, Any] | None,
    observed_bindings: Mapping[str, Any] | None,
) -> dict[str, str | None]:
    """Merge observed bindings into available state under authority rules."""
    merged = normalize_bindings(available_bindings)
    observed = normalize_bindings(observed_bindings)

    for key in HARD_BINDING_KEYS:
        if merged.get(key) is None and observed.get(key):
            merged[key] = observed[key]

    for key in SOFT_BINDING_KEYS:
        if observed.get(key):
            merged[key] = observed[key]

    return merged


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(payload: Any) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return sha256_text(raw)


def new_event_id() -> str:
    return f"evt_{uuid.uuid4().hex}"


def new_session_id() -> str:
    return f"sess_{uuid.uuid4().hex}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def coerce_run_id(run_id: str | None, trace_id: str | None = None) -> str:
    raw = (run_id or "").strip()
    if not raw:
        raw = (trace_id or "").strip()
    if not raw:
        raw = uuid.uuid4().hex
    safe = re.sub(r"[^A-Za-z0-9._:-]+", "_", raw)
    if not safe.startswith("run_"):
        safe = f"run_{safe}"
    return safe


_EVIDENCE_REF_ADAPTER: TypeAdapter[EvidenceRef] = TypeAdapter(EvidenceRef)
_ARTIFACT_ENVELOPE_ADAPTER: TypeAdapter[ArtifactEnvelope] = TypeAdapter(ArtifactEnvelope)


def validate_evidence_ref(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one evidence reference payload."""
    parsed = _EVIDENCE_REF_ADAPTER.validate_python(payload)
    return parsed.model_dump(mode="json", exclude_none=True)


def parse_evidence_ref(payload: Any) -> dict[str, Any] | None:
    """Best-effort EvidenceRef parse; returns None instead of raising."""
    if not isinstance(payload, Mapping):
        return None
    try:
        return validate_evidence_ref(payload)
    except Exception:
        return None


def evidence_pointer_label(raw: Any) -> str | None:
    """Canonicalize one evidence reference into a stable pointer label."""
    if isinstance(raw, str):
        ref = raw.strip()
        if not ref:
            return None
        if ref.startswith("chunk_"):
            return f"chunk:{ref}"
        if ":" in ref:
            kind, locator = ref.split(":", 1)
            kind_norm = kind.strip().lower()
            locator_norm = locator.strip()
            if kind_norm and locator_norm:
                return f"{kind_norm}:{locator_norm}"
            return None
        return f"raw:{ref}"

    parsed = parse_evidence_ref(raw)
    if parsed is None:
        return None

    base: str | None = None
    if parsed.get("chunk_id"):
        base = f"chunk:{parsed['chunk_id']}"
    elif parsed.get("artifact_id"):
        base = f"artifact:{parsed['artifact_id']}"
    elif parsed.get("doc_id"):
        base = f"doc:{parsed['doc_id']}"
    elif parsed.get("uri"):
        base = f"uri:{parsed['uri']}"

    if not base:
        return None

    char_start = parsed.get("char_start")
    char_end = parsed.get("char_end")
    if isinstance(char_start, int) and isinstance(char_end, int):
        return f"{base}#char:{char_start}-{char_end}"

    line_start = parsed.get("line_start")
    line_end = parsed.get("line_end")
    if isinstance(line_start, int) and isinstance(line_end, int):
        return f"{base}#line:{line_start}-{line_end}"

    return base


def validate_artifact_envelope(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one artifact-envelope payload."""
    parsed = _ARTIFACT_ENVELOPE_ADAPTER.validate_python(payload)
    return parsed.model_dump(mode="json", exclude_none=True)


def parse_artifact_envelope(payload: Any) -> dict[str, Any] | None:
    """Best-effort ArtifactEnvelope parse; returns None instead of raising."""
    if not isinstance(payload, Mapping):
        return None
    if not {"artifact_id", "artifact_type", "schema_version"} <= set(payload.keys()):
        return None
    try:
        return validate_artifact_envelope(payload)
    except Exception:
        return None


def extract_artifact_envelopes(payload: Any) -> list[dict[str, Any]]:
    """Recursively extract normalized artifact envelopes from arbitrary JSON payloads."""
    found: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    def _walk(value: Any) -> None:
        envelope = parse_artifact_envelope(value)
        if envelope is not None:
            artifact_id = str(envelope.get("artifact_id", "")).strip()
            if artifact_id and artifact_id not in seen_ids:
                seen_ids.add(artifact_id)
                found.append(envelope)

        if isinstance(value, Mapping):
            for child in value.values():
                if isinstance(child, (Mapping, list, tuple)):
                    _walk(child)
            return

        if isinstance(value, (list, tuple)):
            for child in value:
                if isinstance(child, (Mapping, list, tuple)):
                    _walk(child)

    _walk(payload)
    return found


class EventOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    version: str | None = None


class EventInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifact_ids: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    bindings: BindingSet = Field(default_factory=BindingSet)


class EventOutputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifact_ids: list[str] = Field(default_factory=list)
    payload_hashes: list[str] = Field(default_factory=list)


class FoundationEventBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(pattern=r"^evt_[A-Za-z0-9._:-]+$")
    event_type: Literal[
        "ToolCalled",
        "ToolFailed",
        "ArtifactCreated",
        "LLMCalled",
        "LLMCallLifecycle",
        "DecisionMade",
        "RuleRegistered",
        "ConfigChanged",
        "BindingChanged",
    ]
    timestamp: str
    run_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    actor_id: str = Field(min_length=1)
    operation: EventOperation
    inputs: EventInputs = Field(default_factory=EventInputs)
    outputs: EventOutputs = Field(default_factory=EventOutputs)


class ToolFailedPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    error_code: str = Field(min_length=1)
    category: Literal["validation", "execution", "provider", "policy"]
    phase: Literal["input_validation", "binding_validation", "execution", "post_validation"]
    retryable: bool
    tool_name: str = Field(min_length=1)
    user_message: str = Field(min_length=1)
    debug_ref: str | None = None


class LLMCalledPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str = Field(min_length=1)
    content_persisted: Literal["hash_only", "full_redacted", "full_raw"]
    prompt_sha256: str = Field(pattern=r"^sha256:[0-9a-fA-F]{64}$")
    response_sha256: str = Field(pattern=r"^sha256:[0-9a-fA-F]{64}$")
    token_usage: dict[str, Any] = Field(default_factory=dict)
    cost_usd: float | None = Field(default=None, ge=0)


class LLMCallLifecyclePayload(BaseModel):
    """Lifecycle state for one public LLM call.

    This payload distinguishes plain liveness from explicit observed progress.
    `heartbeat` means the client runtime is still waiting. `progress` is only
    emitted on paths where the runtime can truthfully observe forward motion,
    such as stream chunks or successful background polls.
    """

    model_config = ConfigDict(extra="forbid")

    call_id: str = Field(pattern=r"^llmcall_[A-Za-z0-9._:-]+$")
    phase: Literal["started", "heartbeat", "progress", "stalled", "completed", "failed"]
    call_kind: Literal["text", "structured"]
    requested_model_id: str = Field(min_length=1)
    resolved_model_id: str | None = None
    provider_timeout_s: int | None = Field(default=None, ge=0)
    timeout_policy: Literal["allow", "ban"]
    prompt_ref: str | None = None
    host_name: str | None = None
    process_id: int | None = Field(default=None, ge=1)
    process_start_token: str | None = None
    progress_observable: bool | None = None
    progress_source: str | None = None
    progress_event_count: int | None = Field(default=None, ge=0)
    elapsed_s: float | None = Field(default=None, ge=0)
    latency_s: float | None = Field(default=None, ge=0)
    heartbeat_interval_s: float | None = Field(default=None, ge=0)
    stall_after_s: float | None = Field(default=None, ge=0)
    error_type: str | None = None
    error_message: str | None = None


class DecisionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision_id: str = Field(min_length=1)
    chosen_action: str = Field(min_length=1)
    action_args_summary: str = ""
    signals: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    alternatives_considered: list[str] = Field(default_factory=list)


class BindingChangedPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    old_bindings: BindingSet
    new_bindings: BindingSet
    reason: str = Field(min_length=1)


class ToolCalledEvent(FoundationEventBase):
    event_type: Literal["ToolCalled"]


class ToolFailedEvent(FoundationEventBase):
    event_type: Literal["ToolFailed"]
    failure: ToolFailedPayload


class ArtifactCreatedEvent(FoundationEventBase):
    event_type: Literal["ArtifactCreated"]
    artifacts: list[ArtifactEnvelope] = Field(default_factory=list)


class LLMCalledEvent(FoundationEventBase):
    event_type: Literal["LLMCalled"]
    llm: LLMCalledPayload


class LLMCallLifecycleEvent(FoundationEventBase):
    """Foundation event for public-call lifecycle telemetry."""

    event_type: Literal["LLMCallLifecycle"]
    llm_call_lifecycle: LLMCallLifecyclePayload


class DecisionMadeEvent(FoundationEventBase):
    event_type: Literal["DecisionMade"]
    decision: DecisionPayload


class RuleRegisteredEvent(FoundationEventBase):
    event_type: Literal["RuleRegistered"]


class ConfigChangedEvent(FoundationEventBase):
    event_type: Literal["ConfigChanged"]


class BindingChangedEvent(FoundationEventBase):
    event_type: Literal["BindingChanged"]
    binding_change: BindingChangedPayload


FoundationEvent = Annotated[
    ToolCalledEvent
    | ToolFailedEvent
    | ArtifactCreatedEvent
    | LLMCalledEvent
    | LLMCallLifecycleEvent
    | DecisionMadeEvent
    | RuleRegisteredEvent
    | ConfigChangedEvent
    | BindingChangedEvent,
    Field(discriminator="event_type"),
]

_FOUNDATION_EVENT_ADAPTER: TypeAdapter[FoundationEvent] = TypeAdapter(FoundationEvent)


def validate_foundation_event(event_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one foundation event payload."""
    parsed = _FOUNDATION_EVENT_ADAPTER.validate_python(event_payload)
    return parsed.model_dump(mode="json", exclude_none=True)


__all__ = [
    "BINDING_KEYS",
    "HARD_BINDING_KEYS",
    "SOFT_BINDING_KEYS",
    "ArtifactCapability",
    "ArtifactEnvelope",
    "ArtifactProducer",
    "ArtifactProvenance",
    "BindingSet",
    "EvidenceRef",
    "check_binding_conflicts",
    "coerce_run_id",
    "empty_bindings",
    "evidence_pointer_label",
    "extract_artifact_envelopes",
    "extract_bindings_from_tool_args",
    "merge_binding_state",
    "new_event_id",
    "new_session_id",
    "normalize_bindings",
    "now_iso",
    "parse_artifact_envelope",
    "parse_evidence_ref",
    "sha256_json",
    "sha256_text",
    "validate_artifact_envelope",
    "validate_evidence_ref",
    "validate_foundation_event",
]
