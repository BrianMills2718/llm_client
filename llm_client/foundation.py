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

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


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


class LLMCalledEvent(FoundationEventBase):
    event_type: Literal["LLMCalled"]
    llm: LLMCalledPayload


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
    "BindingSet",
    "check_binding_conflicts",
    "coerce_run_id",
    "empty_bindings",
    "extract_bindings_from_tool_args",
    "merge_binding_state",
    "new_event_id",
    "new_session_id",
    "normalize_bindings",
    "now_iso",
    "sha256_json",
    "sha256_text",
    "validate_foundation_event",
]
