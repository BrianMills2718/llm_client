from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_client.foundation import (
    check_binding_conflicts,
    evidence_pointer_label,
    extract_artifact_envelopes,
    merge_binding_state,
    validate_artifact_envelope,
    validate_evidence_ref,
    validate_foundation_event,
)


def test_check_binding_conflicts_detects_hard_mismatch() -> None:
    ok, reason, conflicts, proposed = check_binding_conflicts(
        available_bindings={"dataset_id": "musique", "graph_id": "graph_a"},
        proposed_bindings={"dataset_name": "musique", "graph_reference_id": "graph_b"},
    )
    assert ok is False
    assert "binding_conflict" in reason
    assert "graph_id" in conflicts
    assert proposed["graph_id"] == "graph_b"


def test_merge_binding_state_adopts_new_hard_and_updates_soft() -> None:
    merged = merge_binding_state(
        available_bindings={"dataset_id": "musique", "graph_id": None, "model_id": "m1"},
        observed_bindings={"graph_reference_id": "graph_a", "model_id": "m2"},
    )
    assert merged["dataset_id"] == "musique"
    assert merged["graph_id"] == "graph_a"
    assert merged["model_id"] == "m2"


def test_validate_foundation_event_tool_failed_shape() -> None:
    payload = {
        "event_id": "evt_test1",
        "event_type": "ToolFailed",
        "timestamp": "2026-02-21T00:00:00Z",
        "run_id": "run_test",
        "session_id": "sess_test",
        "actor_id": "agent:test",
        "operation": {"name": "entity_onehop", "version": "1.0.0"},
        "inputs": {
            "artifact_ids": ["QUERY_TEXT"],
            "params": {"graph_reference_id": "g1"},
            "bindings": {
                "scope_id": "scope:musique:v1",
                "dataset_id": "musique",
                "corpus_id": "musique_corpus_v1",
                "graph_id": "g1",
                "vector_store_id": None,
                "table_id": None,
                "model_id": "openrouter/deepseek/deepseek-chat",
            },
        },
        "outputs": {"artifact_ids": [], "payload_hashes": []},
        "failure": {
            "error_code": "binding_conflict",
            "category": "validation",
            "phase": "binding_validation",
            "retryable": False,
            "tool_name": "entity_onehop",
            "user_message": "conflict",
        },
    }
    validated = validate_foundation_event(payload)
    assert validated["event_type"] == "ToolFailed"
    assert validated["failure"]["error_code"] == "binding_conflict"


def test_validate_foundation_event_rejects_custom_event_type() -> None:
    payload = {
        "event_id": "evt_test_custom",
        "event_type": "BeliefStatusChanged",
        "timestamp": "2026-02-23T00:00:00Z",
        "run_id": "run_test",
        "session_id": "sess_test",
        "actor_id": "agent:test",
        "operation": {"name": "quarantine", "version": "1.0.0"},
        "inputs": {"artifact_ids": [], "params": {}, "bindings": {}},
        "outputs": {"artifact_ids": [], "payload_hashes": []},
    }
    with pytest.raises(ValidationError):
        validate_foundation_event(payload)


def test_validate_foundation_event_rejects_tool_failed_extra_fields() -> None:
    payload = {
        "event_id": "evt_test2",
        "event_type": "ToolFailed",
        "timestamp": "2026-02-23T00:00:00Z",
        "run_id": "run_test",
        "session_id": "sess_test",
        "actor_id": "agent:test",
        "operation": {"name": "entity_onehop", "version": "1.0.0"},
        "inputs": {"artifact_ids": [], "params": {}, "bindings": {}},
        "outputs": {"artifact_ids": [], "payload_hashes": []},
        "failure": {
            "error_code": "PROVIDER_EMPTY_CANDIDATES",
            "category": "provider",
            "phase": "execution",
            "retryable": True,
            "tool_name": "entity_onehop",
            "user_message": "provider empty response",
            "provider_classification": "empty_candidates",
        },
    }
    with pytest.raises(ValidationError):
        validate_foundation_event(payload)


def test_validate_evidence_ref_requires_locator() -> None:
    with pytest.raises(ValidationError):
        validate_evidence_ref({"backend": "corpus"})


def test_validate_evidence_ref_normalizes_chunk_span_pointer() -> None:
    validated = validate_evidence_ref(
        {
            "backend": "corpus",
            "chunk_id": " chunk_42 ",
            "char_start": 5,
            "char_end": 18,
        }
    )
    assert validated["chunk_id"] == "chunk_42"
    assert evidence_pointer_label(validated) == "chunk:chunk_42#char:5-18"


def test_validate_artifact_envelope_with_provenance_and_capabilities() -> None:
    validated = validate_artifact_envelope(
        {
            "artifact_id": "art_1",
            "artifact_type": "entity_set",
            "schema_version": "1.0.0",
            "bindings": {"dataset_id": "musique", "graph_id": "g1"},
            "capabilities": [
                {"kind": "entity_set", "ref_type": "id"},
            ],
            "provenance": {
                "event_id": "evt_source",
                "evidence_refs": [
                    {"chunk_id": "chunk_99", "char_start": 0, "char_end": 12},
                ],
            },
            "payload": {"items": [{"entity_id": "e1"}]},
        }
    )
    assert validated["artifact_type"] == "ENTITY_SET"
    assert validated["capabilities"][0]["kind"] == "ENTITY_SET"
    assert validated["provenance"]["evidence_refs"][0]["chunk_id"] == "chunk_99"


def test_extract_artifact_envelopes_finds_nested_envelopes_once() -> None:
    payload = {
        "results": [
            {
                "artifact_id": "art_nested",
                "artifact_type": "chunk_set",
                "schema_version": "1.0.0",
                "payload": {"items": [{"chunk_id": "chunk_1"}]},
            },
            {
                "artifact_id": "art_nested",
                "artifact_type": "chunk_set",
                "schema_version": "1.0.0",
                "payload": {"items": [{"chunk_id": "chunk_1"}]},
            },
        ]
    }
    envelopes = extract_artifact_envelopes(payload)
    assert len(envelopes) == 1
    assert envelopes[0]["artifact_type"] == "CHUNK_SET"


def test_validate_foundation_event_artifact_created_with_typed_artifacts() -> None:
    payload = {
        "event_id": "evt_artifacts_1",
        "event_type": "ArtifactCreated",
        "timestamp": "2026-03-15T00:00:00Z",
        "run_id": "run_test",
        "session_id": "sess_test",
        "actor_id": "agent:test",
        "operation": {"name": "chunk_text_search", "version": "1.0.0"},
        "inputs": {"artifact_ids": ["QUERY_TEXT"], "params": {}, "bindings": {}},
        "outputs": {"artifact_ids": ["art_chunk_1"], "payload_hashes": ["sha256:" + "a" * 64]},
        "artifacts": [
            {
                "artifact_id": "art_chunk_1",
                "artifact_type": "chunk_set",
                "schema_version": "1.0.0",
                "provenance": {
                    "evidence_refs": [{"chunk_id": "chunk_1"}],
                },
                "payload": {"items": [{"chunk_id": "chunk_1"}]},
            }
        ],
    }
    validated = validate_foundation_event(payload)
    assert validated["event_type"] == "ArtifactCreated"
    assert validated["artifacts"][0]["artifact_type"] == "CHUNK_SET"
