from __future__ import annotations

from llm_client.foundation import (
    check_binding_conflicts,
    merge_binding_state,
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

