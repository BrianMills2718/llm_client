"""Tests for contract/composability helpers extracted from mcp_agent."""

from __future__ import annotations

from llm_client.agent.agent_contracts import (
    _analyze_lane_closure,
    _effective_contract_requirements,
    _normalize_tool_contracts,
    _validate_tool_contract_call,
)


EVENT_CODE_TOOL_VALIDATION_MISSING_PREREQUISITE = "TOOL_VALIDATION_REJECTED_MISSING_PREREQUISITE"
EVENT_CODE_TOOL_VALIDATION_MISSING_CAPABILITY = "TOOL_VALIDATION_REJECTED_MISSING_CAPABILITY"
EVENT_CODE_TOOL_VALIDATION_BINDING_CONFLICT = "TOOL_VALIDATION_REJECTED_BINDING_CONFLICT"


class TestDynamicContractRequirements:
    def test_call_modes_allow_arg_conditional_prereq_bypass(self) -> None:
        requires_all, requires_any = _effective_contract_requirements(
            "chunk_get_text",
            _normalize_tool_contracts(
                {
                    "chunk_get_text": {
                        "requires_all": ["CHUNK_SET"],
                        "call_modes": [
                            {
                                "name": "by_chunk_ids",
                                "when_args_present_any": ["chunk_id", "chunk_ids"],
                                "artifact_prereqs": "none",
                                "produces": [{"kind": "CHUNK_SET", "ref_type": "fulltext"}],
                            }
                        ],
                    }
                }
            )["chunk_get_text"],
            {"chunk_ids": ["chunk_84"]},
        )
        assert requires_all == set()
        assert requires_any == set()


class TestHandleInputValidation:
    def test_handle_input_reports_missing_runtime_artifact(self) -> None:
        contract = _normalize_tool_contracts(
            {
                "extract_date_mentions_from_artifacts": {
                    "handle_inputs": [
                        {
                            "arg": "chunk_artifact_ids",
                            "inject_arg": "chunk_artifacts",
                            "representation": "envelope",
                            "accepts": [{"kind": "CHUNK_SET", "ref_type": "fulltext"}],
                        }
                    ]
                }
            }
        )["extract_date_mentions_from_artifacts"]

        validation = _validate_tool_contract_call(
            tool_name="extract_date_mentions_from_artifacts",
            contract=contract,
            parsed_args={"chunk_artifact_ids": ["art_missing"]},
            available_artifacts={"QUERY_TEXT"},
            artifact_registry_by_id={},
            event_code_missing_prerequisite=EVENT_CODE_TOOL_VALIDATION_MISSING_PREREQUISITE,
            event_code_missing_capability=EVENT_CODE_TOOL_VALIDATION_MISSING_CAPABILITY,
            event_code_binding_conflict=EVENT_CODE_TOOL_VALIDATION_BINDING_CONFLICT,
        )

        assert validation.is_valid is False
        assert validation.error_code == EVENT_CODE_TOOL_VALIDATION_MISSING_PREREQUISITE
        assert validation.missing_requirements == [{"artifact_id": "art_missing"}]


class TestLaneClosureAnalysis:
    def test_lane_closure_resolves_via_conversion_tool(self) -> None:
        normalized = _normalize_tool_contracts(
            {
                "entity_resolve_names_to_ids": {
                    "requires_all": ["QUERY_TEXT"],
                    "produces": [{"kind": "ENTITY_SET", "ref_type": "id"}],
                },
                "relationship_onehop": {
                    "requires_all": [{"kind": "ENTITY_SET", "ref_type": "id"}],
                    "produces": ["RELATIONSHIP_SET"],
                },
            }
        )
        analysis = _analyze_lane_closure(
            normalized_tool_contracts=normalized,
            initial_artifacts={"QUERY_TEXT"},
            initial_capabilities={"QUERY_TEXT": {(None, None, None)}},
            available_bindings={},
            event_code_missing_prerequisite=EVENT_CODE_TOOL_VALIDATION_MISSING_PREREQUISITE,
            event_code_missing_capability=EVENT_CODE_TOOL_VALIDATION_MISSING_CAPABILITY,
            event_code_binding_conflict=EVENT_CODE_TOOL_VALIDATION_BINDING_CONFLICT,
        )

        assert analysis["lane_closed"] is True
        assert analysis["unresolved_tool_count"] == 0
        assert "ENTITY_SET" in analysis["reachable_artifacts"]
