"""Tests for disclosure helpers extracted from mcp_agent."""

from __future__ import annotations

from llm_client.agent.agent_contracts import (
    _capability_requirement_from_raw,
    _normalize_tool_contracts,
    _validate_tool_contract_call,
)
from llm_client.agent.agent_disclosure import (
    _deficit_labels_from_hidden_entries,
    _disclosure_message,
    _disclosure_reason_from_entry,
    _filter_tools_for_disclosure,
)


EVENT_CODE_TOOL_VALIDATION_MISSING_PREREQUISITE = "TOOL_VALIDATION_REJECTED_MISSING_PREREQUISITE"
EVENT_CODE_TOOL_VALIDATION_MISSING_CAPABILITY = "TOOL_VALIDATION_REJECTED_MISSING_CAPABILITY"
EVENT_CODE_TOOL_VALIDATION_BINDING_CONFLICT = "TOOL_VALIDATION_REJECTED_BINDING_CONFLICT"


def _short_requirement(req: object) -> str:
    return getattr(req, "short_label")()


class TestDisclosureHelpers:
    def test_disclosure_reason_prefers_capability_labels_and_repairs(self) -> None:
        entry = {
            "tool": "relationship_onehop",
            "missing_requirements": [{"kind": "ENTITY_SET", "ref_type": "id"}],
            "repair_tools": ["entity_resolve_names_to_ids"],
        }

        reason = _disclosure_reason_from_entry(
            entry,
            capability_requirement_from_raw=_capability_requirement_from_raw,
            short_requirement=_short_requirement,
        )

        assert "ENTITY_SET[ref_type=id]" in reason
        assert "entity_resolve_names_to_ids" in reason

    def test_deficit_labels_are_deduplicated(self) -> None:
        hidden = [
            {"missing_requirements": [{"kind": "ENTITY_SET", "ref_type": "id"}]},
            {"missing_requirements": [{"kind": "ENTITY_SET", "ref_type": "id"}]},
        ]
        labels = _deficit_labels_from_hidden_entries(
            hidden,
            capability_requirement_from_raw=_capability_requirement_from_raw,
            short_requirement=_short_requirement,
        )
        assert labels == ["ENTITY_SET[ref_type=id]"]


class TestDisclosureFiltering:
    def test_filter_hides_unavailable_tool_and_surfaces_repair(self) -> None:
        contracts = _normalize_tool_contracts(
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
        openai_tools = [
            {"type": "function", "function": {"name": "entity_resolve_names_to_ids"}},
            {"type": "function", "function": {"name": "relationship_onehop"}},
        ]

        visible, hidden, hidden_total = _filter_tools_for_disclosure(
            openai_tools=openai_tools,
            normalized_tool_contracts=contracts,
            available_artifacts={"QUERY_TEXT"},
            available_capabilities={"QUERY_TEXT": {(None, None, None)}},
            available_bindings={},
            max_unavailable=10,
            max_missing_per_tool=2,
            max_repair_tools=2,
            tool_declares_no_artifact_prereqs=lambda tool_name, contract: False,
            validate_tool_contract_call=lambda **kwargs: _validate_tool_contract_call(
                **kwargs,
                event_code_missing_prerequisite=EVENT_CODE_TOOL_VALIDATION_MISSING_PREREQUISITE,
                event_code_missing_capability=EVENT_CODE_TOOL_VALIDATION_MISSING_CAPABILITY,
                event_code_binding_conflict=EVENT_CODE_TOOL_VALIDATION_BINDING_CONFLICT,
            ),
            find_repair_tools_for_missing_requirements=lambda **kwargs: ["entity_resolve_names_to_ids"],
        )

        assert [tool["function"]["name"] for tool in visible] == ["entity_resolve_names_to_ids"]
        assert hidden_total == 1
        assert hidden[0]["tool"] == "relationship_onehop"
        assert hidden[0]["repair_tools"] == ["entity_resolve_names_to_ids"]
        msg = _disclosure_message(
            hidden,
            disclosure_reason_from_entry=lambda entry: _disclosure_reason_from_entry(
                entry,
                capability_requirement_from_raw=_capability_requirement_from_raw,
                short_requirement=_short_requirement,
            ),
            trim_text=lambda text, _max_chars: text,
            max_reason_chars=200,
        )
        assert "relationship_onehop" in msg
