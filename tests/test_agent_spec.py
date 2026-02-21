import json

import pytest

from llm_client.agent_spec import (
    AgentSpecValidationError,
    load_agent_spec,
    validate_agent_spec,
)


def _valid_spec() -> dict:
    return {
        "prompts": {"benchmark": {"path": "prompts/agent.yaml"}},
        "tools": [{"name": "chunk_text_search"}, {"name": "submit_answer"}],
        "artifact_contracts": {
            "chunk_text_search": {"requires_all": ["QUERY_TEXT"], "produces": ["CHUNK_SET"]},
            "submit_answer": {"is_control": True},
        },
        "answer_schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
        "error_taxonomy": {"canonical": ["tool_unavailable", "tool_runtime_error"]},
        "observability": {"required_fields": ["trace_id", "tool_calls"]},
        "evaluation": {"metrics": ["em", "f1", "llm_em"]},
        "gates": {"fail_if": {"tool_unavailable_gt": 0}},
    }


def test_validate_agent_spec_accepts_valid_spec():
    validated = validate_agent_spec(_valid_spec())
    assert validated["evaluation"]["metrics"] == ["em", "f1", "llm_em"]


def test_validate_agent_spec_rejects_missing_sections():
    bad = _valid_spec()
    bad.pop("gates")
    with pytest.raises(AgentSpecValidationError, match="missing required sections"):
        validate_agent_spec(bad)


def test_validate_agent_spec_rejects_unknown_contract_tools():
    bad = _valid_spec()
    bad["artifact_contracts"]["not_a_tool"] = {"requires_all": ["QUERY_TEXT"], "produces": ["CHUNK_SET"]}
    with pytest.raises(AgentSpecValidationError, match="unknown tools"):
        validate_agent_spec(bad)


def test_load_agent_spec_from_file(tmp_path):
    path = tmp_path / "agent_spec.json"
    path.write_text(json.dumps(_valid_spec()))
    loaded, summary = load_agent_spec(path)
    assert loaded["answer_schema"]["required"] == ["answer"]
    assert summary["tool_count"] == 2
    assert summary["contract_count"] == 2
