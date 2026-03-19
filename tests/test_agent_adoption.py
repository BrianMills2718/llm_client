from __future__ import annotations

from llm_client.agent_adoption import (
    DEFAULT_ADOPTION_PROFILE,
    assess_adoption_profile,
    assess_tool_registry_quality,
    normalize_adoption_profile,
    tool_schema_is_nontrivial,
)


TOOL_REASONING_FIELD = "tool_reasoning"


def test_normalize_adoption_profile_defaults_for_unknown_values() -> None:
    assert normalize_adoption_profile("STRICT") == "strict"
    assert normalize_adoption_profile("unknown") == DEFAULT_ADOPTION_PROFILE
    assert normalize_adoption_profile(None) == DEFAULT_ADOPTION_PROFILE


def test_tool_schema_is_nontrivial_ignores_tool_reasoning_only() -> None:
    simple_tool = {
        "function": {
            "parameters": {
                "properties": {
                    TOOL_REASONING_FIELD: {"type": "string"},
                    "query": {"type": "string"},
                }
            }
        }
    }
    complex_tool = {
        "function": {
            "parameters": {
                "properties": {
                    TOOL_REASONING_FIELD: {"type": "string"},
                    "items": {"type": "array", "items": {"type": "string"}},
                }
            }
        }
    }
    assert tool_schema_is_nontrivial(simple_tool, tool_reasoning_field=TOOL_REASONING_FIELD) is False
    assert tool_schema_is_nontrivial(complex_tool, tool_reasoning_field=TOOL_REASONING_FIELD) is True


def test_assess_tool_registry_quality_flags_missing_examples_and_contracts() -> None:
    tool = {
        "function": {
            "name": "search_entities",
            "description": "Search entities.",
            "parameters": {
                "properties": {
                    TOOL_REASONING_FIELD: {"type": "string"},
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                }
            },
        }
    }
    violations = assess_tool_registry_quality(
        openai_tools=[tool],
        normalized_tool_contracts={},
        tool_reasoning_field=TOOL_REASONING_FIELD,
    )
    assert "nontrivial tool search_entities must include input examples" in violations
    assert "nontrivial tool search_entities must declare tool_contracts" in violations


def test_assess_adoption_profile_strict_captures_runtime_and_tooling_violations() -> None:
    tool = {
        "function": {
            "name": "search_entities",
            "description": "Search entities.\n\nInput examples:\n- {\"query\": \"alan turing\"}",
            "parameters": {
                "properties": {
                    TOOL_REASONING_FIELD: {"type": "string"},
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                }
            },
        }
    }
    assessment = assess_adoption_profile(
        requested_profile="strict",
        enforce=False,
        openai_tools=[tool],
        normalized_tool_contracts={"search_entities": {"produces": ["ENTITY_SET"]}},
        require_tool_reasoning=False,
        enforce_tool_contracts=True,
        progressive_tool_disclosure=False,
        lane_closure_analysis={"lane_closed": False},
        tool_reasoning_field=TOOL_REASONING_FIELD,
    )
    assert assessment.effective_profile == "strict"
    assert assessment.satisfied is False
    assert "require_tool_reasoning must be enabled" in assessment.violations
    assert "progressive_tool_disclosure must be enabled" in assessment.violations
    assert "lane_closure_analysis must be closed for strict profile" in assessment.violations
