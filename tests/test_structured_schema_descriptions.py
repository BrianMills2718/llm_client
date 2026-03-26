"""Tests verifying structured output schema descriptions propagate to response_format.

Ported from llm_client v2
(~/projects/archive/llm_client_v2/tests/test_structured_schema_descriptions.py).

These tests ensure that Pydantic ``Field(description=...)`` values flow through
``model_json_schema()`` -> ``_strict_json_schema()`` -> the ``response_format``
dict sent to litellm. This matters because schema field descriptions constrain
LLM behavior at decode time -- they are a prompting surface, not just
documentation.

Documents the 18.8% -> 69.5% onto-canon predicate accuracy gain from field
descriptions.

# mock-ok: litellm.completion and litellm.supports_response_schema mocked
# because these are unit tests verifying the schema-generation chain, not
# provider integration.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from llm_client.core.client import _strict_json_schema


# ---------------------------------------------------------------------------
# Test models -- field descriptions are the key assertion surface
# ---------------------------------------------------------------------------


class RelationshipExtraction(BaseModel):
    """A single extracted relationship between two entities."""

    subject: str = Field(description="The source entity name, exactly as it appears in text")
    verb: str = Field(description="Bare verb infinitive describing the relationship (e.g. deploy, invest)")
    object: str = Field(description="The target entity name, exactly as it appears in text")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")


class PartialResult(BaseModel):
    """Model with optional fields for permissive-parsing tests."""

    name: str = Field(description="Entity name")
    category: Optional[str] = Field(default=None, description="Optional category label")
    score: float = Field(default=0.0, description="Relevance score, defaults to 0.0")
    tags: list[str] = Field(default_factory=list, description="Optional tags list")


class MetadataBlock(BaseModel):
    """Metadata for a nested schema test."""

    source: str = Field(description="Source document identifier")
    version: int = Field(default=1, description="Schema version number")


class ItemEntry(BaseModel):
    """Single item in a nested schema test."""

    label: str = Field(description="Item label")
    value: float = Field(description="Numeric value")


class NestedSchema(BaseModel):
    """Model with nested objects to test recursive schema transforms."""

    metadata: MetadataBlock = Field(description="Metadata about the extraction")
    items: list[ItemEntry] = Field(description="Extracted items")


# ---------------------------------------------------------------------------
# Test: Field descriptions propagate to response_format
# ---------------------------------------------------------------------------


class TestFieldDescriptionsPropagate:
    """Verify that Field(description=...) values appear in the json_schema response_format."""

    def test_field_descriptions_in_model_json_schema(self) -> None:
        """model_json_schema() includes Field descriptions -- the foundation of propagation."""
        schema = RelationshipExtraction.model_json_schema()
        props = schema["properties"]

        assert props["subject"]["description"] == (
            "The source entity name, exactly as it appears in text"
        )
        assert props["verb"]["description"] == (
            "Bare verb infinitive describing the relationship (e.g. deploy, invest)"
        )
        assert props["object"]["description"] == (
            "The target entity name, exactly as it appears in text"
        )
        assert props["confidence"]["description"] == "Confidence score between 0.0 and 1.0"

    def test_field_descriptions_survive_strict_json_schema(self) -> None:
        """_strict_json_schema preserves field descriptions while adding strict-mode fields."""
        raw_schema = RelationshipExtraction.model_json_schema()
        strict_schema = _strict_json_schema(raw_schema)

        props = strict_schema["properties"]
        assert props["subject"]["description"] == (
            "The source entity name, exactly as it appears in text"
        )
        assert props["verb"]["description"] == (
            "Bare verb infinitive describing the relationship (e.g. deploy, invest)"
        )
        # Also verify strict-mode additions didn't clobber descriptions
        assert strict_schema["additionalProperties"] is False
        assert set(strict_schema["required"]) == {"subject", "verb", "object", "confidence"}

    @patch("litellm.supports_response_schema", return_value=True)
    @patch("litellm.completion")
    @patch("litellm.completion_cost", return_value=0.001)
    def test_field_descriptions_reach_litellm_completion(
        self,
        mock_cost: MagicMock,
        mock_completion: MagicMock,
        mock_supports: MagicMock,
    ) -> None:
        """call_llm_structured sends field descriptions to litellm.completion via response_format.

        This is the end-to-end proof: define a model with descriptive fields,
        call the public API, and verify the response_format dict passed to
        litellm.completion contains those descriptions.
        """
        from llm_client.core.client import call_llm_structured

        # Build a realistic mock response with valid JSON for our model
        mock_response = _make_structured_mock_response(
            RelationshipExtraction(
                subject="Shield AI",
                verb="deploy",
                object="autonomous drones",
                confidence=0.92,
            )
        )
        mock_completion.return_value = mock_response

        parsed, result = call_llm_structured(
            "gpt-4o",
            [{"role": "user", "content": "Extract relationships"}],
            RelationshipExtraction,
            task="test",
            trace_id="desc-propagation",
            max_budget=1.0,
        )

        # Verify litellm.completion was called with response_format containing descriptions
        assert mock_completion.called
        call_kwargs = mock_completion.call_args
        response_format = call_kwargs.kwargs.get("response_format") or call_kwargs[1].get("response_format")
        assert response_format is not None, "response_format not passed to litellm.completion"
        assert response_format["type"] == "json_schema"

        schema = response_format["json_schema"]["schema"]
        props = schema["properties"]
        assert "description" in props["subject"], "subject field missing description"
        assert props["subject"]["description"] == (
            "The source entity name, exactly as it appears in text"
        )
        assert props["verb"]["description"] == (
            "Bare verb infinitive describing the relationship (e.g. deploy, invest)"
        )

        # Verify the parsed result is correct
        assert isinstance(parsed, RelationshipExtraction)
        assert parsed.subject == "Shield AI"
        assert parsed.verb == "deploy"


# ---------------------------------------------------------------------------
# Test: Permissive parsing with optional/default fields
# ---------------------------------------------------------------------------


class TestPermissiveParsing:
    """Verify structured calls handle partial output when optional fields are missing."""

    def test_missing_optional_fields_parse_successfully(self) -> None:
        """model_validate_json succeeds when optional/defaulted fields are absent."""
        # Simulate LLM returning only the required field
        partial_json = json.dumps({"name": "Shield AI"})
        result = PartialResult.model_validate_json(partial_json)

        assert result.name == "Shield AI"
        assert result.category is None
        assert result.score == 0.0
        assert result.tags == []

    def test_extra_fields_ignored_by_default(self) -> None:
        """Pydantic ignores extra fields the LLM may hallucinate (default model_config)."""
        json_with_extras = json.dumps({
            "name": "Shield AI",
            "category": "defense",
            "score": 0.8,
            "tags": ["autonomy"],
            "hallucinated_field": "should be ignored",
        })
        result = PartialResult.model_validate_json(json_with_extras)
        assert result.name == "Shield AI"
        assert result.category == "defense"
        assert not hasattr(result, "hallucinated_field")

    @patch("litellm.supports_response_schema", return_value=True)
    @patch("litellm.completion")
    @patch("litellm.completion_cost", return_value=0.001)
    def test_partial_response_via_call_llm_structured(
        self,
        mock_cost: MagicMock,
        mock_completion: MagicMock,
        mock_supports: MagicMock,
    ) -> None:
        """call_llm_structured returns a valid result when optional fields are missing."""
        from llm_client.core.client import call_llm_structured

        # LLM returns only the required field
        partial_json = json.dumps({"name": "Anduril"})
        mock_response = _make_raw_mock_response(content=partial_json)
        mock_completion.return_value = mock_response

        parsed, result = call_llm_structured(
            "gpt-4o",
            [{"role": "user", "content": "Extract entity"}],
            PartialResult,
            task="test",
            trace_id="permissive-parse",
            max_budget=1.0,
        )

        assert isinstance(parsed, PartialResult)
        assert parsed.name == "Anduril"
        assert parsed.category is None
        assert parsed.score == 0.0


# ---------------------------------------------------------------------------
# Test: Schema name derived from model class name
# ---------------------------------------------------------------------------


class TestSchemaNameFromModel:
    """Verify the json_schema name field comes from the Pydantic model class name."""

    @patch("litellm.supports_response_schema", return_value=True)
    @patch("litellm.completion")
    @patch("litellm.completion_cost", return_value=0.001)
    def test_schema_name_matches_class_name(
        self,
        mock_cost: MagicMock,
        mock_completion: MagicMock,
        mock_supports: MagicMock,
    ) -> None:
        """response_format json_schema name equals the Pydantic model __name__."""
        from llm_client.core.client import call_llm_structured

        mock_response = _make_structured_mock_response(
            RelationshipExtraction(
                subject="A", verb="fund", object="B", confidence=0.5,
            )
        )
        mock_completion.return_value = mock_response

        call_llm_structured(
            "gpt-4o",
            [{"role": "user", "content": "Extract"}],
            RelationshipExtraction,
            task="test",
            trace_id="schema-name",
            max_budget=1.0,
        )

        call_kwargs = mock_completion.call_args
        response_format = call_kwargs.kwargs.get("response_format") or call_kwargs[1].get("response_format")
        assert response_format["json_schema"]["name"] == "RelationshipExtraction"

    @patch("litellm.supports_response_schema", return_value=True)
    @patch("litellm.completion")
    @patch("litellm.completion_cost", return_value=0.001)
    def test_schema_name_for_different_model(
        self,
        mock_cost: MagicMock,
        mock_completion: MagicMock,
        mock_supports: MagicMock,
    ) -> None:
        """Schema name changes when a different Pydantic model is used."""
        from llm_client.core.client import call_llm_structured

        mock_response = _make_raw_mock_response(
            content=json.dumps({"name": "test", "category": None, "score": 0.0, "tags": []})
        )
        mock_completion.return_value = mock_response

        call_llm_structured(
            "gpt-4o",
            [{"role": "user", "content": "Extract"}],
            PartialResult,
            task="test",
            trace_id="schema-name-2",
            max_budget=1.0,
        )

        call_kwargs = mock_completion.call_args
        response_format = call_kwargs.kwargs.get("response_format") or call_kwargs[1].get("response_format")
        assert response_format["json_schema"]["name"] == "PartialResult"


# ---------------------------------------------------------------------------
# Test: _strict_json_schema correctness
# ---------------------------------------------------------------------------


class TestStrictJsonSchemaTransform:
    """Verify _strict_json_schema adds additionalProperties: false correctly."""

    def test_top_level_additional_properties_false(self) -> None:
        """Top-level object gets additionalProperties: false."""
        schema = RelationshipExtraction.model_json_schema()
        strict = _strict_json_schema(schema)
        assert strict["additionalProperties"] is False

    def test_all_properties_required(self) -> None:
        """All properties become required in strict mode."""
        schema = RelationshipExtraction.model_json_schema()
        strict = _strict_json_schema(schema)
        assert set(strict["required"]) == {"subject", "verb", "object", "confidence"}

    def test_nested_objects_get_additional_properties_false(self) -> None:
        """Nested object types also get additionalProperties: false."""
        schema = NestedSchema.model_json_schema()
        strict = _strict_json_schema(schema)

        # Pydantic v2 may use $defs for nested models -- check both inline and $defs
        defs = strict.get("$defs", {})
        if "MetadataBlock" in defs:
            assert defs["MetadataBlock"]["additionalProperties"] is False
            assert "source" in defs["MetadataBlock"].get("required", [])
        if "ItemEntry" in defs:
            assert defs["ItemEntry"]["additionalProperties"] is False
            assert "label" in defs["ItemEntry"].get("required", [])

    def test_descriptions_preserved_in_nested_strict_schema(self) -> None:
        """Field descriptions survive _strict_json_schema even in nested models."""
        schema = NestedSchema.model_json_schema()
        strict = _strict_json_schema(schema)

        # Check top-level descriptions
        assert strict["properties"]["metadata"]["description"] == "Metadata about the extraction"
        assert strict["properties"]["items"]["description"] == "Extracted items"

        # Check nested model descriptions in $defs
        defs = strict.get("$defs", {})
        if "MetadataBlock" in defs:
            assert defs["MetadataBlock"]["properties"]["source"]["description"] == (
                "Source document identifier"
            )

    def test_mutates_input_in_place(self) -> None:
        """v1 _strict_json_schema mutates the input dict in place (unlike v2 which copies).

        Callers should deepcopy before calling if they need the original preserved.
        """
        original = RelationshipExtraction.model_json_schema()
        frozen = copy.deepcopy(original)
        result = _strict_json_schema(original)
        # v1 mutates in place -- result IS the original
        assert result is original
        # The original should now differ from the frozen copy (strict fields added)
        assert original != frozen
        assert original["additionalProperties"] is False

    def test_anyof_branches_get_strict(self) -> None:
        """Optional fields using anyOf get additionalProperties: false on object branches."""
        schema = {
            "type": "object",
            "properties": {
                "maybe_obj": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {"inner": {"type": "string"}},
                        },
                        {"type": "null"},
                    ]
                }
            },
        }
        strict = _strict_json_schema(schema)
        obj_branch = strict["properties"]["maybe_obj"]["anyOf"][0]
        assert obj_branch["additionalProperties"] is False
        assert obj_branch["required"] == ["inner"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_structured_mock_response(
    parsed_model: BaseModel,
    model: str = "gpt-4o",
) -> MagicMock:
    """Build a mock litellm response containing the JSON of a parsed Pydantic model."""
    return _make_raw_mock_response(
        content=parsed_model.model_dump_json(),
        model=model,
    )


def _make_raw_mock_response(
    content: str,
    model: str = "gpt-4o",
    finish_reason: str = "stop",
) -> MagicMock:
    """Build a mock litellm response with arbitrary string content."""
    response = MagicMock()
    response.model = model

    message = MagicMock()
    message.content = content
    message.tool_calls = None

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    response.choices = [choice]
    response.usage = MagicMock()
    response.usage.prompt_tokens = 20
    response.usage.completion_tokens = 15
    response.usage.total_tokens = 35
    response.usage.prompt_tokens_details = None
    response._hidden_params = {}

    return response
