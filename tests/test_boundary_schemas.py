"""Tests that LLMCallResult boundary schemas are registered correctly.

The 6 @boundary-decorated entry points in client.py return LLMCallResult
(a dataclass), so the decorator cannot auto-extract a Pydantic schema.
We inject a schema from LLMCallResultSchema post-decoration. These tests
verify the schema is present and structurally correct.
"""

from __future__ import annotations

import pytest

from data_contracts import registry
from llm_client.core.client import (
    acall_llm,
    acall_llm_structured,
    acall_llm_with_tools,
    call_llm,
    call_llm_structured,
    call_llm_with_tools,
)
from llm_client.core.data_types import LLMCallResult
from llm_client.schemas import LLMCallResultSchema


BOUNDARY_FUNCTIONS = [
    call_llm,
    call_llm_structured,
    call_llm_with_tools,
    acall_llm,
    acall_llm_structured,
    acall_llm_with_tools,
]


@pytest.mark.parametrize("fn", BOUNDARY_FUNCTIONS, ids=lambda fn: fn.__name__)
def test_boundary_has_output_schema(fn):
    """Each boundary-decorated function has a non-None output schema."""
    info = fn._boundary_info
    assert info.output_schema is not None, f"{info.name} missing output_schema"


@pytest.mark.parametrize("fn", BOUNDARY_FUNCTIONS, ids=lambda fn: fn.__name__)
def test_output_schema_matches_pydantic_model(fn):
    """The injected schema matches LLMCallResultSchema.model_json_schema()."""
    expected = LLMCallResultSchema.model_json_schema()
    actual = fn._boundary_info.output_schema
    assert actual == expected


def test_schema_covers_all_dataclass_fields():
    """LLMCallResultSchema has a property for every LLMCallResult field.

    Excludes raw_response which is intentionally omitted (Any type,
    provider-specific, not serializable).
    """
    import dataclasses

    dc_fields = {f.name for f in dataclasses.fields(LLMCallResult)}
    schema_props = set(LLMCallResultSchema.model_json_schema()["properties"].keys())

    # raw_response is excluded from schema (Any type, not serializable)
    dc_fields.discard("raw_response")

    missing_from_schema = dc_fields - schema_props
    assert not missing_from_schema, (
        f"Dataclass fields missing from schema: {missing_from_schema}"
    )


def test_registry_has_output_schemas():
    """The global contract registry has output schemas for all llm_client boundaries."""
    llm_boundaries = registry.list_by_producer("llm_client")
    assert len(llm_boundaries) >= 6

    for info in llm_boundaries:
        assert info.output_schema is not None, f"{info.name} missing output_schema in registry"
        assert "properties" in info.output_schema, f"{info.name} schema has no properties"
