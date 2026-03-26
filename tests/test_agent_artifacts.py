"""Tests for artifact/evidence/context helpers extracted from mcp_agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from llm_client.agent.agent_artifacts import (
    _build_active_artifact_context_content,
    _runtime_artifact_read_result,
    _tool_evidence_pointer_labels,
)


@dataclass
class _FakeToolCallRecord:
    server: str
    tool: str
    arguments: dict[str, Any]
    tool_call_id: str | None = None
    tool_reasoning: str | None = None
    result: str | None = None
    error: str | None = None


def _extract_args(tc: dict[str, Any]) -> dict[str, Any] | None:
    raw = tc.get("function", {}).get("arguments")
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _truncate(value: str, max_length: int) -> str:
    return value if len(value) <= max_length else value[:max_length]


class TestRuntimeArtifactRead:
    def test_reopens_typed_artifacts_and_can_strip_payload(self) -> None:
        record, tool_msg = _runtime_artifact_read_result(
            artifact_registry_by_id={
                "art_chunk_1": {
                    "artifact_id": "art_chunk_1",
                    "artifact_type": "CHUNK_SET",
                    "payload": {"chunk_id": "chunk_1", "text": "Reached on August 3, 1769."},
                    "provenance": {"evidence_refs": [{"chunk_id": "chunk_1"}]},
                }
            },
            tc={
                "id": "tc_runtime_read",
                "function": {
                    "name": "runtime_artifact_read",
                    "arguments": json.dumps(
                        {
                            "artifact_ids": ["art_chunk_1"],
                            "include_payload": False,
                            "tool_reasoning": "reopen artifact",
                        }
                    ),
                },
            },
            max_result_length=2000,
            require_tool_reasoning=True,
            tool_name="runtime_artifact_read",
            tool_reasoning_field="tool_reasoning",
            record_factory=_FakeToolCallRecord,
            extract_tool_call_args=_extract_args,
            truncate_text=_truncate,
        )

        assert record.error is None
        assert record.tool_reasoning == "reopen artifact"
        payload = json.loads(str(tool_msg["content"]))
        assert payload["artifact_ids"] == ["art_chunk_1"]
        assert "payload" not in payload["artifacts"][0]
        assert payload["artifacts"][0]["provenance"]["evidence_refs"][0]["chunk_id"] == "chunk_1"


class TestEvidencePointers:
    def test_extracts_typed_evidence_refs_and_removes_redundant_chunk_label(self) -> None:
        record = _FakeToolCallRecord(
            server="srv",
            tool="chunk_text_search",
            arguments={"query": "q"},
            result=json.dumps(
                {
                    "artifact_id": "art_chunk_1",
                    "artifact_type": "CHUNK_SET",
                    "provenance": {
                        "evidence_refs": [
                            {"chunk_id": "chunk_42", "char_start": 3, "char_end": 9}
                        ]
                    },
                    "payload": {
                        "chunk_id": "chunk_42",
                        "text": "Reached on August 3, 1769.",
                    },
                }
            ),
        )

        labels = _tool_evidence_pointer_labels(
            record,
            budget_exempt_tool_names=frozenset({"submit_answer", "todo_write", "runtime_artifact_read"}),
        )

        assert "chunk:chunk_42#char:3-9" in labels
        assert "chunk:chunk_42" not in labels


class TestActiveArtifactContext:
    def test_builds_summary_from_handles_and_capabilities(self) -> None:
        content = _build_active_artifact_context_content(
            available_artifacts={"QUERY_TEXT", "CHUNK_SET", "ENTITY_SET"},
            available_capabilities=[
                {"kind": "CHUNK_SET", "ref_type": "fulltext"},
                {"kind": "ENTITY_SET", "ref_type": "id", "namespace": "wiki"},
            ],
            tool_result_metadata_by_id={
                "tc1": {
                    "artifact_handles": [
                        {
                            "artifact_id": "art_chunk_1",
                            "artifact_type": "CHUNK_SET",
                            "kind": "CHUNK_SET",
                            "ref_type": "fulltext",
                        },
                        {
                            "artifact_id": "art_entity_1",
                            "artifact_type": "ENTITY_SET",
                            "kind": "ENTITY_SET",
                            "ref_type": "id",
                            "namespace": "wiki",
                        },
                    ]
                }
            },
            max_handles=4,
            max_chars=400,
            runtime_artifact_read_tool_name="runtime_artifact_read",
        )

        assert content is not None
        assert "Active artifact context" in content
        assert "art_chunk_1 CHUNK_SET ref_type=fulltext" in content
        assert "ENTITY_SET[ref_type=id, namespace=wiki]" in content
        assert "use runtime_artifact_read" in content
        assert len(content) <= 400
