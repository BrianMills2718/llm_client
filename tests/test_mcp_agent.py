"""Tests for MCP agent loop. All mocked (no real MCP servers or LLM calls).

Tests cover:
- _mcp_tool_to_openai() schema conversion
- _extract_usage() with OpenAI and Anthropic conventions
- _truncate() behavior
- _acall_with_mcp() full agent loop (start, discover, loop, cleanup)
- Routing: mcp_servers on non-agent model goes through MCP loop
- Routing: mcp_servers on agent model goes through existing agent path
- max_turns exhaustion
- Error handling: no tools, unknown tool, MCP tool error
- MCP_LOOP_KWARGS are popped before inner acall_llm
"""

# mock-ok: MCP servers require subprocess lifecycle; unit tests must mock

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_client import (
    DEFAULT_MAX_TOOL_CALLS,
    DEFAULT_MAX_TURNS,
    DEFAULT_MCP_INIT_TIMEOUT,
    DEFAULT_TOOL_RESULT_MAX_LENGTH,
    LLMCallResult,
    MCPAgentResult,
    MCPToolCallRecord,
    acall_llm,
    call_llm,
)
from llm_client.agent.mcp_agent import (
    MCP_LOOP_KWARGS,
    _build_active_artifact_context_content,
    _clear_old_tool_results_for_context,
    _effective_contract_requirements,
    _extract_usage,
    _find_repair_tools_for_missing_requirements,
    _mcp_tool_to_openai,
    _normalize_tool_contracts,
    _tool_evidence_pointer_labels,
    _truncate,
)


# ---------------------------------------------------------------------------
# Schema conversion
# ---------------------------------------------------------------------------


class TestMcpToolToOpenai:
    def test_basic_conversion(self) -> None:
        tool = MagicMock()
        tool.name = "search"
        tool.description = "Search for entities"
        tool.inputSchema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        result = _mcp_tool_to_openai(tool)
        assert result["type"] == "function"
        assert result["function"]["name"] == "search"
        assert result["function"]["description"] == "Search for entities"
        required = result["function"]["parameters"]["required"]
        assert "query" in required
        assert "tool_reasoning" in required
        props = result["function"]["parameters"]["properties"]
        assert props["query"] == {"type": "string"}
        assert "tool_reasoning" in props

    def test_missing_description(self) -> None:
        tool = MagicMock()
        tool.name = "foo"
        tool.description = None
        tool.inputSchema = {"type": "object", "properties": {}}
        result = _mcp_tool_to_openai(tool)
        assert result["function"]["description"] == ""

    def test_missing_schema(self) -> None:
        tool = MagicMock()
        tool.name = "bar"
        tool.description = "desc"
        tool.inputSchema = None
        result = _mcp_tool_to_openai(tool)
        params = result["function"]["parameters"]
        assert params["type"] == "object"
        assert "tool_reasoning" in params["properties"]

    def test_appends_input_examples_to_description(self) -> None:
        tool = MagicMock()
        tool.name = "search"
        tool.description = "Search for entities"
        tool.inputSchema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "examples": [{"query": "Messi Copa del Rey goals comparison"}],
        }
        result = _mcp_tool_to_openai(tool)
        desc = result["function"]["description"]
        assert "Search for entities" in desc
        assert "Input examples:" in desc
        assert "Messi Copa del Rey goals comparison" in desc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestExtractUsage:
    def test_openai_convention(self) -> None:
        inp, out, cached, cache_create = _extract_usage({"prompt_tokens": 100, "completion_tokens": 50})
        assert inp == 100
        assert out == 50
        assert cached == 0
        assert cache_create == 0

    def test_anthropic_convention(self) -> None:
        inp, out, cached, cache_create = _extract_usage({"input_tokens": 200, "output_tokens": 75})
        assert inp == 200
        assert out == 75

    def test_empty_usage(self) -> None:
        inp, out, cached, cache_create = _extract_usage({})
        assert inp == 0
        assert out == 0
        assert cached == 0
        assert cache_create == 0

    def test_input_tokens_takes_priority(self) -> None:
        """input_tokens is checked before prompt_tokens."""
        inp, out, cached, cache_create = _extract_usage({
            "input_tokens": 300,
            "prompt_tokens": 100,
            "output_tokens": 50,
        })
        assert inp == 300

    def test_cached_tokens(self) -> None:
        """Provider-level prompt caching fields are extracted."""
        inp, out, cached, cache_create = _extract_usage({
            "input_tokens": 500,
            "output_tokens": 100,
            "cached_tokens": 400,
            "cache_creation_tokens": 50,
        })
        assert inp == 500
        assert out == 100
        assert cached == 400
        assert cache_create == 50


class TestClearOldToolResultsForContext:
    def test_clears_older_tool_messages_and_keeps_recent(self) -> None:
        messages = [
            {"role": "user", "content": "q"},
            {"role": "tool", "tool_call_id": "tc1", "content": "alpha " * 60},
            {"role": "tool", "tool_call_id": "tc2", "content": "beta " * 60},
            {"role": "tool", "tool_call_id": "tc3", "content": "gamma " * 60},
        ]
        cleared, saved = _clear_old_tool_results_for_context(
            messages,
            keep_recent=1,
            preview_chars=80,
        )
        assert cleared == 2
        assert saved > 0
        # Most recent tool result stays verbatim.
        assert messages[3]["content"].startswith("gamma ")
        # Older results are replaced with stubs that keep trace handles.
        assert "Tool result cleared from active context" in messages[1]["content"]
        assert '"tool_call_id": "tc1"' in messages[1]["content"]
        assert '"content_sha256":' in messages[1]["content"]

    def test_keep_recent_zero_clears_all_tool_payloads(self) -> None:
        messages = [
            {"role": "tool", "tool_call_id": "tc1", "content": "x" * 200},
            {"role": "tool", "tool_call_id": "tc2", "content": "y" * 200},
        ]
        cleared, _ = _clear_old_tool_results_for_context(
            messages,
            keep_recent=0,
            preview_chars=60,
        )
        assert cleared == 2
        assert all("Tool result cleared from active context" in m["content"] for m in messages)

    def test_stub_preserves_artifact_handle_metadata_when_available(self) -> None:
        messages = [
            {"role": "tool", "tool_call_id": "tc1", "content": "x" * 200},
            {"role": "tool", "tool_call_id": "tc2", "content": "y" * 200},
        ]
        cleared, _ = _clear_old_tool_results_for_context(
            messages,
            keep_recent=1,
            preview_chars=60,
            tool_result_metadata_by_id={
                "tc1": {
                    "artifact_ids": ["art_123"],
                    "artifact_handles": [{"artifact_id": "art_123", "artifact_type": "CHUNK_SET"}],
                }
            },
        )
        assert cleared == 1
        assert '"artifact_ids": ["art_123"]' in messages[0]["content"]
        assert '"artifact_type": "CHUNK_SET"' in messages[0]["content"]


class TestActiveArtifactContext:
    def test_builds_bounded_summary_from_recent_handles_and_capabilities(self) -> None:
        content = _build_active_artifact_context_content(
            available_artifacts={"QUERY_TEXT", "CHUNK_SET", "ENTITY_SET"},
            available_capabilities={
                "CHUNK_SET": {("fulltext", None, None)},
                "ENTITY_SET": {("id", "wiki", None)},
            },
            tool_result_metadata_by_id={
                "tc1": {
                    "artifact_handles": [
                        {"artifact_id": "art_chunk_1", "artifact_type": "CHUNK_SET", "kind": "CHUNK_SET", "ref_type": "fulltext"},
                        {"artifact_id": "art_entity_1", "artifact_type": "ENTITY_SET", "kind": "ENTITY_SET", "ref_type": "id", "namespace": "wiki"},
                    ]
                }
            },
            max_handles=4,
            max_chars=400,
        )
        assert content is not None
        assert "Active artifact context" in content
        assert "art_chunk_1 CHUNK_SET ref_type=fulltext" in content
        assert "ENTITY_SET[ref_type=id, namespace=wiki]" in content
        assert len(content) <= 400


class TestEvidencePointerExtraction:
    def test_extracts_typed_evidence_refs_from_artifact_envelope(self) -> None:
        record = MCPToolCallRecord(
            server="srv",
            tool="chunk_text_search",
            arguments={"tool_reasoning": "find evidence"},
            result=json.dumps(
                {
                    "artifact_id": "art_chunk_1",
                    "artifact_type": "CHUNK_SET",
                    "schema_version": "1.0.0",
                    "provenance": {
                        "evidence_refs": [
                            {
                                "chunk_id": "chunk_42",
                                "char_start": 3,
                                "char_end": 9,
                            }
                        ]
                    },
                    "payload": {"items": [{"chunk_id": "chunk_42"}]},
                }
            ),
        )

        labels = _tool_evidence_pointer_labels(record)
        assert labels == {"chunk:chunk_42#char:3-9"}


class TestDynamicContractRequirements:
    def test_artifact_prereqs_none_overrides_artifact_requirements(self) -> None:
        requires_all, requires_any = _effective_contract_requirements(
            "chunk_get_text_by_chunk_ids",
            {"artifact_prereqs_none": True, "requires_all": ["CHUNK_SET"]},
            {},
        )
        assert requires_all == set()
        assert requires_any == set()

    def test_chunk_get_text_explicit_chunk_id_is_self_contained(self) -> None:
        requires_all, requires_any = _effective_contract_requirements(
            "chunk_get_text",
            {
                "artifact_prereqs_none": True,
                "requires_all": ["CHUNK_SET"],
                "requires_any": ["ENTITY_SET"],
            },
            {"chunk_id": "chunk_84"},
        )
        assert requires_all == set()
        assert requires_any == set()

    def test_chunk_get_text_explicit_entity_id_alias_is_self_contained(self) -> None:
        requires_all, requires_any = _effective_contract_requirements(
            "chunk_get_text",
            {"artifact_prereqs_none": True, "requires_all": ["ENTITY_SET"]},
            {"entity_id": "godiva"},
        )
        assert requires_all == set()
        assert requires_any == set()

    def test_chunk_get_text_without_explicit_refs_uses_contract(self) -> None:
        requires_all, requires_any = _effective_contract_requirements(
            "chunk_get_text",
            {"requires_all": ["CHUNK_SET"], "requires_any": ["ENTITY_SET"]},
            {},
        )
        assert requires_all == {"CHUNK_SET"}
        assert requires_any == {"ENTITY_SET"}

    def test_chunk_get_text_by_chunk_ids_explicit_ref_is_self_contained(self) -> None:
        requires_all, requires_any = _effective_contract_requirements(
            "chunk_get_text_by_chunk_ids",
            {"artifact_prereqs_none": True, "requires_all": ["CHUNK_SET"]},
            {"chunk_ids": ["chunk_84"]},
        )
        assert requires_all == set()
        assert requires_any == set()

    def test_chunk_get_text_by_entity_ids_explicit_ref_is_self_contained(self) -> None:
        requires_all, requires_any = _effective_contract_requirements(
            "chunk_get_text_by_entity_ids",
            {"artifact_prereqs_none": True, "requires_all": ["ENTITY_SET"]},
            {"entity_ids": ["godiva"]},
        )
        assert requires_all == set()
        assert requires_any == set()

    def test_explicit_refs_do_not_bypass_prereqs_without_declarative_metadata(self) -> None:
        requires_all, requires_any = _effective_contract_requirements(
            "chunk_get_text_by_chunk_ids",
            {"requires_all": ["CHUNK_SET"]},
            {"chunk_ids": ["chunk_84"]},
        )
        assert requires_all == {"CHUNK_SET"}
        assert requires_any == set()

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

    def test_call_modes_support_discriminated_union_style_mode_field(self) -> None:
        requires_all, requires_any = _effective_contract_requirements(
            "chunk_get_text",
            _normalize_tool_contracts(
                {
                    "chunk_get_text": {
                        "requires_all": ["CHUNK_SET"],
                        "call_modes": [
                            {
                                "name": "by_chunk_mode",
                                "when_arg_equals": {"mode": "by_chunk_id"},
                                "artifact_prereqs": "none",
                            }
                        ],
                    }
                }
            )["chunk_get_text"],
            {"mode": "by_chunk_id", "chunk_ids": ["chunk_84"]},
        )
        assert requires_all == set()
        assert requires_any == set()


class TestRepairToolSuggestions:
    def test_prefers_bootstrap_search_over_self_dependent_get_text(self) -> None:
        contracts = _normalize_tool_contracts({
            "extract_date_mentions": {
                "requires_all": [{"kind": "CHUNK_SET", "ref_type": "fulltext"}],
                "produces": [{"kind": "CHUNK_SET", "ref_type": "fulltext"}],
            },
            "chunk_text_search": {
                "requires_all": ["QUERY_TEXT"],
                "produces": [{"kind": "CHUNK_SET", "ref_type": "id"}],
            },
            "chunk_aggregator": {
                "requires_any": ["ENTITY_SET", "RELATIONSHIP_SET"],
                "produces": [{"kind": "CHUNK_SET", "ref_type": "id"}],
            },
            "chunk_get_text_by_chunk_ids": {
                "artifact_prereqs": "none",
                "requires_all": [{"kind": "CHUNK_SET", "ref_type": "id"}],
                "produces": [{"kind": "CHUNK_SET", "ref_type": "fulltext"}],
            },
        })

        suggestions = _find_repair_tools_for_missing_requirements(
            current_tool_name="extract_date_mentions",
            missing_requirements=[{"kind": "CHUNK_SET", "ref_type": "fulltext"}],
            normalized_tool_contracts=contracts,
            available_artifacts={"QUERY_TEXT", "ENTITY_SET"},
            available_capabilities={
                "QUERY_TEXT": {(None, None, None)},
                "ENTITY_SET": {("id", None, None)},
            },
            available_bindings={},
            max_repair_tools=2,
        )

        assert suggestions
        assert suggestions[0] == "chunk_text_search"
        assert "chunk_get_text_by_chunk_ids" not in suggestions


class TestTruncate:
    def test_no_truncation(self) -> None:
        assert _truncate("hello", 100) == "hello"

    def test_exact_limit(self) -> None:
        assert _truncate("hello", 5) == "hello"

    def test_truncation(self) -> None:
        result = _truncate("hello world", 5)
        assert result.startswith("hello")
        assert "truncated at 5 chars" in result


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_default_max_tool_calls(self) -> None:
        assert DEFAULT_MAX_TOOL_CALLS == 20

    def test_default_max_turns(self) -> None:
        assert DEFAULT_MAX_TURNS == 20

    def test_default_init_timeout(self) -> None:
        assert DEFAULT_MCP_INIT_TIMEOUT == 30.0

    def test_default_tool_result_max_length(self) -> None:
        assert DEFAULT_TOOL_RESULT_MAX_LENGTH == 50_000

    def test_mcp_loop_kwargs(self) -> None:
        assert "mcp_servers" in MCP_LOOP_KWARGS
        assert "max_turns" in MCP_LOOP_KWARGS
        assert "max_tool_calls" in MCP_LOOP_KWARGS
        assert "mcp_init_timeout" in MCP_LOOP_KWARGS
        assert "tool_result_max_length" in MCP_LOOP_KWARGS


class TestFailureTaxonomy:
    _EXPECTED_FAILURE_EVENT_CODES: set[str] = {
        "TOOL_VALIDATION_REJECTED_MISSING_PREREQUISITE",
        "TOOL_VALIDATION_REJECTED_MISSING_CAPABILITY",
        "TOOL_VALIDATION_REJECTED_BINDING_CONFLICT",
        "TOOL_VALIDATION_REJECTED_MISSING_TOOL_REASONING",
        "TOOL_VALIDATION_REJECTED_SCHEMA",
        "NO_LEGAL_NONCONTROL_TOOLS",
        "CONTROL_CHURN_SUPPRESSED",
        "CONTROL_CHURN_THRESHOLD_EXCEEDED",
        "TOOL_EXECUTION_RUNTIME_ERROR",
        "PROVIDER_EMPTY_CANDIDATES",
        "PROVIDER_CREDITS_EXHAUSTED",
        "FINALIZATION_CIRCUIT_BREAKER_OPEN",
        "FINALIZATION_TOOL_CALL_DISALLOWED",
        "REQUIRED_SUBMIT_NOT_ATTEMPTED",
        "REQUIRED_SUBMIT_NOT_ACCEPTED",
        "SUBMIT_FORCED_ACCEPT_BUDGET_EXHAUSTION",
        "SUBMIT_FORCED_ACCEPT_TURN_EXHAUSTION",
        "SUBMIT_FORCED_ACCEPT_FORCED_FINAL",
        "RETRIEVAL_NO_HITS",
        "RETRIEVAL_STAGNATION",
        "RETRIEVAL_STAGNATION_OBSERVED",
    }

    def test_runtime_event_codes_match_canonical_failure_set(self) -> None:
        import llm_client.agent.mcp_agent as mcp_agent

        runtime_codes = {
            value
            for name, value in vars(mcp_agent).items()
            if name.startswith("EVENT_CODE_") and isinstance(value, str)
        }
        runtime_codes.add("RETRIEVAL_NO_HITS")
        assert runtime_codes == self._EXPECTED_FAILURE_EVENT_CODES

    @pytest.mark.parametrize(
        ("event_code", "expected_class"),
        [
            ("TOOL_VALIDATION_REJECTED_MISSING_PREREQUISITE", "composability"),
            ("TOOL_VALIDATION_REJECTED_MISSING_CAPABILITY", "composability"),
            ("TOOL_VALIDATION_REJECTED_BINDING_CONFLICT", "composability"),
            ("TOOL_VALIDATION_REJECTED_MISSING_TOOL_REASONING", "composability"),
            ("TOOL_VALIDATION_REJECTED_SCHEMA", "composability"),
            ("NO_LEGAL_NONCONTROL_TOOLS", "composability"),
            ("CONTROL_CHURN_SUPPRESSED", "control_churn"),
            ("CONTROL_CHURN_THRESHOLD_EXCEEDED", "control_churn"),
            ("TOOL_EXECUTION_RUNTIME_ERROR", "none"),
            ("PROVIDER_EMPTY_CANDIDATES", "provider"),
            ("PROVIDER_CREDITS_EXHAUSTED", "provider"),
            ("FINALIZATION_CIRCUIT_BREAKER_OPEN", "provider"),
            ("FINALIZATION_TOOL_CALL_DISALLOWED", "policy"),
            ("REQUIRED_SUBMIT_NOT_ATTEMPTED", "policy"),
            ("REQUIRED_SUBMIT_NOT_ACCEPTED", "policy"),
            ("SUBMIT_FORCED_ACCEPT_BUDGET_EXHAUSTION", "none"),
            ("SUBMIT_FORCED_ACCEPT_TURN_EXHAUSTION", "none"),
            ("SUBMIT_FORCED_ACCEPT_FORCED_FINAL", "none"),
            ("RETRIEVAL_NO_HITS", "retrieval"),
            ("RETRIEVAL_STAGNATION", "retrieval"),
            ("RETRIEVAL_STAGNATION_OBSERVED", "retrieval"),
        ],
    )
    def test_event_code_primary_failure_class_mapping(
        self,
        event_code: str,
        expected_class: str,
    ) -> None:
        import llm_client.agent.mcp_agent as mcp_agent

        primary, _secondary = mcp_agent._classify_failure_signals(
            failure_event_codes=[event_code],
            retrieval_no_hits_count=0,
            control_loop_suppressed_calls=0,
            force_final_reason=None,
            submit_answer_succeeded=True,
        )
        assert primary == expected_class

    def test_first_terminal_failure_event_code_prefers_earliest_terminal_event(self) -> None:
        import llm_client.agent.mcp_agent as mcp_agent

        first_terminal = mcp_agent._first_terminal_failure_event_code(
            [
                "TOOL_VALIDATION_REJECTED_SCHEMA",
                "CONTROL_CHURN_THRESHOLD_EXCEEDED",
                "PROVIDER_EMPTY_CANDIDATES",
            ]
        )
        assert first_terminal == "CONTROL_CHURN_THRESHOLD_EXCEEDED"

    def test_submit_forced_accept_codes_are_informational(self) -> None:
        import llm_client.agent.mcp_agent as mcp_agent

        for event_code in (
            "SUBMIT_FORCED_ACCEPT_BUDGET_EXHAUSTION",
            "SUBMIT_FORCED_ACCEPT_TURN_EXHAUSTION",
            "SUBMIT_FORCED_ACCEPT_FORCED_FINAL",
        ):
            primary, secondary = mcp_agent._classify_failure_signals(
                failure_event_codes=[event_code],
                retrieval_no_hits_count=0,
                control_loop_suppressed_calls=0,
                force_final_reason=None,
                submit_answer_succeeded=True,
            )
            first_terminal = mcp_agent._first_terminal_failure_event_code([event_code])

            assert primary == "none"
            assert secondary == []
            assert first_terminal is None

    def test_provider_credits_exhausted_classification(self) -> None:
        import llm_client.agent.mcp_agent as mcp_agent

        is_provider, event_code, classification, retryable = (
            mcp_agent._provider_failure_classification(
                Exception(
                    'OpenrouterException - {"error":{"message":"Insufficient credits","code":402}}',
                ),
                'OpenrouterException - {"error":{"message":"Insufficient credits","code":402}}',
            )
        )

        assert is_provider is True
        assert event_code == "PROVIDER_CREDITS_EXHAUSTED"
        assert classification == "provider_credits_exhausted"
        assert retryable is False


class TestLaneClosureAnalysis:
    def test_lane_closure_detects_unresolved_requirements(self) -> None:
        import llm_client.agent.mcp_agent as mcp_agent

        normalized = mcp_agent._normalize_tool_contracts(
            {
                "relationship_onehop": {
                    "requires_all": [{"kind": "ENTITY_SET", "ref_type": "id"}],
                    "produces": ["RELATIONSHIP_SET"],
                },
            }
        )
        analysis = mcp_agent._analyze_lane_closure(
            normalized_tool_contracts=normalized,
            initial_artifacts={"QUERY_TEXT"},
            initial_capabilities={"QUERY_TEXT": {(None, None, None)}},
            available_bindings={},
        )

        assert analysis["lane_closed"] is False
        assert analysis["unresolved_tool_count"] == 1
        assert analysis["unresolved_tools"][0]["tool"] == "relationship_onehop"

    def test_lane_closure_resolves_via_conversion_tool(self) -> None:
        import llm_client.agent.mcp_agent as mcp_agent

        normalized = mcp_agent._normalize_tool_contracts(
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
        analysis = mcp_agent._analyze_lane_closure(
            normalized_tool_contracts=normalized,
            initial_artifacts={"QUERY_TEXT"},
            initial_capabilities={"QUERY_TEXT": {(None, None, None)}},
            available_bindings={},
        )

        assert analysis["lane_closed"] is True
        assert analysis["unresolved_tool_count"] == 0
        assert "ENTITY_SET" in analysis["reachable_artifacts"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_tool(name: str, desc: str = "tool") -> MagicMock:
    t = MagicMock()
    t.name = name
    t.description = desc
    t.inputSchema = {"type": "object", "properties": {}}
    return t


def _make_tool_result(text: str, is_error: bool = False) -> MagicMock:
    content_item = MagicMock()
    content_item.text = text
    result = MagicMock()
    result.content = [content_item]
    result.isError = is_error
    return result


def _nontrivial_direct_tool(query: str, limit: int = 5) -> str:
    return f"{query}:{limit}"


def _make_llm_result(
    content: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
    cost: float = 0.001,
    finish_reason: str = "stop",
    model: str = "test-model",
    warnings: list[str] | None = None,
) -> LLMCallResult:
    return LLMCallResult(
        content=content,
        usage=usage or {"input_tokens": 100, "output_tokens": 50},
        cost=cost,
        model=model,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        warnings=warnings or [],
    )


# ---------------------------------------------------------------------------
# Full agent loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAcallWithMcp:
    """Test _acall_with_mcp with fully mocked MCP and LLM."""

    async def test_single_turn_no_tools(self) -> None:
        """LLM returns a text answer immediately (no tool calls)."""
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(
            tools=[_make_tool("search")],
        ))

        with (
            patch("llm_client.agent.mcp_agent._import_mcp") as mock_import,
            patch("llm_client.agent.mcp_agent._inner_acall_llm") as mock_acall,
        ):
            # Setup MCP mocks
            mock_stdio = AsyncMock()
            mock_stdio.__aenter__ = AsyncMock(return_value=("read", "write"))
            mock_stdio.__aexit__ = AsyncMock(return_value=False)
            mock_import.return_value = (
                MagicMock(return_value=mock_stdio),  # stdio_client
                MagicMock,  # StdioServerParameters
                MagicMock(return_value=mock_session),  # ClientSession
            )

            # Setup session context manager
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            # LLM returns text answer
            mock_acall.return_value = _make_llm_result(content="Paris", finish_reason="stop")

            from llm_client.agent.mcp_agent import _acall_with_mcp
            result = await _acall_with_mcp(
                "test-model",
                [{"role": "user", "content": "What is the capital of France?"}],
                mcp_servers={"server": {"command": "python", "args": ["s.py"]}},
            )

            assert result.content == "Paris"
            assert result.finish_reason == "stop"
            assert isinstance(result.raw_response, MCPAgentResult)
            assert result.raw_response.turns == 1
            assert result.raw_response.tool_calls == []
            # No tool calls → trace has just the final assistant message
            trace = result.raw_response.conversation_trace
            non_budget_trace = [
                msg for msg in trace
                if "budget:" not in str(msg.get("content", "")).lower()
                and msg.get("synthetic") != "active_artifact_context"
            ]
            assert len(non_budget_trace) == 1
            assert non_budget_trace[0]["role"] == "assistant"
            assert non_budget_trace[0]["content"] == "Paris"
            metadata = result.raw_response.metadata
            assert isinstance(metadata.get("hard_bindings_hash"), str)
            assert isinstance(metadata.get("full_bindings_hash"), str)
            assert isinstance(metadata.get("run_config_hash"), str)
            assert metadata.get("first_terminal_failure_event_code") is None
            assert metadata["answer_present"] is True
            assert metadata["grounded_completed"] is False
            assert metadata["forced_terminal_accepted"] is False
            assert metadata["reliability_completed"] is True
            assert metadata["run_completed"] is True

    async def test_strict_adoption_profile_warns_when_runtime_guards_are_missing(self) -> None:
        """Strict adoption profile should surface structured violations even when not enforced."""
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(
            tools=[_make_tool("search")],
        ))

        with (
            patch("llm_client.agent.mcp_agent._import_mcp") as mock_import,
            patch("llm_client.agent.mcp_agent._inner_acall_llm") as mock_acall,
        ):
            mock_stdio = AsyncMock()
            mock_stdio.__aenter__ = AsyncMock(return_value=("read", "write"))
            mock_stdio.__aexit__ = AsyncMock(return_value=False)
            mock_import.return_value = (
                MagicMock(return_value=mock_stdio),
                MagicMock,
                MagicMock(return_value=mock_session),
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_acall.return_value = _make_llm_result(content="Paris", finish_reason="stop")

            from llm_client.agent.mcp_agent import _acall_with_mcp
            result = await _acall_with_mcp(
                "test-model",
                [{"role": "user", "content": "What is the capital of France?"}],
                mcp_servers={"server": {"command": "python", "args": ["s.py"]}},
                adoption_profile="strict",
            )

            metadata = result.raw_response.metadata
            assert metadata["adoption_profile_effective"] == "strict"
            assert metadata["adoption_profile_satisfied"] is False
            assert "require_tool_reasoning must be enabled" in metadata["adoption_profile_violations"]
            assert any("ADOPTION_PROFILE_VIOLATION[strict]" in warning for warning in result.warnings)

    async def test_strict_adoption_profile_can_fail_fast(self) -> None:
        """Strict adoption profile can raise immediately when enforcement is requested."""
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(
            tools=[_make_tool("search")],
        ))

        with patch("llm_client.agent.mcp_agent._import_mcp") as mock_import:
            mock_stdio = AsyncMock()
            mock_stdio.__aenter__ = AsyncMock(return_value=("read", "write"))
            mock_stdio.__aexit__ = AsyncMock(return_value=False)
            mock_import.return_value = (
                MagicMock(return_value=mock_stdio),
                MagicMock,
                MagicMock(return_value=mock_session),
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            from llm_client.agent.mcp_agent import _acall_with_mcp
            with pytest.raises(ValueError, match="ADOPTION_PROFILE_VIOLATION\\[strict\\]"):
                await _acall_with_mcp(
                    "test-model",
                    [{"role": "user", "content": "What is the capital of France?"}],
                    mcp_servers={"server": {"command": "python", "args": ["s.py"]}},
                    adoption_profile="strict",
                    adoption_profile_enforce=True,
                )

    async def test_strict_adoption_profile_warns_for_nontrivial_direct_tool_quality(self) -> None:
        with patch("llm_client.agent.mcp_agent._inner_acall_llm") as mock_acall:
            mock_acall.return_value = _make_llm_result(content="done", finish_reason="stop")

            from llm_client.agent.mcp_agent import _acall_with_tools

            result = await _acall_with_tools(
                "test-model",
                [{"role": "user", "content": "Q"}],
                python_tools=[_nontrivial_direct_tool],
                require_tool_reasoning=True,
                enforce_tool_contracts=True,
                progressive_tool_disclosure=True,
                adoption_profile="strict",
                task="test",
                trace_id="test_strict_adoption_profile_warns_for_nontrivial_direct_tool_quality",
            )

            metadata = result.raw_response.metadata
            violations = metadata["adoption_profile_violations"]
            assert metadata["adoption_profile_effective"] == "strict"
            assert metadata["adoption_profile_satisfied"] is False
            assert any("must include input examples" in violation for violation in violations)
            assert any("must declare tool_contracts" in violation for violation in violations)

    async def test_multi_turn_with_tool_calls(self) -> None:
        """LLM calls a tool, gets result, then answers."""
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(
            tools=[_make_tool("search")],
        ))
        mock_session.call_tool = AsyncMock(
            return_value=_make_tool_result('{"entities": ["Paris"]}'),
        )

        with (
            patch("llm_client.agent.mcp_agent._import_mcp") as mock_import,
            patch("llm_client.agent.mcp_agent._inner_acall_llm") as mock_acall,
        ):
            mock_stdio = AsyncMock()
            mock_stdio.__aenter__ = AsyncMock(return_value=("read", "write"))
            mock_stdio.__aexit__ = AsyncMock(return_value=False)
            mock_import.return_value = (
                MagicMock(return_value=mock_stdio),
                MagicMock,
                MagicMock(return_value=mock_session),
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            # Turn 1: LLM calls a tool
            # Turn 2: LLM answers
            mock_acall.side_effect = [
                _make_llm_result(tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"query": "capital of France"}',
                    },
                }]),
                _make_llm_result(content="Paris"),
            ]

            from llm_client.agent.mcp_agent import _acall_with_mcp
            result = await _acall_with_mcp(
                "test-model",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
            )

            assert result.content == "Paris"
            agent_result = result.raw_response
            assert isinstance(agent_result, MCPAgentResult)
            assert agent_result.turns == 2
            assert len(agent_result.tool_calls) == 1
            assert agent_result.tool_calls[0].tool == "search"
            assert agent_result.tool_calls[0].server == "srv"
            assert agent_result.tool_calls[0].result is not None
            assert agent_result.tool_calls[0].error is None

            # Verify conversation trace captures all messages
            trace = agent_result.conversation_trace
            non_budget_trace = [
                msg for msg in trace
                if ("budget:" not in str(msg.get("content", "")).lower())
                and ("Observability requirement" not in str(msg.get("content", "")))
                and msg.get("synthetic") != "active_artifact_context"
            ]
            assert len(non_budget_trace) == 3  # assistant(tool_call) + tool_result + assistant(answer)
            assert non_budget_trace[0]["role"] == "assistant"
            assert len(non_budget_trace[0]["tool_calls"]) == 1
            assert non_budget_trace[0]["tool_calls"][0]["name"] == "search"
            assert non_budget_trace[1]["role"] == "tool"
            assert non_budget_trace[2]["role"] == "assistant"
            assert non_budget_trace[2]["content"] == "Paris"

            # Verify the session.call_tool was called correctly
            mock_session.call_tool.assert_called_once_with(
                "search", {"query": "capital of France"},
            )

    async def test_max_turns_exhausted(self) -> None:
        """Loop reaches max_turns and makes a final call without tools."""
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(
            tools=[_make_tool("search")],
        ))
        mock_session.call_tool = AsyncMock(
            return_value=_make_tool_result("result"),
        )

        with (
            patch("llm_client.agent.mcp_agent._import_mcp") as mock_import,
            patch("llm_client.agent.mcp_agent._inner_acall_llm") as mock_acall,
        ):
            mock_stdio = AsyncMock()
            mock_stdio.__aenter__ = AsyncMock(return_value=("read", "write"))
            mock_stdio.__aexit__ = AsyncMock(return_value=False)
            mock_import.return_value = (
                MagicMock(return_value=mock_stdio),
                MagicMock,
                MagicMock(return_value=mock_session),
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            # Always return tool calls (never stops on its own)
            tool_call_result = _make_llm_result(tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": "{}"},
            }])
            final_result = _make_llm_result(content="forced answer")

            # max_turns=2: 2 loop iterations + 1 final call = 3 calls total
            mock_acall.side_effect = [tool_call_result, tool_call_result, final_result]

            from llm_client.agent.mcp_agent import _acall_with_mcp
            result = await _acall_with_mcp(
                "test-model",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
                max_turns=2,
            )

            assert result.content == "forced answer"
            assert result.raw_response.turns == 3  # 2 loop + 1 forced

    async def test_max_tool_calls_exhausted(self) -> None:
        """Loop stops when max_tool_calls is exhausted and forces final answer."""
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(
            tools=[_make_tool("search")],
        ))
        mock_session.call_tool = AsyncMock(
            return_value=_make_tool_result("result"),
        )

        with (
            patch("llm_client.agent.mcp_agent._import_mcp") as mock_import,
            patch("llm_client.agent.mcp_agent._inner_acall_llm") as mock_acall,
        ):
            mock_stdio = AsyncMock()
            mock_stdio.__aenter__ = AsyncMock(return_value=("read", "write"))
            mock_stdio.__aexit__ = AsyncMock(return_value=False)
            mock_import.return_value = (
                MagicMock(return_value=mock_stdio),
                MagicMock,
                MagicMock(return_value=mock_session),
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            tool_call_result = _make_llm_result(tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": "{}"},
            }])
            final_result = _make_llm_result(content="forced by tool budget")
            mock_acall.side_effect = [tool_call_result, final_result]

            from llm_client.agent.mcp_agent import _acall_with_mcp
            result = await _acall_with_mcp(
                "test-model",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
                max_turns=10,
                max_tool_calls=1,
            )

            assert result.content == "forced by tool budget"
            assert result.raw_response.turns == 2  # 1 loop + 1 forced final
            assert len(result.raw_response.tool_calls) == 1

    async def test_forced_final_llm_exception_preserves_tool_history(self) -> None:
        """Forced-final provider failure should keep prior tool trace instead of raising."""
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(
            tools=[_make_tool("search")],
        ))
        mock_session.call_tool = AsyncMock(
            return_value=_make_tool_result("result"),
        )

        with (
            patch("llm_client.agent.mcp_agent._import_mcp") as mock_import,
            patch("llm_client.agent.mcp_agent._inner_acall_llm") as mock_acall,
        ):
            mock_stdio = AsyncMock()
            mock_stdio.__aenter__ = AsyncMock(return_value=("read", "write"))
            mock_stdio.__aexit__ = AsyncMock(return_value=False)
            mock_import.return_value = (
                MagicMock(return_value=mock_stdio),
                MagicMock,
                MagicMock(return_value=mock_session),
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            tool_call_result = _make_llm_result(tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": "{}"},
            }])
            mock_acall.side_effect = [
                tool_call_result,
                Exception(
                    "Empty content from LLM [litellm_completion:provider_empty_candidates retryable=True]"
                ),
            ]

            from llm_client.agent.mcp_agent import _acall_with_mcp
            result = await _acall_with_mcp(
                "test-model",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
                max_turns=10,
                max_tool_calls=1,
            )

            assert result.finish_reason == "error"
            assert "Empty content from LLM" in result.content
            assert len(result.raw_response.tool_calls) == 1
            metadata = result.raw_response.metadata
            assert metadata["forced_final_breaker_effective"] is False
            assert metadata["foundation_event_validation_errors"] == 0
            assert "PROVIDER_EMPTY_RESPONSE" not in metadata.get("failure_event_codes", [])
            assert "PROVIDER_EMPTY_RESPONSE_FIRST_TURN" not in metadata.get("failure_event_codes", [])
            assert any(
                "AGENT_LLM_CALL_FAILED" in w for w in result.raw_response.warnings
            )

    async def test_forced_final_fallback_model_recovers_primary_failure(self) -> None:
        """Forced-final can recover via finalization-only fallback without masking primary failure event metadata."""
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(
            tools=[_make_tool("search")],
        ))
        mock_session.call_tool = AsyncMock(return_value=_make_tool_result("result"))

        with (
            patch("llm_client.agent.mcp_agent._import_mcp") as mock_import,
            patch("llm_client.agent.mcp_agent._inner_acall_llm") as mock_acall,
        ):
            mock_stdio = AsyncMock()
            mock_stdio.__aenter__ = AsyncMock(return_value=("read", "write"))
            mock_stdio.__aexit__ = AsyncMock(return_value=False)
            mock_import.return_value = (
                MagicMock(return_value=mock_stdio),
                MagicMock,
                MagicMock(return_value=mock_session),
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            tool_call_result = _make_llm_result(tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": "{}"},
            }])
            mock_acall.side_effect = [
                tool_call_result,
                Exception(
                    "Empty content from LLM [litellm_completion:provider_empty_candidates retryable=True]"
                ),
                _make_llm_result(content="fallback final answer", model="fallback-model"),
            ]

            from llm_client.agent.mcp_agent import _acall_with_mcp
            result = await _acall_with_mcp(
                "test-model",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
                max_turns=10,
                max_tool_calls=1,
                forced_final_max_attempts=2,
                finalization_fallback_models=["fallback-model"],
            )

            assert result.content == "fallback final answer"
            assert result.finish_reason == "stop"
            metadata = result.raw_response.metadata
            assert metadata["forced_final_attempts"] == 2
            assert metadata["finalization_fallback_used"] is True
            assert metadata["finalization_fallback_succeeded"] is True
            assert metadata["finalization_fallback_usage_rate"] == 0.5
            assert metadata["finalization_attempt_counts_by_model"] == {
                "test-model": 1,
                "fallback-model": 1,
            }
            assert metadata["finalization_failure_counts_by_model"] == {"test-model": 1}
            assert metadata["finalization_success_counts_by_model"] == {"fallback-model": 1}
            assert metadata["provider_failure_event_total"] == 0
            assert metadata["provider_caused_incompletion"] is False
            assert "FINALIZATION_PRIMARY_FAILED" in metadata.get("finalization_events", [])
            assert "FINALIZATION_FALLBACK_SUCCEEDED" in metadata.get("finalization_events", [])
            # Recovered finalization should not count as terminal run failure.
            assert metadata.get("primary_failure_class") == "none"

    async def test_max_tool_calls_ignores_todo_tools(self) -> None:
        """todo_write calls do not consume max_tool_calls budget."""
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(
            tools=[_make_tool("todo_write"), _make_tool("search")],
        ))
        mock_session.call_tool = AsyncMock(
            side_effect=[_make_tool_result("todo-ok"), _make_tool_result("search-ok")],
        )

        with (
            patch("llm_client.agent.mcp_agent._import_mcp") as mock_import,
            patch("llm_client.agent.mcp_agent._inner_acall_llm") as mock_acall,
        ):
            mock_stdio = AsyncMock()
            mock_stdio.__aenter__ = AsyncMock(return_value=("read", "write"))
            mock_stdio.__aexit__ = AsyncMock(return_value=False)
            mock_import.return_value = (
                MagicMock(return_value=mock_stdio),
                MagicMock,
                MagicMock(return_value=mock_session),
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_acall.side_effect = [
                _make_llm_result(tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "todo_write", "arguments": "{}"},
                }]),
                _make_llm_result(tool_calls=[{
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }]),
                _make_llm_result(content="forced by retrieval budget"),
            ]

            from llm_client.agent.mcp_agent import _acall_with_mcp
            result = await _acall_with_mcp(
                "test-model",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
                max_turns=10,
                max_tool_calls=1,
            )

            assert result.content == "forced by retrieval budget"
            assert result.raw_response.turns == 3  # 2 loop turns + 1 forced final
            assert len(result.raw_response.tool_calls) == 2
            assert result.raw_response.metadata["budgeted_tool_calls_used"] == 1
            assert mock_session.call_tool.call_count == 2

    async def test_max_tool_calls_ignores_plan_tools(self) -> None:
        """create_plan/update_plan calls do not consume max_tool_calls budget."""
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(
            tools=[_make_tool("search")],
        ))
        mock_session.call_tool = AsyncMock(return_value=_make_tool_result("search-ok"))

        with (
            patch("llm_client.agent.mcp_agent._import_mcp") as mock_import,
            patch("llm_client.agent.mcp_agent._inner_acall_llm") as mock_acall,
        ):
            mock_stdio = AsyncMock()
            mock_stdio.__aenter__ = AsyncMock(return_value=("read", "write"))
            mock_stdio.__aexit__ = AsyncMock(return_value=False)
            mock_import.return_value = (
                MagicMock(return_value=mock_stdio),
                MagicMock,
                MagicMock(return_value=mock_session),
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_acall.side_effect = [
                _make_llm_result(tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "create_plan",
                        "arguments": '{"steps":[{"step_id":"s1","description":"Search"}]}',
                    },
                }]),
                _make_llm_result(tool_calls=[{
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "update_plan",
                        "arguments": '{"step_id":"s1","status":"done","result":"done"}',
                    },
                }]),
                _make_llm_result(tool_calls=[{
                    "id": "call_3",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }]),
                _make_llm_result(content="forced by retrieval budget"),
            ]

            from llm_client.agent.agent_planning import PlanningConfig
            from llm_client.agent.mcp_agent import _acall_with_mcp

            result = await _acall_with_mcp(
                "test-model",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
                max_turns=10,
                max_tool_calls=1,
                planning_config=PlanningConfig(enabled=True),
            )

            assert result.content == "forced by retrieval budget"
            assert result.raw_response.turns == 4  # 3 loop turns + 1 forced final
            assert len(result.raw_response.tool_calls) == 3
            assert result.raw_response.metadata["budgeted_tool_calls_used"] == 1
            assert [record.tool for record in result.raw_response.tool_calls[:2]] == [
                "create_plan",
                "update_plan",
            ]
            assert mock_session.call_tool.call_count == 1

    async def test_forced_final_circuit_breaker_opens_on_same_class_failures(self) -> None:
        """Repeated same-class forced-final failures should open circuit breaker."""
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(
            tools=[_make_tool("search")],
        ))
        mock_session.call_tool = AsyncMock(return_value=_make_tool_result("result"))

        with (
            patch("llm_client.agent.mcp_agent._import_mcp") as mock_import,
            patch("llm_client.agent.mcp_agent._inner_acall_llm") as mock_acall,
        ):
            mock_stdio = AsyncMock()
            mock_stdio.__aenter__ = AsyncMock(return_value=("read", "write"))
            mock_stdio.__aexit__ = AsyncMock(return_value=False)
            mock_import.return_value = (
                MagicMock(return_value=mock_stdio),
                MagicMock,
                MagicMock(return_value=mock_session),
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            tool_call_result = _make_llm_result(tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": "{}"},
            }])
            provider_empty_exc = Exception(
                "Empty content from LLM [litellm_completion:provider_empty_candidates retryable=True]"
            )
            mock_acall.side_effect = [tool_call_result, provider_empty_exc, provider_empty_exc]

            from llm_client.agent.mcp_agent import _acall_with_mcp
            result = await _acall_with_mcp(
                "test-model",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
                max_turns=10,
                max_tool_calls=1,
                forced_final_max_attempts=4,
                forced_final_circuit_breaker_threshold=2,
            )

            assert result.finish_reason == "error"
            metadata = result.raw_response.metadata
            assert metadata["forced_final_attempts"] == 2
            assert metadata["forced_final_circuit_breaker_opened"] is True
            assert metadata["finalization_breaker_open_rate"] == 1.0
            assert metadata["provider_caused_incompletion"] is True
            assert metadata["provider_failure_event_total"] >= 2
            assert metadata["provider_failure_event_code_counts"].get("PROVIDER_EMPTY_CANDIDATES", 0) >= 2
            assert "FINALIZATION_CIRCUIT_BREAKER_OPEN" in metadata.get("failure_event_codes", [])
            assert any(
                "FINALIZATION_CIRCUIT_BREAKER_OPEN" in warning
                for warning in result.raw_response.warnings
            )
            assert mock_acall.call_count == 3  # 1 loop turn + 2 forced-final attempts

    async def test_forced_final_mixed_failure_classes_do_not_open_breaker(self) -> None:
        """Mixed forced-final failure classes should not open same-class breaker."""
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(
            tools=[_make_tool("search")],
        ))
        mock_session.call_tool = AsyncMock(return_value=_make_tool_result("result"))

        with (
            patch("llm_client.agent.mcp_agent._import_mcp") as mock_import,
            patch("llm_client.agent.mcp_agent._inner_acall_llm") as mock_acall,
        ):
            mock_stdio = AsyncMock()
            mock_stdio.__aenter__ = AsyncMock(return_value=("read", "write"))
            mock_stdio.__aexit__ = AsyncMock(return_value=False)
            mock_import.return_value = (
                MagicMock(return_value=mock_stdio),
                MagicMock,
                MagicMock(return_value=mock_session),
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            tool_call_result = _make_llm_result(tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": "{}"},
            }])
            provider_empty_exc = Exception(
                "Empty content from LLM [litellm_completion:provider_empty_candidates retryable=True]"
            )
            parse_exc = Exception("JSON parse error in finalization payload")
            mock_acall.side_effect = [tool_call_result, provider_empty_exc, parse_exc]

            from llm_client.agent.mcp_agent import _acall_with_mcp
            result = await _acall_with_mcp(
                "test-model",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
                max_turns=10,
                max_tool_calls=1,
                forced_final_max_attempts=2,
                forced_final_circuit_breaker_threshold=2,
            )

            assert result.finish_reason == "error"
            metadata = result.raw_response.metadata
            assert metadata["forced_final_attempts"] == 2
            assert metadata["forced_final_circuit_breaker_opened"] is False
            assert metadata["finalization_breaker_open_rate"] == 0.0
            assert metadata["finalization_failure_code_counts"].get("PROVIDER_EMPTY_CANDIDATES", 0) == 1
            assert metadata["finalization_failure_code_counts"].get("TOOL_EXECUTION_RUNTIME_ERROR", 0) == 1
            assert "FINALIZATION_CIRCUIT_BREAKER_OPEN" not in metadata.get("failure_event_codes", [])

    async def test_forced_final_tool_calls_are_disallowed_without_execution(self) -> None:
        """Forced-final tool-call-shaped outputs must not execute tools and should classify deterministically."""
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(
            tools=[_make_tool("search")],
        ))
        mock_session.call_tool = AsyncMock(return_value=_make_tool_result("result"))

        with (
            patch("llm_client.agent.mcp_agent._import_mcp") as mock_import,
            patch("llm_client.agent.mcp_agent._inner_acall_llm") as mock_acall,
        ):
            mock_stdio = AsyncMock()
            mock_stdio.__aenter__ = AsyncMock(return_value=("read", "write"))
            mock_stdio.__aexit__ = AsyncMock(return_value=False)
            mock_import.return_value = (
                MagicMock(return_value=mock_stdio),
                MagicMock,
                MagicMock(return_value=mock_session),
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            tool_call_result = _make_llm_result(tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": "{}"},
            }])
            forced_final_tool_call = _make_llm_result(
                content="",
                tool_calls=[{
                    "id": "forced_call",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }],
                finish_reason="tool_calls",
            )
            mock_acall.side_effect = [
                tool_call_result,
                forced_final_tool_call,
                _make_llm_result(content="final answer via fallback", model="fallback-model"),
            ]

            from llm_client.agent.mcp_agent import _acall_with_mcp
            result = await _acall_with_mcp(
                "test-model",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
                max_turns=10,
                max_tool_calls=1,
                forced_final_max_attempts=2,
                finalization_fallback_models=["fallback-model"],
            )

            assert result.content == "final answer via fallback"
            metadata = result.raw_response.metadata
            assert any(
                attempt.get("error_code") == "FINALIZATION_TOOL_CALL_DISALLOWED"
                for attempt in metadata.get("finalization_fallback_attempts", [])
            )
            assert metadata["finalization_failure_code_counts"].get("FINALIZATION_TOOL_CALL_DISALLOWED", 0) == 1
            assert metadata["finalization_fallback_usage_rate"] == 0.5
            # Forced-final tool-calls are never executed; only the original loop call runs.
            assert mock_session.call_tool.call_count == 1

    async def test_foundation_schema_strict_raises_on_invalid_event(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """FOUNDATION_SCHEMA_STRICT=1 should raise when event validation fails."""
        monkeypatch.setenv("FOUNDATION_SCHEMA_STRICT", "1")

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(
            tools=[_make_tool("search")],
        ))
        mock_session.call_tool = AsyncMock(return_value=_make_tool_result("result"))

        with (
            patch("llm_client.agent.mcp_agent._import_mcp") as mock_import,
            patch("llm_client.agent.mcp_agent._inner_acall_llm") as mock_acall,
            patch("llm_client.agent.mcp_agent.validate_foundation_event", side_effect=ValueError("schema mismatch")),
        ):
            mock_stdio = AsyncMock()
            mock_stdio.__aenter__ = AsyncMock(return_value=("read", "write"))
            mock_stdio.__aexit__ = AsyncMock(return_value=False)
            mock_import.return_value = (
                MagicMock(return_value=mock_stdio),
                MagicMock,
                MagicMock(return_value=mock_session),
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_acall.side_effect = [
                _make_llm_result(tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }]),
                _make_llm_result(content="answer"),
            ]

            from llm_client.agent.mcp_agent import _acall_with_mcp
            with pytest.raises(RuntimeError, match="FOUNDATION_EVENT_INVALID"):
                await _acall_with_mcp(
                    "test-model",
                    [{"role": "user", "content": "Q"}],
                    mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
                    max_turns=10,
                )

    async def test_require_tool_reasoning_rejects_mcp_call(self) -> None:
        """Strict mode rejects tool calls that omit tool_reasoning."""
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(
            tools=[_make_tool("search")],
        ))
        mock_session.call_tool = AsyncMock(return_value=_make_tool_result("result"))

        with (
            patch("llm_client.agent.mcp_agent._import_mcp") as mock_import,
            patch("llm_client.agent.mcp_agent._inner_acall_llm") as mock_acall,
        ):
            mock_stdio = AsyncMock()
            mock_stdio.__aenter__ = AsyncMock(return_value=("read", "write"))
            mock_stdio.__aexit__ = AsyncMock(return_value=False)
            mock_import.return_value = (
                MagicMock(return_value=mock_stdio),
                MagicMock,
                MagicMock(return_value=mock_session),
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_acall.side_effect = [
                _make_llm_result(tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }]),
                _make_llm_result(content="answer after rejection"),
            ]

            from llm_client.agent.mcp_agent import _acall_with_mcp
            result = await _acall_with_mcp(
                "test-model",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
                require_tool_reasoning=True,
            )

            assert result.content == "answer after rejection"
            # Invalid call is rejected by compliance gate pre-execution.
            assert len(result.raw_response.tool_calls) == 1
            assert result.raw_response.tool_calls[0].server == "__compliance__"
            assert result.raw_response.metadata["rejected_missing_reasoning_calls"] == 1
            assert result.raw_response.metadata["tool_gate_rejections"] == 1
            assert (
                result.raw_response.metadata["failure_event_code_counts"].get(
                    "TOOL_VALIDATION_REJECTED_MISSING_TOOL_REASONING", 0,
                )
                >= 1
            )
            assert any("missing tool_reasoning" in w.lower() for w in (result.warnings or []))
            # Synthetic tool rejection is still visible in trace for observability
            assert any(
                "Missing required argument: tool_reasoning" in str(msg.get("content", ""))
                for msg in result.raw_response.conversation_trace
                if msg.get("role") == "tool"
            )
            mock_session.call_tool.assert_not_called()

    async def test_compliance_gate_rejects_unknown_schema_args(self) -> None:
        """Schema gate rejects unknown args before MCP execution."""
        strict_tool = MagicMock()
        strict_tool.name = "search"
        strict_tool.description = "search"
        strict_tool.inputSchema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        }
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(
            tools=[strict_tool],
        ))
        mock_session.call_tool = AsyncMock(return_value=_make_tool_result("result"))

        with (
            patch("llm_client.agent.mcp_agent._import_mcp") as mock_import,
            patch("llm_client.agent.mcp_agent._inner_acall_llm") as mock_acall,
        ):
            mock_stdio = AsyncMock()
            mock_stdio.__aenter__ = AsyncMock(return_value=("read", "write"))
            mock_stdio.__aexit__ = AsyncMock(return_value=False)
            mock_import.return_value = (
                MagicMock(return_value=mock_stdio),
                MagicMock,
                MagicMock(return_value=mock_session),
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_acall.side_effect = [
                _make_llm_result(
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": '{"tool_reasoning":"lookup","foo":1}',
                            },
                        }
                    ]
                ),
                _make_llm_result(content="answer after schema rejection"),
            ]

            from llm_client.agent.mcp_agent import _acall_with_mcp
            result = await _acall_with_mcp(
                "test-model",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
            )

            assert result.content == "answer after schema rejection"
            assert result.raw_response.metadata["tool_gate_rejections"] == 1
            assert (
                result.raw_response.metadata["failure_event_code_counts"].get(
                    "TOOL_VALIDATION_REJECTED_SCHEMA", 0,
                )
                >= 1
            )
            assert result.raw_response.metadata["primary_failure_class"] == "composability"
            mock_session.call_tool.assert_not_called()

    async def test_enforce_tool_contracts_rejects_incompatible_call(self) -> None:
        """Contract mode rejects tool calls when required artifacts are unavailable."""
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(
            tools=[_make_tool("search")],
        ))
        mock_session.call_tool = AsyncMock(return_value=_make_tool_result("result"))

        with (
            patch("llm_client.agent.mcp_agent._import_mcp") as mock_import,
            patch("llm_client.agent.mcp_agent._inner_acall_llm") as mock_acall,
        ):
            mock_stdio = AsyncMock()
            mock_stdio.__aenter__ = AsyncMock(return_value=("read", "write"))
            mock_stdio.__aexit__ = AsyncMock(return_value=False)
            mock_import.return_value = (
                MagicMock(return_value=mock_stdio),
                MagicMock,
                MagicMock(return_value=mock_session),
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_acall.side_effect = [
                _make_llm_result(tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"tool_reasoning":"test"}',
                    },
                }]),
                _make_llm_result(content="answer after contract rejection"),
            ]

            from llm_client.agent.mcp_agent import _acall_with_mcp
            result = await _acall_with_mcp(
                "test-model",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
                enforce_tool_contracts=True,
                tool_contracts={
                    "search": {
                        "requires_all": ["ENTITY_SET"],
                        "produces": ["CHUNK_SET"],
                    },
                },
                initial_artifacts=("QUERY_TEXT",),
            )

            assert result.content == "answer after contract rejection"
            assert result.raw_response.metadata["tool_contract_rejections"] == 1
            assert result.raw_response.metadata["available_artifacts_final"] == ["QUERY_TEXT"]
            assert result.raw_response.metadata["primary_failure_class"] == "composability"
            assert (
                result.raw_response.metadata["failure_event_code_counts"].get(
                    "TOOL_VALIDATION_REJECTED_MISSING_PREREQUISITE", 0,
                )
                >= 1
            )
            assert any(
                "Tool contract violation" in str(msg.get("content", ""))
                for msg in result.raw_response.conversation_trace
                if msg.get("role") == "tool"
            )
            mock_session.call_tool.assert_not_called()

    async def test_enforce_tool_contracts_rejects_binding_conflict(self) -> None:
        """Hard binding conflicts are rejected pre-execution with stable validation code."""
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(
            tools=[_make_tool("search")],
        ))
        mock_session.call_tool = AsyncMock(return_value=_make_tool_result("result"))

        with (
            patch("llm_client.agent.mcp_agent._import_mcp") as mock_import,
            patch("llm_client.agent.mcp_agent._inner_acall_llm") as mock_acall,
        ):
            mock_stdio = AsyncMock()
            mock_stdio.__aenter__ = AsyncMock(return_value=("read", "write"))
            mock_stdio.__aexit__ = AsyncMock(return_value=False)
            mock_import.return_value = (
                MagicMock(return_value=mock_stdio),
                MagicMock,
                MagicMock(return_value=mock_session),
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_acall.side_effect = [
                _make_llm_result(tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"tool_reasoning":"lookup","graph_reference_id":"graph_b"}',
                    },
                }]),
                _make_llm_result(content="binding conflict handled"),
            ]

            from llm_client.agent.mcp_agent import _acall_with_mcp
            result = await _acall_with_mcp(
                "test-model",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
                enforce_tool_contracts=True,
                tool_contracts={
                    "search": {
                        "requires_all": ["QUERY_TEXT"],
                        "produces": ["CHUNK_SET"],
                    },
                },
                initial_artifacts=("QUERY_TEXT",),
                initial_bindings={"graph_id": "graph_a"},
            )

            assert result.content == "binding conflict handled"
            assert result.raw_response.metadata["tool_contract_rejections"] == 0
            assert result.raw_response.metadata["tool_gate_rejections"] == 1
            assert result.raw_response.metadata["available_bindings_final"]["graph_id"] == "graph_a"
            violations = result.raw_response.metadata["tool_gate_violation_events"]
            assert len(violations) == 1
            assert (
                violations[0]["error_code"]
                == "TOOL_VALIDATION_REJECTED_BINDING_CONFLICT"
            )
            assert violations[0]["failure_phase"] == "binding_validation"
            mock_session.call_tool.assert_not_called()

    async def test_enforce_tool_contracts_propagates_artifacts(self) -> None:
        """Successful contract-validated calls grow available artifacts."""
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(
            tools=[_make_tool("entity_tfidf"), _make_tool("entity_onehop")],
        ))
        mock_session.call_tool = AsyncMock(
            side_effect=[_make_tool_result("entity hit"), _make_tool_result("onehop hit")],
        )

        with (
            patch("llm_client.agent.mcp_agent._import_mcp") as mock_import,
            patch("llm_client.agent.mcp_agent._inner_acall_llm") as mock_acall,
        ):
            mock_stdio = AsyncMock()
            mock_stdio.__aenter__ = AsyncMock(return_value=("read", "write"))
            mock_stdio.__aexit__ = AsyncMock(return_value=False)
            mock_import.return_value = (
                MagicMock(return_value=mock_stdio),
                MagicMock,
                MagicMock(return_value=mock_session),
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_acall.side_effect = [
                _make_llm_result(tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "entity_tfidf",
                        "arguments": '{"tool_reasoning":"seed entities from query"}',
                    },
                }]),
                _make_llm_result(tool_calls=[{
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "entity_onehop",
                        "arguments": '{"tool_reasoning":"expand entity neighborhood"}',
                    },
                }]),
                _make_llm_result(content="answer"),
            ]

            from llm_client.agent.mcp_agent import _acall_with_mcp
            result = await _acall_with_mcp(
                "test-model",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
                enforce_tool_contracts=True,
                tool_contracts={
                    "entity_tfidf": {
                        "requires_all": ["QUERY_TEXT"],
                        "produces": ["ENTITY_SET"],
                    },
                    "entity_onehop": {
                        "requires_all": ["ENTITY_SET"],
                        "produces": ["ENTITY_SET"],
                    },
                },
                initial_artifacts=("QUERY_TEXT",),
            )

            assert result.content == "answer"
            assert mock_session.call_tool.call_count == 2
            assert result.raw_response.metadata["tool_contract_rejections"] == 0
            assert "ENTITY_SET" in result.raw_response.metadata["available_artifacts_final"]

    async def test_artifact_envelope_output_unlocks_next_tool_without_contract_produces(self) -> None:
        """Typed artifact envelopes should drive downstream legality even when contract produces metadata is absent."""
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(
            tools=[_make_tool("entity_tfidf"), _make_tool("relationship_onehop")],
        ))
        mock_session.call_tool = AsyncMock(
            side_effect=[
                _make_tool_result(
                    json.dumps(
                        {
                            "artifact_id": "art_entities_1",
                            "artifact_type": "ENTITY_SET",
                            "schema_version": "1.0.0",
                            "bindings": {"graph_id": "graph_a"},
                            "capabilities": [
                                {
                                    "kind": "ENTITY_SET",
                                    "ref_type": "id",
                                    "namespace": "graph_a",
                                }
                            ],
                            "provenance": {
                                "evidence_refs": [{"chunk_id": "chunk_1"}],
                            },
                            "payload": {"items": [{"entity_id": "e1"}]},
                        }
                    )
                ),
                _make_tool_result("onehop hit"),
            ],
        )

        with (
            patch("llm_client.agent.mcp_agent._import_mcp") as mock_import,
            patch("llm_client.agent.mcp_agent._inner_acall_llm") as mock_acall,
        ):
            mock_stdio = AsyncMock()
            mock_stdio.__aenter__ = AsyncMock(return_value=("read", "write"))
            mock_stdio.__aexit__ = AsyncMock(return_value=False)
            mock_import.return_value = (
                MagicMock(return_value=mock_stdio),
                MagicMock,
                MagicMock(return_value=mock_session),
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_acall.side_effect = [
                _make_llm_result(tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "entity_tfidf",
                        "arguments": '{"tool_reasoning":"seed from query"}',
                    },
                }]),
                _make_llm_result(tool_calls=[{
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "relationship_onehop",
                        "arguments": '{"tool_reasoning":"expand seeded entity"}',
                    },
                }]),
                _make_llm_result(content="answer"),
            ]

            from llm_client.agent.mcp_agent import _acall_with_mcp
            result = await _acall_with_mcp(
                "test-model",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
                enforce_tool_contracts=True,
                tool_contracts={
                    "entity_tfidf": {
                        "requires_all": ["QUERY_TEXT"],
                    },
                    "relationship_onehop": {
                        "requires_all": [
                            {"kind": "ENTITY_SET", "ref_type": "id"},
                        ],
                        "produces": ["RELATIONSHIP_SET"],
                    },
                },
                initial_artifacts=("QUERY_TEXT",),
                initial_bindings={"graph_id": "graph_a"},
            )

            assert result.content == "answer"
            assert mock_session.call_tool.call_count == 2
            assert result.raw_response.metadata["tool_contract_rejections"] == 0
            assert "ENTITY_SET" in result.raw_response.metadata["available_artifacts_final"]
            assert {
                "kind": "ENTITY_SET",
                "ref_type": "id",
                "namespace": "graph_a",
                "bindings_hash": result.raw_response.metadata["hard_bindings_hash"],
            } in result.raw_response.metadata["available_capabilities_final"]
            artifact_events = [
                event
                for event in result.raw_response.metadata["foundation_events"]
                if event.get("event_type") == "ArtifactCreated"
            ]
            assert artifact_events
            assert artifact_events[0]["artifacts"][0]["artifact_id"] == "art_entities_1"
            assert result.raw_response.metadata["active_artifact_context_updates"] >= 1
            assert result.raw_response.metadata["tool_result_metadata_tracked"] >= 1
            trace_texts = [
                str(entry.get("content") or "")
                for entry in result.raw_response.conversation_trace
                if isinstance(entry, dict)
            ]
            assert any("Active artifact context" in text for text in trace_texts)

    async def test_call_mode_outputs_unlock_next_tool_from_discriminated_union_contract(self) -> None:
        """Arg-conditional call_modes should drive actual call legality and produced capabilities."""
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(
            tools=[_make_tool("chunk_get_text"), _make_tool("extract_date_mentions")],
        ))
        mock_session.call_tool = AsyncMock(
            side_effect=[
                _make_tool_result('{"results":[{"chunk_id":"chunk_1","text":"Reached on August 3, 1769."}]}'),
                _make_tool_result('{"dates":[{"normalized_date":"August 3, 1769","chunk_id":"chunk_1"}]}'),
            ],
        )

        with (
            patch("llm_client.agent.mcp_agent._import_mcp") as mock_import,
            patch("llm_client.agent.mcp_agent._inner_acall_llm") as mock_acall,
        ):
            mock_stdio = AsyncMock()
            mock_stdio.__aenter__ = AsyncMock(return_value=("read", "write"))
            mock_stdio.__aexit__ = AsyncMock(return_value=False)
            mock_import.return_value = (
                MagicMock(return_value=mock_stdio),
                MagicMock,
                MagicMock(return_value=mock_session),
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_acall.side_effect = [
                _make_llm_result(tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "chunk_get_text",
                        "arguments": '{"mode":"by_chunk_id","chunk_ids":["chunk_1"],"tool_reasoning":"read the exact chunk"}',
                    },
                }]),
                _make_llm_result(tool_calls=[{
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "extract_date_mentions",
                        "arguments": '{"tool_reasoning":"extract date from the retrieved text"}',
                    },
                }]),
                _make_llm_result(content="answer"),
            ]

            from llm_client.agent.mcp_agent import _acall_with_mcp
            result = await _acall_with_mcp(
                "test-model",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
                enforce_tool_contracts=True,
                tool_contracts={
                    "chunk_get_text": {
                        "requires_all": ["CHUNK_SET"],
                        "call_modes": [
                            {
                                "name": "by_chunk_id",
                                "when_arg_equals": {"mode": "by_chunk_id"},
                                "artifact_prereqs": "none",
                                "produces": [{"kind": "CHUNK_SET", "ref_type": "fulltext"}],
                            }
                        ],
                    },
                    "extract_date_mentions": {
                        "requires_all": [{"kind": "CHUNK_SET", "ref_type": "fulltext"}],
                        "produces": [{"kind": "CHUNK_SET", "ref_type": "fulltext"}],
                    },
                },
                initial_artifacts=("QUERY_TEXT",),
            )

            assert result.content == "answer"
            assert mock_session.call_tool.call_count == 2
            assert result.raw_response.metadata["tool_contract_rejections"] == 0
            assert any(
                cap.get("kind") == "CHUNK_SET" and cap.get("ref_type") == "fulltext"
                for cap in result.raw_response.metadata["available_capabilities_final"]
            )

    async def test_enforce_tool_contracts_rejects_missing_capability_shape(self) -> None:
        """Capability requirements reject calls when kind exists but ref_type is incompatible."""
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(
            tools=[_make_tool("relationship_onehop")],
        ))
        mock_session.call_tool = AsyncMock(return_value=_make_tool_result("result"))

        with (
            patch("llm_client.agent.mcp_agent._import_mcp") as mock_import,
            patch("llm_client.agent.mcp_agent._inner_acall_llm") as mock_acall,
        ):
            mock_stdio = AsyncMock()
            mock_stdio.__aenter__ = AsyncMock(return_value=("read", "write"))
            mock_stdio.__aexit__ = AsyncMock(return_value=False)
            mock_import.return_value = (
                MagicMock(return_value=mock_stdio),
                MagicMock,
                MagicMock(return_value=mock_session),
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_acall.side_effect = [
                _make_llm_result(tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "relationship_onehop",
                        "arguments": '{"tool_reasoning":"expand entity"}',
                    },
                }]),
                _make_llm_result(content="capability mismatch handled"),
            ]

            from llm_client.agent.mcp_agent import _acall_with_mcp
            result = await _acall_with_mcp(
                "test-model",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
                enforce_tool_contracts=True,
                tool_contracts={
                    "relationship_onehop": {
                        "requires_all": [
                            {"kind": "ENTITY_SET", "ref_type": "id"},
                        ],
                        "produces": ["RELATIONSHIP_SET"],
                    },
                },
                initial_artifacts=("QUERY_TEXT", "ENTITY_SET"),
                initial_capabilities=[{"kind": "ENTITY_SET", "ref_type": "name"}],
            )

            assert result.content == "capability mismatch handled"
            assert result.raw_response.metadata["tool_contract_rejections"] == 1
            violations = result.raw_response.metadata["tool_contract_violation_events"]
            assert len(violations) == 1
            assert (
                violations[0]["error_code"]
                == "TOOL_VALIDATION_REJECTED_MISSING_CAPABILITY"
            )
            assert violations[0]["missing_requirements"] == [
                {"kind": "ENTITY_SET", "ref_type": "id"},
            ]
            assert result.raw_response.metadata["primary_failure_class"] == "composability"
            mock_session.call_tool.assert_not_called()

    async def test_no_tools_raises(self) -> None:
        """Error if MCP servers provide no tools."""
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=[]))

        with (
            patch("llm_client.agent.mcp_agent._import_mcp") as mock_import,
        ):
            mock_stdio = AsyncMock()
            mock_stdio.__aenter__ = AsyncMock(return_value=("read", "write"))
            mock_stdio.__aexit__ = AsyncMock(return_value=False)
            mock_import.return_value = (
                MagicMock(return_value=mock_stdio),
                MagicMock,
                MagicMock(return_value=mock_session),
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            from llm_client.agent.mcp_agent import _acall_with_mcp
            with pytest.raises(ValueError, match="No tools discovered"):
                await _acall_with_mcp(
                    "test-model",
                    [{"role": "user", "content": "Q"}],
                    mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
                )

    async def test_unknown_tool_handled(self) -> None:
        """Tool call for a tool not in any server is recorded as error."""
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(
            tools=[_make_tool("real_tool")],
        ))

        with (
            patch("llm_client.agent.mcp_agent._import_mcp") as mock_import,
            patch("llm_client.agent.mcp_agent._inner_acall_llm") as mock_acall,
        ):
            mock_stdio = AsyncMock()
            mock_stdio.__aenter__ = AsyncMock(return_value=("read", "write"))
            mock_stdio.__aexit__ = AsyncMock(return_value=False)
            mock_import.return_value = (
                MagicMock(return_value=mock_stdio),
                MagicMock,
                MagicMock(return_value=mock_session),
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_acall.side_effect = [
                _make_llm_result(tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "nonexistent_tool", "arguments": "{}"},
                }]),
                _make_llm_result(content="answer"),
            ]

            from llm_client.agent.mcp_agent import _acall_with_mcp
            result = await _acall_with_mcp(
                "test-model",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
            )

            assert result.content == "answer"
            assert len(result.raw_response.tool_calls) == 1
            assert result.raw_response.tool_calls[0].error == "Unknown tool: nonexistent_tool"

    async def test_mcp_tool_error_handled(self) -> None:
        """MCP tool returning isError=true is recorded as error."""
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(
            tools=[_make_tool("search")],
        ))
        mock_session.call_tool = AsyncMock(
            return_value=_make_tool_result("something failed", is_error=True),
        )

        with (
            patch("llm_client.agent.mcp_agent._import_mcp") as mock_import,
            patch("llm_client.agent.mcp_agent._inner_acall_llm") as mock_acall,
        ):
            mock_stdio = AsyncMock()
            mock_stdio.__aenter__ = AsyncMock(return_value=("read", "write"))
            mock_stdio.__aexit__ = AsyncMock(return_value=False)
            mock_import.return_value = (
                MagicMock(return_value=mock_stdio),
                MagicMock,
                MagicMock(return_value=mock_session),
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_acall.side_effect = [
                _make_llm_result(tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }]),
                _make_llm_result(content="answer"),
            ]

            from llm_client.agent.mcp_agent import _acall_with_mcp
            result = await _acall_with_mcp(
                "test-model",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
            )

            assert result.raw_response.tool_calls[0].error == "something failed"
            assert result.raw_response.tool_calls[0].result is None

    async def test_usage_accumulates(self) -> None:
        """Usage tokens accumulate across turns."""
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(
            tools=[_make_tool("search")],
        ))
        mock_session.call_tool = AsyncMock(
            return_value=_make_tool_result("data"),
        )

        with (
            patch("llm_client.agent.mcp_agent._import_mcp") as mock_import,
            patch("llm_client.agent.mcp_agent._inner_acall_llm") as mock_acall,
        ):
            mock_stdio = AsyncMock()
            mock_stdio.__aenter__ = AsyncMock(return_value=("read", "write"))
            mock_stdio.__aexit__ = AsyncMock(return_value=False)
            mock_import.return_value = (
                MagicMock(return_value=mock_stdio),
                MagicMock,
                MagicMock(return_value=mock_session),
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_acall.side_effect = [
                _make_llm_result(
                    tool_calls=[{
                        "id": "c1", "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }],
                    usage={"input_tokens": 100, "output_tokens": 20},
                    cost=0.01,
                ),
                _make_llm_result(
                    content="answer",
                    usage={"input_tokens": 200, "output_tokens": 30},
                    cost=0.02,
                ),
            ]

            from llm_client.agent.mcp_agent import _acall_with_mcp
            result = await _acall_with_mcp(
                "test-model",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
            )

            assert result.usage["input_tokens"] == 300
            assert result.usage["output_tokens"] == 50
            assert result.usage["total_tokens"] == 350
            assert result.usage["num_turns"] == 2
            assert result.usage["n_tool_calls"] == 1
            assert result.usage["n_budgeted_tool_calls"] == 1
            assert abs(result.cost - 0.03) < 0.001


# ---------------------------------------------------------------------------
# Routing through call_llm / acall_llm
# ---------------------------------------------------------------------------


class TestRouting:
    @pytest.mark.asyncio
    async def test_non_agent_with_mcp_routes_to_loop(self) -> None:
        """Non-agent model + mcp_servers → MCP agent loop."""
        with patch("llm_client.agent.mcp_agent._acall_with_mcp") as mock_loop:
            mock_loop.return_value = _make_llm_result(content="answer")

            result = await acall_llm(
                "gemini/gemini-3-flash-preview",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
                max_turns=5,
                max_tool_calls=7,
                task="test",
                trace_id="test_non_agent_mcp_routing",
                max_budget=0,
            )

            mock_loop.assert_called_once()
            call_kwargs = mock_loop.call_args
            assert call_kwargs.kwargs.get("mcp_servers") == {
                "srv": {"command": "python", "args": ["s.py"]},
            }
            assert call_kwargs.kwargs.get("max_turns") == 5
            assert call_kwargs.kwargs.get("max_tool_calls") == 7

    @pytest.mark.asyncio
    async def test_agent_model_with_mcp_skips_loop(self) -> None:
        """Agent model + mcp_servers → existing agent SDK path (not MCP loop)."""
        with (
            patch("llm_client.agent.mcp_agent._acall_with_mcp") as mock_loop,
            patch("llm_client.sdk.agents._route_acall") as mock_route,
        ):
            mock_route.return_value = _make_llm_result(content="agent answer")

            result = await acall_llm(
                "codex",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
                task="test",
                trace_id="test_agent_model_skips_loop",
                max_budget=0,
            )

            mock_loop.assert_not_called()
            mock_route.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_mcp_servers_normal_routing(self) -> None:
        """No mcp_servers → normal litellm routing."""
        with (
            patch("llm_client.agent.mcp_agent._acall_with_mcp") as mock_loop,
            patch("litellm.acompletion") as mock_completion,
        ):
            mock_msg = MagicMock()
            mock_msg.content = "hello"
            mock_msg.tool_calls = None
            mock_choice = MagicMock()
            mock_choice.message = mock_msg
            mock_choice.finish_reason = "stop"
            mock_resp = MagicMock()
            mock_resp.choices = [mock_choice]
            mock_resp.usage = MagicMock(
                prompt_tokens=10, completion_tokens=5, total_tokens=15,
            )
            mock_resp.model = "gemini/gemini-3-flash-preview"
            mock_completion.return_value = mock_resp

            result = await acall_llm(
                "gemini/gemini-3-flash-preview",
                [{"role": "user", "content": "Q"}],
                task="test",
                trace_id="test_no_mcp_normal_routing",
                max_budget=0,
            )

            mock_loop.assert_not_called()
            mock_completion.assert_called_once()

    def test_sync_call_llm_with_mcp(self) -> None:
        """Sync call_llm with mcp_servers routes to MCP loop."""
        with patch("llm_client.agent.mcp_agent._acall_with_mcp") as mock_loop:
            mock_loop.return_value = _make_llm_result(content="sync answer")

            result = call_llm(
                "gemini/gemini-3-flash-preview",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
                task="test",
                trace_id="test_sync_call_llm_with_mcp",
                max_budget=0,
            )

            # _run_sync wraps the async call
            mock_loop.assert_called_once()

    @pytest.mark.asyncio
    async def test_mcp_kwargs_popped_from_inner_calls(self) -> None:
        """MCP-specific kwargs don't leak to inner acall_llm."""
        with patch("llm_client.agent.mcp_agent._acall_with_mcp") as mock_loop:
            mock_loop.return_value = _make_llm_result(content="answer")
            from llm_client.agent.agent_planning import PlanningConfig
            planning_config = PlanningConfig(enabled=True)

            await acall_llm(
                "gemini/gemini-3-flash-preview",
                [{"role": "user", "content": "Q"}],
                mcp_servers={"srv": {"command": "python", "args": ["s.py"]}},
                max_turns=10,
                max_tool_calls=15,
                mcp_init_timeout=60.0,
                tool_result_max_length=10000,
                planning_config=planning_config,
                temperature=0.5,  # regular litellm kwarg
                task="test",
                trace_id="test_mcp_kwargs_popped",
                max_budget=0,
            )

            call_kwargs = mock_loop.call_args.kwargs
            # MCP kwargs present
            assert call_kwargs["mcp_servers"] == {"srv": {"command": "python", "args": ["s.py"]}}
            assert call_kwargs["max_turns"] == 10
            assert call_kwargs["max_tool_calls"] == 15
            assert call_kwargs["mcp_init_timeout"] == 60.0
            assert call_kwargs["tool_result_max_length"] == 10000
            assert call_kwargs["planning_config"] is planning_config
            # Regular kwargs also present (passed through)
            assert call_kwargs["temperature"] == 0.5


# ---------------------------------------------------------------------------
# Warnings, sticky fallback, models_used
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAgentDiagnostics:
    """Tests for warnings, models_used, and sticky fallback in agent loop."""

    async def test_sticky_fallback(self) -> None:
        """When inner call returns a different model (fallback), remaining turns use it."""
        from llm_client.agent.mcp_agent import MCPAgentResult, _agent_loop

        call_count = 0

        async def mock_executor(
            tool_calls: list[dict[str, Any]], max_len: int,
        ) -> tuple[list, list]:
            records = [MCPToolCallRecord(server="s", tool="t", arguments={}, result="ok")]
            msgs = [{"role": "tool", "tool_call_id": "tc1", "content": "result"}]
            return records, msgs

        async def mock_inner_acall(model, messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: tool call, but model fell back
                return _make_llm_result(
                    content="",
                    model="fallback-model",
                    tool_calls=[{
                        "id": "tc1",
                        "function": {"name": "t", "arguments": "{}"},
                    }],
                    finish_reason="tool_calls",
                    warnings=["FALLBACK: test-model -> fallback-model (Exception: error)"],
                )
            else:
                # Second call: final answer with sticky model
                return _make_llm_result(content="answer", model="fallback-model")

        agent_result = MCPAgentResult()

        with patch("llm_client.agent.mcp_agent._inner_acall_llm", side_effect=mock_inner_acall):
            content, finish = await _agent_loop(
                "test-model",
                [{"role": "user", "content": "q"}],
                [{"type": "function", "function": {"name": "t"}}],
                agent_result,
                mock_executor,
                5,
                None,
                False,
                50000,
                max_message_chars=60,
                timeout=60,
                kwargs={},
            )

        assert content == "answer"
        assert "fallback-model" in agent_result.models_used
        assert any("STICKY_FALLBACK" in w for w in agent_result.warnings)
        # Verify inner calls used the sticky model on turn 2
        assert call_count == 2

    async def test_warnings_propagated_from_turns(self) -> None:
        """Per-turn warnings from inner acall_llm accumulate in MCPAgentResult."""
        from llm_client.agent.mcp_agent import MCPAgentResult, _agent_loop

        async def mock_executor(tc, ml):
            records = [MCPToolCallRecord(server="s", tool="t", arguments={}, result="ok")]
            msgs = [{"role": "tool", "tool_call_id": "tc1", "content": "result"}]
            return records, msgs

        results = [
            _make_llm_result(
                content="",
                tool_calls=[{"id": "tc1", "function": {"name": "t", "arguments": "{}"}}],
                finish_reason="tool_calls",
                warnings=["RETRY 1/3: test-model (Exception: rate limit)"],
            ),
            _make_llm_result(content="done", warnings=["RETRY 1/3: test-model (Exception: timeout)"]),
        ]

        agent_result = MCPAgentResult()

        with patch("llm_client.agent.mcp_agent._inner_acall_llm", side_effect=results):
            await _agent_loop(
                "test-model",
                [{"role": "user", "content": "q"}],
                [{"type": "function", "function": {"name": "t"}}],
                agent_result,
                mock_executor,
                5,
                None,
                False,
                50000,
                max_message_chars=60,
                timeout=60,
                kwargs={},
            )

        assert len(agent_result.warnings) >= 2
        assert any("rate limit" in w for w in agent_result.warnings)
        assert any("timeout" in w for w in agent_result.warnings)

    async def test_models_used_tracked(self) -> None:
        """models_used set tracks all models that responded."""
        from llm_client.agent.mcp_agent import MCPAgentResult, _agent_loop

        result = _make_llm_result(content="ok", model="gemini/gemini-2.5-flash")
        agent_result = MCPAgentResult()

        with patch("llm_client.agent.mcp_agent._inner_acall_llm", return_value=result):
            await _agent_loop(
                "gemini/gemini-2.5-flash",
                [{"role": "user", "content": "q"}],
                [{"type": "function", "function": {"name": "t"}}],
                agent_result,
                AsyncMock(),
                5,
                None,
                False,
                50000,
                max_message_chars=60,
                timeout=60,
                kwargs={},
            )

        assert "gemini/gemini-2.5-flash" in agent_result.models_used

    async def test_progressive_tool_disclosure_filters_then_unlocks(self) -> None:
        """Only composable tools are exposed per turn; newly-produced artifacts unlock others."""
        from llm_client.agent.mcp_agent import MCPAgentResult, _agent_loop

        observed_tool_surfaces: list[list[str]] = []

        async def mock_inner_acall(model, messages, **kwargs):
            tools = kwargs.get("tools") or []
            observed_tool_surfaces.append([
                str(t.get("function", {}).get("name", ""))
                for t in tools
                if isinstance(t, dict)
            ])
            if len(observed_tool_surfaces) == 1:
                return _make_llm_result(
                    content="",
                    tool_calls=[{
                        "id": "tc_entity_seed",
                        "function": {
                            "name": "entity_tfidf",
                            "arguments": '{"tool_reasoning":"seed from query"}',
                        },
                    }],
                    finish_reason="tool_calls",
                )
            return _make_llm_result(content="done")

        async def mock_executor(tool_calls, max_len):
            return (
                [
                    MCPToolCallRecord(
                        server="srv",
                        tool="entity_tfidf",
                        arguments={"tool_reasoning": "seed from query"},
                        result='{"entities":["e1"]}',
                    ),
                ],
                [{
                    "role": "tool",
                    "tool_call_id": "tc_entity_seed",
                    "content": '{"entities":["e1"]}',
                }],
            )

        openai_tools = [
            {"type": "function", "function": {"name": "entity_tfidf"}},
            {"type": "function", "function": {"name": "entity_onehop"}},
            {"type": "function", "function": {"name": "todo_list"}},
        ]
        contracts = {
            "entity_tfidf": {
                "requires_all": ["QUERY_TEXT"],
                "produces": ["ENTITY_SET"],
            },
            "entity_onehop": {
                "requires_all": ["ENTITY_SET"],
                "produces": ["ENTITY_SET"],
            },
            "todo_list": {"is_control": True},
        }

        agent_result = MCPAgentResult()
        with patch("llm_client.agent.mcp_agent._inner_acall_llm", side_effect=mock_inner_acall):
            content, finish = await _agent_loop(
                "test-model",
                [{"role": "user", "content": "q"}],
                openai_tools,
                agent_result,
                mock_executor,
                3,
                None,
                False,
                50000,
                max_message_chars=60,
                enforce_tool_contracts=True,
                progressive_tool_disclosure=True,
                tool_contracts=contracts,
                initial_artifacts=("QUERY_TEXT",),
                timeout=60,
                kwargs={},
            )

        assert content == "done"
        assert finish == "stop"
        assert len(observed_tool_surfaces) == 2
        # Turn 1: entity_onehop is hidden because ENTITY_SET is not available yet.
        assert observed_tool_surfaces[0] == ["entity_tfidf", "todo_list"]
        # Turn 2: entity_tfidf produced ENTITY_SET, so entity_onehop is now visible.
        assert observed_tool_surfaces[1] == ["entity_tfidf", "entity_onehop", "todo_list"]
        assert agent_result.metadata["tool_disclosure_turns"] == 1
        assert agent_result.metadata["tool_disclosure_hidden_total"] == 1

    async def test_runtime_artifact_read_reopens_typed_artifact_handles(self) -> None:
        """Typed artifacts can be reopened explicitly by artifact_id without another external tool call."""
        from llm_client.agent.mcp_agent import MCPAgentResult, _agent_loop

        observed_tool_surfaces: list[list[str]] = []
        executor_call_count = 0

        async def mock_inner_acall(model, messages, **kwargs):
            tools = kwargs.get("tools") or []
            observed_tool_surfaces.append([
                str(t.get("function", {}).get("name", ""))
                for t in tools
                if isinstance(t, dict)
            ])
            if len(observed_tool_surfaces) == 1:
                return _make_llm_result(
                    content="",
                    tool_calls=[{
                        "id": "tc_search",
                        "function": {
                            "name": "chunk_text_search",
                            "arguments": '{"query":"q","tool_reasoning":"find the source chunk"}',
                        },
                    }],
                    finish_reason="tool_calls",
                )
            if len(observed_tool_surfaces) == 2:
                return _make_llm_result(
                    content="",
                    tool_calls=[{
                        "id": "tc_runtime_read",
                        "function": {
                            "name": "runtime_artifact_read",
                            "arguments": '{"artifact_ids":["art_chunk_1"],"tool_reasoning":"reopen the cleared typed artifact"}',
                        },
                    }],
                    finish_reason="tool_calls",
                )
            return _make_llm_result(content="done")

        async def mock_executor(tool_calls, max_len):
            nonlocal executor_call_count
            executor_call_count += 1
            payload = json.dumps(
                {
                    "artifact_id": "art_chunk_1",
                    "artifact_type": "CHUNK_SET",
                    "schema_version": "1.0.0",
                    "capabilities": [{"kind": "CHUNK_SET", "ref_type": "fulltext"}],
                    "payload": {
                        "chunk_id": "chunk_1",
                        "text": "Reached on August 3, 1769.",
                    },
                    "provenance": {
                        "evidence_refs": [{"chunk_id": "chunk_1"}],
                    },
                }
            )
            return (
                [
                    MCPToolCallRecord(
                        server="srv",
                        tool="chunk_text_search",
                        arguments={"query": "q"},
                        result=payload,
                    ),
                ],
                [{
                    "role": "tool",
                    "tool_call_id": "tc_search",
                    "content": payload,
                }],
            )

        openai_tools = [
            {"type": "function", "function": {"name": "chunk_text_search"}},
            {"type": "function", "function": {"name": "todo_list"}},
        ]
        contracts = {
            "chunk_text_search": {
                "requires_all": ["QUERY_TEXT"],
                "produces": [{"kind": "CHUNK_SET", "ref_type": "fulltext"}],
            },
            "todo_list": {"is_control": True},
        }

        agent_result = MCPAgentResult()
        with patch("llm_client.agent.mcp_agent._inner_acall_llm", side_effect=mock_inner_acall):
            content, finish = await _agent_loop(
                "test-model",
                [{"role": "user", "content": "q"}],
                openai_tools,
                agent_result,
                mock_executor,
                4,
                None,
                False,
                50000,
                max_message_chars=400,
                enforce_tool_contracts=True,
                progressive_tool_disclosure=True,
                tool_contracts=contracts,
                initial_artifacts=("QUERY_TEXT",),
                timeout=60,
                kwargs={
                    "tool_result_keep_recent": 0,
                    "tool_result_context_preview_chars": 80,
                },
            )

        assert content == "done"
        assert finish == "stop"
        assert executor_call_count == 1
        assert len(observed_tool_surfaces) == 3
        assert "runtime_artifact_read" not in observed_tool_surfaces[0]
        assert "runtime_artifact_read" in observed_tool_surfaces[1]
        assert agent_result.metadata["runtime_artifact_registry_size"] == 1
        assert agent_result.metadata["runtime_artifact_registry_ids"] == ["art_chunk_1"]
        runtime_records = [
            record for record in agent_result.tool_calls
            if record.tool == "runtime_artifact_read"
        ]
        assert runtime_records
        runtime_payload = json.loads(str(runtime_records[0].result or "{}"))
        assert runtime_payload["artifact_ids"] == ["art_chunk_1"]
        assert runtime_payload["artifacts"][0]["artifact_id"] == "art_chunk_1"
        assert agent_result.metadata["context_tool_results_cleared"] >= 1
        assert any(
            "use runtime_artifact_read with artifact_ids" in str(msg.get("content", ""))
            for msg in agent_result.conversation_trace
            if isinstance(msg, dict)
        )

    async def test_handle_input_contract_injects_resolved_artifacts_for_consumer_tool(self) -> None:
        """Declarative handle inputs can resolve artifact handles into injected executor args."""
        from llm_client.agent.mcp_agent import MCPAgentResult, _agent_loop

        executor_calls: list[dict[str, Any]] = []

        async def mock_inner_acall(model, messages, **kwargs):
            if len(executor_calls) == 0:
                return _make_llm_result(
                    content="",
                    tool_calls=[{
                        "id": "tc_search",
                        "function": {
                            "name": "chunk_text_search",
                            "arguments": '{"query":"q","tool_reasoning":"find a chunk"}',
                        },
                    }],
                    finish_reason="tool_calls",
                )
            if len(executor_calls) == 1:
                return _make_llm_result(
                    content="",
                    tool_calls=[{
                        "id": "tc_extract",
                        "function": {
                            "name": "extract_date_mentions_from_artifacts",
                            "arguments": '{"chunk_artifact_ids":["art_chunk_1"],"tool_reasoning":"extract date from the artifact handle"}',
                        },
                    }],
                    finish_reason="tool_calls",
                )
            return _make_llm_result(content="done")

        async def mock_executor(tool_calls, max_len):
            executor_calls.extend(tool_calls)
            tool_name = tool_calls[0]["function"]["name"]
            if tool_name == "chunk_text_search":
                payload = json.dumps(
                    {
                        "artifact_id": "art_chunk_1",
                        "artifact_type": "CHUNK_SET",
                        "schema_version": "1.0.0",
                        "capabilities": [{"kind": "CHUNK_SET", "ref_type": "fulltext"}],
                        "payload": {
                            "chunk_id": "chunk_1",
                            "text": "Reached on August 3, 1769.",
                        },
                    }
                )
                return (
                    [
                        MCPToolCallRecord(
                            server="srv",
                            tool="chunk_text_search",
                            arguments={"query": "q"},
                            result=payload,
                        ),
                    ],
                    [{
                        "role": "tool",
                        "tool_call_id": "tc_search",
                        "content": payload,
                    }],
                )

            injected_args = json.loads(tool_calls[0]["function"]["arguments"])
            assert injected_args["chunk_artifact_ids"] == ["art_chunk_1"]
            assert injected_args["chunk_artifacts"] == [
                {
                    "chunk_id": "chunk_1",
                    "text": "Reached on August 3, 1769.",
                }
            ]
            return (
                [
                    MCPToolCallRecord(
                        server="srv",
                        tool="extract_date_mentions_from_artifacts",
                        arguments=injected_args,
                        result='{"dates":[{"normalized_date":"August 3, 1769"}]}',
                    ),
                ],
                [{
                    "role": "tool",
                    "tool_call_id": "tc_extract",
                    "content": '{"dates":[{"normalized_date":"August 3, 1769"}]}',
                }],
            )

        openai_tools = [
            {"type": "function", "function": {"name": "chunk_text_search"}},
            {"type": "function", "function": {"name": "extract_date_mentions_from_artifacts"}},
        ]
        contracts = {
            "chunk_text_search": {
                "requires_all": ["QUERY_TEXT"],
                "produces": [{"kind": "CHUNK_SET", "ref_type": "fulltext"}],
            },
            "extract_date_mentions_from_artifacts": {
                "artifact_prereqs": "none",
                "handle_inputs": [
                    {
                        "arg": "chunk_artifact_ids",
                        "inject_arg": "chunk_artifacts",
                        "representation": "payload",
                        "accepts": [{"kind": "CHUNK_SET", "ref_type": "fulltext"}],
                    }
                ],
                "produces": [{"kind": "CHUNK_SET", "ref_type": "fulltext"}],
            },
        }

        agent_result = MCPAgentResult()
        with patch("llm_client.agent.mcp_agent._inner_acall_llm", side_effect=mock_inner_acall):
            content, finish = await _agent_loop(
                "test-model",
                [{"role": "user", "content": "q"}],
                openai_tools,
                agent_result,
                mock_executor,
                4,
                None,
                False,
                50000,
                max_message_chars=400,
                enforce_tool_contracts=True,
                progressive_tool_disclosure=True,
                tool_contracts=contracts,
                initial_artifacts=("QUERY_TEXT",),
                timeout=60,
                kwargs={},
            )

        assert content == "done"
        assert finish == "stop"
        assert len(executor_calls) == 2
        assert agent_result.metadata["handle_input_resolution_count"] >= 1
        assert agent_result.metadata["handle_input_resolved_artifact_count"] >= 1

    async def test_handle_input_contract_rejects_unknown_artifact_handle(self) -> None:
        """Handle-aware contracts should reject unknown artifact handles before executor runs."""
        from llm_client.agent.mcp_agent import MCPAgentResult, _agent_loop

        async def mock_inner_acall(model, messages, **kwargs):
            if not any(msg.get("role") == "tool" for msg in messages if isinstance(msg, dict)):
                return _make_llm_result(
                    content="",
                    tool_calls=[{
                        "id": "tc_extract",
                        "function": {
                            "name": "extract_date_mentions_from_artifacts",
                            "arguments": '{"chunk_artifact_ids":["art_missing"],"tool_reasoning":"try the missing handle"}',
                        },
                    }],
                    finish_reason="tool_calls",
                )
            return _make_llm_result(content="done")

        async def mock_executor(tool_calls, max_len):
            raise AssertionError("executor should not run when handle contract validation fails")

        openai_tools = [
            {"type": "function", "function": {"name": "extract_date_mentions_from_artifacts"}},
        ]
        contracts = {
            "extract_date_mentions_from_artifacts": {
                "artifact_prereqs": "none",
                "handle_inputs": [
                    {
                        "arg": "chunk_artifact_ids",
                        "inject_arg": "chunk_artifacts",
                        "representation": "payload",
                        "accepts": [{"kind": "CHUNK_SET", "ref_type": "fulltext"}],
                    }
                ],
            },
        }

        agent_result = MCPAgentResult()
        with patch("llm_client.agent.mcp_agent._inner_acall_llm", side_effect=mock_inner_acall):
            content, finish = await _agent_loop(
                "test-model",
                [{"role": "user", "content": "q"}],
                openai_tools,
                agent_result,
                mock_executor,
                2,
                None,
                False,
                50000,
                max_message_chars=200,
                enforce_tool_contracts=True,
                progressive_tool_disclosure=True,
                tool_contracts=contracts,
                initial_artifacts=("QUERY_TEXT",),
                timeout=60,
                kwargs={},
            )

        assert content == "done"
        assert finish == "stop"
        assert agent_result.metadata["tool_contract_rejections"] == 1
        violation = agent_result.metadata["tool_contract_violation_events"][0]
        assert violation["tool"] == "extract_date_mentions_from_artifacts"
        assert violation["error_code"] == "TOOL_VALIDATION_REJECTED_MISSING_PREREQUISITE"
        assert "requires known runtime artifact handles" in violation["reason"]

    async def test_progressive_disclosure_reports_bounded_unavailable_reasons(self) -> None:
        """Disclosure adds bounded unavailable reason guidance and tracks overhead metadata."""
        from llm_client.agent.mcp_agent import MCPAgentResult, _agent_loop

        observed_tools: list[list[str]] = []

        async def mock_inner_acall(model, messages, **kwargs):
            tools = kwargs.get("tools") or []
            observed_tools.append([
                str(t.get("function", {}).get("name", ""))
                for t in tools
                if isinstance(t, dict)
            ])
            return _make_llm_result(content="done")

        async def mock_executor(tool_calls, max_len):
            return [], []

        blocked_tools = [
            {"type": "function", "function": {"name": f"blocked_{i}"}}
            for i in range(12)
        ]
        openai_tools = blocked_tools + [
            {"type": "function", "function": {"name": "todo_list"}},
        ]
        contracts = {
            **{
                f"blocked_{i}": {
                    "requires_all": [
                        {"kind": "ENTITY_SET", "ref_type": "id"},
                    ],
                    "produces": ["CHUNK_SET"],
                }
                for i in range(12)
            },
            "todo_list": {"is_control": True},
        }

        agent_result = MCPAgentResult()
        with patch("llm_client.agent.mcp_agent._inner_acall_llm", side_effect=mock_inner_acall):
            content, finish = await _agent_loop(
                "test-model",
                [{"role": "user", "content": "q"}],
                openai_tools,
                agent_result,
                mock_executor,
                2,
                None,
                False,
                50000,
                max_message_chars=60,
                enforce_tool_contracts=True,
                progressive_tool_disclosure=True,
                tool_contracts=contracts,
                initial_artifacts=("QUERY_TEXT",),
                timeout=60,
                kwargs={},
            )

        assert content == "done"
        assert finish == "stop"
        assert observed_tools[0] == ["todo_list"]
        assert agent_result.metadata["tool_disclosure_turns"] == 1
        # Total hidden before cap is tracked.
        assert agent_result.metadata["tool_disclosure_hidden_total"] == 12
        # Bounded unavailable message overhead is tracked.
        assert agent_result.metadata["tool_disclosure_unavailable_msgs"] == 1
        assert agent_result.metadata["tool_disclosure_unavailable_reason_chars"] > 0
        assert agent_result.metadata["tool_disclosure_unavailable_reason_tokens_est"] > 0
        assert agent_result.metadata["no_legal_noncontrol_turns"] == 1
        assert any(
            "Currently unavailable tools with missing requirements" in str(msg.get("content", ""))
            for msg in agent_result.conversation_trace
            if msg.get("role") == "user"
        )

    async def test_progressive_disclosure_suggests_repair_tools(self) -> None:
        """Unavailable guidance includes legal conversion/repair tool hints."""
        from llm_client.agent.mcp_agent import MCPAgentResult, _agent_loop

        async def mock_inner_acall(model, messages, **kwargs):
            return _make_llm_result(content="done")

        async def mock_executor(tool_calls, max_len):
            return [], []

        openai_tools = [
            {"type": "function", "function": {"name": "entity_resolve_names_to_ids"}},
            {"type": "function", "function": {"name": "relationship_onehop"}},
            {"type": "function", "function": {"name": "todo_list"}},
        ]
        contracts = {
            "entity_resolve_names_to_ids": {
                "requires_all": ["QUERY_TEXT"],
                "produces": [{"kind": "ENTITY_SET", "ref_type": "id"}],
            },
            "relationship_onehop": {
                "requires_all": [{"kind": "ENTITY_SET", "ref_type": "id"}],
                "produces": ["RELATIONSHIP_SET"],
            },
            "todo_list": {"is_control": True},
        }

        agent_result = MCPAgentResult()
        with patch("llm_client.agent.mcp_agent._inner_acall_llm", side_effect=mock_inner_acall):
            content, finish = await _agent_loop(
                "test-model",
                [{"role": "user", "content": "q"}],
                openai_tools,
                agent_result,
                mock_executor,
                2,
                None,
                False,
                50000,
                max_message_chars=60,
                enforce_tool_contracts=True,
                progressive_tool_disclosure=True,
                tool_contracts=contracts,
                initial_artifacts=("QUERY_TEXT",),
                timeout=60,
                kwargs={},
            )

        assert content == "done"
        assert finish == "stop"
        assert agent_result.metadata["tool_disclosure_repair_suggestions"] > 0
        assert any(
            "try entity_resolve_names_to_ids" in str(msg.get("content", ""))
            for msg in agent_result.conversation_trace
            if msg.get("role") == "user"
        )
        closure = agent_result.metadata.get("lane_closure_analysis") or {}
        assert closure.get("lane_closed") is True

    async def test_context_tool_result_clearing_emits_metadata(self) -> None:
        """Older tool results are proactively cleared and surfaced in metadata counters."""
        from llm_client.agent.mcp_agent import MCPAgentResult, _agent_loop

        llm_results = [
            _make_llm_result(
                content="",
                tool_calls=[{
                    "id": "tc1",
                    "function": {"name": "chunk_text_search", "arguments": '{"query":"q1"}'},
                }],
                finish_reason="tool_calls",
            ),
            _make_llm_result(
                content="",
                tool_calls=[{
                    "id": "tc2",
                    "function": {"name": "chunk_text_search", "arguments": '{"query":"q2"}'},
                }],
                finish_reason="tool_calls",
            ),
            _make_llm_result(content="done", finish_reason="stop"),
        ]

        async def mock_executor(tool_calls, max_len):
            tc_id = tool_calls[0]["id"]
            payload = ("tool-result " + tc_id + " ") * 120
            return (
                [MCPToolCallRecord(server="srv", tool="chunk_text_search", arguments={}, result=payload)],
                [{"role": "tool", "tool_call_id": tc_id, "content": payload}],
            )

        agent_result = MCPAgentResult()
        with patch("llm_client.agent.mcp_agent._inner_acall_llm", side_effect=llm_results):
            content, finish = await _agent_loop(
                "test-model",
                [{"role": "user", "content": "q"}],
                [{"type": "function", "function": {"name": "chunk_text_search"}}],
                agent_result,
                mock_executor,
                5,
                None,
                False,
                50000,
                max_message_chars=500000,
                timeout=60,
                kwargs={
                    "tool_result_keep_recent": 1,
                    "tool_result_context_preview_chars": 80,
                },
            )

        assert content == "done"
        assert finish == "stop"
        assert agent_result.metadata["tool_result_keep_recent"] == 1
        assert agent_result.metadata["context_tool_results_cleared"] >= 1
        assert agent_result.metadata["context_tool_result_clearings"] >= 1
        assert agent_result.metadata["context_tool_result_cleared_chars"] > 0

    async def test_submit_answer_enforced_when_tool_available(self) -> None:
        """When submit_answer exists, plain-text response is nudged into explicit submission."""
        from llm_client.agent.mcp_agent import MCPAgentResult, _agent_loop

        llm_results = [
            _make_llm_result(content="June 1982"),
            _make_llm_result(
                content="",
                tool_calls=[{
                    "id": "tc_submit",
                    "function": {
                        "name": "submit_answer",
                        "arguments": '{"reasoning":"from chunk_1","answer":"June 1982"}',
                    },
                }],
                finish_reason="tool_calls",
            ),
        ]

        async def mock_executor(tc, ml):
            return (
                [
                    MCPToolCallRecord(
                        server="srv",
                        tool="submit_answer",
                        arguments={"reasoning": "from chunk_1", "answer": "June 1982"},
                        result='{"status":"submitted","answer":"June 1982"}',
                    ),
                ],
                [{
                    "role": "tool",
                    "tool_call_id": "tc_submit",
                    "content": '{"status":"submitted","answer":"June 1982"}',
                }],
            )

        agent_result = MCPAgentResult()
        with patch("llm_client.agent.mcp_agent._inner_acall_llm", side_effect=llm_results):
            content, finish = await _agent_loop(
                "test-model",
                [{"role": "user", "content": "q"}],
                [{"type": "function", "function": {"name": "submit_answer"}}],
                agent_result,
                mock_executor,
                5,
                None,
                False,
                50000,
                max_message_chars=60,
                timeout=60,
                kwargs={},
            )

        assert content == "June 1982"
        assert finish == "submitted"
        assert any(
            msg.get("role") == "user"
            and (
                "MUST call submit_answer" in msg.get("content", "")
                or "Do NOT finalize yet" in msg.get("content", "")
            )
            for msg in agent_result.conversation_trace
        )

    async def test_required_submit_not_attempted_is_policy_failure(self) -> None:
        """If submit_answer is available but never called, classify as required-submit policy failure."""
        from llm_client.agent.mcp_agent import MCPAgentResult, _agent_loop

        llm_results = [
            _make_llm_result(content="June 1982", finish_reason="stop"),
        ]

        async def mock_executor(tc, ml):
            return ([], [])

        agent_result = MCPAgentResult()
        with patch("llm_client.agent.mcp_agent._inner_acall_llm", side_effect=llm_results):
            content, finish = await _agent_loop(
                "test-model",
                [{"role": "user", "content": "q"}],
                [{"type": "function", "function": {"name": "submit_answer"}}],
                agent_result,
                mock_executor,
                1,
                None,
                False,
                50000,
                max_message_chars=60,
                timeout=60,
                kwargs={},
            )

        assert content == "June 1982"
        assert finish == "stop"
        assert agent_result.metadata["requires_submit_answer"] is True
        assert agent_result.metadata["submit_answer_call_count"] == 0
        assert agent_result.metadata["submit_answer_attempted"] is False
        assert agent_result.metadata["submit_answer_succeeded"] is False
        assert agent_result.metadata["required_submit_missing"] is True
        assert "REQUIRED_SUBMIT_NOT_ATTEMPTED" in agent_result.metadata["failure_event_codes"]
        assert agent_result.metadata["primary_failure_class"] == "policy"
        assert agent_result.metadata["first_terminal_failure_event_code"] == "REQUIRED_SUBMIT_NOT_ATTEMPTED"

    async def test_autofill_reasoning_for_todo_write(self) -> None:
        """todo_write missing tool_reasoning is auto-filled and executed in strict mode."""
        from llm_client.agent.mcp_agent import MCPAgentResult, _agent_loop

        observed_tool_calls: list[dict[str, Any]] = []

        async def mock_executor(tool_calls, max_len):
            observed_tool_calls.extend(tool_calls)
            return (
                [MCPToolCallRecord(server="srv", tool="todo_write", arguments={}, result='{"status":"ok"}')],
                [{"role": "tool", "tool_call_id": "tc_write", "content": '{"status":"ok"}'}],
            )

        llm_results = [
            _make_llm_result(
                content="",
                tool_calls=[{
                    "id": "tc_write",
                    "function": {"name": "todo_write", "arguments": "{}"},
                }],
                finish_reason="tool_calls",
            ),
            _make_llm_result(content="done"),
        ]

        agent_result = MCPAgentResult()
        with patch("llm_client.agent.mcp_agent._inner_acall_llm", side_effect=llm_results):
            content, finish = await _agent_loop(
                "test-model",
                [{"role": "user", "content": "q"}],
                [{"type": "function", "function": {"name": "todo_write"}}],
                agent_result,
                mock_executor,
                3,
                None,
                True,
                50000,
                max_message_chars=60,
                timeout=60,
                kwargs={},
            )

        assert content == "done"
        assert finish == "stop"
        assert observed_tool_calls
        args = json.loads(observed_tool_calls[0]["function"]["arguments"])
        assert "tool_reasoning" in args
        assert agent_result.metadata["rejected_missing_reasoning_calls"] == 0
        assert not any(
            "observability: missing tool_reasoning on tools:" in w.lower()
            for w in agent_result.warnings
        )

    async def test_retrieval_stagnation_forces_final_answer(self) -> None:
        """Repeated evidence calls with unchanged evidence digest trigger retrieval stagnation fuse."""
        from llm_client.agent.mcp_agent import MCPAgentResult, _agent_loop

        executor_call_count = 0

        async def mock_executor(tool_calls, max_len):
            nonlocal executor_call_count
            executor_call_count += 1
            return (
                [
                    MCPToolCallRecord(
                        server="srv",
                        tool="chunk_text_search",
                        arguments={"query_text": "same query"},
                        result='{"chunks":[{"chunk_id":"chunk_1","text":"same evidence","score":0.9}]}',
                    ),
                ],
                [
                    {
                        "role": "tool",
                        "tool_call_id": f"tc_search_{executor_call_count}",
                        "content": '{"chunks":[{"chunk_id":"chunk_1","text":"same evidence","score":0.9}]}',
                    }
                ],
            )

        llm_results = [
            _make_llm_result(
                content="",
                tool_calls=[{
                    "id": "tc_search_1",
                    "function": {
                        "name": "chunk_text_search",
                        "arguments": '{"query_text":"same query","tool_reasoning":"search 1"}',
                    },
                }],
                finish_reason="tool_calls",
            ),
            _make_llm_result(
                content="",
                tool_calls=[{
                    "id": "tc_search_2",
                    "function": {
                        "name": "chunk_text_search",
                        "arguments": '{"query_text":"same query","tool_reasoning":"search 2"}',
                    },
                }],
                finish_reason="tool_calls",
            ),
            _make_llm_result(
                content="",
                tool_calls=[{
                    "id": "tc_search_3",
                    "function": {
                        "name": "chunk_text_search",
                        "arguments": '{"query_text":"same query","tool_reasoning":"search 3"}',
                    },
                }],
                finish_reason="tool_calls",
            ),
            _make_llm_result(content="best effort answer"),
        ]

        agent_result = MCPAgentResult()
        with patch("llm_client.agent.mcp_agent._inner_acall_llm", side_effect=llm_results):
            content, finish = await _agent_loop(
                "test-model",
                [{"role": "user", "content": "q"}],
                [{"type": "function", "function": {"name": "chunk_text_search"}}],
                agent_result,
                mock_executor,
                10,
                None,
                False,
                50000,
                max_message_chars=60,
                timeout=60,
                kwargs={"retrieval_stagnation_turns": 2},
            )

        assert content == "best effort answer"
        assert finish == "stop"
        assert executor_call_count == 3
        assert agent_result.metadata["retrieval_stagnation_triggered"] is True
        failure_counts = agent_result.metadata["failure_event_code_counts"]
        assert failure_counts.get("RETRIEVAL_STAGNATION", 0) >= 1

    async def test_retrieval_stagnation_observe_does_not_force_final(self) -> None:
        """Observe mode logs stagnation but allows the loop to continue."""
        from llm_client.agent.mcp_agent import MCPAgentResult, _agent_loop

        executor_call_count = 0

        async def mock_executor(tool_calls, max_len):
            nonlocal executor_call_count
            executor_call_count += 1
            return (
                [
                    MCPToolCallRecord(
                        server="srv",
                        tool="chunk_text_search",
                        arguments={"query_text": "same query"},
                        result='{"chunks":[{"chunk_id":"chunk_1","text":"same evidence","score":0.9}]}',
                    ),
                ],
                [
                    {
                        "role": "tool",
                        "tool_call_id": f"tc_search_{executor_call_count}",
                        "content": '{"chunks":[{"chunk_id":"chunk_1","text":"same evidence","score":0.9}]}',
                    }
                ],
            )

        llm_results = [
            _make_llm_result(
                content="",
                tool_calls=[{
                    "id": "tc_search_1",
                    "function": {
                        "name": "chunk_text_search",
                        "arguments": '{"query_text":"same query","tool_reasoning":"search 1"}',
                    },
                }],
                finish_reason="tool_calls",
            ),
            _make_llm_result(
                content="",
                tool_calls=[{
                    "id": "tc_search_2",
                    "function": {
                        "name": "chunk_text_search",
                        "arguments": '{"query_text":"same query","tool_reasoning":"search 2"}',
                    },
                }],
                finish_reason="tool_calls",
            ),
            _make_llm_result(
                content="",
                tool_calls=[{
                    "id": "tc_search_3",
                    "function": {
                        "name": "chunk_text_search",
                        "arguments": '{"query_text":"same query","tool_reasoning":"search 3"}',
                    },
                }],
                finish_reason="tool_calls",
            ),
            _make_llm_result(
                content="",
                tool_calls=[{
                    "id": "tc_search_4",
                    "function": {
                        "name": "chunk_text_search",
                        "arguments": '{"query_text":"same query","tool_reasoning":"search 4"}',
                    },
                }],
                finish_reason="tool_calls",
            ),
            _make_llm_result(content="best effort answer"),
        ]

        agent_result = MCPAgentResult()
        with patch("llm_client.agent.mcp_agent._inner_acall_llm", side_effect=llm_results):
            content, finish = await _agent_loop(
                "test-model",
                [{"role": "user", "content": "q"}],
                [{"type": "function", "function": {"name": "chunk_text_search"}}],
                agent_result,
                mock_executor,
                10,
                None,
                False,
                50000,
                max_message_chars=60,
                timeout=60,
                kwargs={
                    "retrieval_stagnation_turns": 2,
                    "retrieval_stagnation_action": "observe",
                },
            )

        assert content == "best effort answer"
        assert finish == "stop"
        assert executor_call_count == 4
        assert agent_result.metadata["retrieval_stagnation_triggered"] is True
        assert agent_result.metadata["retrieval_stagnation_action"] == "observe"
        assert agent_result.metadata["first_terminal_failure_event_code"] is None
        failure_counts = agent_result.metadata["failure_event_code_counts"]
        assert failure_counts.get("RETRIEVAL_STAGNATION_OBSERVED", 0) >= 1

    async def test_submit_retry_requires_new_evidence_signal(self) -> None:
        """If submit validation requires new evidence, suppress resubmit until evidence tool succeeds."""
        from llm_client.agent.mcp_agent import MCPAgentResult, _agent_loop

        executor_call_count = 0

        async def mock_executor(tool_calls, max_len):
            nonlocal executor_call_count
            executor_call_count += 1
            tool_name = tool_calls[0]["function"]["name"]
            if tool_name == "submit_answer" and executor_call_count == 1:
                return (
                    [
                        MCPToolCallRecord(
                            server="srv",
                            tool="submit_answer",
                            arguments={"reasoning": "r1", "answer": "2000"},
                            result='{"status":"needs_revision","recovery_policy":{"new_evidence_required_before_retry":true},"validation_error":{"reason_code":"answer_not_grounded","message":"need fresh evidence"}}',
                        ),
                    ],
                    [
                        {
                            "role": "tool",
                            "tool_call_id": "tc_submit_1",
                            "content": '{"status":"needs_revision","recovery_policy":{"new_evidence_required_before_retry":true},"validation_error":{"reason_code":"answer_not_grounded","message":"need fresh evidence"}}',
                        }
                    ],
                )
            if tool_name == "chunk_text_search":
                return (
                    [
                        MCPToolCallRecord(
                            server="srv",
                            tool="chunk_text_search",
                            arguments={"query_text": "fresh evidence"},
                            result='{"results":[{"chunk_id":"chunk_1"}]}',
                        ),
                    ],
                    [
                        {
                            "role": "tool",
                            "tool_call_id": "tc_search",
                            "content": '{"results":[{"chunk_id":"chunk_1"}]}',
                        }
                    ],
                )
            return (
                [
                    MCPToolCallRecord(
                        server="srv",
                        tool="submit_answer",
                        arguments={"reasoning": "r2", "answer": "August 3, 1769"},
                        result='{"status":"submitted","answer":"August 3, 1769"}',
                    ),
                ],
                [
                    {
                        "role": "tool",
                        "tool_call_id": "tc_submit_3",
                        "content": '{"status":"submitted","answer":"August 3, 1769"}',
                    }
                ],
            )

        llm_results = [
            _make_llm_result(
                content="",
                tool_calls=[{
                    "id": "tc_submit_1",
                    "function": {
                        "name": "submit_answer",
                        "arguments": '{"reasoning":"r1","answer":"2000","tool_reasoning":"first submit"}',
                    },
                }],
                finish_reason="tool_calls",
            ),
            _make_llm_result(
                content="",
                tool_calls=[{
                    "id": "tc_submit_2",
                    "function": {
                        "name": "submit_answer",
                        "arguments": '{"reasoning":"r1","answer":"2000","tool_reasoning":"retry too early"}',
                    },
                }],
                finish_reason="tool_calls",
            ),
            _make_llm_result(
                content="",
                tool_calls=[{
                    "id": "tc_search",
                    "function": {
                        "name": "chunk_text_search",
                        "arguments": '{"query_text":"fresh evidence","tool_reasoning":"gather new evidence"}',
                    },
                }],
                finish_reason="tool_calls",
            ),
            _make_llm_result(
                content="",
                tool_calls=[{
                    "id": "tc_submit_3",
                    "function": {
                        "name": "submit_answer",
                        "arguments": '{"reasoning":"r2","answer":"August 3, 1769","tool_reasoning":"submit with new evidence"}',
                    },
                }],
                finish_reason="tool_calls",
            ),
        ]

        agent_result = MCPAgentResult()
        with patch("llm_client.agent.mcp_agent._inner_acall_llm", side_effect=llm_results):
            content, finish = await _agent_loop(
                "test-model",
                [{"role": "user", "content": "q"}],
                [
                    {"type": "function", "function": {"name": "submit_answer"}},
                    {"type": "function", "function": {"name": "chunk_text_search"}},
                ],
                agent_result,
                mock_executor,
                8,
                None,
                False,
                50000,
                max_message_chars=60,
                suppress_control_loop_calls=True,
                timeout=60,
                kwargs={},
            )

        assert content == "August 3, 1769"
        assert finish == "submitted"
        # First submit + evidence search + final submit executed; middle submit suppressed.
        assert executor_call_count == 3
        assert agent_result.metadata["control_loop_suppressed_calls"] >= 1
        assert any(
            record.tool == "submit_answer" and "requires NEW evidence" in (record.error or "")
            for record in agent_result.tool_calls
        )
        assert isinstance(agent_result.metadata.get("evidence_digest"), str)
        # Successful evidence call should clear pending digest gate.
        assert agent_result.metadata.get("submit_evidence_digest_at_last_failure") is None

    async def test_repeated_submit_rejections_can_force_final_early(self) -> None:
        """Repeated validator rejections that require forced-terminal submission should stop churn early."""
        from llm_client.agent.mcp_agent import MCPAgentResult, _agent_loop

        executor_call_names: list[str] = []
        search_count = 0

        async def mock_executor(tool_calls, max_len):
            nonlocal search_count
            tool_name = tool_calls[0]["function"]["name"]
            executor_call_names.append(tool_name)
            if tool_name == "chunk_text_search":
                search_count += 1
                chunk_id = f"chunk_{search_count}"
                return (
                    [
                        MCPToolCallRecord(
                            server="srv",
                            tool="chunk_text_search",
                            arguments={"query_text": f"evidence {search_count}"},
                            result=json.dumps({"results": [{"chunk_id": chunk_id}]}),
                        ),
                    ],
                    [
                        {
                            "role": "tool",
                            "tool_call_id": f"tc_search_{search_count}",
                            "content": json.dumps({"results": [{"chunk_id": chunk_id}]}),
                        }
                    ],
                )
            return (
                [
                    MCPToolCallRecord(
                        server="srv",
                        tool="submit_answer",
                        arguments={"reasoning": "grounded", "answer": "12"},
                        result=json.dumps(
                            {
                                "status": "rejected",
                                "pending_atoms": 1,
                                "pending_ids": ["a2"],
                                "validation_error": {
                                    "reason_code": "pending_atoms",
                                    "message": "atom a2 still unresolved",
                                },
                                "recovery_policy": {
                                    "new_evidence_required_before_retry": True,
                                    "requires_forced_terminal_path": True,
                                },
                            }
                        ),
                    ),
                ],
                [
                    {
                        "role": "tool",
                        "tool_call_id": f"tc_submit_{len([n for n in executor_call_names if n == 'submit_answer'])}",
                        "content": json.dumps(
                            {
                                "status": "rejected",
                                "pending_atoms": 1,
                                "pending_ids": ["a2"],
                                "validation_error": {
                                    "reason_code": "pending_atoms",
                                    "message": "atom a2 still unresolved",
                                },
                                "recovery_policy": {
                                    "new_evidence_required_before_retry": True,
                                    "requires_forced_terminal_path": True,
                                },
                            }
                        ),
                    }
                ],
            )

        llm_results = [
            _make_llm_result(
                content="",
                tool_calls=[{
                    "id": "tc_search_1",
                    "function": {
                        "name": "chunk_text_search",
                        "arguments": '{"query_text":"evidence 1","tool_reasoning":"get initial evidence"}',
                    },
                }],
                finish_reason="tool_calls",
            ),
            _make_llm_result(
                content="",
                tool_calls=[{
                    "id": "tc_submit_1",
                    "function": {
                        "name": "submit_answer",
                        "arguments": '{"reasoning":"grounded","answer":"12","tool_reasoning":"first submit"}',
                    },
                }],
                finish_reason="tool_calls",
            ),
            _make_llm_result(
                content="",
                tool_calls=[{
                    "id": "tc_search_2",
                    "function": {
                        "name": "chunk_text_search",
                        "arguments": '{"query_text":"evidence 2","tool_reasoning":"fresh evidence retry"}',
                    },
                }],
                finish_reason="tool_calls",
            ),
            _make_llm_result(
                content="",
                tool_calls=[{
                    "id": "tc_submit_2",
                    "function": {
                        "name": "submit_answer",
                        "arguments": '{"reasoning":"grounded","answer":"12","tool_reasoning":"second submit"}',
                    },
                }],
                finish_reason="tool_calls",
            ),
            _make_llm_result(content="12", finish_reason="stop"),
        ]

        agent_result = MCPAgentResult()
        with patch("llm_client.agent.mcp_agent._inner_acall_llm", side_effect=llm_results):
            content, finish = await _agent_loop(
                "test-model",
                [{"role": "user", "content": "q"}],
                [
                    {"type": "function", "function": {"name": "submit_answer"}},
                    {"type": "function", "function": {"name": "chunk_text_search"}},
                ],
                agent_result,
                mock_executor,
                10,
                None,
                False,
                50000,
                max_message_chars=60,
                suppress_control_loop_calls=True,
                timeout=60,
                kwargs={},
            )

        assert content == "12"
        assert finish == "stop"
        assert executor_call_names == [
            "chunk_text_search",
            "submit_answer",
            "chunk_text_search",
            "submit_answer",
        ]
        assert agent_result.metadata["submit_validation_reason_counts"]["pending_atoms"] == 2
        assert "CONTROL_CHURN_THRESHOLD_EXCEEDED" in agent_result.metadata["failure_event_codes"]
        assert "SUBMIT_FORCED_ACCEPT_FORCED_FINAL" in agent_result.metadata["failure_event_codes"]
