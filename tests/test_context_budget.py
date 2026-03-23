"""Tests for context budget visibility (Plan 02, Slice E).

Tests the ContextBudget dataclass, token estimation heuristics, model
context limit lookup, budget computation, serialization, and the synthetic
check_context_budget tool definition.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from llm_client.context_budget import (
    CONTEXT_BUDGET_TOOL_NAME,
    ContextBudget,
    compute_context_budget,
    context_budget_tool_definition,
    estimate_conversation_tokens,
    estimate_tokens,
    estimate_tool_definition_tokens,
    execute_context_budget_tool,
    get_model_context_limit,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(name: str, description: str = "") -> dict[str, Any]:
    """Build a minimal OpenAI-format tool definition for testing."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description or f"Tool {name}",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    }


def _make_messages(count: int) -> list[dict[str, Any]]:
    """Build a list of alternating user/assistant messages."""
    messages: list[dict[str, Any]] = []
    for i in range(count):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({
            "role": role,
            "content": f"Message number {i} with some content to estimate tokens from.",
        })
    return messages


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    """Token estimation heuristic tests."""

    def test_estimate_tokens_known_text(self) -> None:
        """Known text produces a reasonable estimate (4 chars/token)."""
        text = "Hello, world!"  # 13 chars -> 3 tokens
        result = estimate_tokens(text)
        assert result == 3

    def test_estimate_tokens_empty_string(self) -> None:
        """Empty string returns 1 (minimum)."""
        assert estimate_tokens("") == 1

    def test_estimate_tokens_short_string(self) -> None:
        """String shorter than 4 chars returns 1."""
        assert estimate_tokens("Hi") == 1

    def test_estimate_tokens_exact_multiple(self) -> None:
        """String that is exact multiple of 4 chars."""
        text = "abcd" * 10  # 40 chars -> 10 tokens
        assert estimate_tokens(text) == 10


class TestEstimateToolTokens:
    """Tool definition token estimation tests."""

    def test_estimate_tool_tokens_five_tools(self) -> None:
        """List of 5 tool definitions produces a positive token count."""
        tools = [_make_tool(f"tool_{i}") for i in range(5)]
        result = estimate_tool_definition_tokens(tools)
        assert result > 0

    def test_estimate_tool_tokens_empty_list(self) -> None:
        """Empty tool list returns 0."""
        assert estimate_tool_definition_tokens([]) == 0

    def test_estimate_tool_tokens_scales_with_count(self) -> None:
        """More tools produce higher token estimates."""
        small = [_make_tool(f"tool_{i}") for i in range(2)]
        large = [_make_tool(f"tool_{i}") for i in range(20)]
        assert estimate_tool_definition_tokens(large) > estimate_tool_definition_tokens(small)


class TestEstimateConversationTokens:
    """Conversation token estimation tests."""

    def test_estimate_conversation_tokens_ten_messages(self) -> None:
        """10-message conversation produces a positive count."""
        messages = _make_messages(10)
        result = estimate_conversation_tokens(messages)
        assert result > 0

    def test_estimate_conversation_tokens_empty(self) -> None:
        """Empty message list returns 0."""
        assert estimate_conversation_tokens([]) == 0


class TestModelContextLimits:
    """Model context limit lookup tests."""

    def test_opus_4_is_1m(self) -> None:
        """claude-opus-4.6 returns 1M context."""
        assert get_model_context_limit("claude-opus-4.6") == 1_000_000

    def test_sonnet_4_is_1m(self) -> None:
        """claude-sonnet-4 returns 1M context."""
        assert get_model_context_limit("anthropic/claude-sonnet-4-20250514") == 1_000_000

    def test_gpt5_is_1m(self) -> None:
        """gpt-5 returns 1M context."""
        assert get_model_context_limit("gpt-5.2") == 1_000_000

    def test_gemini_25_is_1m(self) -> None:
        """gemini-2.5 returns 1M context."""
        assert get_model_context_limit("gemini/gemini-2.5-flash") == 1_000_000

    def test_gpt4_is_128k(self) -> None:
        """gpt-4 returns 128K context."""
        assert get_model_context_limit("gpt-4o") == 128_000

    def test_unknown_model_is_200k(self) -> None:
        """Unknown model returns 200K default."""
        assert get_model_context_limit("unknown/mystery-model-7b") == 200_000


class TestComputeBudget:
    """Budget computation integration tests."""

    def test_compute_budget_basic(self) -> None:
        """Compute budget with known inputs and verify all fields."""
        tools = [_make_tool(f"tool_{i}") for i in range(3)]
        messages = _make_messages(5)
        budget = compute_context_budget(
            model="gpt-4o",
            openai_tools=tools,
            messages=messages,
            system_prompt="You are a helpful assistant.",
            max_context_tokens=128_000,
        )
        assert budget.model == "gpt-4o"
        assert budget.max_context_tokens == 128_000
        assert budget.tool_definition_tokens > 0
        assert budget.system_prompt_tokens > 0
        assert budget.conversation_tokens > 0
        assert budget.used_tokens == (
            budget.tool_definition_tokens
            + budget.system_prompt_tokens
            + budget.conversation_tokens
        )
        assert budget.remaining_tokens == budget.max_context_tokens - budget.used_tokens
        assert 0.0 < budget.utilization < 1.0

    def test_utilization_calculation(self) -> None:
        """Budget with ~50% used returns utilization near 0.5."""
        # Create a budget where we control the exact values
        budget = ContextBudget(
            model="test",
            max_context_tokens=1000,
            tool_definition_tokens=200,
            system_prompt_tokens=100,
            conversation_tokens=200,
        )
        assert budget.used_tokens == 500
        assert budget.utilization == pytest.approx(0.5)
        assert budget.remaining_tokens == 500

    def test_remaining_never_negative(self) -> None:
        """Budget with more used than available returns remaining=0."""
        budget = ContextBudget(
            model="test",
            max_context_tokens=100,
            tool_definition_tokens=50,
            system_prompt_tokens=30,
            conversation_tokens=50,
        )
        assert budget.used_tokens == 130  # exceeds max
        assert budget.remaining_tokens == 0
        assert budget.utilization == 1.0  # capped at 1.0

    def test_budget_to_dict(self) -> None:
        """Serialize and verify all keys present."""
        budget = ContextBudget(
            model="test-model",
            max_context_tokens=200_000,
            tool_definition_tokens=5000,
            system_prompt_tokens=1000,
            conversation_tokens=3000,
        )
        d = budget.to_dict()
        expected_keys = {
            "model",
            "max_context_tokens",
            "tool_definition_tokens",
            "system_prompt_tokens",
            "conversation_tokens",
            "used_tokens",
            "remaining_tokens",
            "utilization",
        }
        assert set(d.keys()) == expected_keys
        assert d["model"] == "test-model"
        assert d["used_tokens"] == 9000
        assert d["remaining_tokens"] == 191_000
        assert d["utilization"] == pytest.approx(0.045, abs=0.001)

    def test_empty_tools_and_messages(self) -> None:
        """Empty state produces low utilization."""
        budget = compute_context_budget(
            model="claude-opus-4.6",
            openai_tools=[],
            messages=[],
            system_prompt="",
        )
        assert budget.max_context_tokens == 1_000_000
        assert budget.tool_definition_tokens == 0
        assert budget.system_prompt_tokens == 0
        assert budget.conversation_tokens == 0
        assert budget.used_tokens == 0
        assert budget.remaining_tokens == 1_000_000
        assert budget.utilization == 0.0

    def test_override_max_context(self) -> None:
        """Custom max_context_tokens overrides model lookup."""
        budget = compute_context_budget(
            model="claude-opus-4.6",  # would normally be 1M
            openai_tools=[],
            messages=[],
            max_context_tokens=50_000,
        )
        assert budget.max_context_tokens == 50_000

    def test_zero_max_context_utilization(self) -> None:
        """Zero max_context_tokens avoids division by zero."""
        budget = ContextBudget(
            model="test",
            max_context_tokens=0,
            tool_definition_tokens=100,
            system_prompt_tokens=0,
            conversation_tokens=0,
        )
        assert budget.utilization == 0.0


class TestSyntheticTool:
    """Synthetic check_context_budget tool tests."""

    def test_tool_definition_schema(self) -> None:
        """Tool definition has correct name and empty parameters."""
        defn = context_budget_tool_definition()
        assert defn["type"] == "function"
        fn = defn["function"]
        assert fn["name"] == CONTEXT_BUDGET_TOOL_NAME
        assert fn["parameters"]["properties"] == {}
        assert fn["parameters"]["required"] == []
        assert fn["description"]  # non-empty

    def test_execute_returns_valid_json(self) -> None:
        """Execute tool returns parseable JSON with budget fields."""
        result_text = execute_context_budget_tool(
            model="gpt-4o",
            openai_tools=[_make_tool("search")],
            messages=[{"role": "user", "content": "hello"}],
            system_prompt="You are helpful.",
            max_context_tokens=128_000,
        )
        data = json.loads(result_text)
        assert data["model"] == "gpt-4o"
        assert data["max_context_tokens"] == 128_000
        assert data["used_tokens"] > 0
        assert data["remaining_tokens"] < 128_000
        assert 0.0 <= data["utilization"] < 1.0
