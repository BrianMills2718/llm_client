"""Context budget tracking for agent sessions.

Provides visibility into how much of the context window is consumed by
tool definitions, conversation history, system prompts, and remaining
capacity.  Enables agents to make informed decisions about when to compact,
use subagents, or write scripts instead of sequential tool calls.
"""

from __future__ import annotations

import json as _json
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTEXT_BUDGET_TOOL_NAME = "check_context_budget"
"""Synthetic tool name for on-demand context budget inspection."""

_CHARS_PER_TOKEN_ESTIMATE = 4
"""Heuristic: 1 token ~ 4 characters.  Intentionally conservative."""


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextBudget:
    """Snapshot of context window utilization.

    All token counts are estimates — exact counts depend on the tokenizer
    which varies by model.  Estimates use the 1 token ~ 4 chars heuristic
    unless actual token counts are available from the API.
    """

    model: str
    max_context_tokens: int
    tool_definition_tokens: int
    system_prompt_tokens: int
    conversation_tokens: int

    @property
    def used_tokens(self) -> int:
        """Total tokens currently consumed."""
        return (
            self.tool_definition_tokens
            + self.system_prompt_tokens
            + self.conversation_tokens
        )

    @property
    def remaining_tokens(self) -> int:
        """Tokens available for new content."""
        return max(0, self.max_context_tokens - self.used_tokens)

    @property
    def utilization(self) -> float:
        """Fraction of context window consumed (0.0 to 1.0)."""
        if self.max_context_tokens <= 0:
            return 0.0
        return min(1.0, self.used_tokens / self.max_context_tokens)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for tool output or metadata."""
        return {
            "model": self.model,
            "max_context_tokens": self.max_context_tokens,
            "tool_definition_tokens": self.tool_definition_tokens,
            "system_prompt_tokens": self.system_prompt_tokens,
            "conversation_tokens": self.conversation_tokens,
            "used_tokens": self.used_tokens,
            "remaining_tokens": self.remaining_tokens,
            "utilization": round(self.utilization, 3),
        }


# ---------------------------------------------------------------------------
# Estimation helpers
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Estimate token count from text using the 4-chars-per-token heuristic.

    This is a rough approximation.  Real token counts vary by tokenizer
    and language.  For English prose, this tends to overcount slightly,
    which is the safe direction for budget tracking.
    """
    return max(1, len(text) // _CHARS_PER_TOKEN_ESTIMATE)


def estimate_tool_definition_tokens(openai_tools: list[dict[str, Any]]) -> int:
    """Estimate tokens consumed by tool definitions in OpenAI format.

    Serializes the full tool schema list to JSON and applies the
    character heuristic.  This captures parameter schemas and descriptions.
    """
    if not openai_tools:
        return 0
    return estimate_tokens(_json.dumps(openai_tools))


def estimate_conversation_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate tokens consumed by conversation history.

    Serializes all messages to JSON and applies the character heuristic.
    Includes role tokens, content, tool call payloads, and metadata.
    """
    if not messages:
        return 0
    return estimate_tokens(_json.dumps(messages))


# ---------------------------------------------------------------------------
# Model context limits
# ---------------------------------------------------------------------------


def get_model_context_limit(model: str) -> int:
    """Return the context window size for a model.

    Returns conservative estimates.  For unknown models, returns 200_000
    as a safe default since most current models support at least that.

    These limits will go stale as providers update models — callers can
    override via the ``max_context_tokens`` parameter on
    ``compute_context_budget``.
    """
    model_lower = model.lower()
    # 1M context models
    if any(
        k in model_lower
        for k in ("opus-4", "sonnet-4", "gpt-5", "gemini-2.5")
    ):
        return 1_000_000
    # 200K models
    if any(k in model_lower for k in ("claude", "haiku")):
        return 200_000
    # 128K models
    if any(k in model_lower for k in ("gpt-4", "o1", "o3", "o4")):
        return 128_000
    return 200_000


# ---------------------------------------------------------------------------
# Budget computation
# ---------------------------------------------------------------------------


def compute_context_budget(
    *,
    model: str,
    openai_tools: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    system_prompt: str = "",
    max_context_tokens: int | None = None,
) -> ContextBudget:
    """Compute a context budget snapshot from current agent state.

    Args:
        model: Model identifier (used for context limit lookup).
        openai_tools: Current tool definitions in OpenAI format.
        messages: Current conversation history.
        system_prompt: Extracted system prompt text.  If empty, the system
            prompt tokens are estimated from the first system message in
            ``messages`` (which means they also count toward conversation
            tokens — a small double-count that errs on the conservative side).
        max_context_tokens: Override model context limit (for testing or
            when the caller knows the exact limit).
    """
    return ContextBudget(
        model=model,
        max_context_tokens=max_context_tokens or get_model_context_limit(model),
        tool_definition_tokens=estimate_tool_definition_tokens(openai_tools),
        system_prompt_tokens=estimate_tokens(system_prompt) if system_prompt else 0,
        conversation_tokens=estimate_conversation_tokens(messages),
    )


# ---------------------------------------------------------------------------
# Synthetic tool definition
# ---------------------------------------------------------------------------


def context_budget_tool_definition() -> dict[str, Any]:
    """Return the OpenAI-format schema for the synthetic context budget tool.

    This tool takes no arguments — the agent simply calls it to receive
    a snapshot of context window utilization.
    """
    return {
        "type": "function",
        "function": {
            "name": CONTEXT_BUDGET_TOOL_NAME,
            "description": (
                "Check how much of the context window is used. "
                "Call this to decide whether to compact, use subagents, "
                "or write scripts instead of sequential calls."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }


def execute_context_budget_tool(
    *,
    model: str,
    openai_tools: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    system_prompt: str = "",
    max_context_tokens: int | None = None,
) -> str:
    """Execute the synthetic context budget tool and return JSON result text.

    Called by the agent loop when the agent invokes ``check_context_budget``.
    Returns a JSON string suitable for a tool-result message.
    """
    budget = compute_context_budget(
        model=model,
        openai_tools=openai_tools,
        messages=messages,
        system_prompt=system_prompt,
        max_context_tokens=max_context_tokens,
    )
    return _json.dumps(budget.to_dict(), indent=2)
