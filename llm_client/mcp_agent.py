"""MCP agent loop for llm_client.

Enables any litellm model to act as a tool-calling agent by connecting
to MCP servers, discovering their tools, and running an autonomous
tool-calling loop.

Usage (via call_llm/acall_llm — routing is automatic):
    result = await acall_llm(
        "gemini/gemini-3-flash-preview",
        messages,
        mcp_servers={
            "my-server": {
                "command": "python",
                "args": ["-u", "server.py"],
                "env": {"KEY": "value"},
            }
        },
        max_turns=20,
    )

The loop:
    1. Start MCP server subprocesses (stdio transport)
    2. Discover tools via session.list_tools()
    3. Convert MCP tool schemas to OpenAI function-calling format
    4. Call LLM with tools → if tool_calls → execute via MCP → repeat
    5. Return LLMCallResult with accumulated usage/cost
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import os
import re
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from llm_client.client import LLMCallResult
from llm_client.compliance_gate import (
    build_tool_parameter_index,
    validate_tool_call_inputs,
)
from llm_client.foundation import (
    BINDING_KEYS,
    HARD_BINDING_KEYS,
    check_binding_conflicts,
    coerce_run_id,
    extract_bindings_from_tool_args,
    merge_binding_state,
    new_event_id,
    new_session_id,
    normalize_bindings,
    now_iso,
    sha256_json,
    sha256_text,
    validate_foundation_event,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configurable defaults — all overridable via kwargs to call_llm/acall_llm
# ---------------------------------------------------------------------------

DEFAULT_MAX_TURNS: int = 20
"""Maximum tool-calling loop iterations before forcing a final answer."""

DEFAULT_MAX_TOOL_CALLS: int | None = 20
"""Maximum tool calls before forcing a final answer. None disables tool budgeting."""

TOOL_REASONING_FIELD: str = "tool_reasoning"
"""Optional argument every tool call can include for action-level observability."""

DEFAULT_REQUIRE_TOOL_REASONING: bool = False
"""Hard-fail tool calls missing tool_reasoning when enabled."""

BUDGET_EXEMPT_TOOL_NAMES: frozenset[str] = frozenset({
    "todo_create",
    "todo_update",
    "todo_list",
    "todo_reset",
    "submit_answer",
})
"""Tools exempt from max_tool_calls budgeting (planning + final submit)."""

AUTO_REASONING_TOOL_DEFAULTS: dict[str, str] = {
    "todo_create": "Create a TODO item to track a required reasoning step.",
    "todo_update": "Update TODO state after checking available evidence for that step.",
    "todo_reset": "Reset TODO state for the current question before planning/execution.",
    "todo_list": "Read TODO states to decide the next unblock action.",
    "submit_answer": "Submit the current best factual answer from gathered evidence.",
}
"""Deterministic fallback reasoning for idempotent planning tools."""

TURN_WARNING_THRESHOLD: int = 3
"""Inject a 'wrap up' system message this many turns before max_turns."""

DEFAULT_MCP_INIT_TIMEOUT: float = 30.0
"""Seconds to wait for each MCP server subprocess to initialize."""

DEFAULT_TOOL_RESULT_MAX_LENGTH: int = 50_000
"""Maximum character length for a single tool result. Longer results are truncated."""

DEFAULT_MAX_MESSAGE_CHARS: int = 260_000
"""Soft cap for serialized message-history size; old tool outputs are compacted above this."""

DEFAULT_TOOL_RESULT_KEEP_RECENT: int = 12
"""Keep this many most-recent tool result payloads fully in-context; older payloads are cleared."""

DEFAULT_TOOL_RESULT_CONTEXT_PREVIEW_CHARS: int = 200
"""Chars of preview retained when older tool payloads are cleared from active prompt context."""

DEFAULT_TOOL_INPUT_EXAMPLES_MAX_ITEMS: int = 2
"""Maximum tool input examples appended to tool descriptions."""

DEFAULT_TOOL_INPUT_EXAMPLE_MAX_CHARS: int = 240
"""Max chars retained per input example snippet in tool descriptions."""

DEFAULT_ENFORCE_TOOL_CONTRACTS: bool = False
"""When enabled, reject tool calls that violate declared composability contracts."""

DEFAULT_PROGRESSIVE_TOOL_DISCLOSURE: bool = True
"""When enabled with contracts, expose only currently composable tools per turn."""

DEFAULT_SUPPRESS_CONTROL_LOOP_CALLS: bool = False
"""When enabled, suppress repeated submit/todo control calls after validation failures."""

DEFAULT_INITIAL_ARTIFACTS: tuple[str, ...] = ("QUERY_TEXT",)
"""Default artifact kinds available before any tool call."""

DEFAULT_TOOL_DISCLOSURE_MAX_UNAVAILABLE: int = 10
"""Maximum unavailable tools reported per turn in disclosure guidance."""

DEFAULT_TOOL_DISCLOSURE_MAX_MISSING_PER_TOOL: int = 2
"""Maximum missing requirements listed per unavailable tool."""

DEFAULT_TOOL_DISCLOSURE_MAX_REPAIR_TOOLS: int = 2
"""Maximum conversion/repair tool hints surfaced per unavailable tool."""

DEFAULT_TOOL_DISCLOSURE_REASON_MAX_CHARS: int = 220
"""Maximum chars allocated to each unavailable-tool reason snippet."""

DEFAULT_TOOL_DISCLOSURE_TOKEN_CHARS: int = 4
"""Approximate chars/token conversion used for prompt-overhead accounting."""

DEFAULT_TOOL_CALL_STALL_TURNS: int = 3
"""Force final answer after this many consecutive tool-call turns with zero executable calls."""

DEFAULT_RETRIEVAL_STAGNATION_TURNS: int = 4
"""Force final answer after this many consecutive evidence turns with zero new evidence."""

DEFAULT_RETRIEVAL_STAGNATION_ACTION: str = "force_final"
"""Stagnation behavior: 'force_final' (default) or 'observe' (diagnostic only)."""

RETRIEVAL_STAGNATION_ACTIONS: frozenset[str] = frozenset({"force_final", "observe"})
"""Allowed stagnation actions."""

DEFAULT_FORCED_FINAL_MAX_ATTEMPTS: int = 1
"""Maximum forced-final LLM attempts before terminating."""

DEFAULT_FORCED_FINAL_CIRCUIT_BREAKER_THRESHOLD: int = 2
"""Open forced-final circuit breaker after this many same-class failures in a row."""

DEFAULT_FORCE_SUBMIT_RETRY_ON_MAX_TOOL_CALLS: bool = True
"""When tool budget is exhausted, count one forced submit retry attempt for observability."""

DEFAULT_ACCEPT_FORCED_ANSWER_ON_MAX_TOOL_CALLS: bool = True
"""When tool budget is exhausted, allow forced-final plain answer to satisfy required submit."""

FOUNDATION_SCHEMA_STRICT_ENV: str = "FOUNDATION_SCHEMA_STRICT"
"""When true, invalid foundation events raise instead of only warning."""

ARG_SELF_CONTAINED_TOOL_NAMES: frozenset[str] = frozenset({
    "chunk_get_text",
    "chunk_get_text_by_chunk_ids",
    "chunk_get_text_by_entity_ids",
})
"""Compatibility fallback for legacy self-contained tools (prefer contract metadata)."""


EVENT_CODE_TOOL_VALIDATION_MISSING_PREREQUISITE = "TOOL_VALIDATION_REJECTED_MISSING_PREREQUISITE"
EVENT_CODE_TOOL_VALIDATION_MISSING_CAPABILITY = "TOOL_VALIDATION_REJECTED_MISSING_CAPABILITY"
EVENT_CODE_TOOL_VALIDATION_BINDING_CONFLICT = "TOOL_VALIDATION_REJECTED_BINDING_CONFLICT"
EVENT_CODE_TOOL_VALIDATION_MISSING_TOOL_REASONING = "TOOL_VALIDATION_REJECTED_MISSING_TOOL_REASONING"
EVENT_CODE_TOOL_VALIDATION_SCHEMA = "TOOL_VALIDATION_REJECTED_SCHEMA"
EVENT_CODE_CONTROL_LOOP_SUPPRESSED = "CONTROL_CHURN_SUPPRESSED"
EVENT_CODE_CONTROL_CHURN_THRESHOLD = "CONTROL_CHURN_THRESHOLD_EXCEEDED"
EVENT_CODE_TOOL_RUNTIME_ERROR = "TOOL_EXECUTION_RUNTIME_ERROR"
EVENT_CODE_PROVIDER_EMPTY = "PROVIDER_EMPTY_CANDIDATES"
EVENT_CODE_PROVIDER_CREDITS_EXHAUSTED = "PROVIDER_CREDITS_EXHAUSTED"
# Compatibility aliases retained for external imports; taxonomy now uses one canonical code.
EVENT_CODE_PROVIDER_EMPTY_FIRST_TURN = EVENT_CODE_PROVIDER_EMPTY
EVENT_CODE_PROVIDER_EMPTY_RESPONSE = EVENT_CODE_PROVIDER_EMPTY
EVENT_CODE_FINALIZATION_CIRCUIT_BREAKER_OPEN = "FINALIZATION_CIRCUIT_BREAKER_OPEN"
EVENT_CODE_FINALIZATION_TOOL_CALL_DISALLOWED = "FINALIZATION_TOOL_CALL_DISALLOWED"
EVENT_CODE_RETRIEVAL_STAGNATION = "RETRIEVAL_STAGNATION"
EVENT_CODE_RETRIEVAL_STAGNATION_OBSERVED = "RETRIEVAL_STAGNATION_OBSERVED"
EVENT_CODE_NO_LEGAL_NONCONTROL_TOOLS = "NO_LEGAL_NONCONTROL_TOOLS"
EVENT_CODE_REQUIRED_SUBMIT_NOT_ATTEMPTED = "REQUIRED_SUBMIT_NOT_ATTEMPTED"
EVENT_CODE_REQUIRED_SUBMIT_NOT_ACCEPTED = "REQUIRED_SUBMIT_NOT_ACCEPTED"
EVENT_CODE_SUBMIT_FORCED_ACCEPT_BUDGET_EXHAUSTION = "SUBMIT_FORCED_ACCEPT_BUDGET_EXHAUSTION"
EVENT_CODE_SUBMIT_FORCED_ACCEPT_TURN_EXHAUSTION = "SUBMIT_FORCED_ACCEPT_TURN_EXHAUSTION"
EVENT_CODE_SUBMIT_FORCED_ACCEPT_FORCED_FINAL = "SUBMIT_FORCED_ACCEPT_FORCED_FINAL"

# Kwargs consumed by the MCP agent loop (popped before passing to inner acall_llm)
MCP_LOOP_KWARGS = frozenset({
    "mcp_servers",
    "mcp_sessions",
    "max_turns",
    "max_tool_calls",
    "require_tool_reasoning",
    "mcp_init_timeout",
    "tool_result_max_length",
    "max_message_chars",
    "tool_result_keep_recent",
    "tool_result_context_preview_chars",
    "enforce_tool_contracts",
    "progressive_tool_disclosure",
    "suppress_control_loop_calls",
    "tool_contracts",
    "initial_artifacts",
    "initial_bindings",
    "forced_final_max_attempts",
    "forced_final_circuit_breaker_threshold",
    "force_submit_retry_on_max_tool_calls",
    "accept_forced_answer_on_max_tool_calls",
    "finalization_fallback_models",
    "retrieval_stagnation_turns",
    "retrieval_stagnation_action",
})

# Kwargs consumed by the direct tool loop
TOOL_LOOP_KWARGS = frozenset({
    "python_tools",
    "max_turns",
    "max_tool_calls",
    "require_tool_reasoning",
    "tool_result_max_length",
    "max_message_chars",
    "tool_result_keep_recent",
    "tool_result_context_preview_chars",
    "enforce_tool_contracts",
    "progressive_tool_disclosure",
    "suppress_control_loop_calls",
    "tool_contracts",
    "initial_artifacts",
    "initial_bindings",
    "forced_final_max_attempts",
    "forced_final_circuit_breaker_threshold",
    "force_submit_retry_on_max_tool_calls",
    "accept_forced_answer_on_max_tool_calls",
    "finalization_fallback_models",
    "retrieval_stagnation_turns",
    "retrieval_stagnation_action",
})


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class MCPToolCallRecord:
    """Record of a single MCP tool call during the agent loop."""

    server: str
    tool: str
    arguments: dict[str, Any]
    tool_reasoning: str | None = None
    arg_coercions: list[dict[str, Any]] = field(default_factory=list)
    result: str | None = None
    error: str | None = None
    latency_s: float = 0.0


@dataclass(frozen=True)
class CapabilityRequirement:
    """Normalized capability requirement for composability checks."""

    kind: str
    ref_type: str | None = None
    namespace: str | None = None
    bindings_hash: str | None = None

    def to_dict(self) -> dict[str, str]:
        payload: dict[str, str] = {"kind": self.kind}
        if self.ref_type:
            payload["ref_type"] = self.ref_type
        if self.namespace:
            payload["namespace"] = self.namespace
        if self.bindings_hash:
            payload["bindings_hash"] = self.bindings_hash
        return payload

    def short_label(self) -> str:
        suffix_parts: list[str] = []
        if self.ref_type:
            suffix_parts.append(f"ref_type={self.ref_type}")
        if self.namespace:
            suffix_parts.append(f"namespace={self.namespace}")
        if self.bindings_hash:
            suffix_parts.append(f"bindings_hash={self.bindings_hash[:12]}")
        if not suffix_parts:
            return self.kind
        return f"{self.kind}[{', '.join(suffix_parts)}]"


@dataclass
class ToolCallValidation:
    """Normalized pre-execution validation outcome for a tool call."""

    is_valid: bool
    reason: str = ""
    error_code: str | None = None
    failure_phase: str | None = None
    call_bindings: dict[str, str | None] = field(default_factory=dict)
    missing_requirements: list[dict[str, str]] = field(default_factory=list)


@dataclass
class MCPAgentResult:
    """Accumulated result from the MCP agent loop.

    Stored in LLMCallResult.raw_response for introspection.
    """

    tool_calls: list[MCPToolCallRecord] = field(default_factory=list)
    turns: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_tokens: int = 0
    total_cache_creation_tokens: int = 0
    conversation_trace: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    """Diagnostic warnings accumulated across turns (retries, fallbacks, sticky model switches)."""
    models_used: set[str] = field(default_factory=set)
    """Set of model strings actually used during the agent loop (tracks fallback switches)."""


# ---------------------------------------------------------------------------
# Session Pool — reuse MCP servers across multiple calls
# ---------------------------------------------------------------------------


class MCPSessionPool:
    """Persistent MCP server connections for reuse across multiple acall_llm() calls.

    Usage:
        async with MCPSessionPool(mcp_servers) as pool:
            for question in questions:
                result = await acall_llm(model, msgs, mcp_sessions=pool)
    """

    def __init__(
        self,
        mcp_servers: dict[str, dict[str, Any]],
        init_timeout: float = DEFAULT_MCP_INIT_TIMEOUT,
    ):
        self.mcp_servers = mcp_servers
        self.init_timeout = init_timeout
        self._stack: AsyncExitStack | None = None
        self.sessions: dict[str, Any] = {}
        self.tool_to_server: dict[str, str] = {}
        self.openai_tools: list[dict[str, Any]] = []

    async def __aenter__(self) -> "MCPSessionPool":
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        self.sessions, self.tool_to_server, self.openai_tools = await _start_servers(
            self.mcp_servers, self._stack, self.init_timeout,
        )
        logger.info(
            "MCPSessionPool: started %d servers, %d tools",
            len(self.sessions), len(self.openai_tools),
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._stack:
            await self._stack.__aexit__(*exc)
            self._stack = None
            self.sessions = {}
            self.tool_to_server = {}
            self.openai_tools = []


# ---------------------------------------------------------------------------
# Schema conversion
# ---------------------------------------------------------------------------


def _normalize_tool_input_examples(raw: Any) -> list[str]:
    """Normalize raw input examples into compact display strings."""
    if raw is None:
        return []
    items: list[Any]
    if isinstance(raw, list):
        items = raw
    else:
        items = [raw]

    normalized: list[str] = []
    for item in items:
        if len(normalized) >= DEFAULT_TOOL_INPUT_EXAMPLES_MAX_ITEMS:
            break
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, (dict, list)):
            try:
                text = _json.dumps(item, ensure_ascii=False)
            except Exception:
                text = str(item).strip()
        else:
            text = str(item).strip()
        if not text:
            continue
        normalized.append(text[:DEFAULT_TOOL_INPUT_EXAMPLE_MAX_CHARS])
    return normalized


def _append_input_examples_to_description(description: str, examples: list[str]) -> str:
    """Append bounded input examples to tool description text."""
    base = str(description or "").strip()
    if not examples:
        return base
    bullet_lines = "\n".join(f"- {ex}" for ex in examples)
    examples_block = "Input examples:\n" + bullet_lines
    if base:
        return f"{base}\n\n{examples_block}"
    return examples_block


def _mcp_tool_to_openai(tool: Any) -> dict[str, Any]:
    """Convert an MCP Tool object to OpenAI function-calling format.

    MCP: {"name": "foo", "description": "...", "inputSchema": {...}}
    OpenAI: {"type": "function", "function": {"name": "foo", "description": "...", "parameters": {...}}}
    """
    parameters = dict(tool.inputSchema or {"type": "object", "properties": {}})
    if not isinstance(parameters, dict):
        parameters = {"type": "object", "properties": {}}
    parameters.setdefault("type", "object")
    properties = parameters.setdefault("properties", {})
    if not isinstance(properties, dict):
        properties = {}
        parameters["properties"] = properties
    properties.setdefault(
        TOOL_REASONING_FIELD,
        {
            "type": "string",
            "description": "Why this specific tool call is needed right now.",
        },
    )
    required = parameters.get("required")
    if not isinstance(required, list):
        required = []
    if TOOL_REASONING_FIELD not in required:
        required.append(TOOL_REASONING_FIELD)
    parameters["required"] = required

    raw_examples = None
    input_examples_attr = getattr(tool, "inputExamples", None)
    if isinstance(input_examples_attr, (str, list, tuple, dict)):
        raw_examples = input_examples_attr
    elif isinstance(parameters.get("examples"), list):
        raw_examples = parameters.get("examples")
    elif isinstance(getattr(tool, "metadata", None), dict):
        raw_examples = tool.metadata.get("input_examples")

    description = _append_input_examples_to_description(
        str(tool.description or ""),
        _normalize_tool_input_examples(raw_examples),
    )

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": description,
            "parameters": parameters,
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_usage(usage: dict[str, Any]) -> tuple[int, int, int, int]:
    """Extract (input_tokens, output_tokens, cached_tokens, cache_creation_tokens) from usage dict.

    Handles both OpenAI convention (prompt_tokens/completion_tokens)
    and Anthropic convention (input_tokens/output_tokens).
    Extracts provider-level prompt caching details when available.
    """
    inp = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
    out = usage.get("output_tokens") or usage.get("completion_tokens") or 0
    cached = usage.get("cached_tokens") or 0
    cache_creation = usage.get("cache_creation_tokens") or 0
    return int(inp), int(out), int(cached), int(cache_creation)


def _build_agent_usage(agent_result: MCPAgentResult) -> dict[str, int]:
    """Build consistent usage/loop counters for agent-mode calls."""
    usage: dict[str, int] = {
        "input_tokens": int(agent_result.total_input_tokens),
        "output_tokens": int(agent_result.total_output_tokens),
        "total_tokens": int(agent_result.total_input_tokens + agent_result.total_output_tokens),
        "num_turns": int(agent_result.turns or 0),
        "n_tool_calls": int(len(agent_result.tool_calls)),
        "n_budgeted_tool_calls": int(_count_budgeted_records(agent_result.tool_calls)),
    }
    if agent_result.total_cached_tokens:
        usage["cached_tokens"] = int(agent_result.total_cached_tokens)
    if agent_result.total_cache_creation_tokens:
        usage["cache_creation_tokens"] = int(agent_result.total_cache_creation_tokens)
    return usage


def _truncate(text: str, max_length: int) -> str:
    """Truncate text if it exceeds max_length, appending a notice."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + f"\n... [truncated at {max_length} chars]"


def _message_char_length(message: dict[str, Any]) -> int:
    """Best-effort serialized length for one chat message."""
    try:
        return len(_json.dumps(message, ensure_ascii=False, default=str))
    except Exception:
        return len(str(message))


def _compact_tool_history_for_context(
    messages: list[dict[str, Any]],
    max_message_chars: int,
) -> tuple[int, int, int]:
    """Compact verbose historical tool outputs when message history grows too large.

    Returns (compacted_message_count, chars_saved, resulting_chars).
    """
    if max_message_chars <= 0:
        total = sum(_message_char_length(m) for m in messages if isinstance(m, dict))
        return 0, 0, total

    total_chars = sum(_message_char_length(m) for m in messages if isinstance(m, dict))
    if total_chars <= max_message_chars:
        return 0, 0, total_chars

    target_chars = max(int(max_message_chars * 0.75), 32_000)
    compacted = 0
    saved = 0
    replacement = (
        '{"notice":"Earlier tool result compacted to fit context window. '
        'Re-run the tool if this evidence is needed again."}'
    )
    replacement_len = len(replacement)

    for msg in messages:
        if total_chars <= target_chars:
            break
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        content = msg.get("content")
        if not isinstance(content, str) or len(content) <= 512:
            continue
        old_len = len(content)
        if old_len <= replacement_len:
            continue
        msg["content"] = replacement
        delta = old_len - replacement_len
        total_chars -= delta
        saved += delta
        compacted += 1

    return compacted, saved, total_chars


def _clear_old_tool_results_for_context(
    messages: list[dict[str, Any]],
    keep_recent: int,
    preview_chars: int,
) -> tuple[int, int]:
    """Replace older tool payloads with compact stubs while keeping recent results verbatim.

    This proactively limits prompt growth on long tool traces without losing
    traceability: each cleared payload retains tool_call_id, preview, and hash.

    Returns (cleared_message_count, chars_saved).
    """
    if keep_recent < 0:
        return 0, 0

    tool_indices: list[int] = []
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        if not isinstance(msg.get("content"), str):
            continue
        tool_indices.append(idx)

    if len(tool_indices) <= keep_recent:
        return 0, 0

    keep_set = set(tool_indices[-keep_recent:]) if keep_recent > 0 else set()
    preview_limit = max(40, int(preview_chars))
    cleared = 0
    saved = 0

    for idx in tool_indices:
        if idx in keep_set:
            continue
        msg = messages[idx]
        content = msg.get("content")
        if not isinstance(content, str) or not content:
            continue
        # Idempotent clearing: do not keep rewriting previously-cleared stubs.
        if '"notice":"Tool result cleared from active context' in content:
            continue

        content_one_line = " ".join(content.strip().split())
        stub = _json.dumps(
            {
                "notice": "Tool result cleared from active context to reduce prompt size.",
                "tool_call_id": str(msg.get("tool_call_id", "")).strip(),
                "content_sha256": sha256_text(content).replace("sha256:", ""),
                "preview": content_one_line[:preview_limit],
            },
            ensure_ascii=False,
        )
        old_len = len(content)
        new_len = len(stub)
        msg["content"] = stub
        if old_len > new_len:
            saved += old_len - new_len
        cleared += 1

    return cleared, saved


def _is_budget_exempt_tool(tool_name: str) -> bool:
    """Return True when tool is excluded from max_tool_calls accounting."""
    return tool_name in BUDGET_EXEMPT_TOOL_NAMES


def _normalized_tool_name(raw_name: Any) -> str:
    """Normalize tool names for stable matching (budgeting, validation, execution)."""
    return str(raw_name or "").strip()


def _normalize_tool_call_name_inplace(tool_call: dict[str, Any]) -> str:
    """Normalize one tool call name in-place and return the normalized value."""
    fn = tool_call.get("function")
    if not isinstance(fn, dict):
        return ""
    normalized = _normalized_tool_name(fn.get("name"))
    fn["name"] = normalized
    return normalized


def _count_budgeted_records(records: list[MCPToolCallRecord]) -> int:
    """Count tool calls that consume max_tool_calls budget."""
    return sum(1 for r in records if not _is_budget_exempt_tool(r.tool))


def _count_budgeted_tool_calls(tool_calls: list[dict[str, Any]]) -> int:
    """Count proposed LLM tool calls that consume max_tool_calls budget."""
    used = 0
    for tc in tool_calls:
        tool_name = _normalized_tool_name(tc.get("function", {}).get("name", ""))
        if tool_name and not _is_budget_exempt_tool(tool_name):
            used += 1
    return used


def _trim_tool_calls_to_budget(
    tool_calls: list[dict[str, Any]],
    budget_remaining: int,
) -> tuple[list[dict[str, Any]], int]:
    """Keep all budget-exempt tools and trim only budgeted tools over cap."""
    kept: list[dict[str, Any]] = []
    kept_budgeted = 0
    dropped_budgeted = 0
    for tc in tool_calls:
        tool_name = _normalized_tool_name(tc.get("function", {}).get("name", ""))
        if _is_budget_exempt_tool(tool_name):
            kept.append(tc)
            continue
        if kept_budgeted < budget_remaining:
            kept.append(tc)
            kept_budgeted += 1
        else:
            dropped_budgeted += 1
    return kept, dropped_budgeted


def _tool_error_signature(error: str) -> str:
    """Normalize tool error text so repeated failures can be detected."""
    text = (error or "").strip().lower()
    if not text:
        return "unknown error"
    # Keep stable semantic portion while avoiding noisy prefixes.
    if ":" in text:
        text = text.split(":", 1)[1].strip() or text
    text = " ".join(text.split())
    return text[:160]


def _provider_failure_classification(exc: Exception, error_text: str) -> tuple[bool, str, str, bool]:
    """Best-effort provider failure classification for taxonomy + retry semantics."""
    classification = getattr(exc, "classification", None)
    if isinstance(classification, str):
        normalized = classification.strip().lower()
        if normalized.startswith("provider_empty"):
            return True, EVENT_CODE_PROVIDER_EMPTY, normalized, True
        if normalized in {
            "provider_credits_exhausted",
            "provider_insufficient_credits",
            "provider_insufficient_quota",
        }:
            return True, EVENT_CODE_PROVIDER_CREDITS_EXHAUSTED, normalized, False

    lower = (error_text or "").strip().lower()
    if "provider_empty_candidates" in lower or "empty content from llm" in lower:
        return True, EVENT_CODE_PROVIDER_EMPTY, "provider_empty_candidates", True
    if (
        "insufficient credits" in lower
        or "insufficient quota" in lower
        or "\"code\":402" in lower
    ):
        return (
            True,
            EVENT_CODE_PROVIDER_CREDITS_EXHAUSTED,
            "provider_credits_exhausted",
            False,
        )
    return False, "", "", False


def _foundation_schema_strict_enabled() -> bool:
    """Whether invalid foundation events should fail fast."""
    raw = os.environ.get(FOUNDATION_SCHEMA_STRICT_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _coerce_bool(value: Any, default: bool) -> bool:
    """Parse bool-like runtime values with safe defaults."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off", ""}:
            return False
    return default


def _normalize_model_chain(raw: Any) -> list[str]:
    """Normalize fallback model config into a deterministic string list."""
    if raw is None:
        return []

    values: list[str] = []
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",")]
        values.extend([p for p in parts if p])
    elif isinstance(raw, (list, tuple, set)):
        for item in raw:
            if not isinstance(item, str):
                continue
            value = item.strip()
            if value:
                values.append(value)

    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


@dataclass(frozen=True)
class AgentLoopRuntimePolicy:
    """Typed runtime policy consumed by _agent_loop."""

    forced_final_max_attempts: int
    forced_final_circuit_breaker_threshold: int
    forced_final_breaker_effective: bool
    force_submit_retry_on_max_tool_calls: bool
    accept_forced_answer_on_max_tool_calls: bool
    finalization_fallback_models: list[str]
    retrieval_stagnation_turns: int
    retrieval_stagnation_action: str
    tool_result_keep_recent: int
    tool_result_context_preview_chars: int
    run_config_spec: dict[str, Any]
    run_config_hash: str


@dataclass
class AgentLoopToolState:
    """Initial tool/contract state used by _agent_loop execution stages."""

    normalized_tool_contracts: dict[str, dict[str, Any]]
    tool_parameter_index: dict[str, dict[str, Any]]
    available_artifacts: set[str]
    initial_artifact_snapshot: list[str]
    available_capabilities: dict[str, set[tuple[str | None, str | None, str | None]]]
    initial_capability_snapshot: list[dict[str, Any]]
    available_bindings: dict[str, Any]
    initial_binding_snapshot: dict[str, Any]
    lane_closure_analysis: dict[str, Any]
    artifact_timeline: list[dict[str, Any]]
    requires_submit_answer: bool


def _resolve_agent_runtime_policy(
    *,
    model: str,
    max_turns: int,
    max_tool_calls: int | None,
    require_tool_reasoning: bool,
    enforce_tool_contracts: bool,
    progressive_tool_disclosure: bool,
    suppress_control_loop_calls: bool,
    tool_result_max_length: int,
    max_message_chars: int,
    kwargs: dict[str, Any],
    warning_sink: list[str],
) -> AgentLoopRuntimePolicy:
    """Extract, normalize, and hash runtime-policy settings from kwargs."""
    run_config_spec: dict[str, Any] = {
        "requested_model": model,
        "max_turns": max_turns,
        "max_tool_calls": max_tool_calls,
        "require_tool_reasoning": require_tool_reasoning,
        "enforce_tool_contracts": enforce_tool_contracts,
        "progressive_tool_disclosure": progressive_tool_disclosure,
        "suppress_control_loop_calls": suppress_control_loop_calls,
        "tool_result_max_length": tool_result_max_length,
        "max_message_chars": max_message_chars,
    }
    runtime_policy_kwargs = {
        k: kwargs.get(k)
        for k in (
            "temperature",
            "top_p",
            "timeout",
            "num_retries",
            "fallback_models",
            "fallback_limit",
            "execution_mode",
            "api_base",
            "service_tier",
            "forced_final_max_attempts",
            "forced_final_circuit_breaker_threshold",
            "force_submit_retry_on_max_tool_calls",
            "accept_forced_answer_on_max_tool_calls",
            "finalization_fallback_models",
            "retrieval_stagnation_turns",
            "retrieval_stagnation_action",
            "tool_result_keep_recent",
            "tool_result_context_preview_chars",
        )
        if k in kwargs
    }
    run_config_spec["runtime_policy"] = runtime_policy_kwargs
    run_config_hash = sha256_json(run_config_spec).replace("sha256:", "")

    forced_final_max_attempts_raw = kwargs.pop(
        "forced_final_max_attempts",
        DEFAULT_FORCED_FINAL_MAX_ATTEMPTS,
    )
    try:
        forced_final_max_attempts = int(forced_final_max_attempts_raw)
    except Exception:
        forced_final_max_attempts = DEFAULT_FORCED_FINAL_MAX_ATTEMPTS
    forced_final_max_attempts = max(1, forced_final_max_attempts)

    forced_final_breaker_threshold_raw = kwargs.pop(
        "forced_final_circuit_breaker_threshold",
        DEFAULT_FORCED_FINAL_CIRCUIT_BREAKER_THRESHOLD,
    )
    try:
        forced_final_circuit_breaker_threshold = int(forced_final_breaker_threshold_raw)
    except Exception:
        forced_final_circuit_breaker_threshold = DEFAULT_FORCED_FINAL_CIRCUIT_BREAKER_THRESHOLD
    forced_final_circuit_breaker_threshold = max(1, forced_final_circuit_breaker_threshold)
    forced_final_breaker_effective = (
        forced_final_circuit_breaker_threshold <= forced_final_max_attempts
    )
    if not forced_final_breaker_effective:
        warning = (
            "FINALIZATION_BREAKER_INERT: circuit breaker threshold "
            f"({forced_final_circuit_breaker_threshold}) exceeds forced_final_max_attempts "
            f"({forced_final_max_attempts}); breaker cannot trigger under current config."
        )
        warning_sink.append(warning)
        logger.warning(warning)

    force_submit_retry_on_max_tool_calls = _coerce_bool(
        kwargs.pop(
            "force_submit_retry_on_max_tool_calls",
            DEFAULT_FORCE_SUBMIT_RETRY_ON_MAX_TOOL_CALLS,
        ),
        DEFAULT_FORCE_SUBMIT_RETRY_ON_MAX_TOOL_CALLS,
    )
    accept_forced_answer_on_max_tool_calls = _coerce_bool(
        kwargs.pop(
            "accept_forced_answer_on_max_tool_calls",
            DEFAULT_ACCEPT_FORCED_ANSWER_ON_MAX_TOOL_CALLS,
        ),
        DEFAULT_ACCEPT_FORCED_ANSWER_ON_MAX_TOOL_CALLS,
    )

    finalization_fallback_models = _normalize_model_chain(
        kwargs.pop("finalization_fallback_models", None)
    )

    retrieval_stagnation_turns_raw = kwargs.pop(
        "retrieval_stagnation_turns",
        DEFAULT_RETRIEVAL_STAGNATION_TURNS,
    )
    try:
        retrieval_stagnation_turns = int(retrieval_stagnation_turns_raw)
    except Exception:
        retrieval_stagnation_turns = DEFAULT_RETRIEVAL_STAGNATION_TURNS
    retrieval_stagnation_turns = max(2, retrieval_stagnation_turns)

    retrieval_stagnation_action_raw = kwargs.pop(
        "retrieval_stagnation_action",
        DEFAULT_RETRIEVAL_STAGNATION_ACTION,
    )
    retrieval_stagnation_action = str(retrieval_stagnation_action_raw or "").strip().lower()
    if retrieval_stagnation_action not in RETRIEVAL_STAGNATION_ACTIONS:
        warning = (
            "RETRIEVAL_STAGNATION_ACTION_INVALID: unsupported action "
            f"{retrieval_stagnation_action_raw!r}; defaulting to "
            f"{DEFAULT_RETRIEVAL_STAGNATION_ACTION!r}."
        )
        warning_sink.append(warning)
        logger.warning(warning)
        retrieval_stagnation_action = DEFAULT_RETRIEVAL_STAGNATION_ACTION

    tool_result_keep_recent_raw = kwargs.pop(
        "tool_result_keep_recent",
        DEFAULT_TOOL_RESULT_KEEP_RECENT,
    )
    try:
        tool_result_keep_recent = int(tool_result_keep_recent_raw)
    except Exception:
        tool_result_keep_recent = DEFAULT_TOOL_RESULT_KEEP_RECENT
    tool_result_keep_recent = max(0, tool_result_keep_recent)

    tool_result_context_preview_chars_raw = kwargs.pop(
        "tool_result_context_preview_chars",
        DEFAULT_TOOL_RESULT_CONTEXT_PREVIEW_CHARS,
    )
    try:
        tool_result_context_preview_chars = int(tool_result_context_preview_chars_raw)
    except Exception:
        tool_result_context_preview_chars = DEFAULT_TOOL_RESULT_CONTEXT_PREVIEW_CHARS
    tool_result_context_preview_chars = max(40, tool_result_context_preview_chars)
    run_config_spec["tool_result_keep_recent"] = tool_result_keep_recent
    run_config_spec["tool_result_context_preview_chars"] = tool_result_context_preview_chars
    run_config_hash = sha256_json(run_config_spec).replace("sha256:", "")

    return AgentLoopRuntimePolicy(
        forced_final_max_attempts=forced_final_max_attempts,
        forced_final_circuit_breaker_threshold=forced_final_circuit_breaker_threshold,
        forced_final_breaker_effective=forced_final_breaker_effective,
        force_submit_retry_on_max_tool_calls=force_submit_retry_on_max_tool_calls,
        accept_forced_answer_on_max_tool_calls=accept_forced_answer_on_max_tool_calls,
        finalization_fallback_models=finalization_fallback_models,
        retrieval_stagnation_turns=retrieval_stagnation_turns,
        retrieval_stagnation_action=retrieval_stagnation_action,
        tool_result_keep_recent=tool_result_keep_recent,
        tool_result_context_preview_chars=tool_result_context_preview_chars,
        run_config_spec=run_config_spec,
        run_config_hash=run_config_hash,
    )


def _initialize_agent_tool_state(
    *,
    openai_tools: list[dict[str, Any]],
    tool_contracts: dict[str, dict[str, Any]] | None,
    initial_artifacts: list[str] | tuple[str, ...] | None,
    initial_bindings: dict[str, Any] | None,
    kwargs: dict[str, Any],
    enforce_tool_contracts: bool,
    warning_sink: list[str],
) -> AgentLoopToolState:
    """Build initial tool/contract capabilities and lane-closure state."""
    normalized_tool_contracts = _normalize_tool_contracts(tool_contracts)
    tool_parameter_index = build_tool_parameter_index(openai_tools)
    if initial_artifacts is None:
        initial_artifacts = DEFAULT_INITIAL_ARTIFACTS
    available_artifacts = {
        k for k in (_normalize_artifact_kind(v) for v in initial_artifacts) if k
    }
    if not available_artifacts:
        available_artifacts = set(DEFAULT_INITIAL_ARTIFACTS)
    initial_artifact_snapshot = sorted(available_artifacts)
    available_capabilities: dict[str, set[tuple[str | None, str | None, str | None]]] = {}
    for artifact in initial_artifact_snapshot:
        _capability_state_add(available_capabilities, kind=artifact)
    initial_capabilities_raw = kwargs.pop("initial_capabilities", None)
    if isinstance(initial_capabilities_raw, list):
        for item in initial_capabilities_raw:
            req = _capability_requirement_from_raw(item)
            if req is None:
                continue
            _capability_state_add(
                available_capabilities,
                kind=req.kind,
                ref_type=req.ref_type,
                namespace=req.namespace,
                bindings_hash=req.bindings_hash,
            )
    initial_capability_snapshot = _capability_state_snapshot(available_capabilities)
    available_bindings = normalize_bindings(initial_bindings)
    initial_binding_snapshot = dict(available_bindings)
    lane_closure_analysis = _analyze_lane_closure(
        normalized_tool_contracts=normalized_tool_contracts,
        initial_artifacts=set(available_artifacts),
        initial_capabilities=available_capabilities,
        available_bindings=available_bindings,
    )
    if (
        enforce_tool_contracts
        and normalized_tool_contracts
        and not bool(lane_closure_analysis.get("lane_closed"))
    ):
        unresolved_count = int(lane_closure_analysis.get("unresolved_tool_count") or 0)
        warning = (
            "CAPABILITY_CLOSURE_ADVISORY: lane has "
            f"{unresolved_count} unresolved non-control tool(s) from initial state. "
            "Ensure conversion operators are available for required capability transitions."
        )
        warning_sink.append(warning)
        logger.warning(warning)
    artifact_timeline: list[dict[str, Any]] = [
        {
            "turn": 0,
            "phase": "initial",
            "available_artifacts": list(initial_artifact_snapshot),
            "available_capabilities": list(initial_capability_snapshot),
        }
    ]
    if enforce_tool_contracts and not normalized_tool_contracts:
        warning = (
            "TOOL_CONTRACTS: enforce_tool_contracts=True but no contracts were provided; "
            "composability validation is skipped."
        )
        warning_sink.append(warning)
        logger.warning(warning)
    requires_submit_answer = any(
        t.get("function", {}).get("name") == "submit_answer"
        for t in openai_tools
        if isinstance(t, dict)
    )
    return AgentLoopToolState(
        normalized_tool_contracts=normalized_tool_contracts,
        tool_parameter_index=tool_parameter_index,
        available_artifacts=available_artifacts,
        initial_artifact_snapshot=initial_artifact_snapshot,
        available_capabilities=available_capabilities,
        initial_capability_snapshot=initial_capability_snapshot,
        available_bindings=available_bindings,
        initial_binding_snapshot=initial_binding_snapshot,
        lane_closure_analysis=lane_closure_analysis,
        artifact_timeline=artifact_timeline,
        requires_submit_answer=requires_submit_answer,
    )


@dataclass(frozen=True)
class FailureSummary:
    """Aggregated failure counters/classification for agent-loop metadata."""

    primary_failure_class: str
    secondary_failure_classes: list[str]
    first_terminal_failure_event_code: str | None
    failure_event_code_counts: dict[str, int]
    failure_event_class_counts: dict[str, int]
    provider_failure_event_code_counts: dict[str, int]
    provider_failure_event_total: int
    provider_caused_incompletion: bool


def _summarize_failure_events(
    *,
    failure_event_codes: list[str],
    retrieval_no_hits_count: int,
    control_loop_suppressed_calls: int,
    force_final_reason: str | None,
    run_completed: bool,
) -> FailureSummary:
    primary_failure_class, secondary_failure_classes = _classify_failure_signals(
        failure_event_codes=failure_event_codes,
        retrieval_no_hits_count=retrieval_no_hits_count,
        control_loop_suppressed_calls=control_loop_suppressed_calls,
        force_final_reason=force_final_reason,
        submit_answer_succeeded=run_completed,
    )
    first_terminal_failure_event_code = _first_terminal_failure_event_code(failure_event_codes)
    failure_event_code_counts: dict[str, int] = {}
    for code in failure_event_codes:
        failure_event_code_counts[code] = failure_event_code_counts.get(code, 0) + 1
    failure_event_class_counts: dict[str, int] = {}
    provider_failure_event_code_counts: dict[str, int] = {}
    for code, count in failure_event_code_counts.items():
        cls = _failure_class_for_event_code(code)
        if cls is None:
            continue
        failure_event_class_counts[cls] = failure_event_class_counts.get(cls, 0) + count
        if cls == "provider":
            provider_failure_event_code_counts[code] = count
    provider_failure_event_total = sum(provider_failure_event_code_counts.values())
    provider_caused_incompletion = (not run_completed) and primary_failure_class == "provider"
    return FailureSummary(
        primary_failure_class=primary_failure_class,
        secondary_failure_classes=secondary_failure_classes,
        first_terminal_failure_event_code=first_terminal_failure_event_code,
        failure_event_code_counts=failure_event_code_counts,
        failure_event_class_counts=failure_event_class_counts,
        provider_failure_event_code_counts=provider_failure_event_code_counts,
        provider_failure_event_total=provider_failure_event_total,
        provider_caused_incompletion=provider_caused_incompletion,
    )


@dataclass(frozen=True)
class FinalizationSummary:
    """Aggregated finalization counters for agent-loop metadata."""

    finalization_attempt_counts_by_model: dict[str, int]
    finalization_failure_counts_by_model: dict[str, int]
    finalization_success_counts_by_model: dict[str, int]
    finalization_failure_code_counts: dict[str, int]
    provider_empty_attempt_counts_by_model: dict[str, int]
    finalization_fallback_attempt_count: int
    finalization_fallback_usage_rate: float
    finalization_breaker_open_rate: float
    finalization_breaker_open_by_model: dict[str, int]


def _summarize_finalization_attempts(
    *,
    finalization_fallback_attempts: list[dict[str, Any]],
    finalization_primary_model: str | None,
    forced_final_attempts: int,
    forced_final_circuit_breaker_opened: bool,
) -> FinalizationSummary:
    finalization_attempt_counts_by_model: dict[str, int] = {}
    finalization_failure_counts_by_model: dict[str, int] = {}
    finalization_success_counts_by_model: dict[str, int] = {}
    finalization_failure_code_counts: dict[str, int] = {}
    provider_empty_attempt_counts_by_model: dict[str, int] = {}
    finalization_fallback_attempt_count = 0
    for attempt in finalization_fallback_attempts:
        model_name = str(attempt.get("model", "")).strip() or "<unknown>"
        status = str(attempt.get("status", "")).strip().lower()
        error_code = attempt.get("error_code")
        finalization_attempt_counts_by_model[model_name] = (
            finalization_attempt_counts_by_model.get(model_name, 0) + 1
        )
        if (
            isinstance(finalization_primary_model, str)
            and finalization_primary_model.strip()
            and model_name.lower() != finalization_primary_model.lower()
        ):
            finalization_fallback_attempt_count += 1
        if status == "success":
            finalization_success_counts_by_model[model_name] = (
                finalization_success_counts_by_model.get(model_name, 0) + 1
            )
            continue
        finalization_failure_counts_by_model[model_name] = (
            finalization_failure_counts_by_model.get(model_name, 0) + 1
        )
        if isinstance(error_code, str) and error_code:
            finalization_failure_code_counts[error_code] = (
                finalization_failure_code_counts.get(error_code, 0) + 1
            )
            if error_code == EVENT_CODE_PROVIDER_EMPTY:
                provider_empty_attempt_counts_by_model[model_name] = (
                    provider_empty_attempt_counts_by_model.get(model_name, 0) + 1
                )
    finalization_fallback_usage_rate = (
        (finalization_fallback_attempt_count / forced_final_attempts)
        if forced_final_attempts > 0
        else 0.0
    )
    finalization_breaker_open_rate = 1.0 if forced_final_circuit_breaker_opened else 0.0
    finalization_breaker_open_by_model: dict[str, int] = {}
    if forced_final_circuit_breaker_opened and finalization_fallback_attempts:
        breaker_model = str(finalization_fallback_attempts[-1].get("model", "")).strip() or "<unknown>"
        finalization_breaker_open_by_model[breaker_model] = 1
    return FinalizationSummary(
        finalization_attempt_counts_by_model=finalization_attempt_counts_by_model,
        finalization_failure_counts_by_model=finalization_failure_counts_by_model,
        finalization_success_counts_by_model=finalization_success_counts_by_model,
        finalization_failure_code_counts=finalization_failure_code_counts,
        provider_empty_attempt_counts_by_model=provider_empty_attempt_counts_by_model,
        finalization_fallback_attempt_count=finalization_fallback_attempt_count,
        finalization_fallback_usage_rate=finalization_fallback_usage_rate,
        finalization_breaker_open_rate=finalization_breaker_open_rate,
        finalization_breaker_open_by_model=finalization_breaker_open_by_model,
    )


@dataclass(frozen=True)
class ForcedFinalizationResult:
    """Outcome and metric deltas from the forced-final execution path."""

    final_content: str
    final_finish_reason: str
    finalization_primary_model: str
    forced_final_attempts: int
    forced_final_circuit_breaker_opened: bool
    finalization_fallback_used: bool
    finalization_fallback_succeeded: bool
    finalization_events: list[str]
    finalization_fallback_attempts: list[dict[str, Any]]
    failure_event_codes: list[str]
    context_tool_result_clearings_delta: int
    context_tool_results_cleared_delta: int
    context_tool_result_cleared_chars_delta: int
    context_compactions_delta: int
    context_compacted_messages_delta: int
    context_compacted_chars_delta: int
    total_cost_delta: float
    total_input_tokens_delta: int
    total_output_tokens_delta: int
    total_cached_tokens_delta: int
    total_cache_creation_tokens_delta: int
    turns_delta: int


async def _execute_forced_finalization(
    *,
    force_final_reason: str,
    max_turns: int,
    max_message_chars: int,
    tool_result_keep_recent: int,
    tool_result_context_preview_chars: int,
    messages: list[dict[str, Any]],
    kwargs: dict[str, Any],
    timeout: int,
    effective_model: str,
    attempted_models: list[str],
    finalization_fallback_models: list[str],
    forced_final_max_attempts: int,
    forced_final_circuit_breaker_threshold: int,
    available_artifacts: set[str],
    available_bindings: dict[str, Any],
    foundation_run_id: str,
    foundation_session_id: str,
    foundation_actor_id: str,
    agent_result: MCPAgentResult,
    emit_foundation_event: Any,
) -> ForcedFinalizationResult:
    """Run the forced-final no-tools lane and return aggregate deltas/outcomes."""
    if force_final_reason == "max_tool_calls":
        force_msg = {
            "role": "user",
            "content": (
                "[SYSTEM: Tool-call budget exhausted. You MUST give your best answer now "
                "using evidence already retrieved. Extract the best name/date/number and answer.]"
            ),
        }
    elif force_final_reason == "control_churn":
        force_msg = {
            "role": "user",
            "content": (
                "[SYSTEM: Tool-call loop stalled with repeated non-executable calls. "
                "You MUST submit your best factual answer now from current evidence.]"
            ),
        }
    elif force_final_reason == "retrieval_stagnation":
        force_msg = {
            "role": "user",
            "content": (
                "[SYSTEM: Retrieval stagnated with no new evidence. "
                "Submit your best factual answer now based only on retrieved evidence.]"
            ),
        }
    else:
        logger.warning(
            "Agent loop exhausted max_turns=%d, forcing final answer",
            max_turns,
        )
        force_msg = {
            "role": "user",
            "content": (
                "[SYSTEM: All turns exhausted. You MUST give your best answer now "
                "based on the evidence you found. Extract any name, date, or number "
                "from the tool results. 'I don't know' is NOT acceptable — guess.]"
            ),
        }

    messages.append(force_msg)
    agent_result.conversation_trace.append(force_msg)

    context_tool_result_clearings_delta = 0
    context_tool_results_cleared_delta = 0
    context_tool_result_cleared_chars_delta = 0
    context_compactions_delta = 0
    context_compacted_messages_delta = 0
    context_compacted_chars_delta = 0
    cleared_count, cleared_chars = _clear_old_tool_results_for_context(
        messages,
        keep_recent=tool_result_keep_recent,
        preview_chars=tool_result_context_preview_chars,
    )
    if cleared_count:
        context_tool_result_clearings_delta = 1
        context_tool_results_cleared_delta = cleared_count
        context_tool_result_cleared_chars_delta = cleared_chars
        warning = (
            "CONTEXT_TOOL_RESULT_CLEARING: replaced "
            f"{cleared_count} older tool payload(s) with compact stubs "
            f"(saved ~{cleared_chars} chars; keep_recent={tool_result_keep_recent}) before forced-final call"
        )
        agent_result.warnings.append(warning)
        logger.warning(warning)

    compacted_count, compacted_chars, current_chars = _compact_tool_history_for_context(
        messages, max_message_chars,
    )
    if compacted_count:
        context_compactions_delta = 1
        context_compacted_messages_delta = compacted_count
        context_compacted_chars_delta = compacted_chars
        warning = (
            "CONTEXT_COMPACTION: compacted "
            f"{compacted_count} tool message(s), saved ~{compacted_chars} chars "
            f"(history ~{current_chars} chars) before forced-final call"
        )
        agent_result.warnings.append(warning)
        logger.warning(warning)

    # Finalization fallback lane:
    # - Primary forced-final call uses the current effective model.
    # - Optional fallback models are attempted only for finalization.
    # - No tools are passed, and per-turn fallback chains are disabled.
    forced_kwargs = dict(kwargs)
    forced_kwargs.pop("fallback_models", None)
    finalization_primary_model = effective_model
    base_model_chain: list[str] = [effective_model]
    for fb_model in finalization_fallback_models:
        if fb_model.lower() == effective_model.lower():
            continue
        base_model_chain.append(fb_model)

    def _forced_final_model_for_attempt(attempt_idx: int) -> str:
        if attempt_idx < len(base_model_chain):
            return base_model_chain[attempt_idx]
        return base_model_chain[-1]

    total_cost_delta = 0.0
    total_input_tokens_delta = 0
    total_output_tokens_delta = 0
    total_cached_tokens_delta = 0
    total_cache_creation_tokens_delta = 0
    turns_delta = 0

    forced_final_attempts = 0
    forced_final_circuit_breaker_opened = False
    forced_final_error_streak = 0
    last_forced_final_error_class: str | None = None
    finalization_fallback_used = False
    finalization_fallback_succeeded = False
    finalization_events: list[str] = []
    finalization_fallback_attempts: list[dict[str, Any]] = []
    final_result = None
    terminal_error_text: str | None = None
    forced_final_failure_codes: list[str] = []

    def _record_forced_final_error(
        *,
        attempt_idx: int,
        final_model: str,
        event_code: str,
        error_text: str,
        error_class: str,
        is_provider_failure: bool,
    ) -> bool:
        nonlocal forced_final_error_streak, last_forced_final_error_class
        nonlocal terminal_error_text, forced_final_circuit_breaker_opened
        forced_final_failure_codes.append(event_code)
        if final_model.lower() == effective_model.lower():
            finalization_events.append("FINALIZATION_PRIMARY_FAILED")
        else:
            finalization_events.append("FINALIZATION_FALLBACK_FAILED")
        finalization_fallback_attempts.append(
            {
                "attempt_index": attempt_idx + 1,
                "model": final_model,
                "status": "error",
                "error_code": event_code,
                "error_text": error_text,
                "error_class": error_class,
            }
        )
        if error_class == last_forced_final_error_class:
            forced_final_error_streak += 1
        else:
            forced_final_error_streak = 1
            last_forced_final_error_class = error_class
        terminal_error_text = error_text
        if forced_final_error_streak < forced_final_circuit_breaker_threshold:
            return False
        forced_final_circuit_breaker_opened = True
        finalization_events.append("FINALIZATION_CIRCUIT_BREAKER_OPEN")
        forced_final_failure_codes.append(EVENT_CODE_FINALIZATION_CIRCUIT_BREAKER_OPEN)
        emit_foundation_event(
            {
                "event_id": new_event_id(),
                "event_type": "ToolFailed",
                "timestamp": now_iso(),
                "run_id": foundation_run_id,
                "session_id": foundation_session_id,
                "actor_id": foundation_actor_id,
                "operation": {"name": "__finalization__", "version": None},
                "inputs": {
                    "artifact_ids": sorted(available_artifacts),
                    "params": {
                        "attempts": forced_final_attempts,
                        "threshold": forced_final_circuit_breaker_threshold,
                        "error_class": error_class,
                    },
                    "bindings": dict(available_bindings),
                },
                "outputs": {"artifact_ids": [], "payload_hashes": []},
                "failure": {
                    "error_code": EVENT_CODE_FINALIZATION_CIRCUIT_BREAKER_OPEN,
                    "category": "provider" if is_provider_failure else "execution",
                    "phase": "execution",
                    "retryable": False,
                    "tool_name": "__finalization__",
                    "user_message": (
                        "Finalization circuit breaker opened after repeated same-class failures."
                    ),
                    "debug_ref": None,
                },
            }
        )
        return True

    for attempt_idx in range(forced_final_max_attempts):
        if forced_final_circuit_breaker_opened:
            break
        final_model = _forced_final_model_for_attempt(attempt_idx)
        forced_final_attempts += 1
        is_fallback_attempt = final_model.lower() != effective_model.lower()
        if is_fallback_attempt and not finalization_fallback_used:
            finalization_fallback_used = True
            finalization_events.append("FINALIZATION_FALLBACK_USED")

        try:
            candidate_result = await _inner_acall_llm(
                final_model,
                messages,
                timeout=timeout,
                **forced_kwargs,
            )
        except Exception as exc:
            error_text = str(exc).strip() or f"{type(exc).__name__}"
            (
                is_provider_failure,
                event_code,
                provider_classification,
                retryable_provider,
            ) = _provider_failure_classification(
                exc,
                error_text,
            )
            if not event_code:
                event_code = EVENT_CODE_TOOL_RUNTIME_ERROR
            error_class = provider_classification or type(exc).__name__
            warning = (
                "AGENT_LLM_CALL_FAILED: forced_final attempt="
                f"{attempt_idx + 1}/{forced_final_max_attempts} model={final_model} error={error_text}"
            )
            agent_result.warnings.append(warning)
            logger.warning(warning)
            emit_foundation_event(
                {
                    "event_id": new_event_id(),
                    "event_type": "ToolFailed",
                    "timestamp": now_iso(),
                    "run_id": foundation_run_id,
                    "session_id": foundation_session_id,
                    "actor_id": foundation_actor_id,
                    "operation": {"name": "_inner_acall_llm", "version": None},
                    "inputs": {
                        "artifact_ids": sorted(available_artifacts),
                        "params": {
                            "turn": agent_result.turns + 1,
                            "forced_final": True,
                            "attempt": attempt_idx + 1,
                            "model": final_model,
                            "error": error_text,
                            "provider_classification": provider_classification or "",
                            "provider_subevent": "finalization" if is_provider_failure else "",
                        },
                        "bindings": dict(available_bindings),
                    },
                    "outputs": {"artifact_ids": [], "payload_hashes": []},
                    "failure": {
                        "error_code": event_code,
                        "category": "provider" if is_provider_failure else "execution",
                        "phase": "execution",
                        "retryable": bool(retryable_provider),
                        "tool_name": "_inner_acall_llm",
                        "user_message": error_text,
                        "debug_ref": None,
                    },
                }
            )
            _record_forced_final_error(
                attempt_idx=attempt_idx,
                final_model=final_model,
                event_code=event_code,
                error_text=error_text,
                error_class=error_class,
                is_provider_failure=is_provider_failure,
            )
            continue

        final_result_model = str(candidate_result.model).strip() or final_model
        agent_result.models_used.add(final_result_model)
        if final_result_model and final_result_model not in attempted_models:
            attempted_models.append(final_result_model)
        if candidate_result.warnings:
            agent_result.warnings.extend(candidate_result.warnings)

        if candidate_result.tool_calls:
            disallowed_msg = (
                "Forced-final returned tool calls while tools are disallowed."
            )
            emit_foundation_event(
                {
                    "event_id": new_event_id(),
                    "event_type": "ToolFailed",
                    "timestamp": now_iso(),
                    "run_id": foundation_run_id,
                    "session_id": foundation_session_id,
                    "actor_id": foundation_actor_id,
                    "operation": {"name": "_inner_acall_llm", "version": None},
                    "inputs": {
                        "artifact_ids": sorted(available_artifacts),
                        "params": {
                            "turn": agent_result.turns + 1,
                            "forced_final": True,
                            "attempt": attempt_idx + 1,
                            "model": final_model,
                            "tool_calls_count": len(candidate_result.tool_calls),
                        },
                        "bindings": dict(available_bindings),
                    },
                    "outputs": {"artifact_ids": [], "payload_hashes": []},
                    "failure": {
                        "error_code": EVENT_CODE_FINALIZATION_TOOL_CALL_DISALLOWED,
                        "category": "policy",
                        "phase": "post_validation",
                        "retryable": False,
                        "tool_name": "_inner_acall_llm",
                        "user_message": disallowed_msg,
                        "debug_ref": None,
                    },
                }
            )
            _record_forced_final_error(
                attempt_idx=attempt_idx,
                final_model=final_model,
                event_code=EVENT_CODE_FINALIZATION_TOOL_CALL_DISALLOWED,
                error_text=disallowed_msg,
                error_class="finalization_tool_call_disallowed",
                is_provider_failure=False,
            )
            continue

        if not candidate_result.content:
            empty_msg = "Provider returned empty content on forced-final call."
            emit_foundation_event(
                {
                    "event_id": new_event_id(),
                    "event_type": "ToolFailed",
                    "timestamp": now_iso(),
                    "run_id": foundation_run_id,
                    "session_id": foundation_session_id,
                    "actor_id": foundation_actor_id,
                    "operation": {"name": "_inner_acall_llm", "version": None},
                    "inputs": {
                        "artifact_ids": sorted(available_artifacts),
                        "params": {
                            "turn": agent_result.turns + 1,
                            "forced_final": True,
                            "attempt": attempt_idx + 1,
                            "model": final_model,
                            "provider_classification": "provider_empty_candidates",
                            "provider_subevent": "finalization",
                        },
                        "bindings": dict(available_bindings),
                    },
                    "outputs": {"artifact_ids": [], "payload_hashes": []},
                    "failure": {
                        "error_code": EVENT_CODE_PROVIDER_EMPTY,
                        "category": "provider",
                        "phase": "execution",
                        "retryable": True,
                        "tool_name": "_inner_acall_llm",
                        "user_message": empty_msg,
                        "debug_ref": None,
                    },
                }
            )
            _record_forced_final_error(
                attempt_idx=attempt_idx,
                final_model=final_model,
                event_code=EVENT_CODE_PROVIDER_EMPTY,
                error_text=empty_msg,
                error_class="provider_empty_candidates",
                is_provider_failure=True,
            )
            continue

        forced_final_error_streak = 0
        last_forced_final_error_class = None
        final_result = candidate_result
        final_content = candidate_result.content
        final_finish_reason = candidate_result.finish_reason
        finalization_fallback_attempts.append(
            {
                "attempt_index": attempt_idx + 1,
                "model": final_model,
                "status": "success",
                "error_code": None,
                "error_text": "",
                "error_class": None,
            }
        )
        if is_fallback_attempt:
            finalization_fallback_succeeded = True
            finalization_events.append("FINALIZATION_FALLBACK_SUCCEEDED")
        else:
            finalization_events.append("FINALIZATION_PRIMARY_SUCCEEDED")
        total_cost_delta += candidate_result.cost
        inp, out, cached, cache_create = _extract_usage(candidate_result.usage or {})
        total_input_tokens_delta += inp
        total_output_tokens_delta += out
        total_cached_tokens_delta += cached
        total_cache_creation_tokens_delta += cache_create
        turns_delta += 1
        agent_result.conversation_trace.append({
            "role": "assistant",
            "content": final_content,
        })
        break

    if final_result is None:
        if forced_final_circuit_breaker_opened:
            warning = (
                "FINALIZATION_CIRCUIT_BREAKER_OPEN: halted forced-final retries after "
                f"{forced_final_attempts} attempt(s)."
            )
            agent_result.warnings.append(warning)
            logger.warning(warning)
        final_content = terminal_error_text or "Forced-final failed with no provider content."
        final_finish_reason = "error"
        failure_event_codes = list(forced_final_failure_codes)
    else:
        failure_event_codes = []

    return ForcedFinalizationResult(
        final_content=final_content,
        final_finish_reason=final_finish_reason,
        finalization_primary_model=finalization_primary_model,
        forced_final_attempts=forced_final_attempts,
        forced_final_circuit_breaker_opened=forced_final_circuit_breaker_opened,
        finalization_fallback_used=finalization_fallback_used,
        finalization_fallback_succeeded=finalization_fallback_succeeded,
        finalization_events=finalization_events,
        finalization_fallback_attempts=finalization_fallback_attempts,
        failure_event_codes=failure_event_codes,
        context_tool_result_clearings_delta=context_tool_result_clearings_delta,
        context_tool_results_cleared_delta=context_tool_results_cleared_delta,
        context_tool_result_cleared_chars_delta=context_tool_result_cleared_chars_delta,
        context_compactions_delta=context_compactions_delta,
        context_compacted_messages_delta=context_compacted_messages_delta,
        context_compacted_chars_delta=context_compacted_chars_delta,
        total_cost_delta=total_cost_delta,
        total_input_tokens_delta=total_input_tokens_delta,
        total_output_tokens_delta=total_output_tokens_delta,
        total_cached_tokens_delta=total_cached_tokens_delta,
        total_cache_creation_tokens_delta=total_cache_creation_tokens_delta,
        turns_delta=turns_delta,
    )


def _is_evidence_tool_name(tool_name: str) -> bool:
    """Best-effort classification for evidence-producing non-control tools."""
    name = (tool_name or "").strip().lower()
    if not name or _is_budget_exempt_tool(name):
        return False
    return name.startswith(("entity_", "relationship_", "chunk_", "subgraph_", "community_", "bridge_"))


def _extract_unfinished_todo_ids(error_text: str) -> list[str]:
    """Extract TODO IDs from submit/todo validation messages."""
    text = (error_text or "").strip()
    if not text:
        return []
    # Prefer explicit unfinished-blocking segment when present.
    # Examples:
    # - "Unfinished TODOs: todo_2, todo_3."
    # - "Unfinished TODOs currently blocking submit: todo_2, todo_3."
    match = re.search(
        r"unfinished\s+todos?(?:\s+[a-z ]+)?\s*:\s*([^\n]+)",
        text,
        flags=re.IGNORECASE,
    )
    segment = match.group(1) if match else text
    # Only treat canonical task IDs as unfinished TODO blockers.
    found = re.findall(r"\btodo_\d+\b", segment)
    # Preserve order while de-duplicating.
    deduped: list[str] = []
    seen: set[str] = set()
    for item in found:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _extract_tool_call_args(tc: dict[str, Any]) -> dict[str, Any] | None:
    """Parse function-call arguments as dict when possible."""
    raw = tc.get("function", {}).get("arguments", {})
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = _json.loads(raw)
        except Exception:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def _parse_record_result_json(record: MCPToolCallRecord) -> dict[str, Any] | None:
    """Best-effort parse of JSON tool result payloads."""
    if not record.result or not isinstance(record.result, str):
        return None
    try:
        parsed = _json.loads(record.result)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _set_tool_call_args(tc: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of tc with updated function arguments preserving argument wire type."""
    out = dict(tc)
    fn = dict(out.get("function", {}))
    raw = fn.get("arguments", {})
    if isinstance(raw, str):
        fn["arguments"] = _json.dumps(args)
    else:
        fn["arguments"] = args
    out["function"] = fn
    return out


def _tool_evidence_signature(record: MCPToolCallRecord) -> str | None:
    """Stable signature for successful non-control evidence-producing tool outputs."""
    if record.error or _is_budget_exempt_tool(record.tool):
        return None
    if record.result is None:
        return None
    result_hash = sha256_text(record.result)
    return f"{record.tool}:{result_hash}"


def _evidence_digest(evidence_signatures: set[str]) -> str:
    """Deterministic digest of accumulated evidence signatures."""
    return sha256_json(sorted(evidence_signatures)).replace("sha256:", "")


def _autofill_tool_reasoning(
    tc: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Autofill tool_reasoning for select deterministic tools when omitted."""
    tool_name = tc.get("function", {}).get("name", "")
    fallback = AUTO_REASONING_TOOL_DEFAULTS.get(tool_name)
    if not fallback:
        return tc, False

    args = _extract_tool_call_args(tc)
    if not isinstance(args, dict):
        return tc, False

    existing = args.get(TOOL_REASONING_FIELD)
    if isinstance(existing, str) and existing.strip():
        return tc, False

    patched_args = dict(args)
    patched_args[TOOL_REASONING_FIELD] = fallback
    return _set_tool_call_args(tc, patched_args), True


def _normalize_artifact_kind(kind: Any) -> str | None:
    """Normalize artifact kind labels (e.g., SlotKind names) to upper snake case strings."""
    if isinstance(kind, str):
        normalized = kind.strip().upper()
        return normalized or None
    return None


def _normalize_capability_value(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _capability_requirement_from_raw(raw: Any) -> CapabilityRequirement | None:
    """Parse one capability requirement from raw contract payload."""
    if isinstance(raw, str):
        kind = _normalize_artifact_kind(raw)
        if kind:
            return CapabilityRequirement(kind=kind)
        return None

    if not isinstance(raw, dict):
        return None

    kind = _normalize_artifact_kind(raw.get("kind"))
    if not kind:
        return None

    ref_type = _normalize_capability_value(raw.get("ref_type"))
    namespace = _normalize_capability_value(raw.get("namespace"))
    bindings_hash = _normalize_capability_value(raw.get("bindings_hash"))
    return CapabilityRequirement(
        kind=kind,
        ref_type=ref_type,
        namespace=namespace,
        bindings_hash=bindings_hash,
    )


def _normalize_capability_requirements(raw: Any) -> list[CapabilityRequirement]:
    if not isinstance(raw, (list, tuple, set)):
        return []
    out: list[CapabilityRequirement] = []
    seen: set[tuple[str, str | None, str | None, str | None]] = set()
    for item in raw:
        req = _capability_requirement_from_raw(item)
        if req is None:
            continue
        key = (req.kind, req.ref_type, req.namespace, req.bindings_hash)
        if key in seen:
            continue
        seen.add(key)
        out.append(req)
    return out


def _capability_state_add(
    state: dict[str, set[tuple[str | None, str | None, str | None]]],
    *,
    kind: str | None,
    ref_type: str | None = None,
    namespace: str | None = None,
    bindings_hash: str | None = None,
) -> bool:
    normalized_kind = _normalize_artifact_kind(kind)
    if not normalized_kind:
        return False
    bucket = state.setdefault(normalized_kind, set())
    entry = (
        _normalize_capability_value(ref_type),
        _normalize_capability_value(namespace),
        _normalize_capability_value(bindings_hash),
    )
    before = len(bucket)
    bucket.add(entry)
    return len(bucket) > before


def _capability_state_has(
    state: dict[str, set[tuple[str | None, str | None, str | None]]],
    req: CapabilityRequirement,
) -> bool:
    bucket = state.get(req.kind)
    if not bucket:
        return False
    for ref_type, namespace, bindings_hash in bucket:
        if req.ref_type and req.ref_type != ref_type:
            continue
        if req.namespace and req.namespace != namespace:
            continue
        if req.bindings_hash and req.bindings_hash != bindings_hash:
            continue
        return True
    return False


def _capability_state_snapshot(
    state: dict[str, set[tuple[str | None, str | None, str | None]]],
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for kind in sorted(state):
        entries = sorted(
            state[kind],
            key=lambda item: (
                item[0] or "",
                item[1] or "",
                item[2] or "",
            ),
        )
        for ref_type, namespace, bindings_hash in entries:
            req = CapabilityRequirement(
                kind=kind,
                ref_type=ref_type,
                namespace=namespace,
                bindings_hash=bindings_hash,
            )
            out.append(req.to_dict())
    return out


def _capability_requirement_matches(
    produced: CapabilityRequirement,
    required: CapabilityRequirement,
) -> bool:
    """Whether a produced capability could satisfy a required capability."""
    if produced.kind != required.kind:
        return False
    if required.ref_type and produced.ref_type and required.ref_type != produced.ref_type:
        return False
    if required.namespace and produced.namespace and required.namespace != produced.namespace:
        return False
    if (
        required.bindings_hash
        and produced.bindings_hash
        and required.bindings_hash != produced.bindings_hash
    ):
        return False
    return True


def _canonical_binding_spec(
    bindings: dict[str, str | None],
    *,
    keys: tuple[str, ...],
) -> dict[str, str | None]:
    """Canonical binding spec for deterministic hashing (stable key subset/order)."""
    return {key: bindings.get(key) for key in keys}


def _hard_bindings_spec(bindings: dict[str, str | None]) -> dict[str, str | None]:
    return _canonical_binding_spec(bindings, keys=HARD_BINDING_KEYS)


def _full_bindings_spec(bindings: dict[str, str | None]) -> dict[str, str | None]:
    return _canonical_binding_spec(bindings, keys=BINDING_KEYS)


def _binding_hash(spec: dict[str, str | None]) -> str:
    return sha256_json(spec).replace("sha256:", "")


def _hard_bindings_state_hash(bindings: dict[str, str | None]) -> str:
    return _binding_hash(_hard_bindings_spec(bindings))


def _full_bindings_state_hash(bindings: dict[str, str | None]) -> str:
    return _binding_hash(_full_bindings_spec(bindings))


def _namespace_from_bindings(bindings: dict[str, str | None]) -> str | None:
    """Best-effort namespace heuristic from authoritative binding state."""
    for key in ("graph_id", "dataset_id", "scope_id", "corpus_id", "vector_store_id"):
        value = bindings.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _explicit_ref_type_from_args(kind: str, args: dict[str, Any] | None) -> str | None:
    if not isinstance(args, dict):
        return None

    def _has_values(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (list, tuple, set)):
            return any(_has_values(v) for v in value)
        return True

    if kind == "ENTITY_SET":
        if _has_values(args.get("entity_ids")) or _has_values(args.get("entity_id")):
            return "id"
        if _has_values(args.get("entity_names")) or _has_values(args.get("entity_name")):
            return "name"
    if kind == "CHUNK_SET":
        if _has_values(args.get("chunk_ids")) or _has_values(args.get("chunk_id")):
            return "id"
    return None


def _infer_output_capabilities(
    *,
    tool_name: str,
    parsed_args: dict[str, Any] | None,
    produced_artifacts: set[str],
    available_bindings: dict[str, str | None],
) -> list[CapabilityRequirement]:
    """Infer lightweight capability tags from tool name + arguments."""
    namespace = _namespace_from_bindings(available_bindings)
    bindings_hash = _hard_bindings_state_hash(available_bindings)
    inferred: list[CapabilityRequirement] = []

    for kind in sorted(produced_artifacts):
        ref_type = _explicit_ref_type_from_args(kind, parsed_args)
        if kind == "CHUNK_SET" and "chunk_get_text" in tool_name:
            ref_type = "fulltext"
        elif kind == "CHUNK_SET" and "search" in tool_name and ref_type is None:
            ref_type = "snippet"

        inferred.append(
            CapabilityRequirement(
                kind=kind,
                ref_type=ref_type,
                namespace=namespace,
                bindings_hash=bindings_hash,
            )
        )

    return inferred


def _trim_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def _short_requirement(req: CapabilityRequirement) -> str:
    return req.short_label()


def _has_explicit_chunk_or_entity_refs(args: dict[str, Any] | None) -> bool:
    if not isinstance(args, dict):
        return False

    def _has_values(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (list, tuple, set)):
            return any(_has_values(item) for item in value)
        return True

    has_explicit_chunk_refs = _has_values(args.get("chunk_id")) or _has_values(args.get("chunk_ids"))
    has_explicit_entity_refs = (
        _has_values(args.get("entity_id"))
        or _has_values(args.get("entity_ids"))
        or _has_values(args.get("entity_name"))
        or _has_values(args.get("entity_names"))
    )
    return bool(has_explicit_chunk_refs or has_explicit_entity_refs)


def _is_arg_self_contained_tool(tool_name: str) -> bool:
    """Compatibility fallback for self-contained tools without explicit contract metadata."""
    return tool_name in ARG_SELF_CONTAINED_TOOL_NAMES


def _contract_declares_no_artifact_prereqs(contract: dict[str, Any] | None) -> bool:
    if not isinstance(contract, dict):
        return False
    raw = contract.get("artifact_prereqs")
    return (
        (isinstance(raw, str) and raw.strip().lower() == "none")
        or bool(contract.get("artifact_prereqs_none"))
        or bool(contract.get("self_contained"))
    )


def _tool_declares_no_artifact_prereqs(
    tool_name: str,
    contract: dict[str, Any] | None,
) -> bool:
    """Prefer declarative contract metadata; retain legacy fallback list for compatibility."""
    if _contract_declares_no_artifact_prereqs(contract):
        return True
    return _is_arg_self_contained_tool(tool_name)


def _normalize_tool_contracts(raw: Any) -> dict[str, dict[str, Any]]:
    """Normalize raw tool contract payload into a predictable dict shape."""
    if not isinstance(raw, dict):
        return {}

    normalized: dict[str, dict[str, Any]] = {}
    for tool_name, spec in raw.items():
        if not isinstance(tool_name, str) or not tool_name.strip():
            continue
        if not isinstance(spec, dict):
            continue

        requires_all_caps = _normalize_capability_requirements(spec.get("requires_all"))
        requires_any_caps = _normalize_capability_requirements(spec.get("requires_any"))
        requires_all_caps.extend(
            _normalize_capability_requirements(spec.get("requires_all_capabilities"))
        )
        requires_any_caps.extend(
            _normalize_capability_requirements(spec.get("requires_any_capabilities"))
        )

        produces_caps = _normalize_capability_requirements(spec.get("produces"))
        produces_caps.extend(
            _normalize_capability_requirements(spec.get("produces_capabilities"))
        )

        requires_all = {req.kind for req in requires_all_caps}
        requires_any = {req.kind for req in requires_any_caps}
        produces = {req.kind for req in produces_caps}

        normalized[tool_name.strip()] = {
            "requires_all_artifacts": requires_all,
            "requires_any_artifacts": requires_any,
            "requires_all_capabilities": requires_all_caps,
            "requires_any_capabilities": requires_any_caps,
            "produces_artifacts": produces,
            "produces_capabilities": produces_caps,
            "is_control": bool(spec.get("is_control", False)),
            "artifact_prereqs": (
                str(spec.get("artifact_prereqs", "")).strip().lower()
                if isinstance(spec.get("artifact_prereqs"), str)
                else None
            ),
            "artifact_prereqs_none": (
                str(spec.get("artifact_prereqs", "")).strip().lower() == "none"
            ),
        }

    return normalized


def _effective_contract_requirements(
    tool_name: str,
    contract: dict[str, Any],
    parsed_args: dict[str, Any] | None,
) -> tuple[set[str], set[str]]:
    """Resolve dynamic per-call requirements for selected tools."""
    requires_all = set(contract.get("requires_all_artifacts") or contract.get("requires_all") or set())
    requires_any = set(contract.get("requires_any_artifacts") or contract.get("requires_any") or set())

    args = parsed_args or {}

    declarative_no_prereqs = _contract_declares_no_artifact_prereqs(contract)
    if declarative_no_prereqs:
        return set(), set()

    if _is_arg_self_contained_tool(tool_name) and _has_explicit_chunk_or_entity_refs(args):
        # Explicit IDs/names are self-contained: the call can be valid even when
        # the runtime artifact tracker has not yet materialized CHUNK_SET/ENTITY_SET.
        return set(), set()

    return requires_all, requires_any


def _validate_tool_contract_call(
    *,
    tool_name: str,
    contract: dict[str, Any],
    parsed_args: dict[str, Any] | None,
    available_artifacts: set[str],
    available_capabilities: dict[str, set[tuple[str | None, str | None, str | None]]] | None = None,
    available_bindings: dict[str, str | None] | None = None,
) -> ToolCallValidation:
    """Validate whether a tool call is composable given currently available artifacts."""
    call_bindings = extract_bindings_from_tool_args(parsed_args)
    capability_state = available_capabilities or {}

    if contract.get("is_control"):
        return ToolCallValidation(
            is_valid=True,
            call_bindings=call_bindings,
        )

    if call_bindings:
        bindings_ok, bindings_reason, _conflicts, normalized_call_bindings = check_binding_conflicts(
            available_bindings=available_bindings,
            proposed_bindings=call_bindings,
        )
        if not bindings_ok:
            return ToolCallValidation(
                is_valid=False,
                reason=bindings_reason,
                error_code=EVENT_CODE_TOOL_VALIDATION_BINDING_CONFLICT,
                failure_phase="binding_validation",
                call_bindings=normalized_call_bindings,
                missing_requirements=[],
            )

    requires_all, requires_any = _effective_contract_requirements(
        tool_name, contract, parsed_args,
    )

    missing_all = sorted(requires_all - available_artifacts)
    if missing_all:
        return ToolCallValidation(
            is_valid=False,
            reason=f"{tool_name} requires all of {sorted(requires_all)}; missing {missing_all}",
            error_code=EVENT_CODE_TOOL_VALIDATION_MISSING_PREREQUISITE,
            failure_phase="input_validation",
            call_bindings=call_bindings,
            missing_requirements=[{"kind": k} for k in missing_all],
        )

    if requires_any and not (requires_any & available_artifacts):
        missing_any = sorted(requires_any)
        return ToolCallValidation(
            is_valid=False,
            reason=f"{tool_name} requires one of {sorted(requires_any)}; available {sorted(available_artifacts)}",
            error_code=EVENT_CODE_TOOL_VALIDATION_MISSING_PREREQUISITE,
            failure_phase="input_validation",
            call_bindings=call_bindings,
            missing_requirements=[{"kind": k} for k in missing_any],
        )

    declarative_no_prereqs = _contract_declares_no_artifact_prereqs(contract)
    skip_capability_requirements = declarative_no_prereqs or (
        _is_arg_self_contained_tool(tool_name)
        and _has_explicit_chunk_or_entity_refs(parsed_args)
    )
    if skip_capability_requirements:
        return ToolCallValidation(
            is_valid=True,
            call_bindings=call_bindings,
        )

    missing_capabilities: list[CapabilityRequirement] = []
    for req in (contract.get("requires_all_capabilities") or []):
        if isinstance(req, CapabilityRequirement) and not _capability_state_has(capability_state, req):
            missing_capabilities.append(req)
    if missing_capabilities:
        missing_payload = [req.to_dict() for req in missing_capabilities]
        return ToolCallValidation(
            is_valid=False,
            reason=(
                f"{tool_name} missing required capabilities: "
                + ", ".join(_short_requirement(req) for req in missing_capabilities)
            ),
            error_code=EVENT_CODE_TOOL_VALIDATION_MISSING_CAPABILITY,
            failure_phase="input_validation",
            call_bindings=call_bindings,
            missing_requirements=missing_payload,
        )

    requires_any_caps = [
        req
        for req in (contract.get("requires_any_capabilities") or [])
        if isinstance(req, CapabilityRequirement)
    ]
    if requires_any_caps and not any(_capability_state_has(capability_state, req) for req in requires_any_caps):
        missing_payload = [req.to_dict() for req in requires_any_caps]
        return ToolCallValidation(
            is_valid=False,
            reason=(
                f"{tool_name} requires one capability from: "
                + ", ".join(_short_requirement(req) for req in requires_any_caps)
            ),
            error_code=EVENT_CODE_TOOL_VALIDATION_MISSING_CAPABILITY,
            failure_phase="input_validation",
            call_bindings=call_bindings,
            missing_requirements=missing_payload,
        )

    return ToolCallValidation(
        is_valid=True,
        call_bindings=call_bindings,
    )


def _find_repair_tools_for_missing_requirements(
    *,
    current_tool_name: str,
    missing_requirements: list[dict[str, Any]],
    normalized_tool_contracts: dict[str, dict[str, Any]],
    available_artifacts: set[str],
    available_capabilities: dict[str, set[tuple[str | None, str | None, str | None]]],
    available_bindings: dict[str, str | None] | None,
    max_repair_tools: int,
) -> list[str]:
    """Suggest legal-now tools that can produce currently missing requirements."""
    if max_repair_tools <= 0:
        return []

    required_caps: list[CapabilityRequirement] = []
    for raw in missing_requirements or []:
        req = _capability_requirement_from_raw(raw)
        if req is not None:
            required_caps.append(req)
    if not required_caps:
        return []

    required_kinds: set[str] = {req.kind for req in required_caps}
    candidates: list[tuple[tuple[int, int, int, int, int, int, str], str]] = []
    for tool_name in sorted(normalized_tool_contracts):
        if tool_name == current_tool_name:
            continue
        contract = normalized_tool_contracts.get(tool_name)
        if not isinstance(contract, dict):
            continue
        if bool(contract.get("is_control")):
            continue

        produced_caps = _contract_output_capabilities(contract)
        produced_kinds = _contract_outputs(contract)
        can_repair = False
        for req in required_caps:
            for produced in produced_caps:
                if _capability_requirement_matches(produced, req):
                    can_repair = True
                    break
            if can_repair:
                break
            if req.kind in produced_kinds:
                can_repair = True
                break
        if not can_repair:
            continue

        validation = _validate_tool_contract_call(
            tool_name=tool_name,
            contract=contract,
            parsed_args=None,
            available_artifacts=available_artifacts,
            available_capabilities=available_capabilities,
            available_bindings=available_bindings,
        )
        if validation.is_valid:
            raw_requires_all = set(contract.get("requires_all_artifacts") or contract.get("requires_all") or set())
            raw_requires_any = set(contract.get("requires_any_artifacts") or contract.get("requires_any") or set())

            # Prefer tools that can bootstrap from QUERY_TEXT/search over tools
            # that depend on the same missing artifact they are trying to repair.
            exact_capability_match = 1
            for req in required_caps:
                if any(
                    produced.kind == req.kind
                    and (
                        req.ref_type is None
                        or produced.ref_type is None
                        or produced.ref_type == req.ref_type
                    )
                    and (
                        req.namespace is None
                        or produced.namespace is None
                        or produced.namespace == req.namespace
                    )
                    and (
                        req.bindings_hash is None
                        or produced.bindings_hash is None
                        or produced.bindings_hash == req.bindings_hash
                    )
                    for produced in produced_caps
                ):
                    exact_capability_match = 0
                    break

            self_dependency_penalty = 1 if any(kind in raw_requires_all for kind in required_kinds) else 0
            query_bootstrap_penalty = 0 if ("QUERY_TEXT" in raw_requires_all or "QUERY_TEXT" in raw_requires_any) else 1
            search_penalty = 0 if "search" in tool_name else 1
            aggregator_penalty = 1 if any(tok in tool_name for tok in ("aggregator", "score", "optimize")) else 0
            prereq_penalty = len(raw_requires_all) + (1 if raw_requires_any else 0)
            get_text_penalty = 1 if "get_text" in tool_name else 0

            rank_key = (
                self_dependency_penalty,
                query_bootstrap_penalty,
                search_penalty,
                exact_capability_match,
                aggregator_penalty,
                prereq_penalty + get_text_penalty,
                tool_name,
            )
            candidates.append((rank_key, tool_name))

    candidates.sort(key=lambda item: item[0])
    return [name for _, name in candidates[:max_repair_tools]]


def _filter_tools_for_disclosure(
    *,
    openai_tools: list[dict[str, Any]],
    normalized_tool_contracts: dict[str, dict[str, Any]],
    available_artifacts: set[str],
    available_capabilities: dict[str, set[tuple[str | None, str | None, str | None]]],
    available_bindings: dict[str, str | None] | None = None,
    max_unavailable: int = DEFAULT_TOOL_DISCLOSURE_MAX_UNAVAILABLE,
    max_missing_per_tool: int = DEFAULT_TOOL_DISCLOSURE_MAX_MISSING_PER_TOOL,
    max_repair_tools: int = DEFAULT_TOOL_DISCLOSURE_MAX_REPAIR_TOOLS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Return tools currently composable from artifact state + hidden reasons.

    Unknown tools (no contract declared) are kept visible to avoid false negatives.
    Control tools stay visible regardless of artifact state.
    """
    if not openai_tools:
        return [], [], 0
    if not normalized_tool_contracts:
        return list(openai_tools), [], 0

    visible: list[dict[str, Any]] = []
    hidden_all: list[dict[str, Any]] = []
    for tool_def in openai_tools:
        fn = tool_def.get("function", {}) if isinstance(tool_def, dict) else {}
        tool_name = str(fn.get("name", "")).strip()
        if not tool_name:
            visible.append(tool_def)
            continue

        contract = normalized_tool_contracts.get(tool_name)
        if not isinstance(contract, dict):
            visible.append(tool_def)
            continue

        # Keep explicit-ID/self-contained tools visible so disclosure does not
        # create false negatives before arguments are known.
        if _tool_declares_no_artifact_prereqs(tool_name, contract):
            visible.append(tool_def)
            continue

        validation = _validate_tool_contract_call(
            tool_name=tool_name,
            contract=contract,
            parsed_args=None,
            available_artifacts=available_artifacts,
            available_capabilities=available_capabilities,
            available_bindings=available_bindings,
        )
        if validation.is_valid:
            visible.append(tool_def)
        else:
            hidden_all.append({
                "tool": tool_name,
                "reason": validation.reason,
                "missing_requirements": list(validation.missing_requirements or []),
                "missing_count": len(validation.missing_requirements or []),
            })

    hidden_all.sort(key=lambda h: (int(h.get("missing_count", 0) or 0), str(h.get("tool", ""))))
    hidden_total = len(hidden_all)

    if max_missing_per_tool >= 0:
        for item in hidden_all:
            missing = item.get("missing_requirements") or []
            if isinstance(missing, list):
                item["missing_requirements"] = missing[:max_missing_per_tool]
            item["missing_count"] = len(item.get("missing_requirements") or [])
            item["repair_tools"] = _find_repair_tools_for_missing_requirements(
                current_tool_name=str(item.get("tool", "")).strip(),
                missing_requirements=list(item.get("missing_requirements") or []),
                normalized_tool_contracts=normalized_tool_contracts,
                available_artifacts=available_artifacts,
                available_capabilities=available_capabilities,
                available_bindings=available_bindings,
                max_repair_tools=max_repair_tools,
            )

    if max_unavailable >= 0:
        hidden_all = hidden_all[:max_unavailable]

    return visible, hidden_all, hidden_total


def _contract_outputs(contract: dict[str, Any] | None) -> set[str]:
    """Return declared artifact outputs for a tool contract."""
    if not isinstance(contract, dict):
        return set()
    return set(contract.get("produces_artifacts") or contract.get("produces") or set())


def _contract_output_capabilities(contract: dict[str, Any] | None) -> list[CapabilityRequirement]:
    if not isinstance(contract, dict):
        return []
    out = contract.get("produces_capabilities") or []
    return [req for req in out if isinstance(req, CapabilityRequirement)]


def _is_control_tool_name(
    tool_name: str,
    normalized_tool_contracts: dict[str, dict[str, Any]],
) -> bool:
    contract = normalized_tool_contracts.get(tool_name)
    return bool(isinstance(contract, dict) and contract.get("is_control"))


def _disclosure_reason_from_entry(entry: dict[str, Any]) -> str:
    repair_tools = entry.get("repair_tools")
    repair_hint = ""
    if isinstance(repair_tools, list):
        names = [str(n).strip() for n in repair_tools if isinstance(n, str) and n.strip()]
        if names:
            repair_hint = f"; try {', '.join(names)}"

    missing = entry.get("missing_requirements")
    if isinstance(missing, list) and missing:
        labels: list[str] = []
        for item in missing:
            req = _capability_requirement_from_raw(item)
            if req is None:
                continue
            labels.append(_short_requirement(req))
        if labels:
            return "needs " + ", ".join(labels) + repair_hint
    reason = str(entry.get("reason", "")).strip()
    if reason:
        return reason + repair_hint
    return "missing prerequisites" + repair_hint


def _disclosure_message(hidden_entries: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for entry in hidden_entries:
        tool = str(entry.get("tool", "")).strip() or "<unknown>"
        reason = _trim_text(_disclosure_reason_from_entry(entry), DEFAULT_TOOL_DISCLOSURE_REASON_MAX_CHARS)
        parts.append(f"{tool}: {reason}")
    return "; ".join(parts)


def _deficit_labels_from_hidden_entries(hidden_entries: list[dict[str, Any]]) -> list[str]:
    """Canonical missing-artifact/capability labels from hidden tool requirements."""
    labels: set[str] = set()
    for entry in hidden_entries:
        if not isinstance(entry, dict):
            continue
        missing = entry.get("missing_requirements")
        if not isinstance(missing, list):
            continue
        for raw in missing:
            req = _capability_requirement_from_raw(raw)
            if req is None:
                continue
            labels.add(_short_requirement(req))
    return sorted(labels)


def _analyze_lane_closure(
    *,
    normalized_tool_contracts: dict[str, dict[str, Any]],
    initial_artifacts: set[str],
    initial_capabilities: dict[str, set[tuple[str | None, str | None, str | None]]],
    available_bindings: dict[str, str | None] | None,
) -> dict[str, Any]:
    """Advisory closure analysis for current lane (toolset + policies).

    This intentionally ignores argument-level dynamics and provides best-effort
    reachability from the declared initial state.
    """
    reachable_artifacts: set[str] = set(initial_artifacts)
    reachable_capabilities: dict[str, set[tuple[str | None, str | None, str | None]]] = {
        kind: set(entries)
        for kind, entries in (initial_capabilities or {}).items()
    }

    max_iters = max(1, len(normalized_tool_contracts) * 2)
    for _ in range(max_iters):
        changed = False
        for tool_name in sorted(normalized_tool_contracts):
            contract = normalized_tool_contracts.get(tool_name)
            if not isinstance(contract, dict):
                continue
            if bool(contract.get("is_control")):
                continue
            validation = _validate_tool_contract_call(
                tool_name=tool_name,
                contract=contract,
                parsed_args=None,
                available_artifacts=reachable_artifacts,
                available_capabilities=reachable_capabilities,
                available_bindings=available_bindings,
            )
            if not validation.is_valid:
                continue
            for kind in _contract_outputs(contract):
                if kind not in reachable_artifacts:
                    reachable_artifacts.add(kind)
                    changed = True
            for produced in _contract_output_capabilities(contract):
                if _capability_state_add(
                    reachable_capabilities,
                    kind=produced.kind,
                    ref_type=produced.ref_type,
                    namespace=produced.namespace,
                    bindings_hash=produced.bindings_hash,
                ):
                    changed = True
        if not changed:
            break

    unresolved_tools: list[dict[str, Any]] = []
    for tool_name in sorted(normalized_tool_contracts):
        contract = normalized_tool_contracts.get(tool_name)
        if not isinstance(contract, dict):
            continue
        if bool(contract.get("is_control")):
            continue
        validation = _validate_tool_contract_call(
            tool_name=tool_name,
            contract=contract,
            parsed_args=None,
            available_artifacts=reachable_artifacts,
            available_capabilities=reachable_capabilities,
            available_bindings=available_bindings,
        )
        if validation.is_valid:
            continue
        unresolved_tools.append(
            {
                "tool": tool_name,
                "error_code": validation.error_code,
                "missing_requirements": list(validation.missing_requirements or []),
            }
        )

    return {
        "lane_closed": len(unresolved_tools) == 0,
        "reachable_artifacts": sorted(reachable_artifacts),
        "reachable_capabilities": _capability_state_snapshot(reachable_capabilities),
        "unresolved_tools": unresolved_tools,
        "unresolved_tool_count": len(unresolved_tools),
    }


def _is_retrieval_no_hits_result(
    *,
    tool_name: str,
    result_text: str | None,
) -> bool:
    """Best-effort retrieval no-hit detector for taxonomy accounting."""
    if not result_text or _is_budget_exempt_tool(tool_name):
        return False

    lower = result_text.lower()
    if "no results" in lower or "not found" in lower or "\"status\": \"no_results\"" in lower:
        return True

    try:
        payload = _json.loads(result_text)
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False

    for key in ("results", "entities", "chunks", "relationships", "neighbors", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return len(value) == 0

    for key in ("count", "total", "hits"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return int(value) == 0
    return False


_PRIMARY_FAILURE_PRIORITY: tuple[str, ...] = (
    "provider",
    "composability",
    "policy",
    "retrieval",
    "control_churn",
    "reasoning",
)

_TERMINAL_FAILURE_EVENT_CODES: frozenset[str] = frozenset({
    EVENT_CODE_PROVIDER_EMPTY,
    EVENT_CODE_PROVIDER_CREDITS_EXHAUSTED,
    EVENT_CODE_FINALIZATION_CIRCUIT_BREAKER_OPEN,
    EVENT_CODE_RETRIEVAL_STAGNATION,
    EVENT_CODE_CONTROL_CHURN_THRESHOLD,
    EVENT_CODE_REQUIRED_SUBMIT_NOT_ATTEMPTED,
    EVENT_CODE_REQUIRED_SUBMIT_NOT_ACCEPTED,
})


def _first_terminal_failure_event_code(failure_event_codes: list[str]) -> str | None:
    """Return earliest terminal blocker event code in event order, if present."""
    for code in failure_event_codes:
        if code in _TERMINAL_FAILURE_EVENT_CODES:
            return code
    return None


def _classify_failure_signals(
    *,
    failure_event_codes: list[str],
    retrieval_no_hits_count: int,
    control_loop_suppressed_calls: int,
    force_final_reason: str | None,
    submit_answer_succeeded: bool,
) -> tuple[str, list[str]]:
    """Deterministic primary/secondary failure classes from event codes + counters."""
    classes: set[str] = set()

    for code in failure_event_codes:
        cls = _failure_class_for_event_code(code)
        if cls is not None:
            classes.add(cls)

    if retrieval_no_hits_count > 0:
        classes.add("retrieval")
    if control_loop_suppressed_calls > 0:
        classes.add("control_churn")
    if force_final_reason is not None and not classes and not submit_answer_succeeded:
        classes.add("reasoning")

    ordered = [c for c in _PRIMARY_FAILURE_PRIORITY if c in classes]
    if not ordered:
        return "none", []
    return ordered[0], ordered[1:]


def _failure_class_for_event_code(code: str) -> str | None:
    """Map event code to one stable failure class for rollups/classification."""
    if code in {
        EVENT_CODE_SUBMIT_FORCED_ACCEPT_BUDGET_EXHAUSTION,
        EVENT_CODE_SUBMIT_FORCED_ACCEPT_TURN_EXHAUSTION,
        EVENT_CODE_SUBMIT_FORCED_ACCEPT_FORCED_FINAL,
    }:
        # This event records a degraded success path (forced-answer acceptance
        # after runtime exhaustion), not an incompletion/failure class.
        return None
    if code == EVENT_CODE_FINALIZATION_TOOL_CALL_DISALLOWED:
        return "policy"
    if code.startswith("REQUIRED_SUBMIT_"):
        return "policy"
    if code.startswith("PROVIDER_"):
        return "provider"
    if code.startswith("FINALIZATION_"):
        return "provider"
    if code.startswith("TOOL_VALIDATION_REJECTED_") or code == EVENT_CODE_NO_LEGAL_NONCONTROL_TOOLS:
        return "composability"
    if code.startswith("RETRIEVAL_"):
        return "retrieval"
    if code.startswith("CONTROL_CHURN_"):
        return "control_churn"
    if code.startswith("REASONING_"):
        return "reasoning"
    return None


def _is_responses_api_raw_response(raw_response: Any) -> bool:
    """Best-effort check for Responses API objects (vs chat completions)."""
    if raw_response is None:
        return False
    has_output = hasattr(raw_response, "output")
    has_choices = hasattr(raw_response, "choices")
    return bool(has_output and not has_choices)


# ---------------------------------------------------------------------------
# MCP Agent Loop
# ---------------------------------------------------------------------------


def _import_mcp() -> tuple[Any, ...]:
    """Lazily import mcp client components.

    Returns:
        (stdio_client, StdioServerParameters, ClientSession)
    """
    try:
        from mcp.client.stdio import (
            StdioServerParameters,
            stdio_client,
        )
        from mcp import ClientSession
    except ImportError:
        raise ImportError(
            "mcp package is required for MCP agent loop. "
            "Install with: pip install llm_client[mcp]"
        ) from None
    return stdio_client, StdioServerParameters, ClientSession


async def _start_servers(
    mcp_servers: dict[str, dict[str, Any]],
    stack: AsyncExitStack,
    init_timeout: float,
) -> tuple[dict[str, Any], dict[str, str], list[dict[str, Any]]]:
    """Start MCP servers and discover tools.

    Returns:
        (sessions, tool_to_server, openai_tools)
    """
    stdio_client, StdioServerParameters, ClientSession = _import_mcp()

    sessions: dict[str, Any] = {}
    tool_to_server: dict[str, str] = {}
    openai_tools: list[dict[str, Any]] = []

    for server_name, server_cfg in mcp_servers.items():
        params = StdioServerParameters(
            command=server_cfg["command"],
            args=server_cfg.get("args", []),
            env=server_cfg.get("env"),
            cwd=server_cfg.get("cwd"),
        )

        read_stream, write_stream = await stack.enter_async_context(
            stdio_client(params)
        )
        session = await stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await asyncio.wait_for(session.initialize(), timeout=init_timeout)
        sessions[server_name] = session

        # Discover tools
        tools_result = await session.list_tools()
        for tool in tools_result.tools:
            if tool.name in tool_to_server:
                logger.warning(
                    "Duplicate tool %r from server %r (already from %r)",
                    tool.name, server_name, tool_to_server[tool.name],
                )
                continue
            tool_to_server[tool.name] = server_name
            openai_tools.append(_mcp_tool_to_openai(tool))

    logger.info(
        "MCP agent loop: %d tools from %d servers",
        len(openai_tools), len(sessions),
    )
    return sessions, tool_to_server, openai_tools


async def _execute_tool_calls(
    tool_calls: list[dict[str, Any]],
    sessions: dict[str, Any],
    tool_to_server: dict[str, str],
    max_result_length: int,
    require_tool_reasoning: bool = DEFAULT_REQUIRE_TOOL_REASONING,
) -> tuple[list[MCPToolCallRecord], list[dict[str, Any]]]:
    """Execute tool calls against MCP servers.

    Returns:
        (records, tool_messages) — records for tracking, messages to append
    """
    records: list[MCPToolCallRecord] = []
    tool_messages: list[dict[str, Any]] = []

    for tc in tool_calls:
        fn = tc.get("function", {})
        tool_name = fn.get("name", "")
        arguments_str = fn.get("arguments", "{}")
        tc_id = tc.get("id", "")

        try:
            arguments = (
                _json.loads(arguments_str)
                if isinstance(arguments_str, str)
                else arguments_str
            )
        except _json.JSONDecodeError as exc:
            logger.error("Failed to parse tool call arguments for %s: %s", tool_name, str(arguments_str)[:200])
            record = MCPToolCallRecord(
                server=tool_to_server.get(tool_name) or "unknown",
                tool=tool_name,
                arguments={},
                error=f"JSON parse error: {exc}",
            )
            tool_messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": f"ERROR: Invalid JSON arguments: {exc}",
            })
            records.append(record)
            continue

        server_name = tool_to_server.get(tool_name)
        if not isinstance(arguments, dict):
            arguments = {}

        tool_reasoning_raw = arguments.pop(TOOL_REASONING_FIELD, None)
        tool_reasoning = None
        if isinstance(tool_reasoning_raw, str):
            stripped = tool_reasoning_raw.strip()
            if stripped:
                tool_reasoning = stripped

        record = MCPToolCallRecord(
            server=server_name or "unknown",
            tool=tool_name,
            arguments=arguments,
            tool_reasoning=tool_reasoning,
        )

        if require_tool_reasoning and not tool_reasoning:
            record.error = f"Missing required argument: {TOOL_REASONING_FIELD}"
            tool_messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": _json.dumps({"error": record.error}),
            })
            records.append(record)
            continue

        t0 = time.monotonic()
        if server_name is None:
            record.error = f"Unknown tool: {tool_name}"
            tool_content = _json.dumps({"error": record.error})
        else:
            try:
                session = sessions[server_name]
                mcp_result = await session.call_tool(tool_name, arguments)

                parts: list[str] = []
                for content_item in mcp_result.content or []:
                    if hasattr(content_item, "text"):
                        parts.append(content_item.text)
                    else:
                        parts.append(str(content_item))
                tool_content = "\n".join(parts)
                tool_content = _truncate(tool_content, max_result_length)

                if mcp_result.isError:
                    record.error = tool_content
                else:
                    record.result = tool_content
            except Exception as e:
                record.error = str(e)
                tool_content = _json.dumps({"error": str(e)})

        record.latency_s = round(time.monotonic() - t0, 3)
        records.append(record)

        tool_messages.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "content": tool_content,
        })

    return records, tool_messages


async def _inner_acall_llm(
    model: str,
    messages: list[dict[str, Any]],
    **kwargs: Any,
) -> LLMCallResult:
    """Call acall_llm from client module. Separate function for testability."""
    from llm_client.client import acall_llm
    return await acall_llm(model, messages, **kwargs)


async def _acall_with_mcp(
    model: str,
    messages: list[dict[str, Any]],
    mcp_servers: dict[str, dict[str, Any]] | None = None,
    *,
    mcp_sessions: MCPSessionPool | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_tool_calls: int | None = DEFAULT_MAX_TOOL_CALLS,
    require_tool_reasoning: bool = DEFAULT_REQUIRE_TOOL_REASONING,
    mcp_init_timeout: float = DEFAULT_MCP_INIT_TIMEOUT,
    tool_result_max_length: int = DEFAULT_TOOL_RESULT_MAX_LENGTH,
    max_message_chars: int = DEFAULT_MAX_MESSAGE_CHARS,
    enforce_tool_contracts: bool = DEFAULT_ENFORCE_TOOL_CONTRACTS,
    progressive_tool_disclosure: bool = DEFAULT_PROGRESSIVE_TOOL_DISCLOSURE,
    suppress_control_loop_calls: bool = DEFAULT_SUPPRESS_CONTROL_LOOP_CALLS,
    tool_contracts: dict[str, dict[str, Any]] | None = None,
    initial_artifacts: list[str] | tuple[str, ...] | None = DEFAULT_INITIAL_ARTIFACTS,
    initial_bindings: dict[str, Any] | None = None,
    timeout: int = 60,
    **kwargs: Any,
) -> LLMCallResult:
    """Run an MCP tool-calling agent loop with any litellm model.

    Starts MCP server subprocesses, discovers tools, then loops:
    1. Call LLM with tool definitions
    2. If LLM returns tool_calls → execute via MCP → append results
    3. Repeat until LLM returns text (no tool calls) or max_turns

    Args:
        model: Any litellm model string (NOT an agent model)
        messages: Initial messages (system + user)
        mcp_servers: Dict of server_name -> {command, args?, env?, cwd?}
            (ignored if mcp_sessions is provided)
        mcp_sessions: Pre-started MCPSessionPool for server reuse across calls.
            When provided, servers are NOT started/stopped per call.
        max_turns: Maximum loop iterations
        max_tool_calls: Maximum tool calls before final forced answer. None disables.
        require_tool_reasoning: If True, reject tool calls missing tool_reasoning.
        mcp_init_timeout: Seconds to wait for server startup
        tool_result_max_length: Max chars per tool result (truncated if longer)
        enforce_tool_contracts: If True, reject tool calls violating tool contracts.
        progressive_tool_disclosure: If True, hide tools whose contracts are not currently satisfiable.
        suppress_control_loop_calls: If True, suppress repeated submit/todo control calls.
        tool_contracts: Optional per-tool composability contracts.
        initial_artifacts: Artifact kinds available before any tool call.
        initial_bindings: Binding state available before any tool call.
        timeout: Per-turn LLM call timeout
        **kwargs: Passed through to acall_llm (retry, hooks, etc.)
    """
    agent_result = MCPAgentResult()
    messages = list(messages)  # don't mutate caller's list
    total_cost = 0.0
    final_content = ""
    final_finish_reason = "stop"

    if mcp_sessions is not None:
        # Reuse existing sessions — no server spawn/teardown
        sessions = mcp_sessions.sessions
        tool_to_server = mcp_sessions.tool_to_server
        openai_tools = mcp_sessions.openai_tools

        if not openai_tools:
            raise ValueError("MCPSessionPool has no tools — was it entered as context manager?")

        async def _mcp_executor(
            tool_calls: list[dict[str, Any]], max_len: int,
        ) -> tuple[list[MCPToolCallRecord], list[dict[str, Any]]]:
            return await _execute_tool_calls(
                tool_calls,
                sessions,
                tool_to_server,
                max_len,
                require_tool_reasoning=require_tool_reasoning,
            )

        final_content, final_finish_reason = await _agent_loop(
            model, messages, openai_tools,
            agent_result,
            _mcp_executor,
            max_turns,
            max_tool_calls,
            require_tool_reasoning,
            tool_result_max_length,
            max_message_chars,
            enforce_tool_contracts,
            progressive_tool_disclosure,
            suppress_control_loop_calls,
            tool_contracts,
            initial_artifacts,
            initial_bindings,
            timeout,
            kwargs,
        )
        total_cost = sum(r.latency_s for r in agent_result.tool_calls)  # placeholder; real cost tracked below

    elif mcp_servers is not None:
        async with AsyncExitStack() as stack:
            # Start servers and discover tools
            sessions, tool_to_server, openai_tools = await _start_servers(
                mcp_servers, stack, mcp_init_timeout,
            )

            if not openai_tools:
                raise ValueError(
                    f"No tools discovered from MCP servers: {list(mcp_servers.keys())}"
                )

            async def _mcp_executor(
                tool_calls: list[dict[str, Any]], max_len: int,
            ) -> tuple[list[MCPToolCallRecord], list[dict[str, Any]]]:
                return await _execute_tool_calls(
                    tool_calls,
                    sessions,
                    tool_to_server,
                    max_len,
                    require_tool_reasoning=require_tool_reasoning,
                )

            final_content, final_finish_reason = await _agent_loop(
                model, messages, openai_tools,
                agent_result,
                _mcp_executor,
                max_turns,
                max_tool_calls,
                require_tool_reasoning,
                tool_result_max_length,
                max_message_chars,
                enforce_tool_contracts,
                progressive_tool_disclosure,
                suppress_control_loop_calls,
                tool_contracts,
                initial_artifacts,
                initial_bindings,
                timeout,
                kwargs,
            )
    else:
        raise ValueError("Either mcp_servers or mcp_sessions must be provided")

    # Cost is accumulated during _agent_loop via agent_result metadata.
    usage = _build_agent_usage(agent_result)
    resolved_model = agent_result.metadata.get("resolved_model")
    if not isinstance(resolved_model, str) or not resolved_model.strip():
        resolved_model = None
    attempted_models = agent_result.metadata.get("attempted_models")
    if not isinstance(attempted_models, list):
        attempted_models = None
    sticky_fallback = agent_result.metadata.get("sticky_fallback")
    if not isinstance(sticky_fallback, bool):
        sticky_fallback = None
    return LLMCallResult(
        content=final_content,
        usage=usage,
        cost=agent_result.metadata.get("total_cost", 0.0),
        model=resolved_model or model,
        requested_model=model,
        resolved_model=resolved_model,
        routing_trace={
            "attempted_models": attempted_models,
            "sticky_fallback": sticky_fallback,
        },
        finish_reason=final_finish_reason,
        raw_response=agent_result,
        warnings=agent_result.warnings,
    )


async def _agent_loop(
    model: str,
    messages: list[dict[str, Any]],
    openai_tools: list[dict[str, Any]],
    agent_result: MCPAgentResult,
    executor: Any,  # Callable[[list, int], Awaitable[tuple[list[MCPToolCallRecord], list[dict]]]]
    max_turns: int,
    max_tool_calls: int | None,
    require_tool_reasoning: bool,
    tool_result_max_length: int,
    max_message_chars: int = DEFAULT_MAX_MESSAGE_CHARS,
    enforce_tool_contracts: bool = DEFAULT_ENFORCE_TOOL_CONTRACTS,
    progressive_tool_disclosure: bool = DEFAULT_PROGRESSIVE_TOOL_DISCLOSURE,
    suppress_control_loop_calls: bool = DEFAULT_SUPPRESS_CONTROL_LOOP_CALLS,
    tool_contracts: dict[str, dict[str, Any]] | None = None,
    initial_artifacts: list[str] | tuple[str, ...] | None = DEFAULT_INITIAL_ARTIFACTS,
    initial_bindings: dict[str, Any] | None = None,
    timeout: int = 60,
    kwargs: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Core agent loop shared by MCP, direct-tool, and session-pool paths.

    Args:
        executor: async callable (tool_calls, max_result_length) -> (records, tool_messages).
            For MCP: wraps _execute_tool_calls with bound sessions.
            For direct tools: wraps execute_direct_tool_calls with bound tool_map.

    Returns (final_content, final_finish_reason).
    """
    kwargs = dict(kwargs or {})

    total_cost = 0.0
    final_content = ""
    final_finish_reason = "stop"
    effective_model = model
    attempted_models: list[str] = []
    sticky_fallback = False
    tool_call_counts: dict[str, int] = {}
    tool_loop_nudges: dict[str, int] = {}
    tool_error_counts: dict[tuple[str, str], int] = {}
    tool_error_nudges: dict[tuple[str, str], int] = {}
    submit_requires_todo_progress = False
    submit_requires_new_evidence = False
    todo_done_error_streak: dict[str, int] = {}
    control_loop_suppressed_calls = 0
    last_budget_remaining: int | None = None
    rejected_missing_reasoning_calls = 0
    tool_call_turns_total = 0
    tool_call_empty_text_turns = 0
    responses_tool_call_empty_text_turns = 0
    tool_arg_coercions = 0
    tool_arg_coercion_calls = 0
    tool_arg_validation_rejections = 0
    tool_disclosure_turns = 0
    tool_disclosure_hidden_total = 0
    tool_disclosure_unavailable_msgs = 0
    tool_disclosure_unavailable_reason_chars = 0
    tool_disclosure_unavailable_reason_tokens_est = 0
    tool_disclosure_repair_suggestions = 0
    no_legal_noncontrol_turns = 0
    deficit_no_progress_streak = 0
    max_deficit_no_progress_streak = 0
    deficit_no_progress_nudges = 0
    deficit_no_progress_last_nudged = 0
    prev_turn_deficit_digest: str | None = None
    prev_turn_had_evidence_tools = False
    zero_exec_tool_turn_streak = 0
    max_zero_exec_tool_turn_streak = 0
    plain_text_no_tool_turn_streak = 0
    context_compactions = 0
    context_compacted_messages = 0
    context_compacted_chars = 0
    context_tool_result_clearings = 0
    context_tool_results_cleared = 0
    context_tool_result_cleared_chars = 0
    gate_rejected_calls = 0
    gate_violation_events: list[dict[str, Any]] = []
    contract_rejected_calls = 0
    contract_violation_events: list[dict[str, Any]] = []
    failure_event_codes: list[str] = []
    autofilled_tool_reasoning_calls = 0
    autofilled_tool_reasoning_by_tool: dict[str, int] = {}
    submit_validation_reason_counts: dict[str, int] = {}
    evidence_signatures: set[str] = set()
    submit_evidence_digest_at_last_failure: str | None = None
    evidence_digest_change_count = 0
    evidence_turns_total = 0
    evidence_turns_with_new_evidence = 0
    evidence_turns_without_new_evidence = 0
    retrieval_no_hits_count = 0
    tool_state = _initialize_agent_tool_state(
        openai_tools=openai_tools,
        tool_contracts=tool_contracts,
        initial_artifacts=initial_artifacts,
        initial_bindings=initial_bindings,
        kwargs=kwargs,
        enforce_tool_contracts=enforce_tool_contracts,
        warning_sink=agent_result.warnings,
    )
    normalized_tool_contracts = tool_state.normalized_tool_contracts
    tool_parameter_index = tool_state.tool_parameter_index
    available_artifacts = tool_state.available_artifacts
    initial_artifact_snapshot = tool_state.initial_artifact_snapshot
    available_capabilities = tool_state.available_capabilities
    initial_capability_snapshot = tool_state.initial_capability_snapshot
    available_bindings = tool_state.available_bindings
    initial_binding_snapshot = tool_state.initial_binding_snapshot
    lane_closure_analysis = tool_state.lane_closure_analysis
    artifact_timeline = tool_state.artifact_timeline
    requires_submit_answer = tool_state.requires_submit_answer
    submit_answer_succeeded = False
    submit_answer_call_count = 0
    force_final_reason: str | None = None
    pending_submit_todo_ids: list[str] = []

    trace_id = str(kwargs.get("trace_id", "")).strip() or None
    task = str(kwargs.get("task", "")).strip() or None
    foundation_session_id = new_session_id()
    foundation_actor_id = "agent:mcp_loop:default:1"
    foundation_events: list[dict[str, Any]] = []
    foundation_event_types: dict[str, int] = {}
    foundation_event_validation_errors = 0
    foundation_events_logged = 0
    _io_log: Any | None = None
    try:
        from llm_client import io_log as _io_log_module  # local import to avoid module-cycle hazards
        _io_log = _io_log_module
        active_run_id = _io_log_module.get_active_experiment_run_id()
    except Exception:
        active_run_id = None
    foundation_run_id = coerce_run_id(
        active_run_id if isinstance(active_run_id, str) else None,
        trace_id,
    )
    runtime_policy = _resolve_agent_runtime_policy(
        model=model,
        max_turns=max_turns,
        max_tool_calls=max_tool_calls,
        require_tool_reasoning=require_tool_reasoning,
        enforce_tool_contracts=enforce_tool_contracts,
        progressive_tool_disclosure=progressive_tool_disclosure,
        suppress_control_loop_calls=suppress_control_loop_calls,
        tool_result_max_length=tool_result_max_length,
        max_message_chars=max_message_chars,
        kwargs=kwargs,
        warning_sink=agent_result.warnings,
    )
    run_config_spec = runtime_policy.run_config_spec
    run_config_hash = runtime_policy.run_config_hash
    forced_final_max_attempts = runtime_policy.forced_final_max_attempts
    forced_final_circuit_breaker_threshold = runtime_policy.forced_final_circuit_breaker_threshold
    forced_final_breaker_effective = runtime_policy.forced_final_breaker_effective
    force_submit_retry_on_max_tool_calls = runtime_policy.force_submit_retry_on_max_tool_calls
    accept_forced_answer_on_max_tool_calls = runtime_policy.accept_forced_answer_on_max_tool_calls
    finalization_fallback_models = runtime_policy.finalization_fallback_models
    retrieval_stagnation_turns = runtime_policy.retrieval_stagnation_turns
    retrieval_stagnation_action = runtime_policy.retrieval_stagnation_action
    tool_result_keep_recent = runtime_policy.tool_result_keep_recent
    tool_result_context_preview_chars = runtime_policy.tool_result_context_preview_chars

    forced_final_attempts = 0
    forced_final_circuit_breaker_opened = False
    retrieval_stagnation_streak = 0
    retrieval_stagnation_streak_max = 0
    retrieval_stagnation_alerted_for_current_streak = False
    retrieval_stagnation_triggered = False
    retrieval_stagnation_turn: int | None = None
    finalization_events: list[str] = []
    finalization_fallback_used = False
    finalization_fallback_succeeded = False
    finalization_fallback_attempts: list[dict[str, Any]] = []
    finalization_primary_model: str | None = None

    def _emit_foundation_event(payload: dict[str, Any]) -> None:
        nonlocal foundation_event_validation_errors, foundation_events_logged
        strict_foundation_schema = _foundation_schema_strict_enabled()
        try:
            validated = validate_foundation_event(payload)
        except Exception as exc:
            foundation_event_validation_errors += 1
            warning = f"FOUNDATION_EVENT_INVALID: {type(exc).__name__}: {exc}"
            agent_result.warnings.append(warning)
            logger.warning(warning)
            if strict_foundation_schema:
                raise RuntimeError(warning) from exc
            return

        foundation_events.append(validated)
        event_type = str(validated.get("event_type", "unknown"))
        foundation_event_types[event_type] = foundation_event_types.get(event_type, 0) + 1

        if _io_log is None:
            return
        try:
            _io_log.log_foundation_event(
                event=validated,
                caller="llm_client.mcp_agent",
                task=task,
                trace_id=trace_id,
            )
            foundation_events_logged += 1
        except Exception as exc:
            warning = f"FOUNDATION_EVENT_LOG_FAILED: {type(exc).__name__}: {exc}"
            agent_result.warnings.append(warning)
            logger.warning(warning)

    if (
        "max_tokens" not in kwargs
        and "max_completion_tokens" not in kwargs
    ):
        # Tool-calling turns do not need long free-form completions; smaller
        # completions reduce context-window pressure on long agent traces.
        kwargs = dict(kwargs)
        kwargs["max_completion_tokens"] = 4096

    for turn in range(max_turns):
        if max_tool_calls is not None:
            budgeted_calls_used = _count_budgeted_records(agent_result.tool_calls)
            remaining_tool_calls = max_tool_calls - budgeted_calls_used
            if remaining_tool_calls <= 0:
                force_final_reason = "max_tool_calls"
                logger.warning(
                    "Agent loop exhausted max_tool_calls=%d after %d turns (%d budgeted, %d total tool calls); forcing final answer",
                    max_tool_calls,
                    turn,
                    budgeted_calls_used,
                    len(agent_result.tool_calls),
                )
                break
            if remaining_tool_calls != last_budget_remaining:
                budget_msg = {
                    "role": "user",
                    "content": (
                        f"[SYSTEM: Retrieval-tool budget: {budgeted_calls_used}/{max_tool_calls} used, "
                        f"{remaining_tool_calls} remaining. Plan carefully. "
                        f"Budget-exempt tools: {', '.join(sorted(BUDGET_EXEMPT_TOOL_NAMES))}. "
                        "If you already have enough evidence, submit your answer now.]"
                    ),
                }
                messages.append(budget_msg)
                agent_result.conversation_trace.append(budget_msg)
                last_budget_remaining = remaining_tool_calls

        agent_result.turns = turn + 1

        cleared_count, cleared_chars = _clear_old_tool_results_for_context(
            messages,
            keep_recent=tool_result_keep_recent,
            preview_chars=tool_result_context_preview_chars,
        )
        if cleared_count:
            context_tool_result_clearings += 1
            context_tool_results_cleared += cleared_count
            context_tool_result_cleared_chars += cleared_chars
            warning = (
                "CONTEXT_TOOL_RESULT_CLEARING: replaced "
                f"{cleared_count} older tool payload(s) with compact stubs "
                f"(saved ~{cleared_chars} chars; keep_recent={tool_result_keep_recent})."
            )
            agent_result.warnings.append(warning)
            logger.warning(warning)

        compacted_count, compacted_chars, current_chars = _compact_tool_history_for_context(
            messages, max_message_chars,
        )
        if compacted_count:
            context_compactions += 1
            context_compacted_messages += compacted_count
            context_compacted_chars += compacted_chars
            warning = (
                "CONTEXT_COMPACTION: compacted "
                f"{compacted_count} tool message(s), saved ~{compacted_chars} chars "
                f"(history ~{current_chars} chars)"
            )
            agent_result.warnings.append(warning)
            logger.warning(warning)

        disclosed_tools = openai_tools
        hidden_disclosure: list[dict[str, Any]] = []
        hidden_disclosure_total = 0
        disclosure_repair_hints: list[str] = []
        current_turn_deficit_labels: list[str] = []
        current_turn_deficit_digest: str | None = None
        if progressive_tool_disclosure and normalized_tool_contracts:
            disclosed_tools, hidden_disclosure, hidden_disclosure_total = _filter_tools_for_disclosure(
                openai_tools=openai_tools,
                normalized_tool_contracts=normalized_tool_contracts,
                available_artifacts=available_artifacts,
                available_capabilities=available_capabilities,
                available_bindings=available_bindings,
            )
            if hidden_disclosure:
                tool_disclosure_turns += 1
                tool_disclosure_hidden_total += hidden_disclosure_total
                tool_disclosure_repair_suggestions += sum(
                    len(entry.get("repair_tools") or [])
                    for entry in hidden_disclosure
                    if isinstance(entry, dict)
                )
                seen_repair_hints: set[str] = set()
                for entry in hidden_disclosure:
                    if not isinstance(entry, dict):
                        continue
                    for candidate in (entry.get("repair_tools") or []):
                        name = str(candidate or "").strip()
                        if not name or name in seen_repair_hints:
                            continue
                        seen_repair_hints.add(name)
                        disclosure_repair_hints.append(name)
                current_turn_deficit_labels = _deficit_labels_from_hidden_entries(hidden_disclosure)
                if current_turn_deficit_labels:
                    current_turn_deficit_digest = sha256_json(current_turn_deficit_labels).replace("sha256:", "")
                hidden_names = [h["tool"] for h in hidden_disclosure]
                disclosure_reason = _disclosure_message(hidden_disclosure)
                disclosure_msg = {
                    "role": "user",
                    "content": (
                        "[SYSTEM: Currently unavailable tools with missing requirements: "
                        f"{disclosure_reason}]"
                    ),
                }
                disclosure_chars = len(str(disclosure_msg.get("content", "")))
                tool_disclosure_unavailable_msgs += 1
                tool_disclosure_unavailable_reason_chars += disclosure_chars
                tool_disclosure_unavailable_reason_tokens_est += max(
                    1, disclosure_chars // DEFAULT_TOOL_DISCLOSURE_TOKEN_CHARS,
                )
                messages.append(disclosure_msg)
                agent_result.conversation_trace.append(disclosure_msg)
                logger.info(
                    "TOOL_DISCLOSURE turn=%d exposed=%d/%d hidden=%s available_artifacts=%s available_capabilities=%s",
                    turn + 1,
                    len(disclosed_tools),
                    len(openai_tools),
                    hidden_names,
                    sorted(available_artifacts),
                    _capability_state_snapshot(available_capabilities),
                )
                agent_result.warnings.append(
                    "TOOL_DISCLOSURE: hidden currently incompatible tools on turn "
                    f"{turn + 1}: {', '.join(hidden_names)}"
                )

        if prev_turn_had_evidence_tools and current_turn_deficit_digest:
            if prev_turn_deficit_digest == current_turn_deficit_digest:
                deficit_no_progress_streak += 1
            else:
                deficit_no_progress_streak = 0
        elif not prev_turn_had_evidence_tools:
            deficit_no_progress_streak = 0

        if deficit_no_progress_streak > max_deficit_no_progress_streak:
            max_deficit_no_progress_streak = deficit_no_progress_streak

        if (
            deficit_no_progress_streak >= 2
            and deficit_no_progress_streak > deficit_no_progress_last_nudged
            and current_turn_deficit_labels
        ):
            deficit_no_progress_nudges += 1
            deficit_no_progress_last_nudged = deficit_no_progress_streak
            suggested = [
                name
                for name in disclosure_repair_hints
                if name and not _is_budget_exempt_tool(name)
            ][:3]
            suggested_hint = f" Try: {', '.join(suggested)}." if suggested else ""
            deficit_msg = {
                "role": "user",
                "content": (
                    "[SYSTEM: Recent evidence calls did not reduce unresolved artifacts. "
                    "Still missing: "
                    + ", ".join(current_turn_deficit_labels[:4])
                    + ". Choose the next call to close one missing requirement."
                    + suggested_hint
                    + "]"
                ),
            }
            messages.append(deficit_msg)
            agent_result.conversation_trace.append(deficit_msg)

        if progressive_tool_disclosure and normalized_tool_contracts:
            noncontrol_disclosed = [
                t for t in disclosed_tools
                if isinstance(t, dict)
                and not _is_control_tool_name(
                    str(t.get("function", {}).get("name", "")).strip(),
                    normalized_tool_contracts,
                )
            ]
            if not noncontrol_disclosed:
                no_legal_noncontrol_turns += 1
                warning = (
                    "TOOL_DISCLOSURE: no legal non-control tools available this turn. "
                    "Use conversion/planning/control tools to unlock capabilities."
                )
                agent_result.warnings.append(warning)
                logger.warning(warning)
                failure_event_codes.append(EVENT_CODE_NO_LEGAL_NONCONTROL_TOOLS)
                _emit_foundation_event(
                    {
                        "event_id": new_event_id(),
                        "event_type": "ToolFailed",
                        "timestamp": now_iso(),
                        "run_id": foundation_run_id,
                        "session_id": foundation_session_id,
                        "actor_id": foundation_actor_id,
                        "operation": {"name": "__disclosure__", "version": None},
                        "inputs": {
                            "artifact_ids": sorted(available_artifacts),
                            "params": {
                                "hidden_tools": [h.get("tool") for h in hidden_disclosure],
                                "hidden_total": hidden_disclosure_total,
                            },
                            "bindings": dict(available_bindings),
                        },
                        "outputs": {"artifact_ids": [], "payload_hashes": []},
                        "failure": {
                            "error_code": EVENT_CODE_NO_LEGAL_NONCONTROL_TOOLS,
                            "category": "validation",
                            "phase": "input_validation",
                            "retryable": True,
                            "tool_name": "__disclosure__",
                            "user_message": warning,
                            "debug_ref": None,
                        },
                    }
                )

        try:
            result = await _inner_acall_llm(
                effective_model, messages, timeout=timeout, tools=disclosed_tools, **kwargs,
            )
        except Exception as exc:
            error_text = str(exc).strip() or f"{type(exc).__name__}"
            (
                is_provider_failure,
                event_code,
                provider_classification,
                retryable_provider,
            ) = _provider_failure_classification(
                exc,
                error_text,
            )
            if not event_code:
                event_code = EVENT_CODE_TOOL_RUNTIME_ERROR
            provider_subevent = "first_turn" if turn == 0 else "turn"
            failure_event_codes.append(event_code)
            warning = (
                "AGENT_LLM_CALL_FAILED: turn="
                f"{turn + 1} model={effective_model} error={error_text}"
            )
            agent_result.warnings.append(warning)
            logger.warning(warning)
            _emit_foundation_event(
                {
                    "event_id": new_event_id(),
                    "event_type": "ToolFailed",
                    "timestamp": now_iso(),
                    "run_id": foundation_run_id,
                    "session_id": foundation_session_id,
                    "actor_id": foundation_actor_id,
                    "operation": {"name": "_inner_acall_llm", "version": None},
                    "inputs": {
                        "artifact_ids": sorted(available_artifacts),
                        "params": {
                            "turn": turn + 1,
                            "error": error_text,
                            "provider_classification": provider_classification or "",
                            "provider_subevent": provider_subevent if is_provider_failure else "",
                        },
                        "bindings": dict(available_bindings),
                    },
                    "outputs": {"artifact_ids": [], "payload_hashes": []},
                    "failure": {
                        "error_code": event_code,
                        "category": "provider" if is_provider_failure else "execution",
                        "phase": "execution",
                        "retryable": bool(retryable_provider),
                        "tool_name": "_inner_acall_llm",
                        "user_message": error_text,
                        "debug_ref": None,
                    },
                }
            )
            final_content = error_text
            final_finish_reason = "error"
            break

        # Track per-turn diagnostics
        result_model = str(result.model).strip() or effective_model
        agent_result.models_used.add(result_model)
        if result_model and result_model not in attempted_models:
            attempted_models.append(result_model)
        if result.warnings:
            agent_result.warnings.extend(result.warnings)

        # Sticky fallback: if inner call fell back to a different model,
        # use that model for remaining turns (avoids re-hitting dead primary).
        if result_model != effective_model:
            agent_result.warnings.append(
                f"STICKY_FALLBACK: {effective_model} failed, "
                f"using {result_model} for remaining turns"
            )
            effective_model = result_model
            sticky_fallback = True

        _emit_foundation_event(
            {
                "event_id": new_event_id(),
                "event_type": "LLMCalled",
                "timestamp": now_iso(),
                "run_id": foundation_run_id,
                "session_id": foundation_session_id,
                "actor_id": foundation_actor_id,
                "operation": {"name": "_inner_acall_llm", "version": None},
                "inputs": {
                    "artifact_ids": sorted(available_artifacts),
                    "params": {
                        "turn": turn + 1,
                    "model": effective_model,
                        "capabilities_sha256": sha256_json(_capability_state_snapshot(available_capabilities)),
                    },
                    "bindings": dict(available_bindings),
                },
                "outputs": {
                    "artifact_ids": [],
                    "payload_hashes": [sha256_text(result.content or "")],
                },
                "llm": {
                    "model_id": result_model,
                    "content_persisted": "hash_only",
                    "prompt_sha256": sha256_json(messages),
                    "response_sha256": sha256_text(result.content or ""),
                    "token_usage": dict(result.usage or {}),
                    "cost_usd": float(result.cost or 0.0),
                },
            }
        )

        # Accumulate usage
        inp, out, cached, cache_create = _extract_usage(result.usage or {})
        agent_result.total_input_tokens += inp
        agent_result.total_output_tokens += out
        agent_result.total_cached_tokens += cached
        agent_result.total_cache_creation_tokens += cache_create
        total_cost += result.cost

        # No tool calls → done
        if not result.tool_calls:
            remaining_turns = max_turns - (turn + 1)
            if requires_submit_answer and not submit_answer_succeeded and remaining_turns > 0:
                # Benchmark loops require explicit submit_answer() so scorers can
                # reliably parse the final answer from tool calls.
                if result.content:
                    draft_msg = {
                        "role": "assistant",
                        "content": result.content,
                    }
                    messages.append(draft_msg)
                    agent_result.conversation_trace.append(draft_msg)

                # Two-stage recovery for plain-text replies:
                # 1) first no-tool turn -> continue tool-based solving (avoid premature submit loops)
                # 2) repeated no-tool turn (or near end) -> require explicit submit_answer
                near_end = remaining_turns <= TURN_WARNING_THRESHOLD
                if plain_text_no_tool_turn_streak == 0 and not near_end:
                    continue_nudge = {
                        "role": "user",
                        "content": (
                            "[SYSTEM: Do NOT finalize yet. Continue with tools to resolve remaining TODO atoms. "
                            "Call todo_list() to inspect unresolved steps, gather missing evidence, then submit.]"
                        ),
                    }
                    messages.append(continue_nudge)
                    agent_result.conversation_trace.append(continue_nudge)
                    plain_text_no_tool_turn_streak += 1
                    logger.warning(
                        "Agent loop: model returned plain text without tool calls on turn %d/%d; nudging continued tool use before submission.",
                        turn + 1, max_turns,
                    )
                    continue

                submit_nudge = {
                    "role": "user",
                    "content": (
                        "[SYSTEM: Do NOT answer in plain text. You MUST call "
                        "submit_answer(reasoning, answer) now. Use a short factual "
                        "answer (<=8 words). No additional searches.]"
                    ),
                }
                messages.append(submit_nudge)
                agent_result.conversation_trace.append(submit_nudge)
                plain_text_no_tool_turn_streak += 1
                logger.warning(
                    "Agent loop: model returned plain text without submit_answer on turn %d/%d; nudging explicit submission.",
                    turn + 1, max_turns,
                )
                continue

            final_content = result.content
            final_finish_reason = result.finish_reason
            # Log visibility: empty content on first turn is almost always a model failure
            if not result.content and turn == 0:
                failure_event_codes.append(EVENT_CODE_PROVIDER_EMPTY)
                _emit_foundation_event(
                    {
                        "event_id": new_event_id(),
                        "event_type": "ToolFailed",
                        "timestamp": now_iso(),
                        "run_id": foundation_run_id,
                        "session_id": foundation_session_id,
                        "actor_id": foundation_actor_id,
                        "operation": {"name": "_inner_acall_llm", "version": None},
                        "inputs": {
                            "artifact_ids": sorted(available_artifacts),
                            "params": {
                                "turn": turn + 1,
                                "finish_reason": result.finish_reason,
                                "provider_classification": "provider_empty_candidates",
                                "provider_subevent": "first_turn",
                            },
                            "bindings": dict(available_bindings),
                        },
                        "outputs": {"artifact_ids": [], "payload_hashes": []},
                        "failure": {
                            "error_code": EVENT_CODE_PROVIDER_EMPTY,
                            "category": "provider",
                            "phase": "execution",
                            "retryable": True,
                            "tool_name": "_inner_acall_llm",
                            "user_message": "Provider returned empty content with no tool calls on first turn.",
                            "debug_ref": None,
                        },
                    }
                )
                logger.error(
                    "Agent loop: model=%s returned empty content with 0 tool calls on turn 1 "
                    "(finish_reason=%s). All %d retries + fallback exhausted at the per-turn level.",
                    model, result.finish_reason, kwargs.get("num_retries", 2),
                )
            elif not result.content:
                failure_event_codes.append(EVENT_CODE_PROVIDER_EMPTY)
                _emit_foundation_event(
                    {
                        "event_id": new_event_id(),
                        "event_type": "ToolFailed",
                        "timestamp": now_iso(),
                        "run_id": foundation_run_id,
                        "session_id": foundation_session_id,
                        "actor_id": foundation_actor_id,
                        "operation": {"name": "_inner_acall_llm", "version": None},
                        "inputs": {
                            "artifact_ids": sorted(available_artifacts),
                            "params": {
                                "turn": turn + 1,
                                "finish_reason": result.finish_reason,
                                "provider_classification": "provider_empty_candidates",
                                "provider_subevent": "turn",
                            },
                            "bindings": dict(available_bindings),
                        },
                        "outputs": {"artifact_ids": [], "payload_hashes": []},
                        "failure": {
                            "error_code": EVENT_CODE_PROVIDER_EMPTY,
                            "category": "provider",
                            "phase": "execution",
                            "retryable": True,
                            "tool_name": "_inner_acall_llm",
                            "user_message": "Provider returned empty content with no tool calls.",
                            "debug_ref": None,
                        },
                    }
                )
                logger.warning(
                    "Agent loop: model=%s returned empty content on turn %d/%d "
                    "(finish_reason=%s, %d tool calls so far).",
                    model, turn + 1, max_turns, result.finish_reason,
                    len(agent_result.tool_calls),
                )
            # Capture final assistant message in trace
            if result.content:
                agent_result.conversation_trace.append({
                    "role": "assistant",
                    "content": result.content,
                })
            break

        # Reset plain-text streak once the model returns at least one tool call.
        plain_text_no_tool_turn_streak = 0

        # Append assistant message with tool calls
        tool_calls_this_turn = list(result.tool_calls)
        autofilled_reasoning_tools: list[str] = []
        patched_calls: list[dict[str, Any]] = []
        for tc in tool_calls_this_turn:
            patched, changed = _autofill_tool_reasoning(tc)
            normalized_name = _normalize_tool_call_name_inplace(patched)
            if changed:
                name = normalized_name
                if name:
                    autofilled_reasoning_tools.append(name)
                    autofilled_tool_reasoning_by_tool[name] = (
                        autofilled_tool_reasoning_by_tool.get(name, 0) + 1
                    )
                autofilled_tool_reasoning_calls += 1
            patched_calls.append(patched)
        tool_calls_this_turn = patched_calls
        if autofilled_reasoning_tools:
            agent_result.warnings.append(
                "OBSERVABILITY: auto-filled missing tool_reasoning on tools: "
                + ", ".join(autofilled_reasoning_tools)
            )
        tool_call_turns_total += 1
        if not (result.content or "").strip():
            tool_call_empty_text_turns += 1
            is_responses_turn = _is_responses_api_raw_response(result.raw_response)
            if is_responses_turn:
                responses_tool_call_empty_text_turns += 1
            logger.info(
                "Agent loop metric: turn %d/%d model=%s returned %d tool call(s) with empty assistant text%s",
                turn + 1,
                max_turns,
                result.model,
                len(tool_calls_this_turn),
                " [responses-api]" if is_responses_turn else "",
            )

        if max_tool_calls is not None:
            budgeted_used = _count_budgeted_records(agent_result.tool_calls)
            remaining_tool_calls = max_tool_calls - budgeted_used
            budgeted_requested = _count_budgeted_tool_calls(tool_calls_this_turn)
            if budgeted_requested > remaining_tool_calls:
                tool_calls_this_turn, dropped = _trim_tool_calls_to_budget(
                    tool_calls_this_turn,
                    max(remaining_tool_calls, 0),
                )
                trim_msg = {
                    "role": "user",
                    "content": (
                        f"[SYSTEM: Retrieval-tool budget allows only {remaining_tool_calls} more budgeted call(s). "
                        f"Ignored {dropped} over-budget retrieval call(s). "
                        f"Budget-exempt tools ({', '.join(sorted(BUDGET_EXEMPT_TOOL_NAMES))}) are still allowed.]"
                    ),
                }
                messages.append(trim_msg)
                agent_result.conversation_trace.append(trim_msg)
                logger.warning(
                    "Agent loop trimmed %d over-budget retrieval call(s) on turn %d; budget remaining=%d",
                    dropped,
                    turn + 1,
                    remaining_tool_calls,
                )

        missing_reasoning_tools: list[str] = []
        for tc in tool_calls_this_turn:
            fn = tc.get("function", {})
            tc_name = fn.get("name", "")
            tc_args = _extract_tool_call_args(tc)
            if not isinstance(tc_args, dict):
                missing_reasoning_tools.append(tc_name or "<unknown>")
                continue
            tc_reasoning = tc_args.get(TOOL_REASONING_FIELD)
            if not isinstance(tc_reasoning, str) or not tc_reasoning.strip():
                missing_reasoning_tools.append(tc_name or "<unknown>")

        if missing_reasoning_tools:
            reasoning_nudge = {
                "role": "user",
                "content": (
                    "[SYSTEM: Observability requirement: every tool call must include "
                    f"'{TOOL_REASONING_FIELD}' with one concise sentence explaining why this call is needed."
                    + (
                        " Calls without it are rejected."
                        if require_tool_reasoning else
                        ""
                    )
                    + "]"
                ),
            }
            messages.append(reasoning_nudge)
            agent_result.conversation_trace.append(reasoning_nudge)
            agent_result.warnings.append(
                "OBSERVABILITY: missing tool_reasoning on tools: "
                + ", ".join(missing_reasoning_tools)
            )

        assistant_msg = {
            "role": "assistant",
            "content": result.content or None,
            "tool_calls": tool_calls_this_turn,
        }
        messages.append(assistant_msg)

        # Capture assistant message in trace (with tool call names for readability)
        agent_result.conversation_trace.append({
            "role": "assistant",
            "content": result.content or "",
            "tool_calls": [
                {
                    "id": tc.get("id", ""),
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": tc.get("function", {}).get("arguments", ""),
                }
                for tc in tool_calls_this_turn
            ],
        })

        tool_calls_to_execute = tool_calls_this_turn
        gate_rejected_records: list[MCPToolCallRecord] = []
        gate_rejected_messages: list[dict[str, Any]] = []
        pending_gate_msg: dict[str, Any] | None = None

        filtered_gate_calls: list[dict[str, Any]] = []
        for tc in tool_calls_to_execute:
            tool_name = str(tc.get("function", {}).get("name", "")).strip()
            tc_id = tc.get("id", "")
            parsed_args = _extract_tool_call_args(tc)
            gate_validation = validate_tool_call_inputs(
                tool_name=tool_name or "<unknown>",
                parsed_args=parsed_args,
                tool_parameters=tool_parameter_index.get(tool_name),
                require_tool_reasoning=require_tool_reasoning,
                tool_reasoning_field=TOOL_REASONING_FIELD,
                available_bindings=available_bindings,
                error_code_schema=EVENT_CODE_TOOL_VALIDATION_SCHEMA,
                error_code_missing_reasoning=EVENT_CODE_TOOL_VALIDATION_MISSING_TOOL_REASONING,
                error_code_binding_conflict=EVENT_CODE_TOOL_VALIDATION_BINDING_CONFLICT,
            )
            if gate_validation.is_valid:
                filtered_gate_calls.append(tc)
                continue

            gate_rejected_calls += 1
            if gate_validation.error_code == EVENT_CODE_TOOL_VALIDATION_MISSING_TOOL_REASONING:
                rejected_missing_reasoning_calls += 1
            gate_reason = gate_validation.reason or "Compliance gate rejected tool call."
            gate_error = f"Validation error: {gate_reason}"
            gate_error_code = gate_validation.error_code or EVENT_CODE_TOOL_VALIDATION_SCHEMA
            failure_event_codes.append(gate_error_code)
            gate_violation_events.append(
                {
                    "turn": turn + 1,
                    "tool": tool_name or "<unknown>",
                    "reason": gate_reason,
                    "error_code": gate_error_code,
                    "failure_phase": gate_validation.failure_phase or "input_validation",
                    "available_artifacts": sorted(available_artifacts),
                    "available_capabilities": _capability_state_snapshot(available_capabilities),
                    "available_bindings": dict(available_bindings),
                    "call_bindings": dict(gate_validation.call_bindings),
                    "arg_keys": sorted(parsed_args.keys()) if isinstance(parsed_args, dict) else [],
                    "missing_requirements": list(gate_validation.missing_requirements or []),
                }
            )
            gate_rejected_records.append(
                MCPToolCallRecord(
                    server="__compliance__",
                    tool=tool_name or "<unknown>",
                    arguments=parsed_args if isinstance(parsed_args, dict) else {},
                    error=gate_error,
                )
            )
            gate_rejected_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": _json.dumps(
                        {
                            "error": gate_error,
                            "error_code": gate_error_code,
                            "failure_phase": gate_validation.failure_phase or "input_validation",
                            "call_bindings": gate_validation.call_bindings,
                            "missing_requirements": gate_validation.missing_requirements,
                        }
                    ),
                }
            )
            _emit_foundation_event(
                {
                    "event_id": new_event_id(),
                    "event_type": "ToolFailed",
                    "timestamp": now_iso(),
                    "run_id": foundation_run_id,
                    "session_id": foundation_session_id,
                    "actor_id": foundation_actor_id,
                    "operation": {"name": tool_name or "<unknown>", "version": None},
                    "inputs": {
                        "artifact_ids": sorted(available_artifacts),
                        "params": parsed_args if isinstance(parsed_args, dict) else {},
                        "bindings": dict(available_bindings),
                    },
                    "outputs": {"artifact_ids": [], "payload_hashes": []},
                    "failure": {
                        "error_code": gate_error_code,
                        "category": "validation",
                        "phase": gate_validation.failure_phase or "input_validation",
                        "retryable": True,
                        "tool_name": tool_name or "<unknown>",
                        "user_message": gate_error,
                        "debug_ref": None,
                    },
                }
            )

        tool_calls_to_execute = filtered_gate_calls
        if gate_rejected_records:
            pending_gate_msg = {
                "role": "user",
                "content": (
                    "[SYSTEM: One or more tool calls were rejected by compliance checks "
                    "(schema/binding/observability). Re-issue corrected calls only if still needed.]"
                ),
            }

        contract_rejected_records: list[MCPToolCallRecord] = []
        contract_rejected_messages: list[dict[str, Any]] = []
        pending_contract_msg: dict[str, Any] | None = None
        if enforce_tool_contracts and normalized_tool_contracts:
            filtered_contract_calls: list[dict[str, Any]] = []
            for tc in tool_calls_to_execute:
                tool_name = tc.get("function", {}).get("name", "")
                tc_id = tc.get("id", "")
                parsed_args = _extract_tool_call_args(tc)
                contract = normalized_tool_contracts.get(tool_name)
                if not isinstance(contract, dict):
                    filtered_contract_calls.append(tc)
                    continue

                contract_validation = _validate_tool_contract_call(
                    tool_name=tool_name,
                    contract=contract,
                    parsed_args=parsed_args,
                    available_artifacts=available_artifacts,
                    available_capabilities=available_capabilities,
                    available_bindings=available_bindings,
                )
                if contract_validation.is_valid:
                    filtered_contract_calls.append(tc)
                    continue

                contract_rejected_calls += 1
                err = f"Tool contract violation: {contract_validation.reason}"
                contract_violation_events.append({
                    "turn": turn + 1,
                    "tool": tool_name or "<unknown>",
                    "reason": contract_validation.reason,
                    "error_code": contract_validation.error_code or EVENT_CODE_TOOL_VALIDATION_MISSING_PREREQUISITE,
                    "failure_phase": contract_validation.failure_phase or "input_validation",
                    "available_artifacts": sorted(available_artifacts),
                    "available_capabilities": _capability_state_snapshot(available_capabilities),
                    "available_bindings": dict(available_bindings),
                    "call_bindings": dict(contract_validation.call_bindings),
                    "arg_keys": sorted(parsed_args.keys()) if isinstance(parsed_args, dict) else [],
                    "missing_requirements": list(contract_validation.missing_requirements or []),
                })
                failure_event_codes.append(
                    contract_validation.error_code or EVENT_CODE_TOOL_VALIDATION_MISSING_PREREQUISITE
                )
                contract_rejected_records.append(
                    MCPToolCallRecord(
                        server="__contract__",
                        tool=tool_name or "<unknown>",
                        arguments=parsed_args if isinstance(parsed_args, dict) else {},
                        error=err,
                    )
                )
                contract_rejected_messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": _json.dumps(
                        {
                            "error": err,
                            "error_code": contract_validation.error_code or EVENT_CODE_TOOL_VALIDATION_MISSING_PREREQUISITE,
                            "failure_phase": contract_validation.failure_phase or "input_validation",
                            "call_bindings": contract_validation.call_bindings,
                            "missing_requirements": contract_validation.missing_requirements,
                        }
                    ),
                })
                _emit_foundation_event(
                    {
                        "event_id": new_event_id(),
                        "event_type": "ToolFailed",
                        "timestamp": now_iso(),
                        "run_id": foundation_run_id,
                        "session_id": foundation_session_id,
                        "actor_id": foundation_actor_id,
                        "operation": {"name": tool_name or "<unknown>", "version": None},
                        "inputs": {
                            "artifact_ids": sorted(available_artifacts),
                            "params": parsed_args if isinstance(parsed_args, dict) else {},
                            "bindings": dict(available_bindings),
                        },
                        "outputs": {"artifact_ids": [], "payload_hashes": []},
                        "failure": {
                            "error_code": contract_validation.error_code or EVENT_CODE_TOOL_VALIDATION_MISSING_PREREQUISITE,
                            "category": "validation",
                            "phase": contract_validation.failure_phase or "input_validation",
                            "retryable": False,
                            "tool_name": tool_name or "<unknown>",
                            "user_message": err,
                            "debug_ref": None,
                        },
                    }
                )

            tool_calls_to_execute = filtered_contract_calls
            if contract_rejected_records:
                pending_contract_msg = {
                    "role": "user",
                    "content": (
                        "[SYSTEM: One or more tool calls were rejected by tool composability contracts. "
                        f"Use tools compatible with available artifacts: {sorted(available_artifacts)}. "
                        f"Capabilities: {_capability_state_snapshot(available_capabilities)}. "
                        "Adjust plan and try again.]"
                    ),
                }

        # Control-loop suppression for repeated invalid control actions:
        # 1) block repeated submit_answer attempts until TODO state changes;
        # 2) block repeated todo_update(done) retries on the same TODO after
        #    repeated validation failures.
        current_evidence_digest = _evidence_digest(evidence_signatures)
        suppressed_records: list[MCPToolCallRecord] = []
        suppressed_tool_messages: list[dict[str, Any]] = []
        filtered_tool_calls: list[dict[str, Any]] = []
        if suppress_control_loop_calls:
            for tc in tool_calls_to_execute:
                tool_name = tc.get("function", {}).get("name", "")
                tc_id = tc.get("id", "")
                parsed_args = _extract_tool_call_args(tc) or {}

                if tool_name == "submit_answer" and submit_requires_new_evidence:
                    if (
                        submit_evidence_digest_at_last_failure is not None
                        and current_evidence_digest != submit_evidence_digest_at_last_failure
                    ):
                        submit_requires_new_evidence = False
                        submit_evidence_digest_at_last_failure = None
                    else:
                        err = (
                            "submit_answer suppressed: validator requires NEW evidence before retry. "
                            "Evidence digest has not changed since last rejected submit. "
                            "Run at least one non-control evidence tool call "
                            "(entity_*, chunk_*, relationship_*, subgraph_*) first, then retry submit."
                        )
                        suppressed_records.append(
                            MCPToolCallRecord(
                                server="__agent__",
                                tool=tool_name,
                                arguments=parsed_args,
                                error=err,
                            )
                        )
                        suppressed_tool_messages.append({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": _json.dumps(
                                {
                                    "error": err,
                                    "error_code": EVENT_CODE_CONTROL_LOOP_SUPPRESSED,
                                    "evidence_digest": current_evidence_digest,
                                    "required_evidence_digest_change_from": submit_evidence_digest_at_last_failure,
                                }
                            ),
                        })
                        continue

                if tool_name == "submit_answer" and submit_requires_todo_progress:
                    unresolved_hint = (
                        f" Unfinished TODOs currently blocking submit: {', '.join(pending_submit_todo_ids)}. "
                        "Call todo_update(todo_id=<id>, status='done' with evidence_refs/confidence) "
                        "or status='blocked' with a concise note, then retry submit."
                        if pending_submit_todo_ids else
                        ""
                    )
                    err = (
                        "submit_answer suppressed: TODO state has not changed since the last unfinished-TODO "
                        "submission failure. Call todo_update(...) or todo_list() to unblock first."
                        + unresolved_hint
                    )
                    suppressed_records.append(
                        MCPToolCallRecord(
                            server="__agent__",
                            tool=tool_name,
                            arguments=parsed_args,
                            error=err,
                        )
                    )
                    suppressed_tool_messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": _json.dumps(
                            {
                                "error": err,
                                "error_code": EVENT_CODE_CONTROL_LOOP_SUPPRESSED,
                            }
                        ),
                    })
                    continue

                if tool_name == "todo_update":
                    todo_id = str(parsed_args.get("todo_id", "")).strip()
                    status = str(parsed_args.get("status", "")).strip().lower()
                    if (
                        todo_id and status == "done"
                        and todo_done_error_streak.get(todo_id, 0) >= 2
                    ):
                        blocked_example = (
                            f"todo_update(todo_id='{todo_id}', status='blocked', "
                            "note='blocked: missing evidence/dependency')"
                        )
                        err = (
                            "todo_update(done) suppressed: repeated completion failures for this TODO. "
                            f"TODO={todo_id}. Change strategy first (set status='blocked' with a concise note, "
                            f"or repair upstream bridge/evidence) before retrying done. Example: {blocked_example}."
                        )
                        suppressed_records.append(
                            MCPToolCallRecord(
                                server="__agent__",
                                tool=tool_name,
                                arguments=parsed_args,
                                error=err,
                            )
                        )
                        suppressed_tool_messages.append({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": _json.dumps(
                                {
                                    "error": err,
                                    "error_code": EVENT_CODE_CONTROL_LOOP_SUPPRESSED,
                                    "recovery_policy": {
                                        "change_todo_status_before_retry": True,
                                        "suggested_next_actions": [
                                            blocked_example,
                                            "todo_list()",
                                        ],
                                    },
                                }
                            ),
                        })
                        continue

                filtered_tool_calls.append(tc)
        else:
            filtered_tool_calls = list(tool_calls_to_execute)

        tool_calls_to_execute = filtered_tool_calls
        pending_control_loop_msg: dict[str, Any] | None = None
        if suppressed_records:
            control_loop_suppressed_calls += len(suppressed_records)
            pending_control_loop_msg = {
                "role": "user",
                "content": (
                    "[SYSTEM: Repeated control-call loop detected. Some tool calls were suppressed. "
                    "Update TODO state or change hypotheses before retrying submit/completion.]"
                ),
            }
            for rec in suppressed_records:
                _emit_foundation_event(
                    {
                        "event_id": new_event_id(),
                        "event_type": "ToolFailed",
                        "timestamp": now_iso(),
                        "run_id": foundation_run_id,
                        "session_id": foundation_session_id,
                        "actor_id": foundation_actor_id,
                        "operation": {"name": rec.tool or "<unknown>", "version": None},
                        "inputs": {
                            "artifact_ids": sorted(available_artifacts),
                            "params": rec.arguments if isinstance(rec.arguments, dict) else {},
                            "bindings": dict(available_bindings),
                        },
                        "outputs": {"artifact_ids": [], "payload_hashes": []},
                        "failure": {
                            "error_code": EVENT_CODE_CONTROL_LOOP_SUPPRESSED,
                            "category": "policy",
                            "phase": "post_validation",
                            "retryable": False,
                            "tool_name": rec.tool or "<unknown>",
                            "user_message": rec.error or "suppressed",
                            "debug_ref": None,
                        },
                    }
                )
                failure_event_codes.append(EVENT_CODE_CONTROL_LOOP_SUPPRESSED)

        for tc in tool_calls_to_execute:
            tool_name = tc.get("function", {}).get("name", "") or "<unknown>"
            parsed_args = _extract_tool_call_args(tc)
            _emit_foundation_event(
                {
                    "event_id": new_event_id(),
                    "event_type": "ToolCalled",
                    "timestamp": now_iso(),
                    "run_id": foundation_run_id,
                    "session_id": foundation_session_id,
                    "actor_id": foundation_actor_id,
                    "operation": {"name": tool_name, "version": None},
                    "inputs": {
                        "artifact_ids": sorted(available_artifacts),
                        "params": parsed_args if isinstance(parsed_args, dict) else {},
                        "bindings": dict(available_bindings),
                    },
                    "outputs": {"artifact_ids": [], "payload_hashes": []},
                }
            )

        # Execute valid tool calls via executor
        if tool_calls_to_execute:
            executed_records, executed_tool_messages = await executor(
                tool_calls_to_execute, tool_result_max_length,
            )
        else:
            executed_records, executed_tool_messages = [], []

        for record in executed_records:
            if record.error:
                _emit_foundation_event(
                    {
                        "event_id": new_event_id(),
                        "event_type": "ToolFailed",
                        "timestamp": now_iso(),
                        "run_id": foundation_run_id,
                        "session_id": foundation_session_id,
                        "actor_id": foundation_actor_id,
                        "operation": {"name": record.tool or "<unknown>", "version": None},
                        "inputs": {
                            "artifact_ids": sorted(available_artifacts),
                            "params": record.arguments if isinstance(record.arguments, dict) else {},
                            "bindings": dict(available_bindings),
                        },
                        "outputs": {"artifact_ids": [], "payload_hashes": []},
                        "failure": {
                            "error_code": EVENT_CODE_TOOL_RUNTIME_ERROR,
                            "category": "execution",
                            "phase": "execution",
                            "retryable": False,
                            "tool_name": record.tool or "<unknown>",
                            "user_message": record.error,
                            "debug_ref": None,
                        },
                    }
                )
                failure_event_codes.append(EVENT_CODE_TOOL_RUNTIME_ERROR)
                continue

            observed_bindings = extract_bindings_from_tool_args(record.arguments)
            merged_bindings = merge_binding_state(
                available_bindings=available_bindings,
                observed_bindings=observed_bindings,
            )
            if merged_bindings != available_bindings:
                old_bindings = dict(available_bindings)
                available_bindings = merged_bindings
                _emit_foundation_event(
                    {
                        "event_id": new_event_id(),
                        "event_type": "BindingChanged",
                        "timestamp": now_iso(),
                        "run_id": foundation_run_id,
                        "session_id": foundation_session_id,
                        "actor_id": foundation_actor_id,
                        "operation": {"name": record.tool or "<unknown>", "version": None},
                        "inputs": {
                            "artifact_ids": sorted(available_artifacts),
                            "params": record.arguments if isinstance(record.arguments, dict) else {},
                            "bindings": old_bindings,
                        },
                        "outputs": {"artifact_ids": [], "payload_hashes": []},
                        "binding_change": {
                            "old_bindings": old_bindings,
                            "new_bindings": dict(available_bindings),
                            "reason": "adopted_new_binding_from_successful_tool_call",
                        },
                    }
                )

            produced: set[str] = set()
            produced_capabilities: list[CapabilityRequirement] = []
            if enforce_tool_contracts and normalized_tool_contracts:
                contract = normalized_tool_contracts.get(record.tool)
                produced = _contract_outputs(contract)
                produced_capabilities = _contract_output_capabilities(contract)
                if produced:
                    available_artifacts.update(produced)

            inferred_caps = _infer_output_capabilities(
                tool_name=record.tool,
                parsed_args=record.arguments if isinstance(record.arguments, dict) else {},
                produced_artifacts=produced,
                available_bindings=available_bindings,
            )

            produced_capability_payload: list[dict[str, str]] = []
            for req in produced_capabilities + inferred_caps:
                added = _capability_state_add(
                    available_capabilities,
                    kind=req.kind,
                    ref_type=req.ref_type,
                    namespace=req.namespace,
                    bindings_hash=req.bindings_hash,
                )
                if added:
                    produced_capability_payload.append(req.to_dict())

            if produced:
                artifact_timeline.append({
                    "turn": turn + 1,
                    "phase": "tool_success",
                    "tool": record.tool,
                    "produced": sorted(produced),
                    "produced_capabilities": produced_capability_payload,
                    "available_artifacts": sorted(available_artifacts),
                    "available_capabilities": _capability_state_snapshot(available_capabilities),
                })

            if _is_retrieval_no_hits_result(tool_name=record.tool, result_text=record.result):
                retrieval_no_hits_count += 1
                failure_event_codes.append("RETRIEVAL_NO_HITS")

            _emit_foundation_event(
                {
                    "event_id": new_event_id(),
                    "event_type": "ArtifactCreated",
                    "timestamp": now_iso(),
                    "run_id": foundation_run_id,
                    "session_id": foundation_session_id,
                    "actor_id": foundation_actor_id,
                    "operation": {"name": record.tool or "<unknown>", "version": None},
                    "inputs": {
                        "artifact_ids": sorted(available_artifacts),
                        "params": record.arguments if isinstance(record.arguments, dict) else {},
                        "bindings": dict(available_bindings),
                    },
                    "outputs": {
                        "artifact_ids": sorted(produced),
                        "payload_hashes": (
                            [sha256_text(record.result or "")]
                            if record.result is not None else []
                        ),
                    },
                }
            )

        records = gate_rejected_records + contract_rejected_records + suppressed_records + executed_records
        tool_messages = gate_rejected_messages + contract_rejected_messages + suppressed_tool_messages + executed_tool_messages
        agent_result.tool_calls.extend(records)
        messages.extend(tool_messages)
        if pending_gate_msg is not None:
            messages.append(pending_gate_msg)
            agent_result.conversation_trace.append(pending_gate_msg)
        if pending_contract_msg is not None:
            messages.append(pending_contract_msg)
            agent_result.conversation_trace.append(pending_contract_msg)
        if pending_control_loop_msg is not None:
            messages.append(pending_control_loop_msg)
            agent_result.conversation_trace.append(pending_control_loop_msg)

        turn_arg_coercions = 0
        turn_arg_coercion_calls = 0
        turn_arg_validation_rejections = 0
        for record in records:
            coercions = getattr(record, "arg_coercions", None) or []
            if coercions:
                turn_arg_coercions += len(coercions)
                turn_arg_coercion_calls += 1
            if record.error and "validation error:" in record.error.lower():
                turn_arg_validation_rejections += 1

        if turn_arg_coercions:
            tool_arg_coercions += turn_arg_coercions
            tool_arg_coercion_calls += turn_arg_coercion_calls
            warning = (
                "TOOL_ARG_COERCION: turn "
                f"{turn + 1}/{max_turns} applied {turn_arg_coercions} coercion(s) "
                f"across {turn_arg_coercion_calls} call(s)"
            )
            agent_result.warnings.append(warning)
            logger.warning(warning)
        if turn_arg_validation_rejections:
            tool_arg_validation_rejections += turn_arg_validation_rejections
            warning = (
                "TOOL_ARG_VALIDATION: turn "
                f"{turn + 1}/{max_turns} rejected {turn_arg_validation_rejections} "
                "tool call(s) due to argument/schema mismatch"
            )
            agent_result.warnings.append(warning)
            logger.warning(warning)

        submitted_answer_value: str | None = None
        submit_error_this_turn = False
        submit_errors: list[str] = []
        submit_reason_codes_this_turn: list[str] = []
        submit_needs_new_evidence_signal = False
        evidence_digest_before_turn = _evidence_digest(evidence_signatures)
        for record in executed_records:
            signature = _tool_evidence_signature(record)
            if signature:
                evidence_signatures.add(signature)
        evidence_digest_after_turn = _evidence_digest(evidence_signatures)
        new_evidence_this_turn = evidence_digest_after_turn != evidence_digest_before_turn
        if new_evidence_this_turn:
            evidence_digest_change_count += 1
        if (
            new_evidence_this_turn
            and submit_requires_new_evidence
            and submit_evidence_digest_at_last_failure is not None
            and evidence_digest_after_turn != submit_evidence_digest_at_last_failure
        ):
            submit_requires_new_evidence = False
            submit_evidence_digest_at_last_failure = None

        evidence_tools_executed = [
            rec.tool
            for rec in executed_records
            if _is_evidence_tool_name(rec.tool) and not rec.error
        ]
        if evidence_tools_executed:
            evidence_turns_total += 1
            if new_evidence_this_turn:
                evidence_turns_with_new_evidence += 1
                retrieval_stagnation_streak = 0
                retrieval_stagnation_alerted_for_current_streak = False
            else:
                evidence_turns_without_new_evidence += 1
                retrieval_stagnation_streak += 1
        else:
            retrieval_stagnation_streak = 0
            retrieval_stagnation_alerted_for_current_streak = False
        if retrieval_stagnation_streak > retrieval_stagnation_streak_max:
            retrieval_stagnation_streak_max = retrieval_stagnation_streak

        # Persist deficit snapshot for next-turn deficit-progress evaluation.
        prev_turn_had_evidence_tools = bool(evidence_tools_executed)
        prev_turn_deficit_digest = current_turn_deficit_digest

        if (
            retrieval_stagnation_streak >= retrieval_stagnation_turns
            and not retrieval_stagnation_alerted_for_current_streak
        ):
            retrieval_stagnation_triggered = True
            if retrieval_stagnation_turn is None:
                retrieval_stagnation_turn = turn + 1
            stagnation_event_code = (
                EVENT_CODE_RETRIEVAL_STAGNATION
                if retrieval_stagnation_action == "force_final"
                else EVENT_CODE_RETRIEVAL_STAGNATION_OBSERVED
            )
            failure_event_codes.append(stagnation_event_code)
            warning = (
                "RETRIEVAL_STAGNATION: "
                f"{retrieval_stagnation_streak} consecutive evidence turns produced no new evidence."
            )
            agent_result.warnings.append(warning)
            logger.warning(warning)
            _emit_foundation_event(
                {
                    "event_id": new_event_id(),
                    "event_type": "ToolFailed",
                    "timestamp": now_iso(),
                    "run_id": foundation_run_id,
                    "session_id": foundation_session_id,
                    "actor_id": foundation_actor_id,
                    "operation": {"name": "__retrieval_stagnation__", "version": None},
                    "inputs": {
                        "artifact_ids": sorted(available_artifacts),
                        "params": {
                            "streak": retrieval_stagnation_streak,
                            "turn": turn + 1,
                            "evidence_tools": evidence_tools_executed,
                        },
                        "bindings": dict(available_bindings),
                    },
                    "outputs": {"artifact_ids": [], "payload_hashes": []},
                    "failure": {
                        "error_code": stagnation_event_code,
                        "category": "policy",
                        "phase": "post_validation",
                        "retryable": False,
                        "tool_name": "__retrieval_stagnation__",
                        "user_message": warning,
                        "debug_ref": None,
                    },
                }
            )
            if retrieval_stagnation_action == "force_final":
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "[SYSTEM: Evidence has stagnated across consecutive retrieval turns. "
                            "Stop repeating equivalent searches. Verify a different bridge entity, "
                            "run a conversion tool, or submit your best answer now.]"
                        ),
                    }
                )
                agent_result.conversation_trace.append(messages[-1])
                force_final_reason = "retrieval_stagnation"
                retrieval_stagnation_alerted_for_current_streak = True
                break
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "[SYSTEM: Evidence stagnation observed. Do NOT repeat equivalent retrieval. "
                        "Pivot strategy now (different bridge, conversion, or graph path) and continue.]"
                    ),
                }
            )
            agent_result.conversation_trace.append(messages[-1])
            retrieval_stagnation_alerted_for_current_streak = True

        for record in records:
            if record.tool != "submit_answer":
                continue
            submit_answer_call_count += 1
            if record.error:
                submit_error_this_turn = True
                submit_errors.append(str(record.error))
                continue
            parsed = _parse_record_result_json(record)
            if parsed is not None:
                status = str(parsed.get("status", "")).strip().lower()
                if status and status not in {"submitted", "ok", "success"}:
                    submit_error_this_turn = True
                    validation_payload = parsed.get("validation_error")
                    reason_code = ""
                    detail = ""
                    recovery_policy = parsed.get("recovery_policy")
                    if (
                        isinstance(recovery_policy, dict)
                        and bool(recovery_policy.get("new_evidence_required_before_retry"))
                    ):
                        submit_needs_new_evidence_signal = True
                    if isinstance(validation_payload, dict):
                        reason_code = str(validation_payload.get("reason_code", "")).strip()
                        detail = str(validation_payload.get("message", "")).strip()
                    if reason_code:
                        submit_validation_reason_counts[reason_code] = (
                            submit_validation_reason_counts.get(reason_code, 0) + 1
                        )
                        submit_reason_codes_this_turn.append(reason_code)
                    err_parts = [f"submit_answer not accepted (status={status})"]
                    if reason_code:
                        err_parts.append(f"reason_code={reason_code}")
                    if detail:
                        err_parts.append(detail)
                    submit_errors.append(" | ".join(err_parts))
                    continue

            submit_answer_succeeded = True
            arg_answer = record.arguments.get("answer") if isinstance(record.arguments, dict) else None
            if isinstance(arg_answer, str) and arg_answer.strip():
                submitted_answer_value = arg_answer.strip()
                continue
            if isinstance(parsed, dict):
                parsed_answer = parsed.get("answer")
                if isinstance(parsed_answer, str) and parsed_answer.strip():
                    submitted_answer_value = parsed_answer.strip()

        todo_progress_this_turn = False
        for record in records:
            if record.tool != "todo_update":
                continue
            args = record.arguments if isinstance(record.arguments, dict) else {}
            todo_id = str(args.get("todo_id", "")).strip()
            status = str(args.get("status", "")).strip().lower()
            parsed = _parse_record_result_json(record)
            result_status = str(parsed.get("status", "")).strip().lower() if isinstance(parsed, dict) else ""
            needs_revision = result_status == "needs_revision"
            if record.error or needs_revision:
                if todo_id and status == "done":
                    todo_done_error_streak[todo_id] = todo_done_error_streak.get(todo_id, 0) + 1
                continue
            todo_progress_this_turn = True
            if todo_id:
                todo_done_error_streak[todo_id] = 0
        if todo_progress_this_turn:
            submit_requires_todo_progress = False

        if submit_error_this_turn:
            todo_blocking_reason_codes = {
                "todos_required",
                "unfinished_todos",
                "blocked_missing_note",
                "todo_missing_evidence_or_span",
            }
            todo_blocked = any(
                code in todo_blocking_reason_codes
                for code in submit_reason_codes_this_turn
            )
            if (
                not todo_blocked
                and "answer_not_grounded" in submit_reason_codes_this_turn
                and any("todo answer_span" in err.lower() for err in submit_errors)
            ):
                todo_blocked = True
            if not todo_blocked:
                todo_blocked = any(
                    (
                        "unfinished todo" in err.lower()
                        or "must use todos" in err.lower()
                        or "blocked todos" in err.lower()
                        or "missing evidence/span" in err.lower()
                    )
                    for err in submit_errors
                )
            pending_submit_todo_ids = []
            for err in submit_errors:
                pending_submit_todo_ids.extend(_extract_unfinished_todo_ids(err))
            # Keep stable order and uniqueness
            if pending_submit_todo_ids:
                deduped_ids: list[str] = []
                seen_ids: set[str] = set()
                for todo_id in pending_submit_todo_ids:
                    key = todo_id.lower()
                    if key in seen_ids:
                        continue
                    seen_ids.add(key)
                    deduped_ids.append(todo_id)
                pending_submit_todo_ids = deduped_ids
            refusal_blocked = any(
                ("refusal-style" in err.lower()) or ("not acceptable" in err.lower())
                for err in submit_errors
            )
            if todo_blocked and not todo_progress_this_turn:
                submit_requires_todo_progress = True
            if submit_needs_new_evidence_signal and not new_evidence_this_turn:
                submit_requires_new_evidence = True
                submit_evidence_digest_at_last_failure = evidence_digest_after_turn
            todo_fix_hint = ""
            if pending_submit_todo_ids:
                todo_fix_hint = (
                    " Unfinished TODO IDs: "
                    + ", ".join(pending_submit_todo_ids)
                    + ". For each unresolved leaf TODO, call "
                    "todo_update(todo_id=<id>, status='blocked', note='why evidence is insufficient') "
                    "or mark done with evidence refs before submit."
                )
            evidence_fix_hint = (
                " Validator requires NEW evidence before retry. "
                "Run at least one non-control evidence tool call "
                "(entity_*/chunk_*/relationship_*/subgraph_*) before submit."
                if submit_requires_new_evidence else
                ""
            )
            retry_submit_msg = {
                "role": "user",
                "content": (
                    "[SYSTEM: submit_answer failed. "
                    + (
                        "Create/update TODOs first: todo_create(...), "
                        "todo_update(..., status='done'), todo_list(). Then call submit_answer again. "
                        if todo_blocked else
                        ""
                    )
                    + todo_fix_hint
                    + evidence_fix_hint
                    + (
                        "Do NOT use refusal text (cannot, unknown, insufficient, no such, not found). "
                        "Submit the single best factual guess from retrieved evidence. "
                        if refusal_blocked else
                        ""
                    )
                    + "Call submit_answer again with answer as a short fact only "
                    "(name/date/number/yes/no, <=8 words).]"
                ),
            }
            messages.append(retry_submit_msg)
            agent_result.conversation_trace.append(retry_submit_msg)
        elif todo_progress_this_turn:
            pending_submit_todo_ids = []

        if tool_calls_this_turn and not tool_calls_to_execute:
            zero_exec_tool_turn_streak += 1
        else:
            zero_exec_tool_turn_streak = 0
        if zero_exec_tool_turn_streak > max_zero_exec_tool_turn_streak:
            max_zero_exec_tool_turn_streak = zero_exec_tool_turn_streak

        if zero_exec_tool_turn_streak >= DEFAULT_TOOL_CALL_STALL_TURNS:
            blocked_tools = sorted({
                str(tc.get("function", {}).get("name", "")).strip() or "<unknown>"
                for tc in tool_calls_this_turn
                if isinstance(tc, dict)
            })
            warning = (
                "CONTROL_CHURN: consecutive turns with tool calls but no executable calls "
                f"({zero_exec_tool_turn_streak} turns). Forcing final answer."
            )
            agent_result.warnings.append(warning)
            logger.warning(warning)
            failure_event_codes.append(EVENT_CODE_CONTROL_CHURN_THRESHOLD)
            stall_msg = {
                "role": "user",
                "content": (
                    "[SYSTEM: Repeated non-executable tool-call loop detected "
                    f"({zero_exec_tool_turn_streak} turns). Stop calling tools and "
                    "submit your best answer now from existing evidence.]"
                ),
            }
            messages.append(stall_msg)
            agent_result.conversation_trace.append(stall_msg)
            _emit_foundation_event(
                {
                    "event_id": new_event_id(),
                    "event_type": "ToolFailed",
                    "timestamp": now_iso(),
                    "run_id": foundation_run_id,
                    "session_id": foundation_session_id,
                    "actor_id": foundation_actor_id,
                    "operation": {"name": "__control_churn__", "version": None},
                    "inputs": {
                        "artifact_ids": sorted(available_artifacts),
                        "params": {
                            "blocked_tools": blocked_tools,
                            "streak": zero_exec_tool_turn_streak,
                            "contract_rejections": len(contract_rejected_records),
                            "suppressed_calls": len(suppressed_records),
                        },
                        "bindings": dict(available_bindings),
                    },
                    "outputs": {"artifact_ids": [], "payload_hashes": []},
                    "failure": {
                        "error_code": EVENT_CODE_CONTROL_CHURN_THRESHOLD,
                        "category": "policy",
                        "phase": "post_validation",
                        "retryable": False,
                        "tool_name": "__control_churn__",
                        "user_message": warning,
                        "debug_ref": None,
                    },
                }
            )
            force_final_reason = "control_churn"
            break

        # Repeated tool-error detection: nudge strategy change instead of
        # repeatedly retrying near-identical invalid calls.
        for record in records:
            if not record.error:
                continue
            sig = _tool_error_signature(str(record.error))
            error_key = (record.tool, sig)
            count = tool_error_counts.get(error_key, 0) + 1
            tool_error_counts[error_key] = count

            last_nudged = tool_error_nudges.get(error_key, 0)
            if count >= 2 and count > last_nudged:
                todo_hint = (
                    " If this is a TODO completion error, reopen upstream TODO(s) "
                    "and gather new evidence before retrying."
                    if record.tool == "todo_update" else
                    ""
                )
                err_msg = {
                    "role": "user",
                    "content": (
                        f"[SYSTEM: Repeated tool error from '{record.tool}': {sig}. "
                        "STOP retrying near-identical calls. Change strategy "
                        "(new evidence, alternate hypothesis, or upstream repair) before continuing.]"
                        + todo_hint
                    ),
                }
                messages.append(err_msg)
                agent_result.conversation_trace.append(err_msg)
                tool_error_nudges[error_key] = count
                logger.warning(
                    "Loop detection: repeated tool error from '%s' (count=%d): %s",
                    record.tool, count, sig,
                )

        # Track tool call frequency for loop detection
        for tc in tool_calls_to_execute:
            tc_name = tc.get("function", {}).get("name", "")
            if tc_name and not _is_budget_exempt_tool(tc_name):
                tool_call_counts[tc_name] = tool_call_counts.get(tc_name, 0) + 1

        # Loop detection: if same tool called 3+ times, nudge model to switch
        for tool_name, count in tool_call_counts.items():
            last_nudged = tool_loop_nudges.get(tool_name, 0)
            if count >= 3 and count % 3 == 0 and count > last_nudged:
                loop_hint = ""
                if disclosure_repair_hints:
                    suggested = [
                        name
                        for name in disclosure_repair_hints
                        if name != tool_name and not _is_budget_exempt_tool(name)
                    ][:3]
                    if suggested:
                        loop_hint = "Try these alternatives now: " + ", ".join(suggested) + ". "
                loop_msg = {
                    "role": "user",
                    "content": (
                        f"[SYSTEM: You have called '{tool_name}' {count} times. "
                        "STOP repeating the same searches. Read what you already have. "
                        + loop_hint
                        + "If still blocked, submit your answer now.]"
                    ),
                }
                messages.append(loop_msg)
                agent_result.conversation_trace.append(loop_msg)
                tool_loop_nudges[tool_name] = count
                logger.warning(
                    "Loop detection: tool '%s' called %d times on turn %d",
                    tool_name, count, turn + 1,
                )

        # Capture tool results in trace
        for tmsg in tool_messages:
            agent_result.conversation_trace.append({
                "role": "tool",
                "tool_call_id": tmsg.get("tool_call_id", ""),
                "content": tmsg.get("content", ""),
            })

        if submit_answer_succeeded:
            final_content = submitted_answer_value or (result.content or "").strip() or "submitted"
            final_finish_reason = "submitted"
            logger.info(
                "Agent loop: submit_answer succeeded on turn %d/%d",
                turn + 1, max_turns,
            )
            break

        logger.debug(
            "Agent turn %d/%d: %d tool calls",
            turn + 1, max_turns, len(tool_calls_this_turn),
        )

        # Turn countdown: warn agent when it's running low on turns
        remaining = max_turns - (turn + 1)
        if remaining == TURN_WARNING_THRESHOLD:
            countdown_msg = {
                "role": "user",
                "content": (
                    f"[SYSTEM: You have {remaining} turns remaining. "
                    "Submit your best answer now based on the evidence you have. "
                    "Do not search for more information.]"
                ),
            }
            messages.append(countdown_msg)
            agent_result.conversation_trace.append(countdown_msg)
    else:
        force_final_reason = "max_turns"

    if force_final_reason is not None and not submit_answer_succeeded:
        forced_final_result = await _execute_forced_finalization(
            force_final_reason=force_final_reason,
            max_turns=max_turns,
            max_message_chars=max_message_chars,
            tool_result_keep_recent=tool_result_keep_recent,
            tool_result_context_preview_chars=tool_result_context_preview_chars,
            messages=messages,
            kwargs=kwargs,
            timeout=timeout,
            effective_model=effective_model,
            attempted_models=attempted_models,
            finalization_fallback_models=finalization_fallback_models,
            forced_final_max_attempts=forced_final_max_attempts,
            forced_final_circuit_breaker_threshold=forced_final_circuit_breaker_threshold,
            available_artifacts=available_artifacts,
            available_bindings=available_bindings,
            foundation_run_id=foundation_run_id,
            foundation_session_id=foundation_session_id,
            foundation_actor_id=foundation_actor_id,
            agent_result=agent_result,
            emit_foundation_event=_emit_foundation_event,
        )
        final_content = forced_final_result.final_content
        final_finish_reason = forced_final_result.final_finish_reason
        finalization_primary_model = forced_final_result.finalization_primary_model
        forced_final_attempts = forced_final_result.forced_final_attempts
        forced_final_circuit_breaker_opened = forced_final_result.forced_final_circuit_breaker_opened
        finalization_fallback_used = forced_final_result.finalization_fallback_used
        finalization_fallback_succeeded = forced_final_result.finalization_fallback_succeeded
        finalization_events = list(forced_final_result.finalization_events)
        finalization_fallback_attempts = list(forced_final_result.finalization_fallback_attempts)
        failure_event_codes.extend(forced_final_result.failure_event_codes)
        context_tool_result_clearings += forced_final_result.context_tool_result_clearings_delta
        context_tool_results_cleared += forced_final_result.context_tool_results_cleared_delta
        context_tool_result_cleared_chars += (
            forced_final_result.context_tool_result_cleared_chars_delta
        )
        context_compactions += forced_final_result.context_compactions_delta
        context_compacted_messages += forced_final_result.context_compacted_messages_delta
        context_compacted_chars += forced_final_result.context_compacted_chars_delta
        total_cost += forced_final_result.total_cost_delta
        agent_result.total_input_tokens += forced_final_result.total_input_tokens_delta
        agent_result.total_output_tokens += forced_final_result.total_output_tokens_delta
        agent_result.total_cached_tokens += forced_final_result.total_cached_tokens_delta
        agent_result.total_cache_creation_tokens += (
            forced_final_result.total_cache_creation_tokens_delta
        )
        agent_result.turns += forced_final_result.turns_delta

    required_submit_missing = requires_submit_answer and not submit_answer_succeeded
    submit_forced_retry_on_budget_exhaustion = False
    submit_forced_accept_on_budget_exhaustion = False
    forced_exhaustion_reason: str | None = None
    if force_final_reason == "max_tool_calls":
        forced_exhaustion_reason = "budget"
    elif force_final_reason == "max_turns":
        forced_exhaustion_reason = "turns"
    elif force_final_reason is not None:
        forced_exhaustion_reason = force_final_reason
    if (
        forced_exhaustion_reason is not None
        and requires_submit_answer
        and not submit_answer_succeeded
    ):
        if force_submit_retry_on_max_tool_calls:
            submit_forced_retry_on_budget_exhaustion = True
            submit_answer_call_count += 1
            if forced_exhaustion_reason == "budget":
                warning = (
                    "SUBMIT_FORCED_RETRY_BUDGET_EXHAUSTION: counted one forced submit retry "
                    "attempt after tool budget exhaustion."
                )
            elif forced_exhaustion_reason == "turns":
                warning = (
                    "SUBMIT_FORCED_RETRY_TURN_EXHAUSTION: counted one forced submit retry "
                    "attempt after max-turn exhaustion."
                )
            else:
                warning = (
                    "SUBMIT_FORCED_RETRY_FORCED_FINAL: counted one forced submit retry "
                    f"attempt after forced-final reason '{forced_exhaustion_reason}'."
                )
            agent_result.warnings.append(warning)
            logger.warning(warning)
        if accept_forced_answer_on_max_tool_calls and final_content.strip():
            submit_forced_accept_on_budget_exhaustion = True
            submit_answer_succeeded = True
            submitted_answer_value = submitted_answer_value or final_content.strip()
            required_submit_missing = False
            if forced_exhaustion_reason == "budget":
                warning = (
                    "SUBMIT_FORCED_ACCEPT_BUDGET_EXHAUSTION: accepted forced-final answer "
                    "without grounding validation because tool budget was exhausted."
                )
                failure_event_codes.append(EVENT_CODE_SUBMIT_FORCED_ACCEPT_BUDGET_EXHAUSTION)
            elif forced_exhaustion_reason == "turns":
                warning = (
                    "SUBMIT_FORCED_ACCEPT_TURN_EXHAUSTION: accepted forced-final answer "
                    "without grounding validation because max turns were exhausted."
                )
                failure_event_codes.append(EVENT_CODE_SUBMIT_FORCED_ACCEPT_TURN_EXHAUSTION)
            else:
                warning = (
                    "SUBMIT_FORCED_ACCEPT_FORCED_FINAL: accepted forced-final answer "
                    "without grounding validation after forced-final termination "
                    f"('{forced_exhaustion_reason}')."
                )
                failure_event_codes.append(EVENT_CODE_SUBMIT_FORCED_ACCEPT_FORCED_FINAL)
            agent_result.warnings.append(warning)
            logger.warning(warning)

    if required_submit_missing:
        submit_failure_code = (
            EVENT_CODE_REQUIRED_SUBMIT_NOT_ATTEMPTED
            if submit_answer_call_count == 0
            else EVENT_CODE_REQUIRED_SUBMIT_NOT_ACCEPTED
        )
        if not failure_event_codes:
            # Classify missing required submit as terminal/policy only when no
            # stronger prior failure signal has already explained the run.
            failure_event_codes.append(submit_failure_code)
        warning = (
            "REQUIRED_SUBMIT: submit_answer tool is available but no accepted submit was recorded. "
            f"submit_answer_call_count={submit_answer_call_count}."
        )
        agent_result.warnings.append(warning)
        logger.warning(warning)
        _emit_foundation_event(
            {
                "event_id": new_event_id(),
                "event_type": "ToolFailed",
                "timestamp": now_iso(),
                "run_id": foundation_run_id,
                "session_id": foundation_session_id,
                "actor_id": foundation_actor_id,
                "operation": {"name": "submit_answer", "version": None},
                "inputs": {
                    "artifact_ids": sorted(available_artifacts),
                    "params": {
                        "submit_answer_call_count": submit_answer_call_count,
                        "submit_answer_succeeded": submit_answer_succeeded,
                        "reason": "required_submit_not_satisfied",
                    },
                    "bindings": dict(available_bindings),
                },
                "outputs": {"artifact_ids": [], "payload_hashes": []},
                "failure": {
                    "error_code": submit_failure_code,
                    "category": "policy",
                    "phase": "post_validation",
                    "retryable": False,
                    "tool_name": "submit_answer",
                    "user_message": warning,
                    "debug_ref": None,
                },
            }
        )

    run_completed = (not required_submit_missing) and (
        submit_answer_succeeded or final_finish_reason != "error"
    )
    failure_summary = _summarize_failure_events(
        failure_event_codes=failure_event_codes,
        retrieval_no_hits_count=retrieval_no_hits_count,
        control_loop_suppressed_calls=control_loop_suppressed_calls,
        force_final_reason=force_final_reason,
        run_completed=run_completed,
    )
    finalization_summary = _summarize_finalization_attempts(
        finalization_fallback_attempts=finalization_fallback_attempts,
        finalization_primary_model=finalization_primary_model,
        forced_final_attempts=forced_final_attempts,
        forced_final_circuit_breaker_opened=forced_final_circuit_breaker_opened,
    )

    hard_bindings_spec = _hard_bindings_spec(available_bindings)
    full_bindings_spec = _full_bindings_spec(available_bindings)
    hard_bindings_hash = _hard_bindings_state_hash(available_bindings)
    full_bindings_hash = _full_bindings_state_hash(available_bindings)
    initial_hard_bindings_spec = _hard_bindings_spec(initial_binding_snapshot)
    initial_full_bindings_spec = _full_bindings_spec(initial_binding_snapshot)
    initial_hard_bindings_hash = _hard_bindings_state_hash(initial_binding_snapshot)
    initial_full_bindings_hash = _full_bindings_state_hash(initial_binding_snapshot)
    final_evidence_digest = _evidence_digest(evidence_signatures)

    agent_result.metadata["total_cost"] = total_cost
    agent_result.metadata["requested_model"] = model
    agent_result.metadata["resolved_model"] = effective_model
    agent_result.metadata["attempted_models"] = attempted_models
    agent_result.metadata["sticky_fallback"] = sticky_fallback
    agent_result.metadata["max_turns"] = max_turns
    agent_result.metadata["max_tool_calls"] = max_tool_calls
    agent_result.metadata["forced_final_attempts"] = forced_final_attempts
    agent_result.metadata["forced_final_max_attempts"] = forced_final_max_attempts
    agent_result.metadata["forced_final_circuit_breaker_threshold"] = forced_final_circuit_breaker_threshold
    agent_result.metadata["forced_final_breaker_effective"] = forced_final_breaker_effective
    agent_result.metadata["force_submit_retry_on_max_tool_calls"] = force_submit_retry_on_max_tool_calls
    agent_result.metadata["accept_forced_answer_on_max_tool_calls"] = accept_forced_answer_on_max_tool_calls
    agent_result.metadata["forced_final_circuit_breaker_opened"] = forced_final_circuit_breaker_opened
    agent_result.metadata["finalization_primary_model"] = finalization_primary_model
    agent_result.metadata["finalization_fallback_models"] = list(finalization_fallback_models)
    agent_result.metadata["finalization_fallback_used"] = finalization_fallback_used
    agent_result.metadata["finalization_fallback_attempt_count"] = finalization_summary.finalization_fallback_attempt_count
    agent_result.metadata["finalization_fallback_usage_rate"] = finalization_summary.finalization_fallback_usage_rate
    agent_result.metadata["finalization_fallback_succeeded"] = finalization_fallback_succeeded
    agent_result.metadata["finalization_fallback_attempts"] = list(finalization_fallback_attempts)
    agent_result.metadata["finalization_attempt_counts_by_model"] = finalization_summary.finalization_attempt_counts_by_model
    agent_result.metadata["finalization_failure_counts_by_model"] = finalization_summary.finalization_failure_counts_by_model
    agent_result.metadata["finalization_success_counts_by_model"] = finalization_summary.finalization_success_counts_by_model
    agent_result.metadata["finalization_failure_code_counts"] = finalization_summary.finalization_failure_code_counts
    agent_result.metadata["provider_empty_attempt_counts_by_model"] = finalization_summary.provider_empty_attempt_counts_by_model
    agent_result.metadata["finalization_breaker_open_rate"] = finalization_summary.finalization_breaker_open_rate
    agent_result.metadata["finalization_breaker_open_by_model"] = finalization_summary.finalization_breaker_open_by_model
    agent_result.metadata["finalization_events"] = list(finalization_events)
    agent_result.metadata["tool_calls_used"] = len(agent_result.tool_calls)
    agent_result.metadata["budgeted_tool_calls_used"] = _count_budgeted_records(agent_result.tool_calls)
    agent_result.metadata["budget_exempt_tools"] = sorted(BUDGET_EXEMPT_TOOL_NAMES)
    agent_result.metadata["requires_submit_answer"] = requires_submit_answer
    agent_result.metadata["submit_answer_call_count"] = submit_answer_call_count
    agent_result.metadata["submit_answer_attempted"] = submit_answer_call_count > 0
    agent_result.metadata["submit_answer_succeeded"] = submit_answer_succeeded
    agent_result.metadata["required_submit_missing"] = required_submit_missing
    agent_result.metadata["submit_forced_retry_on_budget_exhaustion"] = submit_forced_retry_on_budget_exhaustion
    agent_result.metadata["submit_forced_accept_on_budget_exhaustion"] = submit_forced_accept_on_budget_exhaustion
    agent_result.metadata["require_tool_reasoning"] = require_tool_reasoning
    agent_result.metadata["rejected_missing_reasoning_calls"] = rejected_missing_reasoning_calls
    agent_result.metadata["control_loop_suppressed_calls"] = control_loop_suppressed_calls
    agent_result.metadata["submit_validation_reason_counts"] = dict(
        sorted(submit_validation_reason_counts.items())
    )
    agent_result.metadata["tool_call_turns_total"] = tool_call_turns_total
    agent_result.metadata["tool_call_empty_text_turns"] = tool_call_empty_text_turns
    agent_result.metadata["responses_tool_call_empty_text_turns"] = responses_tool_call_empty_text_turns
    agent_result.metadata["tool_call_empty_text_turn_ratio"] = (
        (tool_call_empty_text_turns / tool_call_turns_total)
        if tool_call_turns_total else 0.0
    )
    agent_result.metadata["tool_arg_coercions"] = tool_arg_coercions
    agent_result.metadata["tool_arg_coercion_calls"] = tool_arg_coercion_calls
    agent_result.metadata["tool_arg_validation_rejections"] = tool_arg_validation_rejections
    agent_result.metadata["tool_result_keep_recent"] = tool_result_keep_recent
    agent_result.metadata["tool_result_context_preview_chars"] = tool_result_context_preview_chars
    agent_result.metadata["context_tool_result_clearings"] = context_tool_result_clearings
    agent_result.metadata["context_tool_results_cleared"] = context_tool_results_cleared
    agent_result.metadata["context_tool_result_cleared_chars"] = context_tool_result_cleared_chars
    agent_result.metadata["context_compactions"] = context_compactions
    agent_result.metadata["context_compacted_messages"] = context_compacted_messages
    agent_result.metadata["context_compacted_chars"] = context_compacted_chars
    agent_result.metadata["enforce_tool_contracts"] = enforce_tool_contracts
    agent_result.metadata["progressive_tool_disclosure"] = progressive_tool_disclosure
    agent_result.metadata["suppress_control_loop_calls"] = suppress_control_loop_calls
    agent_result.metadata["tool_gate_rejections"] = gate_rejected_calls
    agent_result.metadata["tool_gate_violation_events"] = gate_violation_events
    agent_result.metadata["tool_contract_rejections"] = contract_rejected_calls
    agent_result.metadata["tool_contract_violation_events"] = contract_violation_events
    agent_result.metadata["tool_contracts_declared"] = sorted(normalized_tool_contracts.keys())
    agent_result.metadata["initial_artifacts"] = initial_artifact_snapshot
    agent_result.metadata["available_artifacts_final"] = sorted(available_artifacts)
    agent_result.metadata["initial_capabilities"] = initial_capability_snapshot
    agent_result.metadata["available_capabilities_final"] = _capability_state_snapshot(available_capabilities)
    agent_result.metadata["artifact_timeline"] = artifact_timeline
    agent_result.metadata["initial_bindings"] = initial_binding_snapshot
    agent_result.metadata["available_bindings_final"] = dict(available_bindings)
    agent_result.metadata["initial_hard_bindings_spec"] = initial_hard_bindings_spec
    agent_result.metadata["initial_full_bindings_spec"] = initial_full_bindings_spec
    agent_result.metadata["initial_hard_bindings_hash"] = initial_hard_bindings_hash
    agent_result.metadata["initial_full_bindings_hash"] = initial_full_bindings_hash
    agent_result.metadata["hard_bindings_spec"] = hard_bindings_spec
    agent_result.metadata["full_bindings_spec"] = full_bindings_spec
    agent_result.metadata["hard_bindings_hash"] = hard_bindings_hash
    agent_result.metadata["full_bindings_hash"] = full_bindings_hash
    agent_result.metadata["run_config_spec"] = run_config_spec
    agent_result.metadata["run_config_hash"] = run_config_hash
    agent_result.metadata["tool_disclosure_turns"] = tool_disclosure_turns
    agent_result.metadata["tool_disclosure_hidden_total"] = tool_disclosure_hidden_total
    agent_result.metadata["tool_disclosure_unavailable_msgs"] = tool_disclosure_unavailable_msgs
    agent_result.metadata["tool_disclosure_unavailable_reason_chars"] = tool_disclosure_unavailable_reason_chars
    agent_result.metadata["tool_disclosure_unavailable_reason_tokens_est"] = tool_disclosure_unavailable_reason_tokens_est
    agent_result.metadata["tool_disclosure_repair_suggestions"] = tool_disclosure_repair_suggestions
    agent_result.metadata["lane_closure_analysis"] = lane_closure_analysis
    agent_result.metadata["no_legal_noncontrol_turns"] = no_legal_noncontrol_turns
    agent_result.metadata["deficit_no_progress_streak_max"] = max_deficit_no_progress_streak
    agent_result.metadata["deficit_no_progress_nudges"] = deficit_no_progress_nudges
    agent_result.metadata["max_zero_exec_tool_turn_streak"] = max_zero_exec_tool_turn_streak
    agent_result.metadata["retrieval_no_hits_count"] = retrieval_no_hits_count
    agent_result.metadata["retrieval_stagnation_turns"] = retrieval_stagnation_turns
    agent_result.metadata["retrieval_stagnation_action"] = retrieval_stagnation_action
    agent_result.metadata["retrieval_stagnation_streak"] = retrieval_stagnation_streak
    agent_result.metadata["retrieval_stagnation_streak_max"] = retrieval_stagnation_streak_max
    agent_result.metadata["retrieval_stagnation_triggered"] = retrieval_stagnation_triggered
    agent_result.metadata["retrieval_stagnation_turn"] = retrieval_stagnation_turn
    agent_result.metadata["evidence_digest_change_count"] = evidence_digest_change_count
    agent_result.metadata["evidence_turns_total"] = evidence_turns_total
    agent_result.metadata["evidence_turns_with_new_evidence"] = evidence_turns_with_new_evidence
    agent_result.metadata["evidence_turns_without_new_evidence"] = evidence_turns_without_new_evidence
    agent_result.metadata["failure_event_codes"] = list(failure_event_codes)
    agent_result.metadata["failure_event_code_counts"] = failure_summary.failure_event_code_counts
    agent_result.metadata["failure_event_class_counts"] = failure_summary.failure_event_class_counts
    agent_result.metadata["provider_failure_event_code_counts"] = failure_summary.provider_failure_event_code_counts
    agent_result.metadata["provider_failure_event_total"] = failure_summary.provider_failure_event_total
    agent_result.metadata["provider_caused_incompletion"] = failure_summary.provider_caused_incompletion
    agent_result.metadata["primary_failure_class"] = failure_summary.primary_failure_class
    agent_result.metadata["secondary_failure_classes"] = failure_summary.secondary_failure_classes
    agent_result.metadata["first_terminal_failure_event_code"] = failure_summary.first_terminal_failure_event_code
    agent_result.metadata["failure_priority_order"] = list(_PRIMARY_FAILURE_PRIORITY)
    agent_result.metadata["terminal_failure_event_codes"] = sorted(_TERMINAL_FAILURE_EVENT_CODES)
    agent_result.metadata["autofilled_tool_reasoning_calls"] = autofilled_tool_reasoning_calls
    agent_result.metadata["autofilled_tool_reasoning_by_tool"] = dict(autofilled_tool_reasoning_by_tool)
    agent_result.metadata["evidence_digest"] = final_evidence_digest
    agent_result.metadata["submit_evidence_digest_at_last_failure"] = submit_evidence_digest_at_last_failure
    agent_result.metadata["foundation_event_count"] = len(foundation_events)
    agent_result.metadata["foundation_event_types"] = dict(foundation_event_types)
    agent_result.metadata["foundation_event_validation_errors"] = foundation_event_validation_errors
    agent_result.metadata["foundation_events_logged"] = foundation_events_logged
    agent_result.metadata["foundation_events"] = foundation_events
    if tool_call_turns_total > 0:
        agent_result.warnings.append(
            "METRIC: tool_call_empty_text_turns="
            f"{tool_call_empty_text_turns}/{tool_call_turns_total}, "
            f"responses_tool_call_empty_text_turns={responses_tool_call_empty_text_turns}"
        )
    if tool_arg_coercions or tool_arg_validation_rejections:
        agent_result.warnings.append(
            "METRIC: tool_arg_coercions="
            f"{tool_arg_coercions} across {tool_arg_coercion_calls} calls, "
            f"tool_arg_validation_rejections={tool_arg_validation_rejections}"
        )
    if evidence_turns_total > 0:
        agent_result.warnings.append(
            "METRIC: evidence_turns="
            f"{evidence_turns_total}, new_evidence_turns={evidence_turns_with_new_evidence}, "
            f"stagnant_evidence_turns={evidence_turns_without_new_evidence}, "
            f"retrieval_stagnation_streak_max={retrieval_stagnation_streak_max}"
        )
    logger.info(
        "Agent loop metrics: tool_call_turns=%d empty_text_tool_call_turns=%d responses_empty_text_tool_call_turns=%d "
        "tool_arg_coercions=%d tool_arg_validation_rejections=%d",
        tool_call_turns_total,
        tool_call_empty_text_turns,
        responses_tool_call_empty_text_turns,
        tool_arg_coercions,
        tool_arg_validation_rejections,
    )
    return final_content, final_finish_reason


# ---------------------------------------------------------------------------
# Direct Python Tool Loop
# ---------------------------------------------------------------------------


async def _acall_with_tools(
    model: str,
    messages: list[dict[str, Any]],
    python_tools: list[Any],
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_tool_calls: int | None = DEFAULT_MAX_TOOL_CALLS,
    require_tool_reasoning: bool = DEFAULT_REQUIRE_TOOL_REASONING,
    tool_result_max_length: int = DEFAULT_TOOL_RESULT_MAX_LENGTH,
    max_message_chars: int = DEFAULT_MAX_MESSAGE_CHARS,
    enforce_tool_contracts: bool = DEFAULT_ENFORCE_TOOL_CONTRACTS,
    progressive_tool_disclosure: bool = DEFAULT_PROGRESSIVE_TOOL_DISCLOSURE,
    suppress_control_loop_calls: bool = DEFAULT_SUPPRESS_CONTROL_LOOP_CALLS,
    tool_contracts: dict[str, dict[str, Any]] | None = None,
    initial_artifacts: list[str] | tuple[str, ...] | None = DEFAULT_INITIAL_ARTIFACTS,
    initial_bindings: dict[str, Any] | None = None,
    timeout: int = 60,
    **kwargs: Any,
) -> LLMCallResult:
    """Run a tool-calling agent loop with direct Python functions.

    Same loop as _acall_with_mcp but calls Python functions in-process
    instead of going through MCP subprocess/stdio/JSON-RPC.

    Args:
        model: Any litellm model string (NOT an agent model).
        messages: Initial messages (system + user).
        python_tools: List of typed Python callables (sync or async).
        max_turns: Maximum loop iterations.
        max_tool_calls: Maximum tool calls before final forced answer. None disables.
        require_tool_reasoning: If True, reject tool calls missing tool_reasoning.
        tool_result_max_length: Max chars per tool result.
        progressive_tool_disclosure: If True, hide tools whose contracts are not currently satisfiable.
        suppress_control_loop_calls: If True, suppress repeated submit/todo control calls.
        initial_bindings: Binding state available before any tool call.
        timeout: Per-turn LLM call timeout.
        **kwargs: Passed through to acall_llm.
    """
    from llm_client.tool_utils import execute_direct_tool_calls, prepare_direct_tools

    tool_map, openai_tools = prepare_direct_tools(python_tools)

    if not openai_tools:
        raise ValueError("python_tools list is empty — no tools to call.")

    agent_result = MCPAgentResult()
    messages = list(messages)  # don't mutate caller's list

    async def _direct_executor(
        tool_calls: list[dict[str, Any]], max_len: int,
    ) -> tuple[list[MCPToolCallRecord], list[dict[str, Any]]]:
        return await execute_direct_tool_calls(
            tool_calls,
            tool_map,
            max_len,
            require_tool_reasoning=require_tool_reasoning,
        )

    final_content, final_finish_reason = await _agent_loop(
        model, messages, openai_tools,
        agent_result,
        _direct_executor,
        max_turns,
        max_tool_calls,
        require_tool_reasoning,
        tool_result_max_length,
        max_message_chars,
        enforce_tool_contracts,
        progressive_tool_disclosure,
        suppress_control_loop_calls,
        tool_contracts,
        initial_artifacts,
        initial_bindings,
        timeout,
        kwargs,
    )

    usage = _build_agent_usage(agent_result)
    resolved_model = agent_result.metadata.get("resolved_model")
    if not isinstance(resolved_model, str) or not resolved_model.strip():
        resolved_model = None
    attempted_models = agent_result.metadata.get("attempted_models")
    if not isinstance(attempted_models, list):
        attempted_models = None
    sticky_fallback = agent_result.metadata.get("sticky_fallback")
    if not isinstance(sticky_fallback, bool):
        sticky_fallback = None
    return LLMCallResult(
        content=final_content,
        usage=usage,
        cost=agent_result.metadata.get("total_cost", 0.0),
        model=resolved_model or model,
        requested_model=model,
        resolved_model=resolved_model,
        routing_trace={
            "attempted_models": attempted_models,
            "sticky_fallback": sticky_fallback,
        },
        finish_reason=final_finish_reason,
        raw_response=agent_result,
        warnings=agent_result.warnings,
    )
