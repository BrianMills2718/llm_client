"""Structured-output tool-calling shim for models without native tool support.

Models like gemini-2.5-flash-lite support structured output (response_format)
but return empty content when ``tools=`` is passed.  This module emulates tool
calling by embedding tool schemas in the system prompt and asking the model to
respond with JSON that encodes either a tool call or a final answer.

The loop mirrors ``_agent_loop`` in mcp_agent.py but never passes ``tools=``
to the LLM — instead it uses ``response_format={"type": "json_object"}``.

Only used for ``python_tools=`` (direct backend).  NOT for MCP, which
requires OpenAI-format ``tool_calls`` in the LLM response.
"""

from __future__ import annotations

import json as _json
import logging
from typing import Any, Callable

from llm_client.client import LLMCallResult
from llm_client.mcp_agent import (
    DEFAULT_MAX_TURNS,
    DEFAULT_TOOL_RESULT_MAX_LENGTH,
    MCPAgentResult,
    _extract_usage,
    _inner_acall_llm,
)
from llm_client.tool_utils import execute_direct_tool_calls, prepare_direct_tools

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first valid JSON object from text that may have trailing garbage."""
    # Find the first '{' and use a decoder to parse just that object
    start = text.find("{")
    if start == -1:
        return None
    decoder = _json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text, start)
        if isinstance(obj, dict):
            return obj
    except _json.JSONDecodeError:
        pass
    return None


# ---------------------------------------------------------------------------
# System-prompt generation
# ---------------------------------------------------------------------------


def _build_tool_system_prompt(openai_tools: list[dict[str, Any]]) -> str:
    """Build the system-prompt section that describes available tools."""
    lines = [
        "You have access to these tools. You MUST respond with valid JSON.",
        "",
        "To use a tool, respond with exactly:",
        '{"action": "tool_call", "tool_name": "<name>", "arguments": {<args matching schema>}}',
        "",
        "When you have enough information to answer, respond with exactly:",
        '{"action": "final_answer", "content": "<your answer>"}',
        "",
        "Available tools:",
    ]
    for tool in openai_tools:
        fn = tool["function"]
        name = fn["name"]
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])

        # Build a compact signature
        sig_parts: list[str] = []
        for pname, pschema in props.items():
            ptype = pschema.get("type", "any")
            default = pschema.get("default")
            if pname in required:
                sig_parts.append(f"{pname}: {ptype}")
            else:
                sig_parts.append(f"{pname}: {ptype} = {_json.dumps(default)}")

        sig = ", ".join(sig_parts)
        lines.append(f"- {name}({sig}): {desc}")
        lines.append(f"  Parameters schema: {_json.dumps(params)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core shim loop
# ---------------------------------------------------------------------------


async def _acall_with_tool_shim(
    model: str,
    messages: list[dict[str, Any]],
    python_tools: list[Callable[..., Any]],
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    tool_result_max_length: int = DEFAULT_TOOL_RESULT_MAX_LENGTH,
    timeout: int = 60,
    **kwargs: Any,
) -> LLMCallResult:
    """Run a tool-calling loop via structured-output JSON emulation.

    Same interface as ``_acall_with_tools`` — callers should not need to
    know whether native tool calling or the shim is in use.
    """
    tool_map, openai_tools = prepare_direct_tools(python_tools)
    if not openai_tools:
        raise ValueError("python_tools list is empty — no tools to call.")

    tool_prompt = _build_tool_system_prompt(openai_tools)
    messages = list(messages)  # don't mutate caller's list

    # Prepend or append tool documentation to the system message
    if messages and messages[0].get("role") == "system":
        messages[0] = dict(messages[0])
        messages[0]["content"] = messages[0]["content"] + "\n\n" + tool_prompt
    else:
        messages.insert(0, {"role": "system", "content": tool_prompt})

    agent_result = MCPAgentResult()
    final_content = ""
    final_finish_reason = "stop"
    total_cost = 0.0

    for turn in range(max_turns):
        agent_result.turns = turn + 1

        # Call LLM — no tools=, use json_object response format
        result = await _inner_acall_llm(
            model,
            messages,
            timeout=timeout,
            response_format={"type": "json_object"},
            **kwargs,
        )

        agent_result.models_used.add(result.model)
        if result.warnings:
            agent_result.warnings.extend(result.warnings)

        # Accumulate usage
        inp, out, cached, cache_create = _extract_usage(result.usage or {})
        agent_result.total_input_tokens += inp
        agent_result.total_output_tokens += out
        agent_result.total_cached_tokens += cached
        agent_result.total_cache_creation_tokens += cache_create
        total_cost += result.cost

        raw_content = (result.content or "").strip()

        # Parse the JSON response — try full content first, then extract first object
        try:
            parsed = _json.loads(raw_content)
        except _json.JSONDecodeError:
            # Model may emit valid JSON followed by trailing garbage.
            # Try to extract the first complete JSON object.
            parsed = _extract_first_json_object(raw_content)
            if parsed is not None:
                msg = f"SHIM_JSON_RECOVERED: turn {turn + 1} had trailing garbage, extracted first JSON object"
                logger.info(msg)
                agent_result.warnings.append(msg)
        if parsed is None:
            msg = f"SHIM_JSON_ERROR: turn {turn + 1} unparseable (content: {raw_content[:200]})"
            logger.warning(msg)
            agent_result.warnings.append(msg)
            messages.append({"role": "assistant", "content": raw_content})
            messages.append({
                "role": "user",
                "content": (
                    "Your response was not valid JSON. "
                    "Please respond with valid JSON matching the required format."
                ),
            })
            agent_result.conversation_trace.append(
                {"role": "assistant", "content": raw_content}
            )
            continue

        action = parsed.get("action", "")

        if action == "final_answer":
            final_content = parsed.get("content", "")
            final_finish_reason = "stop"
            agent_result.conversation_trace.append(
                {"role": "assistant", "content": raw_content}
            )
            break

        if action == "tool_call":
            tool_name = parsed.get("tool_name", "")
            arguments = parsed.get("arguments", {})

            # Build a synthetic OpenAI-format tool_call for execute_direct_tool_calls
            synthetic_tc = {
                "id": f"shim_{turn}",
                "function": {
                    "name": tool_name,
                    "arguments": arguments,  # already a dict
                },
            }

            agent_result.conversation_trace.append({
                "role": "assistant",
                "content": raw_content,
                "tool_calls": [{"name": tool_name, "arguments": arguments}],
            })

            records, _ = await execute_direct_tool_calls(
                [synthetic_tc], tool_map, tool_result_max_length,
            )
            agent_result.tool_calls.extend(records)

            # Get the result text from the record
            record = records[0]
            if record.error:
                tool_result_text = f"ERROR: {record.error}"
            else:
                tool_result_text = record.result or ""

            # Inject as conversation messages (not role=tool — model doesn't understand that)
            messages.append({"role": "assistant", "content": raw_content})
            messages.append({
                "role": "user",
                "content": f"Tool '{tool_name}' returned:\n{tool_result_text}",
            })

            logger.debug(
                "Tool shim turn %d/%d: called %s",
                turn + 1, max_turns, tool_name,
            )
            continue

        # Unknown action — nudge the model
        msg = f"SHIM_UNKNOWN_ACTION: turn {turn + 1} action={action!r}"
        logger.warning(msg)
        agent_result.warnings.append(msg)
        messages.append({"role": "assistant", "content": raw_content})
        messages.append({
            "role": "user",
            "content": (
                f"Unknown action {action!r}. "
                'Use {{"action": "tool_call", ...}} or {{"action": "final_answer", ...}}.'
            ),
        })
        agent_result.conversation_trace.append(
            {"role": "assistant", "content": raw_content}
        )
    else:
        # max_turns exhausted — one final call without tool prompt to get an answer
        logger.warning(
            "Tool shim exhausted max_turns=%d, forcing final answer", max_turns,
        )
        messages.append({
            "role": "user",
            "content": "You have used all available turns. Give your final answer now as plain text.",
        })
        final_result = await _inner_acall_llm(
            model, messages, timeout=timeout, **kwargs,
        )
        agent_result.models_used.add(final_result.model)
        if final_result.warnings:
            agent_result.warnings.extend(final_result.warnings)
        final_content = final_result.content
        final_finish_reason = final_result.finish_reason
        total_cost += final_result.cost
        inp, out, cached, cache_create = _extract_usage(final_result.usage or {})
        agent_result.total_input_tokens += inp
        agent_result.total_output_tokens += out
        agent_result.total_cached_tokens += cached
        agent_result.total_cache_creation_tokens += cache_create
        agent_result.turns += 1

    agent_result.metadata["total_cost"] = total_cost

    usage = {
        "input_tokens": agent_result.total_input_tokens,
        "output_tokens": agent_result.total_output_tokens,
        "total_tokens": (
            agent_result.total_input_tokens + agent_result.total_output_tokens
        ),
    }
    if agent_result.total_cached_tokens:
        usage["cached_tokens"] = agent_result.total_cached_tokens
    if agent_result.total_cache_creation_tokens:
        usage["cache_creation_tokens"] = agent_result.total_cache_creation_tokens

    return LLMCallResult(
        content=final_content,
        usage=usage,
        cost=total_cost,
        model=model,
        finish_reason=final_finish_reason,
        raw_response=agent_result,
        warnings=agent_result.warnings,
    )
