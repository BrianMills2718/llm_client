"""Direct Python tool-calling for llm_client.

Generate OpenAI-compatible tool schemas from plain Python functions and execute
tool calls in-process — no MCP subprocess, no stdio, no JSON-RPC overhead.

Usage:
    from llm_client.tool_utils import callable_to_openai_tool, prepare_direct_tools

    async def search(query: str, limit: int = 10) -> str:
        '''Search for entities.'''
        ...

    tool_map, openai_tools = prepare_direct_tools([search])
    # openai_tools is ready for litellm tools= parameter
    # tool_map is name->callable for execute_direct_tool_calls
"""

from __future__ import annotations

import asyncio
import inspect
import json as _json
import logging
import time
from typing import Any, Callable, Optional, Union, get_args, get_origin, get_type_hints

from llm_client.mcp_agent import MCPToolCallRecord, _truncate

logger = logging.getLogger(__name__)

# Python type → JSON Schema type
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _type_to_json_schema(tp: type) -> dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema fragment.

    Supports: str, int, float, bool, list[X], dict, Optional[X].
    Raises ValueError for unsupported types.
    """
    origin = get_origin(tp)
    args = get_args(tp)

    # Optional[X] → unwrap to X (nullable not needed for OpenAI function calling)
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _type_to_json_schema(non_none[0])

    # list[X]
    if origin is list:
        schema: dict[str, Any] = {"type": "array"}
        if args:
            schema["items"] = _type_to_json_schema(args[0])
        return schema

    # dict (any dict)
    if origin is dict or tp is dict:
        return {"type": "object"}

    # Basic types
    if tp in _TYPE_MAP:
        return {"type": _TYPE_MAP[tp]}

    raise ValueError(
        f"Unsupported type annotation: {tp!r}. "
        f"Supported: str, int, float, bool, list[X], dict, Optional[X]."
    )


def callable_to_openai_tool(fn: Callable[..., Any]) -> dict[str, Any]:
    """Convert a Python callable to an OpenAI function-calling tool schema.

    Inspects the function's name, type hints, and docstring.
    Every parameter must have a type annotation (raises ValueError otherwise).

    Args:
        fn: An async or sync function with typed parameters.

    Returns:
        OpenAI tool schema dict: {"type": "function", "function": {...}}
    """
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        if name not in hints:
            raise ValueError(
                f"Parameter {name!r} of {fn.__name__!r} has no type annotation. "
                f"All parameters must be typed for schema generation."
            )

        tp = hints[name]
        prop = _type_to_json_schema(tp)

        # Default value
        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(name)

        properties[name] = prop

    # Description from docstring first line
    description = ""
    if fn.__doc__:
        first_line = fn.__doc__.strip().split("\n")[0].strip()
        if first_line:
            description = first_line

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        parameters["required"] = required

    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": description,
            "parameters": parameters,
        },
    }


def prepare_direct_tools(
    tools: list[Callable[..., Any]],
) -> tuple[dict[str, Callable[..., Any]], list[dict[str, Any]]]:
    """Prepare a list of Python callables for use as direct tools.

    Args:
        tools: List of typed Python functions (sync or async).

    Returns:
        (tool_map, openai_tools):
        - tool_map: dict mapping function name → callable
        - openai_tools: list of OpenAI tool schema dicts

    Raises:
        ValueError: If two functions have the same __name__.
    """
    tool_map: dict[str, Callable[..., Any]] = {}
    openai_tools: list[dict[str, Any]] = []

    for fn in tools:
        schema = callable_to_openai_tool(fn)
        name = schema["function"]["name"]
        if name in tool_map:
            raise ValueError(
                f"Duplicate tool name {name!r}: "
                f"{tool_map[name]!r} and {fn!r} have the same __name__."
            )
        tool_map[name] = fn
        openai_tools.append(schema)

    return tool_map, openai_tools


async def execute_direct_tool_calls(
    tool_calls: list[dict[str, Any]],
    tool_map: dict[str, Callable[..., Any]],
    max_result_length: int,
) -> tuple[list[MCPToolCallRecord], list[dict[str, Any]]]:
    """Execute tool calls by calling Python functions directly.

    Same return type as _execute_tool_calls in mcp_agent.py for drop-in use.

    Args:
        tool_calls: Tool calls from LLM response (OpenAI format).
        tool_map: name → callable mapping from prepare_direct_tools().
        max_result_length: Max chars per tool result (truncated if longer).

    Returns:
        (records, tool_messages) — records for tracking, messages to append.
    """
    records: list[MCPToolCallRecord] = []
    tool_messages: list[dict[str, Any]] = []

    for tc in tool_calls:
        fn_info = tc.get("function", {})
        tool_name = fn_info.get("name", "")
        arguments_str = fn_info.get("arguments", "{}")
        tc_id = tc.get("id", "")

        try:
            arguments = (
                _json.loads(arguments_str)
                if isinstance(arguments_str, str)
                else arguments_str
            )
        except _json.JSONDecodeError:
            arguments = {}

        record = MCPToolCallRecord(
            server="__direct__",
            tool=tool_name,
            arguments=arguments,
        )

        t0 = time.monotonic()
        fn = tool_map.get(tool_name)

        if fn is None:
            record.error = f"Unknown tool: {tool_name}"
            tool_content = _json.dumps({"error": record.error})
        else:
            try:
                if asyncio.iscoroutinefunction(fn):
                    raw_result = await fn(**arguments)
                else:
                    raw_result = fn(**arguments)

                # Serialize: str passed through, anything else json.dumps
                if isinstance(raw_result, str):
                    tool_content = raw_result
                else:
                    tool_content = _json.dumps(raw_result)

                tool_content = _truncate(tool_content, max_result_length)
                record.result = tool_content
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                record.error = error_msg
                tool_content = _json.dumps({"error": error_msg})

        record.latency_s = round(time.monotonic() - t0, 3)
        records.append(record)

        tool_messages.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "content": tool_content,
        })

    return records, tool_messages
