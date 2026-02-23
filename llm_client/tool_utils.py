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

from llm_client.mcp_agent import (
    MCPToolCallRecord,
    TOOL_REASONING_FIELD,
    _append_input_examples_to_description,
    _normalize_tool_input_examples,
    _truncate,
)

logger = logging.getLogger(__name__)

# Python type → JSON Schema type
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _candidate_arg_aliases(key: str) -> list[str]:
    """Return structural singular/plural alias candidates for an argument key."""
    candidates: list[str] = []
    if key.endswith("_id"):
        candidates.append(f"{key}s")
    if key.endswith("_ids"):
        candidates.append(key[:-1])
    if key.endswith("s"):
        candidates.append(key[:-1])
    else:
        candidates.append(f"{key}s")
    return [c for c in candidates if c and c != key]


def _coerce_alias_value(source_key: str, target_key: str, value: Any) -> tuple[Any, str]:
    """Coerce value shape when mapping between singular/plural parameter names."""
    singular_to_plural = (
        (source_key.endswith("_id") and target_key.endswith("_ids"))
        or (not source_key.endswith("s") and target_key.endswith("s"))
    )
    plural_to_singular = (
        (source_key.endswith("_ids") and target_key.endswith("_id"))
        or (source_key.endswith("s") and not target_key.endswith("s"))
    )

    if singular_to_plural:
        if isinstance(value, list):
            return value, "singular_to_plural_passthrough"
        return [value], "singular_to_plural_wrap"

    if plural_to_singular:
        if isinstance(value, list):
            if len(value) == 1:
                return value[0], "plural_to_singular_unwrap"
            raise ValueError(
                f"cannot coerce {source_key!r}->{target_key!r}: expected single-item list, got {len(value)} items"
            )
        return value, "plural_to_singular_passthrough"

    return value, "rename_passthrough"


def _normalize_direct_tool_arguments(
    fn: Callable[..., Any],
    arguments: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], list[str], list[str], list[str]]:
    """Normalize tool args against callable signature.

    Returns:
      normalized_args, coercions, unknown_args, missing_required, accepted_params, coercion_errors
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        # Fall back to raw args when signature introspection is unavailable.
        return dict(arguments), [], {}, [], sorted(arguments.keys()), []
    accepted: set[str] = set()
    required: set[str] = set()
    accepts_var_kwargs = False

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            accepts_var_kwargs = True
            continue
        if param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            accepted.add(name)
            if param.default is inspect.Parameter.empty:
                required.add(name)

    normalized: dict[str, Any] = {}
    unknown: dict[str, Any] = {}
    coercions: list[dict[str, Any]] = []
    coercion_errors: list[str] = []

    for key, value in arguments.items():
        if key in accepted:
            normalized[key] = value
            continue
        if accepts_var_kwargs:
            normalized[key] = value
            continue

        alias_target = next(
            (cand for cand in _candidate_arg_aliases(key) if cand in accepted),
            None,
        )
        if (
            alias_target
            and alias_target not in arguments
            and alias_target not in normalized
        ):
            try:
                coerced_value, rule = _coerce_alias_value(key, alias_target, value)
            except ValueError as exc:
                coercion_errors.append(str(exc))
                unknown[key] = value
                continue
            normalized[alias_target] = coerced_value
            coercions.append(
                {
                    "source_arg": key,
                    "target_arg": alias_target,
                    "rule": rule,
                }
            )
            continue

        unknown[key] = value

    missing_required = sorted(name for name in required if name not in normalized)
    return (
        normalized,
        coercions,
        unknown,
        missing_required,
        sorted(accepted),
        coercion_errors,
    )


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

    # Description from explicit override attribute or docstring first line
    description = ""
    override_desc = getattr(fn, "__tool_description__", None)
    if isinstance(override_desc, str) and override_desc.strip():
        description = override_desc.strip()
    elif fn.__doc__:
        first_line = fn.__doc__.strip().split("\n")[0].strip()
        if first_line:
            description = first_line
    raw_input_examples = getattr(fn, "__tool_input_examples__", None)
    if raw_input_examples is None:
        raw_input_examples = getattr(fn, "__tool_examples__", None)
    description = _append_input_examples_to_description(
        description,
        _normalize_tool_input_examples(raw_input_examples),
    )

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    # Cross-tool observability: caller can provide a short reason for this call.
    if TOOL_REASONING_FIELD not in parameters["properties"]:
        parameters["properties"][TOOL_REASONING_FIELD] = {
            "type": "string",
            "description": "Why this specific tool call is needed right now.",
        }
    if TOOL_REASONING_FIELD not in required:
        required.append(TOOL_REASONING_FIELD)
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
    require_tool_reasoning: bool = False,
) -> tuple[list[MCPToolCallRecord], list[dict[str, Any]]]:
    """Execute tool calls by calling Python functions directly.

    Same return type as _execute_tool_calls in mcp_agent.py for drop-in use.

    Args:
        tool_calls: Tool calls from LLM response (OpenAI format).
        tool_map: name → callable mapping from prepare_direct_tools().
        max_result_length: Max chars per tool result (truncated if longer).
        require_tool_reasoning: If True, reject tool calls missing tool_reasoning.

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
        except _json.JSONDecodeError as exc:
            logger.error("Failed to parse tool call arguments for %s: %s", tool_name, str(arguments_str)[:200])
            record = MCPToolCallRecord(
                server="__direct__",
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

        if not isinstance(arguments, dict):
            arguments = {}

        tool_reasoning_raw = arguments.pop(TOOL_REASONING_FIELD, None)
        tool_reasoning = None
        if isinstance(tool_reasoning_raw, str):
            stripped = tool_reasoning_raw.strip()
            if stripped:
                tool_reasoning = stripped

        record = MCPToolCallRecord(
            server="__direct__",
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
        fn = tool_map.get(tool_name)

        if fn is None:
            record.error = f"Unknown tool: {tool_name}"
            tool_content = _json.dumps({"error": record.error})
        else:
            (
                normalized_args,
                coercions,
                unknown_args,
                missing_required,
                accepted_params,
                coercion_errors,
            ) = _normalize_direct_tool_arguments(fn, arguments)
            record.arguments = normalized_args
            record.arg_coercions = coercions
            if coercions:
                for event in coercions:
                    logger.warning(
                        "TOOL_ARG_COERCION tool=%s source=%s target=%s rule=%s",
                        tool_name,
                        event.get("source_arg"),
                        event.get("target_arg"),
                        event.get("rule"),
                    )

            if unknown_args or missing_required or coercion_errors:
                parts: list[str] = []
                if unknown_args:
                    parts.append(
                        "unsupported args: "
                        + ", ".join(sorted(unknown_args.keys()))
                    )
                if missing_required:
                    parts.append(
                        "missing required args: "
                        + ", ".join(missing_required)
                    )
                if coercion_errors:
                    parts.append("coercion errors: " + " | ".join(coercion_errors))
                parts.append("allowed args: " + ", ".join(accepted_params))
                validation_msg = "; ".join(parts)
                record.error = f"Validation error: {validation_msg}"
                tool_content = _json.dumps(
                    {
                        "error": record.error,
                        "arg_coercions": coercions,
                    }
                )
                record.latency_s = round(time.monotonic() - t0, 3)
                records.append(record)
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": tool_content,
                    }
                )
                continue

            try:
                if asyncio.iscoroutinefunction(fn):
                    raw_result = await fn(**normalized_args)
                else:
                    raw_result = fn(**normalized_args)

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
