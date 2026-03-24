"""Direct Python tool-calling for llm_client.

Generate OpenAI-compatible tool schemas from plain Python functions and execute
tool calls in-process — no MCP subprocess, no stdio, no JSON-RPC overhead.

Usage:
    from llm_client.tools.tool_utils import callable_to_openai_tool, prepare_direct_tools

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

from llm_client.tools.tool_runtime_common import (
    MCPToolCallRecord,
    TOOL_REASONING_FIELD,
    append_input_examples_to_description as _append_input_examples_to_description,
    normalize_tool_contracts,
    normalize_tool_input_examples as _normalize_tool_input_examples,
    truncate_text as _truncate,
)

logger = logging.getLogger(__name__)

# Python type → JSON Schema type
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _tool_description_line(fn: Callable[..., Any]) -> str:
    override_desc = getattr(fn, "__tool_description__", None)
    if isinstance(override_desc, str) and override_desc.strip():
        return override_desc.strip()
    if fn.__doc__:
        first_line = fn.__doc__.strip().split("\n")[0].strip()
        if first_line:
            return first_line
    return ""


def _tool_user_parameters(fn: Callable[..., Any]) -> list[inspect.Parameter]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return []
    params: list[inspect.Parameter] = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls", TOOL_REASONING_FIELD):
            continue
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            params.append(param)
    return params


def _is_complex_annotation(tp: Any) -> bool:
    origin = get_origin(tp)
    args = get_args(tp)
    if origin in {list, dict, set, tuple}:
        return True
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        return len(non_none) > 1 or any(_is_complex_annotation(a) for a in non_none)
    return False


def _tool_is_nontrivial(fn: Callable[..., Any]) -> tuple[bool, list[str]]:
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}
    params = _tool_user_parameters(fn)
    reasons: list[str] = []
    if len(params) > 1:
        reasons.append("multiple_parameters")
    for param in params:
        if param.default is not inspect.Parameter.empty:
            reasons.append(f"optional_param:{param.name}")
        hint = hints.get(param.name)
        if hint is not None and _is_complex_annotation(hint):
            reasons.append(f"complex_type:{param.name}")
    return (bool(reasons), reasons)


def lint_tool_callable(
    fn: Callable[..., Any],
    *,
    contract: dict[str, Any] | None = None,
    require_examples_for_nontrivial: bool = True,
    require_contract_for_nontrivial: bool = True,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    tool_name = getattr(fn, "__name__", "<callable>")

    try:
        callable_to_openai_tool(fn)
    except Exception as exc:
        findings.append(
            {
                "tool_name": tool_name,
                "severity": "error",
                "code": "schema_generation_failed",
                "message": str(exc),
            }
        )
        return findings

    description = _tool_description_line(fn)
    if not description:
        findings.append(
            {
                "tool_name": tool_name,
                "severity": "warning",
                "code": "missing_description",
                "message": "tool is missing a one-line description or docstring summary",
            }
        )

    normalized_examples = _normalize_tool_input_examples(
        getattr(fn, "__tool_input_examples__", getattr(fn, "__tool_examples__", None))
    )
    is_nontrivial, nontrivial_reasons = _tool_is_nontrivial(fn)
    if require_examples_for_nontrivial and is_nontrivial and not normalized_examples:
        findings.append(
            {
                "tool_name": tool_name,
                "severity": "warning",
                "code": "missing_examples",
                "message": "nontrivial tool is missing input examples",
                "details": {"nontrivial_reasons": nontrivial_reasons},
            }
        )

    if require_contract_for_nontrivial and is_nontrivial and contract is None:
        findings.append(
            {
                "tool_name": tool_name,
                "severity": "warning",
                "code": "missing_contract",
                "message": "nontrivial tool is missing declarative tool_contracts metadata",
                "details": {"nontrivial_reasons": nontrivial_reasons},
            }
        )

    if contract is not None:
        try:
            normalized = normalize_tool_contracts({tool_name: contract})
            normalized_contract = normalized.get(tool_name) or {}
            accepted_param_names = {param.name for param in _tool_user_parameters(fn)}
            call_modes = normalized_contract.get("call_modes") or []
            if call_modes:
                for index, mode in enumerate(call_modes):
                    has_selector = bool(
                        mode.get("when_args_present_any")
                        or mode.get("when_args_present_all")
                        or mode.get("when_arg_equals")
                    )
                    if not has_selector:
                        findings.append(
                            {
                                "tool_name": tool_name,
                                "severity": "error",
                                "code": "call_mode_missing_selector",
                                "message": f"call_modes[{index}] must declare a matching selector",
                            }
                        )
                    for handle_index, handle_spec in enumerate(mode.get("handle_inputs") or []):
                        arg_name = str(handle_spec.get("arg") or "").strip()
                        inject_arg = str(handle_spec.get("inject_arg") or "").strip()
                        if arg_name and arg_name not in accepted_param_names:
                            findings.append(
                                {
                                    "tool_name": tool_name,
                                    "severity": "error",
                                    "code": "handle_input_missing_arg",
                                    "message": (
                                        f"call_modes[{index}].handle_inputs[{handle_index}] "
                                        f"arg={arg_name!r} is not accepted by the callable"
                                    ),
                                }
                            )
                        if inject_arg and inject_arg not in accepted_param_names:
                            findings.append(
                                {
                                    "tool_name": tool_name,
                                    "severity": "error",
                                    "code": "handle_input_missing_inject_arg",
                                    "message": (
                                        f"call_modes[{index}].handle_inputs[{handle_index}] "
                                        f"inject_arg={inject_arg!r} is not accepted by the callable"
                                    ),
                                }
                            )
            for handle_index, handle_spec in enumerate(normalized_contract.get("handle_inputs") or []):
                arg_name = str(handle_spec.get("arg") or "").strip()
                inject_arg = str(handle_spec.get("inject_arg") or "").strip()
                if arg_name and arg_name not in accepted_param_names:
                    findings.append(
                        {
                            "tool_name": tool_name,
                            "severity": "error",
                            "code": "handle_input_missing_arg",
                            "message": (
                                f"handle_inputs[{handle_index}] arg={arg_name!r} "
                                "is not accepted by the callable"
                            ),
                        }
                    )
                if inject_arg and inject_arg not in accepted_param_names:
                    findings.append(
                        {
                            "tool_name": tool_name,
                            "severity": "error",
                            "code": "handle_input_missing_inject_arg",
                            "message": (
                                f"handle_inputs[{handle_index}] inject_arg={inject_arg!r} "
                                "is not accepted by the callable"
                            ),
                        }
                    )
        except Exception as exc:
            findings.append(
                {
                    "tool_name": tool_name,
                    "severity": "error",
                    "code": "invalid_contract",
                    "message": str(exc),
                }
            )

    return findings


def lint_tool_registry(
    tools: list[Callable[..., Any]],
    *,
    tool_contracts: dict[str, Any] | None = None,
    require_examples_for_nontrivial: bool = True,
    require_contract_for_nontrivial: bool = True,
) -> dict[str, Any]:
    per_tool: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    for fn in tools:
        tool_name = getattr(fn, "__name__", "<callable>")
        tool_findings = lint_tool_callable(
            fn,
            contract=(tool_contracts or {}).get(tool_name),
            require_examples_for_nontrivial=require_examples_for_nontrivial,
            require_contract_for_nontrivial=require_contract_for_nontrivial,
        )
        per_tool.append({"tool_name": tool_name, "findings": tool_findings})
        findings.extend(tool_findings)
    n_errors = sum(1 for finding in findings if finding.get("severity") == "error")
    n_warnings = sum(1 for finding in findings if finding.get("severity") == "warning")
    return {
        "n_tools": len(tools),
        "n_findings": len(findings),
        "n_errors": n_errors,
        "n_warnings": n_warnings,
        "passed": n_errors == 0,
        "tools": per_tool,
        "findings": findings,
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

    # Union types: Optional[X] unwraps to X; multi-type unions use anyOf.
    # Handles both typing.Union and Python 3.10+ X | Y syntax (types.UnionType).
    import types as _types
    _is_union = origin is Union or isinstance(tp, _types.UnionType)
    if _is_union:
        _union_args = args or get_args(tp)
        non_none = [a for a in _union_args if a is not type(None)]
        if len(non_none) == 1:
            return _type_to_json_schema(non_none[0])
        if len(non_none) > 1:
            return {"anyOf": [_type_to_json_schema(a) for a in non_none]}

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

    # Any → no type constraint (accept anything)
    if tp is Any:
        return {}

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
                tool_call_id=tc_id,
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
            tool_call_id=tc_id,
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
