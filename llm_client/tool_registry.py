"""Tool callable registry for programmatic tool calling (PTC).

Exposes registered tool functions as await-able callables that agent code
can import and call directly. Intermediates stay in code variables — only
the final output enters agent context.

This is the in-process version of PTC — for trusted tools that run in the
same process. For sandboxed execution, use the API-level code_execution tool.

Security note: ``execute_tool_chain`` uses ``exec()`` internally. It is NOT
sandboxed and must only be used with trusted code. Arbitrary user-provided
code strings should never be passed to ``execute_tool_chain`` without
independent validation.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import io
import logging
import sys
import textwrap
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

ToolCallable = Callable[..., Awaitable[str]]
"""An async callable that takes keyword args and returns a string result."""

# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

_TOOL_REGISTRY: dict[str, ToolCallable] = {}
_TOOL_DESCRIPTIONS: dict[str, str] = {}


def _wrap_sync_as_async(fn: Callable[..., Any]) -> ToolCallable:
    """Wrap a synchronous callable so it presents an async interface.

    The wrapper converts the return value to ``str`` for consistency with the
    tool callable contract.
    """

    @functools.wraps(fn)
    async def _async_wrapper(**kwargs: Any) -> str:
        result = fn(**kwargs)
        return str(result)

    return _async_wrapper


# ---------------------------------------------------------------------------
# Public API — registration
# ---------------------------------------------------------------------------


def register_tool_callable(
    name: str,
    fn: Callable[..., Any],
    *,
    description: str = "",
) -> None:
    """Register an async (or sync) callable as a tool available for PTC.

    Sync callables are automatically wrapped in an async shim whose return
    value is coerced to ``str``.

    Args:
        name: Tool name (must be unique within the registry).
        fn: Callable that takes kwargs and returns a string-coercible result.
        description: Human-readable description for discovery.

    Raises:
        ValueError: If *name* is already registered.
    """
    if name in _TOOL_REGISTRY:
        raise ValueError(
            f"Tool {name!r} is already registered. "
            "Call clear_registry() first or choose a different name."
        )

    if asyncio.iscoroutinefunction(fn):
        _TOOL_REGISTRY[name] = fn
    else:
        _TOOL_REGISTRY[name] = _wrap_sync_as_async(fn)

    _TOOL_DESCRIPTIONS[name] = description or (fn.__doc__ or "").strip().split("\n")[0]
    logger.debug("Registered tool callable: %s", name)


def get_tool_callable(name: str) -> ToolCallable:
    """Get a registered tool callable by name.

    Raises:
        KeyError: If *name* is not registered.
    """
    try:
        return _TOOL_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(_TOOL_REGISTRY)) or "(none)"
        raise KeyError(
            f"Tool {name!r} is not registered. Available: {available}"
        ) from None


def list_tool_callables() -> list[dict[str, str]]:
    """List all registered tool callables with their descriptions."""
    return [
        {"name": name, "description": _TOOL_DESCRIPTIONS.get(name, "")}
        for name in sorted(_TOOL_REGISTRY)
    ]


def clear_registry() -> None:
    """Clear all registered callables. Mainly for testing."""
    _TOOL_REGISTRY.clear()
    _TOOL_DESCRIPTIONS.clear()


def register_tools_from_list(tools: list[Callable[..., Any]]) -> int:
    """Register multiple tool callables from a list of functions.

    Uses ``__name__`` as the tool name and ``__doc__`` (first line) as the
    description. Skips tools whose name is already registered and logs a
    warning for each skip.

    Args:
        tools: List of sync or async callables to register.

    Returns:
        Count of successfully registered tools.
    """
    count = 0
    for fn in tools:
        name = getattr(fn, "__name__", None)
        if name is None:
            logger.warning("Skipping tool with no __name__: %r", fn)
            continue
        doc = (fn.__doc__ or "").strip().split("\n")[0]
        try:
            register_tool_callable(name, fn, description=doc)
            count += 1
        except ValueError:
            logger.warning("Skipping already-registered tool: %s", name)
    return count


# ---------------------------------------------------------------------------
# Public API — execution
# ---------------------------------------------------------------------------


async def execute_tool_chain(
    code: str,
    *,
    tools: dict[str, Callable[..., Any]] | None = None,
    timeout: float = 300.0,
) -> str:
    """Execute a Python code string with tool callables available as awaitable functions.

    The code runs in a restricted namespace where registered tools (or the
    provided *tools* dict) are available as async functions. Only the captured
    ``print()`` output is returned — intermediates stay in local variables.

    Args:
        code: Python code to execute. Should use ``await tool_name(...)`` for
              tool calls and ``print(result)`` for the final output.
        tools: Override the global registry with an explicit name-to-callable
              dict. If ``None``, uses the global registry.
        timeout: Maximum execution time in seconds (default 300).

    Returns:
        The captured stdout from the code execution (stripped of trailing
        whitespace).

    Raises:
        TimeoutError: If execution exceeds *timeout* seconds.
        RuntimeError: If the code raises an unhandled exception.

    **Security warning:** This function uses ``exec()`` and is NOT sandboxed.
    It runs in the current process with full access to the Python runtime.
    Only use with trusted code.
    """
    # Build the namespace with available tools
    namespace: dict[str, Any] = {"asyncio": asyncio}

    source_tools: dict[str, Callable[..., Any]]
    if tools is not None:
        source_tools = tools
    else:
        source_tools = dict(_TOOL_REGISTRY)

    # Ensure every tool in the namespace is async
    for name, fn in source_tools.items():
        if asyncio.iscoroutinefunction(fn):
            namespace[name] = fn
        else:
            namespace[name] = _wrap_sync_as_async(fn)

    # Capture stdout
    captured = io.StringIO()

    # Wrap user code in an async function so `await` works at top level
    indented_code = textwrap.indent(code, "    ")
    wrapper_code = f"async def __ptc_main__():\n{indented_code}\n"

    try:
        exec(compile(wrapper_code, "<tool_chain>", "exec"), namespace)  # noqa: S102
    except SyntaxError as exc:
        raise RuntimeError(f"Syntax error in tool chain code: {exc}") from exc

    main_fn = namespace["__ptc_main__"]

    async def _run_capturing() -> None:
        """Execute the user code with stdout redirected to our StringIO."""
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            await main_fn()
        finally:
            sys.stdout = old_stdout

    try:
        await asyncio.wait_for(_run_capturing(), timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError(
            f"Tool chain execution exceeded {timeout}s timeout"
        ) from None
    except TimeoutError:
        raise
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Tool chain execution failed: {type(exc).__name__}: {exc}"
        ) from exc

    return captured.getvalue().rstrip()
