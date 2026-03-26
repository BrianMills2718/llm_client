"""Tests for llm_client.tools.tool_registry — PTC tool callable registry.

Covers registration, lookup, listing, clearing, bulk registration from lists,
and the execute_tool_chain code-execution surface (including chaining,
stdout isolation, timeouts, error handling, and explicit tool overrides).
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

import pytest

from llm_client.tools.tool_registry import (
    clear_registry,
    execute_tool_chain,
    get_tool_callable,
    list_tool_callables,
    register_tool_callable,
    register_tools_from_list,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry() -> Any:
    """Ensure every test starts and ends with a clean global registry."""
    clear_registry()
    yield
    clear_registry()


# ---------------------------------------------------------------------------
# Helpers — sample tools
# ---------------------------------------------------------------------------


async def _async_echo(text: str) -> str:
    """Echo the input text."""
    return f"echo:{text}"


def _sync_add(a: int, b: int) -> str:
    """Add two numbers."""
    return str(a + b)


async def _async_upper(text: str) -> str:
    """Uppercase text."""
    return text.upper()


async def _async_wrap(text: str, prefix: str = "[", suffix: str = "]") -> str:
    """Wrap text with prefix and suffix."""
    return f"{prefix}{text}{suffix}"


# ---------------------------------------------------------------------------
# Test: register_tool_callable / get_tool_callable
# ---------------------------------------------------------------------------


class TestRegisterAndGet:
    """Registration and retrieval of individual tool callables."""

    def test_register_and_get(self) -> None:
        """Register a tool, then retrieve it by name."""
        register_tool_callable("echo", _async_echo, description="Echo tool")
        fn = get_tool_callable("echo")
        result: str = asyncio.run(fn(text="hello"))  # type: ignore[arg-type]
        assert result == "echo:hello"

    def test_get_missing_raises(self) -> None:
        """Getting an unregistered name raises KeyError with helpful message."""
        with pytest.raises(KeyError, match="not registered"):
            get_tool_callable("nonexistent")

    def test_duplicate_name_raises(self) -> None:
        """Registering the same name twice raises ValueError."""
        register_tool_callable("echo", _async_echo)
        with pytest.raises(ValueError, match="already registered"):
            register_tool_callable("echo", _async_upper)

    def test_sync_callable_wrapped(self) -> None:
        """A sync callable is automatically wrapped as async."""
        register_tool_callable("add", _sync_add)
        fn = get_tool_callable("add")
        result: str = asyncio.run(fn(a=3, b=4))  # type: ignore[arg-type]
        assert result == "7"


# ---------------------------------------------------------------------------
# Test: list_tool_callables
# ---------------------------------------------------------------------------


class TestListCallables:
    """Listing registered tool callables."""

    def test_list_callables(self) -> None:
        """Register 3 tools, list returns all with descriptions."""
        register_tool_callable("echo", _async_echo, description="Echo tool")
        register_tool_callable("add", _sync_add, description="Add numbers")
        register_tool_callable("upper", _async_upper, description="Uppercase")

        entries = list_tool_callables()
        assert len(entries) == 3
        names = {e["name"] for e in entries}
        assert names == {"echo", "add", "upper"}

        # Descriptions preserved
        by_name = {e["name"]: e["description"] for e in entries}
        assert by_name["echo"] == "Echo tool"
        assert by_name["add"] == "Add numbers"

    def test_list_empty(self) -> None:
        """Listing when nothing is registered returns empty list."""
        assert list_tool_callables() == []


# ---------------------------------------------------------------------------
# Test: clear_registry
# ---------------------------------------------------------------------------


class TestClearRegistry:
    """Clearing the global registry."""

    def test_clear_registry(self) -> None:
        """Register tools, clear, list is empty."""
        register_tool_callable("echo", _async_echo)
        register_tool_callable("add", _sync_add)
        assert len(list_tool_callables()) == 2

        clear_registry()
        assert list_tool_callables() == []

    def test_clear_allows_re_registration(self) -> None:
        """After clearing, the same name can be registered again."""
        register_tool_callable("echo", _async_echo)
        clear_registry()
        register_tool_callable("echo", _async_upper)  # no ValueError
        fn = get_tool_callable("echo")
        result: str = asyncio.run(fn(text="hi"))  # type: ignore[arg-type]
        assert result == "HI"


# ---------------------------------------------------------------------------
# Test: execute_tool_chain — simple cases
# ---------------------------------------------------------------------------


class TestExecuteChainSimple:
    """Basic execute_tool_chain scenarios."""

    def test_execute_chain_simple(self) -> None:
        """Register a tool, execute code that calls it and prints result."""
        register_tool_callable("echo", _async_echo)

        code = """\
result = await echo(text="world")
print(result)
"""
        output = asyncio.run(execute_tool_chain(code))
        assert output == "echo:world"

    def test_execute_chain_multi_tool(self) -> None:
        """Register 3 tools, execute code that chains them (A -> B -> C)."""

        async def tool_a(x: str) -> str:
            """Step A: prefix."""
            return f"A({x})"

        async def tool_b(x: str) -> str:
            """Step B: prefix."""
            return f"B({x})"

        async def tool_c(x: str) -> str:
            """Step C: prefix."""
            return f"C({x})"

        register_tool_callable("tool_a", tool_a)
        register_tool_callable("tool_b", tool_b)
        register_tool_callable("tool_c", tool_c)

        code = """\
r1 = await tool_a(x="start")
r2 = await tool_b(x=r1)
r3 = await tool_c(x=r2)
print(r3)
"""
        output = asyncio.run(execute_tool_chain(code))
        assert output == "C(B(A(start)))"

    def test_intermediates_not_in_output(self) -> None:
        """Only the final print appears in output — intermediates stay local."""

        async def step1() -> str:
            """First step."""
            return "intermediate_secret_value"

        async def step2(x: str) -> str:
            """Second step."""
            return f"final:{len(x)}"

        register_tool_callable("step1", step1)
        register_tool_callable("step2", step2)

        code = """\
mid = await step1()
result = await step2(x=mid)
print(result)
"""
        output = asyncio.run(execute_tool_chain(code))
        assert output == "final:25"
        # The intermediate value must not appear in captured output
        assert "intermediate_secret_value" not in output


# ---------------------------------------------------------------------------
# Test: execute_tool_chain — error and edge cases
# ---------------------------------------------------------------------------


class TestExecuteChainErrors:
    """Error handling and edge cases in execute_tool_chain."""

    def test_execute_chain_timeout(self) -> None:
        """Code that sleeps forever triggers a TimeoutError."""

        async def slow_tool() -> str:
            """Never finishes."""
            await asyncio.sleep(9999)
            return "never"

        register_tool_callable("slow_tool", slow_tool)

        code = """\
result = await slow_tool()
print(result)
"""
        with pytest.raises(TimeoutError, match="timeout"):
            asyncio.run(execute_tool_chain(code, timeout=0.1))

    def test_execute_chain_error_handling(self) -> None:
        """Code that raises is surfaced as RuntimeError, not swallowed."""

        async def bad_tool() -> str:
            """Always fails."""
            raise ValueError("tool broke")

        register_tool_callable("bad_tool", bad_tool)

        code = """\
result = await bad_tool()
print(result)
"""
        with pytest.raises(RuntimeError, match="tool broke"):
            asyncio.run(execute_tool_chain(code))

    def test_execute_chain_syntax_error(self) -> None:
        """Malformed code raises RuntimeError with syntax info."""
        code = "this is not valid python @@@@"
        with pytest.raises(RuntimeError, match="Syntax error"):
            asyncio.run(execute_tool_chain(code))

    def test_execute_chain_no_print_returns_empty(self) -> None:
        """Code that produces no print output returns empty string."""

        async def noop() -> str:
            """Do nothing interesting."""
            return "ignored"

        register_tool_callable("noop", noop)

        code = """\
x = await noop()
"""
        output = asyncio.run(execute_tool_chain(code))
        assert output == ""


# ---------------------------------------------------------------------------
# Test: register_tools_from_list
# ---------------------------------------------------------------------------


class TestRegisterFromList:
    """Bulk registration from a list of callables."""

    def test_register_from_list(self) -> None:
        """Register tools from a list of functions, verify all registered."""
        tools: list[Callable[..., Any]] = [_async_echo, _sync_add, _async_upper]
        count = register_tools_from_list(tools)
        assert count == 3

        entries = list_tool_callables()
        names = {e["name"] for e in entries}
        assert names == {"_async_echo", "_sync_add", "_async_upper"}

    def test_register_from_list_skips_duplicates(self) -> None:
        """Duplicate names in the list are skipped with a warning."""
        register_tool_callable("_async_echo", _async_echo)
        count = register_tools_from_list([_async_echo, _sync_add])
        # _async_echo already registered → skipped, _sync_add succeeds
        assert count == 1

    def test_register_from_list_uses_docstring(self) -> None:
        """Descriptions come from __doc__ first line."""
        register_tools_from_list([_async_echo])
        entries = list_tool_callables()
        assert len(entries) == 1
        assert entries[0]["description"] == "Echo the input text."


# ---------------------------------------------------------------------------
# Test: execute_tool_chain with explicit tools dict
# ---------------------------------------------------------------------------


class TestExecuteWithExplicitTools:
    """Pass tools dict to execute_tool_chain instead of using registry."""

    def test_execute_with_explicit_tools(self) -> None:
        """Explicit tools dict overrides the global registry."""

        async def custom_greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}!"

        # Registry is empty — explicit tools should still work
        code = """\
msg = await custom_greet(name="Alice")
print(msg)
"""
        output = asyncio.run(
            execute_tool_chain(code, tools={"custom_greet": custom_greet})
        )
        assert output == "Hello, Alice!"

    def test_explicit_tools_ignores_registry(self) -> None:
        """When tools dict is provided, registry tools are not available."""

        async def registry_tool() -> str:
            """In registry."""
            return "from_registry"

        async def explicit_tool() -> str:
            """Explicit only."""
            return "from_explicit"

        register_tool_callable("registry_tool", registry_tool)

        code = """\
result = await explicit_tool()
print(result)
"""
        output = asyncio.run(
            execute_tool_chain(code, tools={"explicit_tool": explicit_tool})
        )
        assert output == "from_explicit"

    def test_explicit_sync_tool_wrapped(self) -> None:
        """Sync tools in explicit dict are auto-wrapped as async."""

        def sync_double(n: int) -> str:
            """Double a number."""
            return str(n * 2)

        code = """\
result = await sync_double(n=21)
print(result)
"""
        output = asyncio.run(
            execute_tool_chain(code, tools={"sync_double": sync_double})
        )
        assert output == "42"


# ---------------------------------------------------------------------------
# Test: async tool callable
# ---------------------------------------------------------------------------


class TestAsyncToolCallable:
    """Verify async tool callables work through the full path."""

    def test_async_tool_callable(self) -> None:
        """Register an async tool, execute code that awaits it."""

        async def async_concat(a: str, b: str) -> str:
            """Concatenate two strings with a space."""
            # Simulate async I/O
            await asyncio.sleep(0.01)
            return f"{a} {b}"

        register_tool_callable("async_concat", async_concat)

        code = """\
result = await async_concat(a="hello", b="world")
print(result)
"""
        output = asyncio.run(execute_tool_chain(code))
        assert output == "hello world"
