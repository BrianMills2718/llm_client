"""Tests for llm_client.tools.decorator — @tool decorator, ToolResult, and ToolRegistry.

Covers:
- Decorator wraps async functions correctly
- Decorator wraps sync functions correctly
- ToolResult returned on success with correct fields
- ToolResult returned on error (no exception propagation)
- Registry auto-populates on decoration
- registry.list_all() returns registered tools
- registry.list_by_domain() filters correctly
- registry.has() checks existence
- ToolResult has correct latency_s (non-zero for real work)
- Already-ToolResult return values pass through
- trace_id and task kwargs flow through
- ToolInfo validates cost_tier
- ToolInfo captures input_type and output_type
- Registry rejects duplicate names
- Registry clear works
- _tool_info attached to wrapper
- ToolRegistry __len__ and __contains__
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from llm_client.tools.decorator import (
    ToolInfo,
    ToolRegistry,
    ToolResult,
    registry,
    tool,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry() -> Any:
    """Ensure every test starts and ends with a clean decorator registry."""
    registry.clear()
    yield
    registry.clear()


# ---------------------------------------------------------------------------
# Test: basic decoration and ToolResult on success
# ---------------------------------------------------------------------------


class TestToolDecoratorSuccess:
    """The @tool decorator wraps async functions and returns ToolResult."""

    def test_decorator_returns_tool_result(self) -> None:
        """Decorating an async function makes it return ToolResult."""

        @tool(name="echo")
        async def echo(text: str) -> str:
            return f"echo:{text}"

        result = asyncio.run(echo(text="hello"))
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.data == "echo:hello"
        assert result.error is None
        assert result.error_type is None
        assert result.tool_name == "echo"

    def test_latency_is_positive(self) -> None:
        """ToolResult.latency_s is a positive float for real work."""

        @tool(name="slow_op")
        async def slow_op() -> str:
            await asyncio.sleep(0.05)
            return "done"

        result = asyncio.run(slow_op())
        assert result.success is True
        assert result.latency_s >= 0.04  # generous lower bound

    def test_preserves_function_name_and_doc(self) -> None:
        """functools.wraps preserves __name__ and __doc__."""

        @tool(name="documented")
        async def my_func() -> str:
            """This is the docstring."""
            return "ok"

        assert my_func.__name__ == "my_func"
        assert my_func.__doc__ is not None
        assert "docstring" in my_func.__doc__

    def test_data_contains_complex_types(self) -> None:
        """ToolResult.data can hold dicts, lists, any type."""

        @tool(name="complex_return")
        async def get_data() -> dict[str, Any]:
            return {"items": [1, 2, 3], "count": 3}

        result = asyncio.run(get_data())
        assert result.success is True
        assert result.data == {"items": [1, 2, 3], "count": 3}


# ---------------------------------------------------------------------------
# Test: error handling
# ---------------------------------------------------------------------------


class TestToolDecoratorErrors:
    """Errors are caught and wrapped in ToolResult, never propagated."""

    def test_exception_wrapped_in_tool_result(self) -> None:
        """Exceptions become ToolResult(success=False) with error info."""

        @tool(name="fail_op")
        async def fail_op() -> str:
            raise ValueError("something broke")

        result = asyncio.run(fail_op())
        assert isinstance(result, ToolResult)
        assert result.success is False
        assert result.data is None
        assert result.error == "something broke"
        assert result.error_type == "ValueError"
        assert result.tool_name == "fail_op"

    def test_runtime_error_captured(self) -> None:
        """RuntimeError is also captured, not re-raised."""

        @tool(name="runtime_fail")
        async def runtime_fail() -> str:
            raise RuntimeError("critical failure")

        result = asyncio.run(runtime_fail())
        assert result.success is False
        assert result.error_type == "RuntimeError"

    def test_latency_recorded_on_failure(self) -> None:
        """Even on failure, latency_s is set."""

        @tool(name="slow_fail")
        async def slow_fail() -> str:
            await asyncio.sleep(0.03)
            raise IOError("network down")

        result = asyncio.run(slow_fail())
        assert result.success is False
        assert result.latency_s >= 0.02


# ---------------------------------------------------------------------------
# Test: sync function support
# ---------------------------------------------------------------------------


class TestSyncFunctionSupport:
    """The @tool decorator wraps sync functions and returns ToolResult."""

    def test_sync_function_returns_tool_result(self) -> None:
        """Decorating a sync function returns ToolResult on call."""

        @tool(name="sync_echo")
        def sync_echo(text: str) -> str:
            return f"sync:{text}"

        result = asyncio.run(sync_echo(text="hello"))
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.data == "sync:hello"
        assert result.tool_name == "sync_echo"

    def test_sync_function_error_handling(self) -> None:
        """Sync function exceptions are caught and wrapped in ToolResult."""

        @tool(name="sync_fail")
        def sync_fail() -> str:
            raise ValueError("sync error")

        result = asyncio.run(sync_fail())
        assert result.success is False
        assert result.error == "sync error"
        assert result.error_type == "ValueError"

    def test_sync_function_registered(self) -> None:
        """Sync functions are registered in the global registry."""

        @tool(name="sync_reg")
        def sync_reg() -> str:
            return "ok"

        assert "sync_reg" in registry
        info = registry.get("sync_reg")
        assert info is not None
        assert info.name == "sync_reg"

    def test_sync_function_latency_recorded(self) -> None:
        """Sync functions have latency_s recorded."""
        import time as _time

        @tool(name="sync_slow")
        def sync_slow() -> str:
            _time.sleep(0.03)
            return "done"

        result = asyncio.run(sync_slow())
        assert result.success is True
        assert result.latency_s >= 0.02


# ---------------------------------------------------------------------------
# Test: registry auto-population
# ---------------------------------------------------------------------------


class TestRegistryAutoPopulation:
    """The @tool decorator registers functions in the global registry."""

    def test_decoration_registers_tool(self) -> None:
        """After decoration, the tool appears in the registry."""
        assert len(registry) == 0

        @tool(name="auto_reg")
        async def auto_reg() -> str:
            return "ok"

        assert len(registry) == 1
        assert "auto_reg" in registry
        info = registry.get("auto_reg")
        assert info is not None
        assert info.name == "auto_reg"
        assert info.domain == "general"

    def test_duplicate_name_raises(self) -> None:
        """Registering two tools with the same name raises ValueError."""

        @tool(name="dup")
        async def first() -> str:
            return "1"

        with pytest.raises(ValueError, match="already registered"):

            @tool(name="dup")
            async def second() -> str:
                return "2"

    def test_tool_info_attached_to_wrapper(self) -> None:
        """The wrapper function has a _tool_info attribute."""

        @tool(name="with_info")
        async def with_info() -> str:
            return "ok"

        assert hasattr(with_info, "_tool_info")
        assert with_info._tool_info.name == "with_info"


# ---------------------------------------------------------------------------
# Test: registry queries
# ---------------------------------------------------------------------------


class TestRegistryQueries:
    """ToolRegistry.list_all() and list_by_domain() work correctly."""

    def test_list_all_returns_all_sorted(self) -> None:
        """list_all() returns all tools sorted by name."""

        @tool(name="zeta", domain="web")
        async def zeta() -> str:
            return "z"

        @tool(name="alpha", domain="gov")
        async def alpha() -> str:
            return "a"

        @tool(name="mid", domain="web")
        async def mid() -> str:
            return "m"

        tools = registry.list_all()
        assert len(tools) == 3
        assert [t.name for t in tools] == ["alpha", "mid", "zeta"]

    def test_list_by_domain_filters(self) -> None:
        """list_by_domain() returns only tools in the requested domain."""

        @tool(name="web1", domain="web")
        async def web1() -> str:
            return "w1"

        @tool(name="web2", domain="web")
        async def web2() -> str:
            return "w2"

        @tool(name="gov1", domain="government")
        async def gov1() -> str:
            return "g1"

        web_tools = registry.list_by_domain("web")
        assert len(web_tools) == 2
        assert all(t.domain == "web" for t in web_tools)

        gov_tools = registry.list_by_domain("government")
        assert len(gov_tools) == 1
        assert gov_tools[0].name == "gov1"

        # Non-existent domain returns empty list
        assert registry.list_by_domain("nonexistent") == []

    def test_registry_get_missing_returns_none(self) -> None:
        """get() returns None for unknown names."""
        assert registry.get("not_here") is None

    def test_registry_clear(self) -> None:
        """clear() removes all registered tools."""

        @tool(name="temp")
        async def temp() -> str:
            return "t"

        assert len(registry) == 1
        registry.clear()
        assert len(registry) == 0
        assert "temp" not in registry

    def test_registry_contains(self) -> None:
        """__contains__ supports 'in' operator."""

        @tool(name="exists")
        async def exists() -> str:
            return "e"

        assert "exists" in registry
        assert "missing" not in registry


# ---------------------------------------------------------------------------
# Test: ToolResult pass-through
# ---------------------------------------------------------------------------


class TestToolResultPassThrough:
    """When a function already returns ToolResult, it's enriched not re-wrapped."""

    def test_existing_tool_result_enriched(self) -> None:
        """A ToolResult return value gets latency_s and tool_name set."""

        @tool(name="pre_wrapped")
        async def pre_wrapped() -> ToolResult[str]:
            return ToolResult(success=True, data="already wrapped", text="summary")

        result = asyncio.run(pre_wrapped())
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.data == "already wrapped"
        assert result.text == "summary"
        assert result.tool_name == "pre_wrapped"
        assert result.latency_s >= 0


# ---------------------------------------------------------------------------
# Test: trace_id and task flow through
# ---------------------------------------------------------------------------


class TestKwargsFlow:
    """trace_id and task kwargs are captured in the ToolResult."""

    def test_trace_id_flows_through(self) -> None:
        """trace_id kwarg is captured in the ToolResult."""

        @tool(name="traced")
        async def traced() -> str:
            return "ok"

        result = asyncio.run(traced(trace_id="test/trace/123"))
        assert result.trace_id == "test/trace/123"

    def test_default_trace_id_is_empty(self) -> None:
        """Without trace_id kwarg, it defaults to empty string."""

        @tool(name="untraced")
        async def untraced() -> str:
            return "ok"

        result = asyncio.run(untraced())
        assert result.trace_id == ""


# ---------------------------------------------------------------------------
# Test: ToolInfo validation
# ---------------------------------------------------------------------------


class TestToolInfoValidation:
    """ToolInfo validates its fields on construction."""

    def test_invalid_cost_tier_raises(self) -> None:
        """ToolInfo rejects unknown cost_tier values."""
        with pytest.raises(ValueError, match="cost_tier"):
            ToolInfo(
                name="bad",
                domain="test",
                description="test",
                cost_tier="ultra_premium",
                func=lambda: None,
            )

    def test_valid_cost_tiers_accepted(self) -> None:
        """All four valid cost_tier values are accepted."""
        for tier in ("free", "cheap", "moderate", "expensive"):
            info = ToolInfo(
                name=f"t_{tier}",
                domain="test",
                description="test",
                cost_tier=tier,
                func=lambda: None,
            )
            assert info.cost_tier == tier

    def test_description_fallback_to_docstring(self) -> None:
        """When description is empty, @tool uses the first line of __doc__."""

        @tool(name="docstring_desc")
        async def documented() -> str:
            """This is auto-extracted."""
            return "ok"

        info = registry.get("docstring_desc")
        assert info is not None
        assert info.description == "This is auto-extracted."

    def test_explicit_description_wins(self) -> None:
        """Explicit description overrides docstring."""

        @tool(name="explicit_desc", description="My explicit desc")
        async def has_doc() -> str:
            """This is ignored."""
            return "ok"

        info = registry.get("explicit_desc")
        assert info is not None
        assert info.description == "My explicit desc"


# ---------------------------------------------------------------------------
# Test: ToolRegistry as standalone
# ---------------------------------------------------------------------------


class TestToolRegistryStandalone:
    """ToolRegistry works independently of the decorator."""

    def test_standalone_registry(self) -> None:
        """A separate ToolRegistry instance works independently."""
        custom = ToolRegistry()
        assert len(custom) == 0

        info = ToolInfo(
            name="custom_tool",
            domain="test",
            description="test tool",
            cost_tier="free",
            func=lambda: None,
        )
        custom.register(info)
        assert len(custom) == 1
        assert custom.get("custom_tool") is info
        assert custom.list_all() == [info]

        custom.clear()
        assert len(custom) == 0

    def test_has_method(self) -> None:
        """has() returns True for registered tools, False otherwise."""
        custom = ToolRegistry()
        info = ToolInfo(
            name="check_me",
            domain="test",
            description="test",
            cost_tier="free",
            func=lambda: None,
        )
        assert custom.has("check_me") is False
        custom.register(info)
        assert custom.has("check_me") is True
        assert custom.has("missing") is False


# ---------------------------------------------------------------------------
# Test: ToolInfo input_type and output_type
# ---------------------------------------------------------------------------


class TestToolInfoTypes:
    """ToolInfo captures input_type and output_type from annotations."""

    def test_output_type_captured(self) -> None:
        """output_type is extracted from return annotation when it's a concrete type."""

        @tool(name="typed_return")
        async def typed_return() -> str:
            return "ok"

        info = registry.get("typed_return")
        assert info is not None
        assert info.output_type is str

    def test_input_type_captured_single_param(self) -> None:
        """input_type is extracted when function has a single typed parameter."""

        @tool(name="single_input")
        async def single_input(query: str) -> str:
            return query

        info = registry.get("single_input")
        assert info is not None
        assert info.input_type is str

    def test_input_type_none_for_multi_params(self) -> None:
        """input_type is None when function has multiple parameters."""

        @tool(name="multi_input")
        async def multi_input(a: str, b: int) -> str:
            return f"{a}{b}"

        info = registry.get("multi_input")
        assert info is not None
        assert info.input_type is None

    def test_types_default_to_none(self) -> None:
        """input_type and output_type default to None when not annotated."""

        @tool(name="untyped")
        async def untyped():
            return "ok"

        info = registry.get("untyped")
        assert info is not None
        assert info.input_type is None
        assert info.output_type is None
