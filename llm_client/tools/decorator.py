"""@tool decorator: ToolResult envelope, observability logging, and registry.

Wraps any async function so it:
1. Returns a ToolResult envelope (success/failure, data, latency, error info)
2. Logs the call to llm_client's observability backend
3. Registers itself in a global ToolRegistry for discovery

Usage::

    from llm_client.tools import tool, registry

    @tool(name="fetch_page", domain="web", description="Fetch a URL", cost_tier="cheap")
    async def fetch_page(url: str) -> str:
        ...
        return html

    result = await fetch_page(url="https://example.com")
    assert isinstance(result, ToolResult)
    assert result.success
    print(result.data)  # the raw return value
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")
logger = logging.getLogger(__name__)


@dataclass
class ToolResult(Generic[T]):
    """Universal result envelope for all tool calls.

    Wraps the return value (or error) of a tool invocation with metadata
    needed for observability and context management.

    Attributes:
        success: Whether the tool call completed without raising.
        data: The raw return value from the tool function (None on failure).
        text: Linearized text summary for LLM consumption. Tools can set this
            to provide a compact representation of structured data.
        error: Error message string if the call failed.
        error_type: Exception class name if the call failed.
        latency_s: Wall-clock execution time in seconds.
        tool_name: Name of the tool as registered in the registry.
        trace_id: Trace ID for correlating with other observability events.
        retries: Number of retry attempts (0 = first attempt succeeded/failed).
    """

    success: bool
    data: T | None = None
    text: str | None = None
    error: str | None = None
    error_type: str | None = None
    latency_s: float = 0.0
    tool_name: str = ""
    trace_id: str = ""
    retries: int = 0


@dataclass
class ToolInfo:
    """Metadata about a registered tool function.

    Attached to the decorated function as ``func._tool_info`` for introspection
    and used by ToolRegistry for discovery.
    """

    name: str
    domain: str
    description: str
    cost_tier: str  # free | cheap | moderate | expensive
    func: Callable[..., Any]

    def __post_init__(self) -> None:
        """Validate cost_tier is one of the allowed values."""
        allowed = {"free", "cheap", "moderate", "expensive"}
        if self.cost_tier not in allowed:
            raise ValueError(
                f"cost_tier must be one of {allowed}, got {self.cost_tier!r}"
            )


class ToolRegistry:
    """Auto-populated registry of @tool decorated functions.

    A singleton instance (``registry``) is created at module level. The @tool
    decorator registers each function automatically on decoration.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolInfo] = {}

    def register(self, info: ToolInfo) -> None:
        """Register a tool. Raises ValueError if name already taken."""
        if info.name in self._tools:
            raise ValueError(
                f"Tool {info.name!r} is already registered in the @tool registry. "
                "Use a different name or call registry.clear() first."
            )
        self._tools[info.name] = info
        logger.debug("Registered @tool: %s (domain=%s)", info.name, info.domain)

    def get(self, name: str) -> ToolInfo | None:
        """Look up a tool by name. Returns None if not found."""
        return self._tools.get(name)

    def list_all(self) -> list[ToolInfo]:
        """Return all registered tools, sorted by name."""
        return sorted(self._tools.values(), key=lambda t: t.name)

    def list_by_domain(self, domain: str) -> list[ToolInfo]:
        """Return tools in a specific domain, sorted by name."""
        return sorted(
            (t for t in self._tools.values() if t.domain == domain),
            key=lambda t: t.name,
        )

    def clear(self) -> None:
        """Remove all registered tools. Primarily for testing."""
        self._tools.clear()

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# Module-level singleton
registry = ToolRegistry()


def tool(
    name: str,
    *,
    domain: str = "general",
    description: str = "",
    cost_tier: str = "cheap",
) -> Callable[..., Any]:
    """Decorator that wraps an async function with ToolResult, observability, and registration.

    The decorated function will:
    - Return a ``ToolResult`` envelope instead of a raw value
    - Catch all exceptions and wrap them in ``ToolResult(success=False, ...)``
    - Log each call to the observability backend via ``log_tool_call``
    - Register itself in the global ``registry`` on decoration

    Args:
        name: Unique tool name for the registry and observability.
        domain: Logical grouping (e.g. "web", "government", "social").
        description: Human-readable description. Falls back to docstring.
        cost_tier: One of "free", "cheap", "moderate", "expensive".

    Raises:
        TypeError: If the decorated function is not async.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if not asyncio.iscoroutinefunction(func):
            raise TypeError(
                f"@tool requires an async function, but {func.__name__!r} is sync. "
                "Use 'async def' instead."
            )

        desc = description or (func.__doc__ or "").strip().split("\n")[0]
        info = ToolInfo(
            name=name,
            domain=domain,
            description=desc,
            cost_tier=cost_tier,
            func=func,
        )
        registry.register(info)

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> ToolResult[Any]:
            """Execute the wrapped tool with ToolResult envelope and observability."""
            call_id = uuid.uuid4().hex[:8]
            trace_id = kwargs.get("trace_id", "")
            task = kwargs.get("task", name)
            start = time.monotonic()
            started_at = datetime.now(timezone.utc).isoformat()
            result: ToolResult[Any] | None = None

            try:
                raw = await func(*args, **kwargs)
                latency = time.monotonic() - start

                # If the function already returns a ToolResult, enrich it
                if isinstance(raw, ToolResult):
                    raw.latency_s = latency
                    raw.tool_name = name
                    raw.trace_id = trace_id
                    result = raw
                else:
                    result = ToolResult(
                        success=True,
                        data=raw,
                        latency_s=latency,
                        tool_name=name,
                        trace_id=trace_id,
                    )
                return result
            except Exception as exc:
                latency = time.monotonic() - start
                result = ToolResult(
                    success=False,
                    error=str(exc),
                    error_type=type(exc).__name__,
                    latency_s=latency,
                    tool_name=name,
                    trace_id=trace_id,
                )
                return result
            finally:
                # Log to observability — never let observability break the tool
                try:
                    _log_to_observability(
                        call_id=call_id,
                        name=name,
                        task=task,
                        trace_id=trace_id,
                        started_at=started_at,
                        result=result,
                    )
                except Exception:
                    logger.debug(
                        "Observability logging failed for tool %s (non-fatal)", name,
                        exc_info=True,
                    )

        wrapper._tool_info = info  # type: ignore[attr-defined]
        return wrapper

    return decorator


def _log_to_observability(
    *,
    call_id: str,
    name: str,
    task: str,
    trace_id: str,
    started_at: str,
    result: ToolResult[Any] | None,
) -> None:
    """Write one tool call record to the observability backend.

    Isolated in its own function so the try/except in the decorator stays clean.
    """
    from llm_client.observability.tool_calls import (
        ToolCallResult,
        log_tool_call,
    )

    if result is None:
        return

    duration_ms = int(result.latency_s * 1000)
    ended_at = datetime.now(timezone.utc).isoformat()

    log_tool_call(
        ToolCallResult(
            call_id=call_id,
            tool_name=name,
            operation=name,
            status="succeeded" if result.success else "failed",
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=duration_ms,
            task=task,
            trace_id=trace_id or None,
            error_type=result.error_type,
            error_message=result.error,
        )
    )
