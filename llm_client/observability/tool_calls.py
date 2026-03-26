"""Typed shared observability contract for non-LLM tool calls.

Wave 0 keeps this deliberately small: one dataclass plus one logging helper.
Projects and shared libraries can emit retrieval, fetch, and extraction
outcomes into the existing llm_client observability backend without adopting
the heavier foundation-event schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from llm_client import io_log as _io_log


ToolCallStatus = Literal["started", "succeeded", "failed"]


@dataclass(frozen=True, slots=True)
class ToolCallResult:
    """Record one shared non-LLM tool call lifecycle event.

    The contract is intentionally compact and SQL-friendly. A single operation
    attempt may emit a `started` row followed by either `succeeded` or `failed`
    using the same `call_id`.
    """

    call_id: str
    tool_name: str
    operation: str
    status: ToolCallStatus
    started_at: str
    provider: str | None = None
    target: str | None = None
    ended_at: str | None = None
    duration_ms: int | None = None
    attempt: int = 1
    task: str | None = None
    trace_id: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        """Reject structurally invalid records before they hit persistence."""

        if not self.call_id.strip():
            raise ValueError("call_id must be non-empty")
        if not self.tool_name.strip():
            raise ValueError("tool_name must be non-empty")
        if not self.operation.strip():
            raise ValueError("operation must be non-empty")
        if self.attempt < 1:
            raise ValueError("attempt must be >= 1")
        if self.status == "started":
            if self.ended_at is not None:
                raise ValueError("started status must not set ended_at")
            if self.duration_ms is not None:
                raise ValueError("started status must not set duration_ms")
        if self.status in {"succeeded", "failed"}:
            if self.ended_at is None:
                raise ValueError("final status must set ended_at")
            if self.duration_ms is None:
                raise ValueError("final status must set duration_ms")
            if self.duration_ms < 0:
                raise ValueError("duration_ms must be >= 0")


def log_tool_call(result: ToolCallResult) -> None:
    """Persist one tool-call record through the shared llm_client backend."""

    _io_log.log_tool_call_record(
        call_id=result.call_id,
        tool_name=result.tool_name,
        operation=result.operation,
        provider=result.provider,
        target=result.target,
        status=result.status,
        started_at=result.started_at,
        ended_at=result.ended_at,
        duration_ms=result.duration_ms,
        attempt=result.attempt,
        task=result.task,
        trace_id=result.trace_id,
        metrics=result.metrics,
        error_type=result.error_type,
        error_message=result.error_message,
    )


__all__ = ["ToolCallResult", "ToolCallStatus", "log_tool_call"]
