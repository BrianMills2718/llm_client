"""Typed shared observability contract for non-LLM tool calls.

Wave 0 keeps this deliberately small: one dataclass plus one logging helper.
Projects and shared libraries can emit retrieval, fetch, and extraction
outcomes into the existing llm_client observability backend without adopting
the heavier foundation-event schema.

Wave 1 adds size tracking and data-loss detection: ``raw_size`` and
``processed_size`` fields enable automated alerts when post-processing
(linearization, truncation) silently drops content.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from llm_client import io_log as _io_log

logger = logging.getLogger(__name__)

ToolCallStatus = Literal["started", "succeeded", "failed"]

# Data-loss detection thresholds (configurable per-project via overrides).
_DATA_LOSS_RATIO_THRESHOLD = 0.1
_DATA_LOSS_RAW_SIZE_FLOOR = 100


@dataclass(frozen=True, slots=True)
class ToolCallResult:
    """Record one shared non-LLM tool call lifecycle event.

    The contract is intentionally compact and SQL-friendly. A single operation
    attempt may emit a ``started`` row followed by either ``succeeded`` or
    ``failed`` using the same ``call_id``.

    **Wave 1 fields** (all optional, backward-compatible):

    - ``result_count``: number of items returned (search hits, rows, etc.)
    - ``cost``: monetary cost of the API call, if applicable
    - ``raw_size``: bytes/chars of raw response before post-processing
    - ``processed_size``: bytes/chars after post-processing (linearization,
      truncation, extraction)
    - ``query_json``: input parameters dict for reproducibility

    **Data-loss detection**: If ``processed_size / raw_size < 0.1`` and
    ``raw_size > 100``, the ``data_loss_warning`` property returns ``True``.
    This catches linearization bugs, truncation errors, and format mismatches
    where post-processing silently drops content.
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
    # Wave 1: size tracking and data-loss detection
    result_count: int | None = None
    cost: float | None = None
    raw_size: int = 0
    processed_size: int = 0
    query_json: dict[str, Any] | None = None

    @property
    def data_loss_warning(self) -> bool:
        """Return True if post-processing appears to have silently dropped content.

        Triggers when the processed/raw size ratio falls below 10% and the raw
        payload was non-trivial (> 100 bytes/chars). Designed to catch
        linearization bugs like the DIGIMON ``chunk_retrieve`` incident where
        847 chars of valid data became "No chunks found."
        """
        if self.raw_size <= _DATA_LOSS_RAW_SIZE_FLOOR:
            return False
        if self.processed_size < 0 or self.raw_size < 0:
            return False
        ratio = self.processed_size / self.raw_size
        return ratio < _DATA_LOSS_RATIO_THRESHOLD

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
    """Persist one tool-call record through the shared llm_client backend.

    When ``data_loss_warning`` is True, a warning is emitted to the logger
    before writing -- this makes silent data loss visible in process logs
    even before anyone queries the observability DB.
    """

    if result.data_loss_warning:
        logger.warning(
            "Data loss detected: %s/%s processed_size=%d raw_size=%d "
            "(ratio=%.2f < %.2f threshold)",
            result.tool_name,
            result.operation,
            result.processed_size,
            result.raw_size,
            result.processed_size / result.raw_size if result.raw_size else 0,
            _DATA_LOSS_RATIO_THRESHOLD,
        )

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
        result_count=result.result_count,
        cost=result.cost,
        raw_size=result.raw_size,
        processed_size=result.processed_size,
        query_json=result.query_json,
        data_loss_warning=result.data_loss_warning,
    )


__all__ = ["ToolCallResult", "ToolCallStatus", "log_tool_call"]
