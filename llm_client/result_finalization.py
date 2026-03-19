"""Finalize completed result objects after dispatch.

This module owns the pure bookkeeping that happens only after a provider or
agent runtime has already produced a result object:

1. create a cache-hit view without mutating the cached source object,
2. attach stable identity/routing metadata for return to callers.

The module does not parse provider responses and does not decide whether
results should be cached, logged, or counted toward budgets. Those policy
decisions stay in the runtime entrypoints that call into this seam.
"""

from __future__ import annotations

from copy import copy as _copy
from typing import Any, Protocol, TypeVar

from llm_client.result_metadata import ResultIdentityTarget, annotate_result_identity


class FinalizableResult(ResultIdentityTarget, Protocol):
    """Structural contract for results that support cache-hit normalization."""

    cost_source: str
    marginal_cost: float | None
    cache_hit: bool


FinalizableResultT = TypeVar("FinalizableResultT", bound=FinalizableResult)


def cache_hit_view(result: FinalizableResultT) -> FinalizableResultT:
    """Return a shallow cache-hit view without mutating cached source state."""
    cached_result = _copy(result)
    cached_result.cache_hit = True
    cached_result.marginal_cost = 0.0
    cached_result.cost_source = "cache_hit"
    return cached_result


def finalize_result(
    result: FinalizableResultT,
    *,
    requested_model: str,
    resolved_model: str | None = None,
    routing_trace: dict[str, Any] | None = None,
    warning_records: list[dict[str, Any]] | None = None,
    cache_hit: bool = False,
) -> FinalizableResultT:
    """Finalize a completed result for caller return.

    When ``cache_hit`` is true, a shallow copy is created first so the cached
    source object keeps its original accounting fields.
    """
    final_result = cache_hit_view(result) if cache_hit else result
    return annotate_result_identity(
        final_result,
        requested_model=requested_model,
        resolved_model=resolved_model,
        routing_trace=routing_trace,
        warning_records=warning_records,
    )
