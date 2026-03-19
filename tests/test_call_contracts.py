"""Tests for the extracted pre-call contract helpers.

# mock-ok: these helpers depend on observability backend queries and guardrail
# hooks; patching isolates the contract logic without requiring real SQLite
# state or process-wide profile configuration.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from llm_client.call_contracts import check_budget, normalize_prompt_ref, require_tags
from llm_client.errors import LLMBudgetExceededError


def test_require_tags_calls_observability_guardrails() -> None:
    """Resolved tasks still invoke the same feature-profile and experiment guards."""
    with (
        patch("llm_client.call_contracts._io_log.enforce_feature_profile") as mock_feature_enforce,
        patch("llm_client.call_contracts._io_log.enforce_experiment_context") as mock_experiment_enforce,
    ):
        require_tags(
            "digimon.benchmark",
            "trace.required.tags",
            0,
            caller="test_call_contracts",
        )

    mock_feature_enforce.assert_called_once_with(
        "digimon.benchmark",
        caller="llm_client.client",
    )
    mock_experiment_enforce.assert_called_once_with(
        "digimon.benchmark",
        caller="llm_client.client",
    )


def test_normalize_prompt_ref_rejects_blank_values() -> None:
    """Blank prompt references fail loudly before they reach observability."""
    with pytest.raises(ValueError, match="prompt_ref must not be empty"):
        normalize_prompt_ref("   ")


def test_check_budget_raises_when_trace_is_spent() -> None:
    """Budget enforcement happens before dispatch using current trace spend."""
    with patch("llm_client.call_contracts._io_log.get_cost", return_value=5.0):
        with pytest.raises(LLMBudgetExceededError, match="Budget exceeded for trace trace/budget"):
            check_budget("trace/budget", 5.0)
