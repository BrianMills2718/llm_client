"""Shared test fixtures for llm_client test suite.

Disables observability logging during tests to prevent test mocks and fixtures
from polluting the production JSONL + SQLite observability data.
"""

import os

import pytest


@pytest.fixture(autouse=True)
def _disable_observability_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent tests from writing to the real observability DB/JSONL."""
    monkeypatch.setenv("LLM_CLIENT_LOG_ENABLED", "0")
