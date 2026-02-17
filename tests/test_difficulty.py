"""Tests for llm_client.difficulty."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_client.difficulty import (
    DifficultyTier,
    get_effective_tier,
    get_model_for_difficulty,
    load_model_floors,
    save_model_floors,
    _is_model_available,
    _reset_tiers,
)


@pytest.fixture(autouse=True)
def _reset():
    _reset_tiers()
    yield
    _reset_tiers()


# --- get_model_for_difficulty ---


def test_tier_0_returns_none():
    assert get_model_for_difficulty(0) is None


def test_tier_invalid_raises():
    with pytest.raises(ValueError, match="Invalid difficulty tier"):
        get_model_for_difficulty(99)


def test_override_model_bypasses_routing():
    result = get_model_for_difficulty(2, override_model="my-custom-model")
    assert result == "my-custom-model"


def test_available_only_false_returns_first():
    # With available_only=False, should return first model in tier
    model = get_model_for_difficulty(1, available_only=False)
    assert model == "ollama/llama3.1"


@patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}, clear=False)
def test_tier_1_skips_ollama_when_unavailable():
    # mock-ok: ollama availability check is system-dependent
    with patch("llm_client.difficulty._is_ollama_available", return_value=False):
        model = get_model_for_difficulty(1)
        assert model == "deepseek/deepseek-chat"


@patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=False)
def test_tier_2_returns_gemini():
    model = get_model_for_difficulty(2)
    assert model == "gemini/gemini-2.5-flash"


def test_tier_4_returns_agent():
    # Agent SDKs are always "available"
    model = get_model_for_difficulty(4)
    assert model in ("codex", "claude-code")


@patch.dict("os.environ", {}, clear=True)
def test_no_available_models_raises():
    # mock-ok: testing edge case where no API keys are set
    with patch("llm_client.difficulty._is_ollama_available", return_value=False):
        with pytest.raises(RuntimeError, match="No available models"):
            get_model_for_difficulty(1)


# --- _is_model_available ---


@patch.dict("os.environ", {"DEEPSEEK_API_KEY": "key"}, clear=False)
def test_deepseek_available():
    assert _is_model_available("deepseek/deepseek-chat") is True


@patch.dict("os.environ", {}, clear=True)
def test_deepseek_unavailable():
    assert _is_model_available("deepseek/deepseek-chat") is False


def test_agent_models_always_available():
    assert _is_model_available("codex") is True
    assert _is_model_available("claude-code") is True
    assert _is_model_available("codex/gpt-5.3-codex") is True
    assert _is_model_available("claude-code/opus") is True


# --- model floors ---


def test_load_model_floors_missing_file(tmp_path: Path):
    floors = load_model_floors(tmp_path / "nope.json")
    assert floors == {}


def test_save_and_load_model_floors(tmp_path: Path):
    path = tmp_path / "floors.json"
    data = {
        "collect_sources": {"floor": 1, "ceiling": 2, "last_tested": "2026-02-20", "runs": 14},
    }
    save_model_floors(data, path)
    loaded = load_model_floors(path)
    assert loaded == data


def test_save_creates_parent_dirs(tmp_path: Path):
    path = tmp_path / "sub" / "dir" / "floors.json"
    save_model_floors({"task": {"floor": 1}}, path)
    assert path.exists()


# --- get_effective_tier ---


def test_effective_tier_no_floors():
    assert get_effective_tier("new_task", 3) == 3


def test_effective_tier_with_lower_floor():
    floors = {"my_task": {"floor": 1, "ceiling": 3, "last_tested": "2026-02-20", "runs": 10}}
    assert get_effective_tier("my_task", 3, floors) == 1


def test_effective_tier_never_exceeds_declared():
    floors = {"my_task": {"floor": 4, "ceiling": 4, "last_tested": "2026-02-20", "runs": 5}}
    # Floor is 4, declared is 2 — should use 2 (never upgrade above declared)
    assert get_effective_tier("my_task", 2, floors) == 2


def test_effective_tier_floor_at_zero():
    floors = {"my_task": {"floor": 0, "ceiling": 1, "last_tested": "2026-02-20", "runs": 20}}
    assert get_effective_tier("my_task", 2, floors) == 0
