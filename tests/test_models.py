"""Tests for llm_client.models — registry, task selection, performance tracking."""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_client.models import (
    ModelInfo,
    TaskProfile,
    _DEFAULT_CONFIG,
    _reset_config,
    get_model,
    list_models,
    query_performance,
)


@pytest.fixture(autouse=True)
def _reset():
    """Reset config cache between tests."""
    _reset_config()
    yield
    _reset_config()


# ---------------------------------------------------------------------------
# get_model
# ---------------------------------------------------------------------------


class TestGetModel:
    def test_extraction_returns_highest_intelligence_structured(self):
        # available_only=False so we don't need env vars set
        model = get_model("extraction", available_only=False)
        # gemini-3-flash has intelligence=46 and structured_output=True
        assert model == "gemini/gemini-3-flash"

    def test_bulk_cheap_returns_cheapest(self):
        model = get_model("bulk_cheap", available_only=False)
        # gpt-5-nano is cheapest at $0.14
        assert model == "gpt-5-nano"

    def test_graph_building_returns_cheapest_structured(self):
        model = get_model("graph_building", available_only=False)
        # cheapest with structured_output and intel>=30:
        # gpt-5-nano (intel=27) fails intel check
        # gemini-2.5-flash-lite (intel=28) fails intel check
        # grok-4.1-fast (intel=39, cost=0.28) or deepseek (intel=42, cost=0.32)
        # grok-4.1-fast at $0.28 wins
        assert model == "xai/grok-4.1-fast"

    def test_agent_reasoning_filters_high_intelligence(self):
        model = get_model("agent_reasoning", available_only=False)
        # min_intelligence=42, prefer intelligence
        # gemini-3-flash (46), gpt-5 (45), deepseek (42)
        assert model == "gemini/gemini-3-flash"

    def test_synthesis_prefers_intelligence_then_cost(self):
        model = get_model("synthesis", available_only=False)
        # min_intelligence=40, prefer intelligence then -cost
        # gemini-3-flash (46), gpt-5 (45), deepseek (42), gpt-5-mini (41)
        assert model == "gemini/gemini-3-flash"

    def test_code_generation_prefers_intelligence_then_speed(self):
        model = get_model("code_generation", available_only=False)
        # min_intelligence=38, prefer intelligence then speed
        # gemini-3-flash (46, speed=207), gpt-5 (45, speed=98), deepseek (42, speed=36)
        assert model == "gemini/gemini-3-flash"

    def test_unknown_task_raises_keyerror(self):
        with pytest.raises(KeyError, match="Unknown task"):
            get_model("nonexistent_task")

    def test_no_models_qualify_raises_runtimeerror(self):
        # Set min_intelligence impossibly high via custom config
        with patch.dict(os.environ, {"LLM_CLIENT_MODELS_CONFIG": "/nonexistent"}):
            _reset_config()
        # Override the config cache directly
        from llm_client import models as m
        m._config_cache = {
            "models": _DEFAULT_CONFIG["models"],
            "tasks": {
                "impossible": {
                    "description": "Impossible task",
                    "require": {"min_intelligence": 999},
                    "prefer": [],
                },
            },
        }
        with pytest.raises(RuntimeError, match="No models qualify"):
            get_model("impossible", available_only=False)

    def test_available_only_filters_by_env_var(self):
        # Only set DEEPSEEK_API_KEY
        env = {
            "DEEPSEEK_API_KEY": "test-key",
            "OPENAI_API_KEY": "",
            "GEMINI_API_KEY": "",
            "XAI_API_KEY": "",
        }
        with patch.dict(os.environ, env, clear=False):
            # Remove keys that might be set
            for k in ["OPENAI_API_KEY", "GEMINI_API_KEY", "XAI_API_KEY"]:
                os.environ.pop(k, None)
            model = get_model("bulk_cheap", available_only=True)
            assert model == "deepseek/deepseek-chat"

    def test_available_only_no_keys_raises(self):
        env_clear = {
            "DEEPSEEK_API_KEY": "",
            "OPENAI_API_KEY": "",
            "GEMINI_API_KEY": "",
            "XAI_API_KEY": "",
        }
        with patch.dict(os.environ, env_clear, clear=False):
            for k in env_clear:
                os.environ.pop(k, None)
            with pytest.raises(RuntimeError, match="No models qualify"):
                get_model("extraction", available_only=True)


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------


class TestListModels:
    def test_returns_all_models_when_no_task(self):
        models = list_models(available_only=False)
        assert len(models) == len(_DEFAULT_CONFIG["models"])
        assert all("available" in m for m in models)

    def test_filters_by_task(self):
        models = list_models(task="extraction", available_only=False)
        # All should have structured_output=True and intelligence>=35
        for m in models:
            assert m["structured_output"] is True
            assert m["intelligence"] >= 35

    def test_unknown_task_raises(self):
        with pytest.raises(KeyError, match="Unknown task"):
            list_models(task="nonexistent")

    def test_sorted_by_task_prefer(self):
        models = list_models(task="extraction", available_only=False)
        # Should be sorted by intelligence desc, then cost asc
        intels = [m["intelligence"] for m in models]
        assert intels == sorted(intels, reverse=True)


# ---------------------------------------------------------------------------
# query_performance
# ---------------------------------------------------------------------------


class TestQueryPerformance:
    def test_empty_when_no_log_file(self):
        with patch("llm_client.io_log._log_dir", return_value=Path("/tmp/nonexistent_dir_xyz")):
            result = query_performance()
            assert result == []

    def test_parses_jsonl_correctly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            now = datetime.now(timezone.utc).isoformat()
            records = [
                {
                    "timestamp": now,
                    "model": "deepseek/deepseek-chat",
                    "task": "extraction",
                    "cost": 0.001,
                    "latency_s": 1.5,
                    "error": None,
                    "usage": {"total_tokens": 500},
                },
                {
                    "timestamp": now,
                    "model": "deepseek/deepseek-chat",
                    "task": "extraction",
                    "cost": 0.002,
                    "latency_s": 2.0,
                    "error": None,
                    "usage": {"total_tokens": 600},
                },
                {
                    "timestamp": now,
                    "model": "deepseek/deepseek-chat",
                    "task": "extraction",
                    "cost": 0.001,
                    "latency_s": 1.0,
                    "error": "timeout",
                    "usage": {"total_tokens": 0},
                },
            ]
            (log_dir / "calls.jsonl").write_text(
                "\n".join(json.dumps(r) for r in records) + "\n"
            )

            with patch("llm_client.io_log._log_dir", return_value=log_dir):
                result = query_performance(task="extraction")

            assert len(result) == 1
            r = result[0]
            assert r["task"] == "extraction"
            assert r["model"] == "deepseek/deepseek-chat"
            assert r["call_count"] == 3
            assert r["total_cost"] == 0.004
            assert r["avg_latency_s"] == pytest.approx(1.5, abs=0.01)
            assert r["error_rate"] == pytest.approx(0.333, abs=0.01)
            assert r["avg_tokens"] == 367  # (500+600+0)/3 rounded

    def test_filters_by_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            now = datetime.now(timezone.utc).isoformat()
            records = [
                {"timestamp": now, "model": "gpt-5", "task": "synthesis", "cost": 0.01, "latency_s": 2.0, "error": None, "usage": {"total_tokens": 1000}},
                {"timestamp": now, "model": "deepseek/deepseek-chat", "task": "synthesis", "cost": 0.001, "latency_s": 1.0, "error": None, "usage": {"total_tokens": 500}},
            ]
            (log_dir / "calls.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")

            with patch("llm_client.io_log._log_dir", return_value=log_dir):
                result = query_performance(model="gpt-5")

            assert len(result) == 1
            assert result[0]["model"] == "gpt-5"

    def test_untagged_calls_grouped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            now = datetime.now(timezone.utc).isoformat()
            records = [
                {"timestamp": now, "model": "gpt-5", "cost": 0.01, "latency_s": 1.0, "error": None, "usage": {"total_tokens": 100}},
            ]
            (log_dir / "calls.jsonl").write_text(json.dumps(records[0]) + "\n")

            with patch("llm_client.io_log._log_dir", return_value=log_dir):
                result = query_performance()

            assert len(result) == 1
            assert result[0]["task"] == "untagged"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


class TestConfigLoading:
    def test_yaml_override(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
models:
  - name: custom-model
    litellm_id: custom/model
    provider: custom
    api_key_env: CUSTOM_API_KEY
    intelligence: 50
    speed: 100
    cost: 1.0
    context: 32000
    structured_output: true
    tags: [custom]
tasks:
  custom_task:
    description: Custom task
    require:
      min_intelligence: 40
    prefer: [intelligence]
""")
            f.flush()
            try:
                with patch.dict(os.environ, {"LLM_CLIENT_MODELS_CONFIG": f.name}):
                    _reset_config()
                    model = get_model("custom_task", available_only=False)
                    assert model == "custom/model"
            finally:
                os.unlink(f.name)

    def test_env_var_nonexistent_file_raises(self):
        with patch.dict(os.environ, {"LLM_CLIENT_MODELS_CONFIG": "/tmp/nonexistent_config_xyz.yaml"}):
            _reset_config()
            with pytest.raises(RuntimeError, match="non-existent file"):
                get_model("extraction")


# ---------------------------------------------------------------------------
# io_log task field
# ---------------------------------------------------------------------------


class TestIoLogTaskField:
    def test_log_call_includes_task(self):
        """Verify log_call accepts task param and includes it in records."""
        from llm_client.io_log import log_call

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("llm_client.io_log._log_dir", return_value=Path(tmpdir)):
                with patch("llm_client.io_log._enabled", True):
                    log_call(model="test-model", task="extraction")

            log_file = Path(tmpdir) / "calls.jsonl"
            assert log_file.exists()
            record = json.loads(log_file.read_text().strip())
            assert record["task"] == "extraction"

    def test_log_call_task_none_by_default(self):
        from llm_client.io_log import log_call

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("llm_client.io_log._log_dir", return_value=Path(tmpdir)):
                with patch("llm_client.io_log._enabled", True):
                    log_call(model="test-model")

            log_file = Path(tmpdir) / "calls.jsonl"
            record = json.loads(log_file.read_text().strip())
            assert record["task"] is None
