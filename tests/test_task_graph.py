"""Tests for llm_client.task_graph."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from llm_client.task_graph import (
    ExecutionReport,
    ExperimentRecord,
    GraphMeta,
    TaskDef,
    TaskGraph,
    TaskResult,
    TaskStatus,
    _make_experiment_record,
    _resolve_templates,
    _validate_dag,
    load_graph,
    run_graph,
    toposort_waves,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_yaml(tmp_path: Path) -> Path:
    """A simple 3-task linear graph."""
    out_a = tmp_path / "a.txt"
    out_b = tmp_path / "b.txt"
    content = f"""
graph:
  id: test_graph
  description: "Test graph"
  timeout_minutes: 10
  checkpoint: none

tasks:
  task_a:
    difficulty: 1
    agent: codex
    prompt: "Do task A"
    validate:
      - type: file_exists
        path: {out_a}
    outputs:
      result: "{out_a}"

  task_b:
    difficulty: 2
    agent: codex
    depends_on: [task_a]
    prompt: "Process {{task_a.outputs.result}}"
    validate:
      - type: file_exists
        path: {out_b}

  task_c:
    difficulty: 1
    agent: codex
    prompt: "Independent task C"
"""
    f = tmp_path / "graph.yaml"
    f.write_text(content)
    return f


@pytest.fixture
def parallel_yaml(tmp_path: Path) -> Path:
    """A graph with parallel tasks and a join."""
    content = """
graph:
  id: parallel_test
  description: "Parallel tasks"
  timeout_minutes: 5
  checkpoint: none

tasks:
  fetch_a:
    difficulty: 1
    prompt: "Fetch A"
  fetch_b:
    difficulty: 1
    prompt: "Fetch B"
  merge:
    difficulty: 2
    depends_on: [fetch_a, fetch_b]
    prompt: "Merge A and B"
"""
    f = tmp_path / "parallel.yaml"
    f.write_text(content)
    return f


@pytest.fixture
def cyclic_yaml(tmp_path: Path) -> Path:
    content = """
graph:
  id: cyclic
  description: "Has a cycle"
  timeout_minutes: 1
  checkpoint: none

tasks:
  a:
    difficulty: 1
    prompt: "A"
    depends_on: [b]
  b:
    difficulty: 1
    prompt: "B"
    depends_on: [a]
"""
    f = tmp_path / "cyclic.yaml"
    f.write_text(content)
    return f


@pytest.fixture
def missing_dep_yaml(tmp_path: Path) -> Path:
    content = """
graph:
  id: bad_dep
  description: "Missing dep"
  timeout_minutes: 1
  checkpoint: none

tasks:
  a:
    difficulty: 1
    prompt: "A"
    depends_on: [nonexistent]
"""
    f = tmp_path / "bad_dep.yaml"
    f.write_text(content)
    return f


# ---------------------------------------------------------------------------
# Graph parsing
# ---------------------------------------------------------------------------


def test_load_graph(simple_yaml: Path):
    graph = load_graph(simple_yaml)
    assert graph.meta.id == "test_graph"
    assert len(graph.tasks) == 3
    assert "task_a" in graph.tasks
    assert "task_b" in graph.tasks
    assert "task_c" in graph.tasks
    assert graph.tasks["task_b"].depends_on == ["task_a"]


def test_load_graph_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_graph(tmp_path / "nope.yaml")


# ---------------------------------------------------------------------------
# DAG validation
# ---------------------------------------------------------------------------


def test_validate_dag_cycle(cyclic_yaml: Path):
    with pytest.raises(ValueError, match="Cycle detected"):
        load_graph(cyclic_yaml)


def test_validate_dag_missing_dep(missing_dep_yaml: Path):
    with pytest.raises(ValueError, match="doesn't exist"):
        load_graph(missing_dep_yaml)


# ---------------------------------------------------------------------------
# Topological sort
# ---------------------------------------------------------------------------


def test_toposort_linear(simple_yaml: Path):
    graph = load_graph(simple_yaml)
    waves = graph.waves
    # task_a and task_c are independent (wave 0), task_b depends on task_a (wave 1)
    assert len(waves) == 2
    assert set(waves[0]) == {"task_a", "task_c"}
    assert waves[1] == ["task_b"]


def test_toposort_parallel(parallel_yaml: Path):
    graph = load_graph(parallel_yaml)
    waves = graph.waves
    assert len(waves) == 2
    assert set(waves[0]) == {"fetch_a", "fetch_b"}
    assert waves[1] == ["merge"]


# ---------------------------------------------------------------------------
# Template resolution
# ---------------------------------------------------------------------------


def test_resolve_templates():
    text = "Process {task_a.outputs.result} and {task_b.outputs.data}"
    completed_outputs = {
        "task_a": {"result": "/tmp/a.txt"},
        "task_b": {"data": "/tmp/b.json"},
    }
    resolved = _resolve_templates(text, {}, completed_outputs)
    assert resolved == "Process /tmp/a.txt and /tmp/b.json"


def test_resolve_templates_missing_ref():
    text = "Process {missing.outputs.thing}"
    resolved = _resolve_templates(text, {}, {})
    assert resolved == text  # Unresolved references left as-is


# ---------------------------------------------------------------------------
# run_graph (dry_run)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_graph_dry_run(simple_yaml: Path):
    graph = load_graph(simple_yaml)
    # mock-ok: dry_run=True doesn't dispatch agents, but difficulty router needs models
    with patch("llm_client.difficulty._is_ollama_available", return_value=False), \
         patch.dict("os.environ", {"DEEPSEEK_API_KEY": "k", "GEMINI_API_KEY": "k"}):
        report = await run_graph(graph, dry_run=True)
    assert report.status == "completed"
    assert report.waves_completed == 2
    assert len(report.task_results) == 3
    assert all(tr.status == TaskStatus.COMPLETED for tr in report.task_results)


# ---------------------------------------------------------------------------
# run_graph (mocked dispatch)
# ---------------------------------------------------------------------------


class _FakeResult:
    """Minimal mock of LLMCallResult."""
    def __init__(
        self,
        content: str = "done",
        cost: float = 0.01,
        routing_trace: dict[str, object] | None = None,
    ):
        self.content = content
        self.cost = cost
        self.usage = {"prompt_tokens": 100, "completion_tokens": 50}
        self.finish_reason = "stop"
        self.routing_trace = routing_trace


@pytest.mark.asyncio
async def test_run_graph_success(simple_yaml: Path, tmp_path: Path):
    """Full graph execution with mocked acall_llm and passing validators."""
    graph = load_graph(simple_yaml)
    exp_log = tmp_path / "experiments.jsonl"

    # Create files that validators expect (in tmp_path, matching the YAML fixture)
    (tmp_path / "a.txt").write_text("output a")
    (tmp_path / "b.txt").write_text("output b")

    # mock-ok: testing graph execution flow, not real LLM calls
    mock_acall = AsyncMock(return_value=_FakeResult())
    with patch("llm_client.task_graph._acall_llm", mock_acall), \
         patch("llm_client.difficulty._is_ollama_available", return_value=False), \
         patch.dict("os.environ", {"DEEPSEEK_API_KEY": "k", "GEMINI_API_KEY": "k"}):
        report = await run_graph(graph, experiment_log=exp_log)

    assert report.status == "completed"
    assert report.waves_completed == 2
    assert len(report.task_results) == 3
    assert all(tr.status == TaskStatus.COMPLETED for tr in report.task_results)
    assert report.total_cost_usd > 0

    # Check experiment log was written
    assert exp_log.exists()
    lines = exp_log.read_text().strip().splitlines()
    assert len(lines) == 3  # One per task

    # Verify experiment record structure
    record = json.loads(lines[0])
    assert "run_id" in record
    assert "task_id" in record
    assert record["run_id"] == "test_graph"


@pytest.mark.asyncio
async def test_run_graph_validation_failure(simple_yaml: Path, tmp_path: Path):
    """Graph stops when a task fails validation."""
    graph = load_graph(simple_yaml)
    exp_log = tmp_path / "experiments.jsonl"

    # Don't create tmp_path/a.txt — task_a's file_exists validator will fail

    # mock-ok: testing graph execution flow
    mock_acall = AsyncMock(return_value=_FakeResult())
    with patch("llm_client.task_graph._acall_llm", mock_acall), \
         patch("llm_client.difficulty._is_ollama_available", return_value=False), \
         patch.dict("os.environ", {"DEEPSEEK_API_KEY": "k", "GEMINI_API_KEY": "k"}):
        report = await run_graph(graph, experiment_log=exp_log)

    # task_a failed validation, so graph should be partial
    assert report.status == "partial"
    # task_b should not have been attempted (depends on task_a)
    task_a = next(tr for tr in report.task_results if tr.task_id == "task_a")
    assert task_a.status == TaskStatus.FAILED
    # task_c is in the same wave as task_a, so it still ran
    task_c = next(tr for tr in report.task_results if tr.task_id == "task_c")
    assert task_c.status == TaskStatus.COMPLETED  # No validators on task_c


@pytest.mark.asyncio
async def test_run_graph_agent_error(simple_yaml: Path, tmp_path: Path):
    """Graph handles agent exceptions gracefully."""
    graph = load_graph(simple_yaml)
    exp_log = tmp_path / "experiments.jsonl"

    # mock-ok: testing error handling
    mock_acall = AsyncMock(side_effect=RuntimeError("API crashed"))
    with patch("llm_client.task_graph._acall_llm", mock_acall), \
         patch("llm_client.difficulty._is_ollama_available", return_value=False), \
         patch.dict("os.environ", {"DEEPSEEK_API_KEY": "k", "GEMINI_API_KEY": "k"}):
        report = await run_graph(graph, experiment_log=exp_log)

    assert report.status == "partial"
    for tr in report.task_results:
        assert tr.status == TaskStatus.FAILED
        assert "API crashed" in tr.error


# ---------------------------------------------------------------------------
# Agent/non-agent kwargs routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_override_non_agent_strips_agent_kwargs(tmp_path: Path):
    """Gemini override should not inherit Codex-only kwargs."""
    content = f"""
graph:
  id: non_agent_override
  description: "Model override routes to non-agent runtime"
  timeout_minutes: 5
  checkpoint: none

tasks:
  t1:
    difficulty: 1
    agent: codex
    model: gemini/gemini-2.5-flash
    prompt: "Hello"
    working_directory: {tmp_path}
"""
    f = tmp_path / "non_agent_override.yaml"
    f.write_text(content)
    graph = load_graph(f)

    captured_kwargs = {}

    async def capture_acall(model, messages, **kwargs):
        captured_kwargs.update(kwargs)
        return _FakeResult()

    mock_acall = AsyncMock(side_effect=capture_acall)
    with patch("llm_client.task_graph._acall_llm", mock_acall), \
         patch.dict("os.environ", {"GEMINI_API_KEY": "k"}):
        report = await run_graph(graph, experiment_log=tmp_path / "exp.jsonl")

    assert report.status == "completed"
    assert "approval_policy" not in captured_kwargs
    assert "working_directory" not in captured_kwargs
    assert "cwd" not in captured_kwargs
    assert captured_kwargs.get("execution_mode") == "text"


@pytest.mark.asyncio
async def test_model_override_agent_keeps_agent_kwargs(tmp_path: Path):
    """Codex override should keep Codex runtime kwargs."""
    content = f"""
graph:
  id: agent_override
  description: "Model override routes to codex runtime"
  timeout_minutes: 5
  checkpoint: none

tasks:
  t1:
    difficulty: 1
    agent: codex
    model: codex/gpt-5
    prompt: "Hello"
    working_directory: {tmp_path}
"""
    f = tmp_path / "agent_override.yaml"
    f.write_text(content)
    graph = load_graph(f)

    captured_kwargs = {}

    async def capture_acall(model, messages, **kwargs):
        captured_kwargs.update(kwargs)
        return _FakeResult()

    mock_acall = AsyncMock(side_effect=capture_acall)
    with patch("llm_client.task_graph._acall_llm", mock_acall):
        report = await run_graph(graph, experiment_log=tmp_path / "exp.jsonl")

    assert report.status == "completed"
    assert captured_kwargs.get("approval_policy") == "never"
    assert captured_kwargs.get("working_directory") == str(tmp_path.resolve())
    assert "cwd" not in captured_kwargs
    assert captured_kwargs.get("execution_mode") == "workspace_agent"


@pytest.mark.asyncio
async def test_codex_skip_git_repo_check_passthrough(tmp_path: Path):
    """Codex tasks can opt into skip_git_repo_check in graph YAML."""
    content = f"""
graph:
  id: codex_skip_git_repo_check
  description: "Codex skip git repo check passthrough"
  timeout_minutes: 5
  checkpoint: none

tasks:
  t1:
    difficulty: 4
    agent: codex
    model: codex
    prompt: "Hello"
    working_directory: {tmp_path}
    skip_git_repo_check: true
"""
    f = tmp_path / "codex_skip_git_repo_check.yaml"
    f.write_text(content)
    graph = load_graph(f)

    captured_kwargs = {}

    async def capture_acall(model, messages, **kwargs):
        captured_kwargs.update(kwargs)
        return _FakeResult()

    mock_acall = AsyncMock(side_effect=capture_acall)
    with patch("llm_client.task_graph._acall_llm", mock_acall):
        report = await run_graph(graph, experiment_log=tmp_path / "exp.jsonl")

    assert report.status == "completed"
    assert captured_kwargs.get("approval_policy") == "never"
    assert captured_kwargs.get("skip_git_repo_check") is True
    assert captured_kwargs.get("working_directory") == str(tmp_path.resolve())
    assert captured_kwargs.get("execution_mode") == "workspace_agent"


@pytest.mark.asyncio
async def test_non_agent_mcp_sets_workspace_tools_mode(tmp_path: Path):
    """Non-agent MCP tasks should declare workspace_tools execution mode."""
    content = """
graph:
  id: non_agent_mcp_mode
  description: "Non-agent MCP execution mode"
  timeout_minutes: 5
  checkpoint: none

tasks:
  t1:
    difficulty: 2
    agent: codex
    model: gemini/gemini-2.5-flash
    prompt: "Use MCP tool"
    mcp_servers: [demo_server]
"""
    f = tmp_path / "non_agent_mcp_mode.yaml"
    f.write_text(content)
    graph = load_graph(f)

    captured_kwargs = {}

    async def capture_acall(model, messages, **kwargs):
        captured_kwargs.update(kwargs)
        return _FakeResult()

    mock_acall = AsyncMock(side_effect=capture_acall)
    with patch("llm_client.task_graph._acall_llm", mock_acall), \
         patch.dict("os.environ", {"GEMINI_API_KEY": "k"}):
        report = await run_graph(
            graph,
            experiment_log=tmp_path / "exp.jsonl",
            mcp_server_configs={
                "demo_server": {"command": "python", "args": ["server.py"]},
            },
        )

    assert report.status == "completed"
    assert captured_kwargs.get("execution_mode") == "workspace_tools"
    assert "mcp_servers" in captured_kwargs


@pytest.mark.asyncio
async def test_reasoning_effort_passthrough_to_call_kwargs(tmp_path: Path):
    """Task-level reasoning_effort should be forwarded to acall_llm kwargs."""
    content = """
graph:
  id: reasoning_effort_passthrough
  description: "reasoning_effort passthrough"
  timeout_minutes: 5
  checkpoint: none

tasks:
  t1:
    difficulty: 2
    model: gpt-5.2-pro
    prompt: "Deep review"
    reasoning_effort: xhigh
"""
    f = tmp_path / "reasoning_effort_passthrough.yaml"
    f.write_text(content)
    graph = load_graph(f)

    captured_kwargs = {}

    async def capture_acall(model, messages, **kwargs):
        captured_kwargs.update(kwargs)
        return _FakeResult()

    mock_acall = AsyncMock(side_effect=capture_acall)
    with patch("llm_client.task_graph._acall_llm", mock_acall):
        report = await run_graph(graph, experiment_log=tmp_path / "exp.jsonl")

    assert report.status == "completed"
    assert captured_kwargs.get("reasoning_effort") == "xhigh"
    task_result = report.task_results[0]
    assert task_result.reasoning_effort == "xhigh"
    assert task_result.background_mode is None


@pytest.mark.asyncio
async def test_task_result_captures_background_mode_from_routing_trace(tmp_path: Path):
    """Task result telemetry should capture background_mode from call routing trace."""
    content = """
graph:
  id: background_mode_capture
  description: "background mode capture"
  timeout_minutes: 5
  checkpoint: none

tasks:
  t1:
    difficulty: 2
    model: gpt-5.2-pro
    prompt: "Deep review"
    reasoning_effort: xhigh
"""
    f = tmp_path / "background_mode_capture.yaml"
    f.write_text(content)
    graph = load_graph(f)

    async def capture_acall(model, messages, **kwargs):
        return _FakeResult(routing_trace={"background_mode": True})

    mock_acall = AsyncMock(side_effect=capture_acall)
    with patch("llm_client.task_graph._acall_llm", mock_acall):
        report = await run_graph(graph, experiment_log=tmp_path / "exp.jsonl")

    assert report.status == "completed"
    task_result = report.task_results[0]
    assert task_result.reasoning_effort == "xhigh"
    assert task_result.background_mode is True


# ---------------------------------------------------------------------------
# Task execution with investigation questions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_investigation_questions_prepended(tmp_path: Path):
    """Investigation questions are prepended to the prompt."""
    content = """
graph:
  id: invest_test
  description: "Investigation test"
  timeout_minutes: 5
  checkpoint: none

tasks:
  build:
    difficulty: 1
    prompt: "Build the thing"
    investigate_first:
      - "Does the data directory exist?"
      - "What format are the files?"
"""
    f = tmp_path / "invest.yaml"
    f.write_text(content)
    graph = load_graph(f)

    captured_messages: list = []

    async def capture_acall(model, messages, **kwargs):
        captured_messages.extend(messages)
        return _FakeResult()

    mock_acall = AsyncMock(side_effect=capture_acall)
    with patch("llm_client.task_graph._acall_llm", mock_acall), \
         patch("llm_client.difficulty._is_ollama_available", return_value=False), \
         patch.dict("os.environ", {"DEEPSEEK_API_KEY": "k"}):
        await run_graph(graph, experiment_log=tmp_path / "exp.jsonl")

    assert len(captured_messages) == 1
    prompt = captured_messages[0]["content"]
    assert "INVESTIGATION PHASE" in prompt
    assert "Does the data directory exist?" in prompt
    assert "What format are the files?" in prompt
    assert "Build the thing" in prompt


# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------


def test_task_def_defaults():
    t = TaskDef(id="test", difficulty=2, prompt="do it")
    assert t.agent == "codex"
    assert t.depends_on == []
    assert t.mcp_servers == []
    assert t.timeout == 300


def test_graph_meta_defaults():
    m = GraphMeta(id="test")
    assert m.checkpoint == "git_tag"
    assert m.timeout_minutes == 120


def test_experiment_record_serialization():
    record = ExperimentRecord(
        run_id="test",
        task_id="t1",
        wave=0,
        timestamp="2026-02-16T00:00:00Z",
        hypothesis="test hypothesis",
        difficulty=2,
        model_selected="deepseek/deepseek-chat",
        agent="codex",
        result={"status": "success"},
        outcome="confirmed",
    )
    data = json.loads(record.model_dump_json())
    assert data["run_id"] == "test"
    assert data["outcome"] == "confirmed"


def test_make_experiment_record_defaults_null_agent_to_codex():
    """Nullable task.agent should not break experiment record serialization."""
    graph = TaskGraph(
        meta=GraphMeta(id="agent_default"),
        tasks={
            "t1": TaskDef(id="t1", difficulty=1, prompt="do work", agent=None),
        },
        waves=[["t1"]],
    )
    task_result = TaskResult(
        task_id="t1",
        status=TaskStatus.COMPLETED,
        wave=0,
        difficulty=1,
        reasoning_effort="xhigh",
        background_mode=True,
    )

    record = _make_experiment_record(graph, task_result)
    assert record.agent == "codex"
    assert "codex at tier 1 can handle t1" == record.hypothesis
    assert record.dimensions["reasoning_effort"] == "xhigh"
    assert record.dimensions["background_mode"] is True
