"""LLM client wrapping litellm + agent SDKs.

Swap any model by changing the model string. Everything else stays the same.

Usage:
    from llm_client import call_llm, call_llm_structured, call_llm_with_tools, stream_llm

    # Sync
    result = call_llm("gpt-4o", [{"role": "user", "content": "Hello"}])
    print(result.content, result.cost)

    # Agent SDK (same interface, different routing)
    result = call_llm("claude-code", [{"role": "user", "content": "Fix the bug"}])
    result = call_llm("claude-code/opus", [{"role": "user", "content": "Review code"}])

    # Batch (concurrent)
    results = call_llm_batch("gpt-4o", [msgs1, msgs2, msgs3], max_concurrent=5)

    # Streaming
    for chunk in stream_llm("gpt-4o", [{"role": "user", "content": "Hello"}]):
        print(chunk, end="")

    # Async
    from llm_client import acall_llm, astream_llm

    result = await acall_llm("gpt-4o", [{"role": "user", "content": "Hello"}])
"""

import logging as _logging
import os as _os
from pathlib import Path as _Path

_DEFAULT_KEYS_FILE = _Path.home() / ".secrets" / "api_keys.env"
_log = _logging.getLogger(__name__)


# Keys auto-loaded from env file (not already present in os.environ).
# Agent subprocesses (Claude Code CLI, Codex) should NOT inherit these —
# they use their own auth and auto-loaded keys like ANTHROPIC_API_KEY
# cause the bundled CLI to use the wrong auth mechanism.
_auto_loaded_keys: frozenset[str] = frozenset()


def _load_api_keys() -> int:
    """Load API keys from env file into os.environ on import.

    Reads from LLM_CLIENT_KEYS_FILE env var, or ~/.secrets/api_keys.env.
    Skips comments, empty lines, and keys already set in the environment.
    Returns the number of keys loaded.
    """
    global _auto_loaded_keys  # noqa: PLW0603
    keys_file = _Path(_os.environ.get("LLM_CLIENT_KEYS_FILE", str(_DEFAULT_KEYS_FILE)))
    if not keys_file.is_file():
        return 0
    loaded_names: list[str] = []
    for line in keys_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in _os.environ:
            _os.environ[key] = value
            loaded_names.append(key)
    _auto_loaded_keys = frozenset(loaded_names)
    if loaded_names:
        _log.debug("llm_client: loaded %d API keys from %s", len(loaded_names), keys_file)
    return len(loaded_names)


_load_api_keys()

from llm_client.errors import (
    LLMAuthError,
    LLMCapabilityError,
    LLMBudgetExceededError,
    LLMContentFilterError,
    LLMError,
    LLMModelNotFoundError,
    LLMQuotaExhaustedError,
    LLMRateLimitError,
    LLMTransientError,
    classify_error,
    wrap_error,
)

from llm_client.io_log import configure as configure_logging
from llm_client.rate_limit import configure as configure_rate_limit
from llm_client.io_log import (
    compare_runs,
    finish_run,
    get_completed_traces,
    get_cost,
    get_run_items,
    get_runs,
    get_trace_tree,
    import_jsonl,
    log_embedding,
    log_item,
    lookup_result,
    start_run,
)
from llm_client.models import (
    ModelInfo,
    TaskProfile,
    get_model,
    list_models,
    query_performance,
)
from llm_client.prompts import render_prompt
from llm_client.validators import ValidationResult, run_validators, register_validator, spec_hash
from llm_client.difficulty import (
    DifficultyTier,
    get_model_for_difficulty,
    get_effective_tier,
    load_model_floors,
    save_model_floors,
)
from llm_client.task_graph import (
    ExecutionReport,
    ExperimentRecord,
    GraphMeta,
    TaskDef,
    TaskGraph,
    TaskResult,
    TaskStatus,
    load_graph,
    run_graph,
    toposort_waves,
)
from llm_client.analyzer import (
    AnalysisReport,
    IssueCategory,
    Proposal,
    analyze_history,
    analyze_run,
    analyze_scores,
    check_scorer_reliability,
)
from llm_client.git_utils import (
    CODE_CHANGE,
    CONFIG_CHANGE,
    PROMPT_CHANGE,
    RUBRIC_CHANGE,
    TEST_CHANGE,
    classify_diff_files,
    get_diff_files,
    get_git_head,
)

from llm_client.mcp_agent import (
    DEFAULT_MAX_TURNS,
    DEFAULT_MCP_INIT_TIMEOUT,
    DEFAULT_TOOL_RESULT_MAX_LENGTH,
    MCPAgentResult,
    MCPSessionPool,
    MCPToolCallRecord,
)

from llm_client.tool_utils import (
    callable_to_openai_tool,
    prepare_direct_tools,
)

from llm_client.scoring import (
    CriterionScore,
    Rubric,
    RubricCriterion,
    ScoreResult,
    ascore_output,
    list_rubrics,
    load_rubric,
    score_output,
)

from llm_client.client import (
    AsyncCachePolicy,
    AsyncLLMStream,
    CachePolicy,
    EmbeddingResult,
    Hooks,
    LLMCallResult,
    LLMStream,
    LRUCache,
    RetryPolicy,
    acall_llm,
    acall_llm_batch,
    acall_llm_structured,
    acall_llm_structured_batch,
    acall_llm_with_tools,
    aembed,
    astream_llm,
    astream_llm_with_tools,
    call_llm,
    call_llm_batch,
    call_llm_structured,
    call_llm_structured_batch,
    call_llm_with_tools,
    embed,
    exponential_backoff,
    fixed_backoff,
    linear_backoff,
    stream_llm,
    stream_llm_with_tools,
    strip_fences,
)

__all__ = [
    "LLMAuthError",
    "LLMCapabilityError",
    "LLMBudgetExceededError",
    "LLMContentFilterError",
    "LLMError",
    "LLMModelNotFoundError",
    "LLMQuotaExhaustedError",
    "LLMRateLimitError",
    "LLMTransientError",
    "classify_error",
    "wrap_error",
    "AsyncCachePolicy",
    "DEFAULT_MAX_TURNS",
    "DEFAULT_MCP_INIT_TIMEOUT",
    "DEFAULT_TOOL_RESULT_MAX_LENGTH",
    "MCPAgentResult",
    "MCPSessionPool",
    "MCPToolCallRecord",
    "AsyncLLMStream",
    "CachePolicy",
    "EmbeddingResult",
    "Hooks",
    "LLMCallResult",
    "LLMStream",
    "LRUCache",
    "RetryPolicy",
    "acall_llm",
    "acall_llm_batch",
    "acall_llm_structured",
    "acall_llm_structured_batch",
    "acall_llm_with_tools",
    "aembed",
    "astream_llm",
    "astream_llm_with_tools",
    "call_llm",
    "call_llm_batch",
    "call_llm_structured",
    "call_llm_structured_batch",
    "call_llm_with_tools",
    "embed",
    "exponential_backoff",
    "fixed_backoff",
    "linear_backoff",
    "stream_llm",
    "stream_llm_with_tools",
    "configure_logging",
    "import_jsonl",
    "log_embedding",
    "lookup_result",
    "get_completed_traces",
    "get_cost",
    # experiment logging
    "start_run",
    "log_item",
    "finish_run",
    "get_runs",
    "get_run_items",
    "compare_runs",
    "render_prompt",
    "strip_fences",
    "ModelInfo",
    "TaskProfile",
    "get_model",
    "list_models",
    "query_performance",
    # validators
    "ValidationResult",
    "run_validators",
    "register_validator",
    "spec_hash",
    # difficulty
    "DifficultyTier",
    "get_model_for_difficulty",
    "get_effective_tier",
    "load_model_floors",
    "save_model_floors",
    # task_graph
    "ExecutionReport",
    "ExperimentRecord",
    "GraphMeta",
    "TaskDef",
    "TaskGraph",
    "TaskResult",
    "TaskStatus",
    "load_graph",
    "run_graph",
    "toposort_waves",
    # analyzer
    "AnalysisReport",
    "IssueCategory",
    "Proposal",
    "analyze_history",
    "analyze_run",
    "analyze_scores",
    "check_scorer_reliability",
    # git_utils
    "CODE_CHANGE",
    "CONFIG_CHANGE",
    "PROMPT_CHANGE",
    "RUBRIC_CHANGE",
    "TEST_CHANGE",
    "classify_diff_files",
    "get_diff_files",
    "get_git_head",
    # tool_utils
    "callable_to_openai_tool",
    "prepare_direct_tools",
    # scoring
    "CriterionScore",
    "Rubric",
    "RubricCriterion",
    "ScoreResult",
    "ascore_output",
    "list_rubrics",
    "load_rubric",
    "score_output",
]
