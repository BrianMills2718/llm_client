"""Runtime substrate for multi-agent LLM execution.

`llm_client` is the shared public facade for provider transport, agent SDK
routing, prompt rendering, observability, retries/fallbacks, and related
runtime policy. LiteLLM remains the commodity transport backend, but this
package owns the higher-level execution contract that project code relies on.

Usage:
    from llm_client import call_llm, call_llm_structured, call_llm_with_tools, stream_llm

    # Sync
    result = call_llm(
        "gpt-4o",
        [{"role": "user", "content": "Hello"}],
        task="demo",
        trace_id="init/sync",
        max_budget=1.0,
    )
    print(result.content, result.cost)

    # Agent SDK (same interface, different routing)
    result = call_llm(
        "claude-code",
        [{"role": "user", "content": "Fix the bug"}],
        task="demo_agent",
        trace_id="init/agent",
        max_budget=0,
    )
    result = call_llm(
        "claude-code/opus",
        [{"role": "user", "content": "Review code"}],
        task="demo_agent",
        trace_id="init/agent_opus",
        max_budget=0,
    )

    # Batch (concurrent)
    results = call_llm_batch(
        "gpt-4o",
        [msgs1, msgs2, msgs3],
        max_concurrent=5,
        task="demo_batch",
        trace_id="init/batch",
        max_budget=2.0,
    )

    # Streaming
    for chunk in stream_llm(
        "gpt-4o",
        [{"role": "user", "content": "Hello"}],
        task="demo_stream",
        trace_id="init/stream",
        max_budget=1.0,
    ):
        print(chunk, end="")

    # Async
    from llm_client import acall_llm, astream_llm

    result = await acall_llm(
        "gpt-4o",
        [{"role": "user", "content": "Hello"}],
        task="demo_async",
        trace_id="init/async",
        max_budget=1.0,
    )
"""

# ruff: noqa: E402,F401

from importlib import import_module as _import_module
import logging as _logging
import os as _os
from pathlib import Path as _Path
from typing import TYPE_CHECKING as _TYPE_CHECKING
import warnings as _warnings

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
    LLMConfigurationError,
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

from llm_client.rate_limit import configure as configure_rate_limit
from llm_client.observability import (
    ActiveFeatureProfile,
    ActiveExperimentRun,
    ExperimentRun,
    activate_feature_profile,
    activate_experiment_run,
    compare_runs,
    compare_cohorts,
    configure_logging,
    configure_feature_profile,
    configure_experiment_enforcement,
    configure_agent_spec_enforcement,
    enforce_agent_spec,
    experiment_run,
    finish_run,
    get_active_llm_calls,
    get_background_mode_adoption,
    get_completed_traces,
    get_cost,
    get_active_experiment_run_id,
    get_active_feature_profile,
    get_experiment_aggregates,
    get_run,
    get_run_items,
    get_runs,
    get_trace_tree,
    import_jsonl,
    log_embedding,
    log_experiment_aggregate,
    log_foundation_event,
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
from llm_client.model_selection import (
    ResolvedModelChain,
    ResolvedModelSelection,
    resolve_model_chain,
    resolve_model_selection,
    strict_model_policy,
)
from llm_client.prompt_assets import (
    PromptAssetManifest,
    PromptAssetRef,
    ResolvedPromptAsset,
    load_prompt_asset,
    parse_prompt_ref,
    resolve_prompt_asset,
)
from llm_client.prompts import render_prompt
from llm_client.agent_spec import (
    AgentSpecValidationError,
    REQUIRED_AGENT_SPEC_SECTIONS,
    load_agent_spec,
    validate_agent_spec,
)
from llm_client.validators import ValidationResult, run_validators, register_validator, spec_hash

if _TYPE_CHECKING:
    from llm_client.difficulty import (
        DifficultyTier,
        get_model_for_difficulty,
        get_effective_tier,
        get_model_candidates_for_difficulty,
        load_model_floors,
        save_model_floors,
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
        get_working_tree_files,
        is_git_dirty,
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
    from llm_client.experiment_eval import (
        DEFAULT_DETERMINISTIC_CHECKS,
        build_gate_signals,
        evaluate_gate_policy,
        extract_adoption_profile,
        extract_agent_outcome,
        load_gate_policy,
        review_items_with_rubric,
        run_deterministic_checks_for_item,
        run_deterministic_checks_for_items,
        summarize_adoption_profiles,
        summarize_agent_outcomes,
        triage_items,
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

from llm_client.mcp_agent import (
    DEFAULT_ENFORCE_TOOL_CONTRACTS,
    DEFAULT_INITIAL_ARTIFACTS,
    DEFAULT_MAX_TURNS,
    DEFAULT_MAX_TOOL_CALLS,
    DEFAULT_MCP_INIT_TIMEOUT,
    DEFAULT_TOOL_RESULT_MAX_LENGTH,
    MCPAgentResult,
    MCPSessionPool,
    MCPToolCallRecord,
)

from llm_client.tool_utils import (
    callable_to_openai_tool,
    lint_tool_callable,
    lint_tool_registry,
    prepare_direct_tools,
)

from llm_client.config import ClientConfig
from llm_client.routing import CallRequest, ResolvedCallPlan, resolve_call

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
from llm_client.data_types import TurnEvent

_CORE_SUBSTRATE_EXPORTS: tuple[str, ...] = (
    "LLMAuthError",
    "LLMConfigurationError",
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
    "AsyncLLMStream",
    "CachePolicy",
    "EmbeddingResult",
    "Hooks",
    "LLMCallResult",
    "LLMStream",
    "LRUCache",
    "TurnEvent",
    "ClientConfig",
    "CallRequest",
    "ResolvedCallPlan",
    "resolve_call",
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
    "log_foundation_event",
    "lookup_result",
    "get_cost",
    "start_run",
    "log_item",
    "finish_run",
    "get_runs",
    "get_run",
    "get_run_items",
    "compare_runs",
    "compare_cohorts",
    "ModelInfo",
    "TaskProfile",
    "get_model",
    "list_models",
    "query_performance",
    "render_prompt",
)

_COMPAT_HOLD_EXPORTS: tuple[str, ...] = (
    "DEFAULT_MAX_TURNS",
    "DEFAULT_MAX_TOOL_CALLS",
    "DEFAULT_MCP_INIT_TIMEOUT",
    "DEFAULT_TOOL_RESULT_MAX_LENGTH",
    "DEFAULT_ENFORCE_TOOL_CONTRACTS",
    "DEFAULT_INITIAL_ARTIFACTS",
    "MCPAgentResult",
    "MCPSessionPool",
    "MCPToolCallRecord",
    "ActiveFeatureProfile",
    "ActiveExperimentRun",
    "ExperimentRun",
    "activate_feature_profile",
    "activate_experiment_run",
    "configure_feature_profile",
    "experiment_run",
    "configure_experiment_enforcement",
    "configure_agent_spec_enforcement",
    "enforce_agent_spec",
    "get_active_experiment_run_id",
    "get_active_feature_profile",
    "get_active_llm_calls",
    "get_background_mode_adoption",
    "get_completed_traces",
    "AgentSpecValidationError",
    "REQUIRED_AGENT_SPEC_SECTIONS",
    "load_agent_spec",
    "validate_agent_spec",
    "ValidationResult",
    "run_validators",
    "register_validator",
    "spec_hash",
    "callable_to_openai_tool",
    "prepare_direct_tools",
    "strip_fences",
)

_CANDIDATE_MOVE_EXPORTS: tuple[str, ...] = (
    "DifficultyTier",
    "get_model_for_difficulty",
    "get_effective_tier",
    "load_model_floors",
    "save_model_floors",
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
    "AnalysisReport",
    "IssueCategory",
    "Proposal",
    "analyze_history",
    "analyze_run",
    "analyze_scores",
    "check_scorer_reliability",
    "CODE_CHANGE",
    "CONFIG_CHANGE",
    "PROMPT_CHANGE",
    "RUBRIC_CHANGE",
    "TEST_CHANGE",
    "classify_diff_files",
    "get_diff_files",
    "get_git_head",
    "get_working_tree_files",
    "is_git_dirty",
    "CriterionScore",
    "Rubric",
    "RubricCriterion",
    "ScoreResult",
    "ascore_output",
    "list_rubrics",
    "load_rubric",
    "score_output",
    "DEFAULT_DETERMINISTIC_CHECKS",
    "run_deterministic_checks_for_item",
    "run_deterministic_checks_for_items",
    "review_items_with_rubric",
    "load_gate_policy",
    "build_gate_signals",
    "evaluate_gate_policy",
    "triage_items",
)

__all__ = [
    *_CORE_SUBSTRATE_EXPORTS,
    *_COMPAT_HOLD_EXPORTS,
    *_CANDIDATE_MOVE_EXPORTS,
]

_DEPRECATED_TOP_LEVEL_EXPORTS: dict[str, tuple[str, str]] = {
    "DifficultyTier": (
        "llm_client.difficulty",
        "DifficultyTier",
    ),
    "get_model_for_difficulty": (
        "llm_client.difficulty",
        "get_model_for_difficulty",
    ),
    "get_model_candidates_for_difficulty": (
        "llm_client.difficulty",
        "get_model_candidates_for_difficulty",
    ),
    "get_effective_tier": (
        "llm_client.difficulty",
        "get_effective_tier",
    ),
    "load_model_floors": (
        "llm_client.difficulty",
        "load_model_floors",
    ),
    "save_model_floors": (
        "llm_client.difficulty",
        "save_model_floors",
    ),
    "CODE_CHANGE": (
        "llm_client.git_utils",
        "CODE_CHANGE",
    ),
    "CONFIG_CHANGE": (
        "llm_client.git_utils",
        "CONFIG_CHANGE",
    ),
    "PROMPT_CHANGE": (
        "llm_client.git_utils",
        "PROMPT_CHANGE",
    ),
    "RUBRIC_CHANGE": (
        "llm_client.git_utils",
        "RUBRIC_CHANGE",
    ),
    "TEST_CHANGE": (
        "llm_client.git_utils",
        "TEST_CHANGE",
    ),
    "classify_diff_files": (
        "llm_client.git_utils",
        "classify_diff_files",
    ),
    "get_diff_files": (
        "llm_client.git_utils",
        "get_diff_files",
    ),
    "get_git_head": (
        "llm_client.git_utils",
        "get_git_head",
    ),
    "get_working_tree_files": (
        "llm_client.git_utils",
        "get_working_tree_files",
    ),
    "is_git_dirty": (
        "llm_client.git_utils",
        "is_git_dirty",
    ),
    "CriterionScore": (
        "llm_client.scoring",
        "CriterionScore",
    ),
    "Rubric": (
        "llm_client.scoring",
        "Rubric",
    ),
    "RubricCriterion": (
        "llm_client.scoring",
        "RubricCriterion",
    ),
    "ScoreResult": (
        "llm_client.scoring",
        "ScoreResult",
    ),
    "ascore_output": (
        "llm_client.scoring",
        "ascore_output",
    ),
    "list_rubrics": (
        "llm_client.scoring",
        "list_rubrics",
    ),
    "load_rubric": (
        "llm_client.scoring",
        "load_rubric",
    ),
    "score_output": (
        "llm_client.scoring",
        "score_output",
    ),
    "DEFAULT_DETERMINISTIC_CHECKS": (
        "llm_client.experiment_eval",
        "DEFAULT_DETERMINISTIC_CHECKS",
    ),
    "build_gate_signals": (
        "llm_client.experiment_eval",
        "build_gate_signals",
    ),
    "evaluate_gate_policy": (
        "llm_client.experiment_eval",
        "evaluate_gate_policy",
    ),
    "extract_adoption_profile": (
        "llm_client.experiment_eval",
        "extract_adoption_profile",
    ),
    "extract_agent_outcome": (
        "llm_client.experiment_eval",
        "extract_agent_outcome",
    ),
    "load_gate_policy": (
        "llm_client.experiment_eval",
        "load_gate_policy",
    ),
    "review_items_with_rubric": (
        "llm_client.experiment_eval",
        "review_items_with_rubric",
    ),
    "run_deterministic_checks_for_item": (
        "llm_client.experiment_eval",
        "run_deterministic_checks_for_item",
    ),
    "run_deterministic_checks_for_items": (
        "llm_client.experiment_eval",
        "run_deterministic_checks_for_items",
    ),
    "summarize_adoption_profiles": (
        "llm_client.experiment_eval",
        "summarize_adoption_profiles",
    ),
    "summarize_agent_outcomes": (
        "llm_client.experiment_eval",
        "summarize_agent_outcomes",
    ),
    "triage_items": (
        "llm_client.experiment_eval",
        "triage_items",
    ),
    "ExecutionReport": (
        "llm_client.task_graph",
        "ExecutionReport",
    ),
    "ExperimentRecord": (
        "llm_client.task_graph",
        "ExperimentRecord",
    ),
    "GraphMeta": (
        "llm_client.task_graph",
        "GraphMeta",
    ),
    "TaskDef": (
        "llm_client.task_graph",
        "TaskDef",
    ),
    "TaskGraph": (
        "llm_client.task_graph",
        "TaskGraph",
    ),
    "TaskResult": (
        "llm_client.task_graph",
        "TaskResult",
    ),
    "TaskStatus": (
        "llm_client.task_graph",
        "TaskStatus",
    ),
    "load_graph": (
        "llm_client.task_graph",
        "load_graph",
    ),
    "run_graph": (
        "llm_client.task_graph",
        "run_graph",
    ),
    "toposort_waves": (
        "llm_client.task_graph",
        "toposort_waves",
    ),
    "AnalysisReport": (
        "llm_client.analyzer",
        "AnalysisReport",
    ),
    "IssueCategory": (
        "llm_client.analyzer",
        "IssueCategory",
    ),
    "Proposal": (
        "llm_client.analyzer",
        "Proposal",
    ),
    "analyze_history": (
        "llm_client.analyzer",
        "analyze_history",
    ),
    "analyze_run": (
        "llm_client.analyzer",
        "analyze_run",
    ),
    "analyze_scores": (
        "llm_client.analyzer",
        "analyze_scores",
    ),
    "check_scorer_reliability": (
        "llm_client.analyzer",
        "check_scorer_reliability",
    ),
}


def _load_deprecated_top_level_export(name: str) -> object:
    """Resolve a deprecated top-level export lazily and emit migration guidance.

    The stable home for these names is now their module namespace. The package
    root keeps them only as a compatibility shim during the deprecation window.
    """
    module_name, attr_name = _DEPRECATED_TOP_LEVEL_EXPORTS[name]
    _warnings.warn(
        (
            f"`llm_client.{name}` is deprecated; import it from "
            f"`{module_name}` instead."
        ),
        DeprecationWarning,
        stacklevel=2,
    )
    module = _import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __getattr__(name: str) -> object:
    """Resolve deprecated top-level exports lazily.

    This preserves compatibility for existing imports while steering callers
    toward the module-level home of candidate-move surfaces.
    """
    if name in _DEPRECATED_TOP_LEVEL_EXPORTS:
        return _load_deprecated_top_level_export(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
