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

from llm_client.utils.rate_limit import configure as configure_rate_limit
from llm_client.observability import (
    ActiveFeatureProfile,
    ActiveExperimentRun,
    ExperimentRun,
    activate_feature_profile,
    activate_experiment_run,
    compare_call_snapshots,
    compare_runs,
    compare_cohorts,
    configure_logging,
    configure_feature_profile,
    configure_experiment_enforcement,
    configure_agent_spec_enforcement,
    enforce_agent_spec,
    experiment_run,
    format_call_diff,
    finish_run,
    get_background_mode_adoption,
    get_call_snapshot,
    get_completed_traces,
    get_cost,
    get_active_experiment_run_id,
    get_active_feature_profile,
    get_active_llm_calls,
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
    replay_call_snapshot,
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
# Relocated (Plan #17): agent_spec → project-meta, validators → agentic_scaffolding

if _TYPE_CHECKING:
    from llm_client.difficulty import (
        DifficultyTier,
        get_model_for_difficulty,
        get_effective_tier,
        get_model_candidates_for_difficulty,
        load_model_floors,
        save_model_floors,
    )
    # Relocated (Plan #17): git_utils, scoring, experiment_eval, task_graph, analyzer

from llm_client.agent.agent_contracts import AgentErrorBudget
from llm_client.agent.mcp_agent import (
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

from llm_client.tools.tool_utils import (
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
    "compare_call_snapshots",
    "configure_logging",
    "format_call_diff",
    "get_call_snapshot",
    "import_jsonl",
    "log_embedding",
    "log_foundation_event",
    "lookup_result",
    "replay_call_snapshot",
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
    # Relocated (Plan #17): agent_spec, validators
    "callable_to_openai_tool",
    "prepare_direct_tools",
    "strip_fences",
)

# Relocated modules removed (Plan #17). Difficulty stays (control plane).
_CANDIDATE_MOVE_EXPORTS: tuple[str, ...] = (
    "DifficultyTier",
    "get_model_for_difficulty",
    "get_effective_tier",
    "load_model_floors",
    "save_model_floors",
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
    # Relocated modules removed (Plan #17)
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
