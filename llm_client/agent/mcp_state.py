"""Loop configuration and state initialization for the MCP agent loop.

Contains the typed runtime-policy dataclass, agent tool-state dataclass,
and their construction functions. These hold per-run configuration
and initial capability/artifact/binding state for the agent loop.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from llm_client.agent.agent_adoption import (
    DEFAULT_ADOPTION_PROFILE,
    normalize_adoption_profile,
)
from llm_client.agent.agent_contracts import (
    _analyze_lane_closure as _agent_analyze_lane_closure,
    _capability_requirement_from_raw as _agent_capability_requirement_from_raw,
    _capability_state_add as _agent_capability_state_add,
    _capability_state_snapshot as _agent_capability_state_snapshot,
    _normalize_artifact_kind as _agent_normalize_artifact_kind,
)
from llm_client.agent.compliance_gate import build_tool_parameter_index
from llm_client.foundation import (
    normalize_bindings,
    sha256_json,
)
from llm_client.agent.mcp_context import (
    DEFAULT_TOOL_RESULT_CONTEXT_PREVIEW_CHARS,
    DEFAULT_TOOL_RESULT_KEEP_RECENT,
)
from llm_client.agent.mcp_evidence import (
    DEFAULT_RETRIEVAL_STAGNATION_ACTION,
    DEFAULT_RETRIEVAL_STAGNATION_TURNS,
    RETRIEVAL_STAGNATION_ACTIONS,
)
from llm_client.agent.mcp_finalization import (
    DEFAULT_ACCEPT_FORCED_ANSWER_ON_MAX_TOOL_CALLS,
    DEFAULT_FORCE_SUBMIT_RETRY_ON_MAX_TOOL_CALLS,
    DEFAULT_FORCED_FINAL_CIRCUIT_BREAKER_THRESHOLD,
    DEFAULT_FORCED_FINAL_MAX_ATTEMPTS,
)
from llm_client.tools.tool_runtime_common import (
    normalize_tool_contracts as _shared_normalize_tool_contracts,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_INITIAL_ARTIFACTS: tuple[str, ...] = ("QUERY_TEXT",)
"""Default artifact kinds available before any tool call."""

DEFAULT_ACTIVE_ARTIFACT_CONTEXT_ENABLED: bool = True
"""Keep a rolling compact summary of recent typed artifact handles in active prompt context."""

DEFAULT_ACTIVE_ARTIFACT_CONTEXT_MAX_HANDLES: int = 8
"""Maximum recent typed artifact handles shown in the rolling artifact-context summary."""

DEFAULT_ACTIVE_ARTIFACT_CONTEXT_MAX_CHARS: int = 900
"""Maximum chars for the rolling artifact-context summary injected into active context."""

ADOPTION_PROFILE_ENV: str = "LLM_CLIENT_ADOPTION_PROFILE"
"""Optional default adoption profile when not passed explicitly."""

ADOPTION_PROFILE_ENFORCE_ENV: str = "LLM_CLIENT_ADOPTION_PROFILE_ENFORCE"
"""When true, adoption-profile violations raise instead of only warning."""

EVENT_CODE_TOOL_VALIDATION_MISSING_PREREQUISITE = "TOOL_VALIDATION_REJECTED_MISSING_PREREQUISITE"
EVENT_CODE_TOOL_VALIDATION_MISSING_CAPABILITY = "TOOL_VALIDATION_REJECTED_MISSING_CAPABILITY"
EVENT_CODE_TOOL_VALIDATION_BINDING_CONFLICT = "TOOL_VALIDATION_REJECTED_BINDING_CONFLICT"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _coerce_bool(value: Any, default: bool) -> bool:
    """Parse bool-like runtime values with safe defaults."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off", ""}:
            return False
    return default


def _normalize_model_chain(raw: Any) -> list[str]:
    """Normalize fallback model config into a deterministic string list."""
    if raw is None:
        return []

    values: list[str] = []
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",")]
        values.extend([p for p in parts if p])
    elif isinstance(raw, (list, tuple, set)):
        for item in raw:
            if not isinstance(item, str):
                continue
            value = item.strip()
            if value:
                values.append(value)

    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentLoopRuntimePolicy:
    """Typed runtime policy consumed by _agent_loop."""

    forced_final_max_attempts: int
    forced_final_circuit_breaker_threshold: int
    forced_final_breaker_effective: bool
    force_submit_retry_on_max_tool_calls: bool
    accept_forced_answer_on_max_tool_calls: bool
    finalization_fallback_models: list[str]
    retrieval_stagnation_turns: int
    retrieval_stagnation_action: str
    tool_result_keep_recent: int
    tool_result_context_preview_chars: int
    active_artifact_context_enabled: bool
    active_artifact_context_max_handles: int
    active_artifact_context_max_chars: int
    adoption_profile: str
    adoption_profile_enforce: bool
    run_config_spec: dict[str, Any]
    run_config_hash: str


@dataclass
class AgentLoopToolState:
    """Initial tool/contract state used by _agent_loop execution stages."""

    normalized_tool_contracts: dict[str, dict[str, Any]]
    tool_parameter_index: dict[str, dict[str, Any]]
    available_artifacts: set[str]
    initial_artifact_snapshot: list[str]
    available_capabilities: dict[str, set[tuple[str | None, str | None, str | None]]]
    initial_capability_snapshot: list[dict[str, Any]]
    available_bindings: dict[str, Any]
    initial_binding_snapshot: dict[str, Any]
    lane_closure_analysis: dict[str, Any]
    artifact_timeline: list[dict[str, Any]]
    requires_submit_answer: bool


# ---------------------------------------------------------------------------
# Construction functions
# ---------------------------------------------------------------------------

def _resolve_agent_runtime_policy(
    *,
    model: str,
    max_turns: int,
    max_tool_calls: int | None,
    require_tool_reasoning: bool,
    enforce_tool_contracts: bool,
    progressive_tool_disclosure: bool,
    suppress_control_loop_calls: bool,
    tool_result_max_length: int,
    max_message_chars: int,
    kwargs: dict[str, Any],
    warning_sink: list[str],
) -> AgentLoopRuntimePolicy:
    """Extract, normalize, and hash runtime-policy settings from kwargs."""
    run_config_spec: dict[str, Any] = {
        "requested_model": model,
        "max_turns": max_turns,
        "max_tool_calls": max_tool_calls,
        "require_tool_reasoning": require_tool_reasoning,
        "enforce_tool_contracts": enforce_tool_contracts,
        "progressive_tool_disclosure": progressive_tool_disclosure,
        "suppress_control_loop_calls": suppress_control_loop_calls,
        "tool_result_max_length": tool_result_max_length,
        "max_message_chars": max_message_chars,
    }
    runtime_policy_kwargs = {
        k: kwargs.get(k)
        for k in (
            "temperature",
            "top_p",
            "timeout",
            "num_retries",
            "fallback_models",
            "fallback_limit",
            "execution_mode",
            "api_base",
            "service_tier",
            "forced_final_max_attempts",
            "forced_final_circuit_breaker_threshold",
            "force_submit_retry_on_max_tool_calls",
            "accept_forced_answer_on_max_tool_calls",
            "finalization_fallback_models",
            "retrieval_stagnation_turns",
            "retrieval_stagnation_action",
            "tool_result_keep_recent",
            "tool_result_context_preview_chars",
            "active_artifact_context_enabled",
            "active_artifact_context_max_handles",
            "active_artifact_context_max_chars",
            "adoption_profile",
            "adoption_profile_enforce",
        )
        if k in kwargs
    }
    run_config_spec["runtime_policy"] = runtime_policy_kwargs
    run_config_hash = sha256_json(run_config_spec).replace("sha256:", "")

    forced_final_max_attempts_raw = kwargs.pop(
        "forced_final_max_attempts",
        DEFAULT_FORCED_FINAL_MAX_ATTEMPTS,
    )
    try:
        forced_final_max_attempts = int(forced_final_max_attempts_raw)
    except Exception:
        forced_final_max_attempts = DEFAULT_FORCED_FINAL_MAX_ATTEMPTS
    forced_final_max_attempts = max(1, forced_final_max_attempts)

    forced_final_breaker_threshold_raw = kwargs.pop(
        "forced_final_circuit_breaker_threshold",
        DEFAULT_FORCED_FINAL_CIRCUIT_BREAKER_THRESHOLD,
    )
    try:
        forced_final_circuit_breaker_threshold = int(forced_final_breaker_threshold_raw)
    except Exception:
        forced_final_circuit_breaker_threshold = DEFAULT_FORCED_FINAL_CIRCUIT_BREAKER_THRESHOLD
    forced_final_circuit_breaker_threshold = max(1, forced_final_circuit_breaker_threshold)
    forced_final_breaker_effective = (
        forced_final_circuit_breaker_threshold <= forced_final_max_attempts
    )
    if not forced_final_breaker_effective:
        warning = (
            "FINALIZATION_BREAKER_INERT: circuit breaker threshold "
            f"({forced_final_circuit_breaker_threshold}) exceeds forced_final_max_attempts "
            f"({forced_final_max_attempts}); breaker cannot trigger under current config."
        )
        warning_sink.append(warning)
        logger.warning(warning)

    force_submit_retry_on_max_tool_calls = _coerce_bool(
        kwargs.pop(
            "force_submit_retry_on_max_tool_calls",
            DEFAULT_FORCE_SUBMIT_RETRY_ON_MAX_TOOL_CALLS,
        ),
        DEFAULT_FORCE_SUBMIT_RETRY_ON_MAX_TOOL_CALLS,
    )
    accept_forced_answer_on_max_tool_calls = _coerce_bool(
        kwargs.pop(
            "accept_forced_answer_on_max_tool_calls",
            DEFAULT_ACCEPT_FORCED_ANSWER_ON_MAX_TOOL_CALLS,
        ),
        DEFAULT_ACCEPT_FORCED_ANSWER_ON_MAX_TOOL_CALLS,
    )

    finalization_fallback_models = _normalize_model_chain(
        kwargs.pop("finalization_fallback_models", None)
    )

    retrieval_stagnation_turns_raw = kwargs.pop(
        "retrieval_stagnation_turns",
        DEFAULT_RETRIEVAL_STAGNATION_TURNS,
    )
    try:
        retrieval_stagnation_turns = int(retrieval_stagnation_turns_raw)
    except Exception:
        retrieval_stagnation_turns = DEFAULT_RETRIEVAL_STAGNATION_TURNS
    retrieval_stagnation_turns = max(2, retrieval_stagnation_turns)

    retrieval_stagnation_action_raw = kwargs.pop(
        "retrieval_stagnation_action",
        DEFAULT_RETRIEVAL_STAGNATION_ACTION,
    )
    retrieval_stagnation_action = str(retrieval_stagnation_action_raw or "").strip().lower()
    if retrieval_stagnation_action not in RETRIEVAL_STAGNATION_ACTIONS:
        warning = (
            "RETRIEVAL_STAGNATION_ACTION_INVALID: unsupported action "
            f"{retrieval_stagnation_action_raw!r}; defaulting to "
            f"{DEFAULT_RETRIEVAL_STAGNATION_ACTION!r}."
        )
        warning_sink.append(warning)
        logger.warning(warning)
        retrieval_stagnation_action = DEFAULT_RETRIEVAL_STAGNATION_ACTION

    tool_result_keep_recent_raw = kwargs.pop(
        "tool_result_keep_recent",
        DEFAULT_TOOL_RESULT_KEEP_RECENT,
    )
    try:
        tool_result_keep_recent = int(tool_result_keep_recent_raw)
    except Exception:
        tool_result_keep_recent = DEFAULT_TOOL_RESULT_KEEP_RECENT
    tool_result_keep_recent = max(0, tool_result_keep_recent)

    tool_result_context_preview_chars_raw = kwargs.pop(
        "tool_result_context_preview_chars",
        DEFAULT_TOOL_RESULT_CONTEXT_PREVIEW_CHARS,
    )
    try:
        tool_result_context_preview_chars = int(tool_result_context_preview_chars_raw)
    except Exception:
        tool_result_context_preview_chars = DEFAULT_TOOL_RESULT_CONTEXT_PREVIEW_CHARS
    tool_result_context_preview_chars = max(40, tool_result_context_preview_chars)
    run_config_spec["tool_result_keep_recent"] = tool_result_keep_recent
    run_config_spec["tool_result_context_preview_chars"] = tool_result_context_preview_chars

    active_artifact_context_enabled = _coerce_bool(
        kwargs.pop(
            "active_artifact_context_enabled",
            DEFAULT_ACTIVE_ARTIFACT_CONTEXT_ENABLED,
        ),
        DEFAULT_ACTIVE_ARTIFACT_CONTEXT_ENABLED,
    )
    active_artifact_context_max_handles_raw = kwargs.pop(
        "active_artifact_context_max_handles",
        DEFAULT_ACTIVE_ARTIFACT_CONTEXT_MAX_HANDLES,
    )
    try:
        active_artifact_context_max_handles = int(active_artifact_context_max_handles_raw)
    except Exception:
        active_artifact_context_max_handles = DEFAULT_ACTIVE_ARTIFACT_CONTEXT_MAX_HANDLES
    active_artifact_context_max_handles = max(1, active_artifact_context_max_handles)

    active_artifact_context_max_chars_raw = kwargs.pop(
        "active_artifact_context_max_chars",
        DEFAULT_ACTIVE_ARTIFACT_CONTEXT_MAX_CHARS,
    )
    try:
        active_artifact_context_max_chars = int(active_artifact_context_max_chars_raw)
    except Exception:
        active_artifact_context_max_chars = DEFAULT_ACTIVE_ARTIFACT_CONTEXT_MAX_CHARS
    active_artifact_context_max_chars = max(160, active_artifact_context_max_chars)
    run_config_spec["active_artifact_context_enabled"] = active_artifact_context_enabled
    run_config_spec["active_artifact_context_max_handles"] = active_artifact_context_max_handles
    run_config_spec["active_artifact_context_max_chars"] = active_artifact_context_max_chars

    adoption_profile = normalize_adoption_profile(
        kwargs.pop("adoption_profile", os.getenv(ADOPTION_PROFILE_ENV, DEFAULT_ADOPTION_PROFILE))
    )
    adoption_profile_enforce = _coerce_bool(
        kwargs.pop("adoption_profile_enforce", os.getenv(ADOPTION_PROFILE_ENFORCE_ENV, "")),
        False,
    )
    run_config_spec["adoption_profile"] = adoption_profile
    run_config_spec["adoption_profile_enforce"] = adoption_profile_enforce
    run_config_hash = sha256_json(run_config_spec).replace("sha256:", "")

    return AgentLoopRuntimePolicy(
        forced_final_max_attempts=forced_final_max_attempts,
        forced_final_circuit_breaker_threshold=forced_final_circuit_breaker_threshold,
        forced_final_breaker_effective=forced_final_breaker_effective,
        force_submit_retry_on_max_tool_calls=force_submit_retry_on_max_tool_calls,
        accept_forced_answer_on_max_tool_calls=accept_forced_answer_on_max_tool_calls,
        finalization_fallback_models=finalization_fallback_models,
        retrieval_stagnation_turns=retrieval_stagnation_turns,
        retrieval_stagnation_action=retrieval_stagnation_action,
        tool_result_keep_recent=tool_result_keep_recent,
        tool_result_context_preview_chars=tool_result_context_preview_chars,
        active_artifact_context_enabled=active_artifact_context_enabled,
        active_artifact_context_max_handles=active_artifact_context_max_handles,
        active_artifact_context_max_chars=active_artifact_context_max_chars,
        adoption_profile=adoption_profile,
        adoption_profile_enforce=adoption_profile_enforce,
        run_config_spec=run_config_spec,
        run_config_hash=run_config_hash,
    )


def _initialize_agent_tool_state(
    *,
    openai_tools: list[dict[str, Any]],
    tool_contracts: dict[str, dict[str, Any]] | None,
    initial_artifacts: list[str] | tuple[str, ...] | None,
    initial_bindings: dict[str, Any] | None,
    kwargs: dict[str, Any],
    enforce_tool_contracts: bool,
    warning_sink: list[str],
) -> AgentLoopToolState:
    """Build initial tool/contract capabilities and lane-closure state."""
    normalized_tool_contracts = _shared_normalize_tool_contracts(tool_contracts)
    tool_parameter_index = build_tool_parameter_index(openai_tools)
    if initial_artifacts is None:
        initial_artifacts = DEFAULT_INITIAL_ARTIFACTS
    available_artifacts = {
        k for k in (_agent_normalize_artifact_kind(v) for v in initial_artifacts) if k
    }
    if not available_artifacts:
        available_artifacts = set(DEFAULT_INITIAL_ARTIFACTS)
    initial_artifact_snapshot = sorted(available_artifacts)
    available_capabilities: dict[str, set[tuple[str | None, str | None, str | None]]] = {}
    for artifact in initial_artifact_snapshot:
        _agent_capability_state_add(available_capabilities, kind=artifact)
    initial_capabilities_raw = kwargs.pop("initial_capabilities", None)
    if isinstance(initial_capabilities_raw, list):
        for item in initial_capabilities_raw:
            req = _agent_capability_requirement_from_raw(item)
            if req is None:
                continue
            _agent_capability_state_add(
                available_capabilities,
                kind=req.kind,
                ref_type=req.ref_type,
                namespace=req.namespace,
                bindings_hash=req.bindings_hash,
            )
    initial_capability_snapshot = _agent_capability_state_snapshot(available_capabilities)
    available_bindings = normalize_bindings(initial_bindings)
    initial_binding_snapshot = dict(available_bindings)
    lane_closure_analysis = _agent_analyze_lane_closure(
        normalized_tool_contracts=normalized_tool_contracts,
        initial_artifacts=set(available_artifacts),
        initial_capabilities=available_capabilities,
        available_bindings=available_bindings,
        event_code_missing_prerequisite=EVENT_CODE_TOOL_VALIDATION_MISSING_PREREQUISITE,
        event_code_missing_capability=EVENT_CODE_TOOL_VALIDATION_MISSING_CAPABILITY,
        event_code_binding_conflict=EVENT_CODE_TOOL_VALIDATION_BINDING_CONFLICT,
    )
    if (
        enforce_tool_contracts
        and normalized_tool_contracts
        and not bool(lane_closure_analysis.get("lane_closed"))
    ):
        unresolved_count = int(lane_closure_analysis.get("unresolved_tool_count") or 0)
        warning = (
            "CAPABILITY_CLOSURE_ADVISORY: lane has "
            f"{unresolved_count} unresolved non-control tool(s) from initial state. "
            "Ensure conversion operators are available for required capability transitions."
        )
        warning_sink.append(warning)
        logger.warning(warning)
    artifact_timeline: list[dict[str, Any]] = [
        {
            "turn": 0,
            "phase": "initial",
            "available_artifacts": list(initial_artifact_snapshot),
            "available_capabilities": list(initial_capability_snapshot),
        }
    ]
    if enforce_tool_contracts and not normalized_tool_contracts:
        warning = (
            "TOOL_CONTRACTS: enforce_tool_contracts=True but no contracts were provided; "
            "composability validation is skipped."
        )
        warning_sink.append(warning)
        logger.warning(warning)
    requires_submit_answer = any(
        t.get("function", {}).get("name") == "submit_answer"
        for t in openai_tools
        if isinstance(t, dict)
    )
    return AgentLoopToolState(
        normalized_tool_contracts=normalized_tool_contracts,
        tool_parameter_index=tool_parameter_index,
        available_artifacts=available_artifacts,
        initial_artifact_snapshot=initial_artifact_snapshot,
        available_capabilities=available_capabilities,
        initial_capability_snapshot=initial_capability_snapshot,
        available_bindings=available_bindings,
        initial_binding_snapshot=initial_binding_snapshot,
        lane_closure_analysis=lane_closure_analysis,
        artifact_timeline=artifact_timeline,
        requires_submit_answer=requires_submit_answer,
    )
