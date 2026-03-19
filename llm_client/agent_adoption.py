"""Adoption-profile assessment helpers for agent runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_ADOPTION_PROFILE: str = "minimal"
"""Default agent-runtime adoption profile for workspace tool loops."""

ADOPTION_PROFILE_VALUES: frozenset[str] = frozenset({"minimal", "standard", "strict"})
"""Supported adoption-profile names."""


@dataclass(frozen=True)
class AdoptionProfileAssessment:
    """Normalized adoption-profile assessment for one agent run."""

    requested_profile: str
    effective_profile: str
    enforce: bool
    violations: list[str]

    @property
    def satisfied(self) -> bool:
        return not self.violations


def normalize_adoption_profile(value: Any) -> str:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ADOPTION_PROFILE_VALUES:
            return lowered
    return DEFAULT_ADOPTION_PROFILE


def tool_schema_is_nontrivial(
    tool_schema: dict[str, Any],
    *,
    tool_reasoning_field: str,
) -> bool:
    fn = tool_schema.get("function") or {}
    parameters = fn.get("parameters") or {}
    properties = parameters.get("properties") or {}
    if not isinstance(properties, dict):
        return False
    user_properties = {
        key: value for key, value in properties.items()
        if key != tool_reasoning_field
    }
    if len(user_properties) > 1:
        return True
    for schema in user_properties.values():
        if not isinstance(schema, dict):
            continue
        if schema.get("type") in {"array", "object"}:
            return True
        if schema.get("anyOf"):
            return True
    return False


def assess_tool_registry_quality(
    *,
    openai_tools: list[dict[str, Any]],
    normalized_tool_contracts: dict[str, dict[str, Any]],
    tool_reasoning_field: str,
) -> list[str]:
    violations: list[str] = []
    for tool in openai_tools:
        fn = tool.get("function") or {}
        if not isinstance(fn, dict):
            continue
        tool_name = str(fn.get("name") or "").strip()
        if not tool_name:
            continue
        description = str(fn.get("description") or "").strip()
        if not description:
            violations.append(f"tool {tool_name} must have a one-line description")
        if not tool_schema_is_nontrivial(tool, tool_reasoning_field=tool_reasoning_field):
            continue
        if "input examples:" not in description.lower():
            violations.append(f"nontrivial tool {tool_name} must include input examples")
        if tool_name not in normalized_tool_contracts:
            violations.append(f"nontrivial tool {tool_name} must declare tool_contracts")
    return violations


def assess_adoption_profile(
    *,
    requested_profile: str,
    enforce: bool,
    openai_tools: list[dict[str, Any]],
    normalized_tool_contracts: dict[str, dict[str, Any]],
    require_tool_reasoning: bool,
    enforce_tool_contracts: bool,
    progressive_tool_disclosure: bool,
    lane_closure_analysis: dict[str, Any],
    tool_reasoning_field: str,
) -> AdoptionProfileAssessment:
    """Evaluate whether the current run satisfies the requested adoption profile."""
    effective_profile = normalize_adoption_profile(requested_profile)
    if effective_profile == "minimal":
        return AdoptionProfileAssessment(
            requested_profile=requested_profile,
            effective_profile=effective_profile,
            enforce=enforce,
            violations=[],
        )

    has_tools = bool(openai_tools)
    violations: list[str] = []
    if has_tools and not require_tool_reasoning:
        violations.append("require_tool_reasoning must be enabled")
    if has_tools and not enforce_tool_contracts:
        violations.append("enforce_tool_contracts must be enabled")
    if has_tools and not normalized_tool_contracts:
        violations.append("tool_contracts must be provided")
    if has_tools:
        violations.extend(
            assess_tool_registry_quality(
                openai_tools=openai_tools,
                normalized_tool_contracts=normalized_tool_contracts,
                tool_reasoning_field=tool_reasoning_field,
            )
        )

    if effective_profile == "strict":
        if has_tools and not progressive_tool_disclosure:
            violations.append("progressive_tool_disclosure must be enabled")
        if (
            has_tools
            and normalized_tool_contracts
            and not bool(lane_closure_analysis.get("lane_closed"))
        ):
            violations.append("lane_closure_analysis must be closed for strict profile")

    return AdoptionProfileAssessment(
        requested_profile=requested_profile,
        effective_profile=effective_profile,
        enforce=enforce,
        violations=violations,
    )
