"""Tool-disclosure helpers for agent runtimes."""

from __future__ import annotations

from typing import Any, Callable


def _filter_tools_for_disclosure(
    *,
    openai_tools: list[dict[str, Any]],
    normalized_tool_contracts: dict[str, dict[str, Any]],
    available_artifacts: set[str],
    available_capabilities: dict[str, set[tuple[str | None, str | None, str | None]]],
    available_bindings: dict[str, str | None] | None = None,
    max_unavailable: int,
    max_missing_per_tool: int,
    max_repair_tools: int,
    tool_declares_no_artifact_prereqs: Callable[[str, dict[str, Any] | None], bool],
    validate_tool_contract_call: Callable[..., Any],
    find_repair_tools_for_missing_requirements: Callable[..., list[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Return tools currently composable from artifact state + hidden reasons."""
    if not openai_tools:
        return [], [], 0
    if not normalized_tool_contracts:
        return list(openai_tools), [], 0

    visible: list[dict[str, Any]] = []
    hidden_all: list[dict[str, Any]] = []
    for tool_def in openai_tools:
        fn = tool_def.get("function", {}) if isinstance(tool_def, dict) else {}
        tool_name = str(fn.get("name", "")).strip()
        if not tool_name:
            visible.append(tool_def)
            continue

        contract = normalized_tool_contracts.get(tool_name)
        if not isinstance(contract, dict):
            visible.append(tool_def)
            continue

        if tool_declares_no_artifact_prereqs(tool_name, contract):
            visible.append(tool_def)
            continue

        validation = validate_tool_contract_call(
            tool_name=tool_name,
            contract=contract,
            parsed_args=None,
            available_artifacts=available_artifacts,
            available_capabilities=available_capabilities,
            available_bindings=available_bindings,
        )
        if validation.is_valid:
            visible.append(tool_def)
        else:
            hidden_all.append({
                "tool": tool_name,
                "reason": validation.reason,
                "missing_requirements": list(validation.missing_requirements or []),
                "missing_count": len(validation.missing_requirements or []),
            })

    hidden_all.sort(key=lambda h: (int(h.get("missing_count", 0) or 0), str(h.get("tool", ""))))
    hidden_total = len(hidden_all)

    if max_missing_per_tool >= 0:
        for item in hidden_all:
            missing = item.get("missing_requirements") or []
            if isinstance(missing, list):
                item["missing_requirements"] = missing[:max_missing_per_tool]
            item["missing_count"] = len(item.get("missing_requirements") or [])
            item["repair_tools"] = find_repair_tools_for_missing_requirements(
                current_tool_name=str(item.get("tool", "")).strip(),
                missing_requirements=list(item.get("missing_requirements") or []),
                normalized_tool_contracts=normalized_tool_contracts,
                available_artifacts=available_artifacts,
                available_capabilities=available_capabilities,
                available_bindings=available_bindings,
                max_repair_tools=max_repair_tools,
            )

    if max_unavailable >= 0:
        hidden_all = hidden_all[:max_unavailable]

    return visible, hidden_all, hidden_total


def _disclosure_reason_from_entry(
    entry: dict[str, Any],
    *,
    capability_requirement_from_raw: Callable[[Any], Any | None],
    short_requirement: Callable[[Any], str],
) -> str:
    repair_tools = entry.get("repair_tools")
    repair_hint = ""
    if isinstance(repair_tools, list):
        names = [str(n).strip() for n in repair_tools if isinstance(n, str) and n.strip()]
        if names:
            repair_hint = f"; try {', '.join(names)}"

    missing = entry.get("missing_requirements")
    if isinstance(missing, list) and missing:
        labels: list[str] = []
        for item in missing:
            req = capability_requirement_from_raw(item)
            if req is None:
                continue
            labels.append(short_requirement(req))
        if labels:
            return "needs " + ", ".join(labels) + repair_hint
    reason = str(entry.get("reason", "")).strip()
    if reason:
        return reason + repair_hint
    return "missing prerequisites" + repair_hint


def _disclosure_message(
    hidden_entries: list[dict[str, Any]],
    *,
    disclosure_reason_from_entry: Callable[[dict[str, Any]], str],
    trim_text: Callable[[str, int], str],
    max_reason_chars: int,
) -> str:
    parts: list[str] = []
    for entry in hidden_entries:
        tool = str(entry.get("tool", "")).strip() or "<unknown>"
        reason = trim_text(disclosure_reason_from_entry(entry), max_reason_chars)
        parts.append(f"{tool}: {reason}")
    return "; ".join(parts)


def _deficit_labels_from_hidden_entries(
    hidden_entries: list[dict[str, Any]],
    *,
    capability_requirement_from_raw: Callable[[Any], Any | None],
    short_requirement: Callable[[Any], str],
) -> list[str]:
    """Canonical missing-artifact/capability labels from hidden tool requirements."""
    labels: set[str] = set()
    for entry in hidden_entries:
        if not isinstance(entry, dict):
            continue
        missing = entry.get("missing_requirements")
        if not isinstance(missing, list):
            continue
        for raw in missing:
            req = capability_requirement_from_raw(raw)
            if req is None:
                continue
            labels.add(short_requirement(req))
    return sorted(labels)
