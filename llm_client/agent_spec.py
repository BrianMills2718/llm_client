"""AgentSpec loading, validation, and summarization helpers.

AgentSpec is a project-level declaration that makes benchmark/eval runs
explicitly reproducible and observable. The schema is intentionally strict:
projects must provide required sections up front or explicitly opt out.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

REQUIRED_AGENT_SPEC_SECTIONS: tuple[str, ...] = (
    "prompts",
    "tools",
    "artifact_contracts",
    "answer_schema",
    "error_taxonomy",
    "observability",
    "evaluation",
    "gates",
)


class AgentSpecValidationError(ValueError):
    """Raised when an AgentSpec fails validation."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        joined = "; ".join(errors) if errors else "unknown validation error"
        super().__init__(f"AgentSpec validation failed: {joined}")


def _stable_hash(payload: Any) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_from_path(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"AgentSpec file not found: {path}")
    text = path.read_text()
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise ValueError(
                "YAML AgentSpec requires pyyaml. Install with `pip install pyyaml` "
                f"or use JSON. File: {path}"
            ) from exc
        data = yaml.safe_load(text)
    else:
        raise ValueError(
            f"Unsupported AgentSpec extension {suffix!r} for {path}. "
            "Use .json, .yaml, or .yml."
        )
    if not isinstance(data, dict):
        raise ValueError(f"AgentSpec root must be a mapping. Got: {type(data).__name__}")
    return data


def _tool_names_from_spec(tools: Any) -> set[str]:
    names: set[str] = set()
    if isinstance(tools, dict):
        for key in tools:
            if isinstance(key, str) and key.strip():
                names.add(key.strip())
        return names
    if isinstance(tools, list):
        for item in tools:
            if isinstance(item, str) and item.strip():
                names.add(item.strip())
                continue
            if isinstance(item, Mapping):
                name = item.get("name")
                if isinstance(name, str) and name.strip():
                    names.add(name.strip())
        return names
    return names


def _count_prompts(prompts: Any) -> int:
    if isinstance(prompts, dict):
        return len(prompts)
    if isinstance(prompts, list):
        return len(prompts)
    return 0


def _count_gates(gates: Any) -> int:
    if isinstance(gates, dict):
        if "rules" in gates and isinstance(gates["rules"], list):
            return len(gates["rules"])
        return len(gates)
    if isinstance(gates, list):
        return len(gates)
    return 0


def validate_agent_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate AgentSpec and return a normalized copy.

    Raises:
        AgentSpecValidationError: If validation fails.
    """
    errors: list[str] = []
    normalized = dict(spec)

    missing = [sec for sec in REQUIRED_AGENT_SPEC_SECTIONS if sec not in normalized]
    if missing:
        errors.append(f"missing required sections: {missing}")

    prompts = normalized.get("prompts")
    if not isinstance(prompts, (dict, list)) or _count_prompts(prompts) == 0:
        errors.append("prompts must be a non-empty mapping or list")

    tools = normalized.get("tools")
    tool_names = _tool_names_from_spec(tools)
    if not tool_names:
        errors.append("tools must declare at least one tool name")

    contracts = normalized.get("artifact_contracts")
    if not isinstance(contracts, Mapping) or len(contracts) == 0:
        errors.append("artifact_contracts must be a non-empty mapping")
    else:
        unknown_contract_tools = sorted(
            str(name)
            for name in contracts.keys()
            if isinstance(name, str) and name not in tool_names
        )
        if unknown_contract_tools:
            errors.append(
                "artifact_contracts references unknown tools: "
                + ", ".join(unknown_contract_tools)
            )

    answer_schema = normalized.get("answer_schema")
    if not isinstance(answer_schema, Mapping) or len(answer_schema) == 0:
        errors.append("answer_schema must be a non-empty mapping")

    error_taxonomy = normalized.get("error_taxonomy")
    if not isinstance(error_taxonomy, (Mapping, list)) or len(error_taxonomy) == 0:
        errors.append("error_taxonomy must be a non-empty mapping or list")

    observability = normalized.get("observability")
    if not isinstance(observability, Mapping) or len(observability) == 0:
        errors.append("observability must be a non-empty mapping")
    else:
        required_fields = observability.get("required_fields")
        if not isinstance(required_fields, list) or not required_fields:
            errors.append("observability.required_fields must be a non-empty list")

    evaluation = normalized.get("evaluation")
    if not isinstance(evaluation, Mapping) or len(evaluation) == 0:
        errors.append("evaluation must be a non-empty mapping")
    else:
        metrics = evaluation.get("metrics")
        if not isinstance(metrics, list) or not metrics:
            errors.append("evaluation.metrics must be a non-empty list")

    gates = normalized.get("gates")
    if not isinstance(gates, (Mapping, list)) or _count_gates(gates) == 0:
        errors.append("gates must be a non-empty mapping or list")

    if errors:
        raise AgentSpecValidationError(errors)
    return normalized


def load_agent_spec(spec: str | Path | Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load + validate AgentSpec and return (spec, summary)."""
    source: str
    if isinstance(spec, Mapping):
        raw = dict(spec)
        source = "<in-memory>"
    else:
        path = Path(spec).expanduser()
        raw = _load_from_path(path)
        source = str(path.resolve())

    validated = validate_agent_spec(raw)
    tool_names = sorted(_tool_names_from_spec(validated.get("tools")))
    contracts = validated.get("artifact_contracts")
    contract_count = len(contracts) if isinstance(contracts, Mapping) else 0
    summary = {
        "source": source,
        "sha256": _stable_hash(validated),
        "required_sections": list(REQUIRED_AGENT_SPEC_SECTIONS),
        "tool_count": len(tool_names),
        "tool_names": tool_names,
        "contract_count": contract_count,
        "prompt_count": _count_prompts(validated.get("prompts")),
        "gate_count": _count_gates(validated.get("gates")),
        "evaluation_metrics": list(validated.get("evaluation", {}).get("metrics", []))
        if isinstance(validated.get("evaluation"), Mapping)
        else [],
        "observability_required_fields": list(
            validated.get("observability", {}).get("required_fields", [])
        )
        if isinstance(validated.get("observability"), Mapping)
        else [],
    }
    return validated, summary
