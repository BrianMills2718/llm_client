#!/usr/bin/env python3
"""Dead code sensor for enforced-planning framework.

Wraps vulture (Python) and knip (TypeScript) to detect unused code.
Reads config from meta-process.yaml quality.dead_code section.
Outputs JSON findings for consumption by ecosystem_sweep and task_planner.

Portable: no import dependencies beyond stdlib. Designed to be copied
into target projects via install.sh.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

RETAINED_FINDING_DISPOSITIONS = (
    "keep_reexport",
    "keep_false_positive",
    "keep_planned_feature",
    "framework_sync",
)


@dataclass
class Finding:
    """A single dead code finding from vulture or knip."""

    file: str
    line: int
    name: str
    kind: str
    confidence: int | None


@dataclass
class Result:
    """Aggregated result of dead code detection."""

    passed: bool
    findings: list[Finding] = field(default_factory=list)
    reviewed_findings: list[Finding] = field(default_factory=list)
    actionable_findings: list[Finding] = field(default_factory=list)
    tool_output: str = ""
    tool_available: bool = True
    error: str = ""
    exit_code: int | None = None
    detector: str | None = None
    audit_file: str | None = None


def _repo_python(project_root: Path) -> str:
    """Return the repo-local Python interpreter when one is available.

    Dead-code hooks may run under a generic interpreter that lacks repo-local
    tooling. Governed repos standardize on a per-repo `.venv`, so use that
    interpreter for Python-based checks when present.
    """

    candidates = [
        project_root / ".venv" / "bin" / "python",
        project_root / ".venv" / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def _load_config(project_root: Path) -> dict[str, Any]:
    """Load dead_code config from meta-process.yaml.

    Returns the quality.dead_code section, or sensible defaults
    if the file or section doesn't exist.
    """
    config_path = project_root / "meta-process.yaml"
    defaults: dict[str, Any] = {
        "enabled": False,
        "strict": False,
        "min_confidence": 80,
        "paths": [],
        "whitelist": ".vulture_whitelist.py",
        "audit_file": "dead_code_audit.json",
    }
    if not config_path.is_file():
        return defaults

    raw_text = config_path.read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        dead_code = _load_dead_code_config_without_yaml(raw_text)
        for key, default in defaults.items():
            dead_code.setdefault(key, default)
        return dead_code

    raw = yaml.safe_load(raw_text)
    if not isinstance(raw, dict):
        return defaults

    mp = raw.get("meta_process") or raw.get("meta-process", {})
    quality = mp.get("quality", {}) if isinstance(mp, dict) else {}
    dead_code = quality.get("dead_code", {}) if isinstance(quality, dict) else {}

    for key, default in defaults.items():
        dead_code.setdefault(key, default)
    return dead_code


def _coerce_yaml_scalar(raw_value: str) -> Any:
    """Convert a small YAML scalar into a Python value."""

    value = raw_value.strip()
    if not value:
        return ""
    if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
        return value[1:-1]
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value == "[]":
        return []
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    return value


def _load_dead_code_config_without_yaml(raw_text: str) -> dict[str, Any]:
    """Parse only the quality.dead_code block when PyYAML is unavailable."""

    dead_code: dict[str, Any] = {}
    meta_indent: int | None = None
    quality_indent: int | None = None
    dead_code_indent: int | None = None
    list_key: str | None = None

    for line in raw_text.splitlines():
        content = line.split("#", 1)[0].rstrip()
        if not content.strip():
            continue
        indent = len(content) - len(content.lstrip(" "))
        stripped = content.strip()

        if stripped in {"meta_process:", "meta-process:"}:
            meta_indent = indent
            quality_indent = None
            dead_code_indent = None
            list_key = None
            continue

        if meta_indent is None or indent <= meta_indent:
            continue

        if stripped == "quality:":
            quality_indent = indent
            dead_code_indent = None
            list_key = None
            continue

        if quality_indent is None or indent <= quality_indent:
            continue

        if stripped == "dead_code:":
            dead_code_indent = indent
            list_key = None
            continue

        if dead_code_indent is None:
            continue
        if indent <= dead_code_indent:
            break

        if stripped.startswith("- ") and list_key is not None:
            existing = dead_code.setdefault(list_key, [])
            if isinstance(existing, list):
                existing.append(_coerce_yaml_scalar(stripped[2:]))
            continue

        match = re.match(r"(?P<key>[A-Za-z0-9_]+):\s*(?P<value>.*)$", stripped)
        if not match:
            continue

        key = match.group("key")
        raw_value = match.group("value")
        if raw_value == "":
            dead_code[key] = []
            list_key = key
            continue

        dead_code[key] = _coerce_yaml_scalar(raw_value)
        list_key = None

    return dead_code


def _parse_vulture_line(line: str) -> Finding | None:
    """Parse a vulture output line into a Finding.

    Format: path.py:123: unused function 'foo' (90% confidence)
    """
    match = re.match(
        r"(.+?):(\d+): (unused [^']+) '([^']+)' \((\d+)% confidence\)", line
    )
    if match:
        return Finding(
            file=match.group(1),
            line=int(match.group(2)),
            kind=match.group(3).replace(" ", "-"),
            name=match.group(4),
            confidence=int(match.group(5)),
        )
    return None


def finding_signature(finding: Finding, detector: str | None) -> str:
    """Return the stable review signature for a finding."""

    return "::".join(
        [
            detector or "unknown",
            finding.file,
            finding.kind,
            finding.name,
        ]
    )


def _current_timestamp() -> str:
    """Return a stable UTC timestamp string."""

    return datetime.now(timezone.utc).isoformat()


def _load_audit_payload(project_root: Path, audit_file: str) -> dict[str, Any]:
    """Load the reviewed dead-code audit file when present."""

    path = project_root / audit_file
    if not path.is_file():
        return {"version": 1, "generated_at": None, "findings": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{audit_file} must contain a JSON object")
    findings = payload.get("findings")
    if findings is None:
        payload["findings"] = []
    elif not isinstance(findings, list):
        raise ValueError(f"{audit_file} field 'findings' must be a list")
    return payload


def _retained_audit_entries(
    project_root: Path, audit_file: str
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Return retained audit entries keyed by signature plus the raw list."""

    payload = _load_audit_payload(project_root, audit_file)
    findings = payload.get("findings", [])
    retained: dict[str, dict[str, Any]] = {}
    if not isinstance(findings, list):
        raise ValueError(f"{audit_file} field 'findings' must be a list")
    for index, entry in enumerate(findings, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"{audit_file} finding #{index} must be an object")
        disposition = entry.get("disposition")
        if disposition not in RETAINED_FINDING_DISPOSITIONS:
            continue
        detector = entry.get("detector")
        file_path = entry.get("file")
        kind = entry.get("kind")
        name = entry.get("name")
        if not all(isinstance(value, str) and value for value in (detector, file_path, kind, name)):
            raise ValueError(
                f"{audit_file} retained finding #{index} must include non-empty detector/file/kind/name"
            )
        signature = "::".join([detector, file_path, kind, name])
        retained[signature] = entry
    return retained, findings


def _parse_knip_item(issue_type: str, file_path: str, item: Any) -> Finding | None:
    """Convert one JSON knip issue item into a normalized finding."""

    kind = issue_type.strip().replace("_", "-").replace(" ", "-")
    if isinstance(item, str):
        return Finding(
            file=file_path,
            line=1,
            name=item,
            kind=kind,
            confidence=None,
        )
    if not isinstance(item, dict):
        return None

    name = next(
        (
            value
            for key in ("name", "symbol", "issue", "identifier", "text", "path")
            if isinstance((value := item.get(key)), str) and value
        ),
        Path(file_path).name if file_path else "<unknown>",
    )
    line = item.get("line")
    if not isinstance(line, int) or line < 1:
        line = 1
    return Finding(
        file=file_path or "<unknown>",
        line=line,
        name=name,
        kind=kind,
        confidence=None,
    )


def _parse_knip_report(output: str) -> list[Finding]:
    """Parse the JSON reporter output from knip."""

    raw = output.strip()
    if not raw:
        return []

    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("knip JSON reporter returned a non-object payload")

    findings: list[Finding] = []

    files = payload.get("files")
    if isinstance(files, list):
        for entry in files:
            finding = _parse_knip_item("unused-file", "<unknown>", entry)
            if finding is not None:
                if isinstance(entry, dict):
                    path = entry.get("file") or entry.get("path") or entry.get("name")
                    if isinstance(path, str) and path:
                        finding.file = path
                        finding.name = Path(path).name
                elif isinstance(entry, str):
                    finding.file = entry
                    finding.name = Path(entry).name
                findings.append(finding)

    issues = payload.get("issues")
    if isinstance(issues, list):
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            file_path = issue.get("file") if isinstance(issue.get("file"), str) else "<unknown>"
            for issue_type, items in issue.items():
                if issue_type == "file" or not isinstance(items, list):
                    continue
                for item in items:
                    finding = _parse_knip_item(issue_type, file_path, item)
                    if finding is not None:
                        findings.append(finding)

    return findings


def _run_vulture(
    project_root: Path,
    paths: list[str],
    min_confidence: int,
    whitelist: str,
) -> Result:
    """Run vulture for Python dead code detection."""
    cmd = [_repo_python(project_root), "-m", "vulture"]
    if paths:
        cmd.extend(paths)
    else:
        cmd.append(".")
    cmd.extend(["--min-confidence", str(min_confidence)])

    whitelist_path = project_root / whitelist
    if whitelist_path.is_file():
        cmd.append(str(whitelist_path))

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return Result(
            passed=False,
            tool_available=True,
            error="vulture timed out after 120s",
        )

    findings: list[Finding] = []
    for line in proc.stdout.strip().splitlines():
        f = _parse_vulture_line(line)
        if f:
            findings.append(f)

    tool_output = "\n".join(
        part for part in (proc.stdout.strip(), proc.stderr.strip()) if part
    )
    if findings:
        return Result(
            passed=True,
            findings=findings,
            actionable_findings=list(findings),
            tool_output=tool_output,
            exit_code=proc.returncode,
            detector="vulture",
        )

    if "No module named vulture" in proc.stderr:
        return Result(
            passed=False,
            tool_available=False,
            error="vulture is required when dead_code is enabled",
            tool_output=tool_output,
            exit_code=proc.returncode,
            detector="vulture",
        )

    if proc.returncode != 0:
        return Result(
            passed=False,
            error=f"vulture failed with exit code {proc.returncode}",
            tool_output=tool_output,
            exit_code=proc.returncode,
            detector="vulture",
        )

    return Result(
        passed=True,
        findings=[],
        actionable_findings=[],
        tool_output=tool_output,
        exit_code=proc.returncode,
        detector="vulture",
    )


def _run_knip(project_root: Path) -> Result:
    """Run knip for TypeScript dead code detection."""
    try:
        proc = subprocess.run(
            ["npx", "knip", "--reporter", "json"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except FileNotFoundError:
        return Result(
            passed=False,
            tool_available=False,
            error="npx/knip is required when dead_code is enabled",
            detector="knip",
        )
    except subprocess.TimeoutExpired:
        return Result(
            passed=False,
            tool_available=True,
            error="knip timed out after 120s",
            detector="knip",
        )

    tool_output = "\n".join(
        part for part in (proc.stdout.strip(), proc.stderr.strip()) if part
    )
    try:
        findings = _parse_knip_report(proc.stdout)
    except json.JSONDecodeError as exc:
        return Result(
            passed=False,
            error=f"knip did not return valid JSON: {exc}",
            tool_output=tool_output,
            exit_code=proc.returncode,
            detector="knip",
        )
    except ValueError as exc:
        return Result(
            passed=False,
            error=str(exc),
            tool_output=tool_output,
            exit_code=proc.returncode,
            detector="knip",
        )

    if findings:
        return Result(
            passed=True,
            findings=findings,
            actionable_findings=list(findings),
            tool_output=tool_output,
            exit_code=proc.returncode,
            detector="knip",
        )

    if proc.returncode != 0:
        return Result(
            passed=False,
            error=f"knip failed with exit code {proc.returncode}",
            tool_output=tool_output,
            exit_code=proc.returncode,
            detector="knip",
        )

    return Result(
        passed=True,
        findings=[],
        actionable_findings=[],
        tool_output=tool_output,
        exit_code=proc.returncode,
        detector="knip",
    )


def scan_dead_code(project_root: Path) -> Result:
    """Run the raw dead-code detector without applying review policy.

    Reads config from meta-process.yaml. Auto-detects language from
    project contents (package.json → TypeScript, *.py → Python).
    """
    config = _load_config(project_root)
    if not config.get("enabled"):
        return Result(passed=True)

    # Detect language
    if (project_root / "package.json").is_file():
        result = _run_knip(project_root)
    else:
        result = _run_vulture(
            project_root,
            paths=config.get("paths", []),
            min_confidence=config.get("min_confidence", 80),
            whitelist=config.get("whitelist", ".vulture_whitelist.py"),
        )

    result.audit_file = str(config.get("audit_file", "dead_code_audit.json"))
    if result.findings and not result.actionable_findings:
        result.actionable_findings = list(result.findings)
    return result


def check_dead_code(project_root: Path) -> Result:
    """Run dead code detection and apply reviewed-retention policy."""

    config = _load_config(project_root)
    if not config.get("enabled"):
        return Result(passed=True)

    result = scan_dead_code(project_root)
    if result.error:
        result.passed = False
        return result

    audit_file = str(config.get("audit_file", "dead_code_audit.json"))
    result.audit_file = audit_file
    try:
        retained_entries, _ = _retained_audit_entries(project_root, audit_file)
    except (ValueError, json.JSONDecodeError) as exc:
        result.passed = False
        result.error = str(exc)
        return result
    reviewed: list[Finding] = []
    actionable: list[Finding] = []
    for finding in result.findings:
        signature = finding_signature(finding, result.detector)
        if signature in retained_entries:
            reviewed.append(finding)
        else:
            actionable.append(finding)
    result.reviewed_findings = reviewed
    result.actionable_findings = actionable

    if config.get("strict") and actionable:
        result.passed = False

    return result


def main() -> None:
    """CLI entry point. Outputs JSON to stdout."""
    project_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    project_root = project_root.resolve()

    config = _load_config(project_root)
    if not config.get("enabled"):
        json.dump({"skipped": True, "reason": "dead_code not enabled in meta-process.yaml"}, sys.stdout)
        print()
        return

    result = check_dead_code(project_root)
    output = {
        "passed": result.passed,
        "detector": result.detector,
        "findings_count": len(result.findings),
        "findings": [asdict(f) for f in result.findings],
        "reviewed_findings_count": len(result.reviewed_findings),
        "reviewed_findings": [asdict(f) for f in result.reviewed_findings],
        "actionable_findings_count": len(result.actionable_findings),
        "actionable_findings": [asdict(f) for f in result.actionable_findings],
        "audit_file": result.audit_file,
        "tool_available": result.tool_available,
        "error": result.error,
        "exit_code": result.exit_code,
    }
    json.dump(output, sys.stdout, indent=2)
    print()

    if not result.passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
