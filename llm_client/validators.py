"""Task graph validation framework.

External checks that verify task outputs without trusting agent self-reports.
Validators return structured results with evidence, not just pass/fail.

Usage:
    from llm_client.validators import run_validators, ValidationResult

    configs = [
        {"type": "file_exists", "path": "output.json"},
        {"type": "json_schema", "path": "output.json", "schema": {"type": "object"}},
    ]
    results = run_validators(configs)
    assert all(r.passed for r in results)

    # Dry-run mode (preview checks without executing)
    results = run_validators(configs, dry_run=True)

    # Custom validators
    from llm_client.validators import register_validator

    def check_belief_count(config: dict) -> ValidationResult:
        ...

    register_validator("belief_count", check_belief_count)
"""

from __future__ import annotations

import hashlib
import json
import operator
import re
import sqlite3
import subprocess
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel


class ValidationResult(BaseModel):
    """Result of a single validation check."""

    type: str
    passed: bool
    value: Any = None
    reason: str | None = None


# Type alias for validator callables
ValidatorFn = Callable[[dict[str, Any]], ValidationResult]

# Global registry of validator functions
_registry: dict[str, ValidatorFn] = {}


def register_validator(name: str, fn: ValidatorFn) -> None:
    """Register a custom validator function.

    Args:
        name: Validator type name (used in YAML configs).
        fn: Callable that takes a config dict and returns ValidationResult.
    """
    _registry[name] = fn


def _expand_path(p: str) -> Path:
    """Expand ~ and resolve path."""
    return Path(p).expanduser().resolve()


# --- Comparison expression parser ---

_CMP_OPS: dict[str, Callable[[Any, Any], bool]] = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}

_CMP_RE = re.compile(r"^\s*(>=|<=|!=|==|>|<)\s*(.+)\s*$")


def _eval_check(value: Any, check_expr: str) -> bool:
    """Evaluate a simple comparison expression like '> 0' or '== 5'."""
    m = _CMP_RE.match(check_expr)
    if not m:
        raise ValueError(f"Invalid check expression: {check_expr!r}. Expected format: '> 0', '>= 5', '== 10', etc.")
    op_str, rhs_str = m.group(1), m.group(2).strip()
    rhs: Any
    try:
        rhs = int(rhs_str)
    except ValueError:
        try:
            rhs = float(rhs_str)
        except ValueError:
            rhs = rhs_str
    return _CMP_OPS[op_str](value, rhs)


# --- Built-in validators ---


def _validate_file_exists(config: dict[str, Any]) -> ValidationResult:
    path = _expand_path(config["path"])
    exists = path.exists()
    return ValidationResult(
        type="file_exists",
        passed=exists,
        value=str(path),
        reason=None if exists else f"File not found: {path}",
    )


def _validate_file_not_empty(config: dict[str, Any]) -> ValidationResult:
    path = _expand_path(config["path"])
    min_bytes = config.get("min_bytes", 1)
    if not path.exists():
        return ValidationResult(
            type="file_not_empty",
            passed=False,
            value=0,
            reason=f"File not found: {path}",
        )
    size = path.stat().st_size
    passed = size >= min_bytes
    return ValidationResult(
        type="file_not_empty",
        passed=passed,
        value=size,
        reason=None if passed else f"File too small: {size} bytes (min: {min_bytes})",
    )


def _validate_json_schema(config: dict[str, Any]) -> ValidationResult:
    path = _expand_path(config["path"])
    schema = config["schema"]
    if not path.exists():
        return ValidationResult(
            type="json_schema",
            passed=False,
            value=None,
            reason=f"File not found: {path}",
        )
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return ValidationResult(
            type="json_schema",
            passed=False,
            value=None,
            reason=f"JSON parse error: {e}",
        )
    try:
        import jsonschema
    except ImportError:
        raise ImportError(
            "jsonschema is required for json_schema validation. "
            "Install with: pip install jsonschema"
        )
    try:
        jsonschema.validate(data, schema)
        return ValidationResult(type="json_schema", passed=True, value=data)
    except jsonschema.ValidationError as e:
        return ValidationResult(
            type="json_schema",
            passed=False,
            value=None,
            reason=f"Schema validation failed: {e.message}",
        )


def _validate_command(config: dict[str, Any]) -> ValidationResult:
    command = config["command"]
    timeout = config.get("timeout", 60)
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        passed = proc.returncode == 0
        return ValidationResult(
            type="command",
            passed=passed,
            value=proc.returncode,
            reason=None if passed else f"Exit code {proc.returncode}: {proc.stderr[:500]}",
        )
    except subprocess.TimeoutExpired:
        return ValidationResult(
            type="command",
            passed=False,
            value=None,
            reason=f"Command timed out after {timeout}s",
        )


def _validate_pytest(config: dict[str, Any]) -> ValidationResult:
    path = config["path"]
    markers = config.get("markers")
    timeout = config.get("timeout", 300)
    cmd = ["python", "-m", "pytest", str(_expand_path(path)), "-x", "-q", "--tb=short"]
    if markers:
        cmd.extend(["-m", markers])
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        passed = proc.returncode == 0
        # Extract summary line (e.g., "5 passed in 1.23s")
        summary = ""
        for line in proc.stdout.splitlines():
            if "passed" in line or "failed" in line or "error" in line:
                summary = line.strip()
        return ValidationResult(
            type="pytest",
            passed=passed,
            value=summary or proc.stdout[-500:] if proc.stdout else "",
            reason=None if passed else f"pytest failed: {summary or proc.stderr[:500]}",
        )
    except subprocess.TimeoutExpired:
        return ValidationResult(
            type="pytest",
            passed=False,
            value=None,
            reason=f"pytest timed out after {timeout}s",
        )


def _validate_sql_count(config: dict[str, Any]) -> ValidationResult:
    db_path = _expand_path(config["db"])
    query = config["query"]
    check_expr = config["check"]
    if not db_path.exists():
        return ValidationResult(
            type="sql_count",
            passed=False,
            value=None,
            reason=f"Database not found: {db_path}",
        )
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            cursor = conn.execute(query)
            row = cursor.fetchone()
            count = row[0] if row else 0
        finally:
            conn.close()
        passed = _eval_check(count, check_expr)
        return ValidationResult(
            type="sql_count",
            passed=passed,
            value=count,
            reason=None if passed else f"Count {count} does not satisfy {check_expr!r}",
        )
    except sqlite3.Error as e:
        return ValidationResult(
            type="sql_count",
            passed=False,
            value=None,
            reason=f"SQL error: {e}",
        )


def _validate_mcp_call(config: dict[str, Any]) -> ValidationResult:
    # MCP validation requires the MCP infrastructure — defer to task_graph.py
    # which has access to running MCP sessions. This validator is a placeholder
    # that fails loudly if called directly.
    return ValidationResult(
        type="mcp_call",
        passed=False,
        value=None,
        reason="mcp_call validator must be run through the task graph runner (requires MCP session)",
    )


# --- Registry initialization ---

_BUILTINS: dict[str, ValidatorFn] = {
    "file_exists": _validate_file_exists,
    "file_not_empty": _validate_file_not_empty,
    "json_schema": _validate_json_schema,
    "command": _validate_command,
    "pytest": _validate_pytest,
    "sql_count": _validate_sql_count,
    "mcp_call": _validate_mcp_call,
}

_registry.update(_BUILTINS)


def run_validators(
    configs: list[dict[str, Any]],
    *,
    dry_run: bool = False,
) -> list[ValidationResult]:
    """Run a list of validation checks.

    Args:
        configs: List of validator config dicts. Each must have a "type" key.
        dry_run: If True, return what checks would run without executing them.

    Returns:
        List of ValidationResult in the same order as configs.

    Raises:
        ValueError: If a validator type is not registered.
    """
    results: list[ValidationResult] = []
    for config in configs:
        vtype = config.get("type")
        if not vtype:
            raise ValueError(f"Validator config missing 'type': {config}")
        if vtype not in _registry:
            raise ValueError(f"Unknown validator type: {vtype!r}. Registered: {sorted(_registry.keys())}")
        if dry_run:
            results.append(ValidationResult(
                type=vtype,
                passed=True,
                value=None,
                reason=f"[dry-run] Would check: {vtype} with {config}",
            ))
        else:
            results.append(_registry[vtype](config))
    return results


def spec_hash(task_def: dict[str, Any]) -> str:
    """SHA256 hash of a task definition for spec locking.

    Deterministic: sorts keys, uses separators without spaces.
    """
    canonical = json.dumps(task_def, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
