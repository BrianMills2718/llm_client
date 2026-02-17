"""Tests for llm_client.validators."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from llm_client.validators import (
    ValidationResult,
    _eval_check,
    register_validator,
    run_validators,
    spec_hash,
)


# --- spec_hash ---


def test_spec_hash_deterministic():
    d = {"b": 2, "a": 1}
    assert spec_hash(d) == spec_hash({"a": 1, "b": 2})


def test_spec_hash_different_for_different_inputs():
    assert spec_hash({"a": 1}) != spec_hash({"a": 2})


# --- _eval_check ---


@pytest.mark.parametrize(
    "value,expr,expected",
    [
        (5, "> 0", True),
        (0, "> 0", False),
        (5, ">= 5", True),
        (4, ">= 5", False),
        (3, "< 5", True),
        (5, "< 5", False),
        (5, "<= 5", True),
        (6, "<= 5", False),
        (5, "== 5", True),
        (4, "== 5", False),
        (4, "!= 5", True),
        (5, "!= 5", False),
    ],
)
def test_eval_check(value: int, expr: str, expected: bool):
    assert _eval_check(value, expr) == expected


def test_eval_check_invalid_expression():
    with pytest.raises(ValueError, match="Invalid check expression"):
        _eval_check(5, "is 5")


# --- file_exists ---


def test_file_exists_pass(tmp_path: Path):
    f = tmp_path / "test.txt"
    f.write_text("hello")
    results = run_validators([{"type": "file_exists", "path": str(f)}])
    assert len(results) == 1
    assert results[0].passed is True


def test_file_exists_fail(tmp_path: Path):
    results = run_validators([{"type": "file_exists", "path": str(tmp_path / "missing.txt")}])
    assert results[0].passed is False
    assert "not found" in results[0].reason


# --- file_not_empty ---


def test_file_not_empty_pass(tmp_path: Path):
    f = tmp_path / "data.txt"
    f.write_text("content")
    results = run_validators([{"type": "file_not_empty", "path": str(f)}])
    assert results[0].passed is True
    assert results[0].value > 0


def test_file_not_empty_fail_missing(tmp_path: Path):
    results = run_validators([{"type": "file_not_empty", "path": str(tmp_path / "nope.txt")}])
    assert results[0].passed is False


def test_file_not_empty_fail_empty(tmp_path: Path):
    f = tmp_path / "empty.txt"
    f.write_text("")
    results = run_validators([{"type": "file_not_empty", "path": str(f)}])
    assert results[0].passed is False
    assert results[0].value == 0


def test_file_not_empty_min_bytes(tmp_path: Path):
    f = tmp_path / "small.txt"
    f.write_text("ab")
    results = run_validators([{"type": "file_not_empty", "path": str(f), "min_bytes": 100}])
    assert results[0].passed is False
    assert "too small" in results[0].reason.lower()


# --- json_schema ---


def test_json_schema_pass(tmp_path: Path):
    f = tmp_path / "data.json"
    f.write_text(json.dumps({"name": "test", "value": 42}))
    results = run_validators([{
        "type": "json_schema",
        "path": str(f),
        "schema": {
            "type": "object",
            "required": ["name", "value"],
        },
    }])
    assert results[0].passed is True


def test_json_schema_fail_invalid_json(tmp_path: Path):
    f = tmp_path / "bad.json"
    f.write_text("not json at all")
    results = run_validators([{
        "type": "json_schema",
        "path": str(f),
        "schema": {"type": "object"},
    }])
    assert results[0].passed is False
    assert "parse error" in results[0].reason.lower()


def test_json_schema_fail_missing_file(tmp_path: Path):
    results = run_validators([{
        "type": "json_schema",
        "path": str(tmp_path / "missing.json"),
        "schema": {"type": "object"},
    }])
    assert results[0].passed is False


def test_json_schema_fail_schema_mismatch(tmp_path: Path):
    """Requires jsonschema to be installed."""
    pytest.importorskip("jsonschema")
    f = tmp_path / "data.json"
    f.write_text(json.dumps({"name": "test"}))
    results = run_validators([{
        "type": "json_schema",
        "path": str(f),
        "schema": {
            "type": "object",
            "required": ["name", "missing_field"],
        },
    }])
    assert results[0].passed is False
    assert "schema validation failed" in results[0].reason.lower()


# --- sql_count ---


def test_sql_count_pass(tmp_path: Path):
    db = tmp_path / "test.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("INSERT INTO items (name) VALUES ('a'), ('b'), ('c')")
    conn.commit()
    conn.close()

    results = run_validators([{
        "type": "sql_count",
        "db": str(db),
        "query": "SELECT count(*) FROM items",
        "check": "> 0",
    }])
    assert results[0].passed is True
    assert results[0].value == 3


def test_sql_count_fail(tmp_path: Path):
    db = tmp_path / "test.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()

    results = run_validators([{
        "type": "sql_count",
        "db": str(db),
        "query": "SELECT count(*) FROM items",
        "check": "> 0",
    }])
    assert results[0].passed is False
    assert results[0].value == 0


def test_sql_count_missing_db(tmp_path: Path):
    results = run_validators([{
        "type": "sql_count",
        "db": str(tmp_path / "nope.db"),
        "query": "SELECT 1",
        "check": "> 0",
    }])
    assert results[0].passed is False
    assert "not found" in results[0].reason.lower()


# --- command ---


def test_command_pass():
    results = run_validators([{"type": "command", "command": "true"}])
    assert results[0].passed is True
    assert results[0].value == 0


def test_command_fail():
    results = run_validators([{"type": "command", "command": "false"}])
    assert results[0].passed is False
    assert results[0].value == 1


# --- dry_run ---


def test_dry_run_returns_all_passing():
    configs = [
        {"type": "file_exists", "path": "/nonexistent"},
        {"type": "command", "command": "false"},
    ]
    results = run_validators(configs, dry_run=True)
    assert len(results) == 2
    assert all(r.passed for r in results)
    assert all("[dry-run]" in r.reason for r in results)


# --- unknown type ---


def test_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown validator type"):
        run_validators([{"type": "bogus"}])


def test_missing_type_raises():
    with pytest.raises(ValueError, match="missing 'type'"):
        run_validators([{"path": "/tmp/foo"}])


# --- custom validator ---


def test_custom_validator():
    def my_validator(config: dict) -> ValidationResult:
        return ValidationResult(
            type="custom_check",
            passed=config.get("expected", False),
            value="custom_value",
        )

    register_validator("custom_check", my_validator)
    results = run_validators([{"type": "custom_check", "expected": True}])
    assert results[0].passed is True
    assert results[0].value == "custom_value"


# --- multiple validators ---


def test_multiple_validators(tmp_path: Path):
    f = tmp_path / "output.txt"
    f.write_text("hello world")
    configs = [
        {"type": "file_exists", "path": str(f)},
        {"type": "file_not_empty", "path": str(f)},
    ]
    results = run_validators(configs)
    assert len(results) == 2
    assert all(r.passed for r in results)
