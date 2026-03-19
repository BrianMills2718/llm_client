#!/usr/bin/env python3
"""Append structured JSONL entries for Claude hook activity.

This logger keeps read-gating observable without changing the hook decision
path. Gate hooks record which file was targeted, which governing docs were
required, which were already read, and whether the edit was allowed or blocked.
Read hooks record which file was observed so operators can reconstruct session
context when investigating drift or surprising gate behavior.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from file_context import check_required_reads
from file_context import collect_context
from file_context import load_relationships


DEFAULT_CONFIG = Path("scripts/relationships.yaml")
DEFAULT_LOG_FILE = Path(".claude/hook_log.jsonl")


def _normalize(path_text: str) -> str:
    """Return a stable forward-slash path string for logging."""
    return path_text.replace("\\", "/").strip()


def _repo_root() -> Path:
    """Return the canonical repo root for this script."""
    return Path(__file__).resolve().parents[2]


def _resolve_path(repo_root: Path, raw_path: str) -> Path:
    """Resolve a repo-relative or absolute path against the repo root."""
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return repo_root / path


def _timestamp() -> str:
    """Return an ISO-8601 UTC timestamp suitable for JSONL logs."""
    return datetime.now(timezone.utc).isoformat()


def _write_entry(log_file: Path, entry: dict[str, Any]) -> None:
    """Append one JSON line, creating the parent directory when needed."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def _build_gate_entry(
    repo_root: Path,
    file_path: str,
    tool_name: str,
    decision: str,
    reads_file: Path,
    config_path: Path,
    reason: str,
) -> dict[str, Any]:
    """Build a gate-decision log entry with resolved required-read state."""
    relationships = load_relationships(repo_root=repo_root, config_path=config_path)
    normalized_path = _normalize(file_path)
    context = collect_context(normalized_path, relationships)
    _, required_reads, missing_reads = check_required_reads(
        normalized_path,
        relationships,
        reads_file,
    )
    missing_set = set(missing_reads)
    reads_completed = [doc for doc in required_reads if doc not in missing_set]

    return {
        "schema_version": 1,
        "timestamp": _timestamp(),
        "hook": "gate-edit",
        "tool_name": tool_name,
        "file_path": normalized_path,
        "decision": decision,
        "decision_reason": reason,
        "required_reads": required_reads,
        "reads_completed": reads_completed,
        "missing_reads": missing_reads,
        "coupled_docs": [doc["path"] for doc in context.coupled_docs],
    }


def _build_read_entry(
    repo_root: Path,
    file_path: str,
    reads_file: Path,
    reason: str,
) -> dict[str, Any]:
    """Build a read-tracking log entry after a read is appended."""
    resolved_reads_file = str(reads_file)
    try:
        resolved_reads_file = str(reads_file.relative_to(repo_root))
    except ValueError:
        resolved_reads_file = str(reads_file)

    return {
        "schema_version": 1,
        "timestamp": _timestamp(),
        "hook": "track-reads",
        "tool_name": "Read",
        "file_path": _normalize(file_path),
        "decision": "recorded",
        "decision_reason": reason,
        "reads_file": resolved_reads_file,
    }


def main(argv: list[str] | None = None) -> int:
    """Dispatch hook-log subcommands for gate and read hooks."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    gate_parser = subparsers.add_parser(
        "gate",
        help="Record a gate-edit allow/block/skip decision.",
    )
    gate_parser.add_argument("--file-path", required=True, help="Repo-relative path being edited")
    gate_parser.add_argument("--tool-name", required=True, help="Hook tool name (Edit or Write)")
    gate_parser.add_argument("--decision", required=True, help="allow, block, or skip")
    gate_parser.add_argument("--reason", default="", help="Short machine-readable or human-readable reason")
    gate_parser.add_argument(
        "--reads-file",
        required=True,
        help="Read-tracking session file used by the hook",
    )
    gate_parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Relationships config path relative to repo root",
    )
    gate_parser.add_argument(
        "--log-file",
        default=str(DEFAULT_LOG_FILE),
        help="JSONL log path relative to repo root",
    )

    read_parser = subparsers.add_parser(
        "read",
        help="Record a file read observed by the track-reads hook.",
    )
    read_parser.add_argument("--file-path", required=True, help="Repo-relative path being read")
    read_parser.add_argument(
        "--reads-file",
        required=True,
        help="Read-tracking session file used by the hook",
    )
    read_parser.add_argument("--reason", default="read observed", help="Short note for the log entry")
    read_parser.add_argument(
        "--log-file",
        default=str(DEFAULT_LOG_FILE),
        help="JSONL log path relative to repo root",
    )

    args = parser.parse_args(argv)
    repo_root = _repo_root()
    log_file = _resolve_path(repo_root, args.log_file)

    if args.command == "gate":
        entry = _build_gate_entry(
            repo_root=repo_root,
            file_path=args.file_path,
            tool_name=args.tool_name,
            decision=args.decision,
            reads_file=_resolve_path(repo_root, args.reads_file),
            config_path=_resolve_path(repo_root, args.config),
            reason=args.reason,
        )
    else:
        entry = _build_read_entry(
            repo_root=repo_root,
            file_path=args.file_path,
            reads_file=_resolve_path(repo_root, args.reads_file),
            reason=args.reason,
        )

    _write_entry(log_file, entry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
