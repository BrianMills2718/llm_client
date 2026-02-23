"""Shared CLI helpers for llm_client observability commands."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


def get_db_path() -> Path:
    from llm_client import io_log

    return io_log._db_path


def connect() -> sqlite3.Connection:
    db_path = get_db_path()
    if not db_path.exists():
        print(
            f"No database at {db_path}. Run 'python -m llm_client backfill' first.",
            file=sys.stderr,
        )
        sys.exit(1)
    return sqlite3.connect(str(db_path))


def format_tokens(n: int | None) -> str:
    if n is None or n == 0:
        return "0"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def format_cost(c: float | None) -> str:
    if c is None:
        return "$0.0000"
    return f"${c:.4f}"


def format_latency(s: float | None) -> str:
    if s is None:
        return "-"
    return f"{s:.2f}s"
