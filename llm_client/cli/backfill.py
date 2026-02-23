"""JSONL -> SQLite backfill CLI command."""

from __future__ import annotations

import argparse
from typing import Any

from llm_client.cli.common import get_db_path


def cmd_backfill(args: argparse.Namespace) -> None:
    from llm_client import io_log

    db_path = get_db_path()

    if args.clear and db_path.exists():
        print(f"Clearing existing database at {db_path}")
        if io_log._db_conn is not None:
            io_log._db_conn.close()
            io_log._db_conn = None
        db_path.unlink()

    if io_log._db_conn is not None:
        io_log._db_conn.close()
        io_log._db_conn = None

    data_root = io_log._data_root
    print(f"Scanning {data_root} for JSONL files...")

    total_calls = 0
    total_emb = 0

    if not data_root.exists():
        print(f"Data root {data_root} does not exist.")
        return

    for project_dir in sorted(data_root.iterdir()):
        if not project_dir.is_dir():
            continue
        data_dir = project_dir / f"{project_dir.name}_llm_client_data"
        if not data_dir.is_dir():
            continue

        calls_file = data_dir / "calls.jsonl"
        if calls_file.exists():
            count = io_log.import_jsonl(calls_file, table="llm_calls")
            if count:
                print(f"  {project_dir.name}: {count} LLM calls imported")
                total_calls += count

        emb_file = data_dir / "embeddings.jsonl"
        if emb_file.exists():
            count = io_log.import_jsonl(emb_file, table="embeddings")
            if count:
                print(f"  {project_dir.name}: {count} embeddings imported")
                total_emb += count

    print(f"\nDone: {total_calls} LLM calls + {total_emb} embeddings -> {db_path}")


def register_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("backfill", help="Import JSONL logs into SQLite")
    parser.add_argument("--clear", action="store_true", help="Wipe DB before importing (avoids dupes)")
    parser.set_defaults(handler=cmd_backfill)
