"""Cost aggregation CLI command."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

from llm_client.cli.common import connect, format_cost, format_latency, format_tokens


def cmd_cost(args: argparse.Namespace) -> None:
    db = connect()

    group_cols = [g.strip() for g in args.group_by.split(",")]
    valid_cols = {"project", "model", "caller", "task", "trace_id"}
    for g in group_cols:
        if g not in valid_cols:
            print(
                f"Invalid group-by column: {g!r}. Valid: {', '.join(sorted(valid_cols))}",
                file=sys.stderr,
            )
            sys.exit(1)

    where_parts: list[str] = []
    params: list[str | float] = []

    if args.project:
        where_parts.append("project = ?")
        params.append(args.project)
    if args.trace_id:
        where_parts.append("trace_id = ?")
        params.append(args.trace_id)
    if args.days:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()
        where_parts.append("timestamp >= ?")
        params.append(cutoff)

    where_sql = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""
    group_sql = ", ".join(group_cols)

    query = f"""
        SELECT {group_sql},
               COUNT(*) as calls,
               COALESCE(SUM(COALESCE(marginal_cost, cost)), 0) as total_cost,
               COALESCE(SUM(total_tokens), 0) as total_tokens,
               ROUND(AVG(latency_s), 2) as avg_latency,
               SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as errors
        FROM llm_calls
        {where_sql}
        GROUP BY {group_sql}
        ORDER BY total_cost DESC
    """
    rows = db.execute(query, params).fetchall()

    if args.format == "json":
        data = []
        for row in rows:
            entry: dict[str, Any] = {}
            for i, col in enumerate(group_cols):
                entry[col] = row[i]
            entry["calls"] = row[len(group_cols)]
            entry["cost"] = row[len(group_cols) + 1]
            entry["total_tokens"] = row[len(group_cols) + 2]
            entry["avg_latency_s"] = row[len(group_cols) + 3]
            entry["errors"] = row[len(group_cols) + 4]
            data.append(entry)
        print(json.dumps({"llm_calls": data}, indent=2))
    else:
        _print_calls_table(rows, group_cols)

    emb_query = f"""
        SELECT {group_sql},
               COUNT(*) as calls,
               COALESCE(SUM(cost), 0) as total_cost,
               COALESCE(SUM(input_count), 0) as total_vectors,
               ROUND(AVG(latency_s), 2) as avg_latency
        FROM embeddings
        {where_sql}
        GROUP BY {group_sql}
        ORDER BY total_cost DESC
    """
    emb_rows = db.execute(emb_query, params).fetchall()
    if emb_rows:
        if args.format == "json":
            emb_data = []
            for row in emb_rows:
                entry = {}
                for i, col in enumerate(group_cols):
                    entry[col] = row[i]
                entry["calls"] = row[len(group_cols)]
                entry["cost"] = row[len(group_cols) + 1]
                entry["total_vectors"] = row[len(group_cols) + 2]
                entry["avg_latency_s"] = row[len(group_cols) + 3]
                emb_data.append(entry)
            print(json.dumps({"embeddings": emb_data}, indent=2))
        else:
            _print_embeddings_table(emb_rows, group_cols)

    if args.format != "json":
        total_cost = sum(r[len(group_cols) + 1] for r in rows) + sum(
            r[len(group_cols) + 1] for r in emb_rows
        )
        total_calls = sum(r[len(group_cols)] for r in rows)
        total_emb = sum(r[len(group_cols)] for r in emb_rows)
        print(
            f"\nGrand total: {total_calls} LLM calls + {total_emb} embeddings = {format_cost(total_cost)}"
        )

    db.close()


def _print_calls_table(rows: list[Any], group_cols: list[str]) -> None:
    if not rows:
        print("No LLM calls found.")
        return

    gc = len(group_cols)
    headers = [c.title() for c in group_cols] + ["Calls", "Cost", "Tokens", "Avg Lat", "Errors"]
    col_widths = [max(len(h), 8) for h in headers]

    for row in rows:
        for i in range(gc):
            col_widths[i] = max(col_widths[i], len(str(row[i] or "-")))
        col_widths[gc] = max(col_widths[gc], len(str(row[gc])))
        col_widths[gc + 1] = max(col_widths[gc + 1], len(format_cost(row[gc + 1])))
        col_widths[gc + 2] = max(col_widths[gc + 2], len(format_tokens(row[gc + 2])))
        col_widths[gc + 3] = max(col_widths[gc + 3], len(format_latency(row[gc + 3])))
        col_widths[gc + 4] = max(col_widths[gc + 4], len(str(row[gc + 4])))

    print("\nLLM Calls:")
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("─" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
    for row in rows:
        vals = [str(row[i] or "-") for i in range(gc)]
        vals += [
            str(row[gc]),
            format_cost(row[gc + 1]),
            format_tokens(row[gc + 2]),
            format_latency(row[gc + 3]),
            str(row[gc + 4]),
        ]
        print(fmt.format(*vals))


def _print_embeddings_table(rows: list[Any], group_cols: list[str]) -> None:
    if not rows:
        return

    gc = len(group_cols)
    headers = [c.title() for c in group_cols] + ["Calls", "Cost", "Vectors", "Avg Lat"]
    col_widths = [max(len(h), 8) for h in headers]

    for row in rows:
        for i in range(gc):
            col_widths[i] = max(col_widths[i], len(str(row[i] or "-")))
        col_widths[gc] = max(col_widths[gc], len(str(row[gc])))
        col_widths[gc + 1] = max(col_widths[gc + 1], len(format_cost(row[gc + 1])))
        col_widths[gc + 2] = max(col_widths[gc + 2], len(str(row[gc + 2])))
        col_widths[gc + 3] = max(col_widths[gc + 3], len(format_latency(row[gc + 3])))

    print("\nEmbeddings:")
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("─" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
    for row in rows:
        vals = [str(row[i] or "-") for i in range(gc)]
        vals += [
            str(row[gc]),
            format_cost(row[gc + 1]),
            str(row[gc + 2]),
            format_latency(row[gc + 3]),
        ]
        print(fmt.format(*vals))


def register_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("cost", help="Aggregate LLM spend by grouping dimensions")
    parser.add_argument(
        "--group-by",
        default="project,model",
        help="Comma-separated columns to group by (project, model, caller, task, trace_id)",
    )
    parser.add_argument("--project", help="Filter to a specific project")
    parser.add_argument("--trace-id", help="Filter to a specific trace_id")
    parser.add_argument("--days", type=int, help="Only show last N days")
    parser.add_argument("--format", choices=["table", "json"], default="table", help="Output format")
    parser.set_defaults(handler=cmd_cost)
