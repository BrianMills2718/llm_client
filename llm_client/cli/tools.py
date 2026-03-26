"""Tool call observability CLI command.

Usage:
    python -m llm_client tools
    python -m llm_client tools --group-by tool_name,operation
    python -m llm_client tools --days 7 --project my_project
    python -m llm_client tools --data-loss
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

from llm_client.cli.common import connect, format_cost, format_latency


def cmd_tools(args: argparse.Namespace) -> None:
    """Display tool call aggregations from the observability DB."""

    db = connect()

    # Check if tool_calls table exists
    tables = {
        r[0]
        for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if "tool_calls" not in tables:
        print("No tool_calls table found in database.", file=sys.stderr)
        sys.exit(1)

    if args.data_loss:
        _show_data_loss_warnings(db, args)
        db.close()
        return

    group_cols = [g.strip() for g in args.group_by.split(",")]
    valid_cols = {"project", "tool_name", "operation", "provider", "status", "task", "trace_id"}
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
    if args.tool_name:
        where_parts.append("tool_name = ?")
        params.append(args.tool_name)
    if args.days:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()
        where_parts.append("timestamp >= ?")
        params.append(cutoff)

    where_sql = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""
    group_sql = ", ".join(group_cols)

    query = f"""
        SELECT {group_sql},
               COUNT(*) as calls,
               SUM(CASE WHEN status = 'succeeded' THEN 1 ELSE 0 END) as successes,
               SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failures,
               ROUND(AVG(CASE WHEN duration_ms IS NOT NULL THEN duration_ms / 1000.0 END), 2) as avg_latency_s,
               COALESCE(SUM(cost), 0) as total_cost,
               COALESCE(SUM(result_count), 0) as total_results,
               SUM(CASE WHEN data_loss_warning = 1 THEN 1 ELSE 0 END) as data_loss_warnings
        FROM tool_calls
        {where_sql}
        GROUP BY {group_sql}
        ORDER BY calls DESC
    """

    rows = db.execute(query, params).fetchall()

    if args.format == "json":
        data = []
        for row in rows:
            entry: dict[str, Any] = {}
            for i, col in enumerate(group_cols):
                entry[col] = row[i]
            gc = len(group_cols)
            entry["calls"] = row[gc]
            entry["successes"] = row[gc + 1]
            entry["failures"] = row[gc + 2]
            entry["avg_latency_s"] = row[gc + 3]
            entry["total_cost"] = row[gc + 4]
            entry["total_results"] = row[gc + 5]
            entry["data_loss_warnings"] = row[gc + 6]
            data.append(entry)
        print(json.dumps({"tool_calls": data}, indent=2))
    else:
        _print_tools_table(rows, group_cols)

    db.close()


def _show_data_loss_warnings(db: Any, args: argparse.Namespace) -> None:
    """Show tool calls that triggered data-loss warnings."""

    where_parts: list[str] = ["data_loss_warning = 1"]
    params: list[str | float] = []

    if args.project:
        where_parts.append("project = ?")
        params.append(args.project)
    if args.days:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()
        where_parts.append("timestamp >= ?")
        params.append(cutoff)

    where_sql = " WHERE " + " AND ".join(where_parts)

    query = f"""
        SELECT timestamp, tool_name, operation, raw_size, processed_size,
               ROUND(CAST(processed_size AS REAL) / NULLIF(raw_size, 0), 3) as ratio,
               trace_id, task
        FROM tool_calls
        {where_sql}
        ORDER BY timestamp DESC
        LIMIT {args.limit}
    """

    rows = db.execute(query, params).fetchall()

    if not rows:
        print("No data-loss warnings found.")
        return

    if args.format == "json":
        data = []
        for row in rows:
            data.append({
                "timestamp": row[0],
                "tool_name": row[1],
                "operation": row[2],
                "raw_size": row[3],
                "processed_size": row[4],
                "ratio": row[5],
                "trace_id": row[6],
                "task": row[7],
            })
        print(json.dumps({"data_loss_warnings": data}, indent=2))
        return

    headers = ["Timestamp", "Tool", "Operation", "Raw", "Processed", "Ratio", "Trace ID"]
    col_widths = [max(len(h), 10) for h in headers]

    for row in rows:
        col_widths[0] = max(col_widths[0], len(str(row[0])[:19]))
        col_widths[1] = max(col_widths[1], len(str(row[1] or "-")))
        col_widths[2] = max(col_widths[2], len(str(row[2] or "-")))
        col_widths[3] = max(col_widths[3], len(str(row[3] or 0)))
        col_widths[4] = max(col_widths[4], len(str(row[4] or 0)))
        col_widths[5] = max(col_widths[5], len(f"{row[5]:.3f}" if row[5] else "-"))
        col_widths[6] = max(col_widths[6], min(len(str(row[6] or "-")), 20))

    print(f"\nData Loss Warnings ({len(rows)} found):")
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("\u2500" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
    for row in rows:
        vals = [
            str(row[0])[:19],
            str(row[1] or "-"),
            str(row[2] or "-"),
            str(row[3] or 0),
            str(row[4] or 0),
            f"{row[5]:.3f}" if row[5] else "-",
            str(row[6] or "-")[:20],
        ]
        print(fmt.format(*vals))


def _print_tools_table(rows: list[Any], group_cols: list[str]) -> None:
    """Print tool call aggregation as a formatted table."""

    if not rows:
        print("No tool calls found.")
        return

    gc = len(group_cols)
    headers = [c.replace("_", " ").title() for c in group_cols] + [
        "Calls", "OK", "Fail", "Avg Lat", "Cost", "Results", "DL Warn",
    ]
    col_widths = [max(len(h), 8) for h in headers]

    for row in rows:
        for i in range(gc):
            col_widths[i] = max(col_widths[i], len(str(row[i] or "-")))
        col_widths[gc] = max(col_widths[gc], len(str(row[gc])))
        col_widths[gc + 1] = max(col_widths[gc + 1], len(str(row[gc + 1])))
        col_widths[gc + 2] = max(col_widths[gc + 2], len(str(row[gc + 2])))
        col_widths[gc + 3] = max(col_widths[gc + 3], len(format_latency(row[gc + 3])))
        col_widths[gc + 4] = max(col_widths[gc + 4], len(format_cost(row[gc + 4])))
        col_widths[gc + 5] = max(col_widths[gc + 5], len(str(row[gc + 5])))
        col_widths[gc + 6] = max(col_widths[gc + 6], len(str(row[gc + 6])))

    print("\nTool Calls:")
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("\u2500" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
    for row in rows:
        vals = [str(row[i] or "-") for i in range(gc)]
        vals += [
            str(row[gc]),
            str(row[gc + 1]),
            str(row[gc + 2]),
            format_latency(row[gc + 3]),
            format_cost(row[gc + 4]),
            str(row[gc + 5]),
            str(row[gc + 6]),
        ]
        print(fmt.format(*vals))

    # Summary line
    total_calls = sum(r[gc] for r in rows)
    total_failures = sum(r[gc + 2] for r in rows)
    total_cost = sum(r[gc + 4] for r in rows)
    total_dl = sum(r[gc + 6] for r in rows)
    print(
        f"\nTotal: {total_calls} calls, {total_failures} failures, "
        f"{format_cost(total_cost)} cost"
        + (f", {total_dl} data-loss warnings" if total_dl > 0 else "")
    )


def register_parser(subparsers: Any) -> None:
    """Register the ``tools`` subcommand with the CLI argument parser."""

    parser = subparsers.add_parser(
        "tools",
        help="Aggregate non-LLM tool call stats (search, fetch, extract, API calls)",
    )
    parser.add_argument(
        "--group-by",
        default="tool_name,operation",
        help="Comma-separated columns to group by (tool_name, operation, provider, project, status, task, trace_id)",
    )
    parser.add_argument("--project", help="Filter to a specific project")
    parser.add_argument("--trace-id", help="Filter to a specific trace_id")
    parser.add_argument("--tool-name", help="Filter to a specific tool name")
    parser.add_argument("--days", type=int, help="Only show last N days")
    parser.add_argument(
        "--data-loss",
        action="store_true",
        help="Show tool calls with data-loss warnings (processed/raw ratio < 10%%)",
    )
    parser.add_argument("--limit", type=int, default=50, help="Max rows for --data-loss output")
    parser.add_argument("--format", choices=["table", "json"], default="table", help="Output format")
    parser.set_defaults(handler=cmd_tools)
