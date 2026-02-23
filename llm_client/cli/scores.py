"""Rubric score CLI command."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

from llm_client.cli.common import connect, format_cost


def cmd_scores(args: argparse.Namespace) -> None:
    db = connect()

    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if "task_scores" not in tables:
        print("No task_scores table found. Score some outputs first.", file=sys.stderr)
        sys.exit(1)

    group_cols = [g.strip() for g in args.group_by.split(",")]
    valid_cols = {
        "rubric",
        "task",
        "project",
        "output_model",
        "judge_model",
        "method",
        "agent_spec",
        "prompt_id",
    }
    for g in group_cols:
        if g not in valid_cols:
            print(
                f"Invalid group-by column: {g!r}. Valid: {', '.join(sorted(valid_cols))}",
                file=sys.stderr,
            )
            sys.exit(1)

    where_parts: list[str] = []
    params: list[str | float] = []

    if args.rubric:
        where_parts.append("rubric = ?")
        params.append(args.rubric)
    if args.task:
        where_parts.append("task = ?")
        params.append(args.task)
    if args.project:
        where_parts.append("project = ?")
        params.append(args.project)
    if args.days:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()
        where_parts.append("timestamp >= ?")
        params.append(cutoff)

    where_sql = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""

    if args.trend:
        query = f"""
            SELECT DATE(timestamp) as day,
                   COUNT(*) as count,
                   ROUND(AVG(overall_score), 4) as avg_score,
                   ROUND(MIN(overall_score), 4) as min_score,
                   ROUND(MAX(overall_score), 4) as max_score
            FROM task_scores
            {where_sql}
            GROUP BY DATE(timestamp)
            ORDER BY day DESC
            LIMIT 30
        """
        rows = db.execute(query, params).fetchall()

        if args.format == "json":
            data = [
                {"day": r[0], "count": r[1], "avg": r[2], "min": r[3], "max": r[4]}
                for r in rows
            ]
            print(json.dumps(data, indent=2))
        else:
            if not rows:
                print("No scores found.")
                db.close()
                return
            headers = ["Day", "Count", "Avg Score", "Min", "Max"]
            col_widths = [max(len(h), 10) for h in headers]
            print("\nScore Trend:")
            fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
            print(fmt.format(*headers))
            print("─" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
            for row in rows:
                print(fmt.format(str(row[0]), str(row[1]), str(row[2]), str(row[3]), str(row[4])))
    else:
        group_sql = ", ".join(group_cols)
        query = f"""
            SELECT {group_sql},
                   COUNT(*) as count,
                   ROUND(AVG(overall_score), 4) as avg_score,
                   ROUND(MIN(overall_score), 4) as min_score,
                   ROUND(MAX(overall_score), 4) as max_score,
                   COALESCE(SUM(cost), 0) as total_cost
            FROM task_scores
            {where_sql}
            GROUP BY {group_sql}
            ORDER BY avg_score DESC
        """
        rows = db.execute(query, params).fetchall()

        if args.format == "json":
            data = []
            for row in rows:
                entry: dict = {}
                for i, col in enumerate(group_cols):
                    entry[col] = row[i]
                gc = len(group_cols)
                entry["count"] = row[gc]
                entry["avg_score"] = row[gc + 1]
                entry["min_score"] = row[gc + 2]
                entry["max_score"] = row[gc + 3]
                entry["total_cost"] = row[gc + 4]
                data.append(entry)
            print(json.dumps(data, indent=2))
        else:
            if not rows:
                print("No scores found.")
                db.close()
                return
            gc = len(group_cols)
            headers = [c.title() for c in group_cols] + ["Count", "Avg Score", "Min", "Max", "Cost"]
            col_widths = [max(len(h), 10) for h in headers]
            for row in rows:
                for i in range(gc):
                    col_widths[i] = max(col_widths[i], len(str(row[i] or "-")))

            print("\nRubric Scores:")
            fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
            print(fmt.format(*headers))
            print("─" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
            for row in rows:
                vals = [str(row[i] or "-") for i in range(gc)]
                vals += [
                    str(row[gc]),
                    str(row[gc + 1]),
                    str(row[gc + 2]),
                    str(row[gc + 3]),
                    format_cost(row[gc + 4]),
                ]
                print(fmt.format(*vals))

    db.close()


def register_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("scores", help="Aggregate rubric scores")
    parser.add_argument(
        "--group-by",
        default="rubric,task",
        help="Comma-separated columns (rubric, task, project, output_model, judge_model, method, agent_spec, prompt_id)",
    )
    parser.add_argument("--rubric", help="Filter to a specific rubric")
    parser.add_argument("--task", help="Filter to a specific task")
    parser.add_argument("--project", help="Filter to a specific project")
    parser.add_argument("--days", type=int, help="Only show last N days")
    parser.add_argument("--trend", action="store_true", help="Show daily score trend instead of aggregates")
    parser.add_argument("--format", choices=["table", "json"], default="table", help="Output format")
    parser.set_defaults(handler=cmd_scores)
