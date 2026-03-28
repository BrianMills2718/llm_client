"""Trace summary CLI command."""

from __future__ import annotations

import argparse
import json
from typing import Any

from llm_client.cli.common import connect, format_cost, format_latency, format_tokens
from llm_client.observability.query import summarize_trace


def cmd_traces(args: argparse.Namespace) -> None:
    if args.trace_id:
        _show_trace_summary(args.trace_id, output_format=args.format)
        return

    db = connect()

    query = """
        SELECT trace_id,
               COUNT(*) as calls,
               COALESCE(SUM(COALESCE(marginal_cost, cost)), 0) as total_cost,
               COALESCE(SUM(total_tokens), 0) as total_tokens,
               MIN(timestamp) as first_ts,
               MAX(timestamp) as last_ts
        FROM llm_calls
        WHERE trace_id IS NOT NULL
        GROUP BY trace_id
        ORDER BY first_ts DESC
        LIMIT ?
    """
    rows = db.execute(query, (args.limit,)).fetchall()

    emb_query = """
        SELECT trace_id, COUNT(*) as emb_calls, COALESCE(SUM(cost), 0) as emb_cost
        FROM embeddings
        WHERE trace_id IS NOT NULL
        GROUP BY trace_id
    """
    emb_map = {r[0]: (r[1], r[2]) for r in db.execute(emb_query).fetchall()}

    if args.format == "json":
        data = []
        for row in rows:
            tid = row[0]
            emb_calls, emb_cost = emb_map.get(tid, (0, 0.0))
            data.append(
                {
                    "trace_id": tid,
                    "llm_calls": row[1],
                    "emb_calls": emb_calls,
                    "total_cost": row[2] + emb_cost,
                    "total_tokens": row[3],
                    "first_ts": row[4],
                    "last_ts": row[5],
                }
            )
        print(json.dumps(data, indent=2))
    else:
        if not rows:
            print("No traces found.")
            db.close()
            return

        headers = ["Trace ID", "LLM", "Emb", "Cost", "Tokens", "Started"]
        col_widths = [max(len(h), 10) for h in headers]
        display_rows = []
        for row in rows:
            tid = row[0]
            emb_calls, emb_cost = emb_map.get(tid, (0, 0.0))
            short_tid = tid if len(tid) <= 40 else tid[:37] + "..."
            ts_short = row[4][:19] if row[4] else "-"
            display_rows.append(
                (
                    short_tid,
                    str(row[1]),
                    str(emb_calls),
                    format_cost(row[2] + emb_cost),
                    format_tokens(row[3]),
                    ts_short,
                )
            )
            col_widths[0] = max(col_widths[0], len(short_tid))
            col_widths[1] = max(col_widths[1], len(str(row[1])))
            col_widths[2] = max(col_widths[2], len(str(emb_calls)))
            col_widths[3] = max(col_widths[3], len(format_cost(row[2] + emb_cost)))
            col_widths[4] = max(col_widths[4], len(format_tokens(row[3])))
            col_widths[5] = max(col_widths[5], len(ts_short))

        print("\nRecent Traces:")
        fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
        print(fmt.format(*headers))
        print("─" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
        for vals in display_rows:
            print(fmt.format(*vals))

    db.close()


def _show_trace_summary(trace_id: str, *, output_format: str) -> None:
    """Render one detailed trace summary across LLM and tool calls."""

    summary = summarize_trace(trace_id)
    if summary is None:
        print(f"No trace found for {trace_id}.")
        return

    if output_format == "json":
        print(json.dumps(summary, indent=2))
        return

    print(f"\nTrace: {summary['trace_id']}")
    print(
        "LLM calls: {llm_calls}  Tool calls: {tool_calls}  "
        "LLM errors: {llm_errors}  Tool failures: {tool_failures}".format(**summary)
    )
    print(
        f"Cost: {format_cost(summary['total_cost'])}  "
        f"Tokens: {format_tokens(summary['total_tokens'])}"
    )
    print(f"First: {summary['first_timestamp']}")
    print(f"Last:  {summary['last_timestamp']}")

    last_llm = summary.get("last_completed_llm_call")
    if last_llm:
        print(
            "Last completed LLM: "
            f"{last_llm['timestamp']}  {last_llm['caller']}  {last_llm['model']}  "
            f"{format_latency(last_llm['latency_s'])}"
        )

    last_tool = summary.get("last_tool_call")
    if last_tool:
        print(
            "Last tool call: "
            f"{last_tool['timestamp']}  {last_tool['tool_name']}::{last_tool['operation']}  "
            f"{last_tool['status']}  "
            f"{format_latency((last_tool['duration_ms'] or 0) / 1000.0 if last_tool['duration_ms'] is not None else None)}"
        )

    print("\nTimeline:")
    for event in summary["timeline"]:
        if event["kind"] == "llm_call":
            status = "error" if event["error"] else "ok"
            detail = f"{event['caller']} {event['model']}"
            if event["error"]:
                detail += f" error={event['error']}"
            print(
                f"  {event['timestamp']}  llm   {status:<5}  "
                f"{format_latency(event['latency_s'])}  {detail}"
            )
            continue

        detail = f"{event['tool_name']}::{event['operation']}"
        if event["provider"]:
            detail += f" provider={event['provider']}"
        if event["error_type"]:
            detail += f" error={event['error_type']}"
        duration_s = (
            (event["duration_ms"] or 0) / 1000.0
            if event["duration_ms"] is not None
            else None
        )
        print(
            f"  {event['timestamp']}  tool  {event['status']:<5}  "
            f"{format_latency(duration_s)}  {detail}"
        )


def register_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("traces", help="List recent traces with cost rollup")
    parser.add_argument("--limit", type=int, default=20, help="Max traces to show")
    parser.add_argument("--trace-id", help="Show one detailed trace across llm_calls and tool_calls")
    parser.add_argument("--format", choices=["table", "json"], default="table", help="Output format")
    parser.set_defaults(handler=cmd_traces)
