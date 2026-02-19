"""Cost dashboard CLI for llm_client observability.

Usage:
    python -m llm_client cost                          # group by project,model
    python -m llm_client cost --group-by project       # project totals
    python -m llm_client cost --group-by task,model    # task breakdown
    python -m llm_client cost --project myproj --days 7
    python -m llm_client cost --trace-id "basic_local_abc123"
    python -m llm_client cost --format json

    python -m llm_client traces                        # last 20 traces
    python -m llm_client traces --limit 50

    python -m llm_client scores                        # rubric score summary
    python -m llm_client scores --rubric research_quality --group-by model
    python -m llm_client scores --task sam_gov_research --trend

    python -m llm_client experiments                   # recent experiment runs
    python -m llm_client experiments --dataset MuSiQue  # filter by dataset
    python -m llm_client experiments --compare RUN1 RUN2  # side-by-side
    python -m llm_client experiments --detail RUN_ID    # per-item breakdown
    python -m llm_client experiments --format json      # machine-readable

    python -m llm_client backfill                      # import JSONL → SQLite
    python -m llm_client backfill --clear              # wipe + reimport
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path


def _get_db_path() -> Path:
    from llm_client import io_log
    return io_log._db_path


def _connect() -> sqlite3.Connection:
    db_path = _get_db_path()
    if not db_path.exists():
        print(f"No database at {db_path}. Run 'python -m llm_client backfill' first.", file=sys.stderr)
        sys.exit(1)
    return sqlite3.connect(str(db_path))


def _format_tokens(n: int | None) -> str:
    if n is None or n == 0:
        return "0"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _format_cost(c: float | None) -> str:
    if c is None:
        return "$0.0000"
    return f"${c:.4f}"


def _format_latency(s: float | None) -> str:
    if s is None:
        return "-"
    return f"{s:.2f}s"


# ---------------------------------------------------------------------------
# cost subcommand
# ---------------------------------------------------------------------------


def cmd_cost(args: argparse.Namespace) -> None:
    db = _connect()

    group_cols = [g.strip() for g in args.group_by.split(",")]
    valid_cols = {"project", "model", "caller", "task", "trace_id"}
    for g in group_cols:
        if g not in valid_cols:
            print(f"Invalid group-by column: {g!r}. Valid: {', '.join(sorted(valid_cols))}", file=sys.stderr)
            sys.exit(1)

    # Build WHERE clause
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

    # LLM calls
    query = f"""
        SELECT {group_sql},
               COUNT(*) as calls,
               COALESCE(SUM(cost), 0) as total_cost,
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
            entry: dict = {}
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

    # Embeddings
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

    # Totals
    if args.format != "json":
        total_cost = sum(r[len(group_cols) + 1] for r in rows) + sum(r[len(group_cols) + 1] for r in emb_rows)
        total_calls = sum(r[len(group_cols)] for r in rows)
        total_emb = sum(r[len(group_cols)] for r in emb_rows)
        print(f"\nGrand total: {total_calls} LLM calls + {total_emb} embeddings = {_format_cost(total_cost)}")

    db.close()


def _print_calls_table(rows: list, group_cols: list[str]) -> None:
    if not rows:
        print("No LLM calls found.")
        return

    # Column widths
    gc = len(group_cols)
    headers = [c.title() for c in group_cols] + ["Calls", "Cost", "Tokens", "Avg Lat", "Errors"]
    col_widths = [max(len(h), 8) for h in headers]

    # Compute widths from data
    for row in rows:
        for i in range(gc):
            col_widths[i] = max(col_widths[i], len(str(row[i] or "-")))
        col_widths[gc] = max(col_widths[gc], len(str(row[gc])))
        col_widths[gc + 1] = max(col_widths[gc + 1], len(_format_cost(row[gc + 1])))
        col_widths[gc + 2] = max(col_widths[gc + 2], len(_format_tokens(row[gc + 2])))
        col_widths[gc + 3] = max(col_widths[gc + 3], len(_format_latency(row[gc + 3])))
        col_widths[gc + 4] = max(col_widths[gc + 4], len(str(row[gc + 4])))

    print("\nLLM Calls:")
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("─" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
    for row in rows:
        vals = [str(row[i] or "-") for i in range(gc)]
        vals += [
            str(row[gc]),
            _format_cost(row[gc + 1]),
            _format_tokens(row[gc + 2]),
            _format_latency(row[gc + 3]),
            str(row[gc + 4]),
        ]
        print(fmt.format(*vals))


def _print_embeddings_table(rows: list, group_cols: list[str]) -> None:
    if not rows:
        return

    gc = len(group_cols)
    headers = [c.title() for c in group_cols] + ["Calls", "Cost", "Vectors", "Avg Lat"]
    col_widths = [max(len(h), 8) for h in headers]

    for row in rows:
        for i in range(gc):
            col_widths[i] = max(col_widths[i], len(str(row[i] or "-")))
        col_widths[gc] = max(col_widths[gc], len(str(row[gc])))
        col_widths[gc + 1] = max(col_widths[gc + 1], len(_format_cost(row[gc + 1])))
        col_widths[gc + 2] = max(col_widths[gc + 2], len(str(row[gc + 2])))
        col_widths[gc + 3] = max(col_widths[gc + 3], len(_format_latency(row[gc + 3])))

    print("\nEmbeddings:")
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("─" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
    for row in rows:
        vals = [str(row[i] or "-") for i in range(gc)]
        vals += [
            str(row[gc]),
            _format_cost(row[gc + 1]),
            str(row[gc + 2]),
            _format_latency(row[gc + 3]),
        ]
        print(fmt.format(*vals))


# ---------------------------------------------------------------------------
# traces subcommand
# ---------------------------------------------------------------------------


def cmd_traces(args: argparse.Namespace) -> None:
    db = _connect()

    query = """
        SELECT trace_id,
               COUNT(*) as calls,
               COALESCE(SUM(cost), 0) as total_cost,
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

    # Also get embedding counts per trace
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
            data.append({
                "trace_id": tid,
                "llm_calls": row[1],
                "emb_calls": emb_calls,
                "total_cost": row[2] + emb_cost,
                "total_tokens": row[3],
                "first_ts": row[4],
                "last_ts": row[5],
            })
        print(json.dumps(data, indent=2))
    else:
        if not rows:
            print("No traces found.")
            return

        headers = ["Trace ID", "LLM", "Emb", "Cost", "Tokens", "Started"]
        col_widths = [max(len(h), 10) for h in headers]
        display_rows = []
        for row in rows:
            tid = row[0]
            emb_calls, emb_cost = emb_map.get(tid, (0, 0.0))
            # Truncate trace_id for display
            short_tid = tid if len(tid) <= 40 else tid[:37] + "..."
            ts_short = row[4][:19] if row[4] else "-"
            display_rows.append((
                short_tid, str(row[1]), str(emb_calls),
                _format_cost(row[2] + emb_cost),
                _format_tokens(row[3]), ts_short,
            ))
            col_widths[0] = max(col_widths[0], len(short_tid))
            col_widths[1] = max(col_widths[1], len(str(row[1])))
            col_widths[2] = max(col_widths[2], len(str(emb_calls)))
            col_widths[3] = max(col_widths[3], len(_format_cost(row[2] + emb_cost)))
            col_widths[4] = max(col_widths[4], len(_format_tokens(row[3])))
            col_widths[5] = max(col_widths[5], len(ts_short))

        print("\nRecent Traces:")
        fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
        print(fmt.format(*headers))
        print("─" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
        for vals in display_rows:
            print(fmt.format(*vals))

    db.close()


# ---------------------------------------------------------------------------
# scores subcommand
# ---------------------------------------------------------------------------


def cmd_scores(args: argparse.Namespace) -> None:
    db = _connect()

    # Check if task_scores table exists
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if "task_scores" not in tables:
        print("No task_scores table found. Score some outputs first.", file=sys.stderr)
        sys.exit(1)

    group_cols = [g.strip() for g in args.group_by.split(",")]
    valid_cols = {"rubric", "task", "project", "output_model", "judge_model", "method", "agent_spec", "prompt_id"}
    for g in group_cols:
        if g not in valid_cols:
            print(f"Invalid group-by column: {g!r}. Valid: {', '.join(sorted(valid_cols))}", file=sys.stderr)
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
        # Show scores over time (daily averages)
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
            data = [{"day": r[0], "count": r[1], "avg": r[2], "min": r[3], "max": r[4]} for r in rows]
            print(json.dumps(data, indent=2))
        else:
            if not rows:
                print("No scores found.")
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
        # Aggregate by group columns
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
                vals += [str(row[gc]), str(row[gc + 1]), str(row[gc + 2]), str(row[gc + 3]), _format_cost(row[gc + 4])]
                print(fmt.format(*vals))

    db.close()


# ---------------------------------------------------------------------------
# experiments subcommand
# ---------------------------------------------------------------------------


def cmd_experiments(args: argparse.Namespace) -> None:
    from llm_client import io_log

    if args.compare:
        _cmd_experiments_compare(args)
        return

    if args.detail:
        _cmd_experiments_detail(args)
        return

    # List runs
    runs = io_log.get_runs(
        dataset=args.dataset,
        model=args.model,
        project=args.project,
        since=args.since,
        limit=args.limit,
    )

    if args.format == "json":
        print(json.dumps(runs, indent=2))
        return

    if not runs:
        print("No experiment runs found.")
        return

    headers = ["Run ID", "Dataset", "Model", "Items", "Status", "Metrics", "Cost", "Wall", "Time"]
    col_widths = [max(len(h), 6) for h in headers]

    display_rows = []
    for r in runs:
        rid = r["run_id"][:12]
        ds = (r["dataset"] or "-")[:20]
        mdl = (r["model"] or "-")[:25]
        items = f"{r['n_completed'] or 0}/{r['n_items'] or 0}"
        status = r["status"] or "-"

        # Format summary metrics compactly
        sm = r.get("summary_metrics") or {}
        metrics_str = "  ".join(f"{k}={v}" for k, v in sm.items()) if sm else "-"
        if len(metrics_str) > 40:
            metrics_str = metrics_str[:37] + "..."

        cost = _format_cost(r.get("total_cost"))
        wall = f"{r['wall_time_s']:.0f}s" if r.get("wall_time_s") else "-"
        ts = r["timestamp"][:16] if r.get("timestamp") else "-"

        row = (rid, ds, mdl, items, status, metrics_str, cost, wall, ts)
        display_rows.append(row)
        for i, v in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(v)))

    print("\nExperiment Runs:")
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("─" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
    for row in display_rows:
        print(fmt.format(*row))


def _cmd_experiments_compare(args: argparse.Namespace) -> None:
    from llm_client import io_log

    try:
        result = io_log.compare_runs(args.compare)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.format == "json":
        print(json.dumps(result, indent=2))
        return

    runs = result["runs"]
    deltas = result["deltas_from_first"]

    # Header
    print("\nExperiment Comparison:")
    print("─" * 70)
    for i, r in enumerate(runs):
        label = "BASELINE" if i == 0 else f"vs base"
        sm = r.get("summary_metrics") or {}
        metrics_str = "  ".join(f"{k}={v}" for k, v in sm.items())
        print(f"  [{r['run_id'][:12]}] {r['dataset']} / {r['model']}")
        print(f"    {r['n_completed']}/{r['n_items']} items  ${r['total_cost']:.4f}  {label}")
        print(f"    {metrics_str}")
        if i > 0 and deltas[i - 1]:
            delta_str = "  ".join(
                f"{k}={v:+.2f}" for k, v in deltas[i - 1].items()
            )
            print(f"    Deltas: {delta_str}")
        print()


def _cmd_experiments_detail(args: argparse.Namespace) -> None:
    from llm_client import io_log

    run_id = args.detail
    items = io_log.get_run_items(run_id)

    if args.format == "json":
        print(json.dumps(items, indent=2))
        return

    if not items:
        print(f"No items found for run {run_id}")
        return

    # Get run info
    runs = io_log.get_runs(limit=1000)
    run_info = next((r for r in runs if r["run_id"] == run_id), None)
    if run_info:
        print(f"\nRun: {run_id}")
        print(f"  Dataset: {run_info['dataset']}  Model: {run_info['model']}  Status: {run_info['status']}")
        sm = run_info.get("summary_metrics") or {}
        if sm:
            print(f"  Summary: {'  '.join(f'{k}={v}' for k, v in sm.items())}")
        print()

    headers = ["Item", "Metrics", "Predicted", "Gold", "Cost", "Latency", "Error"]
    col_widths = [max(len(h), 6) for h in headers]

    display_rows = []
    for it in items:
        iid = str(it["item_id"])[:10]
        m = it.get("metrics") or {}
        metrics_str = " ".join(f"{k}={v}" for k, v in m.items())
        if len(metrics_str) > 30:
            metrics_str = metrics_str[:27] + "..."
        pred = (it.get("predicted") or "-")[:30]
        gold = (it.get("gold") or "-")[:30]
        cost = _format_cost(it.get("cost"))
        lat = f"{it['latency_s']:.1f}s" if it.get("latency_s") else "-"
        err = (it.get("error") or "")[:20]

        row = (iid, metrics_str, pred, gold, cost, lat, err)
        display_rows.append(row)
        for i, v in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(v)))

    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("─" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
    for row in display_rows:
        print(fmt.format(*row))


# ---------------------------------------------------------------------------
# backfill subcommand
# ---------------------------------------------------------------------------


def cmd_backfill(args: argparse.Namespace) -> None:
    from llm_client import io_log

    db_path = _get_db_path()

    if args.clear and db_path.exists():
        print(f"Clearing existing database at {db_path}")
        # Close any existing connection
        if io_log._db_conn is not None:
            io_log._db_conn.close()
            io_log._db_conn = None
        db_path.unlink()

    # Force fresh DB creation
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

    print(f"\nDone: {total_calls} LLM calls + {total_emb} embeddings → {db_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llm_client",
        description="LLM cost dashboard and observability tools",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # cost
    cost_p = sub.add_parser("cost", help="Aggregate LLM spend by grouping dimensions")
    cost_p.add_argument("--group-by", default="project,model", help="Comma-separated columns to group by (project, model, caller, task, trace_id)")
    cost_p.add_argument("--project", help="Filter to a specific project")
    cost_p.add_argument("--trace-id", help="Filter to a specific trace_id")
    cost_p.add_argument("--days", type=int, help="Only show last N days")
    cost_p.add_argument("--format", choices=["table", "json"], default="table", help="Output format")

    # traces
    traces_p = sub.add_parser("traces", help="List recent traces with cost rollup")
    traces_p.add_argument("--limit", type=int, default=20, help="Max traces to show")
    traces_p.add_argument("--format", choices=["table", "json"], default="table", help="Output format")

    # scores
    scores_p = sub.add_parser("scores", help="Aggregate rubric scores")
    scores_p.add_argument("--group-by", default="rubric,task", help="Comma-separated columns (rubric, task, project, output_model, judge_model, method, agent_spec, prompt_id)")
    scores_p.add_argument("--rubric", help="Filter to a specific rubric")
    scores_p.add_argument("--task", help="Filter to a specific task")
    scores_p.add_argument("--project", help="Filter to a specific project")
    scores_p.add_argument("--days", type=int, help="Only show last N days")
    scores_p.add_argument("--trend", action="store_true", help="Show daily score trend instead of aggregates")
    scores_p.add_argument("--format", choices=["table", "json"], default="table", help="Output format")

    # experiments
    exp_p = sub.add_parser("experiments", help="List and compare experiment runs")
    exp_p.add_argument("--dataset", help="Filter to a dataset")
    exp_p.add_argument("--model", help="Filter to a model")
    exp_p.add_argument("--project", help="Filter to a project")
    exp_p.add_argument("--since", help="Only show runs since this ISO date")
    exp_p.add_argument("--limit", type=int, default=50, help="Max runs to show")
    exp_p.add_argument("--compare", nargs="+", metavar="RUN_ID", help="Compare 2+ run IDs side-by-side")
    exp_p.add_argument("--detail", metavar="RUN_ID", help="Show per-item detail for a run")
    exp_p.add_argument("--format", choices=["table", "json"], default="table", help="Output format")

    # backfill
    backfill_p = sub.add_parser("backfill", help="Import JSONL logs into SQLite")
    backfill_p.add_argument("--clear", action="store_true", help="Wipe DB before importing (avoids dupes)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "cost":
        cmd_cost(args)
    elif args.command == "traces":
        cmd_traces(args)
    elif args.command == "scores":
        cmd_scores(args)
    elif args.command == "experiments":
        cmd_experiments(args)
    elif args.command == "backfill":
        cmd_backfill(args)


if __name__ == "__main__":
    main()
