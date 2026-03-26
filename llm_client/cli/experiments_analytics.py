"""Extended analytics commands for the experiments CLI.

Provides: breakdown, degradation, tool-analytics, failure-patterns, trace-diff.
Imported and dispatched from experiments.py.
"""

from __future__ import annotations

import argparse
import json
from typing import Any


def _get_item_extra(item: dict[str, Any]) -> dict[str, Any]:
    """Extract and parse the extra field from an experiment item."""
    extra = item.get("extra") or {}
    if isinstance(extra, str):
        try:
            extra = json.loads(extra)
        except json.JSONDecodeError:
            extra = {}
    return extra


def _get_item_metrics(item: dict[str, Any]) -> dict[str, Any]:
    """Extract and parse the metrics field."""
    m = item.get("metrics") or {}
    if isinstance(m, str):
        try:
            m = json.loads(m)
        except json.JSONDecodeError:
            m = {}
    return m


def cmd_trace_diff(args: argparse.Namespace) -> None:
    """Side-by-side trace comparison for the same item across two runs."""
    import llm_client.io_log as io_log
    from llm_client.cli.experiments import _render_conversation_trace

    item_id, run_a, run_b = args.trace_diff

    for label, run_id in [("A", run_a), ("B", run_b)]:
        items = io_log.get_run_items(run_id)
        item = next((i for i in items if i["item_id"] == item_id), None)
        if item is None:
            print(f"Item '{item_id}' not found in run {run_id}")
            return

        m = _get_item_metrics(item)
        status = "PASS" if m.get("llm_em") else "FAIL"
        print(f"{'='*60}")
        print(f"Run {label}: {run_id}")
        print(f"  Status: {status}  EM={m.get('em')}  LLM_EM={m.get('llm_em')}  F1={m.get('f1')}")
        print(f"  Pred: {str(item.get('predicted', ''))[:80]}")
        print(f"  Cost: ${item.get('cost', 0):.4f}  Tools: {item.get('n_tool_calls', 0)}")
        print()

        extra = _get_item_extra(item)
        trace = extra.get("conversation_trace")
        if trace and isinstance(trace, list):
            _render_conversation_trace(trace)
        else:
            tool_calls = extra.get("tool_calls", [])
            if tool_calls:
                for i, tc in enumerate(tool_calls):
                    name = tc if isinstance(tc, str) else tc.get("tool", "?")
                    print(f"  [{i}] {name}")
            else:
                print("  (no trace data)")
        print()


def cmd_breakdown(args: argparse.Namespace) -> None:
    """Show accuracy breakdown by item_id prefix (e.g. 2hop/3hop/4hop)."""
    import llm_client.io_log as io_log

    run_id = args.breakdown
    items = io_log.get_run_items(run_id)
    if not items:
        print(f"No items found for run {run_id}")
        return

    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        prefix = item["item_id"].split("__")[0] if "__" in item["item_id"] else "other"
        groups.setdefault(prefix, []).append(item)

    print(f"Breakdown for run {run_id} ({len(items)} items):")
    print()
    print(f"{'Group':<12} {'N':>4} {'EM%':>6} {'LLM%':>6} {'F1%':>6} {'Avg$':>7} {'AvgTools':>9}")
    print("-" * 56)

    for prefix in sorted(groups.keys()):
        group = groups[prefix]
        n = len(group)
        metrics = [_get_item_metrics(i) for i in group]
        em = sum(1 for m in metrics if m.get("em")) / n * 100
        llm_em = sum(1 for m in metrics if m.get("llm_em")) / n * 100
        f1 = sum(float(m.get("f1") or 0) for m in metrics) / n * 100
        avg_cost = sum(i.get("cost") or 0 for i in group) / n
        avg_tools = sum(i.get("n_tool_calls") or 0 for i in group) / n
        print(f"{prefix:<12} {n:>4} {em:>5.0f}% {llm_em:>5.0f}% {f1:>5.0f}% ${avg_cost:>6.2f} {avg_tools:>9.1f}")

    n = len(items)
    all_m = [_get_item_metrics(i) for i in items]
    em = sum(1 for m in all_m if m.get("em")) / n * 100
    llm_em = sum(1 for m in all_m if m.get("llm_em")) / n * 100
    f1 = sum(float(m.get("f1") or 0) for m in all_m) / n * 100
    avg_cost = sum(i.get("cost") or 0 for i in items) / n
    avg_tools = sum(i.get("n_tool_calls") or 0 for i in items) / n
    print("-" * 56)
    print(f"{'TOTAL':<12} {n:>4} {em:>5.0f}% {llm_em:>5.0f}% {f1:>5.0f}% ${avg_cost:>6.2f} {avg_tools:>9.1f}")


def cmd_degradation(args: argparse.Namespace) -> None:
    """Show rolling accuracy within a run to detect performance degradation."""
    import llm_client.io_log as io_log

    run_id = args.degradation
    items = io_log.get_run_items(run_id)
    if not items:
        print(f"No items found for run {run_id}")
        return

    items.sort(key=lambda x: x.get("timestamp") or "")

    print(f"Degradation analysis for run {run_id} ({len(items)} items):")
    print()

    window = 10
    print(f"{'Position':<10} {'Window':>8} {'EM%':>6} {'LLM%':>6} {'Avg$':>7} {'AvgTools':>9}")
    print("-" * 50)

    for i in range(0, len(items), window):
        chunk = items[i:i + window]
        n = len(chunk)
        metrics = [_get_item_metrics(it) for it in chunk]
        em = sum(1 for m in metrics if m.get("em")) / n * 100
        llm_em = sum(1 for m in metrics if m.get("llm_em")) / n * 100
        avg_cost = sum(it.get("cost") or 0 for it in chunk) / n
        avg_tools = sum(it.get("n_tool_calls") or 0 for it in chunk) / n
        print(f"q{i+1:>3}-{i+n:<4} {n:>5}q   {em:>5.0f}% {llm_em:>5.0f}% ${avg_cost:>6.2f} {avg_tools:>9.1f}")


def cmd_tool_analytics(args: argparse.Namespace) -> None:
    """Show tool usage frequency and pass/fail correlation for a run."""
    import llm_client.io_log as io_log

    run_id = args.tool_analytics
    items = io_log.get_run_items(run_id)
    if not items:
        print(f"No items found for run {run_id}")
        return

    tool_pass: dict[str, int] = {}
    tool_fail: dict[str, int] = {}
    tool_total_calls: dict[str, int] = {}

    for item in items:
        extra = _get_item_extra(item)
        m = _get_item_metrics(item)
        passed = bool(m.get("llm_em"))

        tool_names = extra.get("tool_calls", [])
        seen_tools: set[str] = set()
        for tc in tool_names:
            name = tc if isinstance(tc, str) else (tc.get("tool", "?") if isinstance(tc, dict) else "?")
            tool_total_calls[name] = tool_total_calls.get(name, 0) + 1
            seen_tools.add(name)

        for name in seen_tools:
            if passed:
                tool_pass[name] = tool_pass.get(name, 0) + 1
            else:
                tool_fail[name] = tool_fail.get(name, 0) + 1

    all_tools = sorted(set(list(tool_pass.keys()) + list(tool_fail.keys())))

    print(f"Tool analytics for run {run_id} ({len(items)} items):")
    print()
    print(f"{'Tool':<35} {'Calls':>6} {'Pass':>5} {'Fail':>5} {'PassRate':>9}")
    print("-" * 65)

    for tool in all_tools:
        calls = tool_total_calls.get(tool, 0)
        p = tool_pass.get(tool, 0)
        f = tool_fail.get(tool, 0)
        total = p + f
        rate = p / total * 100 if total > 0 else 0
        print(f"{tool:<35} {calls:>6} {p:>5} {f:>5} {rate:>8.0f}%")

    pass_only = {t for t in all_tools if tool_pass.get(t, 0) > 0 and tool_fail.get(t, 0) == 0}
    fail_only = {t for t in all_tools if tool_fail.get(t, 0) > 0 and tool_pass.get(t, 0) == 0}
    if pass_only:
        print(f"\nTools used ONLY by passing questions: {sorted(pass_only)}")
    if fail_only:
        print(f"Tools used ONLY by failing questions: {sorted(fail_only)}")


def cmd_failure_patterns(args: argparse.Namespace) -> None:
    """Find common patterns in failing traces."""
    import llm_client.io_log as io_log

    run_id = args.failure_patterns
    items = io_log.get_run_items(run_id)
    if not items:
        print(f"No items found for run {run_id}")
        return

    fails = [item for item in items if not _get_item_metrics(item).get("llm_em")]

    if not fails:
        print(f"No failures in run {run_id}")
        return

    print(f"Failure patterns for run {run_id} ({len(fails)}/{len(items)} failed):")
    print()

    categories: dict[str, list[str]] = {}
    for item in fails:
        extra = _get_item_extra(item)
        pred = str(item.get("predicted", ""))
        error = str(item.get("error", ""))
        tool_count = item.get("n_tool_calls", 0)
        m = _get_item_metrics(item)
        f1 = float(m.get("f1") or 0)

        if "ValueError" in error or "ValueError" in pred:
            cat = "VALIDATION_ERROR"
        elif "APIError" in error or "APIError" in pred:
            cat = "PROVIDER_ERROR"
        elif "cannot" in pred.lower() or "not available" in pred.lower() or "not found" in pred.lower():
            cat = "NO_ANSWER"
        elif tool_count >= 70:
            cat = "TOOL_SPIRAL"
        elif tool_count <= 5:
            cat = "GAVE_UP_EARLY"
        elif f1 > 0.3:
            cat = "CLOSE_MISS"
        else:
            cat = "WRONG_ANSWER"

        categories.setdefault(cat, []).append(item["item_id"])

    for cat in sorted(categories.keys()):
        ids = categories[cat]
        print(f"  {cat} ({len(ids)}):")
        for iid in ids[:5]:
            item = next(i for i in fails if i["item_id"] == iid)
            gold = str(item.get("gold", ""))[:30]
            pred = str(item.get("predicted", ""))[:30]
            tools = item.get("n_tool_calls", 0)
            print(f"    {iid[:40]:<40s} t={tools:>3} gold={gold:<30s} pred={pred}")
        if len(ids) > 5:
            print(f"    ... and {len(ids) - 5} more")
        print()

    # Tool chain analysis
    print("Common last-3 tool sequences in failures:")
    sequences: dict[str, int] = {}
    for item in fails:
        extra = _get_item_extra(item)
        tool_calls = extra.get("tool_calls", [])
        names = []
        for tc in tool_calls:
            if isinstance(tc, str):
                names.append(tc)
            elif isinstance(tc, dict):
                names.append(tc.get("tool", "?"))
        if len(names) >= 3:
            last3 = " -> ".join(names[-3:])
            sequences[last3] = sequences.get(last3, 0) + 1

    for seq, count in sorted(sequences.items(), key=lambda x: -x[1])[:5]:
        print(f"  ({count}x) {seq}")


def cmd_interventions(args: argparse.Namespace) -> None:
    """List logged interventions."""
    import llm_client.io_log as io_log

    interventions = io_log.get_interventions(
        project=getattr(args, "project", None),
        dataset=getattr(args, "dataset", None),
        limit=50,
    )

    if not interventions:
        print("No interventions logged.")
        return

    print(f"Interventions ({len(interventions)}):")
    print()
    print(f"{'ID':<14} {'Date':<12} {'Cat':<8} {'Status':<10} {'Description':<40} {'Impact'}")
    print("-" * 100)

    for i in interventions:
        ts = i.get("timestamp", "")[:10]
        cat = i.get("category", "?")[:7]
        status = i.get("status", "?")[:9]
        desc = (i.get("description", "") or "")[:39]
        impact = (i.get("measured_impact", "") or i.get("expected_impact", "") or "")[:30]
        iid = i.get("intervention_id", "?")
        print(f"{iid:<14} {ts:<12} {cat:<8} {status:<10} {desc:<40} {impact}")

    # Show detail for the most recent
    latest = interventions[0]
    print(f"\nLatest: {latest.get('description', '')}")
    print(f"  Problem: {latest.get('problem', '')[:120]}")
    print(f"  Fix: {latest.get('fix', '')[:120]}")
    if latest.get("baseline_run_id"):
        print(f"  Baseline run: {latest['baseline_run_id']}")
    if latest.get("verification_run_id"):
        print(f"  Verification run: {latest['verification_run_id']}")
    if latest.get("affected_items"):
        items = latest["affected_items"]
        if isinstance(items, list):
            print(f"  Affected items: {', '.join(items[:5])}")


def cmd_log_intervention(args: argparse.Namespace) -> None:
    """Log a new intervention from CLI args."""
    import llm_client.io_log as io_log

    iid = io_log.log_intervention(
        description=args.log_intervention[0],
        problem=args.log_intervention[1],
        fix=args.log_intervention[2],
        category=args.intervention_category or "infra",
        dataset=getattr(args, "dataset", None),
        baseline_run_id=args.intervention_baseline or None,
        verification_run_id=args.intervention_verify or None,
        measured_impact=args.intervention_impact or None,
        status="verified" if args.intervention_verify else "proposed",
    )
    print(f"Intervention logged: {iid}")


def register_args(parser: argparse.ArgumentParser) -> None:
    """Register analytics-specific CLI arguments."""
    parser.add_argument(
        "--interventions",
        action="store_true",
        help="List logged interventions (changes and their measured impact)",
    )
    parser.add_argument(
        "--log-intervention",
        nargs=3,
        metavar=("DESCRIPTION", "PROBLEM", "FIX"),
        help="Log a new intervention: description, problem diagnosed, fix applied",
    )
    parser.add_argument("--intervention-category", help="Category: prompt|tool|infra|config|graph|model")
    parser.add_argument("--intervention-baseline", help="Baseline run_id (before fix)")
    parser.add_argument("--intervention-verify", help="Verification run_id (after fix)")
    parser.add_argument("--intervention-impact", help="Measured impact string")
    parser.add_argument(
        "--trace-diff",
        nargs=3,
        metavar=("ITEM_ID", "RUN_ID_A", "RUN_ID_B"),
        help="Side-by-side trace comparison for the same item across two runs",
    )
    parser.add_argument(
        "--breakdown",
        metavar="RUN_ID",
        help="Show accuracy breakdown by item_id prefix (e.g. 2hop/3hop/4hop)",
    )
    parser.add_argument(
        "--degradation",
        metavar="RUN_ID",
        help="Show rolling accuracy within a run to detect performance degradation",
    )
    parser.add_argument(
        "--tool-analytics",
        metavar="RUN_ID",
        help="Show tool usage frequency and pass/fail correlation",
    )
    parser.add_argument(
        "--failure-patterns",
        metavar="RUN_ID",
        help="Find common patterns in failing traces",
    )


def dispatch(args: argparse.Namespace) -> bool:
    """Try to dispatch an analytics command. Returns True if handled."""
    if getattr(args, "interventions", False):
        cmd_interventions(args)
        return True
    if getattr(args, "log_intervention", None):
        cmd_log_intervention(args)
        return True
    if getattr(args, "trace_diff", None):
        cmd_trace_diff(args)
        return True
    if getattr(args, "breakdown", None):
        cmd_breakdown(args)
        return True
    if getattr(args, "degradation", None):
        cmd_degradation(args)
        return True
    if getattr(args, "tool_analytics", None):
        cmd_tool_analytics(args)
        return True
    if getattr(args, "failure_patterns", None):
        cmd_failure_patterns(args)
        return True
    return False
