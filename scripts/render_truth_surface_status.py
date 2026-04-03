#!/usr/bin/env python3
"""Render a compact truth-surface status summary.

This tool can render from validator output JSON or by running the validator from
a config file. The goal is a deterministic, human-readable current-state surface
that can replace hand-maintained status prose for this slice.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_truth_surface_drift import Issue, run_checks  # noqa: E402


def _load_issue_payload(path: Path) -> list[Issue]:
    """Load issues from a validator JSON payload."""
    payload = json.loads(path.read_text())
    issues_raw = payload.get("issues", [])
    issues: list[Issue] = []
    for item in issues_raw:
        issues.append(
            Issue(
                code=item["code"],
                severity=item["severity"],
                message=item["message"],
                evidence=item.get("evidence", {}),
            )
        )
    return issues


def render_status(issues: list[Issue]) -> str:
    """Render a deterministic human-readable truth-surface summary."""
    if not issues:
        return "Truth Surface Status\n- Overall: clean\n- Issues: 0\n"

    counts = Counter(issue.severity for issue in issues)
    if counts.get("fail"):
        overall = "fail"
    elif counts.get("warn"):
        overall = "warn"
    else:
        overall = "info"

    lines = [
        "Truth Surface Status",
        f"- Overall: {overall}",
        f"- Issues: {len(issues)}",
        f"- Fail: {counts.get('fail', 0)}",
        f"- Warn: {counts.get('warn', 0)}",
        f"- Info: {counts.get('info', 0)}",
        "- Findings:",
    ]
    for issue in issues:
        lines.append(f"  - [{issue.severity.upper()}] {issue.code}: {issue.message}")
    return "\n".join(lines) + "\n"


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="Run validator from config and render the result")
    source.add_argument("--input-json", help="Render a previously captured validator JSON payload")
    parser.add_argument("--output", help="Optional output file for rendered text")
    args = parser.parse_args()

    if args.config:
        issues = run_checks(Path(args.config).expanduser())
    else:
        issues = _load_issue_payload(Path(args.input_json).expanduser())

    rendered = render_status(issues)
    if args.output:
        Path(args.output).expanduser().write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
