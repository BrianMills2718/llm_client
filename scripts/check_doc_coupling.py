#!/usr/bin/env python3
"""Compatibility wrapper for doc-coupling checks.

Preferred target:
- scripts/meta/check_doc_coupling.py

Fallback target:
- meta-process/scripts/check_doc_coupling.py
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TARGETS = (
    ROOT / "scripts" / "meta" / "check_doc_coupling.py",
    ROOT / "meta-process" / "scripts" / "check_doc_coupling.py",
)


if __name__ == "__main__":
    for target in TARGETS:
        if target.exists():
            runpy.run_path(str(target), run_name="__main__")
            raise SystemExit(0)
    print(
        "Missing doc-coupling checker. Expected one of:\n"
        f"- {TARGETS[0]}\n"
        f"- {TARGETS[1]}",
        file=sys.stderr,
    )
    raise SystemExit(2)
