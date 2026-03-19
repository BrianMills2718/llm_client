#!/usr/bin/env python3
"""Compatibility wrapper for the repo-local doc-coupling checker."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "meta" / "check_doc_coupling.py"),
        run_name="__main__",
    )
