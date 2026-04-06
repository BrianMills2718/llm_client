"""Logging setup utility for application entry points.

Library code must NOT call these functions — they configure the root logger
and are intended for application entry points (scripts, CLIs, batch runners).

Usage:
    from llm_client import setup_logging

    # Minimal — INFO to stderr
    setup_logging()

    # With file sink (recommended for batch runs)
    setup_logging(level="INFO", log_file="logs/batch_20260405.log")

    # Debug mode
    setup_logging(level="DEBUG")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


_DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(name)-40s %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def setup_logging(
    level: str | int = "INFO",
    log_file: str | Path | None = None,
    fmt: str = _DEFAULT_FORMAT,
    date_fmt: str = _DEFAULT_DATE_FORMAT,
    *,
    quiet_noisy_libs: bool = True,
) -> None:
    """Configure the root logger for an application entry point.

    Call once at the start of a script or CLI entry point.
    Library code must NOT call this.

    Args:
        level: Log level string ("DEBUG", "INFO", "WARNING", "ERROR") or int.
            Defaults to "INFO".
        log_file: Optional path to write logs to in addition to stderr.
            Parent directories are created automatically.
        fmt: Log format string. Defaults to timestamp + level + module + message.
        date_fmt: Date format for timestamp. Defaults to ISO 8601 (no ms).
        quiet_noisy_libs: If True (default), set noisy third-party libraries
            (httpx, httpcore, litellm, anthropic) to WARNING level so they
            don't drown out application logs.
    """
    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {level!r}")
    else:
        numeric_level = level

    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Stderr handler — always present
    if not any(isinstance(h, logging.StreamHandler) and h.stream is sys.stderr for h in root.handlers):
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setFormatter(formatter)
        root.addHandler(stderr_handler)

    # Optional file handler
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    if quiet_noisy_libs:
        for noisy in ("httpx", "httpcore", "litellm", "anthropic", "openai", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
