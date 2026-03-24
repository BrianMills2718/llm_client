"""Tool result cleaning utilities for context management.

Provides decorators and functions to clean tool results before they enter
an agent's context window. The guiding principle: raw API responses are
never acceptable in context. Clean, cap, and summarize.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import html
import re
import tempfile
from pathlib import Path
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# Pattern for stripping HTML tags — intentionally simple (no BeautifulSoup dependency).
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def strip_html_tags(text: str) -> str:
    """Strip HTML tags and decode HTML entities from text.

    Uses regex-based tag removal and ``html.unescape`` for entity decoding.
    This is intentionally lightweight — no BeautifulSoup or trafilatura
    dependency. Handles common cases (``<p>``, ``<div>``, ``<script>`` blocks,
    ``&amp;``, ``&lt;``, etc.) but is not a full HTML parser.
    """
    cleaned = _HTML_TAG_RE.sub("", text)
    return html.unescape(cleaned)


def truncate_at_boundary(text: str, max_chars: int) -> str:
    """Truncate text at the last clean boundary before *max_chars*.

    Tries to break at a newline first, then at sentence-ending punctuation
    (``"."``, ``"!"``, ``"?"``). If neither is found in the last 20% of the
    allowed window, falls back to a hard cut at *max_chars*.

    Text shorter than *max_chars* is returned unchanged.
    """
    if len(text) <= max_chars:
        return text

    original_len = len(text)
    suffix = f"... [truncated, {original_len} chars total]"

    # Reserve space for the suffix so the total output stays within a
    # reasonable size.  The suffix itself is short enough that this won't
    # eat a meaningful chunk of the budget.
    budget = max_chars - len(suffix)
    if budget <= 0:
        # Degenerate case: max_chars is tiny — just return the suffix.
        return suffix

    window = text[:budget]

    # Try newline boundary within the last 20% of the budget.
    search_start = max(0, budget - budget // 5)
    newline_pos = window.rfind("\n", search_start)
    if newline_pos > 0:
        return window[:newline_pos] + "\n" + suffix

    # Try sentence boundary (., !, ?) within the last 20%.
    for char in (".", "!", "?"):
        pos = window.rfind(char, search_start)
        if pos > 0:
            return window[: pos + 1] + " " + suffix

    # Hard cut.
    return window + suffix


def _summarize_and_store(
    text: str,
    storage_dir: Path | None,
    summary_prefix: str,
    summary_lines: int = 20,
) -> str:
    """Write *text* to a file and return a summary with the file path.

    The summary contains the first *summary_lines* lines of the original
    text plus a pointer to the full stored file.
    """
    if storage_dir is None:
        storage_dir = Path(tempfile.mkdtemp(prefix="tool_results_"))
    else:
        storage_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic-ish filename from content hash prefix.
    content_hash = hashlib.sha256(text.encode()).hexdigest()[:12]
    filename = f"result_{content_hash}.txt"
    filepath = storage_dir / filename
    filepath.write_text(text, encoding="utf-8")

    lines = text.split("\n")
    preview = "\n".join(lines[:summary_lines])
    total_lines = len(lines)
    total_chars = len(text)

    return (
        f"{summary_prefix} ({total_chars} chars, {total_lines} lines):\n"
        f"{preview}\n"
        f"... [{total_lines - summary_lines} more lines]\n"
        f"Full result stored at: {filepath}"
    )


def clean_tool_output(
    text: str,
    *,
    max_chars: int = 8000,
    strip_html: bool = False,
) -> str:
    """Clean a tool output string. Standalone function for non-decorator use.

    Applies HTML stripping (if requested) then truncation. This is the
    simple path — no summarize-and-store, no file I/O.

    Args:
        text: Raw tool output.
        max_chars: Maximum characters in returned result.
        strip_html: If True, strip HTML tags and decode entities first.

    Returns:
        Cleaned text, truncated if necessary.
    """
    if strip_html:
        text = strip_html_tags(text)
    return truncate_at_boundary(text, max_chars)


def clean_result(
    *,
    max_chars: int = 8000,
    strip_html: bool = False,
    summarize_threshold: int | None = None,
    summary_prefix: str = "Summary of full result",
    storage_dir: Path | None = None,
) -> Callable[[F], F]:
    """Decorator that cleans tool results before they enter agent context.

    Args:
        max_chars: Maximum characters in returned result. Truncates at boundary.
        strip_html: If True, strip HTML tags and decode entities.
        summarize_threshold: If set and result exceeds this many chars, store
            full result to file and return summary + file path instead.
        summary_prefix: Prefix for the summary when threshold is exceeded.
        storage_dir: Directory for storing full results. Defaults to tempdir.

    The decorator preserves the original function's signature and docstring.
    Works on both sync and async functions.
    """

    def _apply_cleaning(result: Any) -> Any:
        """Apply the cleaning pipeline to a tool result.

        Non-string results are passed through unchanged — cleaning only
        applies to string tool outputs.
        """
        if not isinstance(result, str):
            return result

        text = result

        if strip_html:
            text = strip_html_tags(text)

        if summarize_threshold is not None and len(text) > summarize_threshold:
            return _summarize_and_store(text, storage_dir, summary_prefix)

        return truncate_at_boundary(text, max_chars)

    def decorator(fn: F) -> F:
        """Wrap *fn* so its return value is cleaned before reaching the caller."""
        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                result = await fn(*args, **kwargs)
                return _apply_cleaning(result)

            return async_wrapper  # type: ignore[return-value]
        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                result = fn(*args, **kwargs)
                return _apply_cleaning(result)

            return sync_wrapper  # type: ignore[return-value]

    return decorator
