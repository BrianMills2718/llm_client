"""Tests for tool result cleaning utilities (Plan 02, Slice B)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from llm_client.tool_result_cleaning import (
    clean_result,
    clean_tool_output,
    strip_html_tags,
    truncate_at_boundary,
)


class TestStripHtmlTags:
    """Verify HTML tag removal and entity decoding."""

    def test_strip_html_removes_tags(self) -> None:
        raw = "<p>Hello <b>world</b></p><div>content</div><script>alert(1)</script>"
        result = strip_html_tags(raw)
        assert "<" not in result
        assert ">" not in result
        assert "Hello" in result
        assert "world" in result
        assert "content" in result
        assert "alert(1)" in result

    def test_strip_html_decodes_entities(self) -> None:
        raw = "5 &gt; 3 &amp; 2 &lt; 4 &quot;ok&quot;"
        result = strip_html_tags(raw)
        assert result == '5 > 3 & 2 < 4 "ok"'

    def test_strip_html_handles_plain_text(self) -> None:
        plain = "No HTML here, just plain text."
        assert strip_html_tags(plain) == plain


class TestTruncateAtBoundary:
    """Verify boundary-aware truncation."""

    def test_truncate_preserves_short_text(self) -> None:
        short = "This is short."
        assert truncate_at_boundary(short, 5000) == short

    def test_truncate_at_newline_boundary(self) -> None:
        # Build text with known newline positions.
        lines = [f"Line {i}: " + "x" * 80 for i in range(200)]
        text = "\n".join(lines)
        assert len(text) > 5000

        result = truncate_at_boundary(text, 5000)
        assert len(result) <= 5000
        assert result.endswith(f"chars total]")
        # Should end at a newline boundary, so the last real line is complete.
        # Split on the truncation marker to get the content portion.
        content_part = result.split("... [truncated")[0]
        assert content_part.endswith("\n")

    def test_truncate_adds_indicator(self) -> None:
        text = "a" * 10000
        result = truncate_at_boundary(text, 5000)
        assert "truncated" in result
        assert "10000 chars total" in result

    def test_truncate_at_sentence_boundary(self) -> None:
        # Text with no newlines but sentence endings.
        text = "First sentence. " * 500
        result = truncate_at_boundary(text, 200)
        assert len(result) <= 200
        # Should have cut at a period.
        assert "truncated" in result

    def test_truncate_exact_boundary(self) -> None:
        text = "a" * 8000
        result = truncate_at_boundary(text, 8000)
        assert result == text  # Exactly at limit — no truncation.


class TestCleanToolOutput:
    """Verify the standalone clean_tool_output function."""

    def test_clean_tool_output_standalone(self) -> None:
        raw = "<p>Hello</p> " + "x" * 10000
        result = clean_tool_output(raw, max_chars=5000, strip_html=True)
        assert "<p>" not in result
        assert len(result) <= 5000

    def test_clean_tool_output_no_html_strip(self) -> None:
        raw = "<p>Hello</p>"
        result = clean_tool_output(raw, max_chars=5000, strip_html=False)
        assert "<p>" in result

    def test_clean_tool_output_short_passthrough(self) -> None:
        raw = "Short result."
        assert clean_tool_output(raw, max_chars=5000) == raw


class TestCleanResultDecoratorSync:
    """Verify the @clean_result decorator on synchronous functions."""

    def test_clean_result_decorator_sync(self) -> None:
        @clean_result(max_chars=100, strip_html=True)
        def my_tool() -> str:
            """Return a big HTML blob."""
            return "<div>" + "x" * 500 + "</div>"

        result = my_tool()
        assert "<div>" not in result
        assert len(result) <= 100

    def test_decorator_preserves_docstring(self) -> None:
        @clean_result(max_chars=100)
        def documented_tool() -> str:
            """This is the original docstring."""
            return "result"

        assert documented_tool.__doc__ == "This is the original docstring."

    def test_decorator_preserves_name(self) -> None:
        @clean_result(max_chars=100)
        def named_tool() -> str:
            """A tool."""
            return "result"

        assert named_tool.__name__ == "named_tool"

    def test_decorator_passes_through_non_string(self) -> None:
        @clean_result(max_chars=100, strip_html=True)
        def returns_dict() -> dict[str, int]:
            """Return a dict."""
            return {"count": 42}

        result = returns_dict()
        assert result == {"count": 42}


class TestCleanResultDecoratorAsync:
    """Verify the @clean_result decorator on async functions."""

    def test_clean_result_decorator_async(self) -> None:
        @clean_result(max_chars=100, strip_html=True)
        async def my_async_tool() -> str:
            """Return a big HTML blob asynchronously."""
            return "<div>" + "x" * 500 + "</div>"

        result = asyncio.run(my_async_tool())
        assert "<div>" not in result
        assert len(result) <= 100

    def test_async_decorator_preserves_docstring(self) -> None:
        @clean_result(max_chars=100)
        async def async_documented() -> str:
            """Async original docstring."""
            return "result"

        assert async_documented.__doc__ == "Async original docstring."


class TestSummarizeAndStore:
    """Verify the summarize-and-store threshold behavior."""

    def test_summarize_and_store(self, tmp_path: Path) -> None:
        @clean_result(
            max_chars=50000,
            summarize_threshold=200,
            storage_dir=tmp_path,
        )
        def big_tool() -> str:
            """Return a large result."""
            return "\n".join(f"Line {i}" for i in range(100))

        result = big_tool()
        assert "Full result stored at:" in result
        assert "Summary of full result" in result

        # Verify the file was actually created and contains the full content.
        stored_files = list(tmp_path.glob("result_*.txt"))
        assert len(stored_files) == 1
        stored_content = stored_files[0].read_text()
        assert "Line 0" in stored_content
        assert "Line 99" in stored_content

    def test_summarize_threshold_not_triggered(self, tmp_path: Path) -> None:
        @clean_result(
            max_chars=50000,
            summarize_threshold=10000,
            storage_dir=tmp_path,
        )
        def small_tool() -> str:
            """Return a small result."""
            return "Short output."

        result = small_tool()
        assert result == "Short output."
        # No file should be created.
        assert list(tmp_path.glob("result_*.txt")) == []

    def test_summarize_and_store_async(self, tmp_path: Path) -> None:
        @clean_result(
            max_chars=50000,
            summarize_threshold=200,
            storage_dir=tmp_path,
        )
        async def big_async_tool() -> str:
            """Return a large result asynchronously."""
            return "\n".join(f"Line {i}" for i in range(100))

        result = asyncio.run(big_async_tool())
        assert "Full result stored at:" in result
        stored_files = list(tmp_path.glob("result_*.txt"))
        assert len(stored_files) == 1
