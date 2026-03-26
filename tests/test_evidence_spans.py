"""Tests for shared evidence span resolution."""

from llm_client.utils.evidence_spans import resolve_evidence_span, resolve_evidence_spans, ResolvedSpan


def test_exact_match():
    result = resolve_evidence_span(
        source_text="Gen. Charles R. Holland served as Commander.",
        quoted_text="Gen. Charles R. Holland",
    )
    assert result is not None
    assert result.match_method == "exact"
    assert result.text == "Gen. Charles R. Holland"
    assert result.start_char == 0
    assert result.end_char == 23


def test_hint_offsets_reused_when_exact():
    result = resolve_evidence_span(
        source_text="ABCDEF",
        quoted_text="CD",
        hint_start=2,
        hint_end=4,
    )
    assert result is not None
    assert result.match_method == "exact"
    assert result.start_char == 2


def test_whitespace_normalized_match():
    """Markdown table cells have extra padding spaces."""
    source = "| 2000–2003    | Gen. Charles R. Holland        | USAF          |"
    result = resolve_evidence_span(
        source_text=source,
        quoted_text="Gen. Charles R. Holland",
    )
    assert result is not None
    assert result.text == "Gen. Charles R. Holland"


def test_whitespace_normalized_multiline():
    """LLM collapses newlines to spaces."""
    source = "The 4th PSYOP Group,\nas the principal Army unit"
    result = resolve_evidence_span(
        source_text=source,
        quoted_text="The 4th PSYOP Group, as the principal Army unit",
    )
    assert result is not None
    assert result.match_method == "whitespace_normalized"


def test_stripped_match():
    result = resolve_evidence_span(
        source_text="Hello World",
        quoted_text="  Hello World  ",
    )
    assert result is not None
    assert result.match_method in ("whitespace_normalized", "stripped")


def test_no_match_returns_none():
    result = resolve_evidence_span(
        source_text="The quick brown fox",
        quoted_text="lazy dog",
    )
    assert result is None


def test_resolve_multiple_spans():
    source = "Gen. Holland commanded USSOCOM from 2000 to 2003."
    spans = [
        {"text": "Gen. Holland", "start_char": 0, "end_char": 12},
        {"text": "USSOCOM"},
        {"text": "2000 to 2003"},
    ]
    results = resolve_evidence_spans(source_text=source, spans=spans)
    assert len(results) == 3
    assert results[0].match_method == "exact"
    assert results[1].text == "USSOCOM"
    assert results[2].text == "2000 to 2003"


def test_strict_mode_raises_on_unresolvable():
    import pytest
    with pytest.raises(ValueError, match="did not resolve"):
        resolve_evidence_spans(
            source_text="Hello",
            spans=[{"text": "Goodbye"}],
            strict=True,
        )


def test_non_strict_skips_unresolvable():
    results = resolve_evidence_spans(
        source_text="Hello",
        spans=[{"text": "Goodbye"}],
        strict=False,
    )
    assert len(results) == 0


def test_markdown_table_real_case():
    """Real case from onto-canon6 USSOCOM commanders table."""
    source = (
        "#### Table 2: USSOCOM Commanders (2001–2015)\n\n"
        "| Years        | Commander                       | Service Branch |\n"
        "|--------------|--------------------------------|---------------|\n"
        "| 2000–2003    | Gen. Charles R. Holland        | USAF          |\n"
        "| 2003–2007    | Gen. Bryan D. Brown            | USA           |"
    )
    result = resolve_evidence_span(
        source_text=source,
        quoted_text="Gen. Charles R. Holland",
    )
    assert result is not None
    assert "Gen. Charles R. Holland" in result.text
