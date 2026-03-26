"""Shared evidence-span resolution for structured extraction.

When LLMs extract evidence spans from source text, the quoted text often
doesn't exactly match the source due to whitespace normalization, markdown
formatting, or minor character differences.  This module provides a
resolution pipeline that tries progressively looser matching:

1. Exact match (character-for-character)
2. Whitespace-normalized match (collapse runs of whitespace)
3. Stripped match (remove leading/trailing whitespace from both)

Every project doing structured extraction with evidence grounding should
use this instead of hand-rolling exact-match-or-fail logic.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedSpan:
    """One evidence span resolved to character offsets in the source text."""

    start_char: int
    end_char: int
    text: str
    match_method: str  # "exact", "whitespace_normalized", "stripped"


def resolve_evidence_span(
    *,
    source_text: str,
    quoted_text: str,
    hint_start: int | None = None,
    hint_end: int | None = None,
) -> ResolvedSpan | None:
    """Resolve one quoted evidence span to character offsets in the source.

    Tries progressively looser matching strategies.  Returns None if no
    unique match is found (caller decides whether to fail loud or skip).

    When ``hint_start`` and ``hint_end`` are provided and the substring
    at those offsets matches exactly, they are reused without searching.
    """

    # Strategy 0: Use hints if they match exactly.
    if hint_start is not None and hint_end is not None:
        if 0 <= hint_start < hint_end <= len(source_text):
            candidate = source_text[hint_start:hint_end]
            if candidate == quoted_text:
                return ResolvedSpan(
                    start_char=hint_start,
                    end_char=hint_end,
                    text=quoted_text,
                    match_method="exact",
                )

    # Strategy 1: Exact match.
    exact_matches = _find_all(source_text, quoted_text)
    if len(exact_matches) == 1:
        start, end = exact_matches[0]
        return ResolvedSpan(
            start_char=start, end_char=end, text=quoted_text, match_method="exact",
        )
    if len(exact_matches) > 1:
        logger.debug("evidence span has %d exact matches (ambiguous): %r", len(exact_matches), quoted_text[:50])
        # Ambiguous — fall through to normalized matching which might disambiguate.

    # Strategy 2: Whitespace-normalized match.
    # Collapse runs of whitespace in both source and quoted text, then find
    # the match in the original source by tracking character positions.
    normalized_match = _whitespace_normalized_match(source_text, quoted_text)
    if normalized_match is not None:
        start, end = normalized_match
        return ResolvedSpan(
            start_char=start,
            end_char=end,
            text=source_text[start:end],
            match_method="whitespace_normalized",
        )

    # Strategy 3: Stripped match — trim both sides and retry exact.
    stripped = quoted_text.strip()
    if stripped and stripped != quoted_text:
        stripped_matches = _find_all(source_text, stripped)
        if len(stripped_matches) == 1:
            start, end = stripped_matches[0]
            return ResolvedSpan(
                start_char=start, end_char=end, text=source_text[start:end],
                match_method="stripped",
            )

    return None


def resolve_evidence_spans(
    *,
    source_text: str,
    spans: list[dict],
    text_key: str = "text",
    start_key: str = "start_char",
    end_key: str = "end_char",
    strict: bool = True,
) -> list[ResolvedSpan]:
    """Resolve a list of evidence span dicts to verified offsets.

    Each span dict should have at least a ``text_key`` field with the quoted
    text.  Optional ``start_key``/``end_key`` are used as hints.

    When ``strict`` is True (default), raises ``ValueError`` on unresolvable
    spans.  When False, skips them with a warning.
    """

    resolved: list[ResolvedSpan] = []
    for index, span in enumerate(spans):
        quoted = span.get(text_key, "")
        if not quoted or not quoted.strip():
            if strict:
                raise ValueError(f"evidence span {index} has empty text")
            continue

        result = resolve_evidence_span(
            source_text=source_text,
            quoted_text=quoted,
            hint_start=span.get(start_key),
            hint_end=span.get(end_key),
        )

        if result is None:
            if strict:
                raise ValueError(
                    f"evidence span {index} text did not resolve to a unique match in source: {quoted[:80]!r}"
                )
            logger.warning(
                "evidence span %d could not be resolved (skipped): %r",
                index,
                quoted[:80],
            )
            continue

        if result.match_method != "exact":
            logger.info(
                "evidence span %d resolved via %s: %r → %r",
                index,
                result.match_method,
                quoted[:50],
                result.text[:50],
            )

        resolved.append(result)

    return resolved


def _find_all(source: str, target: str) -> list[tuple[int, int]]:
    """Find all exact occurrences of target in source."""

    matches: list[tuple[int, int]] = []
    start = 0
    while True:
        found = source.find(target, start)
        if found < 0:
            break
        matches.append((found, found + len(target)))
        start = found + 1
    return matches


_WS_RUN = re.compile(r"\s+")


def _whitespace_normalized_match(
    source: str,
    quoted: str,
) -> tuple[int, int] | None:
    """Find a unique match after normalizing whitespace runs to single spaces.

    Returns the (start, end) offsets in the ORIGINAL source text, not the
    normalized version.  This preserves the original formatting in the
    resolved span.
    """

    norm_quoted = _WS_RUN.sub(" ", quoted.strip())
    if not norm_quoted:
        return None

    # Build a mapping from normalized positions back to original positions.
    # Walk the source character by character, tracking both original and
    # normalized positions.
    norm_chars: list[str] = []
    orig_positions: list[int] = []  # orig_positions[i] = source index for norm_chars[i]
    prev_was_space = False

    for i, ch in enumerate(source):
        if ch in (" ", "\t", "\n", "\r"):
            if not prev_was_space and norm_chars:
                norm_chars.append(" ")
                orig_positions.append(i)
            prev_was_space = True
        else:
            norm_chars.append(ch)
            orig_positions.append(i)
            prev_was_space = False

    norm_source = "".join(norm_chars)

    # Find the normalized quoted text in the normalized source.
    matches: list[tuple[int, int]] = []
    start = 0
    while True:
        found = norm_source.find(norm_quoted, start)
        if found < 0:
            break
        matches.append((found, found + len(norm_quoted)))
        start = found + 1

    if len(matches) != 1:
        return None

    norm_start, norm_end = matches[0]
    orig_start = orig_positions[norm_start]
    # For the end, we need the original position AFTER the last matched char.
    orig_end_char_pos = orig_positions[norm_end - 1]
    # Find the next non-whitespace boundary or end of source.
    orig_end = orig_end_char_pos + 1

    return (orig_start, orig_end)
