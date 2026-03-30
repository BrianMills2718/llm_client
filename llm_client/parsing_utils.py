"""JSON extraction and sanitization utilities for LLM response parsing.

Stolen from instructor (567-labs/instructor) per ecosystem-ops deep-dive
recommendation #5 and adapted for llm_client's needs.

LLMs produce JSON in messy ways: wrapped in markdown fences, sprinkled with
control characters, embedded in prose, or truncated mid-object.  These
utilities normalize the mess into clean, parseable JSON before it reaches
Pydantic validation.

No external dependencies -- pure stdlib (json, re).
"""

from __future__ import annotations

import json
import re
from typing import Any

# ---------------------------------------------------------------------------
# Control character stripping
# ---------------------------------------------------------------------------

# Matches C0 controls (\x00-\x1F) EXCEPT \n, \r, \t (which are valid in JSON
# strings) plus DEL (\x7F) and C1 controls (\x80-\x9F).
_CONTROL_CHAR_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
)


def strip_control_chars(text: str) -> str:
    """Remove control characters that LLMs sometimes inject into JSON output.

    Preserves \\n (0x0A), \\r (0x0D), and \\t (0x09) since those are valid
    inside JSON string values.  Strips everything else in the C0 range plus
    DEL and C1 controls.

    Args:
        text: Raw text potentially containing stray control characters.

    Returns:
        Cleaned text with offending characters removed.
    """
    return _CONTROL_CHAR_RE.sub("", text)


# ---------------------------------------------------------------------------
# JSON extraction from various response formats
# ---------------------------------------------------------------------------

# Matches ```json ... ``` or ``` ... ``` fenced blocks (greedy on the inner
# content so we capture the full block).
_FENCED_JSON_RE = re.compile(
    r"```(?:json)?\s*\n?(.*?)\n?\s*```",
    re.DOTALL,
)


def extract_json(text: str) -> str:
    """Extract a JSON object or array from text that may contain wrapping.

    Handles, in order of priority:
      1. Pure JSON string (starts with ``{`` or ``[`` after stripping)
      2. JSON wrapped in markdown code fences (````json ... `````)
      3. JSON embedded in prose (first ``{`` to last ``}``, or ``[`` to ``]``)

    For case 3, both objects and arrays are supported.  If the text contains
    both, the one whose opening delimiter appears first wins.

    Args:
        text: Raw LLM response text.

    Returns:
        Extracted JSON string.

    Raises:
        ValueError: If no JSON-like content can be located.
    """
    stripped = text.strip()

    # Case 1: already pure JSON
    if stripped and stripped[0] in ("{", "["):
        return stripped

    # Case 2: fenced code block
    match = _FENCED_JSON_RE.search(text)
    if match:
        inner = match.group(1).strip()
        if inner:
            return inner

    # Case 3: embedded in prose -- find first/last delimiters
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    first_bracket = text.find("[")
    last_bracket = text.rfind("]")

    candidates: list[tuple[int, int]] = []
    if first_brace != -1 and last_brace > first_brace:
        candidates.append((first_brace, last_brace))
    if first_bracket != -1 and last_bracket > first_bracket:
        candidates.append((first_bracket, last_bracket))

    if candidates:
        # Pick whichever delimiter appears first in the text
        start, end = min(candidates, key=lambda c: c[0])
        return text[start : end + 1]

    raise ValueError(
        f"No JSON object or array found in text (length={len(text)})"
    )


# ---------------------------------------------------------------------------
# Combined safe loader
# ---------------------------------------------------------------------------


def safe_json_loads(text: str) -> dict[str, Any] | list[Any]:
    """Parse JSON from messy LLM output with best-effort extraction.

    Pipeline:
      1. ``strip_control_chars`` -- remove stray C0/C1 bytes
      2. ``extract_json`` -- locate the JSON payload in fences/prose
      3. ``json.loads(strict=False)`` -- parse with relaxed decoder

    Args:
        text: Raw LLM response text.

    Returns:
        Parsed JSON as a dict or list.

    Raises:
        ValueError: If no JSON can be located in the text.
        json.JSONDecodeError: If located text is not valid JSON.
    """
    cleaned = strip_control_chars(text)
    extracted = extract_json(cleaned)
    return json.loads(extracted, strict=False)


# ---------------------------------------------------------------------------
# Truncation detection
# ---------------------------------------------------------------------------


class TruncatedOutputError(RuntimeError):
    """Raised when an LLM response was cut short by a token limit.

    Attributes:
        finish_reason: The raw finish_reason string from the provider.
        content_length: Length of the (possibly truncated) content.
    """

    def __init__(self, finish_reason: str, content_length: int) -> None:
        self.finish_reason = finish_reason
        self.content_length = content_length
        super().__init__(
            f"LLM output truncated (finish_reason={finish_reason!r}, "
            f"content_length={content_length}). "
            "Increase max_tokens or reduce prompt size."
        )


# finish_reason values that mean "ran out of tokens" across providers.
_TRUNCATION_REASONS = frozenset({"length", "max_tokens"})


def check_truncation(
    response: dict[str, Any],
    text: str,
) -> None:
    """Raise ``TruncatedOutputError`` if the response was cut short.

    Checks OpenAI-style ``choices[0].finish_reason == "length"`` and
    Anthropic-style ``stop_reason == "max_tokens"``.

    Args:
        response: Raw provider response dict (or object with the relevant attrs).
        text: The extracted text content (used for diagnostics).

    Raises:
        TruncatedOutputError: If the output was truncated.
    """
    finish_reason: str | None = None

    # OpenAI / LiteLLM format
    if isinstance(response, dict):
        choices = response.get("choices")
        if choices and isinstance(choices, list):
            finish_reason = choices[0].get("finish_reason")
        # Anthropic dict format
        if finish_reason is None:
            finish_reason = response.get("stop_reason")
    else:
        # Object with .choices (OpenAI SDK)
        if hasattr(response, "choices") and response.choices:
            finish_reason = getattr(response.choices[0], "finish_reason", None)
        # Object with .stop_reason (Anthropic SDK)
        if finish_reason is None:
            finish_reason = getattr(response, "stop_reason", None)

    if finish_reason and finish_reason in _TRUNCATION_REASONS:
        raise TruncatedOutputError(
            finish_reason=finish_reason,
            content_length=len(text),
        )
