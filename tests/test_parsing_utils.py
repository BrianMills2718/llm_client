"""Tests for JSON extraction and sanitization utilities.

Covers:
  - strip_control_chars: C0/C1 removal, whitespace preservation
  - extract_json: pure JSON, fenced, embedded in prose, arrays
  - safe_json_loads: end-to-end pipeline
  - check_truncation: finish_reason detection across provider formats
"""

from __future__ import annotations

import json
import pytest

from llm_client.parsing_utils import (
    TruncatedOutputError,
    check_truncation,
    extract_json,
    safe_json_loads,
    strip_control_chars,
)


# =========================================================================
# strip_control_chars
# =========================================================================


class TestStripControlChars:
    """Control character removal from raw LLM text."""

    def test_removes_null_byte(self) -> None:
        assert strip_control_chars('{"a":\x00"b"}') == '{"a":"b"}'

    def test_removes_c0_controls(self) -> None:
        # \x01 (SOH), \x02 (STX), \x1f (US) should all be stripped
        raw = '{"x":\x01"y\x02z\x1f"}'
        assert "\x01" not in strip_control_chars(raw)
        assert "\x02" not in strip_control_chars(raw)
        assert "\x1f" not in strip_control_chars(raw)

    def test_preserves_newline_tab_cr(self) -> None:
        raw = '{"msg": "line1\nline2\ttab\rcarriage"}'
        cleaned = strip_control_chars(raw)
        assert "\n" in cleaned
        assert "\t" in cleaned
        assert "\r" in cleaned

    def test_removes_del_and_c1(self) -> None:
        raw = '{"a":\x7f"b\x80c\x9f"}'
        cleaned = strip_control_chars(raw)
        assert "\x7f" not in cleaned
        assert "\x80" not in cleaned
        assert "\x9f" not in cleaned

    def test_noop_on_clean_string(self) -> None:
        clean = '{"name": "Alice", "age": 30}'
        assert strip_control_chars(clean) == clean

    def test_empty_string(self) -> None:
        assert strip_control_chars("") == ""


# =========================================================================
# extract_json
# =========================================================================


class TestExtractJson:
    """JSON extraction from various LLM response formats."""

    def test_pure_json_object(self) -> None:
        raw = '{"key": "value"}'
        assert extract_json(raw) == raw

    def test_pure_json_array(self) -> None:
        raw = '[1, 2, 3]'
        assert extract_json(raw) == raw

    def test_json_with_leading_trailing_whitespace(self) -> None:
        raw = '  \n  {"key": "value"}  \n  '
        assert extract_json(raw) == '{"key": "value"}'

    def test_fenced_json(self) -> None:
        raw = '```json\n{"key": "value"}\n```'
        result = extract_json(raw)
        assert json.loads(result) == {"key": "value"}

    def test_fenced_without_lang_tag(self) -> None:
        raw = '```\n{"key": "value"}\n```'
        result = extract_json(raw)
        assert json.loads(result) == {"key": "value"}

    def test_json_embedded_in_prose(self) -> None:
        raw = 'Here is the result:\n{"name": "Bob"}\nHope this helps!'
        result = extract_json(raw)
        assert json.loads(result) == {"name": "Bob"}

    def test_array_embedded_in_prose(self) -> None:
        raw = "The items are:\n[1, 2, 3]\nThat's all."
        result = extract_json(raw)
        assert json.loads(result) == [1, 2, 3]

    def test_nested_braces(self) -> None:
        raw = 'Result: {"a": {"b": "c"}} done.'
        result = extract_json(raw)
        assert json.loads(result) == {"a": {"b": "c"}}

    def test_multiple_objects_returns_span_from_first_to_last(self) -> None:
        # extract_json uses first { to last }, which means if there are two
        # separate objects it will try to span them. This is the expected
        # (documented) behaviour — it extracts the maximal span.
        raw = '{"a": 1} some text {"b": 2}'
        result = extract_json(raw)
        # The result spans from first { to last }, which isn't valid JSON
        # on its own, but that's the documented contract. Callers should
        # use safe_json_loads which will handle this via json.loads error.
        assert result.startswith("{")
        assert result.endswith("}")

    def test_no_json_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="No JSON"):
            extract_json("This is just plain text with no JSON.")

    def test_empty_string_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="No JSON"):
            extract_json("")

    def test_fenced_json_with_prose_before(self) -> None:
        raw = "Sure! Here's the data:\n```json\n{\"x\": 42}\n```\nLet me know."
        result = extract_json(raw)
        assert json.loads(result) == {"x": 42}

    def test_object_before_array_prefers_object(self) -> None:
        raw = 'prefix {"obj": 1} and [1, 2] suffix'
        result = extract_json(raw)
        # Object appears first, so it wins
        assert result.startswith("{")

    def test_array_before_object_prefers_array(self) -> None:
        raw = "prefix [1, 2] and {'obj': 1} suffix"
        result = extract_json(raw)
        # Array appears first, so it wins
        assert result.startswith("[")


# =========================================================================
# safe_json_loads
# =========================================================================


class TestSafeJsonLoads:
    """End-to-end parsing: control chars + extraction + json.loads."""

    def test_clean_json(self) -> None:
        assert safe_json_loads('{"a": 1}') == {"a": 1}

    def test_json_with_control_chars(self) -> None:
        raw = '{"data": "hello\x00world"}'
        result = safe_json_loads(raw)
        assert result == {"data": "helloworld"}

    def test_fenced_json_with_control_chars(self) -> None:
        raw = '```json\n{"val\x01ue": "te\x02st"}\n```'
        result = safe_json_loads(raw)
        assert result == {"value": "test"}

    def test_json_with_newlines_in_strings(self) -> None:
        # json.loads(strict=False) allows unescaped newlines in strings
        raw = '{"msg": "line1\nline2"}'
        result = safe_json_loads(raw)
        assert result["msg"] == "line1\nline2"

    def test_embedded_json_with_prose(self) -> None:
        raw = "The answer is: {\"count\": 42}\nDone."
        result = safe_json_loads(raw)
        assert result == {"count": 42}

    def test_no_json_raises_valueerror(self) -> None:
        with pytest.raises(ValueError):
            safe_json_loads("no json here")

    def test_invalid_json_raises_decode_error(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            safe_json_loads("{invalid json}")

    def test_array_input(self) -> None:
        assert safe_json_loads("[1, 2, 3]") == [1, 2, 3]


# =========================================================================
# check_truncation
# =========================================================================


class TestCheckTruncation:
    """Truncation detection across provider response formats."""

    def test_openai_dict_length(self) -> None:
        response = {"choices": [{"finish_reason": "length"}]}
        with pytest.raises(TruncatedOutputError, match="truncated"):
            check_truncation(response, "partial content...")

    def test_openai_dict_stop_is_ok(self) -> None:
        response = {"choices": [{"finish_reason": "stop"}]}
        check_truncation(response, "full content")  # should not raise

    def test_anthropic_dict_max_tokens(self) -> None:
        response = {"stop_reason": "max_tokens"}
        with pytest.raises(TruncatedOutputError, match="truncated"):
            check_truncation(response, "partial...")

    def test_anthropic_dict_end_turn_is_ok(self) -> None:
        response = {"stop_reason": "end_turn"}
        check_truncation(response, "full content")  # should not raise

    def test_openai_object_format(self) -> None:
        class FakeChoice:
            finish_reason = "length"

        class FakeResponse:
            choices = [FakeChoice()]

        with pytest.raises(TruncatedOutputError):
            check_truncation(FakeResponse(), "partial...")  # type: ignore[arg-type]

    def test_anthropic_object_format(self) -> None:
        class FakeResponse:
            choices = None
            stop_reason = "max_tokens"

        with pytest.raises(TruncatedOutputError):
            check_truncation(FakeResponse(), "partial...")  # type: ignore[arg-type]

    def test_error_has_attributes(self) -> None:
        response = {"choices": [{"finish_reason": "length"}]}
        with pytest.raises(TruncatedOutputError) as exc_info:
            check_truncation(response, "some text")
        assert exc_info.value.finish_reason == "length"
        assert exc_info.value.content_length == len("some text")

    def test_empty_response_dict_is_ok(self) -> None:
        check_truncation({}, "content")  # should not raise

    def test_none_finish_reason_is_ok(self) -> None:
        response = {"choices": [{"finish_reason": None}]}
        check_truncation(response, "content")  # should not raise
