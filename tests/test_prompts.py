"""Tests for prompt loading and Jinja2 rendering."""

import textwrap
from pathlib import Path

import pytest

from llm_client import render_prompt


@pytest.fixture()
def prompt_dir(tmp_path: Path) -> Path:
    """Create a temporary prompts directory."""
    d = tmp_path / "prompts"
    d.mkdir()
    return d


def _write(path: Path, content: str) -> Path:
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


class TestRenderPrompt:
    """Core render_prompt functionality."""

    def test_simple_substitution(self, prompt_dir: Path) -> None:
        f = _write(
            prompt_dir / "simple.yaml",
            """\
            name: simple
            version: "1.0"
            messages:
              - role: system
                content: You are a helpful assistant.
              - role: user
                content: "Summarize: {{ text }}"
            """,
        )
        msgs = render_prompt(f, text="Hello world")
        assert len(msgs) == 2
        assert msgs[0] == {"role": "system", "content": "You are a helpful assistant."}
        assert msgs[1] == {"role": "user", "content": "Summarize: Hello world"}

    def test_jinja_loop(self, prompt_dir: Path) -> None:
        f = _write(
            prompt_dir / "loop.yaml",
            """\
            name: loop_test
            version: "1.0"
            messages:
              - role: user
                content: |
                  Items:
                  {% for item in items %}- {{ item }}
                  {% endfor %}
            """,
        )
        msgs = render_prompt(f, items=["alpha", "beta", "gamma"])
        assert len(msgs) == 1
        assert "- alpha" in msgs[0]["content"]
        assert "- beta" in msgs[0]["content"]
        assert "- gamma" in msgs[0]["content"]

    def test_jinja_conditional(self, prompt_dir: Path) -> None:
        f = _write(
            prompt_dir / "cond.yaml",
            """\
            name: cond_test
            version: "1.0"
            messages:
              - role: user
                content: |
                  {% if verbose %}Detailed analysis:{% else %}Brief:{% endif %}
                  {{ text }}
            """,
        )
        msgs_verbose = render_prompt(f, verbose=True, text="data")
        assert "Detailed analysis:" in msgs_verbose[0]["content"]

        msgs_brief = render_prompt(f, verbose=False, text="data")
        assert "Brief:" in msgs_brief[0]["content"]

    def test_nested_object_access(self, prompt_dir: Path) -> None:
        f = _write(
            prompt_dir / "nested.yaml",
            """\
            name: nested
            version: "1.0"
            messages:
              - role: user
                content: "Name: {{ hypothesis.label }}, Score: {{ hypothesis.score }}"
            """,
        )

        class H:
            label = "H1"
            score = 0.85

        msgs = render_prompt(f, hypothesis=H())
        assert msgs[0]["content"] == "Name: H1, Score: 0.85"

    def test_dict_context(self, prompt_dir: Path) -> None:
        f = _write(
            prompt_dir / "dict.yaml",
            """\
            name: dict
            version: "1.0"
            messages:
              - role: user
                content: "{{ data.key }}"
            """,
        )
        msgs = render_prompt(f, data={"key": "value"})
        assert msgs[0]["content"] == "value"

    def test_extra_yaml_fields_ignored(self, prompt_dir: Path) -> None:
        """Extra fields like name, version, description don't break anything."""
        f = _write(
            prompt_dir / "extra.yaml",
            """\
            name: extra_fields
            version: "2.1"
            description: This has extra metadata
            author: brian
            messages:
              - role: user
                content: hello
            """,
        )
        msgs = render_prompt(f)
        assert msgs[0]["content"] == "hello"

    def test_multiline_content_stripped(self, prompt_dir: Path) -> None:
        """Leading/trailing whitespace on rendered content is stripped."""
        f = _write(
            prompt_dir / "whitespace.yaml",
            """\
            name: ws
            version: "1.0"
            messages:
              - role: user
                content: |

                  Hello

            """,
        )
        msgs = render_prompt(f)
        assert msgs[0]["content"] == "Hello"


class TestRenderPromptErrors:
    """Error cases — fail loud."""

    def test_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            render_prompt("/nonexistent/prompt.yaml")

    def test_missing_messages_key(self, prompt_dir: Path) -> None:
        f = _write(
            prompt_dir / "no_msgs.yaml",
            """\
            name: broken
            content: this is wrong
            """,
        )
        with pytest.raises(ValueError, match="missing 'messages'"):
            render_prompt(f)

    def test_messages_not_a_list(self, prompt_dir: Path) -> None:
        f = _write(
            prompt_dir / "bad_msgs.yaml",
            """\
            name: broken
            messages:
              role: user
              content: oops
            """,
        )
        with pytest.raises(ValueError, match="must be a list"):
            render_prompt(f)

    def test_message_missing_role(self, prompt_dir: Path) -> None:
        f = _write(
            prompt_dir / "no_role.yaml",
            """\
            name: broken
            messages:
              - content: no role here
            """,
        )
        with pytest.raises(ValueError, match="'role' and 'content'"):
            render_prompt(f)

    def test_message_missing_content(self, prompt_dir: Path) -> None:
        f = _write(
            prompt_dir / "no_content.yaml",
            """\
            name: broken
            messages:
              - role: user
            """,
        )
        with pytest.raises(ValueError, match="'role' and 'content'"):
            render_prompt(f)

    def test_undefined_variable_fails_loud(self, prompt_dir: Path) -> None:
        """StrictUndefined means missing vars raise, not silently empty."""
        f = _write(
            prompt_dir / "undef.yaml",
            """\
            name: strict
            version: "1.0"
            messages:
              - role: user
                content: "{{ missing_var }}"
            """,
        )
        with pytest.raises(Exception, match="missing_var"):
            render_prompt(f)

    def test_yaml_not_a_mapping(self, prompt_dir: Path) -> None:
        f = _write(prompt_dir / "scalar.yaml", "just a string\n")
        with pytest.raises(ValueError, match="must be a mapping"):
            render_prompt(f)
