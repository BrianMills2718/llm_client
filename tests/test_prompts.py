"""Tests for prompt loading, Jinja2 rendering, and prompt asset resolution."""

import textwrap
from pathlib import Path

import pytest

from llm_client import load_prompt_asset, parse_prompt_ref, render_prompt, resolve_prompt_asset


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

    def test_requires_exactly_one_render_source(self, prompt_dir: Path) -> None:
        f = _write(
            prompt_dir / "source.yaml",
            """\
            name: source
            version: "1.0"
            messages:
              - role: user
                content: "hi"
            """,
        )
        with pytest.raises(ValueError, match="exactly one"):
            render_prompt()
        with pytest.raises(ValueError, match="exactly one"):
            render_prompt(f, prompt_ref="shared.summarize.concise@1")


class TestPromptAssets:
    """Explicit prompt asset identity and deterministic rendering."""

    def test_parse_prompt_ref(self) -> None:
        parsed = parse_prompt_ref("shared.summarize.concise@1")
        assert parsed.asset_id == "shared.summarize.concise"
        assert parsed.version == 1
        assert parsed.prompt_ref == "shared.summarize.concise@1"

    def test_resolve_prompt_asset(self) -> None:
        resolved = resolve_prompt_asset("shared.summarize.bullet@1")
        assert resolved.prompt_ref == "shared.summarize.bullet@1"
        assert resolved.manifest.id == "shared.summarize.bullet"
        assert resolved.manifest.derived_from == "shared.summarize.concise@1"
        assert resolved.template_path.name == "template.yaml"

    def test_load_prompt_asset_metadata(self) -> None:
        manifest = load_prompt_asset("shared.summarize.concise@1")
        assert manifest.status == "canonical"
        assert manifest.input_schema == "prompt_eval.text_input.v1"
        assert manifest.output_schema == "text.summary.v1"

    def test_render_prompt_from_prompt_ref(self) -> None:
        messages = render_prompt(prompt_ref="shared.summarize.concise@1", style="bullet")
        assert messages[0]["role"] == "system"
        assert "concise analyst" in messages[0]["content"]
        assert messages[1]["content"] == (
            "Summarize the following text in bullet style.\n\n{input}"
        )

    def test_render_goal_decompose_prompt_asset(self) -> None:
        messages = render_prompt(
            prompt_ref="shared.investigation_pipeline.goal_decompose@1",
            query="Investigate Acme Corp's federal contracts.",
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "structured investigation scope" in messages[0]["content"]
        assert messages[1]["content"] == (
            "Journalist's query: Investigate Acme Corp's federal contracts.\n\n"
            "Decompose this into a typed investigation scope. Be specific and actionable."
        )

    def test_load_goal_decompose_prompt_asset_metadata(self) -> None:
        manifest = load_prompt_asset("shared.investigation_pipeline.goal_decompose@1")
        assert manifest.status == "canonical"
        assert manifest.input_schema == "investigation.goal_decompose_input.v1"
        assert manifest.output_schema == "investigation.goal_scope.v1"

    def test_render_collect_prompt_asset(self) -> None:
        messages = render_prompt(
            prompt_ref="shared.investigation_pipeline.collect@1",
            original_query="Investigate Acme Corp's federal contracts.",
            scope_key_actors="Acme Corp",
            scope_key_questions="- What contracts did Acme Corp receive?",
            scope_topic_domains="financial, political",
            scope_time_range="2020-01-01/2025-01-01",
            scope_out_of_scope="competitor analysis",
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "AVAILABLE TOOLS" in messages[0]["content"]
        assert messages[1]["content"] == (
            "Investigation: Investigate Acme Corp's federal contracts.\n\n"
            "Collect evidence on the key actors and answer the key questions.\n"
            "Stay within scope. Cite every claim."
        )

    def test_load_collect_prompt_asset_metadata(self) -> None:
        manifest = load_prompt_asset("shared.investigation_pipeline.collect@1")
        assert manifest.status == "canonical"
        assert manifest.input_schema == "investigation.collect_input.v1"
        assert manifest.output_schema == "investigation.collection_synthesis.v1"

    def test_render_investigate_gap_prompt_asset(self) -> None:
        messages = render_prompt(
            prompt_ref="shared.investigation_pipeline.investigate_gap@1",
            original_query="Investigate Acme Corp's federal contracts.",
            scope_key_actors="Acme Corp",
            scope_topic_domains="financial",
            scope_out_of_scope="competitor analysis",
            facts_summary="No facts established yet.",
            entities_summary="No entities identified yet.",
            gap_description="What contracts did Acme Corp receive from DoD?",
            gap_type="open_question",
            gap_priority="high",
            target_entity="Acme Corp",
            tools_hint="search_contracts, search_filings",
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "YOUR SPECIFIC TASK" in messages[0]["content"]
        assert "Target entity: Acme Corp" in messages[0]["content"]
        assert "Suggested tools: search_contracts, search_filings" in messages[0]["content"]
        assert messages[1]["content"] == (
            "Fill this gap: What contracts did Acme Corp receive from DoD?\n\n"
            "Focus only on what is needed to answer this specific question.\n"
            "Cite every claim with its source URL or record ID."
        )

    def test_load_investigate_gap_prompt_asset_metadata(self) -> None:
        manifest = load_prompt_asset("shared.investigation_pipeline.investigate_gap@1")
        assert manifest.status == "canonical"
        assert manifest.input_schema == "investigation.gap_probe_input.v1"
        assert manifest.output_schema == "investigation.collection_synthesis.v1"

    def test_render_corroborate_extract_prompt_asset(self) -> None:
        messages = render_prompt(
            prompt_ref="shared.investigation_pipeline.corroborate_extract@1",
            claim_statement="Acme Corp received a $4M DoD contract.",
            agent_report="A USAspending award record confirms the contract.",
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "confidence_score" in messages[0]["content"]
        assert "investigative journalism workflow" in messages[0]["content"]
        assert messages[1]["content"] == (
            "Claim: Acme Corp received a $4M DoD contract.\n\n"
            "Agent report:\nA USAspending award record confirms the contract.\n\n"
            "Extract the corroboration outcome for this claim."
        )

    def test_load_corroborate_extract_prompt_asset_metadata(self) -> None:
        manifest = load_prompt_asset("shared.investigation_pipeline.corroborate_extract@1")
        assert manifest.status == "canonical"
        assert manifest.input_schema == "investigation.corroborate_extract_input.v1"
        assert manifest.output_schema == "investigation.corroborate_extract_output.v1"

    def test_render_corroborate_prompt_asset(self) -> None:
        messages = render_prompt(
            prompt_ref="shared.investigation_pipeline.corroborate@1",
            original_query="Investigate Acme Corp's federal contracts.",
            claim_statement="Acme Corp received a $4M DoD contract.",
            source_url="https://example.com/claim",
            source_type="web_search",
            source_credibility="secondary",
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "investigative journalists" in messages[0]["content"]
        assert "follow-up investigation would" in messages[0]["content"]
        assert "need to either confirm or refute it" in messages[0]["content"]
        assert messages[1]["content"] == (
            'Evaluate: "Acme Corp received a $4M DoD contract."\n'
            "Original source: https://example.com/claim\n\n"
            "Search for independent confirmation or contradiction, then report your findings."
        )

    def test_load_corroborate_prompt_asset_metadata(self) -> None:
        manifest = load_prompt_asset("shared.investigation_pipeline.corroborate@1")
        assert manifest.status == "canonical"
        assert manifest.input_schema == "investigation.corroborate_input.v1"
        assert manifest.output_schema == "investigation.corroboration_agent_report.v1"

    def test_render_native_ingest_corroborate_prompt_asset(self) -> None:
        messages = render_prompt(
            prompt_ref="research_v3_native_ingest.investigation_pipeline.corroborate@1",
            original_query="Investigate Acme Corp's federal contracts.",
            claim_statement="Acme Corp received a $4M DoD contract.",
            source_url="https://example.com/claim",
            source_type="web_search",
            source_credibility="secondary",
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "A primary government record is worth more" in messages[0]["content"]
        assert "than two aggregators citing the same filing" in messages[0]["content"]
        assert messages[1]["content"] == (
            'Evaluate: "Acme Corp received a $4M DoD contract."\n'
            "Original source: https://example.com/claim\n\n"
            "Search for independent confirmation or contradiction, then report your findings."
        )

    def test_load_native_ingest_corroborate_prompt_asset_metadata(self) -> None:
        manifest = load_prompt_asset("research_v3_native_ingest.investigation_pipeline.corroborate@1")
        assert manifest.status == "canonical"
        assert manifest.derived_from == "shared.investigation_pipeline.corroborate@1"
        assert manifest.input_schema == "investigation.corroborate_input.v1"
        assert manifest.output_schema == "investigation.corroboration_agent_report.v1"

    def test_missing_prompt_asset_fails_loud(self) -> None:
        with pytest.raises(FileNotFoundError, match="Prompt asset manifest not found"):
            render_prompt(prompt_ref="shared.summarize.missing@1")

    def test_invalid_prompt_ref_fails_loud(self) -> None:
        with pytest.raises(ValueError, match="Invalid prompt_ref format"):
            render_prompt(prompt_ref="not a valid ref")
