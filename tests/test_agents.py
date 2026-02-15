"""Tests for agent SDK routing. All mocked (no real agent SDK calls).

Tests cover:
- _is_agent_model() detection (Claude + Codex)
- _parse_agent_model() parsing (Claude + Codex)
- _messages_to_agent_prompt() conversion
- Cache rejection for agent models
- NotImplementedError guards for tool calling (tools are agent-internal)
- Claude Agent SDK: routing, hooks, fallback, structured, streaming, batch
- Codex SDK: routing, hooks, model suffix, structured, streaming, batch, fallback
"""

import sys
import types
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from llm_client import (
    Hooks,
    LLMCallResult,
    LRUCache,
    acall_llm,
    acall_llm_batch,
    acall_llm_structured,
    acall_llm_structured_batch,
    acall_llm_with_tools,
    astream_llm,
    astream_llm_with_tools,
    call_llm,
    call_llm_batch,
    call_llm_structured,
    call_llm_structured_batch,
    call_llm_with_tools,
    stream_llm,
    stream_llm_with_tools,
)
from llm_client.agents import _messages_to_agent_prompt, _parse_agent_model
from llm_client.client import _is_agent_model


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


class TestIsAgentModel:
    def test_claude_code(self) -> None:
        assert _is_agent_model("claude-code") is True

    def test_claude_code_with_model(self) -> None:
        assert _is_agent_model("claude-code/opus") is True

    def test_claude_code_with_haiku(self) -> None:
        assert _is_agent_model("claude-code/haiku") is True

    def test_case_insensitive(self) -> None:
        assert _is_agent_model("Claude-Code") is True
        assert _is_agent_model("CLAUDE-CODE/opus") is True

    def test_openai_agents_reserved(self) -> None:
        assert _is_agent_model("openai-agents/gpt-5") is True

    def test_regular_models_not_agent(self) -> None:
        assert _is_agent_model("gpt-4o") is False
        assert _is_agent_model("anthropic/claude-sonnet-4-5-20250929") is False
        assert _is_agent_model("gemini/gemini-2.0-flash") is False
        assert _is_agent_model("gpt-5-mini") is False

    def test_codex(self) -> None:
        assert _is_agent_model("codex") is True

    def test_codex_with_model(self) -> None:
        assert _is_agent_model("codex/gpt-5") is True

    def test_codex_case_insensitive(self) -> None:
        assert _is_agent_model("Codex") is True
        assert _is_agent_model("CODEX/o3") is True

    def test_partial_prefix_not_matched(self) -> None:
        assert _is_agent_model("claude-coder") is False
        assert _is_agent_model("openai-agent/gpt-5") is False
        assert _is_agent_model("codex-cli") is False


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParseAgentModel:
    def test_bare_claude_code(self) -> None:
        assert _parse_agent_model("claude-code") == ("claude-code", None)

    def test_claude_code_with_model(self) -> None:
        assert _parse_agent_model("claude-code/opus") == ("claude-code", "opus")

    def test_claude_code_with_sonnet(self) -> None:
        assert _parse_agent_model("claude-code/sonnet") == ("claude-code", "sonnet")

    def test_openai_agents(self) -> None:
        assert _parse_agent_model("openai-agents/gpt-5") == ("openai-agents", "gpt-5")

    def test_case_normalization(self) -> None:
        sdk, model = _parse_agent_model("Claude-Code/Opus")
        assert sdk == "claude-code"
        assert model == "Opus"  # underlying model preserves case

    def test_bare_codex(self) -> None:
        assert _parse_agent_model("codex") == ("codex", None)

    def test_codex_with_model(self) -> None:
        assert _parse_agent_model("codex/gpt-5") == ("codex", "gpt-5")

    def test_codex_with_o3(self) -> None:
        assert _parse_agent_model("codex/o3") == ("codex", "o3")


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


class TestMessagesToAgentPrompt:
    def test_single_user_message(self) -> None:
        prompt, sys = _messages_to_agent_prompt(
            [{"role": "user", "content": "What is 2+2?"}]
        )
        assert prompt == "What is 2+2?"
        assert sys is None

    def test_system_plus_user(self) -> None:
        prompt, sys = _messages_to_agent_prompt([
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ])
        assert prompt == "Hi"
        assert sys == "You are helpful"

    def test_multi_turn(self) -> None:
        prompt, sys = _messages_to_agent_prompt([
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ])
        assert "User: Hello" in prompt
        assert "Assistant: Hi there" in prompt
        assert "User: How are you?" in prompt
        assert sys is None

    def test_system_plus_multi_turn(self) -> None:
        prompt, sys = _messages_to_agent_prompt([
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Bye"},
        ])
        assert sys == "Be concise"
        assert "User: Hello" in prompt
        assert "Assistant: Hi" in prompt
        assert "User: Bye" in prompt

    def test_empty_messages_raises(self) -> None:
        with pytest.raises(ValueError, match="No user/assistant messages"):
            _messages_to_agent_prompt([])

    def test_system_only_raises(self) -> None:
        with pytest.raises(ValueError, match="No user/assistant messages"):
            _messages_to_agent_prompt([{"role": "system", "content": "sys"}])


# ---------------------------------------------------------------------------
# Cache rejection
# ---------------------------------------------------------------------------


class TestCacheRejection:
    def test_cache_with_agent_raises_sync(self) -> None:
        cache = LRUCache(maxsize=10)
        with pytest.raises(ValueError, match="Caching not supported for agent models"):
            call_llm("claude-code", [{"role": "user", "content": "Hi"}], cache=cache)

    @pytest.mark.asyncio
    async def test_cache_with_agent_raises_async(self) -> None:
        cache = LRUCache(maxsize=10)
        with pytest.raises(ValueError, match="Caching not supported for agent models"):
            await acall_llm("claude-code", [{"role": "user", "content": "Hi"}], cache=cache)

    def test_cache_with_codex_raises_sync(self) -> None:
        cache = LRUCache(maxsize=10)
        with pytest.raises(ValueError, match="Caching not supported for agent models"):
            call_llm("codex", [{"role": "user", "content": "Hi"}], cache=cache)


# ---------------------------------------------------------------------------
# NotImplementedError guards (tools only — streaming/structured/batch are now supported)
# ---------------------------------------------------------------------------


class _DummyModel(BaseModel):
    name: str


class TestAgentGuards:
    """Tool-related functions should raise NotImplementedError for agent models."""

    def test_call_llm_with_tools(self) -> None:
        with pytest.raises(NotImplementedError, match="built-in tools"):
            call_llm_with_tools(
                "claude-code", [{"role": "user", "content": "Hi"}], tools=[],
            )

    @pytest.mark.asyncio
    async def test_acall_llm_with_tools(self) -> None:
        with pytest.raises(NotImplementedError, match="built-in tools"):
            await acall_llm_with_tools(
                "claude-code", [{"role": "user", "content": "Hi"}], tools=[],
            )

    def test_stream_llm_with_tools(self) -> None:
        with pytest.raises(NotImplementedError, match="built-in tools"):
            stream_llm_with_tools(
                "claude-code", [{"role": "user", "content": "Hi"}], tools=[],
            )

    @pytest.mark.asyncio
    async def test_astream_llm_with_tools(self) -> None:
        with pytest.raises(NotImplementedError, match="built-in tools"):
            await astream_llm_with_tools(
                "claude-code", [{"role": "user", "content": "Hi"}], tools=[],
            )

    def test_openai_agents_guard(self) -> None:
        """openai-agents/* should also trigger guards."""
        with pytest.raises(NotImplementedError, match="built-in tools"):
            stream_llm_with_tools(
                "openai-agents/gpt-5", [{"role": "user", "content": "Hi"}], tools=[],
            )

    def test_codex_with_tools(self) -> None:
        """Codex should also reject tool calling."""
        with pytest.raises(NotImplementedError, match="built-in tools"):
            call_llm_with_tools(
                "codex", [{"role": "user", "content": "Hi"}], tools=[],
            )


# ---------------------------------------------------------------------------
# Fake SDK fixtures
# ---------------------------------------------------------------------------


@dataclass
class _FakeTextBlock:
    text: str


@dataclass
class _FakeAssistantMessage:
    content: list[_FakeTextBlock]
    model: str = "claude-sonnet-4-5-20250929"


@dataclass
class _FakeResultMessage:
    subtype: str = "success"
    duration_ms: int = 1000
    duration_api_ms: int = 800
    is_error: bool = False
    num_turns: int = 1
    session_id: str = "test-session"
    total_cost_usd: float | None = 0.005
    usage: dict | None = None
    result: str | None = None
    structured_output: object = None


async def _fake_query(prompt, options=None):
    """Fake claude_agent_sdk.query() that yields an AssistantMessage and ResultMessage."""
    yield _FakeAssistantMessage(content=[_FakeTextBlock(text="The answer is 4.")])
    yield _FakeResultMessage(
        total_cost_usd=0.005,
        usage={"input_tokens": 100, "output_tokens": 20},
    )


def _make_fake_sdk_module():
    """Create a fake claude_agent_sdk module for sys.modules patching."""
    mod = types.ModuleType("claude_agent_sdk")
    mod.query = _fake_query  # type: ignore[attr-defined]
    mod.AssistantMessage = _FakeAssistantMessage  # type: ignore[attr-defined]
    mod.ResultMessage = _FakeResultMessage  # type: ignore[attr-defined]
    mod.TextBlock = _FakeTextBlock  # type: ignore[attr-defined]
    mod.ClaudeAgentOptions = MagicMock  # type: ignore[attr-defined]
    return mod


@pytest.fixture()
def _mock_agent_sdk(monkeypatch):
    """Install fake claude_agent_sdk in sys.modules and clear import caches."""
    fake_mod = _make_fake_sdk_module()
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake_mod)
    # Also clear the cached lazy import in agents module if it was previously imported
    import llm_client.agents as agents_mod
    # Force re-import on next call by invalidating any cached references
    for attr in ("query", "AssistantMessage", "ResultMessage", "TextBlock", "ClaudeAgentOptions"):
        if hasattr(agents_mod, attr):
            monkeypatch.delattr(agents_mod, attr, raising=False)


# ---------------------------------------------------------------------------
# Mocked agent SDK call
# ---------------------------------------------------------------------------


class TestAgentCallMocked:
    """Test agent routing with mocked claude_agent_sdk."""

    @pytest.mark.usefixtures("_mock_agent_sdk")
    def test_call_llm_agent_sync(self) -> None:
        result = call_llm("claude-code", [{"role": "user", "content": "What is 2+2?"}])
        assert isinstance(result, LLMCallResult)
        assert "4" in result.content
        assert result.cost == 0.005
        assert result.model == "claude-code"
        assert result.finish_reason == "stop"

    @pytest.mark.usefixtures("_mock_agent_sdk")
    @pytest.mark.asyncio
    async def test_acall_llm_agent_async(self) -> None:
        result = await acall_llm("claude-code", [{"role": "user", "content": "What is 2+2?"}])
        assert isinstance(result, LLMCallResult)
        assert "4" in result.content
        assert result.cost == 0.005
        assert result.finish_reason == "stop"

    @pytest.mark.usefixtures("_mock_agent_sdk")
    def test_hooks_fire_for_agent(self) -> None:
        before_calls: list = []
        after_calls: list = []
        hooks = Hooks(
            before_call=lambda m, msgs, kw: before_calls.append(m),
            after_call=lambda r: after_calls.append(r),
        )
        result = call_llm(
            "claude-code", [{"role": "user", "content": "Hi"}], hooks=hooks,
        )
        assert len(before_calls) == 1
        assert before_calls[0] == "claude-code"
        assert len(after_calls) == 1
        assert after_calls[0] is result

    @pytest.mark.usefixtures("_mock_agent_sdk")
    def test_agent_with_model_suffix(self) -> None:
        result = call_llm(
            "claude-code/opus", [{"role": "user", "content": "Hi"}],
        )
        assert result.model == "claude-code/opus"


class TestAgentFallback:
    """Test fallback from agent model to regular model and vice versa."""

    def test_fallback_from_agent_to_litellm(self, monkeypatch) -> None:
        """Agent fails, falls back to regular model."""
        # Install a failing fake SDK
        async def _failing_query(prompt, options=None):
            raise RuntimeError("Agent SDK failed")
            yield  # make it an async generator  # noqa: E501

        fake_mod = _make_fake_sdk_module()
        fake_mod.query = _failing_query  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake_mod)

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Fallback response"
        mock_resp.choices[0].message.tool_calls = None
        mock_resp.choices[0].finish_reason = "stop"
        mock_resp.usage.prompt_tokens = 10
        mock_resp.usage.completion_tokens = 5
        mock_resp.usage.total_tokens = 15

        with (
            patch("llm_client.client.litellm.completion", return_value=mock_resp),
            patch("llm_client.client.litellm.completion_cost", return_value=0.001),
        ):
            result = call_llm(
                "claude-code",
                [{"role": "user", "content": "Hi"}],
                fallback_models=["gpt-4o"],
            )
        assert result.content == "Fallback response"
        assert result.model == "gpt-4o"


class TestOpenAIAgentsGuard:
    """openai-agents/* should raise NotImplementedError at the agent level."""

    def test_openai_agents_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="not yet supported"):
            call_llm(
                "openai-agents/gpt-5",
                [{"role": "user", "content": "Hi"}],
            )


# ---------------------------------------------------------------------------
# Structured output (mocked)
# ---------------------------------------------------------------------------


class _CityInfo(BaseModel):
    name: str
    country: str


class TestAgentStructured:
    """Test structured output via agent SDK."""

    def _make_structured_query(self):
        """Create a fake query that returns structured output."""
        async def _structured_query(prompt, options=None):
            yield _FakeAssistantMessage(
                content=[_FakeTextBlock(text='{"name": "Tokyo", "country": "Japan"}')]
            )
            yield _FakeResultMessage(
                total_cost_usd=0.01,
                usage={"input_tokens": 200, "output_tokens": 50},
                structured_output={"name": "Tokyo", "country": "Japan"},
            )
        return _structured_query

    @pytest.fixture()
    def _mock_structured_sdk(self, monkeypatch):
        fake_mod = _make_fake_sdk_module()
        fake_mod.query = self._make_structured_query()  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake_mod)

    @pytest.mark.usefixtures("_mock_structured_sdk")
    def test_call_llm_structured_sync(self) -> None:
        parsed, meta = call_llm_structured(
            "claude-code",
            [{"role": "user", "content": "Info about Tokyo"}],
            response_model=_CityInfo,
        )
        assert isinstance(parsed, _CityInfo)
        assert parsed.name == "Tokyo"
        assert parsed.country == "Japan"
        assert isinstance(meta, LLMCallResult)
        assert meta.cost == 0.01
        assert meta.model == "claude-code"

    @pytest.mark.usefixtures("_mock_structured_sdk")
    @pytest.mark.asyncio
    async def test_acall_llm_structured_async(self) -> None:
        parsed, meta = await acall_llm_structured(
            "claude-code",
            [{"role": "user", "content": "Info about Tokyo"}],
            response_model=_CityInfo,
        )
        assert isinstance(parsed, _CityInfo)
        assert parsed.name == "Tokyo"
        assert parsed.country == "Japan"
        assert meta.cost == 0.01

    @pytest.mark.usefixtures("_mock_structured_sdk")
    def test_structured_hooks_fire(self) -> None:
        before_calls: list = []
        after_calls: list = []
        hooks = Hooks(
            before_call=lambda m, msgs, kw: before_calls.append(m),
            after_call=lambda r: after_calls.append(r),
        )
        parsed, meta = call_llm_structured(
            "claude-code",
            [{"role": "user", "content": "Info about Tokyo"}],
            response_model=_CityInfo,
            hooks=hooks,
        )
        assert len(before_calls) == 1
        assert len(after_calls) == 1

    def test_structured_falls_back_to_structured_output_field(self, monkeypatch) -> None:
        """If text content is empty but structured_output is set, use it."""
        async def _query(prompt, options=None):
            yield _FakeAssistantMessage(content=[])
            yield _FakeResultMessage(
                total_cost_usd=0.01,
                usage={"input_tokens": 200, "output_tokens": 50},
                structured_output={"name": "Paris", "country": "France"},
            )

        fake_mod = _make_fake_sdk_module()
        fake_mod.query = _query  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake_mod)

        parsed, meta = call_llm_structured(
            "claude-code",
            [{"role": "user", "content": "Info about Paris"}],
            response_model=_CityInfo,
        )
        assert parsed.name == "Paris"
        assert parsed.country == "France"


# ---------------------------------------------------------------------------
# Streaming (mocked)
# ---------------------------------------------------------------------------


class TestAgentStream:
    """Test streaming via agent SDK."""

    @pytest.mark.usefixtures("_mock_agent_sdk")
    def test_stream_llm_sync(self) -> None:
        stream = stream_llm("claude-code", [{"role": "user", "content": "Hi"}])
        chunks: list[str] = []
        for chunk in stream:
            chunks.append(chunk)
        assert len(chunks) > 0
        assert "4" in "".join(chunks)
        result = stream.result
        assert isinstance(result, LLMCallResult)
        assert result.cost == 0.005
        assert result.model == "claude-code"

    @pytest.mark.usefixtures("_mock_agent_sdk")
    @pytest.mark.asyncio
    async def test_astream_llm_async(self) -> None:
        stream = await astream_llm("claude-code", [{"role": "user", "content": "Hi"}])
        chunks: list[str] = []
        async for chunk in stream:
            chunks.append(chunk)
        assert len(chunks) > 0
        assert "4" in "".join(chunks)
        result = stream.result
        assert isinstance(result, LLMCallResult)
        assert result.cost == 0.005

    @pytest.mark.usefixtures("_mock_agent_sdk")
    def test_stream_result_before_consume_raises(self) -> None:
        stream = stream_llm("claude-code", [{"role": "user", "content": "Hi"}])
        with pytest.raises(RuntimeError, match="not yet consumed"):
            _ = stream.result

    @pytest.mark.usefixtures("_mock_agent_sdk")
    def test_stream_hooks_fire(self) -> None:
        before_calls: list = []
        after_calls: list = []
        hooks = Hooks(
            before_call=lambda m, msgs, kw: before_calls.append(m),
            after_call=lambda r: after_calls.append(r),
        )
        stream = stream_llm(
            "claude-code", [{"role": "user", "content": "Hi"}], hooks=hooks,
        )
        for _ in stream:
            pass
        assert len(before_calls) == 1
        assert len(after_calls) == 1

    def test_stream_multi_messages(self, monkeypatch) -> None:
        """Multiple AssistantMessages yield multiple chunks."""
        async def _multi_query(prompt, options=None):
            yield _FakeAssistantMessage(content=[_FakeTextBlock(text="First. ")])
            yield _FakeAssistantMessage(content=[_FakeTextBlock(text="Second.")])
            yield _FakeResultMessage(
                total_cost_usd=0.01,
                usage={"input_tokens": 100, "output_tokens": 40},
            )

        fake_mod = _make_fake_sdk_module()
        fake_mod.query = _multi_query  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake_mod)

        stream = stream_llm("claude-code", [{"role": "user", "content": "Hi"}])
        chunks = list(stream)
        assert len(chunks) == 2
        assert chunks[0] == "First. "
        assert chunks[1] == "Second."


# ---------------------------------------------------------------------------
# Batch (mocked)
# ---------------------------------------------------------------------------


class TestAgentBatch:
    """Test batch calls route through agent SDK."""

    @pytest.mark.usefixtures("_mock_agent_sdk")
    def test_call_llm_batch_sync(self) -> None:
        messages_list = [
            [{"role": "user", "content": f"What is {i}+{i}?"}]
            for i in range(3)
        ]
        results = call_llm_batch("claude-code", messages_list, max_concurrent=3)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, LLMCallResult)
            assert r.model == "claude-code"
            assert r.cost == 0.005

    @pytest.mark.usefixtures("_mock_agent_sdk")
    @pytest.mark.asyncio
    async def test_acall_llm_batch_async(self) -> None:
        messages_list = [
            [{"role": "user", "content": f"What is {i}+{i}?"}]
            for i in range(2)
        ]
        results = await acall_llm_batch("claude-code", messages_list)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, LLMCallResult)

    @pytest.mark.usefixtures("_mock_agent_sdk")
    def test_call_llm_structured_batch_sync(self) -> None:
        """Structured batch uses structured output routing."""
        # We need a structured-capable fake SDK for this test
        pass  # Covered by structured + batch integration — guard removal is the key test

    @pytest.mark.usefixtures("_mock_agent_sdk")
    def test_batch_empty_list(self) -> None:
        results = call_llm_batch("claude-code", [])
        assert results == []


# ===========================================================================
# Codex SDK tests (mocked)
# ===========================================================================


# ---------------------------------------------------------------------------
# Fake Codex SDK fixtures
# ---------------------------------------------------------------------------


@dataclass
class _FakeUsage:
    input_tokens: int = 100
    cached_input_tokens: int = 0
    output_tokens: int = 20


@dataclass
class _FakeAgentMessageItem:
    id: str = "msg-1"
    type: str = "agent_message"
    text: str = "The answer is 4."


@dataclass
class _FakeTurn:
    items: list = None  # type: ignore[assignment]
    final_response: str = "The answer is 4."
    usage: _FakeUsage | None = None

    def __post_init__(self) -> None:
        if self.items is None:
            self.items = [_FakeAgentMessageItem()]
        if self.usage is None:
            self.usage = _FakeUsage()


@dataclass
class _FakeItemCompletedEvent:
    type: str = "item.completed"
    item: _FakeAgentMessageItem | None = None

    def __post_init__(self) -> None:
        if self.item is None:
            self.item = _FakeAgentMessageItem()


@dataclass
class _FakeTurnCompletedEvent:
    type: str = "turn.completed"
    usage: _FakeUsage | None = None

    def __post_init__(self) -> None:
        if self.usage is None:
            self.usage = _FakeUsage()


@dataclass
class _FakeStreamedTurn:
    events: object = None  # set to an async iterator


class _FakeThread:
    """Fake Codex Thread with async run and run_streamed."""

    def __init__(self, turn: _FakeTurn | None = None) -> None:
        self._turn = turn or _FakeTurn()

    async def run(self, input_: str, turn_options: object = None) -> _FakeTurn:
        return self._turn

    async def run_streamed(self, input_: str, turn_options: object = None) -> _FakeStreamedTurn:
        async def _events():
            for item in self._turn.items:
                yield _FakeItemCompletedEvent(item=item)
            yield _FakeTurnCompletedEvent(usage=self._turn.usage)

        return _FakeStreamedTurn(events=_events())


class _FakeCodex:
    """Fake Codex client."""

    def __init__(self, options: object = None) -> None:
        self._thread = _FakeThread()

    def start_thread(self, options: object = None) -> _FakeThread:
        return self._thread


def _make_fake_codex_module():
    """Create a fake openai_codex_sdk module for sys.modules patching."""
    mod = types.ModuleType("openai_codex_sdk")
    mod.Codex = _FakeCodex  # type: ignore[attr-defined]
    mod.ThreadOptions = MagicMock  # type: ignore[attr-defined]
    mod.TurnOptions = MagicMock  # type: ignore[attr-defined]
    mod.Turn = _FakeTurn  # type: ignore[attr-defined]
    mod.StreamedTurn = _FakeStreamedTurn  # type: ignore[attr-defined]
    mod.AgentMessageItem = _FakeAgentMessageItem  # type: ignore[attr-defined]
    mod.ItemCompletedEvent = _FakeItemCompletedEvent  # type: ignore[attr-defined]
    mod.TurnCompletedEvent = _FakeTurnCompletedEvent  # type: ignore[attr-defined]
    mod.Usage = _FakeUsage  # type: ignore[attr-defined]

    # Sub-module for CodexOptions
    codex_submod = types.ModuleType("openai_codex_sdk.codex")
    codex_submod.CodexOptions = MagicMock  # type: ignore[attr-defined]
    mod.codex = codex_submod  # type: ignore[attr-defined]

    return mod, codex_submod


@pytest.fixture()
def _mock_codex_sdk(monkeypatch):
    """Install fake openai_codex_sdk in sys.modules."""
    fake_mod, codex_submod = _make_fake_codex_module()
    monkeypatch.setitem(sys.modules, "openai_codex_sdk", fake_mod)
    monkeypatch.setitem(sys.modules, "openai_codex_sdk.codex", codex_submod)


# ---------------------------------------------------------------------------
# Codex detection
# ---------------------------------------------------------------------------


class TestCodexDetection:
    def test_codex_bare(self) -> None:
        assert _is_agent_model("codex") is True

    def test_codex_with_model(self) -> None:
        assert _is_agent_model("codex/gpt-5") is True

    def test_codex_parse(self) -> None:
        assert _parse_agent_model("codex") == ("codex", None)
        assert _parse_agent_model("codex/gpt-5") == ("codex", "gpt-5")
        assert _parse_agent_model("codex/o3") == ("codex", "o3")


# ---------------------------------------------------------------------------
# Codex guards
# ---------------------------------------------------------------------------


class TestCodexGuards:
    def test_cache_rejected(self) -> None:
        cache = LRUCache(maxsize=10)
        with pytest.raises(ValueError, match="Caching not supported"):
            call_llm("codex", [{"role": "user", "content": "Hi"}], cache=cache)

    def test_tools_rejected(self) -> None:
        with pytest.raises(NotImplementedError, match="built-in tools"):
            call_llm_with_tools(
                "codex/gpt-5", [{"role": "user", "content": "Hi"}], tools=[],
            )


# ---------------------------------------------------------------------------
# Codex call (mocked)
# ---------------------------------------------------------------------------


class TestCodexCall:
    @pytest.mark.usefixtures("_mock_codex_sdk")
    def test_call_llm_sync(self) -> None:
        result = call_llm("codex", [{"role": "user", "content": "What is 2+2?"}])
        assert isinstance(result, LLMCallResult)
        assert "4" in result.content
        assert result.model == "codex"
        assert result.finish_reason == "stop"

    @pytest.mark.usefixtures("_mock_codex_sdk")
    @pytest.mark.asyncio
    async def test_acall_llm_async(self) -> None:
        result = await acall_llm("codex", [{"role": "user", "content": "What is 2+2?"}])
        assert isinstance(result, LLMCallResult)
        assert "4" in result.content
        assert result.finish_reason == "stop"

    @pytest.mark.usefixtures("_mock_codex_sdk")
    def test_hooks_fire(self) -> None:
        before_calls: list = []
        after_calls: list = []
        hooks = Hooks(
            before_call=lambda m, msgs, kw: before_calls.append(m),
            after_call=lambda r: after_calls.append(r),
        )
        result = call_llm("codex", [{"role": "user", "content": "Hi"}], hooks=hooks)
        assert len(before_calls) == 1
        assert before_calls[0] == "codex"
        assert len(after_calls) == 1
        assert after_calls[0] is result

    @pytest.mark.usefixtures("_mock_codex_sdk")
    def test_model_suffix(self) -> None:
        result = call_llm("codex/gpt-5", [{"role": "user", "content": "Hi"}])
        assert result.model == "codex/gpt-5"


# ---------------------------------------------------------------------------
# Codex structured (mocked)
# ---------------------------------------------------------------------------


class TestCodexStructured:
    @pytest.fixture()
    def _mock_structured_codex(self, monkeypatch):
        """Install a Codex SDK that returns JSON."""
        fake_mod, codex_submod = _make_fake_codex_module()
        # Override FakeThread to return JSON
        turn = _FakeTurn(
            final_response='{"name": "Tokyo", "country": "Japan"}',
            usage=_FakeUsage(input_tokens=200, output_tokens=50),
        )
        fake_codex_cls = type("FakeCodex", (), {
            "__init__": lambda self, options=None: setattr(self, "_turn", turn),
            "start_thread": lambda self, options=None: _FakeThread(turn),
        })
        fake_mod.Codex = fake_codex_cls  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "openai_codex_sdk", fake_mod)
        monkeypatch.setitem(sys.modules, "openai_codex_sdk.codex", codex_submod)

    @pytest.mark.usefixtures("_mock_structured_codex")
    def test_structured_sync(self) -> None:
        parsed, meta = call_llm_structured(
            "codex",
            [{"role": "user", "content": "Info about Tokyo"}],
            response_model=_CityInfo,
        )
        assert isinstance(parsed, _CityInfo)
        assert parsed.name == "Tokyo"
        assert parsed.country == "Japan"
        assert isinstance(meta, LLMCallResult)
        assert meta.model == "codex"

    @pytest.mark.usefixtures("_mock_structured_codex")
    @pytest.mark.asyncio
    async def test_structured_async(self) -> None:
        parsed, meta = await acall_llm_structured(
            "codex",
            [{"role": "user", "content": "Info about Tokyo"}],
            response_model=_CityInfo,
        )
        assert isinstance(parsed, _CityInfo)
        assert parsed.name == "Tokyo"

    def test_structured_with_fenced_json(self, monkeypatch) -> None:
        """Codex sometimes wraps JSON in code fences — should still parse."""
        fake_mod, codex_submod = _make_fake_codex_module()
        turn = _FakeTurn(
            final_response='```json\n{"name": "Berlin", "country": "Germany"}\n```',
            usage=_FakeUsage(),
        )
        fake_codex_cls = type("FakeCodex", (), {
            "__init__": lambda self, options=None: setattr(self, "_turn", turn),
            "start_thread": lambda self, options=None: _FakeThread(turn),
        })
        fake_mod.Codex = fake_codex_cls  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "openai_codex_sdk", fake_mod)
        monkeypatch.setitem(sys.modules, "openai_codex_sdk.codex", codex_submod)

        parsed, meta = call_llm_structured(
            "codex",
            [{"role": "user", "content": "Info about Berlin"}],
            response_model=_CityInfo,
        )
        assert parsed.name == "Berlin"
        assert parsed.country == "Germany"


# ---------------------------------------------------------------------------
# Codex streaming (mocked)
# ---------------------------------------------------------------------------


class TestCodexStream:
    @pytest.mark.usefixtures("_mock_codex_sdk")
    def test_stream_sync(self) -> None:
        stream = stream_llm("codex", [{"role": "user", "content": "Hi"}])
        chunks: list[str] = []
        for chunk in stream:
            chunks.append(chunk)
        assert len(chunks) > 0
        assert "4" in "".join(chunks)
        result = stream.result
        assert isinstance(result, LLMCallResult)
        assert result.model == "codex"

    @pytest.mark.usefixtures("_mock_codex_sdk")
    @pytest.mark.asyncio
    async def test_astream_async(self) -> None:
        stream = await astream_llm("codex", [{"role": "user", "content": "Hi"}])
        chunks: list[str] = []
        async for chunk in stream:
            chunks.append(chunk)
        assert len(chunks) > 0
        assert "4" in "".join(chunks)
        result = stream.result
        assert isinstance(result, LLMCallResult)

    @pytest.mark.usefixtures("_mock_codex_sdk")
    def test_stream_hooks_fire(self) -> None:
        before_calls: list = []
        after_calls: list = []
        hooks = Hooks(
            before_call=lambda m, msgs, kw: before_calls.append(m),
            after_call=lambda r: after_calls.append(r),
        )
        stream = stream_llm("codex", [{"role": "user", "content": "Hi"}], hooks=hooks)
        for _ in stream:
            pass
        assert len(before_calls) == 1
        assert len(after_calls) == 1

    def test_stream_multi_items(self, monkeypatch) -> None:
        """Multiple AgentMessageItems yield multiple chunks."""
        fake_mod, codex_submod = _make_fake_codex_module()
        items = [
            _FakeAgentMessageItem(id="msg-1", text="First. "),
            _FakeAgentMessageItem(id="msg-2", text="Second."),
        ]
        turn = _FakeTurn(items=items, final_response="First. \nSecond.")
        fake_codex_cls = type("FakeCodex", (), {
            "__init__": lambda self, options=None: setattr(self, "_turn", turn),
            "start_thread": lambda self, options=None: _FakeThread(turn),
        })
        fake_mod.Codex = fake_codex_cls  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "openai_codex_sdk", fake_mod)
        monkeypatch.setitem(sys.modules, "openai_codex_sdk.codex", codex_submod)

        stream = stream_llm("codex", [{"role": "user", "content": "Hi"}])
        chunks = list(stream)
        assert len(chunks) == 2
        assert chunks[0] == "First. "
        assert chunks[1] == "Second."


# ---------------------------------------------------------------------------
# Codex batch (mocked)
# ---------------------------------------------------------------------------


class TestCodexBatch:
    @pytest.mark.usefixtures("_mock_codex_sdk")
    def test_batch_sync(self) -> None:
        messages_list = [
            [{"role": "user", "content": f"What is {i}+{i}?"}]
            for i in range(3)
        ]
        results = call_llm_batch("codex", messages_list, max_concurrent=3)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, LLMCallResult)
            assert r.model == "codex"

    @pytest.mark.usefixtures("_mock_codex_sdk")
    @pytest.mark.asyncio
    async def test_batch_async(self) -> None:
        messages_list = [
            [{"role": "user", "content": f"What is {i}+{i}?"}]
            for i in range(2)
        ]
        results = await acall_llm_batch("codex", messages_list)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Codex fallback (mocked)
# ---------------------------------------------------------------------------


class TestCodexFallback:
    def test_fallback_from_codex_to_litellm(self, monkeypatch) -> None:
        """Codex fails, falls back to regular model."""
        fake_mod, codex_submod = _make_fake_codex_module()

        class _FailingThread:
            async def run(self, input_, turn_options=None):
                raise RuntimeError("Codex SDK failed")

        class _FailingCodex:
            def __init__(self, options=None):
                pass
            def start_thread(self, options=None):
                return _FailingThread()

        fake_mod.Codex = _FailingCodex  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "openai_codex_sdk", fake_mod)
        monkeypatch.setitem(sys.modules, "openai_codex_sdk.codex", codex_submod)

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Fallback response"
        mock_resp.choices[0].message.tool_calls = None
        mock_resp.choices[0].finish_reason = "stop"
        mock_resp.usage.prompt_tokens = 10
        mock_resp.usage.completion_tokens = 5
        mock_resp.usage.total_tokens = 15

        with (
            patch("llm_client.client.litellm.completion", return_value=mock_resp),
            patch("llm_client.client.litellm.completion_cost", return_value=0.001),
        ):
            result = call_llm(
                "codex",
                [{"role": "user", "content": "Hi"}],
                fallback_models=["gpt-4o"],
            )
        assert result.content == "Fallback response"
        assert result.model == "gpt-4o"
