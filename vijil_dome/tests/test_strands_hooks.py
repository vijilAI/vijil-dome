"""Tests for Dome Strands adapter (DomeHookProvider)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from strands.hooks import AfterModelCallEvent, BeforeModelCallEvent, HookRegistry

from vijil_dome.integrations.strands.hooks import (
    DomeHookProvider,
    DEFAULT_INPUT_BLOCKED_MESSAGE,
    DEFAULT_OUTPUT_BLOCKED_MESSAGE,
    _extract_last_user_text,
    _extract_response_text,
)


# ---------------------------------------------------------------------------
# Helpers to build Strands-shaped objects without importing strands
# ---------------------------------------------------------------------------

def _make_message(role: str, text: str) -> dict:
    """Build a Strands Message (TypedDict) with a single text content block."""
    return {"role": role, "content": [{"text": text}]}


def _make_agent(messages: list[dict]) -> MagicMock:
    """Build a mock Agent with a messages list."""
    agent = MagicMock()
    agent.messages = messages
    return agent


def _make_before_event(messages: list[dict]):
    """Build a BeforeModelCallEvent-shaped object."""
    event = MagicMock()
    event.agent = _make_agent(messages)
    return event


# ---------------------------------------------------------------------------
# _extract_last_user_text
# ---------------------------------------------------------------------------

class TestExtractLastUserText:
    def test_extracts_last_user_message(self):
        messages = [
            _make_message("user", "first"),
            _make_message("assistant", "reply"),
            _make_message("user", "second"),
        ]
        text, idx = _extract_last_user_text(messages)
        assert text == "second"
        assert idx == 2

    def test_returns_none_for_no_user_messages(self):
        messages = [_make_message("assistant", "hello")]
        text, idx = _extract_last_user_text(messages)
        assert text is None
        assert idx is None

    def test_returns_none_for_empty_messages(self):
        text, idx = _extract_last_user_text([])
        assert text is None
        assert idx is None

    def test_concatenates_multiple_text_blocks(self):
        msg = {"role": "user", "content": [{"text": "hello"}, {"text": "world"}]}
        text, idx = _extract_last_user_text([msg])
        assert text == "hello world"

    def test_skips_non_text_content_blocks(self):
        msg = {"role": "user", "content": [{"toolUse": {}}, {"text": "actual text"}]}
        text, idx = _extract_last_user_text([msg])
        assert text == "actual text"


# ---------------------------------------------------------------------------
# Input guarding
# ---------------------------------------------------------------------------

class TestInputGuarding:
    @pytest.fixture
    def dome_mock(self):
        dome = MagicMock()
        dome.async_guard_input = AsyncMock()
        dome.async_guard_output = AsyncMock()
        return dome

    @pytest.mark.asyncio
    async def test_flagged_input_replaces_message(self, dome_mock):
        scan_result = MagicMock()
        scan_result.is_safe.return_value = False
        dome_mock.async_guard_input.return_value = scan_result

        provider = DomeHookProvider(dome_mock, agent_id="a1", team_id="t1")

        messages = [_make_message("user", "malicious prompt")]
        event = _make_before_event(messages)

        await provider._guard_input(event)

        # The user message text should be replaced with blocked message
        assert event.agent.messages[0]["content"] == [{"text": DEFAULT_INPUT_BLOCKED_MESSAGE}]
        dome_mock.async_guard_input.assert_called_once_with(
            "malicious prompt", agent_id="a1"
        )

    @pytest.mark.asyncio
    async def test_clean_input_passes_through(self, dome_mock):
        scan_result = MagicMock()
        scan_result.is_safe.return_value = True
        dome_mock.async_guard_input.return_value = scan_result

        provider = DomeHookProvider(dome_mock)

        messages = [_make_message("user", "normal question")]
        event = _make_before_event(messages)

        await provider._guard_input(event)

        # Message unchanged
        assert event.agent.messages[0]["content"] == [{"text": "normal question"}]

    @pytest.mark.asyncio
    async def test_no_user_messages_skips_scan(self, dome_mock):
        provider = DomeHookProvider(dome_mock)

        messages = [_make_message("assistant", "hi")]
        event = _make_before_event(messages)

        await provider._guard_input(event)

        dome_mock.async_guard_input.assert_not_called()

    @pytest.mark.asyncio
    async def test_custom_blocked_message(self, dome_mock):
        scan_result = MagicMock()
        scan_result.is_safe.return_value = False
        dome_mock.async_guard_input.return_value = scan_result

        custom_msg = "Custom block"
        provider = DomeHookProvider(dome_mock, input_blocked_message=custom_msg)

        messages = [_make_message("user", "bad")]
        event = _make_before_event(messages)

        await provider._guard_input(event)

        assert event.agent.messages[0]["content"] == [{"text": custom_msg}]


# ---------------------------------------------------------------------------
# _extract_response_text
# ---------------------------------------------------------------------------

class TestExtractResponseText:
    def test_extracts_text_from_response(self):
        message = {"role": "assistant", "content": [{"text": "response text"}]}
        assert _extract_response_text(message) == "response text"

    def test_returns_none_for_no_text_blocks(self):
        message = {"role": "assistant", "content": [{"toolUse": {"name": "search"}}]}
        assert _extract_response_text(message) is None

    def test_concatenates_multiple_text_blocks(self):
        message = {
            "role": "assistant",
            "content": [{"text": "hello"}, {"text": "world"}],
        }
        assert _extract_response_text(message) == "hello world"


# ---------------------------------------------------------------------------
# Helpers for output guarding
# ---------------------------------------------------------------------------

def _make_after_event(response_text: str | None):
    """Build an AfterModelCallEvent-shaped object."""
    event = MagicMock()
    event.agent = _make_agent([])
    if response_text is not None:
        event.stop_response = MagicMock()
        event.stop_response.message = _make_message("assistant", response_text)
    else:
        event.stop_response = None
    return event


# ---------------------------------------------------------------------------
# Output guarding
# ---------------------------------------------------------------------------

class TestOutputGuarding:
    @pytest.fixture
    def dome_mock(self):
        dome = MagicMock()
        dome.async_guard_input = AsyncMock()
        dome.async_guard_output = AsyncMock()
        return dome

    @pytest.mark.asyncio
    async def test_flagged_output_replaces_response(self, dome_mock):
        scan_result = MagicMock()
        scan_result.is_safe.return_value = False
        dome_mock.async_guard_output.return_value = scan_result

        provider = DomeHookProvider(dome_mock, agent_id="a1", team_id="t1")

        event = _make_after_event("unsafe response")

        await provider._guard_output(event)

        # The response content should be replaced with blocked message
        assert event.stop_response.message["content"] == [
            {"text": DEFAULT_OUTPUT_BLOCKED_MESSAGE}
        ]
        dome_mock.async_guard_output.assert_called_once_with(
            "unsafe response", agent_id="a1"
        )

    @pytest.mark.asyncio
    async def test_clean_output_passes_through(self, dome_mock):
        scan_result = MagicMock()
        scan_result.is_safe.return_value = True
        dome_mock.async_guard_output.return_value = scan_result

        provider = DomeHookProvider(dome_mock)

        event = _make_after_event("safe response")

        await provider._guard_output(event)

        # Message unchanged
        assert event.stop_response.message["content"] == [{"text": "safe response"}]

    @pytest.mark.asyncio
    async def test_no_stop_response_skips_scan(self, dome_mock):
        provider = DomeHookProvider(dome_mock)

        event = _make_after_event(None)

        await provider._guard_output(event)

        dome_mock.async_guard_output.assert_not_called()

    @pytest.mark.asyncio
    async def test_custom_output_blocked_message(self, dome_mock):
        scan_result = MagicMock()
        scan_result.is_safe.return_value = False
        dome_mock.async_guard_output.return_value = scan_result

        custom_msg = "Custom output block"
        provider = DomeHookProvider(dome_mock, output_blocked_message=custom_msg)

        event = _make_after_event("bad output")

        await provider._guard_output(event)

        assert event.stop_response.message["content"] == [{"text": custom_msg}]


# ---------------------------------------------------------------------------
# Hook registration and protocol conformance
# ---------------------------------------------------------------------------

class TestHookRegistration:
    def test_register_hooks_adds_both_callbacks(self):
        dome = MagicMock()
        provider = DomeHookProvider(dome)
        registry = HookRegistry()

        provider.register_hooks(registry)

        # Both event types should have callbacks registered.
        # Verify by creating events and checking callbacks exist.
        before_event = BeforeModelCallEvent(agent=MagicMock())
        after_event = AfterModelCallEvent(agent=MagicMock())

        before_cbs = list(registry.get_callbacks_for(before_event))
        after_cbs = list(registry.get_callbacks_for(after_event))
        assert len(before_cbs) == 1
        assert len(after_cbs) == 1

    def test_provider_satisfies_hook_provider_protocol(self):
        from strands.hooks import HookProvider

        dome = MagicMock()
        provider = DomeHookProvider(dome)
        assert isinstance(provider, HookProvider)
