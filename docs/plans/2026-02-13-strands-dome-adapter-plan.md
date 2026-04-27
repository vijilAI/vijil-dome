# Dome Strands Adapter Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a Dome framework adapter for Strands agents via `DomeHookProvider`, closing the output guarding gap.

**Architecture:** `DomeHookProvider` implements Strands' `HookProvider` protocol, registering async callbacks for `BeforeModelCallEvent` (input guard) and `AfterModelCallEvent` (output guard). It follows the same pattern as the existing ADK adapter in `vijil_dome/integrations/adk/callbacks.py`.

**Tech Stack:** Python 3.12, Strands SDK (`strands-agents`), vijil-dome, pytest

**Design Doc:** `docs/plans/2026-02-13-strands-dome-adapter-design.md`

---

### Task 1: Create `DomeHookProvider` with input guarding

**Files:**
- Create: `vijil_dome/integrations/strands/__init__.py`
- Create: `vijil_dome/integrations/strands/hooks.py`
- Create: `vijil_dome/tests/test_strands_hooks.py`

**Step 1: Write the failing tests for input guarding**

Create `vijil_dome/tests/test_strands_hooks.py`:

```python
"""Tests for Dome Strands adapter (DomeHookProvider)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

from vijil_dome.integrations.strands.hooks import (
    DomeHookProvider,
    DEFAULT_INPUT_BLOCKED_MESSAGE,
    DEFAULT_OUTPUT_BLOCKED_MESSAGE,
    _extract_last_user_text,
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
        msg = {"role": "user", "content": [{"text": "hello "}, {"text": "world"}]}
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
        scan_result.flagged = True
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
        scan_result.flagged = False
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
        scan_result.flagged = True
        dome_mock.async_guard_input.return_value = scan_result

        custom_msg = "Custom block"
        provider = DomeHookProvider(dome_mock, input_blocked_message=custom_msg)

        messages = [_make_message("user", "bad")]
        event = _make_before_event(messages)

        await provider._guard_input(event)

        assert event.agent.messages[0]["content"] == [{"text": custom_msg}]
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ciphr/Code/Vijil/vijil-dome && python -m pytest vijil_dome/tests/test_strands_hooks.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vijil_dome.integrations.strands'`

**Step 3: Write the implementation**

Create `vijil_dome/integrations/strands/__init__.py`:

```python
from vijil_dome.integrations.strands.hooks import (
    DEFAULT_INPUT_BLOCKED_MESSAGE,
    DEFAULT_OUTPUT_BLOCKED_MESSAGE,
    DomeHookProvider,
)

__all__ = [
    "DomeHookProvider",
    "DEFAULT_INPUT_BLOCKED_MESSAGE",
    "DEFAULT_OUTPUT_BLOCKED_MESSAGE",
]
```

Create `vijil_dome/integrations/strands/hooks.py`:

```python
"""Dome Strands Integration — HookProvider for Strands agents.

Registers hooks that scan model input and output through Dome's
guardrails. Flagged content is replaced with a blocked message.

Usage:
    from vijil_dome import Dome
    from vijil_dome.integrations.strands import DomeHookProvider

    dome = Dome(config)
    agent = Agent(hooks=[DomeHookProvider(dome, agent_id="...", team_id="...")])
"""

import logging
from typing import Any, Optional

from vijil_dome import Dome

try:
    from strands.hooks import (
        AfterModelCallEvent,
        BeforeModelCallEvent,
        HookProvider,
        HookRegistry,
    )
except ImportError:
    raise RuntimeError(
        "Strands SDK is not installed. "
        "Install it with: pip install strands-agents"
    )

logger = logging.getLogger(__name__)

DEFAULT_INPUT_BLOCKED_MESSAGE = (
    "I'm sorry, but I can't help with that request. "
    "It appears to contain content that violates my usage policies. "
    "I'd be happy to assist you with a different question."
)

DEFAULT_OUTPUT_BLOCKED_MESSAGE = (
    "The generated response was blocked by the guardrail policy. "
    "I'd be happy to assist you with a different request."
)


def _extract_last_user_text(messages: list[dict]) -> tuple[Optional[str], Optional[int]]:
    """Find the last user message and extract its text.

    Returns:
        (text, index) if found, (None, None) if no user message exists.
    """
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") != "user":
            continue
        parts = msg.get("content", [])
        texts = [
            block["text"]
            for block in parts
            if isinstance(block, dict) and "text" in block
        ]
        if texts:
            return " ".join(texts), i
    return None, None


def _extract_response_text(message: dict) -> Optional[str]:
    """Extract text from a model response message."""
    parts = message.get("content", [])
    texts = [
        block["text"]
        for block in parts
        if isinstance(block, dict) and "text" in block
    ]
    return " ".join(texts) if texts else None


class DomeHookProvider(HookProvider):
    """Dome guardrail integration for Strands agents.

    Registers hooks for BeforeModelCallEvent (input) and
    AfterModelCallEvent (output) that scan content through Dome's
    guardrails and replace flagged content with a blocked message.
    """

    def __init__(
        self,
        dome: Dome,
        agent_id: str = "",
        team_id: str = "",
        input_blocked_message: str = DEFAULT_INPUT_BLOCKED_MESSAGE,
        output_blocked_message: str = DEFAULT_OUTPUT_BLOCKED_MESSAGE,
    ):
        self.dome = dome
        self.agent_id = agent_id
        self.team_id = team_id
        self.input_blocked_message = input_blocked_message
        self.output_blocked_message = output_blocked_message

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeModelCallEvent, self._guard_input)
        registry.add_callback(AfterModelCallEvent, self._guard_output)

    async def _guard_input(self, event: BeforeModelCallEvent) -> None:
        """Scan the last user message; replace if flagged."""
        text, idx = _extract_last_user_text(event.agent.messages)
        if text is None or idx is None:
            return

        scan = await self.dome.async_guard_input(text, agent_id=self.agent_id)
        if scan.flagged:
            logger.warning("Dome blocked input: %s...", text[:80])
            event.agent.messages[idx]["content"] = [
                {"text": self.input_blocked_message}
            ]

    async def _guard_output(self, event: AfterModelCallEvent) -> None:
        """Scan the model response; replace if flagged."""
        if event.stop_response is None:
            return

        text = _extract_response_text(event.stop_response.message)
        if not text:
            return

        scan = await self.dome.async_guard_output(text, agent_id=self.agent_id)
        if scan.flagged:
            logger.warning("Dome blocked output: %s...", text[:80])
            event.stop_response.message["content"] = [
                {"text": self.output_blocked_message}
            ]
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/ciphr/Code/Vijil/vijil-dome && python -m pytest vijil_dome/tests/test_strands_hooks.py -v`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add vijil_dome/integrations/strands/ vijil_dome/tests/test_strands_hooks.py
git commit -m "feat(strands): add DomeHookProvider for input guarding"
```

---

### Task 2: Add output guarding tests

**Files:**
- Modify: `vijil_dome/tests/test_strands_hooks.py`

**Step 1: Write the failing tests for output guarding**

Append to `vijil_dome/tests/test_strands_hooks.py`:

```python
from vijil_dome.integrations.strands.hooks import _extract_response_text


# ---------------------------------------------------------------------------
# _extract_response_text
# ---------------------------------------------------------------------------

class TestExtractResponseText:
    def test_extracts_text_from_response(self):
        message = _make_message("assistant", "response text")
        assert _extract_response_text(message) == "response text"

    def test_returns_none_for_no_text_blocks(self):
        message = {"role": "assistant", "content": [{"toolUse": {}}]}
        assert _extract_response_text(message) is None

    def test_concatenates_multiple_text_blocks(self):
        message = {"role": "assistant", "content": [{"text": "part1 "}, {"text": "part2"}]}
        assert _extract_response_text(message) == "part1 part2"


# ---------------------------------------------------------------------------
# Output guarding
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
        scan_result.flagged = True
        dome_mock.async_guard_output.return_value = scan_result

        provider = DomeHookProvider(dome_mock, agent_id="a1", team_id="t1")
        event = _make_after_event("toxic response")

        await provider._guard_output(event)

        assert event.stop_response.message["content"] == [
            {"text": DEFAULT_OUTPUT_BLOCKED_MESSAGE}
        ]
        dome_mock.async_guard_output.assert_called_once_with(
            "toxic response", agent_id="a1"
        )

    @pytest.mark.asyncio
    async def test_clean_output_passes_through(self, dome_mock):
        scan_result = MagicMock()
        scan_result.flagged = False
        dome_mock.async_guard_output.return_value = scan_result

        provider = DomeHookProvider(dome_mock)
        event = _make_after_event("normal response")

        await provider._guard_output(event)

        assert event.stop_response.message["content"] == [{"text": "normal response"}]

    @pytest.mark.asyncio
    async def test_no_stop_response_skips_scan(self, dome_mock):
        provider = DomeHookProvider(dome_mock)
        event = _make_after_event(None)

        await provider._guard_output(event)

        dome_mock.async_guard_output.assert_not_called()

    @pytest.mark.asyncio
    async def test_custom_output_blocked_message(self, dome_mock):
        scan_result = MagicMock()
        scan_result.flagged = True
        dome_mock.async_guard_output.return_value = scan_result

        custom_msg = "Custom output block"
        provider = DomeHookProvider(dome_mock, output_blocked_message=custom_msg)
        event = _make_after_event("bad output")

        await provider._guard_output(event)

        assert event.stop_response.message["content"] == [{"text": custom_msg}]
```

**Step 2: Run tests to verify they pass**

Run: `cd /Users/ciphr/Code/Vijil/vijil-dome && python -m pytest vijil_dome/tests/test_strands_hooks.py -v`
Expected: All 15 tests PASS (8 from Task 1 + 7 new)

**Step 3: Commit**

```bash
git add vijil_dome/tests/test_strands_hooks.py
git commit -m "test(strands): add output guarding tests for DomeHookProvider"
```

---

### Task 3: Add HookProvider registration test

**Files:**
- Modify: `vijil_dome/tests/test_strands_hooks.py`

**Step 1: Write registration test**

Append to `vijil_dome/tests/test_strands_hooks.py`:

```python
from strands.hooks import HookRegistry, BeforeModelCallEvent, AfterModelCallEvent


class TestHookRegistration:
    def test_register_hooks_adds_both_callbacks(self):
        dome = MagicMock()
        provider = DomeHookProvider(dome)
        registry = HookRegistry()

        provider.register_hooks(registry)

        # Both event types should have callbacks
        before_cbs = list(registry.get_callbacks_for(BeforeModelCallEvent(agent=MagicMock())))
        after_cbs = list(registry.get_callbacks_for(AfterModelCallEvent(agent=MagicMock())))
        assert len(before_cbs) == 1
        assert len(after_cbs) == 1

    def test_provider_satisfies_hook_provider_protocol(self):
        from strands.hooks import HookProvider
        dome = MagicMock()
        provider = DomeHookProvider(dome)
        assert isinstance(provider, HookProvider)
```

**Step 2: Run tests to verify they pass**

Run: `cd /Users/ciphr/Code/Vijil/vijil-dome && python -m pytest vijil_dome/tests/test_strands_hooks.py -v`
Expected: All 17 tests PASS

**Step 3: Commit**

```bash
git add vijil_dome/tests/test_strands_hooks.py
git commit -m "test(strands): add hook registration and protocol conformance tests"
```

---

### Task 4: Update travel agent to use DomeHookProvider

**Files:**
- Modify: `vijil-travel-agent/agent.py`

This task is in the `vijil-travel-agent` repo, not `vijil-dome`.

**Step 1: Update `create_agent()` to accept hooks**

In `vijil-travel-agent/agent.py`, modify `create_agent()` (currently at line 323) to accept an optional `hooks` parameter:

```python
def create_agent(hooks: list | None = None) -> Agent:
    """Create a fresh travel agent with all tools.

    System prompt is loaded dynamically from genome file (if GENOME_PATH set),
    enabling hot-reload of Darwin mutations without agent restart.
    """
    genome = None
    genome_path = os.environ.get("GENOME_PATH")
    if genome_path:
        try:
            genome = get_current_genome()
            logger.debug(f"Loaded genome v{genome.version} for agent creation")
        except Exception as e:
            logger.warning(f"Failed to load genome: {e}")

    current_prompt = get_effective_system_prompt(genome)

    return Agent(
        name=AGENT_NAME,
        description=AGENT_DESCRIPTION,
        model=OpenAIModel(
            model_id="llama-3.1-8b-instant",
            client_args={
                "base_url": "https://api.groq.com/openai/v1",
                "api_key": os.environ.get("GROQ_API_KEY"),
            },
            params={"max_tokens": 4096},
        ),
        tools=[
            search_flights,
            web_search,
            create_booking,
            auto_rebook,
            save_traveler_profile,
            process_payment,
            redeem_points,
            check_policy_compliance,
            submit_expense,
        ],
        system_prompt=current_prompt,
        hooks=hooks or [],
    )
```

**Step 2: Replace middleware + inline guards with DomeHookProvider in `main()`**

In `main()` (line 522), replace the Dome initialization block. Remove:
- `app.add_middleware(DomeA2AMiddleware, ...)` (line 581)
- Inline input/output guard calls in `add_chat_completions_endpoint()` (lines 446-476)

Replace with `DomeHookProvider` passed to `create_agent()` via a closure:

```python
    # Initialize Dome if enabled
    dome = None
    dome_hooks = None
    dome_active = False
    team_id = os.environ.get("TEAM_ID")

    if DOME_ENABLED:
        effective_dome_config = get_effective_dome_config(startup_genome)
        try:
            from vijil_dome import Dome
            dome = Dome(effective_dome_config)

            # Unified instrumentation: split metrics + Darwin detection spans
            if telemetry_enabled and tracer and meter:
                try:
                    from vijil_dome.integrations.instrumentation.otel_instrumentation import instrument_dome
                    instrument_dome(dome, handler=None, tracer=tracer, meter=meter)
                    logger.info("Dome instrumented via instrument_dome() (split metrics + Darwin spans)")
                except Exception as e:
                    logger.warning(f"Failed to instrument Dome: {e}")

            from vijil_dome.integrations.strands import DomeHookProvider
            dome_hooks = DomeHookProvider(dome, agent_id=AGENT_ID, team_id=team_id or "")
            dome_active = True
            logger.info("Dome guardrails ENABLED via Strands hooks")

            _notify_console_dome_active()

        except ImportError:
            logger.error("DOME_ENABLED=1 but vijil-dome not installed!")
```

Update `create_agent()` calls to pass hooks. Change the factory passed to `create_concurrent_a2a_app`:

```python
    # Create agent factory with dome hooks
    def agent_factory():
        return create_agent(hooks=[dome_hooks] if dome_hooks else None)

    # Create concurrent A2A app
    app = create_concurrent_a2a_app(agent_factory, host, port)
```

**Step 3: Simplify `add_chat_completions_endpoint()`**

Remove the `dome` parameter and all inline guard calls. The function becomes:

```python
def add_chat_completions_endpoint(app: Any) -> None:
    """Register /v1/chat/completions on the FastAPI app.

    Dome guards are handled by DomeHookProvider at the Strands model level,
    so this endpoint only handles protocol translation.
    """
    from pydantic import BaseModel

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatCompletionRequest(BaseModel):
        model: str = "llama-3.1-8b-instant"
        messages: list[ChatMessage]
        temperature: float = 1.0
        max_tokens: int | None = None

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        user_messages = [m for m in request.messages if m.role == "user"]
        if not user_messages:
            return JSONResponse(status_code=400, content={"error": "No user message found"})
        user_text = user_messages[-1].content

        # Run Strands agent in thread pool (hooks handle guarding)
        try:
            agent = create_agent(hooks=[dome_hooks] if dome_hooks else None)
            result = await asyncio.to_thread(agent, user_text)
            response_text = str(result)
        except Exception as e:
            logger.warning(f"Agent execution failed in chat completions: {e}")
            response_text = ConcurrentA2AExecutor.ERROR_MESSAGE

        return _chat_response(response_text, model=request.model)

    logger.info("Chat completions endpoint registered at /v1/chat/completions")
```

**Step 4: Remove DomeA2AMiddleware import and usage**

Remove from `main()`:
- `from dome_a2a import DomeA2AMiddleware` (line 580)
- `app.add_middleware(DomeA2AMiddleware, dome=dome, agent_id=AGENT_ID, team_id=team_id)` (line 581)

Remove the `dome_a2a` import from `add_chat_completions_endpoint()`:
- `from dome_a2a import DEFAULT_BLOCKED_MESSAGE as _DOME_BLOCKED_MESSAGE` (line 426)

**Step 5: Verify the travel agent starts**

Run: `cd /Users/ciphr/Code/Vijil/vijil-travel-agent && DOME_ENABLED=0 python agent.py`
Expected: Agent starts normally on port 9000 (baseline mode, no Dome)

**Step 6: Commit**

```bash
git add agent.py
git commit -m "feat: replace Dome middleware with Strands DomeHookProvider

Moves Dome guarding from A2A middleware + inline code to Strands
hooks. Both input and output are now guarded at the model level,
covering all transport protocols (A2A, /v1/chat/completions)."
```

---

### Task 5: Delete local `dome_a2a.py` from travel agent

**Files:**
- Delete: `vijil-travel-agent/dome_a2a.py`

**Step 1: Verify no remaining imports of dome_a2a**

Run: `grep -r 'dome_a2a' /Users/ciphr/Code/Vijil/vijil-travel-agent/ --include='*.py'`
Expected: No matches (after Task 4 removed all imports)

**Step 2: Delete the file**

```bash
rm /Users/ciphr/Code/Vijil/vijil-travel-agent/dome_a2a.py
```

**Step 3: Commit**

```bash
git add -A dome_a2a.py
git commit -m "chore: remove local dome_a2a.py (now in vijil-dome package)"
```

---

## Summary

| Task | Repo | What | Tests |
|------|------|------|-------|
| 1 | vijil-dome | DomeHookProvider + input guard | 8 tests |
| 2 | vijil-dome | Output guard tests | 7 tests |
| 3 | vijil-dome | Registration + protocol tests | 2 tests |
| 4 | vijil-travel-agent | Replace middleware with hooks | Manual startup test |
| 5 | vijil-travel-agent | Delete dead `dome_a2a.py` | grep verification |

Total: 17 unit tests, 5 commits across 2 repos.
