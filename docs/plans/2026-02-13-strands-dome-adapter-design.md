# Dome Strands Adapter Design

**Date:** 2026-02-13
**Module:** vijil-dome / integrations
**Status:** Approved

## Problem

Dome has framework adapters for ADK, LangChain, and MCP, but not for Strands. Agents built with the Strands SDK (like vijil-travel-agent) lack a proper integration point for Dome guardrails. The current workaround puts guarding in the A2A transport middleware, which is the wrong layer — A2A is a protocol adapter, not a framework integration.

This creates a real gap: A2A responses are not output-guarded because the A2A middleware only wires up `dome.input_guardrail`. The `/v1/chat/completions` handler has inline output guarding, but that's endpoint-specific code, not a reusable adapter.

## Solution

Build a `DomeHookProvider` that implements Strands' `HookProvider` protocol, registering hooks for `BeforeModelCallEvent` (input guarding) and `AfterModelCallEvent` (output guarding). This follows the same pattern as the existing ADK adapter (`integrations/adk/callbacks.py`), adapted to Strands' hook system.

Guarding at the framework level is transport-agnostic — hooks fire on every model call regardless of whether the request came via A2A, `/v1/chat/completions`, or a direct Python call. One integration point covers all protocols.

## Design

### Module structure

```
vijil_dome/integrations/strands/
├── __init__.py          # Exports DomeHookProvider, constants
└── hooks.py             # DomeHookProvider implementation
```

### DomeHookProvider

```python
class DomeHookProvider(HookProvider):
    """Dome guardrail integration for Strands agents.

    Registers hooks that scan model input and output through Dome's
    guardrails. Flagged content is replaced with a blocked message.
    """

    def __init__(
        self,
        dome: Dome,
        agent_id: str = "",
        team_id: str = "",
        input_blocked_message: str = DEFAULT_INPUT_BLOCKED_MESSAGE,
        output_blocked_message: str = DEFAULT_OUTPUT_BLOCKED_MESSAGE,
    ): ...

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(BeforeModelCallEvent, self._guard_input)
        registry.add_callback(AfterModelCallEvent, self._guard_output)
```

### Input guarding (`_guard_input`)

Triggered by `BeforeModelCallEvent`:

1. Walk `event.agent.messages` in reverse to find the last user message.
2. Extract text from content blocks.
3. Call `dome.input_guardrail.async_scan(text, agent_id=..., team_id=...)`.
4. If flagged: replace text content blocks in that message with `input_blocked_message`.
5. If clean: do nothing.

The LLM sees either the original message or the replacement. If replaced, the model responds to the safe text naturally.

### Output guarding (`_guard_output`)

Triggered by `AfterModelCallEvent`:

1. Check `event.stop_response` exists (skip error cases).
2. Extract text from `event.stop_response.message["content"]` blocks.
3. Call `dome.output_guardrail.async_scan(text, agent_id=..., team_id=...)`.
4. If flagged: replace text content blocks with `output_blocked_message`.
5. If clean: do nothing.

### Blocked message constants

```python
DEFAULT_INPUT_BLOCKED_MESSAGE = (
    "I'm sorry, but I can't help with that request. "
    "It appears to contain content that violates my usage policies. "
    "I'd be happy to assist you with a different question."
)

DEFAULT_OUTPUT_BLOCKED_MESSAGE = (
    "The generated response was blocked by the guardrail policy. "
    "I'd be happy to assist you with a different request."
)
```

### Agent usage

```python
from vijil_dome import Dome
from vijil_dome.integrations.strands import DomeHookProvider

dome = Dome(config)
dome_hooks = DomeHookProvider(dome, agent_id="...", team_id="...")

agent = Agent(
    name="My Agent",
    model=OpenAIModel(...),
    tools=[...],
    hooks=[dome_hooks],
)
```

### Impact on travel agent

The travel agent (`vijil-travel-agent/agent.py`) changes:
- Drops `DomeA2AMiddleware` from `app.add_middleware()`.
- Passes `DomeHookProvider` via `hooks=` in `create_agent()`.
- Removes inline input/output guard calls from the `/v1/chat/completions` handler.
- Both A2A and chat completions paths are guarded because hooks fire at the model level.

### Impact on A2A middleware

The `DomeA2AMiddleware` in `integrations/a2a/` remains as-is. It becomes optional defense-in-depth for agents that want early HTTP-level rejection (avoid wasting compute on flagged requests). It is no longer the primary guard for Strands-based agents.

## Strands Hook API Reference

```python
# Hook events used
BeforeModelCallEvent:
    agent: Agent           # Full agent instance (messages, state, etc.)

AfterModelCallEvent:
    agent: Agent
    stop_response: Optional[ModelStopResponse]  # On success
    exception: Optional[Exception]               # On failure

ModelStopResponse:
    message: Message       # {"role": str, "content": list[ContentBlock]}
    stop_reason: StopReason

# Content block text format
{"text": "the message text"}

# Registration
class HookProvider(Protocol):
    def register_hooks(self, registry: HookRegistry, **kwargs) -> None: ...

HookRegistry.add_callback(event_type: Type[HookEvent], callback: HookCallback)
```

## Out of Scope

- Streaming response interception (non-streaming only for now).
- Strands tool call guarding (`BeforeToolCallEvent` / `AfterToolCallEvent`).
- Automatic telemetry instrumentation (handled separately by `instrument_dome()`).
