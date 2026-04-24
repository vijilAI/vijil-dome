"""Strands SDK adapter — HookProvider for trust enforcement.

Strands agents accept a list of HookProvider instances that register
callbacks for model and tool lifecycle events. This adapter creates a
HookProvider that routes through the TrustRuntime for content Guards,
tool-level MAC, and audit.

Strands is NOT a dependency of this package. Imports are deferred.
"""

from __future__ import annotations

import logging
from typing import Any

from vijil_dome.trust.constraints import AgentConstraints
from vijil_dome.trust.runtime import TrustRuntime

logger = logging.getLogger(__name__)


def secure_agent(
    agent: Any,
    *,
    client: Any | None = None,
    agent_id: str,
    constraints: AgentConstraints | dict[str, Any] | None = None,
    manifest: Any = None,
    mode: str = "warn",
) -> Any:
    """Add trust enforcement to a Strands Agent.

    Injects trust hooks into the agent's ``hooks`` list and returns
    the same agent (modified in place). Consistent with the ADK adapter
    which also modifies and returns the agent.

    Usage::

        from vijil_dome.trust.adapters.strands import secure_agent
        secure_agent(agent, agent_id="my-agent", constraints=constraints)

    Parameters
    ----------
    agent:
        A Strands ``Agent`` instance.
    client:
        Optional Vijil client. Not needed if *constraints* is provided.
    agent_id:
        The registered agent ID.
    constraints:
        Agent constraints (dict or AgentConstraints). Skips Console
        fetch if provided.
    manifest:
        Optional signed tool manifest.
    mode:
        ``"warn"`` or ``"enforce"``.

    Returns
    -------
    The same agent, with trust hooks injected.
    """
    hooks = create_trust_hooks(
        client=client,
        agent_id=agent_id,
        constraints=constraints,
        manifest=manifest,
        mode=mode,
    )

    # Inject hooks into the agent
    existing = getattr(agent, "hooks", None)
    if existing is None:
        agent.hooks = [hooks]
    elif isinstance(existing, list):
        existing.append(hooks)
    else:
        agent.hooks = [existing, hooks]

    # Store runtime on agent for external access
    agent._vijil_runtime = hooks.runtime
    agent._vijil_attestation = hooks.attestation

    logger.info(
        "Trust enforcement injected into Strands agent (mode=%s)",
        mode,
    )
    return agent


def create_trust_hooks(
    *,
    client: Any | None = None,
    agent_id: str,
    constraints: AgentConstraints | dict[str, Any] | None = None,
    manifest: Any = None,
    mode: str = "warn",
) -> Any:
    """Create a Strands HookProvider with trust enforcement.

    Low-level API. Returns a ``TrustHookProvider`` for manual injection.
    Prefer ``secure_agent()`` for the common case.

    Usage::

        from vijil_dome.trust.adapters.strands import create_trust_hooks
        from strands import Agent

        hooks = create_trust_hooks(agent_id="my-agent", constraints=constraints)
        agent = Agent(hooks=[hooks])

    Parameters
    ----------
    client:
        Optional Vijil client. Not needed if *constraints* is provided.
    agent_id:
        The registered agent ID.
    constraints:
        Agent constraints (dict or AgentConstraints). Skips Console
        fetch if provided.
    manifest:
        Optional signed tool manifest.
    mode:
        ``"warn"`` or ``"enforce"``.
    """
    try:
        from strands.hooks import (
            AfterModelCallEvent,
            AfterToolCallEvent,
            BeforeModelCallEvent,
            BeforeToolCallEvent,
            HookProvider,
            HookRegistry,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Strands SDK is not installed. Install with: pip install strands-agents"
        ) from exc

    runtime = TrustRuntime(
        client=client,
        agent_id=agent_id,
        constraints=constraints,
        manifest=manifest,
        mode=mode,
    )

    class TrustHookProvider(HookProvider):
        """Vijil Trust Runtime hook provider for Strands agents."""

        def __init__(self) -> None:
            self.runtime = runtime
            self.attestation = runtime.attest()

        def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
            registry.add_callback(BeforeModelCallEvent, self._guard_input)
            registry.add_callback(AfterModelCallEvent, self._guard_output)
            registry.add_callback(BeforeToolCallEvent, self._check_tool)
            registry.add_callback(AfterToolCallEvent, self._guard_tool_response)

        async def _guard_input(self, event: BeforeModelCallEvent) -> None:
            """Guard user input before it reaches the model."""
            messages = getattr(event.agent, "messages", [])
            text = _extract_last_user_text(messages)
            if not text:
                return

            result = await runtime.aguard_input(text)
            if result.flagged and result.enforced:
                # Replace the user message with the guarded response
                if messages:
                    messages[-1]["content"] = [
                        {"text": result.guarded_response or "Request blocked by policy."}
                    ]
            elif result.flagged:
                logger.warning(
                    "Input flagged (warn mode, score=%.2f): %s",
                    result.score,
                    text[:100],
                )

        async def _guard_output(self, event: AfterModelCallEvent) -> None:
            """Guard model output before it reaches the user."""
            stop = getattr(event, "stop_response", None)
            if stop is None:
                return
            message = getattr(stop, "message", None)
            if message is None:
                return

            text = _extract_response_text(message)
            if not text:
                return

            result = await runtime.aguard_output(text)
            if result.flagged and result.enforced:
                message["content"] = [
                    {"text": result.guarded_response or "Response blocked by policy."}
                ]
            elif result.flagged:
                logger.warning("Output flagged (warn mode, score=%.2f)", result.score)

        async def _check_tool(self, event: BeforeToolCallEvent) -> None:
            """Check tool MAC before execution."""
            tool_use = event.tool_use
            tool_name = tool_use.get("name", "unknown")

            mac_result = runtime.check_tool_call(tool_name, dict(tool_use.get("input", {})))

            if not mac_result.permitted and runtime.mode == "enforce":
                event.cancel_tool = f"Tool '{tool_name}' denied: {mac_result.error}"
                logger.warning("Tool '%s' blocked (enforce mode)", tool_name)
            elif not mac_result.permitted:
                logger.warning(
                    "Tool '%s' would be denied in enforce mode: %s",
                    tool_name,
                    mac_result.error,
                )

        async def _guard_tool_response(self, event: AfterToolCallEvent) -> None:
            """Guard tool response content."""
            tool_result = getattr(event, "result", None)
            if tool_result is None:
                return

            # Extract text from the tool result
            content = tool_result.get("content", [])
            texts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and "text" in block
            ]
            text = " ".join(texts)
            if not text:
                return

            tool_name = event.tool_use.get("name", "unknown")
            result = await runtime.aguard_output(text)  # Guard tool response through output guards
            if result.flagged and result.enforced:
                tool_result["content"] = [
                    {"text": result.guarded_response or "Tool response blocked by policy."}
                ]
            elif result.flagged:
                logger.warning(
                    "Tool '%s' response flagged (warn mode, score=%.2f)",
                    tool_name,
                    result.score,
                )

    return TrustHookProvider()


def _extract_last_user_text(messages: list[dict[str, Any]]) -> str | None:
    """Extract text from the last user message."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        parts = msg.get("content", [])
        texts = [
            block["text"]
            for block in parts
            if isinstance(block, dict) and "text" in block
        ]
        if texts:
            return " ".join(texts)
    return None


def _extract_response_text(message: dict[str, Any]) -> str | None:
    """Extract text from a model response message."""
    parts = message.get("content", [])
    texts = [
        block.get("text", "")
        for block in parts
        if isinstance(block, dict) and "text" in block
    ]
    return " ".join(texts) if texts else None
