"""Google ADK adapter — injects trust enforcement via agent callbacks.

ADK agents support callback hooks on model calls and tool calls.
This adapter creates callbacks that route through the TrustRuntime
for input/output Guards and tool-level MAC.

Google ADK is NOT a dependency of this package. Imports are deferred
and the module gracefully reports if ADK is missing.
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
    """Add trust enforcement to a Google ADK Agent via callbacks.

    Unlike the LangGraph adapter (which wraps the graph), this adapter
    injects callbacks directly into the ADK Agent's hook fields:
    - ``before_model_callback``: Dome input Guard on user messages
    - ``after_model_callback``: Dome output Guard on model responses
    - ``before_tool_callback``: Tool MAC (permit/deny) + identity check
    - ``after_tool_callback``: Dome Guard on tool responses

    Parameters
    ----------
    agent:
        A Google ADK ``Agent`` instance.
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
    The same agent, with trust callbacks injected. The agent is
    modified in place and also returned for convenience.
    """
    try:
        from google.adk.models.llm_response import LlmResponse
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "Google ADK is not installed. Install with: pip install google-adk"
        ) from exc

    runtime = TrustRuntime(
        client=client,
        agent_id=agent_id,
        constraints=constraints,
        manifest=manifest,
        mode=mode,
    )

    # Store runtime on the agent for external access
    agent._vijil_runtime = runtime
    agent._vijil_attestation = runtime.attest()

    # ------------------------------------------------------------------
    # before_model_callback — input Guard
    # ------------------------------------------------------------------

    def before_model(callback_context: Any, llm_request: Any) -> Any:
        """Guard user input before it reaches the LLM."""
        user_text = ""
        if hasattr(llm_request, "contents") and llm_request.contents:
            for content in reversed(llm_request.contents):
                if getattr(content, "role", None) == "user" and getattr(content, "parts", None):
                    for part in content.parts:
                        user_text += getattr(part, "text", "") or ""
                    break

        if not user_text:
            return None

        result = runtime.guard_input(user_text)
        if result.flagged and result.enforced:
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=result.guarded_response or "Request blocked by policy.")],
                )
            )
        if result.flagged:
            logger.warning("Input flagged (warn mode, score=%.2f): %s", result.score, user_text[:100])

        return None

    # ------------------------------------------------------------------
    # after_model_callback — output Guard
    # ------------------------------------------------------------------

    def after_model(callback_context: Any, llm_response: Any) -> Any:
        """Guard model output before it reaches the user."""
        model_text = ""
        content = getattr(llm_response, "content", None)
        if content and getattr(content, "role", None) == "model" and getattr(content, "parts", None):
            for part in content.parts:
                model_text += getattr(part, "text", "") or ""

        if not model_text:
            return None

        result = runtime.guard_output(model_text)
        if result.flagged and result.enforced:
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=result.guarded_response or "Response blocked by policy.")],
                )
            )
        if result.flagged:
            logger.warning("Output flagged (warn mode, score=%.2f)", result.score)

        return None

    # ------------------------------------------------------------------
    # before_tool_callback — tool MAC
    # ------------------------------------------------------------------

    def before_tool(tool: Any, args: dict[str, Any], tool_context: Any) -> Any:
        """Check tool MAC before execution."""
        tool_name = getattr(tool, "name", getattr(tool, "__name__", str(tool)))
        mac_result = runtime.check_tool_call(tool_name, args)

        if not mac_result.permitted and runtime.mode == "enforce":
            # Return a dict to short-circuit the tool call
            return {"error": f"Tool '{tool_name}' denied: {mac_result.error}"}
        if not mac_result.permitted:
            logger.warning("Tool '%s' would be denied in enforce mode: %s", tool_name, mac_result.error)

        return None  # Allow tool execution

    # ------------------------------------------------------------------
    # after_tool_callback — tool response Guard
    # ------------------------------------------------------------------

    def after_tool(tool: Any, args: dict[str, Any], tool_context: Any, tool_response: dict[str, Any]) -> Any:
        """Guard tool response content."""
        response_text = str(tool_response) if tool_response else ""
        if not response_text:
            return None

        result = runtime.guard_tool_response(
            getattr(tool, "name", str(tool)),
            response_text,
        )
        if result.flagged and result.enforced:
            return {"result": result.guarded_response or "Tool response blocked by policy."}
        if result.flagged:
            logger.warning("Tool response flagged (warn mode, score=%.2f)", result.score)

        return None

    # ------------------------------------------------------------------
    # Inject callbacks into the agent
    # ------------------------------------------------------------------

    # ADK supports lists of callbacks — append to existing ones
    def _append_callback(agent: Any, field: str, callback: Any) -> None:
        existing = getattr(agent, field, None)
        if existing is None:
            setattr(agent, field, callback)
        elif isinstance(existing, list):
            existing.append(callback)
        else:
            setattr(agent, field, [existing, callback])

    _append_callback(agent, "before_model_callback", before_model)
    _append_callback(agent, "after_model_callback", after_model)
    _append_callback(agent, "before_tool_callback", before_tool)
    _append_callback(agent, "after_tool_callback", after_tool)

    logger.info(
        "Trust enforcement injected into ADK agent '%s' (mode=%s)",
        getattr(agent, "name", agent_id),
        mode,
    )

    return agent
