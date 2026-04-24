"""Unified adapter — one function for all frameworks.

``secure_agent()`` detects the framework from the object type and
dispatches to the correct adapter. Developers use one name regardless
of whether they build with LangGraph, Google ADK, or Strands.

Usage::

    from vijil import secure_agent

    # LangGraph
    app = secure_agent(graph, agent_id="my-agent")

    # Google ADK
    secure_agent(adk_agent, agent_id="my-agent")

    # Strands
    hooks = secure_agent(strands_agent, agent_id="my-agent")
"""

from __future__ import annotations

import logging
from typing import Any

from vijil_dome.trust.constraints import AgentConstraints

logger = logging.getLogger(__name__)


def secure_agent(
    agent: Any,
    *,
    client: Any | None = None,
    agent_id: str,
    constraints: AgentConstraints | dict[str, Any] | None = None,
    manifest: Any = None,
    mode: str = "warn",
    **kwargs: Any,
) -> Any:
    """Add trust enforcement to any supported agent framework.

    Detects the framework from the agent object and dispatches to the
    correct adapter:

    - **LangGraph** ``StateGraph`` or compiled graph → ``secure_graph()``
    - **Google ADK** ``Agent`` → ``secure_agent()`` (callback injection)
    - **Strands** ``Agent`` → ``create_trust_hooks()`` (HookProvider)

    For LangGraph, returns a ``SecureGraph`` (replaces ``graph.compile()``).
    For ADK, modifies the agent in place and returns it.
    For Strands, returns a ``TrustHookProvider`` to pass to the agent's
    ``hooks`` parameter.

    Parameters
    ----------
    agent:
        The agent object. Type determines which adapter is used.
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
    **kwargs:
        Passed through to the framework-specific adapter (e.g.,
        ``compile_kwargs`` for LangGraph).
    """
    framework = _detect_framework(agent)

    if framework == "langgraph":
        from vijil_dome.trust.adapters.langgraph import secure_graph

        return secure_graph(
            agent,
            client=client,
            agent_id=agent_id,
            constraints=constraints,
            manifest=manifest,
            mode=mode,
            **kwargs,
        )

    if framework == "adk":
        from vijil_dome.trust.adapters.adk import secure_agent as _secure_adk

        return _secure_adk(
            agent,
            client=client,
            agent_id=agent_id,
            constraints=constraints,
            manifest=manifest,
            mode=mode,
        )

    if framework == "strands":
        from vijil_dome.trust.adapters.strands import secure_agent as _secure_strands

        return _secure_strands(
            agent,
            client=client,
            agent_id=agent_id,
            constraints=constraints,
            manifest=manifest,
            mode=mode,
        )

    raise TypeError(
        f"Unsupported agent type: {type(agent).__module__}.{type(agent).__qualname__}. "
        f"secure_agent() supports LangGraph (StateGraph), Google ADK (Agent), "
        f"and Strands (Agent). For other frameworks, use the framework-specific "
        f"adapter directly (vijil.adapters.langgraph, vijil.adapters.adk, "
        f"vijil.adapters.strands)."
    )


def _detect_framework(agent: Any) -> str:
    """Detect the agent framework from the object type."""
    module = type(agent).__module__ or ""

    # Google ADK: google.adk.agents.*
    if module.startswith("google.adk") or module.startswith("google.genai"):
        return "adk"

    # Strands: strands.agent.*
    if module.startswith("strands"):
        return "strands"

    # LangGraph: langgraph.graph.* (StateGraph, CompiledGraph)
    if module.startswith("langgraph"):
        return "langgraph"

    # Duck typing fallback: LangGraph graphs have .compile() or .get_graph()
    if hasattr(agent, "compile") or hasattr(agent, "get_graph"):
        return "langgraph"

    # Duck typing fallback: ADK agents have .before_model_callback
    if hasattr(agent, "before_model_callback") and hasattr(agent, "before_tool_callback"):
        return "adk"

    return "unknown"
