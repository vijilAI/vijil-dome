"""LangGraph adapter — wraps a compiled graph with trust enforcement.

LangGraph is NOT a dependency of this package.  All interactions use
duck typing (``Any``) so that the adapter works without importing
langgraph at all.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from vijil_dome.trust.runtime import TrustRuntime

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Message helpers
# ------------------------------------------------------------------

def _extract_text(message: Any) -> str:
    """Extract plain text from a LangGraph message (str or dict)."""
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        return str(message.get("content", ""))
    # LangGraph message objects typically expose .content
    return str(getattr(message, "content", str(message)))


def _extract_last_user_text(input_dict: dict[str, Any]) -> str:
    """Pull the text of the last message from the input dict."""
    messages = input_dict.get("messages", [])
    if not messages:
        return ""
    return _extract_text(messages[-1])


def _extract_output_text(result: dict[str, Any]) -> str:
    """Pull the text of the last message from the graph output."""
    messages = result.get("messages", [])
    if not messages:
        return ""
    return _extract_text(messages[-1])


# ------------------------------------------------------------------
# SecureGraph
# ------------------------------------------------------------------

class SecureGraph:
    """A compiled LangGraph wrapped with Vijil trust enforcement.

    Intercepts ``invoke`` / ``stream`` calls to run Dome input and
    output guards, and exposes the attestation result.
    """

    def __init__(self, *, graph: Any, runtime: TrustRuntime) -> None:
        self._graph = graph
        self._runtime = runtime
        self._attestation = runtime.attest()

    @property
    def runtime(self) -> TrustRuntime:
        return self._runtime

    @property
    def attestation(self) -> Any:
        return self._attestation

    # ------------------------------------------------------------------
    # invoke
    # ------------------------------------------------------------------

    def invoke(self, input: dict[str, Any], config: Any = None) -> dict[str, Any]:
        """Run the graph synchronously with input/output guard passes."""
        user_text = _extract_last_user_text(input)

        # Input guard
        input_result = self._runtime.guard_input(user_text)
        if input_result.flagged and input_result.enforced:
            return {"messages": [input_result.guarded_response]}

        # Delegate to the real graph
        result = self._graph.invoke(input, config)

        # Output guard
        output_text = _extract_output_text(result)
        output_result = self._runtime.guard_output(output_text)
        if output_result.flagged and output_result.enforced:
            return {"messages": [output_result.guarded_response]}

        return result  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # stream
    # ------------------------------------------------------------------

    def stream(self, input: dict[str, Any], config: Any = None) -> Iterator[Any]:
        """Stream graph output with input guard and post-stream output guard."""
        user_text = _extract_last_user_text(input)

        # Input guard
        input_result = self._runtime.guard_input(user_text)
        if input_result.flagged and input_result.enforced:
            yield {"messages": [input_result.guarded_response]}
            return

        # Yield chunks, accumulating output for post-stream guard
        accumulated_texts: list[str] = []
        for chunk in self._graph.stream(input, config):
            chunk_text = _extract_output_text(chunk)
            if chunk_text:
                accumulated_texts.append(chunk_text)
            yield chunk

        # Post-stream output guard (best-effort — cannot retract chunks)
        full_output = " ".join(accumulated_texts)
        if full_output:
            output_result = self._runtime.guard_output(full_output)
            if output_result.flagged:
                logger.warning(
                    "Output guard flagged streamed response (cannot retract): %s",
                    output_result.guarded_response,
                )

    # ------------------------------------------------------------------
    # async variants (thin wrappers for now)
    # ------------------------------------------------------------------

    async def ainvoke(self, input: dict[str, Any], config: Any = None) -> dict[str, Any]:
        """Async invoke — delegates to sync invoke for now."""
        return self.invoke(input, config)

    async def astream(self, input: dict[str, Any], config: Any = None) -> Any:
        """Async stream — wraps sync stream for now."""
        for chunk in self.stream(input, config):
            yield chunk


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def secure_graph(
    graph: Any,
    *,
    client: Any | None = None,
    agent_id: str,
    constraints: Any = None,
    manifest: Any = None,
    mode: str = "warn",
    **compile_kwargs: Any,
) -> SecureGraph:
    """Create a :class:`SecureGraph` from a LangGraph graph.

    If *graph* has a ``compile`` method (i.e. it is an uncompiled
    ``StateGraph``), it will be compiled first.  An already-compiled
    graph is used directly.

    Parameters
    ----------
    graph:
        A LangGraph ``StateGraph`` (uncompiled) or compiled graph.
    client:
        A :class:`~vijil.client.VijilClient` instance. Optional if
        *constraints* is provided directly.
    agent_id:
        The registered agent ID in the Vijil Console.
    constraints:
        An :class:`AgentConstraints` or dict. If provided, skips
        Console fetch. Useful for development and testing.
    manifest:
        Optional tool manifest (path or :class:`ToolManifest`).
    mode:
        ``"warn"`` (log violations) or ``"enforce"`` (block violations).
    **compile_kwargs:
        Passed to ``graph.compile()`` when compiling.
    """
    runtime = TrustRuntime(
        client=client,
        agent_id=agent_id,
        constraints=constraints,
        manifest=manifest,
        mode=mode,
    )

    # Compile if needed
    if hasattr(graph, "compile"):
        compiled = graph.compile(**compile_kwargs)
    else:
        compiled = graph

    # Attempt tool wrapping (best-effort — LangGraph internals vary)
    try:
        if hasattr(compiled, "tools"):
            compiled.tools = runtime.wrap_tools(compiled.tools)
    except Exception:
        logger.debug("Could not wrap graph tools — skipping tool-level enforcement.")

    return SecureGraph(graph=compiled, runtime=runtime)


# ------------------------------------------------------------------
# Adapter registry
# ------------------------------------------------------------------

from vijil_dome.trust.adapters.base import BaseAdapter, register_adapter


@register_adapter("langgraph")
class LangGraphAdapter(BaseAdapter):
    @classmethod
    def detect(cls, agent: Any) -> bool:
        module = type(agent).__module__ or ""
        if module.startswith("langgraph"):
            return True
        return hasattr(agent, "compile") or hasattr(agent, "get_graph")

    @classmethod
    def wrap(cls, agent: Any, **kwargs: Any) -> Any:
        kwargs.pop("policy", None)
        return secure_graph(agent, **kwargs)
