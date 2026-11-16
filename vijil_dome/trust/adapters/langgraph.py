"""LangGraph adapter — wraps a compiled graph with trust enforcement.

LangGraph is NOT a dependency of this package.  All interactions use
duck typing (``Any``) so that the adapter works without importing
langgraph at all.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from vijil_dome.trust.adapters.base import BaseAdapter, register_adapter
from vijil_dome.trust.runtime import TrustRuntime
from vijil_dome.trust.signing import BeaconSigner

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Message helpers
# ------------------------------------------------------------------

def _extract_text(message: Any) -> str:
    """Extract plain text from a LangGraph message (str or dict)."""
    if isinstance(message, str):
        return message
    content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", message)
    return _content_text(content)


def _content_text(content: Any) -> str:
    """Text of a message's content: a string, or the text blocks of a content-block list.

    Non-text blocks (thinking signatures, images, tool_use) are skipped, so guards scan
    what the model said, not opaque encoded payloads.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type", "text") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(part for part in parts if part)
    return str(content)


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

    def invoke(self, input: dict[str, Any], config: Any = None, **kwargs: Any) -> dict[str, Any]:
        """Run the graph synchronously with input/output guard passes.

        Extra keyword arguments (e.g. LangGraph's ``context=``) are passed through.
        """
        user_text = _extract_last_user_text(input)

        # Input guard
        input_result = self._runtime.guard_input(user_text)
        if input_result.flagged and input_result.enforced:
            return {"messages": [input_result.guarded_response]}

        # Delegate to the real graph
        result = self._graph.invoke(input, config, **kwargs)

        # Output guard
        output_text = _extract_output_text(result)
        output_result = self._runtime.guard_output(output_text)
        if output_result.flagged and output_result.enforced:
            return {"messages": [output_result.guarded_response]}

        return result  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # stream
    # ------------------------------------------------------------------

    def stream(self, input: dict[str, Any], config: Any = None, **kwargs: Any) -> Iterator[Any]:
        """Stream graph output with input guard and post-stream output guard."""
        user_text = _extract_last_user_text(input)

        # Input guard
        input_result = self._runtime.guard_input(user_text)
        if input_result.flagged and input_result.enforced:
            yield {"messages": [input_result.guarded_response]}
            return

        # Yield chunks, accumulating output for post-stream guard
        accumulated_texts: list[str] = []
        for chunk in self._graph.stream(input, config, **kwargs):
            chunk_text = _extract_output_text(chunk)
            if chunk_text:
                accumulated_texts.append(chunk_text)
            yield chunk

        # Post-stream output guard (best-effort — cannot retract chunks)
        full_output = " ".join(accumulated_texts)
        if full_output:
            output_result = self._runtime.guard_output(full_output)
            if output_result.flagged:
                if self._runtime.mode == "enforce":
                    logger.warning(
                        "Output guard flagged streamed response in enforce mode "
                        "(chunks already yielded, cannot retract): %s",
                        output_result.guarded_response,
                    )
                else:
                    logger.info(
                        "Output guard flagged streamed response (shadow mode): %s",
                        output_result.guarded_response,
                    )

    # ------------------------------------------------------------------
    # async variants
    #
    # These must not call the sync path: Dome's sync guards use
    # asyncio.run(), which raises inside a running event loop, and graphs
    # with async-only nodes cannot be run by graph.invoke().
    # ------------------------------------------------------------------

    async def ainvoke(self, input: dict[str, Any], config: Any = None, **kwargs: Any) -> dict[str, Any]:
        """Run the graph asynchronously with input/output guard passes."""
        user_text = _extract_last_user_text(input)

        input_result = await self._runtime.aguard_input(user_text)
        if input_result.flagged and input_result.enforced:
            return {"messages": [input_result.guarded_response]}

        result = await self._graph.ainvoke(input, config, **kwargs)

        output_result = await self._runtime.aguard_output(_extract_output_text(result))
        if output_result.flagged and output_result.enforced:
            return {"messages": [output_result.guarded_response]}

        return result  # type: ignore[no-any-return]

    async def astream(self, input: dict[str, Any], config: Any = None, **kwargs: Any) -> Any:
        """Async stream with input guard and post-stream output guard."""
        user_text = _extract_last_user_text(input)

        input_result = await self._runtime.aguard_input(user_text)
        if input_result.flagged and input_result.enforced:
            yield {"messages": [input_result.guarded_response]}
            return

        accumulated_texts: list[str] = []
        async for chunk in self._graph.astream(input, config, **kwargs):
            chunk_text = _extract_output_text(chunk) if isinstance(chunk, dict) else ""
            if chunk_text:
                accumulated_texts.append(chunk_text)
            yield chunk

        full_output = " ".join(accumulated_texts)
        if full_output:
            output_result = await self._runtime.aguard_output(full_output)
            if output_result.flagged:
                logger.warning(
                    "Output guard flagged streamed response (chunks already yielded, cannot retract): %s",
                    output_result.guarded_response,
                )


# ------------------------------------------------------------------
# Tool enforcement inside LangGraph ToolNodes
# ------------------------------------------------------------------


def _tool_message_text(message: Any) -> str | None:
    """Text of a ToolMessage's content, or None for non-ToolMessage results (e.g. Command)."""
    if getattr(message, "type", None) != "tool":
        return None
    return _content_text(message.content)


def _denied_message(call: dict[str, Any], error: str | None) -> Any:
    from langchain_core.messages import ToolMessage

    return ToolMessage(
        content=f"Tool '{call.get('name')}' was denied by Vijil Dome policy: {error}",
        name=call.get("name"),
        tool_call_id=call.get("id"),
        status="error",
    )


def _mac_check(runtime: TrustRuntime, call: dict[str, Any]) -> Any | None:
    """Return a denial ToolMessage if policy blocks this call, else None."""
    name = str(call.get("name"))
    result = runtime.check_tool_call(name, dict(call.get("args") or {}))
    if not result.permitted and result.enforced:
        return _denied_message(call, result.error)
    if not result.permitted:
        logger.warning("Tool '%s' would be denied in enforce mode: %s", name, result.error)
    return None


def _apply_tool_guard(message: Any, guard_result: Any) -> Any:
    if guard_result.flagged and guard_result.enforced:
        message.content = guard_result.guarded_response
    return message


def _wrap_tool_node(node: Any, runtime: TrustRuntime) -> bool:
    """Patch one ToolNode instance so every tool call gets MAC and its result is guarded.

    Hooks the node's per-call methods (``_run_one`` / ``_arun_one``), so it covers every
    tool the node holds, sync or async, including tools that use LangGraph's runtime
    injection. Returns False if this LangGraph version does not have those methods.
    """
    run_one = getattr(node, "_run_one", None)
    arun_one = getattr(node, "_arun_one", None)
    if run_one is None or arun_one is None:
        return False

    def guarded_run_one(call: Any, *args: Any, **kwargs: Any) -> Any:
        denied = _mac_check(runtime, call)
        if denied is not None:
            return denied
        message = run_one(call, *args, **kwargs)
        text = _tool_message_text(message)
        if text:
            message = _apply_tool_guard(message, runtime.guard_tool_response(str(call.get("name")), text))
        return message

    async def guarded_arun_one(call: Any, *args: Any, **kwargs: Any) -> Any:
        denied = _mac_check(runtime, call)
        if denied is not None:
            return denied
        message = await arun_one(call, *args, **kwargs)
        text = _tool_message_text(message)
        if text:
            message = _apply_tool_guard(message, await runtime.aguard_output(text))
        return message

    node._run_one = guarded_run_one
    node._arun_one = guarded_arun_one
    return True


def _wrap_graph_tools(compiled: Any, runtime: TrustRuntime) -> list[str]:
    """Find every ToolNode in a compiled graph and wrap it. Returns the wrapped tool names."""
    wrapped: list[str] = []
    for node in (getattr(compiled, "nodes", None) or {}).values():
        bound = getattr(node, "bound", None)
        tools_by_name = getattr(bound, "tools_by_name", None)
        if isinstance(tools_by_name, dict) and _wrap_tool_node(bound, runtime):
            wrapped.extend(tools_by_name)
    return wrapped


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
    heartbeat_interval: float | None = None,
    beacon_signer: BeaconSigner | None = None,
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
    heartbeat_interval:
        If set, auto-start an enforcement-heartbeat thread on this cadence.
    beacon_signer:
        Optional signer for heartbeat beacons (unsigned by default).
    **compile_kwargs:
        Passed to ``graph.compile()`` when compiling.
    """
    runtime = TrustRuntime(
        client=client,
        agent_id=agent_id,
        constraints=constraints,
        manifest=manifest,
        mode=mode,
        heartbeat_interval=heartbeat_interval,
        beacon_signer=beacon_signer,
    )

    # Compile if needed
    if hasattr(graph, "compile"):
        compiled = graph.compile(**compile_kwargs)
    else:
        compiled = graph

    # Tool enforcement: patch the graph's ToolNodes (MAC before each call, guards on each
    # result). Compiled LangGraph graphs have no ``tools`` attribute, so the old
    # ``compiled.tools`` path is kept only as a fallback for other graph shapes.
    wrapped_tools = _wrap_graph_tools(compiled, runtime)
    if wrapped_tools:
        logger.info("Tool-level enforcement on: %s", ", ".join(sorted(wrapped_tools)))
    elif hasattr(compiled, "tools"):
        try:
            compiled.tools = runtime.wrap_tools(compiled.tools)
        except Exception:
            logger.warning("Could not wrap graph tools; tool-level enforcement is OFF.")
    else:
        logger.warning("No ToolNode found in the graph; tool-level enforcement is OFF.")

    return SecureGraph(graph=compiled, runtime=runtime)


# ------------------------------------------------------------------
# Adapter registry
# ------------------------------------------------------------------


@register_adapter("langgraph")
class LangGraphAdapter(BaseAdapter):
    @classmethod
    def detect(cls, agent: Any) -> bool:
        module = type(agent).__module__ or ""
        return module.startswith("langgraph")

    @classmethod
    def wrap(cls, agent: Any, **kwargs: Any) -> Any:
        return secure_graph(agent, **kwargs)
