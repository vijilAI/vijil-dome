"""LangGraph adapter — wraps a compiled graph with trust enforcement.

LangGraph is NOT a dependency of this package.  All interactions use
duck typing (``Any``) so that the adapter works without importing
langgraph at all.
"""

from __future__ import annotations

import asyncio
import copy
import json
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


# Content blocks that carry no readable text for a guard: model reasoning (and its
# signatures), tool calls, and binary or encoded media. Every other block's "text" is
# scanned, so document blocks ("text-plain") and OpenAI-style "input_text" are kept.
_OPAQUE_BLOCK_TYPES = frozenset({
    "thinking", "redacted_thinking", "reasoning",
    "tool_use", "tool_call", "tool_call_chunk", "server_tool_use", "invalid_tool_call",
    "image", "image_url", "input_image", "audio", "input_audio", "file", "input_file", "video",
})


def _content_text(content: Any) -> str:
    """Text of a message's content: a string, or the text of a content-block list.

    Opaque blocks (reasoning signatures, tool calls, media) are skipped, so guards scan
    what was said, not encoded payloads. Any other block contributes its ``text``, or,
    when it keeps its text elsewhere (Anthropic ``document`` sources, ``tool_result``
    content), the whole block serialized, as ``str(content)`` did before.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") not in _OPAQUE_BLOCK_TYPES:
                text = block.get("text")
                parts.append(text if isinstance(text, str) else json.dumps(block, default=str, ensure_ascii=False))
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
#
# Enforcement is installed through ToolNode's wrap_tool_call / awrap_tool_call
# protocol, composed with any wrapper the node already has. ToolNode exposes those
# only as constructor arguments, so they are set on a private copy of each node:
# the graph passed to secure_graph (and the builder it came from) is never changed,
# and securing the same graph again starts from the original node, not a wrapped one.
# ------------------------------------------------------------------

_ORIGINAL_NODE = "_vijil_original_tool_node"


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


def _result_messages(result: Any) -> list[Any] | None:
    """The ToolMessages a tool result carries, or None if Dome cannot find them.

    A result is a ToolMessage, a ``Command`` whose ``update`` holds messages (routing
    and other state updates are left untouched), or a list of those.
    """
    if getattr(result, "type", None) == "tool":
        return [result]
    if isinstance(result, list):
        found: list[Any] = []
        for item in result:
            messages = _result_messages(item)
            if messages is None:
                return None
            found.extend(messages)
        return found
    if hasattr(result, "goto") and hasattr(result, "update"):  # langgraph.types.Command
        update = result.update
        if update is None:
            return []
        pairs = update.items() if isinstance(update, dict) else update if isinstance(update, (list, tuple)) else None
        if pairs is None:
            return None
        found = []
        for pair in pairs:
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                return None
            _key, value = pair
            values = value if isinstance(value, list) else [value]
            found.extend(item for item in values if getattr(item, "type", None) == "tool")
        return found
    return None


def _unscannable(call: dict[str, Any], result: Any) -> None:
    logger.warning(
        "Tool '%s' returned %s, which Dome cannot scan; its result was not guarded.",
        call.get("name"), type(result).__name__,
    )


def _guard_result(runtime: TrustRuntime, call: dict[str, Any], result: Any) -> Any:
    messages = _result_messages(result)
    if messages is None:
        _unscannable(call, result)
        return result
    for message in messages:
        text = _content_text(message.content)
        if text:
            verdict = runtime.guard_tool_response(str(call.get("name")), text)
            if verdict.flagged and verdict.enforced:
                message.content = verdict.guarded_response
    return result


async def _aguard_result(runtime: TrustRuntime, call: dict[str, Any], result: Any) -> Any:
    messages = _result_messages(result)
    if messages is None:
        _unscannable(call, result)
        return result
    for message in messages:
        text = _content_text(message.content)
        if text:
            verdict = await runtime.aguard_output(text)
            if verdict.flagged and verdict.enforced:
                message.content = verdict.guarded_response
    return result


def _is_tool_node(node: Any) -> bool:
    return isinstance(getattr(node, "tools_by_name", None), dict) and hasattr(node, "_wrap_tool_call")


def _guarded_tool_node(node: Any, runtime: TrustRuntime) -> Any:
    """A copy of ``node`` whose every tool call gets MAC and whose results are guarded."""
    original = getattr(node, _ORIGINAL_NODE, node)
    inner_wrap = original._wrap_tool_call
    inner_awrap = getattr(original, "_awrap_tool_call", None)

    def wrap_tool_call(request: Any, execute: Any) -> Any:
        denied = _mac_check(runtime, request.tool_call)
        if denied is not None:
            return denied
        result = inner_wrap(request, execute) if inner_wrap is not None else execute(request)
        return _guard_result(runtime, request.tool_call, result)

    async def awrap_tool_call(request: Any, execute: Any) -> Any:
        denied = _mac_check(runtime, request.tool_call)
        if denied is not None:
            return denied
        if inner_awrap is not None:
            result = await inner_awrap(request, execute)
        elif inner_wrap is not None:
            # A sync-only wrapper: run it off the event loop, executing tools back on it.
            loop = asyncio.get_running_loop()

            def sync_execute(req: Any) -> Any:
                return asyncio.run_coroutine_threadsafe(execute(req), loop).result()

            result = await asyncio.to_thread(inner_wrap, request, sync_execute)
        else:
            result = await execute(request)
        return await _aguard_result(runtime, request.tool_call, result)

    guarded = copy.copy(original)
    # RunnableCallable stores its entry points as methods bound to the original node.
    guarded.func, guarded.afunc = guarded._func, guarded._afunc
    guarded._wrap_tool_call = wrap_tool_call
    guarded._awrap_tool_call = awrap_tool_call
    setattr(guarded, _ORIGINAL_NODE, original)
    return guarded


def _is_graph(node: Any) -> bool:
    return isinstance(getattr(node, "nodes", None), dict) and callable(getattr(node, "copy", None))


def _secure_graph_tools(graph: Any, runtime: TrustRuntime) -> tuple[Any, list[str], int]:
    """Copy ``graph`` with every ToolNode, in it and in its subgraphs, guarded.

    Returns the copy (``graph`` itself if it has no ToolNode), the guarded tool names, and
    how many subgraphs with tools could not be reached (a subgraph called from inside a
    function node, rather than added as a node, cannot be swapped).
    """
    if not _is_graph(graph):
        return graph, [], 0
    nodes: dict[str, Any] = {}
    wrapped: list[str] = []
    unreachable = 0
    changed = False
    for name, node in graph.nodes.items():
        bound = getattr(node, "bound", None)
        replacement = None
        if _is_tool_node(bound):
            replacement = _guarded_tool_node(bound, runtime)
            wrapped.extend(bound.tools_by_name)
        elif _is_graph(bound):
            secured, sub_wrapped, sub_unreachable = _secure_graph_tools(bound, runtime)
            wrapped.extend(sub_wrapped)
            unreachable += sub_unreachable
            if secured is not bound:
                replacement = secured
        else:
            for subgraph in getattr(node, "subgraphs", None) or []:
                if _secure_graph_tools(subgraph, runtime)[1]:
                    unreachable += 1
        if replacement is not None and callable(getattr(node, "copy", None)):
            nodes[name] = node.copy({"bound": replacement})
            changed = True
        else:
            nodes[name] = node
    if not changed:
        return graph, wrapped, unreachable
    return graph.copy({"nodes": nodes}), wrapped, unreachable


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

    # Tool enforcement: guard every ToolNode, including those in subgraphs (MAC before
    # each call, guards on each result), on a copy of the compiled graph. Compiled
    # LangGraph graphs have no ``tools`` attribute, so the old ``compiled.tools`` path
    # is kept only as a fallback for other graph shapes.
    compiled, wrapped_tools, unreachable = _secure_graph_tools(compiled, runtime)
    if unreachable:
        logger.warning(
            "%d subgraph(s) with tools are called from inside a node, not added as nodes; "
            "Dome cannot enforce tool policy on them.", unreachable,
        )
    if wrapped_tools:
        logger.info("Tool-level enforcement on: %s", ", ".join(sorted(set(wrapped_tools))))
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
