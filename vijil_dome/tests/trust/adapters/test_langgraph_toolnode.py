"""secure_graph() on standard LangGraph graphs: StateGraph + ToolNode, async nodes, runtime context.

Regression tests for #357:
- tool MAC must apply to tools inside a ToolNode (compiled graphs have no ``.tools``);
- ``ainvoke`` must run async-only graphs and pass ``context=`` through;
- guards scan only text content blocks.

No external services required. No Dome guards (passthrough).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from vijil_dome.trust.adapters.langgraph import _content_text, secure_graph

calls: list[str] = []


@tool
def delete_records(table: str) -> str:
    """Delete every record in a table."""
    calls.append(f"delete_records:{table}")
    return f"deleted all rows in {table}"


@tool
def lookup(term: str) -> str:
    """Look up a term."""
    calls.append(f"lookup:{term}")
    return f"{term} is a word"


def _constraints(*, permitted: list[str], denied: list[str]) -> dict[str, Any]:
    return {
        "agent_id": "toolnode-test",
        "dome_config": {"input_guards": [], "output_guards": [], "guards": {}},
        "tool_permissions": [
            {"name": name, "identity": f"spiffe://test/tools/{name}", "endpoint": "local"}
            for name in permitted
        ],
        "organization": {
            "required_input_guards": [],
            "required_output_guards": [],
            "denied_tools": denied,
        },
        "enforcement_mode": "enforce",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _model_calling(tool_name: str, args: dict[str, Any]):
    """A stand-in for an LLM node: asks for one tool call, then answers."""

    def model(state: MessagesState) -> dict[str, Any]:
        if any(m.type == "tool" for m in state["messages"]):
            return {"messages": [AIMessage("done")]}
        call = {"name": tool_name, "args": args, "id": "call-1"}
        return {"messages": [AIMessage("", tool_calls=[call])]}

    return model


def _tool_graph(model, tools) -> StateGraph:
    builder = StateGraph(MessagesState)
    builder.add_node("model", model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "model")
    builder.add_conditional_edges(
        "model", lambda s: "tools" if s["messages"][-1].tool_calls else END
    )
    builder.add_edge("tools", "model")
    return builder


def _tool_messages(result: dict[str, Any]) -> list[ToolMessage]:
    return [m for m in result["messages"] if isinstance(m, ToolMessage)]


class TestToolNodeEnforcement:
    def setup_method(self) -> None:
        calls.clear()

    def test_denied_tool_in_toolnode_does_not_run(self) -> None:
        builder = _tool_graph(
            _model_calling("delete_records", {"table": "customers"}), [delete_records]
        )
        graph = secure_graph(
            builder,
            agent_id="toolnode-test",
            mode="enforce",
            constraints=_constraints(permitted=[], denied=["delete_records"]),
        )

        result = graph.invoke({"messages": [HumanMessage("clean up the customers table")]})

        assert calls == []
        (message,) = _tool_messages(result)
        assert message.status == "error"
        assert "denied by Vijil Dome policy" in message.content

    def test_tool_not_on_allowlist_does_not_run(self) -> None:
        builder = _tool_graph(
            _model_calling("delete_records", {"table": "customers"}),
            [delete_records, lookup],
        )
        graph = secure_graph(
            builder,
            agent_id="toolnode-test",
            mode="enforce",
            constraints=_constraints(permitted=["lookup"], denied=[]),
        )

        graph.invoke({"messages": [HumanMessage("clean up the customers table")]})

        assert calls == []

    def test_permitted_tool_in_toolnode_runs(self) -> None:
        builder = _tool_graph(_model_calling("lookup", {"term": "dome"}), [lookup])
        graph = secure_graph(
            builder,
            agent_id="toolnode-test",
            mode="enforce",
            constraints=_constraints(permitted=["lookup"], denied=[]),
        )

        result = graph.invoke({"messages": [HumanMessage("what is dome?")]})

        assert calls == ["lookup:dome"]
        (message,) = _tool_messages(result)
        assert message.content == "dome is a word"


@dataclass
class _Context:
    answer: str


class TestAsyncGraph:
    def setup_method(self) -> None:
        calls.clear()

    def test_ainvoke_runs_async_only_graph_with_context(self) -> None:
        async def model(state: MessagesState, runtime: Runtime[_Context]) -> dict[str, Any]:
            return {"messages": [AIMessage(runtime.context.answer)]}

        builder = StateGraph(MessagesState, context_schema=_Context)
        builder.add_node("model", model)
        builder.add_edge(START, "model")
        builder.add_edge("model", END)
        graph = secure_graph(
            builder,
            agent_id="toolnode-test",
            mode="enforce",
            constraints=_constraints(permitted=[], denied=[]),
        )

        result = asyncio.run(
            graph.ainvoke({"messages": [HumanMessage("hi")]}, context=_Context(answer="hello"))
        )

        assert result["messages"][-1].content == "hello"

    def test_ainvoke_denies_tool_in_toolnode(self) -> None:
        async def model(state: MessagesState) -> dict[str, Any]:
            if any(m.type == "tool" for m in state["messages"]):
                return {"messages": [AIMessage("done")]}
            call = {"name": "delete_records", "args": {"table": "customers"}, "id": "c1"}
            return {"messages": [AIMessage("", tool_calls=[call])]}

        graph = secure_graph(
            _tool_graph(model, [delete_records]),
            agent_id="toolnode-test",
            mode="enforce",
            constraints=_constraints(permitted=[], denied=["delete_records"]),
        )

        result = asyncio.run(graph.ainvoke({"messages": [HumanMessage("clean up")]}))

        assert calls == []
        (message,) = _tool_messages(result)
        assert message.status == "error"


def test_content_text_keeps_only_text_blocks() -> None:
    content = [
        {"type": "thinking", "thinking": "hmm", "signature": "EpYBCkYIBhgCKkB..."},
        {"type": "text", "text": "The answer is 42."},
        {"type": "tool_use", "id": "t1", "name": "lookup", "input": {}},
    ]

    assert _content_text(content) == "The answer is 42."
    assert _content_text("plain string") == "plain string"
