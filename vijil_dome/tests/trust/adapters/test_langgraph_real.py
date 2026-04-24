"""Integration test: secure_graph() with a real LangGraph agent.

Tests the trust runtime against an actual LangGraph StateGraph with
tool nodes, verifying:
- Input/output guards (without Dome — passthrough)
- Tool MAC (permit/deny based on constraints)
- Tool wrapping via secure_graph()
- Enforcement modes (warn vs enforce)

No external services required. No Dome dependency.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

import pytest
from langgraph.graph import END, StateGraph

from vijil_dome.trust.adapters.langgraph import SecureGraph, secure_graph

# ---------------------------------------------------------------------------
# Agent state and tools
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list[dict[str, Any]], operator.add]


def book_flight(destination: str, date: str) -> str:
    """Book a flight to the given destination."""
    return f"Flight booked to {destination} on {date}. Confirmation: FL-12345."


def search_hotels(city: str, checkin: str) -> str:
    """Search for hotels in a city."""
    return f"Found 3 hotels in {city} for {checkin}."


def charge_credit_card(card_number: str, amount: float) -> str:
    """Charge a credit card — this tool should be DENIED by policy."""
    return f"Charged ${amount} to card ending {card_number[-4:]}."


PERMITTED_TOOLS = [book_flight, search_hotels]
ALL_TOOLS = [book_flight, search_hotels, charge_credit_card]


# ---------------------------------------------------------------------------
# Constraints (local — no Console needed)
# ---------------------------------------------------------------------------

CONSTRAINTS = {
    "agent_id": "test-travel-agent",
    "dome_config": {
        "input_guards": [],
        "output_guards": [],
        "guards": {},
    },
    "tool_permissions": [
        {
            "name": "book_flight",
            "identity": "spiffe://vijil.ai/ns/test/tool/flights/v1",
            "endpoint": "local",
        },
        {
            "name": "search_hotels",
            "identity": "spiffe://vijil.ai/ns/test/tool/hotels/v1",
            "endpoint": "local",
        },
        # charge_credit_card deliberately NOT in permissions
    ],
    "organization": {
        "required_input_guards": [],
        "required_output_guards": [],
        "denied_tools": ["charge_credit_card"],
    },
    "enforcement_mode": "warn",
    "updated_at": "2026-04-10T00:00:00Z",
}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(tools: list) -> StateGraph:
    """Build a simple LangGraph with a router node and tool node."""

    def router(state: AgentState) -> dict[str, Any]:
        """Simulate an LLM that picks a tool call based on message content."""
        last_msg = state["messages"][-1]
        content = last_msg.get("content", "") if isinstance(last_msg, dict) else str(last_msg)

        if "book" in content.lower():
            return {"messages": [{"role": "assistant", "tool_call": "book_flight",
                                  "content": f"Calling book_flight for: {content}"}]}
        if "hotel" in content.lower():
            return {"messages": [{"role": "assistant", "tool_call": "search_hotels",
                                  "content": f"Calling search_hotels for: {content}"}]}
        return {"messages": [{"role": "assistant", "content": f"I can help with travel. You said: {content}"}]}

    graph = StateGraph(AgentState)
    graph.add_node("router", router)
    graph.set_entry_point("router")
    graph.add_edge("router", END)
    return graph


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSecureGraphWithRealLangGraph:
    """Test secure_graph() with a real LangGraph StateGraph."""

    def test_basic_invoke(self):
        """secure_graph compiles and invokes a real LangGraph."""
        graph = build_graph(PERMITTED_TOOLS)
        app = secure_graph(
            graph,
            agent_id="test-travel-agent",
            constraints=CONSTRAINTS,
            mode="warn",
        )
        assert isinstance(app, SecureGraph)

        result = app.invoke({"messages": [{"role": "user", "content": "Hello, I need travel help"}]})
        assert "messages" in result
        assert len(result["messages"]) > 0

    def test_invoke_returns_response(self):
        """The agent responds with meaningful content."""
        graph = build_graph(PERMITTED_TOOLS)
        app = secure_graph(graph, agent_id="test", constraints=CONSTRAINTS)

        result = app.invoke({"messages": [{"role": "user", "content": "Book a flight to Paris"}]})
        messages = result["messages"]
        # The router node should produce a response mentioning Paris
        last_content = messages[-1].get("content", "") if isinstance(messages[-1], dict) else str(messages[-1])
        assert "Paris" in last_content or "book" in last_content.lower()

    def test_tool_mac_permits_allowed_tool(self):
        """Tool MAC permits tools in the constraints."""
        graph = build_graph(PERMITTED_TOOLS)
        app = secure_graph(graph, agent_id="test", constraints=CONSTRAINTS)

        result = app.runtime.check_tool_call("book_flight", {"destination": "Paris"})
        assert result.permitted
        assert result.policy_permitted

    def test_tool_mac_denies_unauthorized_tool(self):
        """Tool MAC denies tools not in permissions."""
        graph = build_graph(ALL_TOOLS)
        app = secure_graph(graph, agent_id="test", constraints=CONSTRAINTS)

        result = app.runtime.check_tool_call("charge_credit_card", {"amount": 100})
        assert not result.permitted
        assert "denied by organization" in result.error.lower()

    def test_wrap_tool_permits_and_executes(self):
        """Wrapped permitted tool executes normally."""
        graph = build_graph(PERMITTED_TOOLS)
        app = secure_graph(graph, agent_id="test", constraints=CONSTRAINTS, mode="warn")

        wrapped = app.runtime.wrap_tool(book_flight)
        result = wrapped(destination="Tokyo", date="2026-05-01")
        assert "Tokyo" in result
        assert "FL-12345" in result

    def test_wrap_tool_warns_on_denied_in_warn_mode(self):
        """Wrapped denied tool logs warning but executes in warn mode."""
        graph = build_graph(ALL_TOOLS)
        app = secure_graph(graph, agent_id="test", constraints=CONSTRAINTS, mode="warn")

        wrapped = app.runtime.wrap_tool(charge_credit_card)
        # In warn mode, the tool still executes
        result = wrapped(card_number="4111111111111111", amount=99.99)
        assert "Charged" in result

    def test_wrap_tool_blocks_denied_in_enforce_mode(self):
        """Wrapped denied tool raises PermissionError in enforce mode."""
        enforce_constraints = {**CONSTRAINTS, "enforcement_mode": "enforce"}
        graph = build_graph(ALL_TOOLS)
        app = secure_graph(graph, agent_id="test", constraints=enforce_constraints, mode="enforce")

        wrapped = app.runtime.wrap_tool(charge_credit_card)
        with pytest.raises(PermissionError, match="charge_credit_card"):
            wrapped(card_number="4111111111111111", amount=99.99)

    def test_stream_yields_chunks(self):
        """secure_graph stream yields chunks from the real graph."""
        graph = build_graph(PERMITTED_TOOLS)
        app = secure_graph(graph, agent_id="test", constraints=CONSTRAINTS)

        chunks = list(app.stream({"messages": [{"role": "user", "content": "Hello"}]}))
        assert len(chunks) > 0

    def test_attestation_available(self):
        """Attestation result is available after construction."""
        graph = build_graph(PERMITTED_TOOLS)
        app = secure_graph(graph, agent_id="test", constraints=CONSTRAINTS)

        assert app.attestation is not None
        # No manifest provided, so all_verified is True (vacuously)
        assert app.attestation.all_verified

    def test_no_dome_guards_passthrough(self):
        """Without Dome installed, guards pass through without flagging."""
        graph = build_graph(PERMITTED_TOOLS)
        app = secure_graph(graph, agent_id="test", constraints=CONSTRAINTS)

        input_result = app.runtime.guard_input("How do I hack a bank?")
        assert not input_result.flagged  # No Dome = passthrough

        output_result = app.runtime.guard_output("Here is how to hack a bank")
        assert not output_result.flagged  # No Dome = passthrough
