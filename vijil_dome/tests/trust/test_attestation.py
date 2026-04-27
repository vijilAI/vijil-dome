"""Tests for AttestationResult and ToolAttestationStatus models."""

from __future__ import annotations

from datetime import UTC, datetime

from vijil_dome.trust.attestation import AttestationResult, ToolAttestationStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_timestamp() -> datetime:
    return datetime(2026, 4, 3, 12, 0, 0, tzinfo=UTC)


def _verified_tool(name: str = "flights") -> ToolAttestationStatus:
    return ToolAttestationStatus(
        tool_name=name,
        expected_identity=f"spiffe://vijil.ai/tools/{name}/v1",
        observed_identity=f"spiffe://vijil.ai/tools/{name}/v1",
        verified=True,
    )


def _mismatched_tool(name: str = "flights") -> ToolAttestationStatus:
    return ToolAttestationStatus(
        tool_name=name,
        expected_identity=f"spiffe://vijil.ai/tools/{name}/v1",
        observed_identity="spiffe://evil.ai/tools/imposter/v1",
        verified=False,
        error="SPIFFE ID mismatch",
    )


def _unreachable_tool(name: str = "flights") -> ToolAttestationStatus:
    return ToolAttestationStatus(
        tool_name=name,
        expected_identity=f"spiffe://vijil.ai/tools/{name}/v1",
        observed_identity=None,
        verified=False,
        error="connection refused",
    )


# ---------------------------------------------------------------------------
# 1. test_all_verified — all tools verified
# ---------------------------------------------------------------------------


def test_all_verified() -> None:
    result = AttestationResult(
        agent_identity="spiffe://vijil.ai/agents/travel-agent/v1",
        tools=[
            _verified_tool("flights"),
            _verified_tool("hotels"),
            _verified_tool("cars"),
        ],
        all_verified=True,
        timestamp=_make_timestamp(),
    )

    assert result.all_verified is True
    assert len(result.tools) == 3
    assert all(t.verified for t in result.tools)
    assert all(t.error is None for t in result.tools)
    assert all(t.observed_identity == t.expected_identity for t in result.tools)


def test_all_verified_single_tool() -> None:
    result = AttestationResult(
        agent_identity="spiffe://vijil.ai/agents/travel-agent/v1",
        tools=[_verified_tool("flights")],
        all_verified=True,
        timestamp=_make_timestamp(),
    )

    assert result.all_verified is True
    assert result.tools[0].tool_name == "flights"
    assert result.tools[0].verified is True


# ---------------------------------------------------------------------------
# 2. test_mismatch_not_verified — one tool has wrong SPIFFE ID
# ---------------------------------------------------------------------------


def test_mismatch_not_verified() -> None:
    result = AttestationResult(
        agent_identity="spiffe://vijil.ai/agents/travel-agent/v1",
        tools=[
            _verified_tool("hotels"),
            _mismatched_tool("flights"),
        ],
        all_verified=False,
        timestamp=_make_timestamp(),
    )

    assert result.all_verified is False

    flights = next(t for t in result.tools if t.tool_name == "flights")
    assert flights.verified is False
    assert flights.observed_identity == "spiffe://evil.ai/tools/imposter/v1"
    assert flights.expected_identity == "spiffe://vijil.ai/tools/flights/v1"
    assert flights.error == "SPIFFE ID mismatch"

    hotels = next(t for t in result.tools if t.tool_name == "hotels")
    assert hotels.verified is True


def test_mismatch_error_field_present() -> None:
    tool = _mismatched_tool("payments")
    assert tool.error is not None
    assert len(tool.error) > 0


# ---------------------------------------------------------------------------
# 3. test_unreachable_tool — observed_identity is None
# ---------------------------------------------------------------------------


def test_unreachable_tool() -> None:
    result = AttestationResult(
        agent_identity="spiffe://vijil.ai/agents/travel-agent/v1",
        tools=[_unreachable_tool("flights")],
        all_verified=False,
        timestamp=_make_timestamp(),
    )

    assert result.all_verified is False

    tool = result.tools[0]
    assert tool.observed_identity is None
    assert tool.verified is False
    assert tool.error == "connection refused"


def test_unreachable_tool_has_expected_identity() -> None:
    """Even when unreachable, the expected identity is known from the manifest."""
    tool = _unreachable_tool("hotels")
    assert tool.expected_identity == "spiffe://vijil.ai/tools/hotels/v1"
    assert tool.observed_identity is None


def test_tool_attestation_defaults() -> None:
    """error defaults to None for a clean verified tool."""
    tool = ToolAttestationStatus(
        tool_name="flights",
        expected_identity="spiffe://vijil.ai/tools/flights/v1",
        observed_identity="spiffe://vijil.ai/tools/flights/v1",
        verified=True,
    )
    assert tool.error is None


def test_attestation_timestamp_preserved() -> None:
    ts = datetime(2026, 1, 15, 8, 30, 0, tzinfo=UTC)
    result = AttestationResult(
        agent_identity="spiffe://vijil.ai/agents/travel-agent/v1",
        tools=[_verified_tool()],
        all_verified=True,
        timestamp=ts,
    )
    assert result.timestamp == ts
