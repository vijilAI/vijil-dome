"""Tool attestation models — SPIFFE identity verification results."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class ToolAttestationStatus(BaseModel):
    """Attestation result for a single tool."""

    tool_name: str
    expected_identity: str  # SPIFFE ID from the signed manifest
    observed_identity: str | None = None  # SPIFFE ID returned by the tool at runtime
    verified: bool
    error: str | None = None  # Set when verification failed or tool unreachable


class AttestationResult(BaseModel):
    """Aggregated attestation result for all tools used by an agent."""

    agent_identity: str  # SPIFFE ID of the agent
    tools: list[ToolAttestationStatus]
    all_verified: bool
    timestamp: datetime
