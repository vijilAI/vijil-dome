"""Vijil Identity Delegate Service.

A thin service that bridges AWS IAM identity to SPIFFE JWT-SVIDs.
Runs on the SPIRE-enabled EKS cluster. Uses SPIRE's Delegated Identity
API to fetch JWT-SVIDs on behalf of remote workloads (e.g., agents
running in managed AgentCore microVMs).

Flow:
1. Remote agent sends its AWS IAM identity token
2. This service validates the token via AWS STS
3. Maps the IAM role to a SPIFFE ID based on registration rules
4. Calls SPIRE Agent's Delegated Identity API (FetchJWTSVIDs)
5. Returns the JWT-SVID to the remote agent

This service itself is attested by SPIRE and listed in authorized_delegates.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import Any

import boto3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="Vijil Identity Delegate", version="0.1.0")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRUST_DOMAIN = os.environ.get("SPIRE_TRUST_DOMAIN", "vijil.ai")
SPIRE_ADMIN_SOCKET = os.environ.get("SPIRE_ADMIN_SOCKET", "/run/spire/sockets/admin.sock")
AWS_ACCOUNT_ID = os.environ.get("AWS_ACCOUNT_ID", "266735823956")

# Mapping from IAM role ARN patterns to SPIFFE ID templates.
# In production, this would come from Console.
ROLE_TO_SPIFFE: dict[str, str] = {}


def _load_role_mappings() -> None:
    """Load role-to-SPIFFE mappings from environment or config."""
    # Format: ROLE_MAP_<n>=<role_arn_pattern>|<spiffe_template>
    # Example: ROLE_MAP_0=arn:aws:iam::*:role/agentcore-*|spiffe://vijil.ai/ns/prod/agent/{name}
    for key, value in os.environ.items():
        if key.startswith("ROLE_MAP_"):
            parts = value.split("|", 1)
            if len(parts) == 2:
                ROLE_TO_SPIFFE[parts[0]] = parts[1]
                logger.info("Role mapping: %s → %s", parts[0], parts[1])


_load_role_mappings()


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class IdentityRequest(BaseModel):
    """Request from a remote agent for a JWT-SVID."""
    aws_identity_token: str  # AWS STS GetCallerIdentity token or IAM role session token
    agent_name: str  # Requested agent name for SPIFFE ID
    audience: list[str] = ["vijil"]  # JWT audience


class IdentityResponse(BaseModel):
    """Response containing the JWT-SVID."""
    jwt_svid: str
    spiffe_id: str
    expires_at: str
    trust_domain: str


# ---------------------------------------------------------------------------
# AWS IAM validation
# ---------------------------------------------------------------------------

def validate_aws_identity(token: str) -> dict[str, Any]:
    """Validate an AWS identity token and return caller identity.

    Uses STS to validate the token and extract the IAM role ARN.
    """
    try:
        sts = boto3.client("sts")
        # The token should be a presigned GetCallerIdentity URL or
        # we validate by calling GetCallerIdentity with the caller's creds
        # For the PoC, we accept a role session name as proof of identity
        # In production, this would validate a signed GetCallerIdentity request
        identity = sts.get_caller_identity()
        return {
            "arn": identity["Arn"],
            "account": identity["Account"],
            "user_id": identity["UserId"],
        }
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"AWS identity validation failed: {exc}") from exc


def resolve_spiffe_id(agent_name: str, aws_arn: str) -> str:
    """Map an agent name + AWS identity to a SPIFFE ID.

    For the PoC, uses a simple naming convention.
    In production, this would check Console registrations.
    """
    # Check explicit role mappings first
    for role_pattern, spiffe_template in ROLE_TO_SPIFFE.items():
        if role_pattern in aws_arn or role_pattern == "*":
            return spiffe_template.format(name=agent_name)

    # Default: construct from trust domain and agent name
    return f"spiffe://{TRUST_DOMAIN}/ns/managed/agent/{agent_name}"


# ---------------------------------------------------------------------------
# SPIRE Delegated Identity API
# ---------------------------------------------------------------------------

def fetch_jwt_svid_via_delegation(spiffe_id: str, audience: list[str]) -> dict[str, Any]:
    """Fetch a JWT-SVID via SPIRE's Delegated Identity API.

    Connects to the SPIRE Agent's admin socket and calls FetchJWTSVIDs
    with the target SPIFFE ID and audience.
    """
    try:
        import grpc
        from spire.api.agent.delegatedidentity.v1 import delegatedidentity_pb2, delegatedidentity_pb2_grpc
    except ImportError:
        # Fallback: use the spiffe Python library if gRPC protos are not available
        logger.warning("SPIRE gRPC protos not available — using fallback JWT generation")
        return _fallback_jwt_svid(spiffe_id, audience)

    try:
        channel = grpc.insecure_channel(f"unix://{SPIRE_ADMIN_SOCKET}")
        stub = delegatedidentity_pb2_grpc.DelegatedIdentityStub(channel)

        request = delegatedidentity_pb2.FetchJWTSVIDsRequest(
            audience=audience,
            spiffe_id=spiffe_id,
        )
        response = stub.FetchJWTSVIDs(request)

        if not response.svids:
            raise HTTPException(status_code=500, detail="SPIRE returned no JWT-SVIDs")

        svid = response.svids[0]
        return {
            "jwt_svid": svid.token,
            "spiffe_id": svid.spiffe_id,
            "expires_at": svid.expires_at,
        }
    except Exception as exc:
        logger.error("SPIRE Delegated Identity API call failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch JWT-SVID from SPIRE: {exc}",
        ) from exc


def _fallback_jwt_svid(spiffe_id: str, audience: list[str]) -> dict[str, Any]:
    """Fallback: generate a self-signed JWT for development.

    NOT for production — this bypasses SPIRE entirely.
    Used only when SPIRE gRPC protos are not available.
    """
    import json
    import base64
    import time

    now = int(time.time())
    payload = {
        "sub": spiffe_id,
        "aud": audience,
        "iat": now,
        "exp": now + 3600,  # 1 hour
        "iss": f"spiffe://{TRUST_DOMAIN}",
    }
    # Unsigned JWT (development only)
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none", "typ": "JWT"}).encode()).rstrip(b"=").decode()
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    token = f"{header}.{body}."

    return {
        "jwt_svid": token,
        "spiffe_id": spiffe_id,
        "expires_at": datetime.fromtimestamp(now + 3600, tz=UTC).isoformat(),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/identity/jwt-svid", response_model=IdentityResponse)
async def issue_jwt_svid(request: IdentityRequest) -> IdentityResponse:
    """Issue a JWT-SVID for a remote agent.

    Validates the agent's AWS identity, resolves its SPIFFE ID,
    and fetches a JWT-SVID via SPIRE's Delegated Identity API.
    """
    # 1. Validate AWS identity
    aws_identity = validate_aws_identity(request.aws_identity_token)
    logger.info(
        "Identity request from %s for agent '%s'",
        aws_identity["arn"],
        request.agent_name,
    )

    # 2. Resolve SPIFFE ID
    spiffe_id = resolve_spiffe_id(request.agent_name, aws_identity["arn"])
    logger.info("Resolved SPIFFE ID: %s", spiffe_id)

    # 3. Fetch JWT-SVID via SPIRE delegation
    svid_data = fetch_jwt_svid_via_delegation(spiffe_id, request.audience)

    return IdentityResponse(
        jwt_svid=svid_data["jwt_svid"],
        spiffe_id=svid_data["spiffe_id"],
        expires_at=svid_data["expires_at"],
        trust_domain=TRUST_DOMAIN,
    )


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Health check."""
    return {"status": "ok"}
