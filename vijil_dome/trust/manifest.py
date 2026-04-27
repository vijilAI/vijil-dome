"""Tool manifest with Ed25519 signing for the Vijil Trust Runtime."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from pydantic import BaseModel


class ToolEntry(BaseModel):
    """A single tool authorized for use by an agent."""

    name: str
    identity: str  # SPIFFE ID, e.g. spiffe://vijil.ai/tools/flights/v2
    endpoint: str  # Tool endpoint, e.g. mcp+tls://flights.internal:8443
    version: str | None = None  # Informational only; used in audit logs


class ToolManifest(BaseModel):
    """Signed manifest declaring which tools an agent is authorized to call."""

    manifest_version: int = 1
    agent_id: str  # SPIFFE ID of the agent
    tools: list[ToolEntry]
    compiled_at: datetime
    signature: str = ""  # Ed25519 signature hex string; empty until signed

    def canonical_json(self) -> str:
        """Return deterministic JSON excluding the signature field.

        Uses sort_keys=True so field ordering in construction does not affect
        the canonical form.  This is the payload that is signed and verified.
        """
        data = self.model_dump(mode="json", exclude={"signature"})
        return json.dumps(data, sort_keys=True)

    def sign(self, private_key: Ed25519PrivateKey) -> ToolManifest:
        """Return a new ToolManifest with the signature field set.

        The original instance is not mutated.
        """
        payload = self.canonical_json().encode()
        raw_signature = private_key.sign(payload)
        return self.model_copy(update={"signature": raw_signature.hex()})

    def verify_signature(self, public_key: Ed25519PublicKey) -> bool:
        """Return True if the signature matches the canonical JSON payload."""
        try:
            raw_signature = bytes.fromhex(self.signature)
            payload = self.canonical_json().encode()
            public_key.verify(raw_signature, payload)
            return True
        except (InvalidSignature, ValueError):
            return False

    @classmethod
    def load(cls, path: Path) -> ToolManifest:
        """Load a ToolManifest from a JSON file."""
        return cls.model_validate_json(path.read_text())
