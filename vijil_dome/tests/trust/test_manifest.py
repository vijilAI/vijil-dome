"""Tests for ToolManifest and ToolEntry models."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from vijil_dome.trust.manifest import ToolEntry, ToolManifest


def _make_entry(name: str = "flights") -> ToolEntry:
    return ToolEntry(
        name=name,
        identity="spiffe://vijil.ai/tools/flights/v2",
        endpoint="mcp+tls://flights.internal:8443",
    )


def _make_manifest(tools: list[ToolEntry] | None = None) -> ToolManifest:
    return ToolManifest(
        agent_id="spiffe://vijil.ai/agents/travel-agent/v1",
        tools=tools or [_make_entry()],
        compiled_at=datetime(2026, 4, 3, 12, 0, 0, tzinfo=UTC),
    )


def _keypair() -> tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    return private_key, public_key


# ---------------------------------------------------------------------------
# 1. Basic construction
# ---------------------------------------------------------------------------


def test_create_entry() -> None:
    entry = ToolEntry(
        name="flights",
        identity="spiffe://vijil.ai/tools/flights/v2",
        endpoint="mcp+tls://flights.internal:8443",
        version="2.1.0",
    )
    assert entry.name == "flights"
    assert entry.identity == "spiffe://vijil.ai/tools/flights/v2"
    assert entry.endpoint == "mcp+tls://flights.internal:8443"
    assert entry.version == "2.1.0"


def test_entry_without_version() -> None:
    entry = ToolEntry(
        name="hotels",
        identity="spiffe://vijil.ai/tools/hotels/v1",
        endpoint="mcp+tls://hotels.internal:8443",
    )
    assert entry.version is None


def test_create_manifest() -> None:
    manifest = _make_manifest()
    assert manifest.manifest_version == 1
    assert manifest.agent_id == "spiffe://vijil.ai/agents/travel-agent/v1"
    assert len(manifest.tools) == 1
    assert manifest.tools[0].name == "flights"
    assert manifest.signature == ""


# ---------------------------------------------------------------------------
# 2. canonical_json
# ---------------------------------------------------------------------------


def test_canonical_json_is_deterministic() -> None:
    manifest = _make_manifest()
    first = manifest.canonical_json()
    second = manifest.canonical_json()
    assert first == second


def test_canonical_json_excludes_signature() -> None:
    manifest = _make_manifest()
    raw = manifest.canonical_json()
    data = json.loads(raw)
    assert "signature" not in data


def test_canonical_json_keys_sorted() -> None:
    manifest = _make_manifest()
    raw = manifest.canonical_json()
    # Verify the JSON string has keys in sorted order (sort_keys=True)
    data = json.loads(raw)
    keys = list(data.keys())
    assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# 3. Sign and verify
# ---------------------------------------------------------------------------


def test_sign_and_verify_roundtrip() -> None:
    private_key, public_key = _keypair()
    manifest = _make_manifest()

    signed = manifest.sign(private_key)

    # signature should now be populated
    assert signed.signature != ""
    # original should be unchanged
    assert manifest.signature == ""

    assert signed.verify_signature(public_key) is True


def test_tampered_manifest_fails_verification() -> None:
    private_key, public_key = _keypair()
    manifest = _make_manifest()
    signed = manifest.sign(private_key)

    # Tamper: change agent_id
    tampered = signed.model_copy(update={"agent_id": "spiffe://evil.ai/agent/x"})

    assert tampered.verify_signature(public_key) is False


def test_sign_returns_new_instance() -> None:
    private_key, _ = _keypair()
    manifest = _make_manifest()
    signed = manifest.sign(private_key)
    assert signed is not manifest


def test_wrong_key_fails_verification() -> None:
    private_key1, _ = _keypair()
    _, public_key2 = _keypair()

    manifest = _make_manifest()
    signed = manifest.sign(private_key1)

    assert signed.verify_signature(public_key2) is False


# ---------------------------------------------------------------------------
# 4. File I/O
# ---------------------------------------------------------------------------


def test_load_from_file(tmp_path: Path) -> None:
    private_key, _ = _keypair()
    manifest = _make_manifest()
    signed = manifest.sign(private_key)

    out = tmp_path / "manifest.json"
    out.write_text(signed.model_dump_json())

    loaded = ToolManifest.load(out)
    assert loaded.agent_id == signed.agent_id
    assert len(loaded.tools) == len(signed.tools)
    assert loaded.tools[0].name == signed.tools[0].name
    assert loaded.signature == signed.signature
    assert loaded.manifest_version == signed.manifest_version
