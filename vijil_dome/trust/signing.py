"""Heartbeat beacon signing — the ``BeaconSigner`` seam (DOME-169).

A signed beacon proves it was emitted by the attested principal it names, so a
forged or replayed "I am enforcing" heartbeat fails verification downstream. The
default :class:`UnwiredBeaconSigner` returns no signature (the beacon ships
``signed=False``) for agents without an SVID; :class:`X509BeaconSigner` signs
with the agent's SPIRE-issued X.509-SVID private key. Wiring the live signer to a
real SVID and the Console-side trust-bundle chaining is deferred to DOME-179
(cluster-gated); this module ships the seam plus a real, synthetic-key-tested
sign/verify implementation that binds the signing cert to the claimed identity.
"""

from __future__ import annotations

import base64
import json
from typing import Protocol

from cryptography import x509
from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa

from vijil_dome.trust.audit import BeaconSignature, Heartbeat

# The identity-bearing fields a signature covers. The ``signature`` field itself
# is excluded — you cannot sign over the signature.
# A set (not frozenset) because Pydantic's ``model_dump(include=...)`` IncEx type
# accepts ``set[str]``. Module-private and read-only in practice.
_SIGNED_FIELDS: set[str] = {
    "configured_mode",
    "guards_constructed",
    "detector_reachable",
    "attested",
    "agent_spiffe_id",
}

_PrivateKey = ec.EllipticCurvePrivateKey | rsa.RSAPrivateKey


class _X509Svid(Protocol):
    """The slice of a ``spiffe`` X509Svid that :meth:`X509BeaconSigner.from_svid` reads."""

    private_key: _PrivateKey
    leaf: x509.Certificate


def canonical_beacon_payload(beacon: Heartbeat) -> bytes:
    """Return a deterministic byte encoding of a beacon's identity-bearing fields.

    Signer and verifier must agree on the exact bytes, so keys are sorted and
    separators are fixed. ``model_dump(include=...)`` selects exactly the signed
    fields (excluding ``signature``) and fails loud if the field set ever drifts
    from the model. Cross-language note: Python ``json`` encodes ``bool`` as
    ``true``/``false`` and ``None`` as ``null`` — a future non-Python verifier
    must match this encoding.
    """
    content = beacon.model_dump(include=_SIGNED_FIELDS)
    return json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _spiffe_id_from_cert(cert: x509.Certificate) -> str | None:
    """Extract the SPIFFE ID from a certificate's SAN URIs, or None if absent."""
    try:
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
    except x509.ExtensionNotFound:
        return None
    for uri in san.value.get_values_for_type(x509.UniformResourceIdentifier):
        if uri.startswith("spiffe://"):
            return uri
    return None


class BeaconSigner(Protocol):
    """Signs an enforcement heartbeat, or declines when no SVID is available."""

    def sign(self, beacon: Heartbeat) -> BeaconSignature | None: ...


class UnwiredBeaconSigner:
    """The honest default: no SVID, so no signature.

    Returns ``None`` rather than fabricating a signature, so an unattested or
    API-key agent's beacon ships ``signed=False`` instead of falsely signed.
    """

    def sign(self, beacon: Heartbeat) -> BeaconSignature | None:
        return None


class X509BeaconSigner:
    """Signs a beacon with an X.509-SVID private key and leaf certificate."""

    def __init__(self, private_key: _PrivateKey, leaf_cert: x509.Certificate) -> None:
        self._private_key = private_key
        self._leaf_cert = leaf_cert

    @classmethod
    def from_svid(cls, svid: _X509Svid) -> X509BeaconSigner:
        """Build a signer from a ``spiffe`` X509Svid.

        Reads ``.private_key`` and ``.leaf`` — the same attributes
        ``AgentIdentity.mtls_context`` uses. Exercising this against a live SVID
        is cluster-gated (DOME-179).
        """
        return cls(svid.private_key, svid.leaf)

    def sign(self, beacon: Heartbeat) -> BeaconSignature | None:
        if beacon.agent_spiffe_id is None:
            # An unattested beacon names no principal, so a signature over it
            # would attest nothing. Degrade to unsigned rather than emit a
            # null-subject signature that a consumer might mistake for proof.
            return None
        payload = canonical_beacon_payload(beacon)
        key = self._private_key
        if isinstance(key, ec.EllipticCurvePrivateKey):
            alg = "ES256"
            raw = key.sign(payload, ec.ECDSA(hashes.SHA256()))
        elif isinstance(key, rsa.RSAPrivateKey):
            alg = "RS256"
            raw = key.sign(payload, padding.PKCS1v15(), hashes.SHA256())
        else:
            raise TypeError(
                f"unsupported SVID key type: {type(key).__name__}; expected EC or RSA"
            )
        leaf_pem = self._leaf_cert.public_bytes(serialization.Encoding.PEM).decode("utf-8")
        return BeaconSignature(
            alg=alg,
            signature=base64.b64encode(raw).decode("ascii"),
            cert_chain=[leaf_pem],
            signed_subject=beacon.agent_spiffe_id,
        )


def verify_beacon_signature(beacon: Heartbeat, signature: BeaconSignature) -> bool:
    """Verify a detached beacon signature against the leaf cert it carries.

    Confirms two things: (1) the leaf cert's SPIFFE SAN equals the beacon's
    ``agent_spiffe_id`` — without this, any holder of a valid SVID could sign a
    beacon impersonating another agent; (2) the signature verifies under the
    leaf's public key over the canonical payload. Algorithm choice is driven by
    the cert's key type, NOT the attacker-supplied ``alg`` field, so an algorithm-
    confusion forgery cannot succeed.

    Pure (no I/O). Returns ``False`` on any mismatch, malformed cert, or bad
    base64 — never raises. Chaining the leaf to the trust bundle (proving the
    cert is SPIRE-issued, not attacker-self-signed) is the Console's job
    (DOME-179); this function alone does not establish that the SVID is genuine.
    """
    if not signature.cert_chain:
        return False
    try:
        leaf = x509.load_pem_x509_certificate(signature.cert_chain[0].encode("utf-8"))
        cert_spiffe_id = _spiffe_id_from_cert(leaf)
        if cert_spiffe_id is None or cert_spiffe_id != beacon.agent_spiffe_id:
            return False
        public_key = leaf.public_key()
        raw = base64.b64decode(signature.signature, validate=True)
        payload = canonical_beacon_payload(beacon)
        if isinstance(public_key, ec.EllipticCurvePublicKey):
            public_key.verify(raw, payload, ec.ECDSA(hashes.SHA256()))
        elif isinstance(public_key, rsa.RSAPublicKey):
            public_key.verify(raw, payload, padding.PKCS1v15(), hashes.SHA256())
        else:
            return False
    except (InvalidSignature, ValueError, UnsupportedAlgorithm):
        # ValueError covers malformed PEM and bad base64 (binascii.Error);
        # UnsupportedAlgorithm covers a leaf carrying a non-EC/RSA key type.
        return False
    return True
