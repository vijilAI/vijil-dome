"""Tests for the BeaconSigner seam (DOME-169).

A signed heartbeat proves the beacon was emitted by the attested principal it
names, so a forged or replayed "I am enforcing" beacon is detectable. These
tests exercise the real ECDSA/RSA sign-and-verify path with a synthetic key —
the live-SVID wiring is deferred to DOME-179.
"""

from __future__ import annotations

import datetime
from types import SimpleNamespace

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.x509.oid import NameOID

from vijil_dome.trust.audit import AuditEmitter, BeaconSignature, Heartbeat
from vijil_dome.trust.signing import (
    UnwiredBeaconSigner,
    X509BeaconSigner,
    canonical_beacon_payload,
    verify_beacon_signature,
)

_SPIFFE = "spiffe://vijil.ai/org/team/agent/uuid"


def _self_signed(
    spiffe_id: str, key: ec.EllipticCurvePrivateKey | rsa.RSAPrivateKey | None = None
) -> tuple[ec.EllipticCurvePrivateKey | rsa.RSAPrivateKey, x509.Certificate]:
    """Build a synthetic SVID-shaped self-signed cert carrying a SPIFFE SAN."""
    signing_key = key or ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test-agent")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(signing_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC))
        .not_valid_after(datetime.datetime(2030, 1, 1, tzinfo=datetime.UTC))
        .add_extension(
            x509.SubjectAlternativeName([x509.UniformResourceIdentifier(spiffe_id)]),
            critical=False,
        )
        .sign(signing_key, hashes.SHA256())
    )
    return signing_key, cert


def _beacon(**overrides: object) -> Heartbeat:
    fields: dict[str, object] = {
        "configured_mode": "enforce",
        "guards_constructed": True,
        "detector_reachable": True,
        "attested": True,
        "agent_spiffe_id": _SPIFFE,
    }
    fields.update(overrides)
    return Heartbeat(**fields)  # type: ignore[arg-type]


def test_x509_signer_round_trip_ec() -> None:
    key, cert = _self_signed(_SPIFFE)
    sig = X509BeaconSigner(key, cert).sign(_beacon())
    assert sig is not None
    assert sig.alg == "ES256"
    assert sig.signed_subject == _SPIFFE
    assert verify_beacon_signature(_beacon(), sig) is True


def test_rsa_signer_round_trip() -> None:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    _, cert = _self_signed(_SPIFFE, key=key)
    sig = X509BeaconSigner(key, cert).sign(_beacon())
    assert sig is not None
    assert sig.alg == "RS256"
    assert verify_beacon_signature(_beacon(), sig) is True


def test_tampered_beacon_fails_verification() -> None:
    # ADVERSARIAL: a beacon signed in enforce posture must not verify when a
    # downgraded warn posture is presented under the same signature.
    key, cert = _self_signed(_SPIFFE)
    sig = X509BeaconSigner(key, cert).sign(_beacon(configured_mode="enforce"))
    assert sig is not None
    assert verify_beacon_signature(_beacon(configured_mode="warn"), sig) is False


def test_signature_from_wrong_key_fails_verification() -> None:
    # ADVERSARIAL: signed by key A but carrying cert B (mismatched public key).
    key_a, _ = _self_signed(_SPIFFE)
    _, cert_b = _self_signed(_SPIFFE)
    sig = X509BeaconSigner(key_a, cert_b).sign(_beacon())
    assert sig is not None
    assert verify_beacon_signature(_beacon(), sig) is False


def test_impersonation_san_mismatch_fails_verification() -> None:
    # ADVERSARIAL: an attacker holding a VALID SVID (key + cert) signs a beacon
    # naming the victim's spiffe id. The signature math is sound, but the cert's
    # SAN names the attacker, so verification must reject the impersonation.
    attacker = "spiffe://vijil.ai/org/team/agent/attacker"
    victim = "spiffe://vijil.ai/org/team/agent/victim"
    key, cert = _self_signed(attacker)
    sig = X509BeaconSigner(key, cert).sign(_beacon(agent_spiffe_id=victim))
    assert sig is not None
    assert verify_beacon_signature(_beacon(agent_spiffe_id=victim), sig) is False


def test_x509_signer_declines_unattested_beacon() -> None:
    # An unattested beacon names no principal; signing it would attest nothing.
    key, cert = _self_signed(_SPIFFE)
    sig = X509BeaconSigner(key, cert).sign(_beacon(attested=False, agent_spiffe_id=None))
    assert sig is None


def test_from_svid_constructs_working_signer() -> None:
    key, cert = _self_signed(_SPIFFE)
    fake_svid = SimpleNamespace(private_key=key, leaf=cert)
    signer = X509BeaconSigner.from_svid(fake_svid)
    sig = signer.sign(_beacon())
    assert sig is not None
    assert verify_beacon_signature(_beacon(), sig) is True


def test_verify_rejects_empty_cert_chain() -> None:
    bad = BeaconSignature(alg="ES256", signature="AA==", cert_chain=[], signed_subject=_SPIFFE)
    assert verify_beacon_signature(_beacon(), bad) is False


def test_unwired_signer_returns_none() -> None:
    assert UnwiredBeaconSigner().sign(_beacon()) is None


def test_unsupported_key_type_raises() -> None:
    # BOUNDARY: a non-EC/RSA key must fail loud, never emit a bogus signature.
    _, cert = _self_signed(_SPIFFE)

    class _FakeKey:
        pass

    with pytest.raises(TypeError):
        X509BeaconSigner(_FakeKey(), cert).sign(_beacon())  # type: ignore[arg-type]


def test_canonical_payload_excludes_signature_field() -> None:
    plain = _beacon()
    with_sig = _beacon(
        signature=BeaconSignature(
            alg="ES256", signature="zz", cert_chain=["pem"], signed_subject=_SPIFFE
        )
    )
    assert canonical_beacon_payload(plain) == canonical_beacon_payload(with_sig)


def test_canonical_payload_changes_with_content() -> None:
    assert canonical_beacon_payload(_beacon(detector_reachable=True)) != (
        canonical_beacon_payload(_beacon(detector_reachable=False))
    )


def test_heartbeat_signature_field_defaults_none() -> None:
    assert _beacon().signature is None


def test_emit_heartbeat_carries_signature_in_attributes() -> None:
    events: list[object] = []
    emitter = AuditEmitter(agent_id="a", sink=events.append)
    sig = BeaconSignature(
        alg="ES256", signature="zz", cert_chain=["pem"], signed_subject=_SPIFFE
    )
    emitter.emit_heartbeat(
        configured_mode="enforce",
        guards_constructed=True,
        detector_reachable=True,
        attested=True,
        agent_spiffe_id=_SPIFFE,
        signature=sig,
    )
    assert events[0].attributes["signature"]["alg"] == "ES256"  # type: ignore[attr-defined]


def test_emit_heartbeat_without_signature_records_none() -> None:
    events: list[object] = []
    emitter = AuditEmitter(agent_id="a", sink=events.append)
    emitter.emit_heartbeat(
        configured_mode="warn",
        guards_constructed=False,
        detector_reachable=False,
        attested=False,
        agent_spiffe_id=None,
    )
    assert events[0].attributes["signature"] is None  # type: ignore[attr-defined]
