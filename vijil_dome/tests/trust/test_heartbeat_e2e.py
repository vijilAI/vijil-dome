"""End-to-end heartbeat: real runtime, real loop, signing, probe, liveness (DOME-169 Task 3.2).

Drives a real ``TrustRuntime`` — not mocks of the unit under test — through the
whole beacon path: the scheduler thread emits on a cadence, the reachability
probe reflects the (stubbed) backend, a synthetic-key ``X509BeaconSigner`` signs
each beacon so it verifies against the carried cert, and ``heartbeat_health``
reports liveness in-process. The unsigned default is exercised too.
"""

from __future__ import annotations

import datetime
import threading
import time
from collections.abc import Callable
from unittest.mock import MagicMock

from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from vijil_dome.trust.audit import AuditEvent
from vijil_dome.trust.constraints import AgentConstraints
from vijil_dome.trust.runtime import TrustRuntime
from vijil_dome.trust.signing import X509BeaconSigner, verify_beacon_signature

_SPIFFE = "spiffe://vijil.ai/org/team/agent/uuid"


def _wait_until(predicate: Callable[[], bool], *, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _self_signed(spiffe_id: str) -> tuple[ec.EllipticCurvePrivateKey, x509.Certificate]:
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "e2e-agent")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC))
        .not_valid_after(datetime.datetime(2030, 1, 1, tzinfo=datetime.UTC))
        .add_extension(
            x509.SubjectAlternativeName([x509.UniformResourceIdentifier(spiffe_id)]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    return key, cert


def _constraints() -> AgentConstraints:
    return AgentConstraints.model_validate(
        {
            "agent_id": "agent-1",
            "dome_config": {"input_guards": [], "output_guards": [], "guards": {}},
            "tool_permissions": [],
            "organization": {
                "required_input_guards": [],
                "required_output_guards": [],
                "denied_tools": [],
            },
            "enforcement_mode": "enforce",
            "updated_at": "2026-04-03T12:00:00+00:00",
        }
    )


def _runtime() -> TrustRuntime:
    client = MagicMock()
    client._http._token = "test-api-key"
    client._http.get.return_value = _constraints().model_dump(mode="json")
    return TrustRuntime(client=client, agent_id="agent-1", mode="warn")


def _attested_identity(spiffe_id: str) -> MagicMock:
    identity = MagicMock()
    identity.is_attested.return_value = True
    identity.spiffe_id = spiffe_id
    return identity


class _StubScanResult:
    def __init__(self, errored_methods: list[str] | None = None) -> None:
        self.errored_methods = errored_methods or []


class _StubDome:
    def __init__(self, *, errored: bool = False) -> None:
        self.input_guardrail = object()
        self.output_guardrail = None
        self._errored = errored

    def guard_input(self, text: str) -> _StubScanResult:
        return _StubScanResult(["pi-detector"] if self._errored else [])


def test_e2e_signed_beacon_verifies_against_carried_cert() -> None:
    key, cert = _self_signed(_SPIFFE)
    runtime = _runtime()
    runtime._identity = _attested_identity(_SPIFFE)
    runtime._beacon_signer = X509BeaconSigner(key, cert)
    runtime._dome = _StubDome()

    heartbeat = runtime.emit_heartbeat()

    assert heartbeat.detector_reachable is True
    assert heartbeat.attested is True
    assert heartbeat.signature is not None
    assert heartbeat.signature.signed_subject == _SPIFFE
    assert verify_beacon_signature(heartbeat, heartbeat.signature) is True


def test_e2e_tampered_beacon_fails_verification() -> None:
    # The primitive's whole point: a downgraded "I am enforcing" beacon emitted
    # by the real runtime must NOT verify under its own signature.
    key, cert = _self_signed(_SPIFFE)
    runtime = _runtime()
    runtime._identity = _attested_identity(_SPIFFE)
    runtime._beacon_signer = X509BeaconSigner(key, cert)
    runtime._dome = _StubDome()

    heartbeat = runtime.emit_heartbeat()
    assert heartbeat.signature is not None
    forged = heartbeat.model_copy(update={"configured_mode": "warn"})
    assert verify_beacon_signature(forged, heartbeat.signature) is False


def test_e2e_unsigned_by_default_for_unattested_agent() -> None:
    runtime = _runtime()  # unattested api-key identity, default UnwiredBeaconSigner
    runtime._dome = _StubDome()
    heartbeat = runtime.emit_heartbeat()
    assert heartbeat.signature is None


def test_e2e_detector_unreachable_when_backend_errors() -> None:
    runtime = _runtime()
    runtime._dome = _StubDome(errored=True)
    heartbeat = runtime.emit_heartbeat()
    assert heartbeat.guards_constructed is True
    assert heartbeat.detector_reachable is False


def test_e2e_scheduler_emits_signed_beacons_and_reports_alive() -> None:
    key, cert = _self_signed(_SPIFFE)
    runtime = _runtime()
    runtime._identity = _attested_identity(_SPIFFE)
    runtime._beacon_signer = X509BeaconSigner(key, cert)
    runtime._dome = _StubDome()
    captured: list[AuditEvent] = []
    beat = threading.Event()

    def sink(event: AuditEvent) -> None:
        captured.append(event)
        beat.set()

    runtime._audit._sink = sink
    runtime.start_heartbeat(interval=0.01)
    try:
        assert beat.wait(timeout=2.0)
        assert _wait_until(lambda: runtime.heartbeat_health().alive, timeout=2.0)
        beacons = [e for e in captured if e.event_type == "enforcement_heartbeat"]
        signature = beacons[-1].attributes["signature"]
        assert signature is not None
        assert signature["signed_subject"] == _SPIFFE
    finally:
        runtime.stop_heartbeat()
    assert runtime.heartbeat_health().running is False
