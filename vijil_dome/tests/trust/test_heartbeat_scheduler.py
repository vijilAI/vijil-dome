"""Supervised heartbeat scheduler + in-process liveness self-check (DOME-169 Task 2.2).

A daemon thread emits the beacon on a cadence; ``heartbeat_health()`` lets a host
(or B4) detect a dead or stalled loop *in-process*, without waiting on the
Console reconciler. ``emit_heartbeat`` signs each beacon via the injected
``BeaconSigner`` (default unsigned). Tests synchronize on the emitted beacon via
a ``threading.Event`` and drive the liveness math with an injected clock — no
fixed sleeps.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from threading import Thread
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest

from vijil_dome.trust.audit import AuditEvent, BeaconSignature, Heartbeat
from vijil_dome.trust.constraints import AgentConstraints
from vijil_dome.trust.runtime import TrustRuntime


def _wait_until(predicate: Callable[[], bool], *, timeout: float) -> bool:
    """Poll until predicate is true or the timeout elapses (bounded, non-flaky)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


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


def _runtime(*, heartbeat_interval: float | None = None) -> TrustRuntime:
    client = MagicMock()
    client._http._token = "test-api-key"
    client._http.get.return_value = _constraints().model_dump(mode="json")
    return TrustRuntime(
        client=client,
        agent_id="agent-1",
        mode="warn",
        heartbeat_interval=heartbeat_interval,
    )


class _FakeSigner:
    """A BeaconSigner that always returns a fixed signature."""

    def sign(self, beacon: Heartbeat) -> BeaconSignature:
        return BeaconSignature(
            alg="ES256",
            signature="zz",
            cert_chain=["pem"],
            signed_subject="spiffe://vijil.ai/org/t/agent/u",
        )


def _raising_sink(event: AuditEvent) -> None:
    raise RuntimeError("audit sink down")


# --- signer wiring ---------------------------------------------------------


def test_emit_heartbeat_unsigned_by_default() -> None:
    events: list[AuditEvent] = []
    runtime = _runtime()
    runtime._audit._sink = events.append
    hb = runtime.emit_heartbeat()
    assert hb.signature is None
    assert events[-1].attributes["signature"] is None


def test_emit_heartbeat_attaches_signature_from_signer() -> None:
    events: list[AuditEvent] = []
    runtime = _runtime()
    runtime._beacon_signer = _FakeSigner()
    runtime._audit._sink = events.append
    hb = runtime.emit_heartbeat()
    assert hb.signature is not None
    assert hb.signature.alg == "ES256"
    assert events[-1].attributes["signature"]["alg"] == "ES256"


# --- per-tick resilience ---------------------------------------------------


def test_emit_safely_advances_last_emit_at() -> None:
    runtime = _runtime()
    runtime._clock = lambda: 42.0
    runtime._emit_heartbeat_safely()
    assert runtime._last_emit_at == 42.0


def test_emit_safely_survives_sink_failure() -> None:
    # A failing emit must not kill the loop, and must NOT advance last_emit_at
    # (so the staleness signal grows -> heartbeat_health turns not-alive).
    runtime = _runtime()
    runtime._audit._sink = _raising_sink
    runtime._emit_heartbeat_safely()  # must not raise
    assert runtime._last_emit_at is None


# --- thread lifecycle ------------------------------------------------------


def test_scheduler_runs_and_stops() -> None:
    runtime = _runtime()
    beat = threading.Event()
    runtime._audit._sink = lambda event: beat.set()
    runtime.start_heartbeat(interval=0.01)
    try:
        assert beat.wait(timeout=2.0)  # a beacon was emitted
        assert runtime.heartbeat_health().running is True
        # the real loop advances _last_emit_at, so alive becomes True end-to-end
        assert _wait_until(lambda: runtime.heartbeat_health().alive, timeout=2.0)
    finally:
        runtime.stop_heartbeat()
    assert runtime.heartbeat_health().running is False


def test_start_heartbeat_rejects_nonpositive_interval() -> None:
    runtime = _runtime()
    with pytest.raises(ValueError, match="positive"):
        runtime.start_heartbeat(interval=0)
    with pytest.raises(ValueError, match="positive"):
        runtime.start_heartbeat(interval=-1.0)


def test_stop_keeps_handle_when_thread_does_not_exit() -> None:
    # When the loop is stuck in a hung probe past the join timeout, stop must NOT
    # orphan the thread (which would let a later start spawn a duplicate); it keeps
    # the handle, logs, and the thread exits once the probe returns.
    runtime = _runtime()
    runtime._heartbeat_join_timeout_s = 0.05
    entered = threading.Event()
    release = threading.Event()

    class _HangingDome:
        input_guardrail = object()
        output_guardrail = None

        def guard_input(self, text: str) -> SimpleNamespace:
            entered.set()
            release.wait(timeout=5.0)  # hang until the test releases it
            return SimpleNamespace(errored_methods=[])

    runtime._dome = _HangingDome()
    runtime.start_heartbeat(interval=0.01)
    try:
        assert entered.wait(timeout=2.0)  # thread is now stuck in the probe
        runtime.stop_heartbeat()  # join times out while the probe hangs
        assert runtime._heartbeat_thread is not None  # handle kept, not orphaned
        assert runtime._heartbeat_thread.is_alive()
    finally:
        release.set()  # let the probe return so the loop can exit
        assert _wait_until(
            lambda: not runtime.heartbeat_health().running, timeout=2.0
        )
        runtime.stop_heartbeat()  # joins cleanly now, clears the handle
    assert runtime._heartbeat_thread is None


def test_stop_heartbeat_is_idempotent() -> None:
    runtime = _runtime()
    runtime.start_heartbeat(interval=0.01)
    runtime.stop_heartbeat()
    runtime.stop_heartbeat()  # second call must not raise
    assert runtime.heartbeat_health().running is False


def test_start_heartbeat_is_idempotent_while_running() -> None:
    runtime = _runtime()
    runtime.start_heartbeat(interval=0.01)
    try:
        first = runtime._heartbeat_thread
        runtime.start_heartbeat(interval=0.01)  # no-op while alive
        assert runtime._heartbeat_thread is first
    finally:
        runtime.stop_heartbeat()


# --- in-process liveness self-check ---------------------------------------


def test_health_not_alive_before_first_emit() -> None:
    runtime = _runtime()
    runtime._heartbeat_interval = 10.0
    health = runtime.heartbeat_health()
    assert health.last_emit_age_s is None
    assert health.alive is False
    assert health.running is False


def test_health_alive_when_running_and_recent_then_stale() -> None:
    runtime = _runtime()
    # simulate a running loop thread (only is_alive() is read by heartbeat_health)
    runtime._heartbeat_thread = cast(Thread, SimpleNamespace(is_alive=lambda: True))
    now = [0.0]
    runtime._clock = lambda: now[0]
    runtime._heartbeat_interval = 10.0
    runtime._last_emit_at = 0.0

    now[0] = 15.0  # running + within 2x interval (20s)
    assert runtime.heartbeat_health().alive is True
    now[0] = 25.0  # past the staleness threshold -> a stuck loop is loud
    assert runtime.heartbeat_health().alive is False


def test_health_not_alive_after_stop_even_if_recent() -> None:
    # A stopped (non-running) scheduler must not report alive, even when the last
    # emit is recent — otherwise alive lingers True for up to 2x interval.
    runtime = _runtime()
    runtime._clock = lambda: 0.0
    runtime._heartbeat_interval = 10.0
    runtime._last_emit_at = 0.0  # very recent
    assert runtime.heartbeat_health().running is False
    assert runtime.heartbeat_health().alive is False


def test_constructor_auto_starts_when_interval_set() -> None:
    runtime = _runtime(heartbeat_interval=0.01)
    try:
        assert runtime.heartbeat_health().running is True
    finally:
        runtime.stop_heartbeat()
