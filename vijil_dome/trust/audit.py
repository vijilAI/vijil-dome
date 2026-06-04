"""Audit event emitter — pluggable sink for trust runtime observability."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from vijil_dome.trust.delta import TrustDelta, TrustVector

logger = logging.getLogger(__name__)


class AuditEvent(BaseModel):
    """A single auditable trust event emitted by the runtime."""

    event_type: str
    agent_id: str
    timestamp: datetime
    attributes: dict[str, Any]


class Heartbeat(BaseModel):
    """An enforcement-alive beacon: 'this agent is still enforcing'.

    Emitted by a wrapped agent so a registered agent that goes dark or
    silently downgrades its enforcement is detectable downstream. The
    fields describe the live enforcement posture at emit time:

    - ``effective_mode`` — the runtime's actual mode (``"warn"`` or
      ``"enforce"``), so a silent downgrade from enforce to warn is visible.
    - ``hooks_attached`` — whether the content guards are wired and live.
    - ``detector_reachable`` — whether the detector backend is reachable; a
      starved or failed detector flips this False without raising.
    - ``agent_spiffe_id`` — the attested principal, or ``None`` when the
      agent is unattested.

    SVID-signing of the beacon and periodic scheduling are a follow-up; this
    model is the unsigned event shape.
    """

    effective_mode: str
    hooks_attached: bool
    detector_reachable: bool
    agent_spiffe_id: str | None


class AuditEmitter:
    """Emits structured audit events to a pluggable sink.

    The default sink logs via ``logging.info``.  In tests, pass a list's
    ``append`` method to capture events.  In production, wire an OTel span
    exporter as the sink.
    """

    def __init__(
        self,
        agent_id: str,
        sink: Callable[[AuditEvent], None] | None = None,
    ) -> None:
        self._agent_id = agent_id
        self._sink = sink if sink is not None else self._log_sink

    # ------------------------------------------------------------------
    # Public emit methods
    # ------------------------------------------------------------------

    def emit_guard(
        self,
        direction: str,
        *,
        flagged: bool,
        score: float,
        exec_time_ms: float,
    ) -> None:
        """Emit a guard execution event (input or output direction)."""
        self._emit(
            "guard",
            direction=direction,
            flagged=flagged,
            score=score,
            exec_time_ms=exec_time_ms,
        )

    def emit_tool_mac(
        self,
        tool_name: str,
        *,
        permitted: bool,
        identity_verified: bool,
        agent_spiffe_id: str | None = None,
    ) -> None:
        """Emit a tool MAC enforcement event.

        ``agent_spiffe_id`` records the attested principal the MAC decision was
        made for, so the binding between the decision and the caller is auditable
        downstream. It is meaningful only when ``identity_verified`` is True.
        """
        self._emit(
            "tool_mac",
            tool_name=tool_name,
            permitted=permitted,
            identity_verified=identity_verified,
            agent_spiffe_id=agent_spiffe_id,
        )

    def emit_heartbeat(
        self,
        *,
        effective_mode: str,
        hooks_attached: bool,
        detector_reachable: bool,
        agent_spiffe_id: str | None = None,
    ) -> None:
        """Emit an enforcement-alive heartbeat ('I am enforcing') event.

        Carries the live enforcement posture so a registered agent that goes
        dark or silently downgrades is detectable downstream. See
        ``Heartbeat`` for field semantics. SVID-signing of the beacon is a
        follow-up; this emits the unsigned event shape.
        """
        self._emit(
            "enforcement_heartbeat",
            effective_mode=effective_mode,
            hooks_attached=hooks_attached,
            detector_reachable=detector_reachable,
            agent_spiffe_id=agent_spiffe_id,
        )

    def emit_attestation(
        self,
        *,
        all_verified: bool,
        tool_count: int,
    ) -> None:
        """Emit a tool manifest attestation event."""
        self._emit(
            "attestation",
            all_verified=all_verified,
            tool_count=tool_count,
        )

    def emit_mtls_downgrade(
        self,
        tool_name: str,
        *,
        error: str,
    ) -> None:
        """Emit an mTLS downgrade event."""
        self._emit(
            "mtls_downgrade",
            tool_name=tool_name,
            error=error,
        )

    def emit_guards_disabled(
        self,
        *,
        error: str,
    ) -> None:
        """Emit a guards-disabled event (Dome init failed in warn mode)."""
        self._emit(
            "guards_disabled",
            error=error,
        )

    def emit_identity_unattested(self) -> None:
        """Emit an event when the agent is running without SPIFFE attestation."""
        self._emit("identity_unattested")

    def emit_trust_delta(
        self,
        *,
        control_name: str,
        delta: TrustDelta,
        before: TrustVector,
        after: TrustVector,
    ) -> None:
        """Emit a trust-delta application event.

        Emitted exactly once per applied delta. Pre-seed deltas are
        queued in ``TrustRuntime._pending_deltas`` and audited at
        ``seed_trust_vector`` replay time, so ``before`` and ``after``
        always carry concrete measured ``TrustVector`` values.
        """
        self._emit(
            "trust_delta",
            control_name=control_name,
            delta=delta.model_dump(),
            before=before.model_dump(),
            after=after.model_dump(),
        )


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit(self, event_type: str, **attrs: Any) -> None:
        event = AuditEvent(
            event_type=event_type,
            agent_id=self._agent_id,
            timestamp=datetime.now(tz=UTC),
            attributes=attrs,
        )
        self._sink(event)

    @staticmethod
    def _log_sink(event: AuditEvent) -> None:
        logger.info(
            "audit event",
            extra={
                "event_type": event.event_type,
                "agent_id": event.agent_id,
                "timestamp": event.timestamp.isoformat(),
                **event.attributes,
            },
        )
