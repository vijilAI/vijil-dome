"""Audit event emitter — pluggable sink for trust runtime observability."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from vijil_dome.trust.models import TrustModel

logger = logging.getLogger(__name__)


class AuditEvent(TrustModel):
    """A single auditable trust event emitted by the runtime."""

    event_type: str
    agent_id: str
    timestamp: datetime
    attributes: dict[str, Any]


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
    ) -> None:
        """Emit a tool MAC enforcement event."""
        self._emit(
            "tool_mac",
            tool_name=tool_name,
            permitted=permitted,
            identity_verified=identity_verified,
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
