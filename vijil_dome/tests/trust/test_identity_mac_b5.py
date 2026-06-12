"""B5 (DOME-163): enforce is a floor — a local ``mode='warn'`` must not silently downgrade a
Console- (or constraint-) mandated ``enforce``. The downgrade attempt is audited, not honored.
"""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock

from vijil_dome.trust.audit import AuditEvent
from vijil_dome.trust.runtime import TrustRuntime


def _runtime(
    *, console_mode: str, local_mode: str, sink: Callable[[AuditEvent], None] | None = None
) -> TrustRuntime:
    constraints = {
        "agent_id": "agent-1",
        "dome_config": {"input_guards": [], "output_guards": [], "guards": {}},
        "tool_permissions": [],
        "organization": {"required_input_guards": [], "required_output_guards": [], "denied_tools": []},
        "enforcement_mode": console_mode,
        "updated_at": "2026-04-03T12:00:00+00:00",
    }
    client = MagicMock()
    client._http._token = "test-api-key"
    client._http.get.return_value = constraints
    return TrustRuntime(client=client, agent_id="agent-1", mode=local_mode, audit_sink=sink)


def test_console_enforce_floors_a_local_warn() -> None:
    assert _runtime(console_mode="enforce", local_mode="warn").mode == "enforce"


def test_local_enforce_upgrades_a_console_warn() -> None:
    assert _runtime(console_mode="warn", local_mode="enforce").mode == "enforce"


def test_both_warn_stays_warn() -> None:
    assert _runtime(console_mode="warn", local_mode="warn").mode == "warn"


def test_both_enforce_stays_enforce() -> None:
    assert _runtime(console_mode="enforce", local_mode="enforce").mode == "enforce"


def test_downgrade_attempt_is_audited() -> None:
    events: list[AuditEvent] = []
    _runtime(console_mode="enforce", local_mode="warn", sink=events.append)
    downgrades = [e for e in events if e.event_type == "mode_downgrade"]
    assert len(downgrades) == 1
    assert downgrades[0].attributes["requested"] == "warn"
    assert downgrades[0].attributes["effective"] == "enforce"


def test_no_downgrade_event_when_not_downgrading() -> None:
    events: list[AuditEvent] = []
    _runtime(console_mode="warn", local_mode="enforce", sink=events.append)
    assert [e for e in events if e.event_type == "mode_downgrade"] == []


def test_dome_init_failure_under_mandated_enforce_raises(monkeypatch: object) -> None:
    """The Dome-init exception handler must gate on effective_mode, not the raw local mode.

    Console mandates enforce, the caller passes warn → effective_mode is enforce. If Dome
    construction fails, the runtime must RAISE (fail-closed) rather than silently disable guards
    — the downgrade-via-error-path the enforcement-core review flagged as critical.
    """
    import pytest
    import vijil_dome

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise ValueError("bad guard config")

    monkeypatch.setattr(vijil_dome, "Dome", _boom)  # type: ignore[attr-defined]
    constraints = {
        "agent_id": "agent-1",
        "dome_config": {"input_guards": ["prompt_injection"], "output_guards": [],
                        "guards": {"prompt_injection": {}}},
        "tool_permissions": [],
        "organization": {"required_input_guards": [], "required_output_guards": [], "denied_tools": []},
        "enforcement_mode": "enforce",
        "updated_at": "2026-04-03T12:00:00+00:00",
    }
    client = MagicMock()
    client._http._token = "test-api-key"
    client._http.get.return_value = constraints
    with pytest.raises(RuntimeError, match="enforce"):
        TrustRuntime(client=client, agent_id="agent-1", mode="warn")  # local warn, mandated enforce
