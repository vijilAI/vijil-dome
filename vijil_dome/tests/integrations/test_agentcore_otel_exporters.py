# Copyright 2025 Vijil, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import json
from unittest.mock import MagicMock, patch

import pytest

from vijil_dome import Dome
from vijil_dome.integrations.agentcore import (
    AgentCoreRuntimeSettings,
    start_agentcore_background_services,
)
from vijil_dome.integrations.agentcore.otel_exporters import (
    setup_agentcore_otel_exporters_from_env,
    setup_agentcore_otel_for_dome,
)


def test_setup_otel_exporters_explicit_disabled():
    pytest.importorskip("opentelemetry.sdk.trace")
    assert setup_agentcore_otel_exporters_from_env(enabled=False) is None


def test_setup_otel_exporters_enabled_missing_endpoint_returns_none(monkeypatch):
    pytest.importorskip("opentelemetry.sdk.trace")
    for k in (
        "DOME_OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
        "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
    ):
        monkeypatch.delenv(k, raising=False)
    assert (
        setup_agentcore_otel_exporters_from_env(
            enabled=True,
            settings=AgentCoreRuntimeSettings(),
        )
        is None
    )


def test_create_agentcore_otel_resource():
    pytest.importorskip("opentelemetry.sdk.resources")
    from vijil_dome.integrations.agentcore.otel_exporters import create_agentcore_otel_resource

    r = create_agentcore_otel_resource("team-1", "agent-1")
    assert r.attributes["service.name"] == "vijil.dome"
    assert r.attributes["service.namespace"] == "team-1"
    assert r.attributes["team.id"] == "team-1"
    assert r.attributes["agent.id"] == "agent-1"
    r2 = create_agentcore_otel_resource("t2", None)
    assert r2.attributes.get("agent.id") is None


def test_setup_otel_uses_settings_for_team_agent(monkeypatch):
    pytest.importorskip("opentelemetry.sdk.trace")
    pytest.importorskip("opentelemetry.instrumentation.logging")
    for k in (
        "DOME_OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
        "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TIMEOUT", "1")
    monkeypatch.delenv("TEAM_ID", raising=False)
    monkeypatch.delenv("AGENT_ID", raising=False)

    settings = AgentCoreRuntimeSettings(
        team_id="from-settings",
        agent_id="agent-s",
        otel_exporter_otlp_endpoint="http://127.0.0.1:4318",
    )
    handle = setup_agentcore_otel_exporters_from_env(
        settings=settings,
        bridge_stdlib_logging=False,
    )
    assert handle is not None

    from opentelemetry import trace

    tp = trace.get_tracer_provider()
    assert tp.resource.attributes.get("team.id") == "from-settings"
    assert tp.resource.attributes.get("agent.id") == "agent-s"

    handle.shutdown()


def test_setup_otel_exporters_inits_providers(monkeypatch):
    pytest.importorskip("opentelemetry.sdk.trace")
    pytest.importorskip("opentelemetry.instrumentation.logging")
    monkeypatch.setenv("DOME_OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TIMEOUT", "1")
    monkeypatch.setenv("TEAM_ID", "t-res")
    monkeypatch.delenv("AGENT_ID", raising=False)

    handle = setup_agentcore_otel_exporters_from_env(
        bridge_stdlib_logging=False, team_id="override-team", agent_id="a1"
    )
    assert handle is not None

    from opentelemetry import metrics, trace
    from opentelemetry.sdk.trace import TracerProvider

    assert isinstance(trace.get_tracer_provider(), TracerProvider)
    mp = metrics.get_meter_provider()
    assert hasattr(mp, "shutdown")
    tp = trace.get_tracer_provider()
    assert tp.resource.attributes.get("team.id") == "override-team"
    assert tp.resource.attributes.get("agent.id") == "a1"

    handle.shutdown()


SAMPLE_DOME_CFG = {
    "input-guards": [
        {"g": {"type": "security", "methods": ["encoding-heuristics"]}}
    ],
    "output-guards": [],
}


def test_setup_agentcore_otel_for_dome_instruments_and_reinstrument(monkeypatch):
    pytest.importorskip("opentelemetry.sdk.trace")
    pytest.importorskip("opentelemetry.instrumentation.logging")
    monkeypatch.setenv("DOME_OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TIMEOUT", "1")

    dome = Dome(SAMPLE_DOME_CFG)
    handle = setup_agentcore_otel_for_dome(
        dome,
        bridge_stdlib_logging=False,
        team_id="t1",
        agent_id="a1",
    )
    assert handle is not None
    assert hasattr(handle, "reinstrument_dome")
    handle.reinstrument_dome(dome)
    handle.shutdown()


def test_setup_skips_global_providers_when_tracer_already_configured(monkeypatch):
    pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    from vijil_dome.integrations.agentcore.otel_lifecycle import (
        shutdown_opentelemetry_providers,
    )

    trace.set_tracer_provider(TracerProvider())
    try:
        handle = setup_agentcore_otel_exporters_from_env(
            enabled=True,
            settings=AgentCoreRuntimeSettings(),
            bridge_stdlib_logging=False,
        )
        assert handle is not None
        assert handle.manages_global_providers is False
        spy = MagicMock()
        monkeypatch.setattr(
            "vijil_dome.integrations.agentcore.otel_lifecycle.shutdown_opentelemetry_providers",
            spy,
        )
        handle.shutdown()
        spy.assert_not_called()
    finally:
        shutdown_opentelemetry_providers()


def test_setup_agentcore_otel_for_dome_after_host_telemetry(monkeypatch):
    pytest.importorskip("opentelemetry.sdk.trace")
    pytest.importorskip("opentelemetry.instrumentation.logging")
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    from vijil_dome.integrations.agentcore.otel_lifecycle import (
        shutdown_opentelemetry_providers,
    )

    trace.set_tracer_provider(TracerProvider())
    try:
        dome = Dome(SAMPLE_DOME_CFG)
        handle = setup_agentcore_otel_for_dome(
            dome,
            enabled=True,
            settings=AgentCoreRuntimeSettings(),
            bridge_stdlib_logging=False,
        )
        assert handle is not None
        assert handle.manages_global_providers is False
        spy = MagicMock()
        monkeypatch.setattr(
            "vijil_dome.integrations.agentcore.otel_lifecycle.shutdown_opentelemetry_providers",
            spy,
        )
        handle.shutdown()
        spy.assert_not_called()
    finally:
        shutdown_opentelemetry_providers()


def test_background_shutdown_prefers_otel_handle():
    pytest.importorskip("opentelemetry.sdk.trace")
    handle = MagicMock()

    cfg = {
        "input-guards": [
            {"g": {"type": "security", "methods": ["encoding-heuristics"]}}
        ],
        "output-guards": [],
    }
    dome = Dome(cfg)

    spy = MagicMock(wraps=handle)
    bg = start_agentcore_background_services(
        dome,
        settings=AgentCoreRuntimeSettings(),
        otel_exporter_handle=spy,
    )
    bg.shutdown()
    spy.shutdown.assert_called_once()
