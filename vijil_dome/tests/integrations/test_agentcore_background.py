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
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from vijil_dome import Dome
from vijil_dome.integrations.agentcore import (
    AgentCoreOtelExporterHandle,
    AgentCoreRuntimeSettings,
    combine_agentcore_s3_reload_callbacks,
    DomeS3ConfigPoller,
    load_agentcore_runtime_settings_from_env,
    start_agentcore_background_services,
)
from vijil_dome.utils.config_loader import build_s3_config_key

SAMPLE_CONFIG = {
    "input-guards": [
        {
            "security_min": {
                "type": "security",
                "methods": ["encoding-heuristics"],
            }
        }
    ],
    "output-guards": [],
}

BUCKET = "b"
TEAM_ID = "t"
AGENT_ID = "a"


@patch("vijil_dome.utils.config_loader._create_s3_client")
def test_dome_reload_from_s3_if_changed(mock_create_client, tmp_path):
    mock_client = MagicMock()
    mock_create_client.return_value = mock_client
    mock_client.get_object.return_value = {
        "Body": io.BytesIO(json.dumps(SAMPLE_CONFIG).encode()),
        "ETag": '"e1"',
    }
    dome = Dome.create_from_s3(
        BUCKET, team_id=TEAM_ID, agent_id=AGENT_ID, cache_dir=str(tmp_path)
    )
    mock_client.get_object.return_value = {
        "Body": io.BytesIO(json.dumps(SAMPLE_CONFIG).encode()),
        "ETag": '"e2"',
    }
    assert dome.reload_from_s3_if_changed() is False

    new_cfg = {**SAMPLE_CONFIG, "id": "new-id"}
    mock_client.get_object.return_value = {
        "Body": io.BytesIO(json.dumps(new_cfg).encode()),
        "ETag": '"e3"',
    }
    assert dome.reload_from_s3_if_changed() is True
    assert dome.config_id == "new-id"


def test_load_agentcore_runtime_settings_from_env(monkeypatch):
    s = load_agentcore_runtime_settings_from_env()
    assert s.team_id == ""
    assert s.agent_id is None
    assert s.s3_config_bucket is None
    assert s.otel_exporter_otlp_endpoint is None

    monkeypatch.setenv("TEAM_ID", " team-x ")
    monkeypatch.setenv("AGENT_ID", " agent-y ")
    monkeypatch.setenv("DOME_CONFIG_S3_BUCKET", "my-bucket")
    monkeypatch.setenv("DOME_OTEL_EXPORTER_OTLP_ENDPOINT", " http://collector:4318 ")
    s2 = load_agentcore_runtime_settings_from_env()
    assert s2.team_id == "team-x"
    assert s2.agent_id == "agent-y"
    assert s2.s3_config_bucket == "my-bucket"
    assert s2.otel_exporter_otlp_endpoint == "http://collector:4318"


@patch("vijil_dome.utils.config_loader._create_s3_client")
def test_s3_poller_stop_joins_thread(mock_create_client, tmp_path):
    mock_client = MagicMock()
    mock_create_client.return_value = mock_client
    mock_client.get_object.return_value = {
        "Body": io.BytesIO(json.dumps(SAMPLE_CONFIG).encode()),
        "ETag": '"e1"',
    }
    dome = Dome.create_from_s3(
        BUCKET, team_id=TEAM_ID, agent_id=AGENT_ID, cache_dir=str(tmp_path)
    )
    poller = DomeS3ConfigPoller(
        dome, AgentCoreRuntimeSettings(), interval_seconds=0.05
    )
    poller.start()
    assert poller.is_running
    deadline = time.time() + 2.0
    while not mock_client.get_object.called and time.time() < deadline:
        time.sleep(0.01)
    poller.stop(join_timeout=5.0)
    assert not poller.is_running


def test_start_agentcore_background_services_no_poll_without_s3_metadata():
    dome = Dome(SAMPLE_CONFIG)
    bg = start_agentcore_background_services(
        dome,
        settings=AgentCoreRuntimeSettings(),
    )
    assert bg._poller is None
    bg.shutdown()


@patch("vijil_dome.utils.config_loader._create_s3_client")
def test_start_agentcore_starts_poller_when_enabled(mock_create_client, tmp_path):
    mock_client = MagicMock()
    mock_create_client.return_value = mock_client
    mock_client.get_object.return_value = {
        "Body": io.BytesIO(json.dumps(SAMPLE_CONFIG).encode()),
        "ETag": '"e1"',
    }
    dome = Dome.create_from_s3(
        BUCKET, team_id=TEAM_ID, agent_id=AGENT_ID, cache_dir=str(tmp_path)
    )
    bg = start_agentcore_background_services(
        dome,
        settings=AgentCoreRuntimeSettings(),
        poll_interval_seconds=30.0,
    )
    assert bg._poller is not None
    assert bg._poller.is_running
    bg.shutdown(join_timeout=5.0)
    assert not bg._poller.is_running


def test_poller_start_raises_without_s3_coordinates():
    dome = Dome(SAMPLE_CONFIG)
    poller = DomeS3ConfigPoller(
        dome, AgentCoreRuntimeSettings(), interval_seconds=1.0
    )
    with pytest.raises(ValueError, match="DomeS3ConfigPoller requires"):
        poller.start()


def test_start_agentcore_no_poller_when_settings_missing_agent():
    dome = Dome(SAMPLE_CONFIG)
    bg = start_agentcore_background_services(
        dome,
        settings=AgentCoreRuntimeSettings(
            s3_config_bucket="b",
            team_id="t",
            agent_id=None,
        ),
    )
    assert bg._poller is None
    bg.shutdown()


@patch("vijil_dome.integrations.agentcore.background.load_dome_config_from_s3")
def test_poller_settings_mode_applies_remote_config(mock_load):
    mock_load.side_effect = [
        {**SAMPLE_CONFIG, "id": "id1"},
        {**SAMPLE_CONFIG, "id": "id2"},
    ]
    dome = Dome(SAMPLE_CONFIG)
    settings = AgentCoreRuntimeSettings(
        s3_config_bucket="my-bucket",
        team_id="t",
        agent_id="a",
    )
    poller = DomeS3ConfigPoller(dome, settings, interval_seconds=0.05)
    poller.start()
    deadline = time.time() + 3.0
    while dome.config_id != "id1" and time.time() < deadline:
        time.sleep(0.02)
    assert dome.config_id == "id1"
    mock_load.assert_called_with(
        bucket="my-bucket",
        key=build_s3_config_key("t", "a"),
        cache_ttl_seconds=0,
    )
    deadline = time.time() + 3.0
    while dome.config_id != "id2" and time.time() < deadline:
        time.sleep(0.02)
    assert dome.config_id == "id2"
    poller.stop(join_timeout=5.0)


def test_combine_agentcore_s3_reload_callbacks_runs_otel_then_user():
    order = []
    handle = MagicMock()
    handle.reinstrument_dome = lambda d: order.append("otel")

    def user(d):
        order.append("user")

    merged = combine_agentcore_s3_reload_callbacks(
        otel_exporter_handle=handle,
        on_s3_reload=user,
    )
    assert merged is not None
    dome = MagicMock()
    merged(dome)
    assert order == ["otel", "user"]


def test_combine_agentcore_s3_reload_callbacks_plain_handle_delegates_user_only():
    order = []
    plain = AgentCoreOtelExporterHandle()

    def user(d):
        order.append("user")

    merged = combine_agentcore_s3_reload_callbacks(
        otel_exporter_handle=plain,
        on_s3_reload=user,
    )
    assert merged is user
    merged(MagicMock())
    assert order == ["user"]


def test_combine_agentcore_s3_reload_callbacks_reinstrument_only():
    order = []
    handle = MagicMock()
    handle.reinstrument_dome = lambda d: order.append("otel")
    merged = combine_agentcore_s3_reload_callbacks(
        otel_exporter_handle=handle,
        on_s3_reload=None,
    )
    assert merged is not None
    merged(MagicMock())
    assert order == ["otel"]
