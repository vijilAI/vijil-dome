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

"""Environment-driven settings for AgentCore-friendly Dome runtime helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _env_opt_str(name: str) -> Optional[str]:
    v = os.environ.get(name, "").strip()
    return v or None


@dataclass(frozen=True)
class AgentCoreRuntimeSettings:
    """Identifiers and Dome-specific OTLP base URL for AgentCore helpers."""

    team_id: str = ""
    """From ``TEAM_ID``; used for OTel resource and S3 config key (with ``agent_id``)."""

    agent_id: Optional[str] = None
    """From ``AGENT_ID``; ``agent.id`` on the OTel resource and S3 key segment."""

    s3_config_bucket: Optional[str] = None
    """From ``DOME_CONFIG_S3_BUCKET``; with ``team_id`` and ``agent_id``, enables settings-based S3 polling."""

    otel_exporter_otlp_endpoint: Optional[str] = None
    """From ``DOME_OTEL_EXPORTER_OTLP_ENDPOINT``; OTLP/HTTP base for Dome exporters (paths appended per signal)."""


def load_agentcore_runtime_settings_from_env() -> AgentCoreRuntimeSettings:
    """Load :class:`AgentCoreRuntimeSettings` from environment variables.

    Variables (all optional):

    - ``TEAM_ID`` / ``AGENT_ID`` — ``team.id`` / ``agent.id`` for OTel and the
      standard S3 object key ``teams/{team}/agents/{agent}/dome/config.json`` when polling
      without :meth:`~vijil_dome.Dome.Dome.create_from_s3`.
    - ``DOME_CONFIG_S3_BUCKET`` — bucket for that key when not using ``create_from_s3``.
      Together with non-empty ``TEAM_ID`` and ``AGENT_ID``, this enables the S3 config poller
      in :func:`~.background.start_agentcore_background_services`.
    - ``DOME_OTEL_EXPORTER_OTLP_ENDPOINT`` — Dome OTLP/HTTP base URL (avoids the host-wide
      ``OTEL_EXPORTER_OTLP_ENDPOINT``). Paths ``v1/traces`` / ``v1/metrics`` / ``v1/logs`` are
      appended unless you set per-signal ``OTEL_EXPORTER_OTLP_*_ENDPOINT`` in the environment
      (see :func:`~.otel_exporters.setup_agentcore_otel_exporters_from_env`).
    """
    return AgentCoreRuntimeSettings(
        team_id=os.getenv("TEAM_ID", "").strip(),
        agent_id=_env_opt_str("AGENT_ID"),
        s3_config_bucket=_env_opt_str("DOME_CONFIG_S3_BUCKET"),
        otel_exporter_otlp_endpoint=_env_opt_str("DOME_OTEL_EXPORTER_OTLP_ENDPOINT"),
    )
