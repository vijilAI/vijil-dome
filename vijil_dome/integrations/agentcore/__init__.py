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

"""AgentCore-oriented helpers: env-driven S3 config polling and OTel lifecycle."""

from .background import (
    AGENTCORE_S3_POLL_INTERVAL_SECONDS,
    AgentCoreBackgroundServices,
    DomeS3ConfigPoller,
    combine_agentcore_s3_reload_callbacks,
    start_agentcore_background_services,
)
from .otel_exporters import (
    AgentCoreDomeOtelExporterHandle,
    AgentCoreOtelExporterHandle,
    create_agentcore_otel_resource,
    setup_agentcore_otel_exporters_from_env,
    setup_agentcore_otel_for_dome,
)
from .otel_lifecycle import shutdown_opentelemetry_providers
from .settings import AgentCoreRuntimeSettings, load_agentcore_runtime_settings_from_env

__all__ = [
    "AGENTCORE_S3_POLL_INTERVAL_SECONDS",
    "AgentCoreBackgroundServices",
    "AgentCoreDomeOtelExporterHandle",
    "AgentCoreOtelExporterHandle",
    "AgentCoreRuntimeSettings",
    "combine_agentcore_s3_reload_callbacks",
    "create_agentcore_otel_resource",
    "DomeS3ConfigPoller",
    "load_agentcore_runtime_settings_from_env",
    "setup_agentcore_otel_exporters_from_env",
    "setup_agentcore_otel_for_dome",
    "shutdown_opentelemetry_providers",
    "start_agentcore_background_services",
]
