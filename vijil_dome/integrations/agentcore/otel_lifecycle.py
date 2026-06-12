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

"""Graceful shutdown for global OpenTelemetry SDK providers (metrics export threads, etc.)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("vijil.dome")


def shutdown_opentelemetry_providers() -> None:
    """Shut down global TracerProvider, MeterProvider, and LoggerProvider if they expose ``shutdown``.

    Safe to call when OpenTelemetry is not installed or only the no-op API is
    registered. Intended to be invoked from application shutdown paths (for
    example after stopping other background threads) so batch exporters flush.
    """
    try:
        from opentelemetry import metrics, trace
    except ImportError:
        logger.debug("OpenTelemetry not installed; skipping provider shutdown")
        return

    _shutdown_if_supported("TracerProvider", trace.get_tracer_provider())
    _shutdown_if_supported("MeterProvider", metrics.get_meter_provider())

    try:
        from opentelemetry._logs import get_logger_provider
    except ImportError:
        return
    _shutdown_if_supported("LoggerProvider", get_logger_provider())


def _shutdown_if_supported(kind: str, provider: Any) -> None:
    shutdown = getattr(provider, "shutdown", None)
    if shutdown is None:
        return
    try:
        shutdown()
        logger.info("Shut down OpenTelemetry %s", kind)
    except Exception:
        logger.exception("Error while shutting down OpenTelemetry %s", kind)
