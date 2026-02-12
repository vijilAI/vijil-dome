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
#
# vijil and vijil-dome are trademarks owned by Vijil Inc.

import logging
from functools import wraps
from typing import Any, Optional
from opentelemetry.sdk.trace import Tracer
from opentelemetry.metrics import Meter
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from vijil_dome.guardrails.instrumentation.instrumentation import (
    instrument_with_monitors,
    instrument_with_tracer,
)
from vijil_dome import Dome
from vijil_dome.guardrails import Guardrail, GuardrailResult
from vijil_dome.integrations.vijil.telemetry import _set_darwin_span_attributes
import socket


class VijilLogFormatter(logging.Formatter):
    def format(self, record):
        # Set the default values of the OTel Logging information if absent
        # this is only the case if OTel instrumentation is disabled
        record.otelTraceID = getattr(record, "otelTraceID", 0)
        record.otelSpanID = getattr(record, "otelSpanID", 0)
        record.otelServiceName = getattr(record, "otelServiceName", "N/A")
        record.otelTraceSampled = getattr(record, "otelTraceSampled", "N/A")

        # Add the IP address to the log record if it doesn't exist
        record.ip = getattr(
            record,
            "ip",
            socket.gethostbyname(socket.gethostname()),
        )
        return super().format(record)


def get_vijil_log_formatter():
    formatter = VijilLogFormatter(
        "%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] [trace_id=%(otelTraceID)s span_id=%(otelSpanID)s resource.service.name=%(otelServiceName)s trace_sampled=%(otelTraceSampled)s resource.service.ip=%(ip)s] - %(msg)s"
    )
    return formatter


def instrument_logger(logger: logging.Logger):
    for handler in logger.handlers:
        formatter = get_vijil_log_formatter()
        handler.setFormatter(formatter)


def _add_darwin_detection_spans(
    guardrail: Guardrail,
    tracer: Tracer,
    guardrail_name: str,
) -> None:
    """Wrap guardrail scan methods to emit Darwin-compatible detection spans.

    Creates 'dome-detection' spans with structured attributes that Darwin's
    TelemetryDetectionAdapter can query from Tempo traces.

    Attributes set on each span:
        - dome.guardrail: guardrail name (e.g., "dome-input", "dome-output")
        - detection.label: "flagged" or "clean"
        - detection.score: max detection score (0.0-1.0)
        - detection.method: name of the triggered guard/detector
        - team.id: team context (from kwargs)
        - agent.id: agent context (from kwargs)

    Args:
        guardrail: The Guardrail instance to wrap.
        tracer: OTEL tracer for span creation.
        guardrail_name: Name prefix (e.g., "dome-input", "dome-output").
    """
    original_scan = guardrail.scan
    original_async_scan = guardrail.async_scan

    @wraps(original_scan)
    def scan_with_darwin_spans(*args: Any, **kwargs: Any) -> Any:
        # Extract team_id before passing to original scan (which doesn't accept it)
        team_id = kwargs.pop("team_id", None)
        agent_id = kwargs.get("agent_id")
        with tracer.start_as_current_span("dome-detection") as span:
            span.set_attribute("dome.guardrail", guardrail_name)
            result = original_scan(*args, **kwargs)
            if isinstance(result, GuardrailResult):
                _set_darwin_span_attributes(
                    span,
                    result,
                    agent_id=agent_id,
                    team_id=team_id,
                )
            return result

    @wraps(original_async_scan)
    async def async_scan_with_darwin_spans(*args: Any, **kwargs: Any) -> Any:
        # Extract team_id before passing to original scan (which doesn't accept it)
        team_id = kwargs.pop("team_id", None)
        agent_id = kwargs.get("agent_id")
        with tracer.start_as_current_span("dome-detection") as span:
            span.set_attribute("dome.guardrail", guardrail_name)
            result = await original_async_scan(*args, **kwargs)
            if isinstance(result, GuardrailResult):
                _set_darwin_span_attributes(
                    span,
                    result,
                    agent_id=agent_id,
                    team_id=team_id,
                )
            return result

    guardrail.scan = scan_with_darwin_spans  # type: ignore[method-assign]
    guardrail.async_scan = async_scan_with_darwin_spans  # type: ignore[method-assign]


def instrument_dome(
    dome: Dome,
    handler: Optional[logging.Handler],
    tracer: Optional[Tracer],
    meter: Optional[Meter],
):
    if not LoggingInstrumentor().is_instrumented_by_opentelemetry:
        LoggingInstrumentor().instrument()

    # Enable OTel logging if a logging handler is provided
    if handler:
        logger = logging.getLogger("vijil.dome")
        logger.addHandler(handler)
        instrument_logger(logger)

    # Add tracer for detailed per-guard/per-detector spans
    if tracer:
        if dome.input_guardrail is not None:
            instrument_with_tracer(dome.input_guardrail, tracer, "Dome-Input-Guardrail")
        if dome.output_guardrail is not None:
            instrument_with_tracer(
                dome.output_guardrail, tracer, "Dome-Output-Guardrail"
            )

    if meter:
        # Add split metrics (dome-input-*, dome-output-*)
        if dome.input_guardrail is not None:
            instrument_with_monitors(dome.input_guardrail, meter, "dome-input")
        if dome.output_guardrail is not None:
            instrument_with_monitors(dome.output_guardrail, meter, "dome-output")

    # Add Darwin-compatible detection spans at the guardrail level.
    # These must be added AFTER monitors and tracer so the "dome-detection"
    # span wraps the full scan chain (metrics + generic traces + original scan).
    if tracer:
        if dome.input_guardrail is not None:
            _add_darwin_detection_spans(dome.input_guardrail, tracer, "dome-input")
        if dome.output_guardrail is not None:
            _add_darwin_detection_spans(dome.output_guardrail, tracer, "dome-output")
