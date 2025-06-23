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

from opentelemetry.trace import Tracer
from opentelemetry.metrics import Meter
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from vijil_dome.instrumentation.tracing import auto_trace
from vijil_dome.guardrails.instrumentation.instrumentation import (
    instrument_with_monitors,
    instrument_with_tracer,
)
from vijil_dome import Dome
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


def instrument_dome(dome: Dome, handler: logging.Handler, tracer: Tracer, meter: Meter):
    if not LoggingInstrumentor().is_instrumented_by_opentelemetry:
        LoggingInstrumentor().instrument()

    # Enable OTel logging
    if handler:
        logger = logging.getLogger("vijil.dome")
        logger.addHandler(handler)
        instrument_logger(logger)

    # By default, the log level is set to warning. For this example, we set it to info
    logger.setLevel(logging.INFO)
    # Add tracer
    if tracer:
        dome.guard_input = auto_trace(tracer, "Dome-Guard-Input")(dome.guard_input)  # type: ignore[method-assign]
        dome.guard_output = auto_trace(tracer, "Dome-Guard-Output")(dome.guard_output)  # type: ignore[method-assign]

        instrument_with_tracer(dome.input_guardrail, tracer, "Dome-Input-Guardrail")
        instrument_with_tracer(dome.output_guardrail, tracer, "Dome-Output-Guardrail")

    if meter:
        # add monitors
        instrument_with_monitors(dome.input_guardrail, meter, "dome-input")
        instrument_with_monitors(dome.output_guardrail, meter, "dome-output")
