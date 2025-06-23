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

"""
Helper file - generate OTel instruments for examples
Includes helper functions to create resources, log handlers, tracers and meters
"""

import logging
import grpc
import socket
import os

from opentelemetry.sdk.resources import Resource

# Opentelemetry Logging Imports
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter,
)
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

# Opentelemetry Metrics imports
from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.sdk import metrics as sdkmetrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    AggregationTemporality,
    PeriodicExportingMetricReader,
)

# Opentelemetry Tracing imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)


# This example uses Uptrace. Make sure the uptrace DSN is set.
def get_uptrace_dsn():
    dsn = os.environ.get("UPTRACE_DSN")
    if not dsn:
        raise ValueError(
            "Uptrace DSN was not set. Please set the environment variable UPTRACE_DSN."
        )
    return dsn


def create_otel_resource():
    resource = Resource(
        attributes={
            "service.name": "ex-core-instrumentation-example",
            "service.version": "1.0.0",
            "service.ip": socket.gethostbyname(
                socket.gethostbyname(socket.gethostname())
            ),
        }
    )
    return resource


def create_otel_log_handler(resource: Resource):
    dsn = get_uptrace_dsn()
    logger_provider = LoggerProvider(resource=resource)
    set_logger_provider(logger_provider)

    log_exporter = OTLPLogExporter(
        endpoint="otlp.uptrace.dev:4317",
        headers=(("uptrace-dsn", dsn),),
        timeout=5,
        compression=grpc.Compression.Gzip,
    )
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))

    otel_handler = LoggingHandler(level=logging.INFO, logger_provider=logger_provider)
    return otel_handler


def create_otel_tracer(resource: Resource):
    dsn = get_uptrace_dsn()
    tracer_provider = TracerProvider(
        resource=resource,
    )
    trace.set_tracer_provider(tracer_provider)

    exporter_span = OTLPSpanExporter(
        endpoint="otlp.uptrace.dev:4317",
        headers=(("uptrace-dsn", dsn),),
        timeout=5,
        compression=grpc.Compression.Gzip,
    )

    span_processor = BatchSpanProcessor(
        exporter_span,
        max_queue_size=1000,
        max_export_batch_size=1000,
    )
    tracer_provider.add_span_processor(span_processor)

    tracer = tracer_provider.get_tracer("ex-core-tracer")
    return tracer


def create_otel_meter(resource: Resource):
    dsn = get_uptrace_dsn()
    temporality_delta = {
        sdkmetrics.Counter: AggregationTemporality.DELTA,
        sdkmetrics.UpDownCounter: AggregationTemporality.DELTA,
        sdkmetrics.Histogram: AggregationTemporality.DELTA,
        sdkmetrics.ObservableCounter: AggregationTemporality.DELTA,
        sdkmetrics.ObservableUpDownCounter: AggregationTemporality.DELTA,
        sdkmetrics.ObservableGauge: AggregationTemporality.DELTA,
    }  # type: dict[type, AggregationTemporality]

    metric_exporter = OTLPMetricExporter(
        endpoint="otlp.uptrace.dev:4317",
        headers=(("uptrace-dsn", dsn),),
        timeout=5,
        compression=grpc.Compression.Gzip,
        preferred_temporality=temporality_delta,
    )

    reader = PeriodicExportingMetricReader(metric_exporter)
    provider = MeterProvider(metric_readers=[reader], resource=resource)
    metrics.set_meter_provider(provider)
    meter = metrics.get_meter("ex-core-meter")
    return meter
