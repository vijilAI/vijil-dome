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

import socket
from opentelemetry import trace
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry import metrics
from opentelemetry.exporter.cloud_monitoring import (
    CloudMonitoringMetricsExporter,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
import google.cloud.logging
from google.cloud.logging_v2.handlers import CloudLoggingHandler


def create_gcp_resource():
    resource = Resource(
        attributes={
            "service.name": "ex-core-gcp-example",
            "service.version": "1.0.0",
            "service.ip": socket.gethostbyname(
                socket.gethostbyname(socket.gethostname())
            ),
        }
    )
    return resource


def create_gcp_tracer(resource: Resource):
    tracer_provider = TracerProvider(
        resource=resource,
    )
    cloud_trace_exporter = CloudTraceSpanExporter()
    tracer_provider.add_span_processor(
        # BatchSpanProcessor buffers spans and sends them in batches in a
        # background thread. The default parameters are sensible, but can be
        # tweaked to optimize your performance
        BatchSpanProcessor(cloud_trace_exporter)
    )
    trace.set_tracer_provider(tracer_provider)
    tracer = tracer_provider.get_tracer("ex-core-gcp-tracer")
    return tracer


def create_gcp_meter(resource: Resource):
    exporter = CloudMonitoringMetricsExporter(add_unique_identifier=True)
    reader = PeriodicExportingMetricReader(exporter, export_interval_millis=6000)
    provider = MeterProvider(metric_readers=[reader], resource=resource)
    metrics.set_meter_provider(provider)
    meter = metrics.get_meter("ex-core-gcp-meter")
    return meter


def create_gcp_log_handler():
    client = google.cloud.logging.Client()
    gcp_log_handler = CloudLoggingHandler(client)
    return gcp_log_handler
