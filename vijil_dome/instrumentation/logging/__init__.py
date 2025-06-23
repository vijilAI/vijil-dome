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
import toml
import socket
from pathlib import Path
from typing import Dict, Any, Optional
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import (
    ConsoleLogExporter,
    BatchLogRecordProcessor,
    SimpleLogRecordProcessor,
)
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.resources import (
    Resource,
    SERVICE_NAME,
    SERVICE_VERSION,
    DEPLOYMENT_ENVIRONMENT,
)
from opentelemetry._logs import get_logger_provider, set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

DEFAULT_CONFIG_PATH = Path(__file__).parent / "default_logging_config.toml"


def set_log_level(level: int) -> None:
    logging.getLogger().setLevel(level)
    for handler in logging.getLogger().handlers:
        handler.setLevel(level)


def get_log_level(level_str: str) -> int:
    return getattr(logging, level_str.upper())


def setup_vijil_logging(config: Optional[Dict[str, Any]] = None) -> LoggerProvider:
    default_config = _load_default_config()
    if config:
        default_config.update(config)
    config = default_config

    resource = Resource.create(
        {
            SERVICE_NAME: config["resource"]["service_name"],
            SERVICE_VERSION: config["resource"]["service_version"],
            DEPLOYMENT_ENVIRONMENT: config["resource"]["deployment_environment"],
        }
    )

    logger_provider = LoggerProvider(resource=resource)

    # Set up processor_exporters
    for pe_config in config.get("processor_exporters", []):
        exporter = _create_exporter(pe_config["exporter"])
        processor = _create_processor(pe_config["processor"], exporter)
        logger_provider.add_log_record_processor(processor)

    # Set the global LoggerProvider if it's not already set
    current_provider = get_logger_provider()
    if not isinstance(current_provider, LoggerProvider):
        set_logger_provider(logger_provider)

    # Create handlers
    handlers = []
    for handler_config in config.get("handlers", []):
        handler = _create_handler(handler_config, logger_provider)
        handlers.append(handler)

    # Configure the root logger with all handlers
    logging.basicConfig(
        handlers=handlers,
        level=get_log_level(config["logging"]["log_level"]),
        force=True,
    )

    # Set up the LoggingInstrumentor if it has not been instrumented yet
    if not LoggingInstrumentor().is_instrumented_by_opentelemetry:
        LoggingInstrumentor().instrument(set_logging_format=True)

    return logger_provider


def _get_ip_address():
    return socket.gethostbyname(socket.gethostname())


CUSTOM_FIELD_FUNCTIONS = {
    "ip_address": _get_ip_address,
    # Add more custom fields here, e.g.:
    # 'process_id': os.getpid,
}


class CustomFormatter(logging.Formatter):
    def format(self, record):
        for field, func in CUSTOM_FIELD_FUNCTIONS.items():
            if not hasattr(record, field):
                setattr(record, field, func())
        return super().format(record)


def _create_exporter(exporter_config):
    exporter_type = exporter_config["type"]
    if exporter_type == "console":
        return ConsoleLogExporter()
    elif exporter_type == "otlp":
        return OTLPLogExporter(
            endpoint=exporter_config.get("endpoint", "http://localhost:4317"),
            insecure=exporter_config.get("insecure", False),
        )
    else:
        raise ValueError(f"Unsupported exporter type: {exporter_type}")


def _create_processor(processor_config, exporter):
    processor_type = processor_config["type"]
    if processor_type == "batch":
        return BatchLogRecordProcessor(
            exporter,
            max_export_batch_size=processor_config.get("max_export_batch_size", 512),
            export_timeout_millis=processor_config.get("export_timeout_millis", 30000),
            max_queue_size=processor_config.get("max_queue_size", 2048),
        )
    elif processor_type == "simple":
        return SimpleLogRecordProcessor(exporter)
    else:
        raise ValueError(f"Unsupported processor type: {processor_type}")


def _create_handler(handler_config, logger_provider):
    handler_type = handler_config["type"]
    if handler_type == "otel":
        handler = LoggingHandler(
            level=get_log_level(handler_config["level"]),
            logger_provider=logger_provider,
        )
    elif handler_type == "stream":
        handler = logging.StreamHandler()
        handler.setLevel(get_log_level(handler_config["level"]))
    elif handler_type == "file":
        handler = logging.FileHandler(handler_config["path"])
        handler.setLevel(get_log_level(handler_config["level"]))
    else:
        raise ValueError(f"Unsupported handler type: {handler_type}")

    formatter = CustomFormatter(handler_config["format"])
    handler.setFormatter(formatter)
    return handler


def _load_default_config() -> Dict[str, Any]:
    with open(DEFAULT_CONFIG_PATH, "r") as f:
        return toml.load(f)
