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

"""Export a Dome's guard spans over OTLP when the standard OTEL_EXPORTER_OTLP_* env vars are set.

This module imports no OpenTelemetry code at import time: ``Dome.__init__``
imports it unconditionally, and an install without the ``opentelemetry``
extra must keep working. OpenTelemetry is imported only once an endpoint
variable is set.

Every ``Dome`` in the process shares one ``TracerProvider``, built on first
use. The SDK registers an atexit hook that flushes and shuts it down. The
app's global tracer provider is never touched.

An explicit ``instrument_dome`` or ``setup_vijil_telemetry`` call on the
same instance wins: both call :func:`undo_env_instrumentation` first, which
puts back the scan/detect methods this module wrapped.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Iterator, Optional

logger = logging.getLogger("vijil.dome")

_ENDPOINT_VARS = ("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "OTEL_EXPORTER_OTLP_ENDPOINT")
_PROTOCOL_VARS = ("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "OTEL_EXPORTER_OTLP_PROTOCOL")
# Every method instrument_dome replaces on a guardrail, guard or detector.
_WRAPPED_METHODS = ("scan", "async_scan", "detect", "sync_detect")
_SNAPSHOT_ATTR = "_otlp_env_snapshot"
_ABSENT = object()

_EXTRA_HINT = "pip install 'vijil-dome[opentelemetry]'"

_lock = threading.Lock()
_provider: Any = None
_warned: set[str] = set()


def otlp_endpoint_configured() -> bool:
    """True when either OTLP endpoint variable is set to a non-blank value."""
    return any(os.environ.get(name, "").strip() for name in _ENDPOINT_VARS)


def _warn_once(key: str, message: str, *args: Any) -> None:
    with _lock:
        if key in _warned:
            return
        _warned.add(key)
    logger.warning(message, *args)


def _opted_out() -> bool:
    if os.environ.get("OTEL_SDK_DISABLED", "").strip().lower() == "true":
        return True
    exporters = os.environ.get("OTEL_TRACES_EXPORTER", "").strip()
    if not exporters:
        return False  # the spec's default exporter is otlp
    return "otlp" not in {e.strip().lower() for e in exporters.split(",")}


def _protocol() -> str:
    for name in _PROTOCOL_VARS:
        value = os.environ.get(name, "").strip().lower()
        if value:
            return value
    return "http/protobuf"


def _exporter_class() -> Optional[type]:
    """The OTLP span exporter class for the configured protocol, or None to export nothing.

    Raises ImportError when the HTTP exporter (part of the opentelemetry extra) is missing.
    """
    protocol = _protocol()
    if protocol == "http/protobuf":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        return OTLPSpanExporter
    if protocol == "grpc":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter as GrpcSpanExporter,
            )
        except ImportError:
            _warn_once(
                "grpc",
                "OTLP protocol is grpc but the gRPC exporter is not installed, so Dome "
                "exports no spans; install opentelemetry-exporter-otlp-proto-grpc",
            )
            return None
        return GrpcSpanExporter
    _warn_once(
        f"protocol:{protocol}",
        "OTLP protocol %r is not supported for Dome spans (use http/protobuf or grpc), "
        "so Dome exports no spans",
        protocol,
    )
    return None


def _shared_tracer() -> Any:
    """A tracer on the process-wide Dome provider, building it on first use; None if unavailable."""
    global _provider
    with _lock:
        provider = _provider
    if provider is None:
        try:
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            exporter_cls = _exporter_class()
        except ImportError:
            _warn_once(
                "extra",
                "OTEL_EXPORTER_OTLP_* is set but Dome cannot export spans without the "
                "opentelemetry extra: %s",
                _EXTRA_HINT,
            )
            return None
        if exporter_cls is None:
            return None
        with _lock:
            if _provider is None:
                # No arguments: the SDK reads endpoint, headers, timeout,
                # compression and certificates from OTEL_EXPORTER_OTLP_*.
                new_provider = TracerProvider(resource=Resource.create())
                new_provider.add_span_processor(BatchSpanProcessor(exporter_cls()))
                _provider = new_provider
            provider = _provider
    return provider.get_tracer("vijil.dome")


def _wrapped_targets(dome: Any) -> Iterator[Any]:
    for guardrail in (dome.input_guardrail, dome.output_guardrail):
        if guardrail is None:
            continue
        yield guardrail
        for guard in guardrail.guard_list:
            yield guard
            yield from guard.detector_list


def _snapshot_methods(dome: Any) -> list[tuple[Any, dict[str, Any]]]:
    return [
        (obj, {name: getattr(obj, "__dict__", {}).get(name, _ABSENT) for name in _WRAPPED_METHODS})
        for obj in _wrapped_targets(dome)
    ]


def is_env_instrumented(dome: Any) -> bool:
    """True when *dome*'s current instrumentation came from the OTLP env vars."""
    return isinstance(getattr(dome, _SNAPSHOT_ATTR, None), list)


def undo_env_instrumentation(dome: Any) -> None:
    """Put back the methods env instrumentation wrapped, so an explicit setup can instrument afresh.

    A no-op on an instance this module did not instrument.
    """
    snapshot = getattr(dome, _SNAPSHOT_ATTR, None)
    if not isinstance(snapshot, list):
        return
    for obj, methods in snapshot:
        for name, value in methods.items():
            if value is _ABSENT:
                obj.__dict__.pop(name, None)
            else:
                setattr(obj, name, value)
    setattr(dome, _SNAPSHOT_ATTR, None)
    dome._instrumented = False


def instrument_from_env(dome: Any) -> None:
    """Wrap *dome*'s guards with spans exported over OTLP, if the OTLP env vars ask for it."""
    if not otlp_endpoint_configured() or _opted_out():
        return
    if getattr(dome, "_instrumented", False):
        return
    tracer = _shared_tracer()
    if tracer is None:
        return
    try:
        from vijil_dome.integrations.instrumentation.otel_instrumentation import (
            instrument_dome,
        )
    except ImportError:
        _warn_once(
            "extra",
            "OTEL_EXPORTER_OTLP_* is set but Dome cannot export spans without the "
            "opentelemetry extra: %s",
            _EXTRA_HINT,
        )
        return
    snapshot = _snapshot_methods(dome)
    instrument_dome(dome, handler=None, tracer=tracer, meter=None)
    setattr(dome, _SNAPSHOT_ATTR, snapshot)
