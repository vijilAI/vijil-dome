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

Telemetry never breaks ``Dome()``: a malformed env value, a missing
exporter or a failure while wrapping logs one warning and leaves the
instance uninstrumented.

Wrapping is tracked per guardrail object, not per Dome, because Domes built
from one ``DomeConfig`` share their guardrails. A shared guardrail is
wrapped once, and stays wrapped while any Dome using it still relies on
the env export.

An explicit ``instrument_dome`` or ``setup_vijil_telemetry`` call on the
same instance wins: both go through :func:`undo_env_instrumentation`,
which puts back the scan/detect methods this module wrapped.
"""

from __future__ import annotations

import logging
import os
import threading
import weakref
from types import SimpleNamespace
from typing import Any, Iterator, Optional

logger = logging.getLogger("vijil.dome")

_ENDPOINT_VARS = ("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "OTEL_EXPORTER_OTLP_ENDPOINT")
_PROTOCOL_VARS = ("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "OTEL_EXPORTER_OTLP_PROTOCOL")
# The OTEL_TRACES_EXPORTER names the Python SDK registers for OTLP, and the
# protocol each implies when no *_PROTOCOL variable is set.
_OTLP_EXPORTER_PROTOCOLS = {
    "otlp": "http/protobuf",
    "otlp_proto_http": "http/protobuf",
    "otlp_proto_grpc": "grpc",
}
# Every method instrument_dome replaces on a guardrail, guard or detector.
_WRAPPED_METHODS = ("scan", "async_scan", "detect", "sync_detect")
_GUARDRAIL_STATE_ATTR = "_otlp_env_state"  # on a Guardrail this module wrapped
_DOME_GUARDRAILS_ATTR = "_otlp_env_guardrails"  # on a Dome relying on the env export
_ABSENT = object()

_EXTRA_HINT = "pip install 'vijil-dome[opentelemetry]'"
_NOT_EXPORTED = "so Dome guard spans are not exported"

# Re-entrant: _exporter_class warns (taking the lock) while _shared_tracer holds it.
_lock = threading.RLock()
_provider: Any = None
_warned: set[str] = set()

Snapshot = list[tuple[Any, dict[str, Any]]]


class _GuardrailState:
    """The methods a guardrail had before env wrapping, and the Domes relying on that wrapping."""

    def __init__(self, snapshot: Snapshot) -> None:
        self.snapshot = snapshot
        self.owners: weakref.WeakSet[Any] = weakref.WeakSet()


def otlp_endpoint_configured() -> bool:
    """True when either OTLP endpoint variable is set to a non-blank value."""
    return any(os.environ.get(name, "").strip() for name in _ENDPOINT_VARS)


def _warn_once(key: str, message: str, *args: Any) -> None:
    with _lock:
        if key in _warned:
            return
        _warned.add(key)
    logger.warning(message, *args)


def _otlp_exporter_name() -> Optional[str]:
    """The first OTLP name in OTEL_TRACES_EXPORTER (default ``otlp``), or None if it lists none."""
    value = os.environ.get("OTEL_TRACES_EXPORTER", "").strip()
    names = [n.strip().lower() for n in value.split(",")] if value else ["otlp"]
    return next((n for n in names if n in _OTLP_EXPORTER_PROTOCOLS), None)


def _export_requested() -> bool:
    """Whether the env asks for Dome spans over OTLP; the one place to add an opt-in gate."""
    if not otlp_endpoint_configured():
        return False
    if os.environ.get("OTEL_SDK_DISABLED", "").strip().lower() == "true":
        return False
    return _otlp_exporter_name() is not None


def _protocol() -> str:
    for name in _PROTOCOL_VARS:
        value = os.environ.get(name, "").strip().lower()
        if value:
            return value
    return _OTLP_EXPORTER_PROTOCOLS[_otlp_exporter_name() or "otlp"]


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
                "OTLP protocol is grpc but the gRPC exporter is not installed, %s; "
                "install opentelemetry-exporter-otlp-proto-grpc",
                _NOT_EXPORTED,
            )
            return None
        return GrpcSpanExporter
    _warn_once(
        f"protocol:{protocol}",
        "OTLP protocol %r is not supported for Dome spans (use http/protobuf or grpc), %s",
        protocol,
        _NOT_EXPORTED,
    )
    return None


def _warn_missing_extra() -> None:
    _warn_once(
        "extra",
        "OTEL_EXPORTER_OTLP_* is set but Dome cannot export spans without the "
        "opentelemetry extra: %s",
        _EXTRA_HINT,
    )


def _shared_tracer() -> Any:
    """A tracer on the process-wide Dome provider, building it on first use; None if unavailable."""
    global _provider
    missing_extra = False
    build_error: Optional[Exception] = None
    with _lock:
        if _provider is None:
            try:
                from opentelemetry.sdk.resources import Resource
                from opentelemetry.sdk.trace import TracerProvider
                from opentelemetry.sdk.trace.export import BatchSpanProcessor

                exporter_cls = _exporter_class()
            except ImportError:
                exporter_cls, missing_extra = None, True
            if exporter_cls is not None:
                try:
                    # No arguments: the SDK reads endpoint, headers, timeout,
                    # compression and certificates from OTEL_EXPORTER_OTLP_*.
                    processor = BatchSpanProcessor(exporter_cls())
                    new_provider = TracerProvider(resource=Resource.create())
                    new_provider.add_span_processor(processor)
                    _provider = new_provider
                except Exception as e:  # a malformed OTEL_* value must not break Dome()
                    build_error = e
        provider = _provider
    if provider is not None:
        return provider.get_tracer("vijil.dome")
    if missing_extra:
        _warn_missing_extra()
    elif build_error is not None:
        _warn_once(
            "build",
            "Could not build Dome's OTLP span exporter from the OTEL_* env vars (%s: %s), %s",
            type(build_error).__name__,
            build_error,
            _NOT_EXPORTED,
        )
    return None


def _guardrail_objects(guardrail: Any) -> Iterator[Any]:
    yield guardrail
    for guard in guardrail.guard_list:
        yield guard
        yield from guard.detector_list


def _snapshot(guardrail: Any) -> Snapshot:
    return [
        (obj, {name: getattr(obj, "__dict__", {}).get(name, _ABSENT) for name in _WRAPPED_METHODS})
        for obj in _guardrail_objects(guardrail)
    ]


def _restore(snapshot: Snapshot) -> None:
    for obj, methods in snapshot:
        for name, value in methods.items():
            if value is _ABSENT:
                obj.__dict__.pop(name, None)
            else:
                setattr(obj, name, value)


def _state(guardrail: Any) -> Optional[_GuardrailState]:
    state = getattr(guardrail, _GUARDRAIL_STATE_ATTR, None)
    return state if isinstance(state, _GuardrailState) else None


def is_env_instrumented(dome: Any) -> bool:
    """True when *dome*'s current instrumentation came from the OTLP env vars."""
    return isinstance(getattr(dome, _DOME_GUARDRAILS_ATTR, None), list)


def undo_env_instrumentation(dome: Any) -> None:
    """Release *dome*'s env instrumentation, so an explicit setup can instrument it afresh.

    A guardrail's methods are put back once no Dome relies on its env
    wrapping. A guardrail shared with another such Dome keeps it, and a
    warning says so. A no-op on an instance this module did not instrument.
    """
    guardrails = getattr(dome, _DOME_GUARDRAILS_ATTR, None)
    if not isinstance(guardrails, list):
        return
    for guardrail in guardrails:
        state = _state(guardrail)
        if state is None:
            continue
        state.owners.discard(dome)
        if len(state.owners) == 0:
            _restore(state.snapshot)
            setattr(guardrail, _GUARDRAIL_STATE_ATTR, None)
        else:
            _warn_once(
                "shared",
                "Guardrails shared with another Dome instance (built from one DomeConfig) "
                "keep exporting spans to OTEL_EXPORTER_OTLP_* while that instance uses them",
            )
    setattr(dome, _DOME_GUARDRAILS_ATTR, None)
    dome._instrumented = False


def _instrument(dome: Any) -> None:
    tracer = _shared_tracer()
    if tracer is None:
        return
    try:
        from vijil_dome.integrations.instrumentation.otel_instrumentation import (
            _instrument_guardrails,
        )
    except ImportError:
        _warn_missing_extra()
        return

    slots = (dome.input_guardrail, dome.output_guardrail)
    fresh = [g if g is not None and _state(g) is None else None for g in slots]
    if fresh[1] is fresh[0]:
        fresh[1] = None  # one object in both slots is wrapped once
    snapshots = {id(g): _snapshot(g) for g in fresh if g is not None}
    try:
        _instrument_guardrails(
            SimpleNamespace(input_guardrail=fresh[0], output_guardrail=fresh[1]),
            None,
            tracer,
            None,
            dome.enforce,
        )
    except Exception:
        # Leave no half-wrapped guardrail behind.
        for snapshot in snapshots.values():
            _restore(snapshot)
        raise

    owned = []
    for guardrail in slots:
        if guardrail is None:
            continue
        state = _state(guardrail)
        if state is None:
            state = _GuardrailState(snapshots[id(guardrail)])
            setattr(guardrail, _GUARDRAIL_STATE_ATTR, state)
        state.owners.add(dome)
        owned.append(guardrail)
    setattr(dome, _DOME_GUARDRAILS_ATTR, owned)
    dome._instrumented = True


def instrument_from_env(dome: Any) -> None:
    """Wrap *dome*'s guards with spans exported over OTLP, if the OTLP env vars ask for it.

    Never raises: any failure logs one warning and leaves *dome* uninstrumented.
    """
    try:
        if not _export_requested() or getattr(dome, "_instrumented", False):
            return
        _instrument(dome)
    except Exception as e:
        _warn_once(
            "instrument",
            "Could not instrument Dome for OTLP export (%s: %s), %s",
            type(e).__name__,
            e,
            _NOT_EXPORTED,
        )
