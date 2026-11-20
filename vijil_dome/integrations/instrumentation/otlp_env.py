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

"""Export a Dome's guard spans over OTLP when the app opts in and the OTLP env vars are set.

Export needs both ``VIJIL_DOME_OTLP_EXPORT`` set to ``1``, ``true`` or
``yes`` and a standard ``OTEL_EXPORTER_OTLP_[TRACES_]ENDPOINT``. Dome's
spans carry prompt and response text, so an endpoint configured for the
app's own telemetry is not enough: sending that text is an explicit choice.
With the endpoint set but no opt-in, one INFO line per process says so.

This module imports no OpenTelemetry code at import time: ``Dome.__init__``
imports it unconditionally, and an install without the ``opentelemetry``
extra must keep working. OpenTelemetry is imported only once the env asks
for export.

Every ``Dome`` in the process shares one ``TracerProvider``, built on first
use. The SDK registers an atexit hook that flushes and shuts it down. The
app's global tracer provider is never touched. Opting in does install the
OpenTelemetry logging, asyncio and threading instrumentors process-wide, as
``instrument_dome`` does: log records gain trace-context fields, and spans
propagate across asyncio tasks and threads.

The "env layer" is the set of wrappers this module installs on a
guardrail's, its guards' and its detectors' ``scan``/``async_scan``/
``detect``/``sync_detect`` methods. It is tracked per guardrail object,
because Domes built from one ``DomeConfig`` share their guardrails. The
contract:

C1. The env layer is installed at most once per guardrail object, however
    many Domes share it and however many threads construct them at once.
C2. Undo for a Dome removes only the env layer, and only once no other Dome
    relies on it. It never removes a wrapper it did not install: each
    method is put back only if it is still the exact wrapper installed.
C3. If another wrapper has been layered over the env layer on a method, that
    part of the env layer stays in place, and one warning says so. Removing
    it would strip the other wrapper too, silently losing someone's export.
C4. An explicit ``instrument_dome`` or ``setup_vijil_telemetry`` with a
    tracer, on an unshared Dome, removes the env layer: only the explicit
    tracer exports.
C5. ``instrument_dome`` with ``tracer=None`` (metrics or logs only) keeps
    the env trace export.
C6. Telemetry never raises from ``Dome()``: any failure logs one warning and
    leaves the instance uninstrumented, with nothing half-wrapped. Log
    handlers never run while this module's lock is held: messages decided
    under the lock are logged after it is released.
"""

from __future__ import annotations

import logging
import os
import threading
import weakref
from types import SimpleNamespace
from typing import Any, Iterator, Optional

logger = logging.getLogger("vijil.dome")

_OPT_IN_VAR = "VIJIL_DOME_OTLP_EXPORT"
_TRUTHY = frozenset({"1", "true", "yes"})
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
_GUARDRAIL_STATE_ATTR = "_otlp_env_state"  # on a Guardrail carrying the env layer
_DOME_GUARDRAILS_ATTR = "_otlp_env_guardrails"  # on a Dome relying on the env layer
_EXPLICIT_ATTR = "_explicitly_instrumented"  # on a Dome instrument_dome has instrumented
_ABSENT = object()

_EXTRA_HINT = "pip install 'vijil-dome[opentelemetry]'"
_NOT_EXPORTED = "so Dome guard spans are not exported"

# Guards _provider, _warned and every env-layer install and undo (C1, C2).
# Re-entrant because _instrument holds it while building the provider.
_lock = threading.RLock()
_provider: Any = None
_warned: set[str] = set()


class _Notice:
    """A message decided under the lock, logged once per process after the lock is released (C6)."""

    def __init__(self, level: int, key: str, message: str, *args: Any) -> None:
        self.level, self.key, self.message, self.args = level, key, message, args


def _emit(notices: list[_Notice]) -> None:
    """Log each notice not yet logged in this process. Must be called without the lock held."""
    for notice in notices:
        with _lock:
            if notice.key in _warned:
                continue
            _warned.add(notice.key)
        logger.log(notice.level, notice.message, *notice.args)


# One installed wrapper: (object, method name, value before the env layer, wrapper installed).
_Layer = tuple[Any, str, Any, Any]


class _GuardrailState:
    """The env layer on one guardrail object, and the Domes relying on it."""

    def __init__(self, layers: list[_Layer]) -> None:
        self.layers = layers
        self.owners: weakref.WeakSet[Any] = weakref.WeakSet()


def otlp_endpoint_configured() -> bool:
    """True when either OTLP endpoint variable is set to a non-blank value."""
    return any(os.environ.get(name, "").strip() for name in _ENDPOINT_VARS)


def _opted_in() -> bool:
    return os.environ.get(_OPT_IN_VAR, "").strip().lower() in _TRUTHY


def _otlp_exporter_name() -> Optional[str]:
    """The first OTLP name in OTEL_TRACES_EXPORTER (default ``otlp``), or None if it lists none."""
    value = os.environ.get("OTEL_TRACES_EXPORTER", "").strip()
    names = [n.strip().lower() for n in value.split(",")] if value else ["otlp"]
    return next((n for n in names if n in _OTLP_EXPORTER_PROTOCOLS), None)


def _export_requested(notices: list[_Notice]) -> bool:
    """Whether the env asks for Dome spans over OTLP. Reads env vars only; imports nothing."""
    if not otlp_endpoint_configured():
        return False
    if os.environ.get("OTEL_SDK_DISABLED", "").strip().lower() == "true":
        return False
    if _otlp_exporter_name() is None:
        return False
    if not _opted_in():
        notices.append(
            _Notice(
                logging.INFO,
                "opt-in",
                "An OTLP endpoint is set, but Dome does not export its guard spans, which "
                "include prompt and response text, unless %s=1 is also set",
                _OPT_IN_VAR,
            )
        )
        return False
    return True


def _protocol() -> str:
    for name in _PROTOCOL_VARS:
        value = os.environ.get(name, "").strip().lower()
        if value:
            return value
    return _OTLP_EXPORTER_PROTOCOLS[_otlp_exporter_name() or "otlp"]


def _missing_extra() -> _Notice:
    return _Notice(
        logging.WARNING,
        "extra",
        "OTEL_EXPORTER_OTLP_* is set but Dome cannot export spans without the "
        "opentelemetry extra: %s",
        _EXTRA_HINT,
    )


def _exporter_class(notices: list[_Notice]) -> Optional[type]:
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
            notices.append(
                _Notice(
                    logging.WARNING,
                    "grpc",
                    "OTLP protocol is grpc but the gRPC exporter is not installed, %s; "
                    "install opentelemetry-exporter-otlp-proto-grpc",
                    _NOT_EXPORTED,
                )
            )
            return None
        return GrpcSpanExporter
    notices.append(
        _Notice(
            logging.WARNING,
            f"protocol:{protocol}",
            "OTLP protocol %r is not supported for Dome spans (use http/protobuf or grpc), %s",
            protocol,
            _NOT_EXPORTED,
        )
    )
    return None


def _shared_tracer(notices: list[_Notice]) -> Any:
    """A tracer on the process-wide Dome provider, building it on first use; None if unavailable."""
    global _provider
    with _lock:
        if _provider is None:
            try:
                from opentelemetry.sdk.resources import Resource
                from opentelemetry.sdk.trace import TracerProvider
                from opentelemetry.sdk.trace.export import BatchSpanProcessor

                exporter_cls = _exporter_class(notices)
            except ImportError:
                notices.append(_missing_extra())
                return None
            if exporter_cls is None:
                return None
            try:
                # No arguments: the SDK reads endpoint, headers, timeout,
                # compression and certificates from OTEL_EXPORTER_OTLP_*.
                processor = BatchSpanProcessor(exporter_cls())
                provider = TracerProvider(resource=Resource.create())
                provider.add_span_processor(processor)
            except Exception as e:  # a malformed OTEL_* value must not break Dome()
                notices.append(
                    _Notice(
                        logging.WARNING,
                        "build",
                        "Could not build Dome's OTLP span exporter from the OTEL_* env vars "
                        "(%s: %s), %s",
                        type(e).__name__,
                        e,
                        _NOT_EXPORTED,
                    )
                )
                return None
            _provider = provider
        return _provider.get_tracer("vijil.dome")


def _guardrail_objects(guardrail: Any) -> Iterator[Any]:
    yield guardrail
    for guard in guardrail.guard_list:
        yield guard
        yield from guard.detector_list


def _method_values(guardrail: Any) -> list[tuple[Any, str, Any]]:
    """(object, method name, instance-level value or _ABSENT) for every wrappable method."""
    return [
        (obj, name, getattr(obj, "__dict__", {}).get(name, _ABSENT))
        for obj in _guardrail_objects(guardrail)
        for name in _WRAPPED_METHODS
    ]


def _put_back(obj: Any, name: str, before: Any) -> None:
    if before is _ABSENT:
        obj.__dict__.pop(name, None)
    else:
        setattr(obj, name, before)


def _state(guardrail: Any) -> Optional[_GuardrailState]:
    state = getattr(guardrail, _GUARDRAIL_STATE_ATTR, None)
    return state if isinstance(state, _GuardrailState) else None


def is_env_instrumented(dome: Any) -> bool:
    """True when *dome* relies on the env layer."""
    return isinstance(getattr(dome, _DOME_GUARDRAILS_ATTR, None), list)


def env_only_instrumented(dome: Any) -> bool:
    """True when *dome*'s only instrumentation is the env layer (no explicit instrument_dome yet)."""
    return is_env_instrumented(dome) and getattr(dome, _EXPLICIT_ATTR, False) is not True


def mark_explicitly_instrumented(dome: Any) -> None:
    setattr(dome, _EXPLICIT_ATTR, True)


def undo_env_instrumentation(dome: Any) -> None:
    """Release *dome*'s reliance on the env layer, removing it where the contract allows (C2, C3).

    A no-op on an instance that does not rely on the env layer.
    """
    notices: list[_Notice] = []
    with _lock:
        guardrails = getattr(dome, _DOME_GUARDRAILS_ATTR, None)
        if not isinstance(guardrails, list):
            return
        for guardrail in guardrails:
            state = _state(guardrail)
            if state is None:
                continue
            state.owners.discard(dome)
            if len(state.owners) > 0:
                notices.append(
                    _Notice(
                        logging.WARNING,
                        "shared",
                        "Guardrails shared with another Dome instance (built from one "
                        "DomeConfig) keep exporting spans to OTEL_EXPORTER_OTLP_* while "
                        "that instance uses them",
                    )
                )
                continue
            kept = []
            for obj, name, before, installed in state.layers:
                if getattr(obj, "__dict__", {}).get(name, _ABSENT) is installed:
                    _put_back(obj, name, before)
                else:
                    kept.append((obj, name, before, installed))
            if kept:
                state.layers = kept
                notices.append(
                    _Notice(
                        logging.WARNING,
                        "layered",
                        "Dome's OTLP env export wrappers were left in place under wrappers "
                        "added later, so Dome guard spans are still also exported to "
                        "OTEL_EXPORTER_OTLP_*; removing them would also remove the later ones",
                    )
                )
            else:
                setattr(guardrail, _GUARDRAIL_STATE_ATTR, None)
        setattr(dome, _DOME_GUARDRAILS_ATTR, None)
        dome._instrumented = False
    _emit(notices)


def _instrument(dome: Any, notices: list[_Notice]) -> None:
    with _lock:
        tracer = _shared_tracer(notices)
        if tracer is None:
            return
        try:
            from vijil_dome.integrations.instrumentation.otel_instrumentation import (
                _instrument_guardrails,
            )
        except ImportError:
            notices.append(_missing_extra())
            return

        slots = [g for g in (dome.input_guardrail, dome.output_guardrail) if g is not None]
        if len(slots) == 2 and slots[0] is slots[1]:
            slots = slots[:1]  # one object in both slots carries one env layer
        fresh = [g if _state(g) is None else None for g in (dome.input_guardrail, dome.output_guardrail)]
        if fresh[0] is not None and fresh[1] is fresh[0]:
            fresh[1] = None
        before = {id(g): _method_values(g) for g in fresh if g is not None}
        try:
            _instrument_guardrails(
                SimpleNamespace(input_guardrail=fresh[0], output_guardrail=fresh[1]),
                None,
                tracer,
                None,
                dome.enforce,
            )
        except Exception:
            # Nothing else can have layered over these while the lock was held.
            for values in before.values():
                for obj, name, value in values:
                    _put_back(obj, name, value)
            raise

        for guardrail in slots:
            state = _state(guardrail)
            if state is None:
                layers = [
                    (obj, name, value, obj.__dict__[name])
                    for obj, name, value in before[id(guardrail)]
                    if obj.__dict__.get(name, _ABSENT) is not value
                ]
                state = _GuardrailState(layers)
                setattr(guardrail, _GUARDRAIL_STATE_ATTR, state)
            state.owners.add(dome)
        setattr(dome, _DOME_GUARDRAILS_ATTR, list(slots))
        dome._instrumented = True


def instrument_from_env(dome: Any) -> None:
    """Wrap *dome*'s guards with spans exported over OTLP, if the env asks for it.

    Never raises: any failure logs one warning and leaves *dome* uninstrumented.
    """
    notices: list[_Notice] = []
    try:
        if _export_requested(notices) and not getattr(dome, "_instrumented", False):
            _instrument(dome, notices)
    except Exception as e:
        notices.append(
            _Notice(
                logging.WARNING,
                "instrument",
                "Could not instrument Dome for OTLP export (%s: %s), %s",
                type(e).__name__,
                e,
                _NOT_EXPORTED,
            )
        )
    _emit(notices)
