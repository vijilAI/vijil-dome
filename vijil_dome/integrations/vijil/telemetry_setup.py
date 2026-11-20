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

"""Send an in-process Dome's telemetry to Vijil Console's OTLP ingest."""

from __future__ import annotations

import atexit
import contextvars
import functools
import inspect
import logging
import os
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple, cast

from vijil_dome.utils.api_config_loader import DEFAULT_CONSOLE_API_BASE_URL

logger = logging.getLogger("vijil.dome")

_KEY_CHECK_TIMEOUT_SECONDS = 5.0
_EXPORT_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class _TelemetryConfig:
    base_url: str
    client_id: str | None
    client_secret: str | None = field(repr=False)
    agent_id: str | None
    user_id: str | None
    send_to_console: bool


def _otlp_url(base_url: str, signal: str) -> str:
    return f"{base_url}/v1/otlp/v1/{signal}"


def _resolve_config(
    dome: Any,
    *,
    client_id: str | None,
    client_secret: str | None,
    agent_id: str | None,
    base_url: str | None,
    user_id: str | None,
    send_to_console: bool,
    has_extra_exporters: bool,
) -> _TelemetryConfig:
    """Apply fallbacks and reject configs that would export nothing."""
    # An instance loaded with create_from_api already knows which Console and key
    # it uses; preferring the env would send a staging key to another Console.
    client_id = client_id or getattr(dome, "_api_client_id", None) or os.getenv("VIJIL_CLIENT_ID")
    client_secret = (
        client_secret or getattr(dome, "_api_client_secret", None) or os.getenv("VIJIL_CLIENT_SECRET")
    )
    # Secrets mounted from files often end in a newline, which no HTTP header can carry.
    client_id = client_id.strip() if client_id else client_id
    client_secret = client_secret.strip() if client_secret else client_secret
    base_url = (
        base_url
        or getattr(dome, "_api_base_url", None)
        or os.getenv("VIJIL_CONSOLE_URL")
        or DEFAULT_CONSOLE_API_BASE_URL
    ).rstrip("/")

    # The agent a create_from_api instance was loaded for is the Console UUID; the
    # config's own agent_id (or legacy agent_config_id) need not be.
    instance_agent_id = getattr(dome, "_api_agent_id", None) or getattr(dome, "agent_id", None)
    given_agent_id = agent_id or os.getenv("VIJIL_AGENT_ID")
    # Defend prefers the span-level agent.id the instance stamps, so a different
    # Resource value would quietly file events under the instance's agent.
    if given_agent_id and instance_agent_id and given_agent_id != instance_agent_id:
        raise ValueError(
            f"agent_id {given_agent_id!r} conflicts with the instance's agent_id "
            f"{instance_agent_id!r}"
        )
    agent_id = given_agent_id or instance_agent_id

    if send_to_console:
        missing = [
            name
            for name, value in (
                ("client_id", client_id),
                ("client_secret", client_secret),
                ("agent_id", agent_id),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                f"Sending telemetry to Console needs {', '.join(missing)} "
                "(arguments or VIJIL_CLIENT_ID / VIJIL_CLIENT_SECRET / VIJIL_AGENT_ID)"
            )
    elif not has_extra_exporters:
        raise ValueError(
            "send_to_console=False with no exporters leaves telemetry nowhere to go"
        )

    return _TelemetryConfig(
        base_url=base_url,
        client_id=client_id,
        client_secret=client_secret,
        agent_id=agent_id,
        user_id=user_id,
        send_to_console=send_to_console,
    )


def _check_console_key(base_url: str, client_id: str, client_secret: str) -> None:
    """Fail at setup on a rejected key or wrong URL; tolerate Console being unreachable."""
    # requests, like the OTLP exporters, so the check trusts exactly the CAs the exports will.
    import requests

    url = _otlp_url(base_url, "traces")
    try:
        response = requests.post(
            url,
            data=b"",
            headers={
                "Authorization": f"Bearer {client_id}:{client_secret}",
                "Content-Type": "application/x-protobuf",
            },
            timeout=_KEY_CHECK_TIMEOUT_SECONDS,
            allow_redirects=False,
        )
    except (requests.exceptions.InvalidHeader, UnicodeEncodeError):
        # Both exceptions carry the header, secret included.
        raise ValueError(
            "The Console API key contains characters an HTTP header can't carry"
        ) from None
    except (requests.exceptions.InvalidURL, requests.exceptions.MissingSchema) as e:
        raise ValueError(f"{url} is not a usable Console URL: {e}") from None
    except requests.RequestException as e:
        logger.warning("Couldn't reach Console at %s to check the API key: %s", url, e)
        return

    if response.status_code in (401, 403):
        try:
            body = response.json()
        except ValueError:
            body = None
        detail = (body.get("message") if isinstance(body, dict) else None) or response.text
        raise ValueError(f"Console at {base_url} rejected the telemetry API key: {detail}")
    status = response.status_code
    if status in (408, 429) or status >= 500:
        logger.warning("Console answered the API key check with HTTP %d; continuing", status)
    elif status >= 300:
        # A redirect or 404 means the URL is wrong, and every export would fail the same way.
        raise ValueError(
            f"{url} answered HTTP {status}; check base_url / VIJIL_CONSOLE_URL "
            "(the Console API host, without /v1)"
        )


class _LoggerState(NamedTuple):
    handlers: list[logging.Handler]
    level: int
    propagate: bool


class _DomeLogRouter(logging.Handler):
    """Exports each Dome record through the setup it belongs to, so it carries that agent's Resource."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        # Replaced, never mutated, so emit can read it without the lock.
        self._setups: tuple[VijilTelemetry, ...] = ()
        # Per Dome logger, its state before install and the state install left.
        self._installed: dict[str, tuple[_LoggerState, _LoggerState]] = {}

    def emit(self, record: logging.LogRecord) -> None:
        from opentelemetry import trace

        setups = self._setups
        resource = getattr(trace.get_current_span(), "resource", None)
        owner = next((s for s in setups if s.tracer_provider.resource is resource), None)
        if owner is None:
            # Between spans of a guard call, e.g. the control decision logged after its span.
            owner = _calling_setup.get()
        # Anything else (a shut-down setup's instance, an instance never set up, code
        # outside any guard call) goes unexported: no guess beats the wrong agent.id.
        if owner in setups and owner._log_handler is not None:
            owner._log_handler.handle(record)

    def register(self, telemetry: VijilTelemetry) -> None:
        from vijil_dome.integrations.instrumentation.otel_instrumentation import (
            _DOME_LOGGER_NAMES,
            _export_dome_logs,
        )

        with self._lock:
            self._setups = (*self._setups, telemetry)
            if self._installed:
                return
            before = {name: _state(logging.getLogger(name)) for name in _DOME_LOGGER_NAMES}
            _export_dome_logs(self)
            self._installed = {
                name: (state, _state(logging.getLogger(name))) for name, state in before.items()
            }

    def unregister(self, telemetry: VijilTelemetry) -> None:
        with self._lock:
            self._setups = tuple(s for s in self._setups if s is not telemetry)
            if self._setups:
                return
            # Undo only what install did, and only where the app hasn't changed it since.
            for name, (before, after) in self._installed.items():
                lg = logging.getLogger(name)
                lg.removeHandler(self)
                added = [h for h in after.handlers if h not in before.handlers]
                if any(h not in after.handlers for h in lg.handlers):
                    # Someone attached a handler while we ran; the forwarder and our
                    # level keep its records flowing as they did.
                    continue
                for h in added:
                    lg.removeHandler(h)
                if lg.level == after.level:
                    lg.setLevel(before.level)
                if lg.propagate == after.propagate:
                    lg.propagate = before.propagate
            self._installed = {}


def _state(lg: logging.Logger) -> _LoggerState:
    return _LoggerState(list(lg.handlers), lg.level, lg.propagate)


# Dome's loggers are process-wide, so one router serves every setup.
_log_router = _DomeLogRouter()

_calling_setup: contextvars.ContextVar[VijilTelemetry | None] = contextvars.ContextVar(
    "vijil_dome_calling_setup", default=None
)

_GUARD_METHODS = (
    "guard_input",
    "guard_output",
    "async_guard_input",
    "async_guard_output",
    "guard_tool_call",
    "async_guard_tool_call",
    "guard_input_batch",
    "guard_output_batch",
    "async_guard_input_batch",
    "async_guard_output_batch",
)


def _mark_guard_calls(dome: Any, telemetry: VijilTelemetry) -> None:
    """Make each guard call announce its setup, for records logged outside any span."""
    for name in _GUARD_METHODS:
        method = getattr(dome, name, None)
        if method is None:
            continue
        if inspect.iscoroutinefunction(method):

            @functools.wraps(method)
            async def marked_async(*args: Any, _method: Any = method, **kwargs: Any) -> Any:
                token = _calling_setup.set(telemetry)
                try:
                    return await _method(*args, **kwargs)
                finally:
                    _calling_setup.reset(token)

            setattr(dome, name, marked_async)
        else:

            @functools.wraps(method)
            def marked(*args: Any, _method: Any = method, **kwargs: Any) -> Any:
                token = _calling_setup.set(telemetry)
                try:
                    return _method(*args, **kwargs)
                finally:
                    _calling_setup.reset(token)

            setattr(dome, name, marked)


# What an OTLP export needs; anything else on the request came from the app's
# OTEL_EXPORTER_OTLP_*HEADERS, which newer SDKs merge in, and is meant for its collector.
_CONSOLE_HEADERS = frozenset(
    {
        "authorization",
        "content-type",
        "content-length",
        "user-agent",
        "accept",
        "accept-encoding",
        "connection",
    }
)


def _console_session(requests: Any) -> Any:
    """A requests session that carries only Vijil's credentials to Console."""

    class ConsoleSession(requests.Session):  # type: ignore[name-defined, misc]
        def send(self, request: Any, **kwargs: Any) -> Any:
            # The app's OTEL_EXPORTER_OTLP_*CLIENT_* certificate is for its own collector,
            # and no exporter argument turns it off.
            kwargs["cert"] = None
            for name in [n for n in request.headers if n.lower() not in _CONSOLE_HEADERS]:
                del request.headers[name]
            return super().send(request, **kwargs)

    return ConsoleSession()


_MISSING_EXTRA = (
    "setup_vijil_telemetry requires the opentelemetry extra: "
    "pip install 'vijil-dome[opentelemetry]'"
)


class VijilTelemetry:
    """The private providers carrying one Dome instance's telemetry."""

    def __init__(self, tracer_provider: Any, meter_provider: Any, logger_provider: Any) -> None:
        self.tracer_provider = tracer_provider
        self.meter_provider = meter_provider
        self.logger_provider = logger_provider
        self._log_handler: logging.Handler | None = None
        self._shut_down = False

    def force_flush(self) -> None:
        self.tracer_provider.force_flush()
        self.meter_provider.force_flush()
        self.logger_provider.force_flush()

    def shutdown(self) -> None:
        # Also registered with atexit, so an explicit shutdown runs it twice.
        if self._shut_down:
            return
        self._shut_down = True
        _log_router.unregister(self)
        self.tracer_provider.shutdown()
        self.meter_provider.shutdown()
        self.logger_provider.shutdown()


def setup_vijil_telemetry(
    dome: Any,
    *,
    client_id: str | None = None,
    client_secret: str | None = None,
    agent_id: str | None = None,
    base_url: str | None = None,
    user_id: str | None = None,
    send_to_console: bool = True,
    span_exporters: Sequence[Any] = (),
    metric_exporters: Sequence[Any] = (),
    log_exporters: Sequence[Any] = (),
) -> VijilTelemetry:
    """Send a Dome or VijilDome's spans, metrics and logs to Console and/or the given exporters."""
    try:
        import requests
        from opentelemetry.exporter.otlp.proto.http import Compression
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.metrics import (
            Counter,
            Histogram,
            MeterProvider,
            ObservableCounter,
            ObservableGauge,
            ObservableUpDownCounter,
            TraceBasedExemplarFilter,
            UpDownCounter,
        )
        from opentelemetry.sdk.metrics.export import (
            AggregationTemporality,
            PeriodicExportingMetricReader,
        )
        from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import SpanLimits, Tracer, TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import ALWAYS_ON

        from vijil_dome.integrations.instrumentation.otel_instrumentation import (
            instrument_dome,
            instrument_vijil_dome,
        )
    except ImportError as e:
        raise ImportError(_MISSING_EXTRA) from e

    if os.getenv("OTEL_SDK_DISABLED", "").strip().lower() == "true":
        logger.warning("OTEL_SDK_DISABLED is set, so setup_vijil_telemetry will export nothing")

    if getattr(dome, "_instrumented", False):
        raise RuntimeError(
            "This Dome instance is already instrumented; a second setup would be ignored"
        )

    config = _resolve_config(
        dome,
        client_id=client_id,
        client_secret=client_secret,
        agent_id=agent_id,
        base_url=base_url,
        user_id=user_id,
        send_to_console=send_to_console,
        has_extra_exporters=bool(span_exporters or metric_exporters or log_exporters),
    )

    span_exporters, metric_exporters, log_exporters = (
        list(span_exporters),
        list(metric_exporters),
        list(log_exporters),
    )
    if config.send_to_console:
        assert config.client_id and config.client_secret
        _check_console_key(config.base_url, config.client_id, config.client_secret)
        headers = {"Authorization": f"Bearer {config.client_id}:{config.client_secret}"}
        # Every argument the exporters would otherwise read from the app's
        # OTEL_EXPORTER_OTLP_* env (its own collector's CA, timeout, compression,
        # credential session, metric temporality) is pinned, and the session drops
        # its client cert and headers. certificate_file=True keeps requests' usual
        # trust: certifi, or REQUESTS_CA_BUNDLE behind a TLS proxy.
        pinned: dict[str, Any] = dict(
            headers=headers,
            certificate_file=True,
            timeout=_EXPORT_TIMEOUT_SECONDS,
            compression=Compression.NoCompression,
        )
        console_exporters = (
            OTLPSpanExporter(
                endpoint=_otlp_url(config.base_url, "traces"), session=_console_session(requests), **pinned
            ),
            OTLPMetricExporter(
                endpoint=_otlp_url(config.base_url, "metrics"),
                session=_console_session(requests),
                preferred_temporality={
                    kind: AggregationTemporality.CUMULATIVE
                    for kind in (
                        Counter,
                        UpDownCounter,
                        Histogram,
                        ObservableCounter,
                        ObservableUpDownCounter,
                        ObservableGauge,
                    )
                },
                preferred_aggregation={Histogram: ExplicitBucketHistogramAggregation()},
                **pinned,
            ),
            OTLPLogExporter(
                endpoint=_otlp_url(config.base_url, "logs"), session=_console_session(requests), **pinned
            ),
        )
        for exporters, exporter in zip(
            (span_exporters, metric_exporters, log_exporters), console_exporters
        ):
            exporters.insert(0, exporter)

    attributes = {"service.name": "vijil.dome"}
    if config.agent_id:
        attributes["agent.id"] = config.agent_id
    if config.user_id:
        attributes["user.id"] = config.user_id
    # Not Resource.create: that merges the app's OTEL_RESOURCE_ATTRIBUTES into what Console receives.
    resource = Resource(attributes)

    # The pipeline below is pinned to SDK defaults rather than the app's OTEL_* env.
    # The exception is log record limits: this SDK reads them from env per record
    # with no provider argument to override. VijilTelemetry.shutdown owns exit-time
    # shutdown, so the providers don't register their own.
    tracer_provider = TracerProvider(
        resource=resource,
        # Not parent-based: an app sampling its own traces must not thin out Defend's events.
        sampler=ALWAYS_ON,
        span_limits=SpanLimits(
            max_attributes=128,
            max_events=128,
            max_links=128,
            max_span_attributes=128,
            max_event_attributes=128,
            max_link_attributes=128,
            max_attribute_length=SpanLimits.UNSET,
            max_span_attribute_length=SpanLimits.UNSET,
        ),
        shutdown_on_exit=False,
    )
    batching = dict(
        max_queue_size=2048,
        schedule_delay_millis=5000,
        max_export_batch_size=512,
        export_timeout_millis=30000,
    )
    for span_exporter in span_exporters:
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter, **batching))
    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[
            PeriodicExportingMetricReader(e, export_interval_millis=60000, export_timeout_millis=30000)
            for e in metric_exporters
        ],
        exemplar_filter=TraceBasedExemplarFilter(),
        shutdown_on_exit=False,
    )
    logger_provider = LoggerProvider(resource=resource, shutdown_on_exit=False)
    for log_exporter in log_exporters:
        logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter, **batching))

    tracer = cast(Tracer, tracer_provider.get_tracer("vijil.dome"))
    meter = meter_provider.get_meter("vijil.dome")
    from vijil_dome.core import VijilDome

    # No handler here: logs go through _log_router, which attributes them per instance.
    if isinstance(dome, VijilDome):
        instrument_vijil_dome(dome, tracer, meter=meter)
    else:
        instrument_dome(dome, None, tracer, meter)

    telemetry = VijilTelemetry(tracer_provider, meter_provider, logger_provider)
    if log_exporters:
        # No formatter: the body is then the rendered message, with trace and span ids
        # in their own OTLP fields.
        telemetry._log_handler = LoggingHandler(logger_provider=logger_provider)
        _log_router.register(telemetry)
        _mark_guard_calls(dome, telemetry)
    atexit.register(telemetry.shutdown)
    return telemetry
