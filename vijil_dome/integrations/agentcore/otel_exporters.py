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

"""Initialize OTLP/HTTP trace, metric, and log exporters (env + :class:`~.settings.AgentCoreRuntimeSettings`)."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from vijil_dome.Dome import Dome

    from .settings import AgentCoreRuntimeSettings

logger = logging.getLogger("vijil.dome")

_INSTRUMENTATION_NAME = "vijil.dome"


def _agentcore_instrumentation_version() -> str:
    try:
        from importlib.metadata import version

        return version("vijil-dome")
    except Exception:
        return "0.0.0"


def _apply_dome_otel_instrumentation(
    dome: "Dome",
    *,
    bridge_stdlib_logging: bool,
    bridge_logger_name: str,
) -> None:
    """Wire :func:`~vijil_dome.integrations.instrumentation.otel_instrumentation.instrument_dome` to global providers."""
    from opentelemetry import metrics, trace

    from vijil_dome.integrations.instrumentation.otel_instrumentation import (
        instrument_dome,
        instrument_logger,
    )

    ver = _agentcore_instrumentation_version()
    tracer = trace.get_tracer(_INSTRUMENTATION_NAME, ver)
    meter = metrics.get_meter(_INSTRUMENTATION_NAME, ver)
    # LoggingHandler is already attached by :func:`setup_agentcore_otel_exporters_from_env`;
    # avoid registering it twice via ``instrument_dome(..., handler=...)``.
    instrument_dome(dome, None, tracer, meter)
    if bridge_stdlib_logging:
        instrument_logger(logging.getLogger(bridge_logger_name))


def _env_nonempty(name: str) -> bool:
    return bool(os.environ.get(name, "").strip())


def _has_any_otlp_endpoint_env() -> bool:
    """True if env supplies per-signal OTLP/HTTP endpoints (not the host base ``OTEL_EXPORTER_OTLP_ENDPOINT``)."""
    keys = (
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
        "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
    )
    return any(_env_nonempty(k) for k in keys)


def _append_otlp_signal_path(base_endpoint: str, signal_path: str) -> str:
    """Append ``v1/traces``-style path the same way OTLP HTTP exporters do."""
    base = base_endpoint.strip()
    if base.endswith("/"):
        return f"{base}{signal_path}"
    return f"{base}/{signal_path}"


def _has_otlp_endpoint_config(settings: "AgentCoreRuntimeSettings") -> bool:
    if (settings.otel_exporter_otlp_endpoint or "").strip():
        return True
    return _has_any_otlp_endpoint_env()


def _resolved_trace_otlp_endpoint(settings: "AgentCoreRuntimeSettings") -> Optional[str]:
    base = (settings.otel_exporter_otlp_endpoint or "").strip()
    if base:
        return _append_otlp_signal_path(base, "v1/traces")
    return os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "").strip() or None


def _resolved_metrics_otlp_endpoint(settings: "AgentCoreRuntimeSettings") -> Optional[str]:
    base = (settings.otel_exporter_otlp_endpoint or "").strip()
    if base:
        return _append_otlp_signal_path(base, "v1/metrics")
    return os.environ.get("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "").strip() or None


def _resolved_logs_otlp_endpoint(settings: "AgentCoreRuntimeSettings") -> Optional[str]:
    base = (settings.otel_exporter_otlp_endpoint or "").strip()
    if base:
        return _append_otlp_signal_path(base, "v1/logs")
    return os.environ.get("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", "").strip() or None


def _global_otel_sdk_providers_already_set() -> bool:
    """True if the process already has SDK tracer or meter providers (not OTel proxies)."""
    try:
        from opentelemetry import metrics, trace
        from opentelemetry.sdk.metrics import MeterProvider as SDKMeterProvider
        from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
    except ImportError:
        return False

    return isinstance(trace.get_tracer_provider(), SDKTracerProvider) or isinstance(
        metrics.get_meter_provider(), SDKMeterProvider
    )


def create_agentcore_otel_resource(
    team_id: str = "",
    agent_id: Optional[str] = None,
):
    """Build a stable :class:`~opentelemetry.sdk.resources.Resource` for AgentCore OTLP export.

    Uses ``team.id`` and optional ``agent.id`` so DELTA counter series stay aligned across
    short-lived sessions for the same team/agent. Omits ``service.instance.id`` and
    ``service.ip`` so labels stay shared and backends can sum deltas into cumulative views.

    ``deployment.environment`` defaults to ``production`` or ``DEPLOYMENT_ENVIRONMENT``.
    ``service.version`` comes from the installed ``vijil-dome`` package metadata (same
    helper as tracer/meter instrumentation); falls back to ``0.0.0`` if unavailable.

    Requires the ``opentelemetry`` optional dependency (``Resource`` from the SDK).
    """
    from opentelemetry.sdk.resources import Resource

    attributes = {
        "service.name": "vijil.dome",
        "service.namespace": team_id,
        "service.version": _agentcore_instrumentation_version(),
        "deployment.environment": os.getenv("DEPLOYMENT_ENVIRONMENT", "production"),
        "team.id": team_id,
    }
    if agent_id:
        attributes["agent.id"] = agent_id
    return Resource(attributes=attributes)


class AgentCoreOtelExporterHandle:
    """Holds references created by :func:`setup_agentcore_otel_exporters_from_env`.

    Call :meth:`shutdown` during agent teardown (after stopping other background
    threads) so batch processors and the metrics export interval flush cleanly.

    For exporter setup **plus** Dome guardrail wiring in one step, use
    :func:`setup_agentcore_otel_for_dome`, which returns a subclass with
    :meth:`AgentCoreDomeOtelExporterHandle.reinstrument_dome`.
    """

    def __init__(
        self,
        *,
        bridge_handler: Optional[logging.Handler] = None,
        bridge_logger_name: Optional[str] = None,
        manages_global_providers: bool = True,
    ) -> None:
        self._bridge_handler = bridge_handler
        self._bridge_logger_name = bridge_logger_name
        self._manages_global_providers = manages_global_providers

    @property
    def manages_global_providers(self) -> bool:
        """False when exporters were skipped because another component owns global providers."""
        return self._manages_global_providers

    @property
    def bridge_handler(self) -> Optional[logging.Handler]:
        return self._bridge_handler

    @property
    def bridge_logger_name(self) -> Optional[str]:
        return self._bridge_logger_name

    def shutdown(self) -> None:
        """Remove optional stdlib bridge handler, then shut down global OTel providers."""
        if self._bridge_handler is not None and self._bridge_logger_name:
            lg = logging.getLogger(self._bridge_logger_name)
            if self._bridge_handler in lg.handlers:
                lg.removeHandler(self._bridge_handler)
            try:
                self._bridge_handler.close()
            except Exception:
                logger.exception("Error closing OTel logging bridge handler")
            self._bridge_handler = None
            self._bridge_logger_name = None

        from .otel_lifecycle import shutdown_opentelemetry_providers

        if self._manages_global_providers:
            shutdown_opentelemetry_providers()


class AgentCoreDomeOtelExporterHandle(AgentCoreOtelExporterHandle):
    """Return type of :func:`setup_agentcore_otel_for_dome` — OTLP handle plus Dome hooks."""

    def __init__(
        self,
        *,
        bridge_handler: Optional[logging.Handler] = None,
        bridge_logger_name: Optional[str] = None,
        attach_stdlib_log_formatting: bool = True,
        use_bridge_logger_name: str = "vijil.dome",
        manages_global_providers: bool = True,
    ) -> None:
        super().__init__(
            bridge_handler=bridge_handler,
            bridge_logger_name=bridge_logger_name,
            manages_global_providers=manages_global_providers,
        )
        self._attach_stdlib_log_formatting = attach_stdlib_log_formatting
        self._use_bridge_logger_name = use_bridge_logger_name

    def reinstrument_dome(self, dome: "Dome") -> None:
        """Re-attach traces, metrics, and Darwin spans after guardrails were replaced.

        Call after :meth:`~vijil_dome.Dome.Dome.apply_config_dict` or any S3 reload that
        swaps ``input_guardrail`` / ``output_guardrail``. Used automatically by
        :func:`start_agentcore_background_services` when this handle is passed as
        *otel_exporter_handle* and the S3 config poller is running.
        """
        _apply_dome_otel_instrumentation(
            dome,
            bridge_stdlib_logging=self._attach_stdlib_log_formatting,
            bridge_logger_name=self._use_bridge_logger_name,
        )


def setup_agentcore_otel_exporters_from_env(
    *,
    enabled: Optional[bool] = None,
    settings: Optional["AgentCoreRuntimeSettings"] = None,
    team_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    bridge_stdlib_logging: bool = True,
    bridge_logger_name: str = "vijil.dome",
) -> Optional[AgentCoreOtelExporterHandle]:
    """Configure global OTLP-over-HTTP exporters for **traces**, **metrics**, and **logs**.

    This is **opt-in** when *enabled* is ``None``: exporters install only if an OTLP HTTP
    endpoint is configured (non-empty ``DOME_OTEL_EXPORTER_OTLP_ENDPOINT`` on *settings*,
    or per-signal ``OTEL_EXPORTER_OTLP_*_ENDPOINT`` in the environment). Pass ``enabled=False``
    to skip, or ``enabled=True`` to force an attempt (logs a warning and returns ``None``
    if no endpoint is found).

    The host-wide ``OTEL_EXPORTER_OTLP_ENDPOINT`` is intentionally **not** read here so it
    does not collide with other OpenTelemetry setup in the same process.

    Other common variables are read **automatically** by the OpenTelemetry Python
    exporters when constructed with default arguments, including:

    - ``OTEL_EXPORTER_OTLP_HEADERS`` / per-signal ``OTEL_EXPORTER_OTLP_*_HEADERS``
    - ``OTEL_EXPORTER_OTLP_TIMEOUT`` / per-signal timeouts

    If the process already has an SDK :class:`~opentelemetry.sdk.trace.TracerProvider`
    or :class:`~opentelemetry.sdk.metrics.MeterProvider` (for example from host
    ``setup_telemetry``), this function **does not** call ``set_tracer_provider`` /
    ``set_meter_provider`` again (which would only log warnings and be ignored). It
    returns a handle with ``manages_global_providers`` false; use
    :func:`setup_agentcore_otel_for_dome` so Dome still attaches to the existing providers.

    Resource attributes use :func:`create_agentcore_otel_resource` (stable ``service.name``,
    ``team.id``, optional ``agent.id``) so DELTA counters from many sessions map to one
    logical series. Resolve ``team_id`` / ``agent_id`` from explicit arguments when
    provided; otherwise from *settings* (or :func:`~.settings.load_agentcore_runtime_settings_from_env`
    when *settings* is omitted), which reads ``TEAM_ID`` / ``AGENT_ID``. Set
    ``DEPLOYMENT_ENVIRONMENT`` for ``deployment.environment`` (default ``production``).

    Metrics: **Counter**, **UpDownCounter**, **ObservableCounter**, and
    **ObservableUpDownCounter** use **delta** temporality on the OTLP/HTTP metric
    exporter (short-lived sessions export per-interval increments; backends can sum
    without a cumulative-to-delta processor). **Histogram** and **ObservableGauge**
    are omitted so they keep exporter/SDK defaults (cumulative-style behavior for
    histogram buckets / percentiles).

    Requires optional dependency group ``opentelemetry`` (and HTTP OTLP exporter
    packages) to be installed.

    Args:
        enabled: If ``None``, install only when ``DOME_OTEL_EXPORTER_OTLP_ENDPOINT`` is set on
            *settings* (after env load if *settings* is omitted) or per-signal
            ``OTEL_EXPORTER_OTLP_*_ENDPOINT`` env vars are set. If ``False``, return ``None``.
            If ``True``, attempt setup and warn if no endpoint is found.
        settings: Optional :class:`~.settings.AgentCoreRuntimeSettings`; when omitted, env
            is loaded once for team/agent ids and ``otel_exporter_otlp_endpoint``
            (``DOME_OTEL_EXPORTER_OTLP_ENDPOINT``).
        team_id: Overrides ``team.id`` / ``service.namespace`` when not ``None``.
        agent_id: Overrides ``agent.id`` when not ``None``.
        bridge_stdlib_logging: If True, attach an OpenTelemetry ``LoggingHandler`` to
            *bridge_logger_name* so stdlib logs are emitted as OTel log records.
        bridge_logger_name: Logger name for the bridge (default ``vijil.dome``).

    Returns:
        :class:`AgentCoreOtelExporterHandle` to shut down later, or ``None`` if
        exporters were not installed. If another component already set global SDK
        providers, returns a handle with ``manages_global_providers`` false so
        :meth:`~AgentCoreOtelExporterHandle.shutdown` does not replace them.
    """
    from .settings import load_agentcore_runtime_settings_from_env

    resolved_settings = settings or load_agentcore_runtime_settings_from_env()
    if enabled is None:
        enabled = _has_otlp_endpoint_config(resolved_settings)
    if not enabled:
        return None

    if _global_otel_sdk_providers_already_set():
        logger.info(
            "AgentCore OTLP exporter setup skipped: an SDK TracerProvider or MeterProvider "
            "is already configured (e.g. host setup_telemetry). Using existing global "
            "providers for Dome; handle.shutdown() will not shut down those providers."
        )
        return AgentCoreOtelExporterHandle(
            bridge_handler=None,
            bridge_logger_name=None,
            manages_global_providers=False,
        )

    try:
        from opentelemetry import metrics, trace
        from opentelemetry._logs import set_logger_provider
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.logging import LoggingInstrumentor
        from opentelemetry.sdk import metrics as sdkmetrics
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import (
            AggregationTemporality,
            PeriodicExportingMetricReader,
        )
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as e:
        logger.warning(
            "AgentCore OTel exporters skipped (install optional 'opentelemetry' extras): %s",
            e,
        )
        return None

    if not _has_otlp_endpoint_config(resolved_settings):
        logger.warning(
            "AgentCore OTLP exporter setup was requested but no Dome OTLP HTTP endpoint was "
            "found. Set ``DOME_OTEL_EXPORTER_OTLP_ENDPOINT`` (base URL; ``v1/traces``, "
            "``v1/metrics``, and ``v1/logs`` are appended per signal) and/or per-signal "
            "``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT``, ``OTEL_EXPORTER_OTLP_METRICS_ENDPOINT``, "
            "``OTEL_EXPORTER_OTLP_LOGS_ENDPOINT``. Skipping exporter setup."
        )
        return None

    resolved_team_id = (
        (team_id or "").strip()
        if team_id is not None
        else resolved_settings.team_id
    )
    if agent_id is not None:
        resolved_agent_id: Optional[str] = agent_id.strip() or None
    else:
        resolved_agent_id = resolved_settings.agent_id
    resource = create_agentcore_otel_resource(resolved_team_id, resolved_agent_id)

    trace_endpoint = _resolved_trace_otlp_endpoint(resolved_settings)
    metrics_endpoint = _resolved_metrics_otlp_endpoint(resolved_settings)
    logs_endpoint = _resolved_logs_otlp_endpoint(resolved_settings)

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=trace_endpoint))
    )
    trace.set_tracer_provider(tracer_provider)

    # Counter-family instruments use DELTA: each short-lived AgentCore session can
    # export per-export increments without needing a cumulativetodelta collector
    # processor. Histograms intentionally omitted — default temporality preserves
    # bucket snapshots for P50/P90/P95/P99-style percentile workflows.
    _agentcore_preferred_temporality: dict[type, AggregationTemporality] = {
        sdkmetrics.Counter: AggregationTemporality.DELTA,
        sdkmetrics.UpDownCounter: AggregationTemporality.DELTA,
        sdkmetrics.ObservableCounter: AggregationTemporality.DELTA,
        sdkmetrics.ObservableUpDownCounter: AggregationTemporality.DELTA,
    }
    metric_exporter = OTLPMetricExporter(
        endpoint=metrics_endpoint,
        preferred_temporality=_agentcore_preferred_temporality,
    )
    reader = PeriodicExportingMetricReader(metric_exporter)
    meter_provider = MeterProvider(metric_readers=[reader], resource=resource)
    metrics.set_meter_provider(meter_provider)

    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(
        BatchLogRecordProcessor(OTLPLogExporter(endpoint=logs_endpoint))
    )
    set_logger_provider(logger_provider)

    bridge: Optional[logging.Handler] = None
    if bridge_stdlib_logging:
        if not LoggingInstrumentor().is_instrumented_by_opentelemetry:
            LoggingInstrumentor().instrument(set_logging_format=True)
        bridge = LoggingHandler(level=logging.NOTSET, logger_provider=logger_provider)
        logging.getLogger(bridge_logger_name).addHandler(bridge)

    logger.info(
        "AgentCore OTLP/HTTP exporters initialized (traces, metrics, logs; "
        "service.name=%s team.id=%s agent.id=%s)",
        resource.attributes.get("service.name", ""),
        resource.attributes.get("team.id", ""),
        resource.attributes.get("agent.id", ""),
    )
    return AgentCoreOtelExporterHandle(
        bridge_handler=bridge,
        bridge_logger_name=bridge_logger_name if bridge_stdlib_logging else None,
        manages_global_providers=True,
    )


def setup_agentcore_otel_for_dome(
    dome: "Dome",
    *,
    enabled: Optional[bool] = None,
    settings: Optional["AgentCoreRuntimeSettings"] = None,
    team_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    bridge_stdlib_logging: bool = True,
    bridge_logger_name: str = "vijil.dome",
) -> Optional[AgentCoreDomeOtelExporterHandle]:
    """Configure OTLP exporters and attach OpenTelemetry to *dome* in one call.

    Runs :func:`setup_agentcore_otel_exporters_from_env`, then wires
    :func:`~vijil_dome.integrations.instrumentation.otel_instrumentation.instrument_dome`
    to the global tracer and meter (and applies log formatters when the stdlib bridge
    is enabled). Returns ``None`` when :func:`setup_agentcore_otel_exporters_from_env`
    would return ``None`` (disabled, missing optional dependencies, or no OTLP endpoint
    when no SDK providers are installed yet).

    If the host runtime already installed global SDK providers (for example
    ``setup_telemetry(service_name=...)``), exporter setup is skipped without calling
    ``set_tracer_provider`` / ``set_meter_provider`` again; Dome instrumentation still
    attaches to the existing providers. The returned handle has
    ``manages_global_providers`` false so :meth:`~AgentCoreOtelExporterHandle.shutdown`
    does not tear down the host's providers.

    After S3 reloads that replace guardrails, :meth:`AgentCoreDomeOtelExporterHandle.reinstrument_dome`
    keeps instrumentation on the new instances. :func:`start_agentcore_background_services`
    invokes it automatically when you pass this handle and the S3 config poller runs.

    Args:
        dome: The :class:`~vijil_dome.Dome.Dome` instance to instrument.
        enabled, settings, team_id, agent_id, bridge_stdlib_logging, bridge_logger_name:
            Forwarded to :func:`setup_agentcore_otel_exporters_from_env`.
    """
    base = setup_agentcore_otel_exporters_from_env(
        enabled=enabled,
        settings=settings,
        team_id=team_id,
        agent_id=agent_id,
        bridge_stdlib_logging=bridge_stdlib_logging,
        bridge_logger_name=bridge_logger_name,
    )
    if base is None:
        return None
    _apply_dome_otel_instrumentation(
        dome,
        bridge_stdlib_logging=bridge_stdlib_logging,
        bridge_logger_name=bridge_logger_name,
    )
    return AgentCoreDomeOtelExporterHandle(
        bridge_handler=base.bridge_handler,
        bridge_logger_name=base.bridge_logger_name,
        attach_stdlib_log_formatting=bridge_stdlib_logging,
        use_bridge_logger_name=bridge_logger_name,
        manages_global_providers=base.manages_global_providers,
    )
