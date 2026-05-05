import logging
import os
import time

from opentelemetry import metrics, trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from vijil_dome import Dome, create_dome_config
from vijil_dome.integrations.instrumentation.otel_instrumentation import instrument_dome


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required env var: {name}")
    return value


def _optional_env(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def _auth_header(token: str) -> dict:
    method = os.getenv("OTEL_AUTH_METHOD", "bearer").title()
    return {"Authorization": f"{method} {token}"}


def _append_otlp_signal_path(base_endpoint: str, signal_path: str) -> str:
    base = base_endpoint.strip()
    if base.endswith("/"):
        return f"{base}{signal_path}"
    return f"{base}/{signal_path}"


def _resolved_signal_url(base: str, per_signal_env: str, rel_path: str) -> str:
    per = os.getenv(per_signal_env, "").strip()
    if per:
        return per
    return _append_otlp_signal_path(base, rel_path)


def _collector_auth_headers() -> dict:
    token = os.getenv("OTEL_COLLECTOR_TOKEN", "").strip()
    if not token:
        return {}
    return _auth_header(token)


def _build_resource(agent_id: str | None, team_id: str | None, user_id: str | None) -> Resource:
    attributes = {
        "service.name": "vijil-dome-telemetry-smoke",
        "service.version": "1.0.0",
        "deployment.environment": "smoke-test",
    }
    if agent_id:
        attributes["agent.id"] = agent_id
    if team_id:
        attributes["team.id"] = team_id
    if user_id:
        attributes["user.id"] = user_id
    return Resource(attributes=attributes)


def _create_log_handler(resource: Resource) -> tuple[LoggerProvider, LoggingHandler]:
    base = _required_env("DOME_OTEL_EXPORTER_OTLP_ENDPOINT")
    endpoint = _resolved_signal_url(base, "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", "v1/logs")
    headers = _collector_auth_headers()

    provider = LoggerProvider(resource=resource)
    set_logger_provider(provider)
    exporter = OTLPLogExporter(endpoint=endpoint, headers=headers)
    provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
    return provider, LoggingHandler(level=logging.INFO, logger_provider=provider)


def _create_tracer(resource: Resource):
    base = _required_env("DOME_OTEL_EXPORTER_OTLP_ENDPOINT")
    endpoint = _resolved_signal_url(base, "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "v1/traces")
    headers = _collector_auth_headers()

    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)
    exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider, provider.get_tracer("vijil-dome-telemetry-smoke")


def _create_meter(resource: Resource):
    base = _required_env("DOME_OTEL_EXPORTER_OTLP_ENDPOINT")
    endpoint = _resolved_signal_url(base, "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "v1/metrics")
    headers = _collector_auth_headers()

    exporter = OTLPMetricExporter(endpoint=endpoint, headers=headers)
    reader = PeriodicExportingMetricReader(exporter)
    provider = MeterProvider(metric_readers=[reader], resource=resource)
    metrics.set_meter_provider(provider)
    return provider, metrics.get_meter("vijil-dome-telemetry-smoke")


def _build_dome_config(agent_id: str | None, team_id: str | None, user_id: str | None) -> dict:
    # Uses deterministic keyword guard for smoke testing.
    cfg = {
        "input-guards": ["input-toxicity"],
        "output-guards": ["output-toxicity"],
        "input-early-exit": False,
        "output-early-exit": False,
        "input-toxicity": {
            "type": "moderation",
            "methods": ["moderation-flashtext"],
        },
        "output-toxicity": {
            "type": "moderation",
            "methods": ["moderation-flashtext"],
        },
    }
    if agent_id:
        cfg["agent_id"] = agent_id
    if team_id:
        cfg["team_id"] = team_id
    if user_id:
        cfg["user_id"] = user_id
    return cfg


def _scan_samples(
    dome: Dome, agent_id: str | None, team_id: str | None, user_id: str | None, logger: logging.Logger
):
    safe_input = "Hello, how are you?"
    flagged_input = "You should kill yourself."
    safe_output = "Here is a neutral summary."
    flagged_output = "Here is how to rob a bank."

    for text in [safe_input, flagged_input]:
        scan = dome.guard_input(text, agent_id=agent_id, team_id=team_id, user_id=user_id)
        logger.info(
            "input_scan flagged=%s score=%s text=%s",
            scan.flagged,
            scan.detection_score,
            text[:80],
        )

    for text in [safe_output, flagged_output]:
        scan = dome.guard_output(text, agent_id=agent_id, team_id=team_id, user_id=user_id)
        logger.info(
            "output_scan flagged=%s score=%s text=%s",
            scan.flagged,
            scan.detection_score,
            text[:80],
        )


def run_smoke_test() -> None:
    agent_id = _optional_env("AGENT_ID")
    user_id = _optional_env("USER_ID")
    team_id = _optional_env("TEAM_ID")

    resource = _build_resource(agent_id=agent_id, team_id=team_id, user_id=user_id)
    log_provider, log_handler = _create_log_handler(resource)
    trace_provider, tracer = _create_tracer(resource)
    meter_provider, meter = _create_meter(resource)

    if not LoggingInstrumentor().is_instrumented_by_opentelemetry:
        LoggingInstrumentor().instrument()

    logger = logging.getLogger("vijil.dome.telemetry_smoke")
    logger.addHandler(log_handler)
    logger.setLevel(logging.INFO)

    dome = Dome(dome_config=create_dome_config(_build_dome_config(agent_id, team_id, user_id)))
    instrument_dome(dome=dome, handler=log_handler, tracer=tracer, meter=meter)

    logger.info("starting telemetry smoke test")
    with tracer.start_as_current_span("custom-smoke-span") as span:
        if agent_id:
            span.set_attribute("agent.id", agent_id)
        if team_id:
            span.set_attribute("team.id", team_id)
        if user_id:
            span.set_attribute("user.id", user_id)
        span.set_attribute("smoke.kind", "manual-span")
        _scan_samples(dome, agent_id, team_id, user_id, logger)

    logger.info("completed telemetry smoke test")

    # Give async exporters time to flush; then shut down providers cleanly.
    flush_seconds = int(os.getenv("SMOKE_FLUSH_SECONDS", "8"))
    time.sleep(max(flush_seconds, 2))
    meter_provider.shutdown()
    trace_provider.shutdown()
    log_provider.shutdown()


if __name__ == "__main__":
    run_smoke_test()
