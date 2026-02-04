"""Vijil platform integrations for Dome.

This module provides integrations with the Vijil platform:
- evaluate: Fetch Dome configs from Vijil Evaluate API
- telemetry: Darwin-compatible OpenTelemetry instrumentation

Example:
    # Fetch config from Vijil Evaluate
    from vijil_dome.integrations.vijil.evaluate import get_config_from_vijil_agent
    config = get_config_from_vijil_agent(api_token, agent_id)

    # Set up Darwin-compatible telemetry
    from vijil_dome.integrations.vijil.telemetry import instrument_for_darwin
    instrument_for_darwin(dome.input_guardrail, tracer, meter)
"""

from vijil_dome.integrations.vijil.evaluate import (
    get_config_from_vijil_agent,
    get_config_from_vijil_evaluation,
)

from vijil_dome.integrations.vijil.telemetry import (
    darwin_trace,
    instrument_for_darwin,
    create_darwin_flagged_counter,
    create_darwin_requests_counter,
    create_darwin_latency_histogram,
    DARWIN_METRIC_PREFIX,
)

__all__ = [
    # Evaluate integration
    "get_config_from_vijil_agent",
    "get_config_from_vijil_evaluation",
    # Darwin telemetry
    "darwin_trace",
    "instrument_for_darwin",
    "create_darwin_flagged_counter",
    "create_darwin_requests_counter",
    "create_darwin_latency_histogram",
    "DARWIN_METRIC_PREFIX",
]
