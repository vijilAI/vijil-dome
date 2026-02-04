"""Darwin-compatible telemetry instrumentation for Dome.

When Dome runs in the Vijil platform context, this module provides
enhanced instrumentation that emits span attributes Darwin can parse
for triggering evolution mutations.

Key differences from generic auto_trace:
- Typed span attributes (detection.score as float, not string dump)
- Team context (team.id for multi-tenant filtering)
- Standardized metric naming (dome-flagged_total vs {guardrail}-flagged_total)

Usage:
    from vijil_dome.integrations.vijil.telemetry import darwin_trace, instrument_for_darwin
    from opentelemetry import trace, metrics

    tracer = trace.get_tracer("service-dome")
    meter = metrics.get_meter("service-dome")

    # Option 1: Use decorator directly
    @darwin_trace(tracer, "input-guardrail.scan")
    def my_scan_function(...):
        ...

    # Option 2: Instrument existing guardrail
    instrument_for_darwin(dome.input_guardrail, tracer, meter)
"""

from functools import wraps
from inspect import iscoroutinefunction
from typing import Callable, Optional, Any, Union, TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import Tracer, Span
    from opentelemetry.sdk.metrics import Meter

from vijil_dome.guardrails import GuardrailResult, GuardResult


# Darwin-compatible metric prefix
DARWIN_METRIC_PREFIX = "dome"


def _extract_detection_score(result: Union[GuardrailResult, GuardResult]) -> float:
    """Extract the maximum detection score from a guard/guardrail result.

    For GuardrailResult: Walks through guard_exec_details to find triggered
    detections and returns the highest score.

    For GuardResult: Walks through details to find the highest detection score.

    Args:
        result: The scan result from a guard or guardrail.

    Returns:
        The maximum detection score (0.0-1.0), or 0.0 if no detections.
    """
    max_score = 0.0

    if isinstance(result, GuardrailResult):
        # GuardrailResult.guard_exec_details: Dict[str, GuardResult]
        for guard_name, guard_result in result.guard_exec_details.items():
            if guard_result.triggered:
                # GuardResult.details: Dict[str, DetectionTimingResult]
                for detector_name, detection in guard_result.details.items():
                    if hasattr(detection, "result") and isinstance(detection.result, dict):
                        score = detection.result.get("score", 0.0)
                        if isinstance(score, (int, float)):
                            max_score = max(max_score, float(score))

    elif isinstance(result, GuardResult):
        # GuardResult.details: Dict[str, DetectionTimingResult]
        for detector_name, detection in result.details.items():
            if hasattr(detection, "result") and isinstance(detection.result, dict):
                score = detection.result.get("score", 0.0)
                if isinstance(score, (int, float)):
                    max_score = max(max_score, float(score))

    return max_score


def _extract_detection_method(result: Union[GuardrailResult, GuardResult]) -> str:
    """Extract the name of the first triggered guard/detector.

    Darwin uses this to understand which detection method flagged the input,
    enabling targeted mutations for specific vulnerability types.

    Args:
        result: The scan result from a guard or guardrail.

    Returns:
        The name of the triggered guard/detector, or "unknown" if none.
    """
    if isinstance(result, GuardrailResult):
        for guard_name, guard_result in result.guard_exec_details.items():
            if guard_result.triggered:
                return guard_name

    elif isinstance(result, GuardResult):
        if result.triggered:
            for detector_name, detection in result.details.items():
                if hasattr(detection, "hit") and detection.hit:
                    return detector_name

    return "unknown"


def _set_darwin_span_attributes(
    span: "Span",
    result: Union[GuardrailResult, GuardResult],
    agent_id: Optional[str] = None,
    team_id: Optional[str] = None,
) -> None:
    """Set Darwin-compatible span attributes on the current span.

    These attributes enable Darwin's TelemetryDetectionAdapter to query
    detections from Tempo traces and create mutation proposals.

    Args:
        span: The current OTEL span.
        result: Guardrail or Guard scan result.
        agent_id: Agent configuration ID (for agent-specific mutations).
        team_id: Team ID for multi-tenant filtering.
    """
    # Team and agent context
    if team_id:
        span.set_attribute("team.id", team_id)
    if agent_id:
        span.set_attribute("agent.id", agent_id)

    # Detection label (Darwin expects "flagged" or "clean")
    is_flagged = (
        result.flagged if isinstance(result, GuardrailResult) else result.triggered
    )
    span.set_attribute("detection.label", "flagged" if is_flagged else "clean")

    # Detection score (0.0-1.0)
    score = _extract_detection_score(result)
    span.set_attribute("detection.score", score)

    # Detection method (which guard/detector triggered)
    method = _extract_detection_method(result)
    span.set_attribute("detection.method", method)


def darwin_trace(tracer: "Tracer", name: str) -> Callable:
    """Decorator for Darwin-compatible tracing of scan functions.

    Wraps scan functions to emit both generic (existing behavior) and
    Darwin-specific (queryable) span attributes for integration with
    the evolution workflow.

    The wrapped function MUST return a GuardrailResult or GuardResult.
    Darwin-specific attributes are only set when the return type is correct.

    Usage:
        @darwin_trace(tracer, "input-guardrail.scan")
        def scan(self, input_text: str, agent_id: str = None, team_id: str = None):
            ...

        @darwin_trace(tracer, "output-guardrail.async-scan")
        async def async_scan(self, output_text: str, **kwargs):
            ...

    Args:
        tracer: OTEL tracer instance (from trace.get_tracer("service-dome")).
        name: Span name (e.g., "input-guardrail.scan").

    Returns:
        Decorator function.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.start_as_current_span(name) as span:
                # Generic attributes (existing behavior for backward compatibility)
                span.set_attribute("function.args", str(args))
                span.set_attribute("function.kwargs", str(kwargs))

                result = func(*args, **kwargs)

                # Generic result attribute (existing behavior)
                if isinstance(result, BaseModel):
                    span.set_attribute("function.result", str(result.model_dump()))
                else:
                    span.set_attribute("function.result", str(result))

                # Darwin-specific attributes (NEW - enables evolution workflow)
                if isinstance(result, (GuardrailResult, GuardResult)):
                    _set_darwin_span_attributes(
                        span,
                        result,
                        agent_id=kwargs.get("agent_id"),
                        team_id=kwargs.get("team_id"),
                    )

                return result

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.start_as_current_span(name) as span:
                # Generic attributes (existing behavior)
                span.set_attribute("function.args", str(args))
                span.set_attribute("function.kwargs", str(kwargs))

                result = await func(*args, **kwargs)

                # Generic result attribute (existing behavior)
                if isinstance(result, BaseModel):
                    span.set_attribute("function.result", str(result.model_dump()))
                else:
                    span.set_attribute("function.result", str(result))

                # Darwin-specific attributes (NEW - enables evolution workflow)
                if isinstance(result, (GuardrailResult, GuardResult)):
                    _set_darwin_span_attributes(
                        span,
                        result,
                        agent_id=kwargs.get("agent_id"),
                        team_id=kwargs.get("team_id"),
                    )

                return result

        return async_wrapper if iscoroutinefunction(func) else sync_wrapper

    return decorator


def create_darwin_flagged_counter(meter: "Meter", level: str = "guardrail"):
    """Create a Darwin-compatible flagged counter metric.

    Uses standardized naming that Darwin's Mimir queries expect:
    - dome-flagged_total (instead of {guardrail_name}-flagged_total)

    Args:
        meter: OTEL meter instance.
        level: Either "guardrail" or "guard" for description context.

    Returns:
        Counter metric for flagged detections.
    """
    return meter.create_counter(
        f"{DARWIN_METRIC_PREFIX}-flagged_total",
        description=f"Number of requests to {level} that are flagged (Darwin-compatible)",
    )


def create_darwin_requests_counter(meter: "Meter", level: str = "guardrail"):
    """Create a Darwin-compatible requests counter metric.

    Args:
        meter: OTEL meter instance.
        level: Either "guardrail" or "guard" for description context.

    Returns:
        Counter metric for total requests.
    """
    return meter.create_counter(
        f"{DARWIN_METRIC_PREFIX}-requests_total",
        description=f"Total requests to {level} (Darwin-compatible)",
    )


def create_darwin_latency_histogram(meter: "Meter", level: str = "guardrail"):
    """Create a Darwin-compatible latency histogram metric.

    Args:
        meter: OTEL meter instance.
        level: Either "guardrail" or "guard" for description context.

    Returns:
        Histogram metric for scan latency.
    """
    return meter.create_histogram(
        f"{DARWIN_METRIC_PREFIX}-latency_seconds",
        description=f"Scan latency for {level} (Darwin-compatible)",
        unit="s",
    )


def instrument_for_darwin(
    guardrail: Any,
    tracer: "Tracer",
    meter: "Meter",
    guardrail_name: Optional[str] = None,
) -> None:
    """Set up Darwin-compatible instrumentation for a guardrail.

    This function patches a guardrail's scan methods to emit Darwin-compatible
    telemetry. It creates metrics with standardized naming and wraps the scan
    functions with darwin_trace.

    Note: This modifies the guardrail in-place. Call once at application startup.

    Args:
        guardrail: The Guardrail instance to instrument.
        tracer: OTEL tracer for span creation.
        meter: OTEL meter for metrics.
        guardrail_name: Optional custom name (default: guardrail.level-guardrail).

    Example:
        from vijil_dome import Dome
        from vijil_dome.integrations.vijil.telemetry import instrument_for_darwin
        from opentelemetry import trace, metrics

        dome = Dome(config)
        tracer = trace.get_tracer("service-dome")
        meter = metrics.get_meter("service-dome")

        instrument_for_darwin(dome.input_guardrail, tracer, meter)
        instrument_for_darwin(dome.output_guardrail, tracer, meter)
    """
    from vijil_dome.guardrails import Guardrail

    if not isinstance(guardrail, Guardrail):
        raise TypeError(f"Expected Guardrail, got {type(guardrail).__name__}")

    name = guardrail_name or f"{guardrail.level}-guardrail"

    # Create Darwin-compatible metrics
    flagged_counter = create_darwin_flagged_counter(meter, name)
    requests_counter = create_darwin_requests_counter(meter, name)
    latency_histogram = create_darwin_latency_histogram(meter, name)

    # Store original methods
    original_scan = guardrail.scan
    original_async_scan = guardrail.async_scan

    # Wrap sync scan with Darwin instrumentation
    @darwin_trace(tracer, f"{name}.scan")
    def instrumented_scan(
        query_string: str, agent_id: Optional[str] = None, team_id: Optional[str] = None
    ) -> GuardrailResult:
        result = original_scan(query_string, agent_id=agent_id)

        # Record metrics with Darwin-compatible attributes
        labels = {}
        if agent_id:
            labels["agent.id"] = agent_id
        if team_id:
            labels["team.id"] = team_id

        requests_counter.add(1, labels)
        latency_histogram.record(result.exec_time, labels)
        if result.flagged:
            flagged_counter.add(1, labels)

        return result

    # Wrap async scan with Darwin instrumentation
    @darwin_trace(tracer, f"{name}.async-scan")
    async def instrumented_async_scan(
        query_string: str, agent_id: Optional[str] = None, team_id: Optional[str] = None
    ) -> GuardrailResult:
        result = await original_async_scan(query_string, agent_id=agent_id)

        # Record metrics with Darwin-compatible attributes
        labels = {}
        if agent_id:
            labels["agent.id"] = agent_id
        if team_id:
            labels["team.id"] = team_id

        requests_counter.add(1, labels)
        latency_histogram.record(result.exec_time, labels)
        if result.flagged:
            flagged_counter.add(1, labels)

        return result

    # Replace methods with instrumented versions
    guardrail.scan = instrumented_scan
    guardrail.async_scan = instrumented_async_scan
