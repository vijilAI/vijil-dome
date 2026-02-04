# Darwin-Compatible Telemetry Integration Design

> **For Claude:** This is a design document for enhancing Dome's telemetry to integrate with Darwin evolution. Use `superpowers:writing-plans` to create implementation tasks.

**Date:** 2026-02-04
**Status:** Draft
**Authors:** Claude (analysis and design)

---

## Executive Summary

Darwin's evolution workflow requires Dome detection telemetry to trigger mutations. Currently, there's a **critical schema mismatch** between Dome's telemetry output and Darwin's expected input format. This design proposes minimal changes to Dome's instrumentation to enable seamless integration.

---

## Problem Statement

### Current State

**Dome emits** (via OpenTelemetry):

| Component | What Dome Sends | Example |
|-----------|-----------------|---------|
| **Metric name** | `{guardrail_name}-flagged_total` | `input-guardrail-flagged_total` |
| **Metric attributes** | `agent.id` only | `{"agent.id": "agent-123"}` |
| **Span attributes** | `function.result` (string dump) | `"{'flagged': True, 'score': 0.85, ...}"` |
| **Service name** | Not explicitly set | (inherits from OTEL config) |

**Darwin expects** (TelemetryDetectionAdapter):

| Component | What Darwin Queries | Example |
|-----------|---------------------|---------|
| **Metric name** | `dome-flagged_total` | Fixed prefix |
| **Metric attributes** | `agent_configuration_id`, `team_id` | Multi-tenant filtering |
| **Span attributes** | `detection.score`, `detection.label`, `detection.method` | Structured typed attributes |
| **Service name filter** | `service.name="service-dome"` | Service-based filtering |

### Impact

- **Metrics queries fail**: Metric naming mismatch
- **Trace parsing fails**: Missing structured attributes
- **Multi-tenant filtering fails**: Missing `team_id`
- **Detection classification fails**: Missing `detection.method`

---

## Proposed Solution

### Option A: Darwin-Aware Tracing Decorator (Recommended)

Add a new decorator that emits Darwin-compatible span attributes alongside the existing generic attributes.

**New file: `vijil_dome/integrations/vijil/telemetry.py`**

```python
"""Darwin-compatible telemetry instrumentation for Dome.

When Dome runs in the Vijil platform context, this module provides
enhanced instrumentation that emits span attributes Darwin can parse.
"""

from functools import wraps
from inspect import iscoroutinefunction
from typing import Callable, Optional

from opentelemetry.sdk.trace import Tracer, Span
from pydantic import BaseModel

from vijil_dome.guardrails import GuardrailResult, GuardResult


def _set_darwin_span_attributes(
    span: Span,
    result: GuardrailResult | GuardResult,
    agent_id: Optional[str] = None,
    team_id: Optional[str] = None,
) -> None:
    """Set Darwin-compatible span attributes.

    Args:
        span: The current OTEL span
        result: Guardrail or Guard scan result
        agent_id: Agent configuration ID (from kwargs)
        team_id: Team ID for multi-tenant filtering
    """
    # Team and agent context
    if team_id:
        span.set_attribute("team.id", team_id)
    if agent_id:
        span.set_attribute("agent.id", agent_id)

    # Detection attributes Darwin expects
    span.set_attribute("detection.label", "flagged" if result.flagged else "clean")

    # Extract score - prefer max_score, fallback to first detection score
    score = 0.0
    if hasattr(result, "trace") and result.trace:
        # GuardrailResult has trace dict with guard results
        for guard_name, guard_result in result.trace.items():
            if isinstance(guard_result, dict) and guard_result.get("triggered"):
                for det_name, det_result in guard_result.get("detections", {}).items():
                    if isinstance(det_result, dict):
                        score = max(score, det_result.get("score", 0.0))
    span.set_attribute("detection.score", score)

    # Extract method from first triggered guard
    method = "unknown"
    if hasattr(result, "trace") and result.trace:
        for guard_name, guard_result in result.trace.items():
            if isinstance(guard_result, dict) and guard_result.get("triggered"):
                method = guard_name
                break
    span.set_attribute("detection.method", method)


def darwin_trace(tracer: Tracer, name: str):
    """Decorator for Darwin-compatible tracing.

    Wraps scan functions to emit both generic and Darwin-specific
    span attributes for integration with the evolution workflow.

    Usage:
        @darwin_trace(tracer, "input-guardrail.scan")
        def scan(self, input_text: str, agent_id: str = None, team_id: str = None):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with tracer.start_as_current_span(name) as span:
                # Generic attributes (existing behavior)
                span.set_attribute("function.args", str(args))
                span.set_attribute("function.kwargs", str(kwargs))

                result = func(*args, **kwargs)

                # Generic result attribute
                if isinstance(result, BaseModel):
                    span.set_attribute("function.result", str(result.model_dump()))
                else:
                    span.set_attribute("function.result", str(result))

                # Darwin-specific attributes
                if isinstance(result, (GuardrailResult, GuardResult)):
                    _set_darwin_span_attributes(
                        span,
                        result,
                        agent_id=kwargs.get("agent_id"),
                        team_id=kwargs.get("team_id"),
                    )

                return result

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with tracer.start_as_current_span(name) as span:
                span.set_attribute("function.args", str(args))
                span.set_attribute("function.kwargs", str(kwargs))

                result = await func(*args, **kwargs)

                if isinstance(result, BaseModel):
                    span.set_attribute("function.result", str(result.model_dump()))
                else:
                    span.set_attribute("function.result", str(result))

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
```

### Metric Naming Enhancement

Add a consistent metric prefix for Darwin compatibility:

**Update: `vijil_dome/guardrails/instrumentation/metrics.py`**

```python
# Add constant for Darwin-compatible metric prefix
DARWIN_METRIC_PREFIX = "dome"

def _create_request_flagged_counter(name: str, meter: Meter, darwin_compatible: bool = False):
    """Create flagged counter with optional Darwin-compatible naming."""
    metric_name = f"{DARWIN_METRIC_PREFIX}-flagged_total" if darwin_compatible else f"{name}-flagged_total"
    return meter.create_counter(
        metric_name,
        description=f"Number of requests to {name} that are flagged",
    )
```

### Integration Function

Provide a single function to set up Darwin-compatible instrumentation:

```python
def instrument_for_darwin(
    guardrail: Guardrail,
    tracer: Tracer,
    meter: Meter,
    guardrail_name: Optional[str] = None,
) -> None:
    """Set up Darwin-compatible instrumentation for a guardrail.

    This function configures both tracing and metrics with Darwin-compatible
    attributes and naming conventions.

    Args:
        guardrail: The guardrail to instrument
        tracer: OTEL tracer for span creation
        meter: OTEL meter for metrics
        guardrail_name: Optional custom name (default: guardrail.level-guardrail)

    Example:
        from vijil_dome.integrations.vijil.telemetry import instrument_for_darwin

        dome = Dome(config)
        tracer = trace.get_tracer("service-dome")
        meter = metrics.get_meter("service-dome")

        instrument_for_darwin(dome.input_guardrail, tracer, meter)
    """
    # Implementation uses darwin_trace decorator and Darwin metric naming
    ...
```

---

## Integration Points

### Console Service (service_dome)

When running Dome in the Console context, use Darwin-compatible instrumentation:

```python
# In vijil-console/src/service_dome/main.py
from vijil_dome import Dome
from vijil_dome.integrations.vijil.telemetry import instrument_for_darwin
from opentelemetry import trace, metrics

dome = Dome(config)
tracer = trace.get_tracer("service-dome")  # Darwin expects this service name
meter = metrics.get_meter("service-dome")

instrument_for_darwin(dome.input_guardrail, tracer, meter)
instrument_for_darwin(dome.output_guardrail, tracer, meter)
```

### Standalone Dome Usage

For standalone Dome usage (not in Vijil platform), the existing instrumentation remains unchanged.

---

## Attribute Reference

### Span Attributes Set by Darwin-Compatible Tracing

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| `team.id` | string | Team ID for multi-tenant filtering | `"7295f905-194b-42a8-95e2-..."` |
| `agent.id` | string | Agent configuration ID | `"agent-cs-helpdesk-v2"` |
| `detection.label` | string | "flagged" or "clean" | `"flagged"` |
| `detection.score` | double | Max detection confidence (0.0-1.0) | `0.92` |
| `detection.method` | string | Guard/detector that triggered | `"prompt-injection"` |
| `function.args` | string | (Existing) String dump of args | `"('input text',)"` |
| `function.kwargs` | string | (Existing) String dump of kwargs | `"{'agent_id': '...'}"` |
| `function.result` | string | (Existing) String dump of result | `"{'flagged': True, ...}"` |

### Metrics Emitted with Darwin-Compatible Naming

| Metric | Type | Attributes | Description |
|--------|------|------------|-------------|
| `dome-flagged_total` | Counter | `agent.id`, `team.id` | Total flagged detections |
| `dome-requests_total` | Counter | `agent.id`, `team.id` | Total scan requests |
| `dome-latency_seconds` | Histogram | `agent.id`, `team.id` | Scan latency |

---

## Migration Path

### Phase 1: Add Darwin-Compatible Module (Non-Breaking)

1. Add `vijil_dome/integrations/vijil/telemetry.py` with Darwin tracing
2. Add `instrument_for_darwin()` convenience function
3. No changes to existing `auto_trace` or metrics

### Phase 2: Console Integration

1. Update `service_dome` in vijil-console to use `instrument_for_darwin()`
2. Set OTEL service name to `service-dome`
3. Verify Darwin can query detections via Tempo/Mimir

### Phase 3: TelemetryDetectionAdapter Alignment

1. Update Darwin's adapter to handle both old and new formats
2. Prefer structured attributes, fallback to string parsing
3. Document expected attribute schema

---

## Testing Strategy

### Unit Tests

```python
def test_darwin_span_attributes():
    """Darwin-compatible attributes are set on spans."""
    result = GuardrailResult(flagged=True, trace={...})
    span = MockSpan()

    _set_darwin_span_attributes(span, result, agent_id="agent-1", team_id="team-1")

    assert span.attributes["team.id"] == "team-1"
    assert span.attributes["detection.label"] == "flagged"
    assert span.attributes["detection.score"] > 0
    assert span.attributes["detection.method"] != "unknown"
```

### Integration Tests

1. Deploy Dome with Darwin instrumentation
2. Generate detections via test prompts
3. Query Tempo for traces with Darwin attributes
4. Query Mimir for `dome-flagged_total` metric
5. Verify Darwin can create mutation proposals from detections

---

## Open Questions

1. **Backward compatibility**: Should Darwin adapter support both formats during migration?
2. **Team ID propagation**: How is `team_id` passed through the request chain?
3. **Metric aggregation**: Should Darwin query by guardrail name or use aggregated `dome-*` metrics?

---

## Next Steps

1. [ ] Implement `vijil_dome/integrations/vijil/telemetry.py`
2. [ ] Add unit tests for Darwin attributes
3. [ ] Update service_dome to use Darwin instrumentation
4. [ ] Update Darwin's TelemetryDetectionAdapter for new format
5. [ ] End-to-end test: Dome detection → Darwin mutation proposal
