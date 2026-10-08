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

import json
import logging
from functools import wraps
from typing import Any, Literal, Optional
from opentelemetry.sdk.trace import Tracer
from opentelemetry.trace.span import Span
from opentelemetry.metrics import Meter
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from vijil_dome.controls.models import ControlMatch, EvaluationResult, Step
from vijil_dome.core import VijilDome
from vijil_dome.guardrails.instrumentation.instrumentation import (
    instrument_with_monitors,
    instrument_with_tracer,
)
from vijil_dome import Dome
from vijil_dome.guardrails import Guardrail, GuardrailResult
from vijil_dome.integrations.vijil.telemetry import _safe_set_attribute, _set_darwin_span_attributes
import socket


class VijilLogFormatter(logging.Formatter):
    def format(self, record):
        # Set the default values of the OTel Logging information if absent
        # this is only the case if OTel instrumentation is disabled
        record.otelTraceID = getattr(record, "otelTraceID", 0)
        record.otelSpanID = getattr(record, "otelSpanID", 0)
        record.otelServiceName = getattr(record, "otelServiceName", "N/A")
        record.otelTraceSampled = getattr(record, "otelTraceSampled", "N/A")

        # Add the IP address to the log record if it doesn't exist
        record.ip = getattr(
            record,
            "ip",
            socket.gethostbyname(socket.gethostname()),
        )
        return super().format(record)


def get_vijil_log_formatter():
    formatter = VijilLogFormatter(
        "%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] [trace_id=%(otelTraceID)s span_id=%(otelSpanID)s resource.service.name=%(otelServiceName)s trace_sampled=%(otelTraceSampled)s resource.service.ip=%(ip)s] - %(msg)s"
    )
    return formatter


def instrument_logger(logger: logging.Logger):
    for handler in logger.handlers:
        formatter = get_vijil_log_formatter()
        handler.setFormatter(formatter)


def _add_darwin_detection_spans(
    guardrail: Guardrail,
    tracer: Tracer,
    guardrail_name: str,
    enforce: bool = True,
) -> None:
    """Wrap guardrail scan methods to emit Darwin-compatible detection spans.

    Creates 'dome-detection' spans with structured attributes that Darwin's
    TelemetryDetectionAdapter can query from Tempo traces.

    Attributes set on each span:
        - dome.guardrail: guardrail name (e.g., "dome-input", "dome-output")
        - detection.label: "flagged" or "clean"
        - detection.score: max detection score (0.0-1.0)
        - detection.method: name of the triggered guard/detector
        - team.id: team context (from kwargs)
        - agent.id: agent context (from kwargs)
        - user.id: user context (from kwargs)

    Args:
        guardrail: The Guardrail instance to wrap.
        tracer: OTEL tracer for span creation.
        guardrail_name: Name prefix (e.g., "dome-input", "dome-output").
    """
    original_scan = guardrail.scan
    original_async_scan = guardrail.async_scan

    @wraps(original_scan)
    def scan_with_darwin_spans(*args: Any, **kwargs: Any) -> Any:
        team_id = kwargs.get("team_id")
        agent_id = kwargs.get("agent_id")
        user_id = kwargs.get("user_id")
        with tracer.start_as_current_span("dome-detection") as span:
            span.set_attribute("dome.guardrail", guardrail_name)
            result = original_scan(*args, **kwargs)
            if isinstance(result, GuardrailResult):
                _set_darwin_span_attributes(
                    span,
                    result,
                    agent_id=agent_id,
                    team_id=team_id,
                    user_id=user_id,
                )
                span.set_attribute("dome.guard.enforced", enforce and result.flagged)
            return result

    @wraps(original_async_scan)
    async def async_scan_with_darwin_spans(*args: Any, **kwargs: Any) -> Any:
        team_id = kwargs.get("team_id")
        agent_id = kwargs.get("agent_id")
        user_id = kwargs.get("user_id")
        with tracer.start_as_current_span("dome-detection") as span:
            span.set_attribute("dome.guardrail", guardrail_name)
            result = await original_async_scan(*args, **kwargs)
            if isinstance(result, GuardrailResult):
                _set_darwin_span_attributes(
                    span,
                    result,
                    agent_id=agent_id,
                    team_id=team_id,
                    user_id=user_id,
                )
                span.set_attribute("dome.guard.enforced", enforce and result.flagged)
            return result

    guardrail.scan = scan_with_darwin_spans  # type: ignore[method-assign]
    guardrail.async_scan = async_scan_with_darwin_spans  # type: ignore[method-assign]


def instrument_dome(
    dome: Dome,
    handler: Optional[logging.Handler],
    tracer: Optional[Tracer],
    meter: Optional[Meter],
):
    if getattr(dome, "_instrumented", False):
        return
    if not LoggingInstrumentor().is_instrumented_by_opentelemetry:
        LoggingInstrumentor().instrument()

    # Enable OTel logging if a logging handler is provided
    if handler:
        logger = logging.getLogger("vijil.dome")
        logger.addHandler(handler)
        instrument_logger(logger)

    # Add tracer for detailed per-guard/per-detector spans
    if tracer:
        if dome.input_guardrail is not None:
            instrument_with_tracer(dome.input_guardrail, tracer, "Dome-Input-Guardrail")
        if dome.output_guardrail is not None:
            instrument_with_tracer(
                dome.output_guardrail, tracer, "Dome-Output-Guardrail"
            )

    if meter:
        # Add split metrics (dome-input-*, dome-output-*)
        if dome.input_guardrail is not None:
            instrument_with_monitors(dome.input_guardrail, meter, "dome-input")
        if dome.output_guardrail is not None:
            instrument_with_monitors(dome.output_guardrail, meter, "dome-output")

    # Add Darwin-compatible detection spans at the guardrail level.
    # These must be added AFTER monitors and tracer so the "dome-detection"
    # span wraps the full scan chain (metrics + generic traces + original scan).
    if tracer:
        if dome.input_guardrail is not None:
            _add_darwin_detection_spans(dome.input_guardrail, tracer, "dome-input", enforce=dome.enforce)
        if dome.output_guardrail is not None:
            _add_darwin_detection_spans(dome.output_guardrail, tracer, "dome-output", enforce=dome.enforce)

    dome._instrumented = True  # type: ignore[attr-defined]


def _set_dome_control_span_attributes(
    span: Span, match: ControlMatch, vijil_dome: VijilDome
) -> None:
    """Set span attributes for one triggered Control match.

    Mirrors ``_set_darwin_span_attributes``'s "the span is generic, the
    metadata is evaluator-specific" split: control.name/control.decision
    and team.id/agent.id/user.id are always set; policy.id/rule.ids/
    consequence.* are only set when the triggering evaluator's metadata
    carries them (currently just ``PolicyRuleJudge`` -- see
    ``vijil_dome/controls/evaluators/policy_rule_judge.py``). Any other
    evaluator's Control still gets a span, just without those extra
    attributes.
    """
    _safe_set_attribute(span, "control.name", match.control_name)
    if match.action is not None:
        _safe_set_attribute(span, "control.decision", match.action.decision)
    _safe_set_attribute(span, "team.id", vijil_dome.team_id)
    _safe_set_attribute(span, "agent.id", vijil_dome.agent_id)
    _safe_set_attribute(span, "user.id", vijil_dome.user_id)
    _safe_set_attribute(span, "detection.confidence", match.confidence)

    control = next(
        (c for c in vijil_dome.engine.controls if c.name == match.control_name),
        None,
    )
    if control is not None:
        _safe_set_attribute(
            span, "policy.id", control.annotations.get("vijil.ai/source-policy-id")
        )

    metadata = match.metadata or {}
    violated_rule_ids = metadata.get("violated_rule_ids")
    rule_verdicts = metadata.get("rule_verdicts")
    if violated_rule_ids:
        _safe_set_attribute(span, "rule.ids", list(violated_rule_ids))
        if isinstance(rule_verdicts, dict):
            worst_id = _pick_worst_violated_rule(violated_rule_ids, rule_verdicts)
            worst = rule_verdicts.get(worst_id, {})
            consequence = worst.get("consequence")
            if isinstance(consequence, dict):
                _safe_set_attribute(span, "consequence.action", consequence.get("action"))
                _safe_set_attribute(span, "consequence.severity", consequence.get("severity"))
            _safe_set_attribute(span, "rule.type", worst.get("rule_type"))
    if isinstance(rule_verdicts, dict):
        # Every rule the judge checked, not just the violated ones -- kept
        # as a JSON blob rather than N separate attributes since the rule
        # count per Control is open-ended.
        _safe_set_attribute(
            span,
            "rule.verdicts",
            _bounded_rule_verdicts_json(rule_verdicts, violated_rule_ids or []),
        )


_SEVERITY_RANK = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


def _pick_worst_violated_rule(
    violated_rule_ids: list[str], rule_verdicts: dict[str, Any]
) -> str:
    """Highest-severity violated rule, not just the first in configured order.

    Configured (bucket) order has nothing to do with severity -- a low
    rule listed before a critical one in the same bucket would otherwise
    make the span report consequence.severity=low, so severity-based
    queries would miss the critical violation entirely.
    """
    def rank(rule_id: str) -> int:
        verdict = rule_verdicts.get(rule_id) or {}
        consequence = verdict.get("consequence") or {}
        severity = consequence.get("severity") if isinstance(consequence, dict) else None
        return _SEVERITY_RANK.get(severity, -1) if isinstance(severity, str) else -1

    return max(violated_rule_ids, key=rank)


def _bounded_rule_verdicts_json(
    rule_verdicts: dict[str, Any],
    violated_rule_ids: list[str],
    max_len: int = 2048,
) -> str:
    """Serialize rule_verdicts as valid JSON within ~max_len, never by
    slicing the serialized string (which can cut a JSON document in half
    and leave consumers unable to parse the audit payload at all).

    Violated rules are kept first -- they're what an audit actually needs
    -- with any entries that don't fit replaced by an explicit
    ``_omitted_rule_ids`` marker rather than silently vanishing.
    """
    ordered_ids = list(violated_rule_ids) + [
        rid for rid in rule_verdicts if rid not in violated_rule_ids
    ]
    kept: dict[str, Any] = {}
    omitted: list[str] = []
    for rule_id in ordered_ids:
        trial = {**kept, rule_id: rule_verdicts[rule_id]}
        if kept and len(json.dumps(trial, default=str)) > max_len:
            omitted.append(rule_id)
            continue
        kept = trial
    if omitted:
        kept["_omitted_rule_ids"] = omitted
    return json.dumps(kept, default=str)


def instrument_vijil_dome(vijil_dome: VijilDome, tracer: Tracer) -> None:
    """Add ``dome-control`` OTel spans to a :class:`VijilDome` instance.

    The Controls-first runtime (``vijil_dome/core.py``) has no span
    emission of its own, unlike the Guardrail path ``instrument_dome``
    wraps with ``dome-detection`` spans. This fills that gap the same
    way: wrap ``vijil_dome.engine.evaluate`` so every *triggered* Control
    emits one span, carrying enough to reconstruct "what fired, and why"
    without needing Dome's own logs -- this, and not a Console
    write-back API, is how enforcement decisions get back to Console
    (point ``DOME_TRACES_COLLECTOR_ENDPOINT`` at Console's collector and
    they show up like any other Dome telemetry).

    Idempotent, same as ``instrument_dome``: calling it twice on an
    already-instrumented instance is a no-op.
    """
    if getattr(vijil_dome, "_instrumented", False):
        return

    original_evaluate = vijil_dome.engine.evaluate

    @wraps(original_evaluate)
    async def evaluate_with_spans(
        step: Step, stage: Literal["pre", "post"] = "pre"
    ) -> EvaluationResult:
        result = await original_evaluate(step, stage)
        for match in result.matches:
            if not match.triggered:
                continue
            with tracer.start_as_current_span("dome-control") as span:
                _set_dome_control_span_attributes(span, match, vijil_dome)
        return result

    vijil_dome.engine.evaluate = evaluate_with_spans  # type: ignore[method-assign]
    vijil_dome._instrumented = True  # type: ignore[attr-defined]
