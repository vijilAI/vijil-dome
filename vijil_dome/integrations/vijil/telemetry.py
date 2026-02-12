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

"""Darwin-compatible telemetry helpers for Dome.

Provides span attribute extraction functions used by instrument_dome()
to emit structured detection data that Darwin's TelemetryDetectionAdapter
can query from Tempo traces.

These are pure functions that extract detection metadata from GuardrailResult
and GuardResult objects. They are called by the Darwin detection span wrapper
in otel_instrumentation.py.
"""

from typing import Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import Span

from vijil_dome.guardrails import GuardrailResult, GuardResult


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
        for guard_name, guard_result in result.guard_exec_details.items():
            if guard_result.triggered:
                for detector_name, detection in guard_result.details.items():
                    if hasattr(detection, "result") and isinstance(detection.result, dict):
                        score = detection.result.get("score", 0.0)
                        if isinstance(score, (int, float)):
                            max_score = max(max_score, float(score))

    elif isinstance(result, GuardResult):
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
    if team_id:
        span.set_attribute("team.id", team_id)
    if agent_id:
        span.set_attribute("agent.id", agent_id)

    is_flagged = (
        result.flagged if isinstance(result, GuardrailResult) else result.triggered
    )
    span.set_attribute("detection.label", "flagged" if is_flagged else "clean")

    score = _extract_detection_score(result)
    span.set_attribute("detection.score", score)

    method = _extract_detection_method(result)
    span.set_attribute("detection.method", method)
