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

"""Darwin-compatible span attribute helpers for Dome.

Reads detection metadata from native GuardrailResult/GuardResult fields
and emits structured span attributes that Darwin's TelemetryDetectionAdapter
can query from Tempo traces.
"""

from typing import Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import Span

from vijil_dome.guardrails import GuardrailResult, GuardResult


def _set_darwin_span_attributes(
    span: "Span",
    result: Union[GuardrailResult, GuardResult],
    agent_id: Optional[str] = None,
    team_id: Optional[str] = None,
) -> None:
    """Set Darwin-compatible span attributes from native result fields.

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
    span.set_attribute("detection.score", result.detection_score)

    if result.triggered_methods:
        span.set_attribute("detection.methods", result.triggered_methods)
        span.set_attribute("detection.method", result.triggered_methods[0])
