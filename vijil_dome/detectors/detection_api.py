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

"""Detection API contract — shared types for Dome client ↔ inference server.

These Pydantic models define the wire format for remote detection calls.
Both the Dome thin client and the inference server import these types
to guarantee schema compatibility.

The API is designed for batching: a single ``POST /v1/detect`` request
can invoke multiple detectors on the same payload, reducing HTTP
round-trips when Dome runs 3-5 detectors per guard pass.

The wire payload is a ``DomePayload`` — the same envelope Dome uses
internally — so callers don't lose the prompt/response distinction at
the client boundary. Detectors that operate on a single string read
``payload.query_string``; detectors that need the prompt and response
separately (hallucination, fact-check) can read the fields directly.

Thresholds are applied CLIENT-SIDE by Dome from the agent's dome.yaml
config. The server returns raw scores so the decision boundary stays
in the config layer where Darwin can mutate it.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vijil_dome.types import DomePayload


# Resource bounds for the wire format. The server enforces these at
# request-validation time so a single client can't tie up the GPU pool
# with an unbounded detector list or megabyte-sized inputs. Keep these
# in sync with detection-server/detection_api.py on the inference repo —
# both sides assert these constants in their test suites so a one-sided
# change becomes a tripwire rather than silent wire-format drift.
_MAX_DETECTORS_PER_REQUEST = 50
# text and response are bounded tighter (single user input or single
# agent response). prompt allows long upstream conversation history fed
# to context-aware detectors (hallucination, fact-check). Asymmetry
# matches the old input_text=64K / context_text=256K pre-DomePayload
# schema and preserves prior capacity for context-heavy callers.
_MAX_TEXT_CHARS = 64_000
_MAX_PROMPT_CHARS = 256_000
_MAX_RESPONSE_CHARS = 64_000


class DetectorInvocation(BaseModel):
    """A single detector to run on the payload.

    Attributes:
        detector_name: Registry name matching the detector implementation
            on the inference server. Use dome's canonical names from
            ``vijil_dome/detectors/__init__.py`` (e.g.,
            ``prompt-injection-mbert``, ``privacy-presidio``,
            ``policy-gpt-oss-safeguard``).
        config: Detector-specific parameters. Passed through to the
            detector's ``detect()`` method. Common keys include
            ``hub_name``, ``model_name``, ``entity_types``.
    """

    # Reject unknown fields. The two detection_api.py mirrors must stay
    # in lockstep; silently dropping unknown fields would let one side
    # add a field the other can't see and produce a zero-result success
    # path (the cardinal-sin pattern from AUDITING.md).
    model_config = ConfigDict(extra="forbid")

    detector_name: str
    config: dict[str, Any] = Field(default_factory=dict)


_PAYLOAD_FIELD_BOUNDS: dict[str, int] = {
    "text": _MAX_TEXT_CHARS,
    "prompt": _MAX_PROMPT_CHARS,
    "response": _MAX_RESPONSE_CHARS,
}


class DetectRequest(BaseModel):
    """Batch detection request — one or more detectors on the same payload.

    Attributes:
        detectors: List of detectors to run (0..50). Each is dispatched
            independently on the server; results are returned in the
            same order. The upper bound prevents a single client from
            exhausting the GPU pool with an unbounded detector list.
            An empty list is permitted (returns an empty results list).
        payload: The ``DomePayload`` to analyze. ``text`` and ``response``
            are bounded at 64K chars each; ``prompt`` is bounded at 256K
            to accommodate long upstream conversation history fed to
            context-aware detectors (hallucination, fact-check).
            ``DomePayload``'s own validator requires at least one of the
            three to be set and forbids mixing ``text`` with
            ``prompt``/``response``.
    """

    model_config = ConfigDict(extra="forbid")

    detectors: list[DetectorInvocation] = Field(
        ..., max_length=_MAX_DETECTORS_PER_REQUEST,
    )
    payload: DomePayload

    @model_validator(mode="after")
    def _validate_payload_field_bounds(self) -> "DetectRequest":
        # DomePayload doesn't bound field lengths (it's a runtime envelope).
        # Bound them here at the wire boundary so a single client can't
        # send a megabyte-sized field and tie up the inference server.
        for field_name, max_chars in _PAYLOAD_FIELD_BOUNDS.items():
            value = getattr(self.payload, field_name, None)
            if value is not None and len(value) > max_chars:
                raise ValueError(
                    f"payload.{field_name} exceeds {max_chars} "
                    f"chars ({len(value)})"
                )
        return self


class DetectorResult(BaseModel):
    """Result from a single detector invocation.

    Attributes:
        detector_name: Which detector produced this result.
        is_flagged: Whether the detector flagged the input at its
            default threshold. Dome client may override this based
            on dome.yaml threshold config.
        score: Raw detection score, 0.0 (safe) to 1.0 (maximum risk).
            Dome client applies thresholds from config against this value.
        category: Detection category (e.g., ``prompt_injection``,
            ``toxicity``, ``pii``, ``hallucination``).
        details: Detector-specific metadata. For PII detectors, this
            includes entity types found. For LLM detectors, includes
            the model's reasoning.
        latency_ms: Server-side processing time for this detector.
        error: Non-None if the detector failed. The result is still
            returned (with ``is_flagged=False``, ``score=0.0``) so
            the client can proceed with partial results.
    """

    detector_name: str
    is_flagged: bool = False
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    category: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    error: str | None = None


class DetectResponse(BaseModel):
    """Batch detection response — one result per invocation, same order.

    Attributes:
        results: Detection results in the same order as the request's
            ``detectors`` list. Always the same length.
        server_version: Inference server version string for debugging.
        total_latency_ms: Total server-side processing time including
            dispatch overhead.
    """

    results: list[DetectorResult]
    server_version: str = ""
    total_latency_ms: float = 0.0
