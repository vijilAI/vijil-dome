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
can invoke multiple detectors on the same input text, reducing HTTP
round-trips when Dome runs 3-5 detectors per guard pass.

Thresholds are applied CLIENT-SIDE by Dome from the agent's dome.yaml
config. The server returns raw scores so the decision boundary stays
in the config layer where Darwin can mutate it.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DetectorInvocation(BaseModel):
    """A single detector to run on the input text.

    Attributes:
        detector_name: Registry name matching the detector implementation
            on the inference server (e.g., ``pi_mbert``, ``pii_presidio``,
            ``gpt_oss_safeguard``).
        config: Detector-specific parameters. Passed through to the
            detector's ``detect()`` method. Common keys include
            ``hub_name``, ``model_name``, ``entity_types``.
    """

    detector_name: str
    config: dict[str, Any] = Field(default_factory=dict)


class DetectRequest(BaseModel):
    """Batch detection request — one or more detectors on the same input.

    Attributes:
        detectors: List of detectors to run. Each is dispatched
            independently on the server; results are returned in the
            same order.
        input_text: The text to analyze (user prompt or agent response).
        context_text: Optional prior conversation context. Used by
            detectors that need dialogue history (e.g., hallucination
            detection compares response against context).
    """

    detectors: list[DetectorInvocation]
    input_text: str
    context_text: str | None = None


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
