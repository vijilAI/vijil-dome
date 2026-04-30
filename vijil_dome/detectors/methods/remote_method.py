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

"""Remote detection method — delegates detection to the inference server.

Drop-in replacement for any local DetectionMethod. The guard pipeline
calls ``detect()`` on this class the same way it calls local detectors.
Internally, it sends the request to the inference server via
``RemoteDetectorDispatcher`` and translates the ``DetectorResult`` back
into the ``(hit, data)`` tuple the pipeline expects.

Usage in config parser:
    Instead of instantiating ``PIHFMbert(**kwargs)``, instantiate
    ``RemoteDetectionMethod(detector_name="pi_mbert", threshold=0.5)``.
    The guard pipeline treats it identically.
"""

from __future__ import annotations

import logging
from typing import Any

from vijil_dome.detectors import DetectionMethod, DetectionResult
from vijil_dome.detectors.detection_api import DetectorInvocation
from vijil_dome.detectors.remote_dispatcher import RemoteDetectorDispatcher
from vijil_dome.types import DomePayload

logger = logging.getLogger(__name__)

# Shared dispatcher singleton — one HTTP client for all remote detectors.
# Initialized lazily on first use.
_dispatcher: RemoteDetectorDispatcher | None = None


def _get_dispatcher() -> RemoteDetectorDispatcher:
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = RemoteDetectorDispatcher()
    return _dispatcher


class RemoteDetectionMethod(DetectionMethod):
    """DetectionMethod that delegates to the inference server.

    Implements the same ``detect(dome_input) -> (hit, data)`` interface
    as local detectors, but sends the input to the inference server
    via httpx instead of running a model locally.

    Args:
        detector_name: Server-side detector name (e.g., ``pi_mbert``).
        threshold: Score threshold for flagging. Scores above this
            value produce ``hit=True``.
        config: Additional detector-specific config passed to the server.
        blocked_response: Response string when the detector flags input.
    """

    # Don't run in a thread executor — we're async-native via httpx
    run_in_executor = False

    def __init__(
        self,
        detector_name: str,
        threshold: float = 0.5,
        config: dict[str, Any] | None = None,
        blocked_response: str = "Blocked by guardrail",
    ) -> None:
        self._detector_name = detector_name
        self._threshold = threshold
        self._config = config or {}
        self._blocked_response = blocked_response

    async def detect(self, dome_input: DomePayload) -> DetectionResult:
        """Send detection request to the inference server.

        Returns:
            Tuple of (hit: bool, data: dict) matching the DetectionMethod
            contract. ``data`` always includes ``response_string``,
            ``score``, and ``detector_name`` so downstream tracing can
            attribute the result regardless of which return path fires.
        """
        dispatcher = _get_dispatcher()
        input_text = dome_input.query_string

        # Per detection_api.py contract: thresholds apply client-side.
        # Don't include self._threshold in the payload; the server returns
        # raw scores and this class compares against the threshold below.
        invocation = DetectorInvocation(
            detector_name=self._detector_name,
            config=dict(self._config),
        )

        # Context text: use response field if available (output guard scenario)
        context_text = dome_input.prompt if dome_input.response else None

        results = await dispatcher.detect(
            input_text=input_text,
            detectors=[invocation],
            context_text=context_text,
        )

        if not results:
            logger.warning("RemoteDetectionMethod: no results for %s", self._detector_name)
            return (False, {
                "response_string": input_text,
                "score": 0.0,
                "detector_name": self._detector_name,
            })

        result = results[0]

        if result.error:
            logger.warning(
                "RemoteDetectionMethod: %s returned error: %s",
                self._detector_name, result.error,
            )
            return (False, {
                "response_string": input_text,
                "score": 0.0,
                "detector_name": self._detector_name,
                "error": result.error,
            })

        hit = result.score >= self._threshold
        response_string = self._blocked_response if hit else input_text

        return (hit, {
            "response_string": response_string,
            "score": result.score,
            "category": result.category,
            "detector_name": self._detector_name,
            "details": result.details,
            "latency_ms": result.latency_ms,
        })
