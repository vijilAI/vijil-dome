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
    ``RemoteDetectionMethod(detector_name="prompt-injection-mbert",
    threshold=0.5)``. The guard pipeline treats it identically. Use
    dome's canonical detector names from ``vijil_dome/detectors/__init__.py``;
    the inference server registers handlers under the same strings.
"""

from __future__ import annotations

import logging
from typing import Any

from vijil_dome.detectors import DetectionMethod, DetectionResult
from vijil_dome.detectors.detection_api import DetectorInvocation
from vijil_dome.detectors.remote_dispatcher import RemoteDetectorDispatcher
from vijil_dome.types import DomePayload

logger = logging.getLogger(__name__)

# Shared dispatcher singleton across all RemoteDetectionMethod instances.
# Initialized lazily on first use. Note: the dispatcher currently spins
# up a fresh httpx.AsyncClient per request (no connection pooling) — a
# follow-up PR will add a long-lived AsyncClient on this singleton.
_dispatcher: RemoteDetectorDispatcher | None = None

# Used only when neither the guard config nor the server supplies a threshold:
# servers that predate detector manifests send no details["threshold"].
_LEGACY_THRESHOLD = 0.5


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
        threshold: Score threshold for flagging; a score at or above it
            produces ``hit=True``. ``None`` (the default) uses the server's
            ``details["threshold"]``, the detector model's own operating point
            from its vijil_detector.json manifest, and 0.5 only when the
            server sends none (VIJIL-1670).
        config: Additional detector-specific config passed to the server.
        blocked_response: Response string when the detector flags input.
    """

    # Don't run in a thread executor — we're async-native via httpx
    run_in_executor = False

    def __init__(
        self,
        detector_name: str,
        threshold: float | None = None,
        config: dict[str, Any] | None = None,
        blocked_response: str = "Blocked by guardrail",
    ) -> None:
        if threshold is not None and not _is_threshold(threshold):
            raise ValueError(
                f"threshold for {detector_name} must be a number in (0, 1], got {threshold!r}"
            )
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

        # Per detection_api.py contract: thresholds apply client-side.
        # Don't include self._threshold in the payload; the server returns
        # raw scores and this class compares against the threshold below.
        invocation = DetectorInvocation(
            detector_name=self._detector_name,
            config=dict(self._config),
        )

        # Send the DomePayload over the wire intact — no client-side
        # flattening. Detectors that need prompt+response separately can
        # read them on the server; flat-text detectors use payload.query_string.
        results = await dispatcher.detect(
            payload=dome_input,
            detectors=[invocation],
        )

        # response_string for the dome trace falls back to the flattened
        # payload when the detector doesn't override it (no hit path).
        flat_text = dome_input.query_string

        if not results:
            logger.warning("RemoteDetectionMethod: no results for %s", self._detector_name)
            return (False, {
                "response_string": flat_text,
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
                "response_string": flat_text,
                "score": 0.0,
                "detector_name": self._detector_name,
                "error": result.error,
            })

        # A malformed details["threshold"] raises ValueError, deliberately not
        # the soft error dict above: detect_with_time records a raise as
        # label="error", so it reaches errored_methods under every policy,
        # like a transport failure. A soft error would read as a clean scan.
        threshold = _resolve_threshold(self._threshold, result.details)
        hit = result.score >= threshold
        response_string = self._blocked_response if hit else flat_text

        return (hit, {
            "response_string": response_string,
            "score": result.score,
            "category": result.category,
            "detector_name": self._detector_name,
            "details": result.details,
            "latency_ms": result.latency_ms,
        })


def _is_threshold(value: Any) -> bool:
    """True for a real number in (0, 1]. Excludes bool, which is an int subtype."""
    # NaN fails both comparisons, so it is refused too.
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and 0.0 < value <= 1.0
    )


def _resolve_threshold(configured: float | None, details: dict[str, Any]) -> float:
    """The threshold to judge a server score against.

    The configured threshold wins; otherwise the server's
    ``details["threshold"]`` applies; otherwise the legacy 0.5. A
    ``details["threshold"]`` that is present but invalid raises ValueError:
    falling back to a default there would judge the score against a number the
    model never specified.
    """
    if configured is not None:
        return configured
    if "threshold" not in details:
        return _LEGACY_THRESHOLD
    sent = details["threshold"]
    if not _is_threshold(sent):
        raise ValueError(f"server details.threshold {sent!r} is not a number in (0, 1]")
    return float(sent)
