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

"""Remote detector dispatcher — sends detection requests to the inference server.

Replaces in-process model loading and litellm routing with a single
httpx POST to ``DOME_INFERENCE_URL/v1/detect``. The inference server
hosts all model-based detectors (ML classifiers via Ray Serve, LLM
detectors via vLLM).

Configuration:
    DOME_INFERENCE_URL: Base URL of the inference server
        (e.g., ``http://inference-server:8000``).
    DOME_INFERENCE_TIMEOUT: Request timeout in seconds (default: 30).
    DOME_INFERENCE_RETRIES: Number of retries on transient failure (default: 1).

Graceful degradation: if the inference server is unreachable, all remote
detectors return ``is_flagged=False`` with an ``error`` field set. Local
detectors continue to function. Detection is defense-in-depth — a server
outage degrades coverage, not availability.
"""

from __future__ import annotations

import logging
import os
import time

import httpx

from vijil_dome.detectors.detection_api import (
    DetectRequest,
    DetectResponse,
    DetectorInvocation,
    DetectorResult,
)
from vijil_dome.types import DomePayload

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30.0
_DEFAULT_RETRIES = 1


def _parse_env_number(name, default, cast):
    """Read a numeric env var, falling back to default on missing/empty/invalid."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except (ValueError, TypeError):
        logger.warning(
            "RemoteDetectorDispatcher: invalid %s=%r; using default %r",
            name, raw, default,
        )
        return default


class RemoteDetectorDispatcher:
    """Dispatch detection requests to the inference server via httpx.

    Batches multiple detector invocations into a single HTTP request
    to reduce round-trip overhead. Returns results in the same order
    as the input invocations.

    Args:
        inference_url: Base URL of the inference server. Read from
            ``DOME_INFERENCE_URL`` if not provided.
        timeout: HTTP request timeout in seconds.
        retries: Number of retries on transient failure (5xx, timeout).
    """

    def __init__(
        self,
        inference_url: str | None = None,
        timeout: float | None = None,
        retries: int | None = None,
    ) -> None:
        # Use `is not None` so caller-supplied falsy values (timeout=0.0,
        # retries=0, inference_url="") are honored exactly. `.strip()`
        # both branches so whitespace-only configuration always reads
        # as "not configured" instead of producing invalid request URLs.
        if inference_url is not None:
            self._url = inference_url.strip()
        else:
            self._url = os.environ.get("DOME_INFERENCE_URL", "").strip()
        if timeout is not None:
            self._timeout = timeout
        else:
            # Guard against empty-string env var (DOME_INFERENCE_TIMEOUT="" in
            # k8s manifests, shell exports) and non-numeric values like "30s".
            # float("")/int("") and float("30s") raise ValueError; treat any
            # of those as "use the default" with a warning.
            self._timeout = _parse_env_number(
                "DOME_INFERENCE_TIMEOUT", _DEFAULT_TIMEOUT, float,
            )
        if retries is not None:
            self._retries = retries
        else:
            self._retries = _parse_env_number(
                "DOME_INFERENCE_RETRIES", _DEFAULT_RETRIES, int,
            )

    @property
    def is_configured(self) -> bool:
        """True if an inference URL is set."""
        return bool(self._url)

    async def detect(
        self,
        payload: DomePayload | str,
        detectors: list[DetectorInvocation],
    ) -> list[DetectorResult]:
        """Send a batch detection request to the inference server.

        Args:
            payload: The ``DomePayload`` to analyze. A bare ``str`` is
                accepted for ergonomics and coerced to ``DomePayload(text=...)``.
            detectors: List of detectors to invoke.

        Returns:
            List of DetectorResult in the same order as ``detectors``.
            On failure, returns error results for all detectors.
        """
        if not self._url:
            logger.warning(
                "RemoteDetectorDispatcher: DOME_INFERENCE_URL not set, "
                "returning error results (is_flagged=False) for %d detectors",
                len(detectors),
            )
            return self._error_results(detectors, "DOME_INFERENCE_URL not configured")

        request = DetectRequest(
            detectors=detectors,
            payload=DomePayload.coerce(payload),
        )

        start = time.monotonic()
        last_error: str = ""

        for attempt in range(1 + self._retries):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.post(
                        f"{self._url.rstrip('/')}/v1/detect",
                        json=request.model_dump(),
                    )

                if resp.status_code >= 500:
                    last_error = f"Server error {resp.status_code}: {resp.text[:200]}"
                    logger.warning(
                        "RemoteDetectorDispatcher: attempt %d/%d failed: %s",
                        attempt + 1, 1 + self._retries, last_error,
                    )
                    continue

                resp.raise_for_status()
                response = DetectResponse.model_validate(resp.json())

                # Contract: server returns one result per requested detector,
                # in the same order. If that's violated, callers would
                # silently misattribute scores to the wrong detector. Fail
                # the request rather than emit wrong data.
                if len(response.results) != len(detectors):
                    last_error = (
                        f"Server returned {len(response.results)} results "
                        f"for {len(detectors)} detectors (contract violation)"
                    )
                    logger.error(
                        "RemoteDetectorDispatcher: %s", last_error,
                    )
                    return self._error_results(detectors, last_error)

                elapsed = (time.monotonic() - start) * 1000
                logger.info(
                    "RemoteDetectorDispatcher: %d detectors in %.0fms (server: %.0fms)",
                    len(detectors), elapsed, response.total_latency_ms,
                )
                return response.results

            except httpx.TimeoutException:
                last_error = f"Timeout after {self._timeout}s"
                logger.warning(
                    "RemoteDetectorDispatcher: attempt %d/%d timed out",
                    attempt + 1, 1 + self._retries,
                )
            except httpx.ConnectError as e:
                last_error = f"Connection failed: {e}"
                logger.warning(
                    "RemoteDetectorDispatcher: attempt %d/%d connection failed: %s",
                    attempt + 1, 1 + self._retries, e,
                )
                break  # Don't retry connection failures — server is down
            except Exception as e:
                last_error = f"Unexpected error: {e}"
                logger.exception(
                    "RemoteDetectorDispatcher: attempt %d/%d unexpected error",
                    attempt + 1, 1 + self._retries,
                )
                break

        logger.error(
            "RemoteDetectorDispatcher: all attempts failed for %d detectors: %s",
            len(detectors), last_error,
        )
        return self._error_results(detectors, last_error)

    @staticmethod
    def _error_results(
        detectors: list[DetectorInvocation],
        error: str,
    ) -> list[DetectorResult]:
        """Build error results for all detectors when the server is unreachable."""
        return [
            DetectorResult(
                detector_name=d.detector_name,
                is_flagged=False,
                score=0.0,
                category="",
                error=error,
            )
            for d in detectors
        ]
