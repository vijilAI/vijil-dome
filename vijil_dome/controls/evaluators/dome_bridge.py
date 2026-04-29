"""Bridge evaluator that wraps Dome detectors as control evaluators."""

from __future__ import annotations

import logging
from typing import Any

from vijil_dome.controls.evaluators.base import Evaluator, EvaluatorResult

logger = logging.getLogger(__name__)


class DomeBridgeEvaluator(Evaluator):
    """Wraps any Dome detector as a control evaluator.

    Created dynamically by :func:`resolve_evaluator` for names starting
    with ``dome:``.  The detector name after the prefix is looked up in
    Dome's ``method_registry`` across all categories.

    Config keys:
        threshold (float): Minimum detection score to consider a match
            (default ``0.5``).
        detector_kwargs (dict): Extra keyword arguments forwarded to the
            detector constructor.
    """

    def __init__(self, detector_name: str) -> None:
        self._detector_name = detector_name
        self._detector_instance: Any | None = None

    def _resolve_detector(self, detector_kwargs: dict[str, Any] | None = None):
        if self._detector_instance is not None:
            return self._detector_instance

        from vijil_dome.detectors import method_registry, DetectionCategory

        for category in DetectionCategory:
            methods = method_registry.get(category, {})
            if self._detector_name in methods:
                cls = methods[self._detector_name]
                self._detector_instance = cls(**(detector_kwargs or {}))
                return self._detector_instance

        available: list[str] = []
        for cat_methods in method_registry.values():
            available.extend(cat_methods.keys())
        raise ValueError(
            f"Dome detector {self._detector_name!r} not found. "
            f"Available: {sorted(set(available))}"
        )

    async def evaluate(
        self, value: Any, config: dict[str, Any]
    ) -> EvaluatorResult:
        threshold = config.get("threshold", 0.5)
        detector_kwargs = config.get("detector_kwargs", {})

        detector = self._resolve_detector(detector_kwargs)

        from vijil_dome.types import DomePayload

        text = str(value) if value is not None else ""
        payload = DomePayload.coerce(text)

        result = await detector.detect(payload)
        hit: bool = result[0]
        data: dict[str, Any] = dict(result[1])
        score = data.get("detection_score", 1.0 if hit else 0.0)

        matched = score >= threshold if not hit else hit

        return EvaluatorResult(
            matched=matched,
            confidence=float(score),
            message=data.get("response_string", ""),
            metadata={
                "detector": self._detector_name,
                "score": score,
                "hit": hit,
                "details": data,
            },
        )
