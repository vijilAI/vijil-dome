"""Tests for the DomeBridgeEvaluator."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from vijil_dome.controls.evaluators.dome_bridge import DomeBridgeEvaluator


class TestDomeBridgeEvaluator:
    @pytest.mark.asyncio
    async def test_threshold_above(self):
        detector = AsyncMock()
        detector.detect.return_value = (True, {"detection_score": 0.9, "response_string": "hit"})

        evaluator = DomeBridgeEvaluator("fake-detector")
        evaluator._detector_instance = detector

        result = await evaluator.evaluate("some text", {"threshold": 0.5})
        assert result.matched is True
        assert result.confidence == 0.9
        assert result.metadata["hit"] is True

    @pytest.mark.asyncio
    async def test_threshold_below(self):
        detector = AsyncMock()
        detector.detect.return_value = (False, {"detection_score": 0.3, "response_string": "miss"})

        evaluator = DomeBridgeEvaluator("fake-detector")
        evaluator._detector_instance = detector

        result = await evaluator.evaluate("safe text", {"threshold": 0.5})
        assert result.matched is False
        assert result.confidence == 0.3

    @pytest.mark.asyncio
    async def test_threshold_exact_boundary(self):
        detector = AsyncMock()
        detector.detect.return_value = (False, {"detection_score": 0.5})

        evaluator = DomeBridgeEvaluator("fake-detector")
        evaluator._detector_instance = detector

        result = await evaluator.evaluate("text", {"threshold": 0.5})
        assert result.matched is True

    @pytest.mark.asyncio
    async def test_threshold_overrides_hit_flag(self):
        """Even when detector returns hit=True, a low score below threshold should not match."""
        detector = AsyncMock()
        detector.detect.return_value = (True, {"detection_score": 0.2})

        evaluator = DomeBridgeEvaluator("fake-detector")
        evaluator._detector_instance = detector

        result = await evaluator.evaluate("text", {"threshold": 0.5})
        assert result.matched is False

    @pytest.mark.asyncio
    async def test_default_threshold(self):
        detector = AsyncMock()
        detector.detect.return_value = (True, {"detection_score": 0.6})

        evaluator = DomeBridgeEvaluator("fake-detector")
        evaluator._detector_instance = detector

        result = await evaluator.evaluate("text", {})
        assert result.matched is True

    @pytest.mark.asyncio
    async def test_score_defaults_zero_when_missing(self):
        """BC-9: When score key is missing, default is 0.0 — no fabrication from hit flag."""
        detector = AsyncMock()
        detector.detect.return_value = (True, {"response_string": "detected"})

        evaluator = DomeBridgeEvaluator("fake-detector")
        evaluator._detector_instance = detector

        result = await evaluator.evaluate("text", {"threshold": 0.5})
        assert result.matched is False
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_score_defaults_zero_on_miss(self):
        detector = AsyncMock()
        detector.detect.return_value = (False, {"response_string": "clean"})

        evaluator = DomeBridgeEvaluator("fake-detector")
        evaluator._detector_instance = detector

        result = await evaluator.evaluate("text", {"threshold": 0.5})
        assert result.matched is False
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_none_value_coerced(self):
        detector = AsyncMock()
        detector.detect.return_value = (False, {"detection_score": 0.1})

        evaluator = DomeBridgeEvaluator("fake-detector")
        evaluator._detector_instance = detector

        result = await evaluator.evaluate(None, {"threshold": 0.5})
        assert result.matched is False
        detector.detect.assert_called_once()

    @pytest.mark.asyncio
    async def test_metadata_includes_details(self):
        detector = AsyncMock()
        detector.detect.return_value = (True, {"detection_score": 0.95, "response_string": "danger"})

        evaluator = DomeBridgeEvaluator("fake-detector")
        evaluator._detector_instance = detector

        result = await evaluator.evaluate("text", {"threshold": 0.5})
        assert result.metadata["detector"] == "fake-detector"
        assert result.metadata["score"] == 0.95
        assert result.metadata["hit"] is True
        assert "details" in result.metadata
        assert result.message == "danger"

    def test_resolve_unknown_detector_raises(self):
        evaluator = DomeBridgeEvaluator("totally-nonexistent-detector-xyz")
        with pytest.raises(ValueError, match="not found"):
            evaluator._resolve_detector()


class TestDomeBridgeResolution:
    def test_dome_prefix_creates_bridge(self):
        from vijil_dome.controls.evaluators import resolve_evaluator

        evaluator = resolve_evaluator("dome:fake-test-name-xyz")
        assert isinstance(evaluator, DomeBridgeEvaluator)
        assert evaluator._detector_name == "fake-test-name-xyz"
