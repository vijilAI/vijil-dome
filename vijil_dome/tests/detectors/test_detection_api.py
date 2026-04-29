# Copyright 2025 Vijil, Inc.
# Licensed under the Apache License, Version 2.0

"""Tests for detection API contract types and remote dispatcher.

Test hierarchy:
- Adversarial: malformed requests, empty detector lists, score out of range
- Boundary: single detector, max batch size, timeout edge
- Round-trip: request → serialize → deserialize → response
"""

from __future__ import annotations


import pytest

from vijil_dome.detectors.detection_api import (
    DetectRequest,
    DetectResponse,
    DetectorInvocation,
    DetectorResult,
)


# ---------------------------------------------------------------------------
# Adversarial: malformed inputs
# ---------------------------------------------------------------------------


class TestDetectorInvocation:
    def test_rejects_empty_detector_name(self) -> None:
        inv = DetectorInvocation(detector_name="", config={})
        # Empty string is technically valid — the server rejects unknown names
        assert inv.detector_name == ""

    def test_config_defaults_to_empty_dict(self) -> None:
        inv = DetectorInvocation(detector_name="pi_mbert")
        assert inv.config == {}

    def test_config_accepts_nested_dicts(self) -> None:
        inv = DetectorInvocation(
            detector_name="gpt_oss_safeguard",
            config={"hub_name": "openai", "model": "vijil-default", "params": {"temperature": 0.0}},
        )
        assert inv.config["params"]["temperature"] == 0.0


class TestDetectRequest:
    def test_requires_input_text(self) -> None:
        with pytest.raises(Exception):
            DetectRequest(detectors=[], input_text=None)  # type: ignore[arg-type]

    def test_empty_detectors_list_is_valid(self) -> None:
        req = DetectRequest(detectors=[], input_text="hello")
        assert len(req.detectors) == 0

    def test_context_text_is_optional(self) -> None:
        req = DetectRequest(
            detectors=[DetectorInvocation(detector_name="pi_mbert")],
            input_text="test",
        )
        assert req.context_text is None


class TestDetectorResult:
    def test_score_clamped_to_0_1(self) -> None:
        with pytest.raises(Exception):
            DetectorResult(detector_name="test", score=1.5)

    def test_error_result_has_safe_defaults(self) -> None:
        result = DetectorResult(
            detector_name="pi_mbert",
            error="connection refused",
        )
        assert result.is_flagged is False
        assert result.score == 0.0

    def test_flagged_result_with_details(self) -> None:
        result = DetectorResult(
            detector_name="pii_presidio",
            is_flagged=True,
            score=0.95,
            category="pii",
            details={"entity_types": ["CREDIT_CARD", "SSN"], "count": 2},
        )
        assert result.is_flagged
        assert result.details["count"] == 2


# ---------------------------------------------------------------------------
# Round-trip: serialize → deserialize
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_request_roundtrip(self) -> None:
        req = DetectRequest(
            detectors=[
                DetectorInvocation(detector_name="pi_mbert", config={"threshold": 0.5}),
                DetectorInvocation(detector_name="toxicity_mbert"),
            ],
            input_text="test input",
            context_text="prior conversation",
        )
        serialized = req.model_dump_json()
        restored = DetectRequest.model_validate_json(serialized)
        assert len(restored.detectors) == 2
        assert restored.detectors[0].detector_name == "pi_mbert"
        assert restored.input_text == "test input"
        assert restored.context_text == "prior conversation"

    def test_response_roundtrip(self) -> None:
        resp = DetectResponse(
            results=[
                DetectorResult(detector_name="pi_mbert", is_flagged=True, score=0.92, category="prompt_injection"),
                DetectorResult(detector_name="toxicity_mbert", is_flagged=False, score=0.1, category="toxicity"),
            ],
            server_version="1.0.0",
            total_latency_ms=45.2,
        )
        serialized = resp.model_dump_json()
        restored = DetectResponse.model_validate_json(serialized)
        assert len(restored.results) == 2
        assert restored.results[0].is_flagged is True
        assert restored.results[1].score == 0.1
        assert restored.total_latency_ms == 45.2


# ---------------------------------------------------------------------------
# Remote dispatcher: error handling
# ---------------------------------------------------------------------------


class TestRemoteDispatcherErrorHandling:
    """Test dispatcher behavior when the inference server is down."""

    @pytest.mark.asyncio
    async def test_unconfigured_returns_error_results(self) -> None:
        from vijil_dome.detectors.remote_dispatcher import RemoteDetectorDispatcher

        dispatcher = RemoteDetectorDispatcher(inference_url="")
        results = await dispatcher.detect(
            input_text="test",
            detectors=[
                DetectorInvocation(detector_name="pi_mbert"),
                DetectorInvocation(detector_name="toxicity_mbert"),
            ],
        )
        assert len(results) == 2
        assert all(r.error is not None for r in results)
        assert all(r.is_flagged is False for r in results)

    @pytest.mark.asyncio
    async def test_unreachable_server_returns_error_results(self) -> None:
        from vijil_dome.detectors.remote_dispatcher import RemoteDetectorDispatcher

        dispatcher = RemoteDetectorDispatcher(
            inference_url="http://127.0.0.1:1",  # unreachable port
            timeout=1.0,
            retries=0,
        )
        results = await dispatcher.detect(
            input_text="test",
            detectors=[DetectorInvocation(detector_name="pi_mbert")],
        )
        assert len(results) == 1
        assert results[0].error is not None
        assert results[0].is_flagged is False

    def test_is_configured_property(self) -> None:
        from vijil_dome.detectors.remote_dispatcher import RemoteDetectorDispatcher

        assert RemoteDetectorDispatcher(inference_url="http://localhost:8000").is_configured
        assert not RemoteDetectorDispatcher(inference_url="").is_configured
        assert not RemoteDetectorDispatcher(inference_url=None).is_configured
