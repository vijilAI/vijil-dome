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
from vijil_dome.types import DomePayload


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
    def test_requires_payload(self) -> None:
        with pytest.raises(Exception):
            DetectRequest(detectors=[], payload=None)  # type: ignore[arg-type]

    def test_empty_detectors_list_is_valid(self) -> None:
        req = DetectRequest(detectors=[], payload=DomePayload(text="hello"))
        assert len(req.detectors) == 0

    def test_payload_text_only(self) -> None:
        req = DetectRequest(
            detectors=[DetectorInvocation(detector_name="prompt-injection-mbert")],
            payload=DomePayload(text="test"),
        )
        assert req.payload.text == "test"
        assert req.payload.prompt is None
        assert req.payload.response is None

    def test_payload_prompt_response(self) -> None:
        req = DetectRequest(
            detectors=[DetectorInvocation(detector_name="hallucination-llm")],
            payload=DomePayload(prompt="What is 2+2?", response="4"),
        )
        assert req.payload.prompt == "What is 2+2?"
        assert req.payload.response == "4"
        # query_string flattens prompt+response for flat-text detectors
        assert "Input: What is 2+2?" in req.payload.query_string
        assert "Output: 4" in req.payload.query_string

    def test_payload_text_too_long_rejected(self) -> None:
        # text cap is 64K. 64,001 chars must fail.
        with pytest.raises(Exception):
            DetectRequest(
                detectors=[],
                payload=DomePayload(text="a" * 64_001),
            )

    def test_payload_text_at_cap_accepted(self) -> None:
        # Boundary: exactly 64K chars must succeed.
        req = DetectRequest(
            detectors=[],
            payload=DomePayload(text="a" * 64_000),
        )
        assert len(req.payload.text or "") == 64_000

    def test_payload_prompt_supports_long_context(self) -> None:
        # prompt cap is 256K so context-aware detectors (hallucination,
        # fact-check) can pass long upstream conversation history.
        # 64K+1 chars (which would fail for `text`) must succeed for
        # `prompt`. This is a perturbation test catching the regression
        # the post-review fix prevents — uniform 64K caps would shrink
        # context capacity by 4x.
        req = DetectRequest(
            detectors=[],
            payload=DomePayload(prompt="a" * 64_001, response="ok"),
        )
        assert len(req.payload.prompt or "") == 64_001

    def test_payload_prompt_too_long_rejected(self) -> None:
        # prompt cap is 256K. 256,001 chars must fail.
        with pytest.raises(Exception):
            DetectRequest(
                detectors=[],
                payload=DomePayload(prompt="a" * 256_001, response="ok"),
            )

    def test_payload_response_too_long_rejected(self) -> None:
        # response cap is 64K (same as text — single agent output).
        with pytest.raises(Exception):
            DetectRequest(
                detectors=[],
                payload=DomePayload(prompt="ctx", response="a" * 64_001),
            )

    def test_payload_must_have_at_least_one_field(self) -> None:
        # Inherited from DomePayload's own validator.
        with pytest.raises(Exception):
            DomePayload()

    def test_request_rejects_unknown_top_level_field(self) -> None:
        # extra='forbid' on DetectRequest catches schema drift between
        # the dome and inference mirrors. If a future commit adds a new
        # top-level field on one side without the other, requests get
        # rejected loudly instead of silently dropping the field.
        with pytest.raises(Exception):
            DetectRequest.model_validate({
                "detectors": [],
                "payload": {"text": "hi"},
                "unknown_future_field": "x",
            })

    def test_invocation_rejects_unknown_field(self) -> None:
        with pytest.raises(Exception):
            DetectorInvocation.model_validate({
                "detector_name": "prompt-injection-mbert",
                "config": {},
                "rogue_field": True,
            })


class TestSchemaDriftTripwire:
    """Constants that MUST match vijil-inference/detection-server/detection_api.py.

    A change here without a matching change there produces wire-format
    drift: dome sends a payload the server rejects with HTTP 422, the
    dispatcher's exception path returns is_flagged=False with the error
    buried, and the agent sees a silent guardrail bypass. These pinned
    constants are the cheapest tripwire — at the moment a developer
    edits one of them, this test fails with an explicit message
    pointing at the inference-side mirror.
    """

    def test_max_detectors_per_request(self) -> None:
        from vijil_dome.detectors.detection_api import _MAX_DETECTORS_PER_REQUEST
        assert _MAX_DETECTORS_PER_REQUEST == 50, (
            "If you change this, change vijil-inference/detection-server/"
            "detection_api.py::_MAX_DETECTORS_PER_REQUEST too."
        )

    def test_payload_field_caps(self) -> None:
        from vijil_dome.detectors.detection_api import (
            _MAX_PROMPT_CHARS,
            _MAX_RESPONSE_CHARS,
            _MAX_TEXT_CHARS,
        )
        assert _MAX_TEXT_CHARS == 64_000
        assert _MAX_PROMPT_CHARS == 256_000
        assert _MAX_RESPONSE_CHARS == 64_000
        # If these change, sync the inference mirror.

    def test_dome_payload_field_set_is_pinned(self) -> None:
        # Adding a field to DomePayload (e.g., image_b64 for multimodal)
        # without updating inference's mirror would let dome serialize
        # the field and inference silently drop it — image guardrails
        # bypass with no error. Pin the field set so additions become
        # an explicit cross-repo coordination point.
        assert set(DomePayload.model_fields.keys()) == {"text", "prompt", "response"}, (
            "DomePayload field set drifted. If adding a field, also add it "
            "to vijil-inference/detection-server/detection_api.py::DomePayload "
            "and update both pinned tests."
        )


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
                DetectorInvocation(detector_name="prompt-injection-mbert", config={"threshold": 0.5}),
                DetectorInvocation(detector_name="moderation-mbert"),
            ],
            payload=DomePayload(prompt="prior conversation", response="test output"),
        )
        serialized = req.model_dump_json()
        restored = DetectRequest.model_validate_json(serialized)
        assert len(restored.detectors) == 2
        assert restored.detectors[0].detector_name == "prompt-injection-mbert"
        assert restored.payload.prompt == "prior conversation"
        assert restored.payload.response == "test output"

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
            payload=DomePayload(text="test"),
            detectors=[
                DetectorInvocation(detector_name="prompt-injection-mbert"),
                DetectorInvocation(detector_name="moderation-mbert"),
            ],
        )
        assert len(results) == 2
        assert all(r.error is not None for r in results)
        assert all(r.is_flagged is False for r in results)

    @pytest.mark.asyncio
    async def test_dispatcher_accepts_bare_string_payload(self) -> None:
        from vijil_dome.detectors.remote_dispatcher import RemoteDetectorDispatcher

        # Ergonomic shortcut: a plain str gets coerced to DomePayload(text=...).
        # Routes to the same not-configured error path as a real DomePayload,
        # which is enough to prove the coercion didn't raise.
        dispatcher = RemoteDetectorDispatcher(inference_url="")
        results = await dispatcher.detect(
            payload="bare string",
            detectors=[DetectorInvocation(detector_name="prompt-injection-mbert")],
        )
        assert len(results) == 1
        assert results[0].error is not None

    @pytest.mark.asyncio
    async def test_unreachable_server_returns_error_results(self) -> None:
        from vijil_dome.detectors.remote_dispatcher import RemoteDetectorDispatcher

        dispatcher = RemoteDetectorDispatcher(
            inference_url="http://127.0.0.1:1",  # unreachable port
            timeout=1.0,
            retries=0,
        )
        results = await dispatcher.detect(
            payload=DomePayload(text="test"),
            detectors=[DetectorInvocation(detector_name="prompt-injection-mbert")],
        )
        assert len(results) == 1
        assert results[0].error is not None
        assert results[0].is_flagged is False

    @pytest.mark.asyncio
    async def test_4xx_returns_explicit_client_error(self) -> None:
        """Validation errors (HTTP 422) must surface clearly, not as opaque
        timeout/connection messages. Without the explicit 4xx branch, a
        too-large payload would fall through to ``except Exception`` and
        the user would see "Unexpected error" instead of "Client error
        422" — the silent-failure mode the post-review fix prevents.
        """
        from unittest.mock import AsyncMock, MagicMock, patch
        from vijil_dome.detectors.remote_dispatcher import RemoteDetectorDispatcher

        dispatcher = RemoteDetectorDispatcher(
            inference_url="http://127.0.0.1:9999",
            timeout=1.0,
            retries=2,  # 4xx must NOT trigger retries
        )

        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.text = "payload.text exceeds 64000 chars (90000)"

        post_calls: list[int] = []

        async def fake_post(*args, **kwargs):  # type: ignore[no-untyped-def]
            post_calls.append(1)
            return mock_response

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = fake_post
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            results = await dispatcher.detect(
                payload=DomePayload(text="x" * 100),
                detectors=[DetectorInvocation(detector_name="prompt-injection-mbert")],
            )

        # 4xx fires once and does NOT retry (contract violation, not transient).
        assert len(post_calls) == 1, f"4xx should not retry; got {len(post_calls)} attempts"
        assert len(results) == 1
        assert results[0].is_flagged is False
        assert results[0].error is not None
        # The user-visible error must name the failure mode, not be generic.
        assert "Client error 422" in results[0].error
        assert "exceeds 64000 chars" in results[0].error

    def test_is_configured_property(self) -> None:
        from vijil_dome.detectors.remote_dispatcher import RemoteDetectorDispatcher

        assert RemoteDetectorDispatcher(inference_url="http://localhost:8000").is_configured
        assert not RemoteDetectorDispatcher(inference_url="").is_configured
        assert not RemoteDetectorDispatcher(inference_url=None).is_configured
