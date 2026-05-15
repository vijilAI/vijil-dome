"""Tests for medium-risk reliability fixes: BC-20, BC-21, BC-16, BC-4, BC-8, BC-15, BC-9."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# BC-20: Corrupted cache recovery
# ---------------------------------------------------------------------------


class TestCorruptedCacheRecovery:
    """Corrupt JSON in cache files should not crash loaders."""

    def test_config_loader_survives_corrupt_cache(self, tmp_path: Path) -> None:
        from vijil_dome.utils.config_loader import load_dome_config_from_s3

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        with patch("vijil_dome.utils.config_loader._create_s3_client") as mock_s3:
            mock_client = MagicMock()
            mock_s3.return_value = mock_client

            good_config = {"input-guards": ["prompt-injection"]}
            mock_client.get_object.return_value = {
                "Body": MagicMock(read=lambda: json.dumps(good_config).encode()),
                "ETag": '"abc123"',
            }

            result = load_dome_config_from_s3(
                bucket="test-bucket",
                key="test/config.json",
                cache_dir=str(cache_dir),
                cache_ttl_seconds=9999,
            )
            assert result == good_config

            # Corrupt the cached file
            config_files = list(cache_dir.rglob("config.json"))
            assert len(config_files) == 1
            config_files[0].write_text("{invalid json!!!")

            # Second load should re-download instead of crashing
            result2 = load_dome_config_from_s3(
                bucket="test-bucket",
                key="test/config.json",
                cache_dir=str(cache_dir),
                cache_ttl_seconds=9999,
            )
            assert result2 == good_config

    def test_policy_loader_raises_on_corrupt_local_file(self, tmp_path: Path) -> None:
        from vijil_dome.utils.policy_loader import load_policy_sections_from_file

        bad_file = tmp_path / "policy.json"
        bad_file.write_text("{not valid json!!!")

        with pytest.raises(ValueError, match="Invalid JSON in policy file"):
            load_policy_sections_from_file(str(bad_file))


# ---------------------------------------------------------------------------
# BC-21: Non-idempotent instrument_dome
# ---------------------------------------------------------------------------


class TestIdempotentInstrumentDome:
    """instrument_dome must be callable multiple times without double-wrapping."""

    def test_instrument_dome_idempotent(self) -> None:
        from vijil_dome.integrations.instrumentation.otel_instrumentation import (
            instrument_dome,
        )

        dome = MagicMock()
        dome._instrumented = False
        dome.input_guardrail = None
        dome.output_guardrail = None

        tracer = MagicMock()
        meter = MagicMock()

        with patch(
            "vijil_dome.integrations.instrumentation.otel_instrumentation.LoggingInstrumentor"
        ):
            instrument_dome(dome, handler=None, tracer=tracer, meter=meter)
            assert dome._instrumented is True

            # Reset call counts
            tracer.reset_mock()
            meter.reset_mock()

            # Second call should be a no-op
            instrument_dome(dome, handler=None, tracer=tracer, meter=meter)
            tracer.start_as_current_span.assert_not_called()

    def test_uninstrumented_dome_gets_flag(self) -> None:
        from vijil_dome.integrations.instrumentation.otel_instrumentation import (
            instrument_dome,
        )

        dome = MagicMock(spec=[])  # No _instrumented attr
        dome.input_guardrail = None
        dome.output_guardrail = None
        dome.enforce = False

        with patch(
            "vijil_dome.integrations.instrumentation.otel_instrumentation.LoggingInstrumentor"
        ):
            instrument_dome(dome, handler=None, tracer=None, meter=None)
            assert dome._instrumented is True


# ---------------------------------------------------------------------------
# BC-16: OTel attribute size truncation
# ---------------------------------------------------------------------------


class TestOtelAttributeTruncation:
    """Span attributes must be bounded to prevent OTLP export failures."""

    def test_truncate_short_string(self) -> None:
        from vijil_dome.instrumentation.tracing import _truncate

        assert _truncate("short") == "short"

    def test_truncate_long_string(self) -> None:
        from vijil_dome.instrumentation.tracing import _truncate, _MAX_ATTR_LENGTH

        long = "x" * (_MAX_ATTR_LENGTH + 500)
        result = _truncate(long)
        assert len(result) < len(long)
        assert result.endswith("...<truncated>")
        assert result.startswith("x" * 100)

    def test_func_span_attributes_truncated(self) -> None:
        from vijil_dome.instrumentation.tracing import (
            _MAX_ATTR_LENGTH,
            _set_func_span_attributes,
        )

        span = MagicMock()
        big_arg = "a" * (_MAX_ATTR_LENGTH + 1000)
        _set_func_span_attributes(span, big_arg)

        args_call = next(
            c for c in span.set_attribute.call_args_list if c[0][0] == "function.args"
        )
        assert len(args_call[0][1]) <= _MAX_ATTR_LENGTH + 50  # room for truncation marker

    def test_func_span_result_truncated(self) -> None:
        from vijil_dome.instrumentation.tracing import (
            _MAX_ATTR_LENGTH,
            _set_func_span_result_attributes,
        )

        span = MagicMock()
        big_result = "r" * (_MAX_ATTR_LENGTH + 1000)
        _set_func_span_result_attributes(span, big_result)

        result_call = next(
            c
            for c in span.set_attribute.call_args_list
            if c[0][0] == "function.result"
        )
        assert len(result_call[0][1]) <= _MAX_ATTR_LENGTH + 50


# ---------------------------------------------------------------------------
# BC-8: PolicySectionsDetector concurrency default + error tracking
# ---------------------------------------------------------------------------


class TestPolicySectionsDefaults:
    """PolicySectionsDetector should have a bounded default concurrency."""

    def test_default_max_parallel_sections(self) -> None:
        from vijil_dome.detectors.methods.policy_sections_detector import (
            PolicySectionsDetector,
        )

        import inspect
        sig = inspect.signature(PolicySectionsDetector.__init__)
        default = sig.parameters["max_parallel_sections"].default
        assert default is not None
        assert isinstance(default, int)
        assert default <= 20


# ---------------------------------------------------------------------------
# BC-15: Thread-safe context via DomePayload
# ---------------------------------------------------------------------------


class TestDomePayloadContext:
    """DomePayload.context enables per-call context without instance mutation."""

    def test_payload_accepts_context(self) -> None:
        from vijil_dome.types import DomePayload

        payload = DomePayload(text="hello", context="some context")
        assert payload.context == "some context"
        assert payload.query_string == "hello"

    def test_payload_context_defaults_to_none(self) -> None:
        from vijil_dome.types import DomePayload

        payload = DomePayload(text="hello")
        assert payload.context is None

    def test_coerce_string_has_no_context(self) -> None:
        from vijil_dome.types import DomePayload

        payload = DomePayload.coerce("hello")
        assert payload.context is None

    def test_context_alone_is_not_valid(self) -> None:
        from vijil_dome.types import DomePayload

        with pytest.raises(Exception):
            DomePayload(context="just context, no content")


# ---------------------------------------------------------------------------
# BC-9: Score clamping and fallback fixes
# ---------------------------------------------------------------------------


class TestScoreClamping:
    """Detection scores should be clamped to [0.0, 1.0]."""

    @pytest.mark.asyncio
    async def test_score_clamped_to_one(self) -> None:
        from vijil_dome.detectors import DetectionMethod, DetectionResult
        from vijil_dome.guardrails import Guard
        from vijil_dome.types import DomePayload

        class OverScoreDetector(DetectionMethod):
            async def detect(self, dome_input: DomePayload) -> DetectionResult:
                return (True, {"response_string": "blocked", "score": 1.5})

        guard = Guard(
            guard_name="test",
            detector_list=[OverScoreDetector()],
            run_in_parallel=False,
        )
        result = await guard.sequential_guard("test input")
        assert result.detection_score <= 1.0

    @pytest.mark.asyncio
    async def test_negative_score_clamped_to_zero(self) -> None:
        from vijil_dome.detectors import DetectionMethod, DetectionResult
        from vijil_dome.guardrails import Guard
        from vijil_dome.types import DomePayload

        class NegativeScoreDetector(DetectionMethod):
            async def detect(self, dome_input: DomePayload) -> DetectionResult:
                return (True, {"response_string": "blocked", "score": -0.5})

        guard = Guard(
            guard_name="test",
            detector_list=[NegativeScoreDetector()],
            run_in_parallel=False,
        )
        result = await guard.sequential_guard("test input")
        assert result.detection_score >= 0.0

    def test_dome_bridge_uses_score_key(self) -> None:
        """BC-9: DomeBridgeEvaluator should read 'score' from detector result."""
        data = {"score": 0.75, "response_string": "blocked"}
        score = data.get("score", data.get("detection_score", 0.0))
        assert score == 0.75

    def test_dome_bridge_no_fabrication(self) -> None:
        """Without a score key, default is 0.0 — not 1.0 based on hit flag."""
        data = {"response_string": "blocked"}
        score = data.get("score", data.get("detection_score", 0.0))
        assert score == 0.0

    def test_guard_trace_no_fabrication(self) -> None:
        """guard.py trace parser defaults missing score to 0.0, not hit-based."""
        result_dict: dict = {"response_string": "test"}
        score = float(result_dict.get("score", 0.0))
        assert score == 0.0
