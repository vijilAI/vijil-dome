"""Tests for OTel instrumentation: idempotent instrument_dome (BC-21), attribute truncation (BC-16)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


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
