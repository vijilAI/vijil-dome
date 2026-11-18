"""Tests for OTel instrumentation: idempotent instrument_dome (BC-21), attribute truncation (BC-16)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


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
        assert len(result) <= _MAX_ATTR_LENGTH
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
        assert len(args_call[0][1]) <= _MAX_ATTR_LENGTH  # room for truncation marker

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
        assert len(result_call[0][1]) <= _MAX_ATTR_LENGTH


class TestVijilLogFormatter:
    def test_unresolvable_hostname_does_not_raise(self) -> None:
        import logging
        import socket

        from vijil_dome.integrations.instrumentation import otel_instrumentation as mod

        mod._host_ip.cache_clear()
        record = logging.LogRecord("vijil.dome", logging.WARNING, "f.py", 1, "hello", None, None)
        with patch.object(socket, "gethostbyname", side_effect=socket.gaierror(8, "not known")):
            formatted = mod.get_vijil_log_formatter().format(record)
        mod._host_ip.cache_clear()

        assert "resource.service.ip=N/A" in formatted
        assert formatted.endswith("hello")

    def test_hostname_resolved_once(self) -> None:
        import logging
        import socket

        from vijil_dome.integrations.instrumentation import otel_instrumentation as mod

        mod._host_ip.cache_clear()
        formatter = mod.get_vijil_log_formatter()
        with patch.object(socket, "gethostbyname", return_value="10.0.0.1") as lookup:
            for _ in range(3):
                formatter.format(logging.LogRecord("vijil.dome", logging.INFO, "f.py", 1, "m", None, None))
        mod._host_ip.cache_clear()

        assert lookup.call_count == 1


class TestInstrumentDomeLogLevel:
    def _instrument(self, handler) -> None:
        from vijil_dome.integrations.instrumentation.otel_instrumentation import (
            instrument_dome,
        )

        dome = MagicMock(spec=[])
        dome.input_guardrail = None
        dome.output_guardrail = None
        with patch(
            "vijil_dome.integrations.instrumentation.otel_instrumentation.LoggingInstrumentor"
        ):
            instrument_dome(dome, handler=handler, tracer=None, meter=None)

    def test_info_enabled_when_level_unset(self) -> None:
        import logging

        logger = logging.getLogger("vijil.dome")
        handler = logging.NullHandler()
        original = logger.level
        logger.setLevel(logging.NOTSET)
        try:
            self._instrument(handler)
            assert logger.level == logging.INFO
        finally:
            logger.removeHandler(handler)
            logger.setLevel(original)

    def test_app_level_preserved(self) -> None:
        import logging

        logger = logging.getLogger("vijil.dome")
        handler = logging.NullHandler()
        original = logger.level
        logger.setLevel(logging.ERROR)
        try:
            self._instrument(handler)
            assert logger.level == logging.ERROR
        finally:
            logger.removeHandler(handler)
            logger.setLevel(original)


class TestConfigLoggingFormatter:
    def test_unresolvable_hostname_does_not_raise(self) -> None:
        import logging
        import socket

        from vijil_dome.instrumentation import logging as dome_logging

        dome_logging._get_ip_address.cache_clear()
        formatter = dome_logging.CustomFormatter("%(ip_address)s %(message)s")
        record = logging.LogRecord("vijil.dome", logging.INFO, "f.py", 1, "hello", None, None)
        with patch.object(socket, "gethostbyname", side_effect=socket.gaierror(8, "not known")):
            formatted = formatter.format(record)
        dome_logging._get_ip_address.cache_clear()

        assert formatted == "N/A hello"
