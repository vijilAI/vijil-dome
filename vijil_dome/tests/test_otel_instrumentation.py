"""Tests for OTel instrumentation: idempotent instrument_dome (BC-21), attribute truncation (BC-16)."""

from __future__ import annotations

import logging
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
    @pytest.fixture(autouse=True)
    def _reset_ip_cache(self):
        from vijil_dome.instrumentation import logging as dome_logging

        dome_logging._ip_address = None
        yield
        dome_logging._ip_address = None

    def test_unresolvable_hostname_does_not_raise(self) -> None:
        import logging
        import socket

        from vijil_dome.integrations.instrumentation import otel_instrumentation as mod

        record = logging.LogRecord("vijil.dome", logging.WARNING, "f.py", 1, "hello", None, None)
        with patch.object(socket, "gethostbyname", side_effect=socket.gaierror(8, "not known")):
            formatted = mod.get_vijil_log_formatter().format(record)

        assert "resource.service.ip=N/A" in formatted
        assert formatted.endswith("hello")

    def test_hostname_resolved_once(self) -> None:
        import logging
        import socket

        from vijil_dome.integrations.instrumentation import otel_instrumentation as mod

        formatter = mod.get_vijil_log_formatter()
        with patch.object(socket, "gethostbyname", return_value="10.0.0.1") as lookup:
            for _ in range(3):
                formatter.format(logging.LogRecord("vijil.dome", logging.INFO, "f.py", 1, "m", None, None))

        assert lookup.call_count == 1

    def test_failed_lookup_is_retried(self) -> None:
        import socket

        from vijil_dome.instrumentation import logging as dome_logging

        with patch.object(socket, "gethostbyname", side_effect=socket.gaierror(8, "not known")):
            assert dome_logging._get_ip_address() == "N/A"
        with patch.object(socket, "gethostbyname", return_value="10.0.0.1"):
            assert dome_logging._get_ip_address() == "10.0.0.1"


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[tuple[str, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append((record.name, record.levelname))


class TestInstrumentDomeLogLevel:
    @pytest.fixture(autouse=True)
    def _restore_loggers(self):
        names = ("vijil.dome", "vijil_dome", "")
        saved = {
            n: (logging.getLogger(n).level, logging.getLogger(n).propagate, list(logging.getLogger(n).handlers))
            for n in names
        }
        for n in ("vijil.dome", "vijil_dome"):
            logging.getLogger(n).setLevel(logging.NOTSET)
        yield
        for n, (level, propagate, handlers) in saved.items():
            logger = logging.getLogger(n)
            logger.setLevel(level)
            logger.propagate = propagate
            logger.handlers[:] = handlers

    def _instrument(self, handler) -> None:
        from vijil_dome.integrations.instrumentation.otel_instrumentation import (
            instrument_dome,
        )

        dome = MagicMock(spec=[])
        dome.input_guardrail = None
        dome.output_guardrail = None
        dome.enforce = True
        with patch(
            "vijil_dome.integrations.instrumentation.otel_instrumentation.LoggingInstrumentor"
        ):
            instrument_dome(dome, handler=handler, tracer=None, meter=None)

    def _root_console(self, level: int) -> _Capture:
        console = _Capture()
        root = logging.getLogger()
        root.addHandler(console)
        root.setLevel(level)
        return console

    def test_info_exported_but_kept_off_the_app_console(self) -> None:
        console = self._root_console(logging.WARNING)
        exported = _Capture()
        self._instrument(exported)

        logging.getLogger("vijil.dome").info("scan")
        logging.getLogger("vijil.dome").warning("degraded")

        assert exported.records == [("vijil.dome", "INFO"), ("vijil.dome", "WARNING")]
        assert console.records == [("vijil.dome", "WARNING")]

    def test_app_console_at_info_still_sees_info(self) -> None:
        console = self._root_console(logging.INFO)
        self._instrument(_Capture())

        logging.getLogger("vijil.dome").info("scan")

        assert console.records == [("vijil.dome", "INFO")]

    def test_module_loggers_are_exported(self) -> None:
        console = self._root_console(logging.WARNING)
        exported = _Capture()
        self._instrument(exported)

        name = "vijil_dome.detectors.remote_dispatcher"
        logging.getLogger(name).info("dispatch")
        logging.getLogger(name).warning("remote down")

        assert exported.records == [(name, "INFO"), (name, "WARNING")]
        assert console.records == [(name, "WARNING")]

    def test_app_level_preserved(self) -> None:
        logger = logging.getLogger("vijil.dome")
        logger.setLevel(logging.ERROR)
        self._instrument(logging.NullHandler())

        assert logger.level == logging.ERROR
        assert logger.propagate is True


class TestConfigLoggingFormatter:
    def test_unresolvable_hostname_does_not_raise(self) -> None:
        import logging
        import socket

        from vijil_dome.instrumentation import logging as dome_logging

        dome_logging._ip_address = None
        formatter = dome_logging.CustomFormatter("%(ip_address)s %(message)s")
        record = logging.LogRecord("vijil.dome", logging.INFO, "f.py", 1, "hello", None, None)
        with patch.object(socket, "gethostbyname", side_effect=socket.gaierror(8, "not known")):
            formatted = formatter.format(record)

        assert formatted == "N/A hello"
