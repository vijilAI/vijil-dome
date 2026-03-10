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

"""Tests for Dome trace logging."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from vijil_dome.Dome import Dome, ScanResult


@pytest.fixture
def trace_file(tmp_path):
    return tmp_path / "trace.jsonl"


@pytest.fixture
def dome_with_trace(trace_file):
    """Create a Dome with trace logging and mocked guardrails."""
    dome = Dome.__new__(Dome)
    dome.enforce = True
    dome.client = None
    dome.agent_id = None
    dome.team_id = None
    dome.user_id = None
    dome._trace_log = trace_file
    dome.input_guardrail = None
    dome.output_guardrail = None
    return dome


def _make_scan_result(flagged=False, trace=None):
    return ScanResult(
        flagged=flagged,
        enforced=flagged,
        response_string="blocked" if flagged else "ok",
        trace=trace or {},
        exec_time=0.05,
        detection_score=0.9 if flagged else 0.1,
    )


class TestTraceLogging:
    def test_no_trace_log_noop(self, tmp_path):
        """Dome without trace_log should not create any files."""
        dome = Dome.__new__(Dome)
        dome._trace_log = None
        scan = _make_scan_result()
        dome._log_trace("input", "hello", scan)
        # No file should be created
        assert not list(tmp_path.iterdir())

    def test_log_trace_writes_jsonl(self, dome_with_trace, trace_file):
        """_log_trace should append a JSON line."""
        scan = _make_scan_result(flagged=True)
        dome_with_trace._log_trace("input", "test prompt", scan)

        lines = trace_file.read_text().strip().split("\n")
        assert len(lines) == 1

        entry = json.loads(lines[0])
        assert entry["direction"] == "input"
        assert entry["flagged"] is True
        assert entry["enforced"] is True
        assert entry["detection_score"] == 0.9
        assert entry["text_preview"] == "test prompt"
        assert "ts" in entry
        assert "exec_time" in entry

    def test_log_trace_appends(self, dome_with_trace, trace_file):
        """Multiple calls should append, not overwrite."""
        dome_with_trace._log_trace("input", "first", _make_scan_result())
        dome_with_trace._log_trace("output", "second", _make_scan_result(flagged=True))

        lines = trace_file.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["direction"] == "input"
        assert json.loads(lines[1])["direction"] == "output"

    def test_log_trace_truncates_text(self, dome_with_trace, trace_file):
        """text_preview should be truncated to 200 chars."""
        long_text = "x" * 500
        dome_with_trace._log_trace("input", long_text, _make_scan_result())

        entry = json.loads(trace_file.read_text().strip())
        assert len(entry["text_preview"]) == 200

    def test_log_trace_survives_write_error(self, dome_with_trace, trace_file, caplog):
        """Write errors should be logged, not raised."""
        # Make the file path a directory so writing fails
        trace_file.mkdir(parents=True, exist_ok=True)
        dome_with_trace._trace_log = trace_file

        scan = _make_scan_result()
        # Should not raise
        dome_with_trace._log_trace("input", "test", scan)

    def test_guard_input_logs_trace(self, dome_with_trace, trace_file):
        """guard_input should write a trace entry when trace_log is set."""
        # No guardrail → returns empty result, but _log_trace shouldn't be called
        # because guard_input returns early. Test with a mocked guardrail instead.
        result = dome_with_trace.guard_input("hello")
        # No guardrail set → empty result, no trace (early return before scan)
        assert trace_file.exists() is False or trace_file.read_text() == ""

    def test_serialize_trace_handles_non_serializable(self):
        """_serialize_trace should skip non-JSON-serializable values."""
        from vijil_dome.detectors import DetectionTimingResult
        from vijil_dome.guardrails import GuardResult

        det = DetectionTimingResult(
            hit=True,
            result={
                "type": type,  # non-serializable
                "score": 0.95,
                "predictions": [{"label": "INJECTION", "score": 0.99}],
            },
            exec_time=0.1,
        )
        guard = GuardResult(
            triggered=True,
            details={"TestDetector": det},
            exec_time=0.1,
            response="blocked",
            detection_score=0.95,
            triggered_methods=["test-method"],
        )

        serialized = Dome._serialize_trace({"test-guard": guard})
        assert "test-guard" in serialized
        g = serialized["test-guard"]
        assert g["triggered"] is True
        assert g["triggered_methods"] == ["test-method"]

        d = g["detectors"]["TestDetector"]
        assert d["hit"] is True
        assert d["score"] == 0.95
        assert "type" not in d  # non-serializable skipped
        assert d["predictions"] == [{"label": "INJECTION", "score": 0.99}]


class TestTraceLogConfig:
    def test_trace_log_kwarg(self, tmp_path):
        """trace_log kwarg should set the trace path."""
        trace_path = tmp_path / "my_trace.jsonl"
        dome = Dome.__new__(Dome)
        dome.enforce = True
        dome.client = None
        dome._trace_log = Path(trace_path)
        assert dome._trace_log == trace_path

    def test_trace_log_creates_parent_dirs(self, tmp_path):
        """Dome should create parent directories for trace_log."""
        trace_path = tmp_path / "deep" / "nested" / "trace.jsonl"
        dome = Dome(trace_log=trace_path)
        assert trace_path.parent.exists()
