"""Guard result models wrapping Dome's ScanResult output."""

from __future__ import annotations

from typing import Any

from vijil_dome.trust.models import TrustModel


class DetectorTrace(TrustModel):
    """Per-detector execution record within a guard pass."""

    detector_name: str
    hit: bool
    score: float
    exec_time_ms: float


class GuardTrace(TrustModel):
    """Per-guard execution record grouping its detector traces."""

    guard_name: str
    detectors: list[DetectorTrace]


class GuardResult(TrustModel):
    """Outcome of a single Dome guard pass.

    Wraps ``ScanResult`` from the vijil-dome library into a typed,
    SDK-portable model.  Construct directly or via ``from_scan_result``.
    """

    flagged: bool
    enforced: bool
    score: float  # detection confidence 0.0–1.0
    guarded_response: str | None
    exec_time_ms: float
    trace: list[GuardTrace]

    @classmethod
    def from_scan_result(cls, scan: Any) -> GuardResult:
        """Convert a Dome ``ScanResult`` to a ``GuardResult``.

        Dome's trace structure::

            {
                guard_name: {
                    detector_name: DetectionTimingResult(
                        hit=bool,
                        result={"score": float, ...},
                        exec_time=float,   # seconds
                    )
                }
            }

        ``exec_time`` values are in seconds; we convert to milliseconds.
        """
        trace: list[GuardTrace] = []
        raw_trace = getattr(scan, "trace", None) or {}

        try:
            if isinstance(raw_trace, dict):
                for guard_name, detectors_map in raw_trace.items():
                    detector_traces: list[DetectorTrace] = []
                    if isinstance(detectors_map, dict):
                        for detector_name, timing in detectors_map.items():
                            hit = getattr(timing, "hit", False)
                            result_data = getattr(timing, "result", {})
                            result_dict = result_data if isinstance(result_data, dict) else {}
                            score = float(result_dict.get("score", 1.0 if hit else 0.0))
                            exec_time = float(getattr(timing, "exec_time", 0.0))
                            detector_traces.append(
                                DetectorTrace(
                                    detector_name=str(detector_name),
                                    hit=bool(hit),
                                    score=score,
                                    exec_time_ms=exec_time * 1000,
                                )
                            )
                    trace.append(GuardTrace(guard_name=str(guard_name), detectors=detector_traces))
        except Exception:
            # If trace parsing fails, continue with empty trace — do not
            # let trace parsing errors block guard enforcement.
            trace = []

        flagged: bool = scan.flagged
        guarded_response: str | None = scan.response_string if flagged else None

        return cls(
            flagged=flagged,
            enforced=scan.enforced,
            score=scan.detection_score,
            guarded_response=guarded_response,
            exec_time_ms=scan.exec_time * 1000,
            trace=trace,
        )
