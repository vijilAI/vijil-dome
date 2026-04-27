"""Guard result models — unified with Dome's guardrail result types.

This module bridges the trust runtime's enforcement context (flagged,
enforced, guarded_response) with Dome's detection output. Rather than
maintaining a parallel set of models, it imports Dome's types and adds
the enforcement layer on top.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class DetectorTrace(BaseModel):
    """Per-detector execution record within a guard pass."""

    detector_name: str
    hit: bool
    score: float
    exec_time_ms: float


class GuardTrace(BaseModel):
    """Per-guard execution record grouping its detector traces."""

    guard_name: str
    detectors: list[DetectorTrace]


class EnforcementResult(BaseModel):
    """Outcome of a Dome guard pass with trust runtime enforcement context.

    This is the trust runtime's view of a guard result. It wraps Dome's
    ``ScanResult`` into a typed model that includes enforcement state
    (was the flagged content blocked or just logged?) and structured
    trace data for audit.

    Named ``EnforcementResult`` to avoid collision with
    ``vijil_dome.guardrails.GuardResult``, which is Dome's per-guard
    detector output. This class adds the enforcement layer.
    """

    flagged: bool
    enforced: bool
    score: float  # detection confidence 0.0–1.0
    guarded_response: str | None
    exec_time_ms: float
    trace: list[GuardTrace]

    @classmethod
    def from_scan_result(cls, scan: Any) -> EnforcementResult:
        """Convert a Dome ``ScanResult`` to an ``EnforcementResult``.

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


# Backwards-compatible alias — trust runtime code uses GuardResult
GuardResult = EnforcementResult
