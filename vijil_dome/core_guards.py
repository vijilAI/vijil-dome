"""Maps guardrail outcomes onto the controls engine's result types."""

from __future__ import annotations

from typing import Literal

from vijil_dome.controls.models import ControlAction, ControlMatch, EvaluationResult
from vijil_dome.guardrails import GuardrailResult

Level = Literal["input", "output"]


def text_after_guards(result: GuardrailResult, original: str) -> str:
    """Text to hand downstream: never the blocked message, keeps earlier masking."""
    if not result.flagged:
        return result.guardrail_response_message
    rewritten = [
        guard.response
        for guard in result.guard_exec_details.values()
        if not guard.triggered and guard.response != original
    ]
    return rewritten[-1] if rewritten else original


def guard_matches(result: GuardrailResult, level: Level) -> list[ControlMatch]:
    """One deny match per triggered guard, or a fail-closed match if none triggered."""
    if not result.flagged:
        return []

    def match(name: str, score: float, methods: list[str], exec_ms: float) -> ControlMatch:
        return ControlMatch(
            control_name=name,
            triggered=True,
            action=ControlAction(decision="deny"),
            message=result.guardrail_response_message,
            confidence=score,
            exec_time_ms=exec_ms,
            metadata={
                "source": "guard",
                "detection_score": score,
                "triggered_methods": methods,
                "errored_methods": result.errored_methods,
            },
        )

    matches = [
        match(name, g.detection_score, g.triggered_methods, g.exec_time * 1000)
        for name, g in result.guard_exec_details.items()
        if g.triggered
    ]
    if matches:
        return matches
    # Flagged with no triggered guard means the pipeline failed closed.
    return [
        match(
            f"{level}-guardrail",
            1.0,
            result.triggered_methods,
            result.exec_time * 1000,
        )
    ]


def merge_results(
    level: Level,
    original: str,
    guard_result: GuardrailResult | None,
    controls: EvaluationResult | None,
) -> EvaluationResult:
    """Combine guard and controls outcomes; a deny from either side wins."""
    if guard_result is None and controls is not None:
        return controls
    guard_hits = guard_matches(guard_result, level) if guard_result else []
    control_matches = controls.matches if controls else []
    matches = guard_hits + control_matches

    if guard_hits or (controls and controls.action == "deny"):
        action: Literal["allow", "deny", "steer"] = "deny"
        denies = [
            m.confidence
            for m in matches
            if m.triggered and m.action and m.action.decision == "deny"
        ]
        confidence = max(denies, default=1.0)
    else:
        action = controls.action if controls else "allow"
        confidence = controls.confidence if controls else 1.0

    return EvaluationResult(
        action=action,
        confidence=confidence,
        matches=matches,
        steering_context=controls.steering_context if controls else None,
        exec_time_ms=(guard_result.exec_time * 1000 if guard_result else 0.0)
        + (controls.exec_time_ms if controls else 0.0),
        guarded_text=text_after_guards(guard_result, original) if guard_result else None,
        guard_result=guard_result,
    )
