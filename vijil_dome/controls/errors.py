"""Exception types for control violations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vijil_dome.controls.models import EvaluationResult, SteeringContext


class ControlViolationError(Exception):
    """Raised when a deny control fires in enforce mode."""

    def __init__(
        self,
        message: str,
        *,
        control_name: str,
        result: EvaluationResult,
    ) -> None:
        super().__init__(message)
        self.control_name = control_name
        self.result = result


class ControlSteerError(Exception):
    """Raised when a steer control fires, carrying correction guidance."""

    def __init__(
        self,
        message: str,
        *,
        control_name: str,
        steering_context: SteeringContext,
        result: EvaluationResult,
    ) -> None:
        super().__init__(message)
        self.control_name = control_name
        self.steering_context = steering_context
        self.result = result
