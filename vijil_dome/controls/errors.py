"""Exception types and result handling for control violations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vijil_dome.controls.models import EvaluationResult, SteeringContext

logger = logging.getLogger(__name__)


class ControlError(Exception):
    """Base class for all control exceptions."""

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


class ControlViolationError(ControlError):
    """Raised when a deny control fires in enforce mode."""


class ControlSteerError(ControlError):
    """Raised when a steer control fires, carrying correction guidance."""

    def __init__(
        self,
        message: str,
        *,
        control_name: str,
        steering_context: SteeringContext,
        result: EvaluationResult,
    ) -> None:
        super().__init__(message, control_name=control_name, result=result)
        self.steering_context = steering_context


def handle_result(
    result: EvaluationResult, enforce: bool, stage: str
) -> None:
    """Raise or log based on evaluation outcome.

    Shared by both the ``@control()`` decorator and ``VijilDome`` guard
    methods so the logic is never duplicated.
    """
    if result.action == "deny":
        triggered = next(
            (m for m in result.matches
             if m.triggered and m.action and m.action.decision == "deny"),
            None,
        )
        name = triggered.control_name if triggered else "unknown"
        msg = triggered.message if triggered else "Denied by control"
        if enforce:
            raise ControlViolationError(
                msg, control_name=name, result=result
            )
        logger.warning(
            "[shadow] Control %s would deny at %s stage: %s",
            name, stage, msg,
        )

    if result.action == "steer":
        triggered = next(
            (m for m in result.matches
             if m.triggered and m.action and m.action.decision == "steer"),
            None,
        )
        name = triggered.control_name if triggered else "unknown"
        if result.steering_context is None:
            logger.warning(
                "Control %s returned steer at %s stage but has no "
                "steering_context — treating as allow",
                name, stage,
            )
            return
        if enforce:
            raise ControlSteerError(
                result.steering_context.message,
                control_name=name,
                steering_context=result.steering_context,
                result=result,
            )
        logger.warning(
            "[shadow] Control %s would steer at %s stage: %s",
            name, stage, result.steering_context.message,
        )
