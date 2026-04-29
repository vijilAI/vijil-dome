"""VijilDome — policy-driven runtime for agent controls.

The policy is the single config surface.  All detection — ML models,
regex, custom evaluators — is defined in the policy controls.  Dome
detectors are referenced via ``dome:*`` evaluators.

The existing :class:`~vijil_dome.Dome.Dome` class is unchanged and
continues to work for standalone content guardrailing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Literal

from vijil_dome.controls.decorator import control as control_decorator
from vijil_dome.controls.engine import ControlEngine
from vijil_dome.controls.errors import ControlSteerError, ControlViolationError
from vijil_dome.controls.models import Control, EvaluationResult, Step

logger = logging.getLogger(__name__)


class VijilDome:
    """Policy-driven runtime.

    All detection — ML models, regex, custom evaluators — is defined
    in the policy controls.  Dome detectors are referenced via
    ``dome:*`` evaluators within the policy.

    Parameters
    ----------
    policy:
        Controls definition.  Can be:
        - A list of :class:`Control` objects or dicts
        - A path (str or Path) to a YAML/JSON file
        - ``None`` (empty policy — controls can be added later)
    enforce:
        If ``True`` (default), deny/steer actions raise exceptions.
        If ``False``, violations are logged but not enforced (shadow mode).
    agent_id:
        Optional agent identifier for audit/logging.
    """

    def __init__(
        self,
        *,
        policy: list[dict | Control] | str | Path | None = None,
        enforce: bool = True,
        agent_id: str | None = None,
    ) -> None:
        self._enforce = enforce
        self._agent_id = agent_id
        self._engine = ControlEngine()

        if isinstance(policy, (str, Path)):
            self._engine.load_controls_from_file(str(policy))
        elif isinstance(policy, list):
            defs = [
                c if isinstance(c, dict) else c.model_dump(by_alias=True)
                for c in policy
            ]
            self._engine.load_controls(defs)

    @property
    def engine(self) -> ControlEngine:
        """The underlying control engine."""
        return self._engine

    @property
    def agent_id(self) -> str | None:
        return self._agent_id

    @property
    def enforce(self) -> bool:
        return self._enforce

    # ------------------------------------------------------------------
    # Guard methods
    # ------------------------------------------------------------------

    def guard_input(
        self,
        text: str | dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
        step_name: str = "input",
    ) -> EvaluationResult:
        """Evaluate input against the policy (sync)."""
        step = Step(
            type="llm",
            name=step_name,
            input=text,
            context=context or {},
        )
        result = self._engine.evaluate_sync(step, stage="pre")
        self._maybe_raise(result, "pre")
        return result

    async def async_guard_input(
        self,
        text: str | dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
        step_name: str = "input",
    ) -> EvaluationResult:
        """Evaluate input against the policy (async)."""
        step = Step(
            type="llm",
            name=step_name,
            input=text,
            context=context or {},
        )
        result = await self._engine.evaluate(step, stage="pre")
        self._maybe_raise(result, "pre")
        return result

    def guard_output(
        self,
        text: str | dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
        step_name: str = "output",
    ) -> EvaluationResult:
        """Evaluate output against the policy (sync)."""
        step = Step(
            type="llm",
            name=step_name,
            output=text,
            context=context or {},
        )
        result = self._engine.evaluate_sync(step, stage="post")
        self._maybe_raise(result, "post")
        return result

    async def async_guard_output(
        self,
        text: str | dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
        step_name: str = "output",
    ) -> EvaluationResult:
        """Evaluate output against the policy (async)."""
        step = Step(
            type="llm",
            name=step_name,
            output=text,
            context=context or {},
        )
        result = await self._engine.evaluate(step, stage="post")
        self._maybe_raise(result, "post")
        return result

    def guard_tool_call(
        self,
        tool_name: str,
        tool_input: Any = None,
        *,
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate a tool call against the policy (sync)."""
        step = Step(
            type="tool",
            name=tool_name,
            input=tool_input,
            context=context or {},
        )
        result = self._engine.evaluate_sync(step, stage="pre")
        self._maybe_raise(result, "pre")
        return result

    async def async_guard_tool_call(
        self,
        tool_name: str,
        tool_input: Any = None,
        *,
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate a tool call against the policy (async)."""
        step = Step(
            type="tool",
            name=tool_name,
            input=tool_input,
            context=context or {},
        )
        result = await self._engine.evaluate(step, stage="pre")
        self._maybe_raise(result, "pre")
        return result

    # ------------------------------------------------------------------
    # Decorator
    # ------------------------------------------------------------------

    def control(
        self,
        *,
        step_type: Literal["tool", "llm"] = "llm",
        step_name: str | None = None,
        input_mapper: Callable | None = None,
        output_mapper: Callable | None = None,
        context_mapper: Callable | None = None,
    ) -> Callable:
        """Return a decorator that uses this instance's engine and enforce mode."""
        return control_decorator(
            engine=self._engine,
            step_type=step_type,
            step_name=step_name,
            enforce=self._enforce,
            input_mapper=input_mapper,
            output_mapper=output_mapper,
            context_mapper=context_mapper,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _maybe_raise(self, result: EvaluationResult, stage: str) -> None:
        if result.action == "deny":
            triggered = next(
                (m for m in result.matches if m.triggered and m.action and m.action.decision == "deny"),
                None,
            )
            name = triggered.control_name if triggered else "unknown"
            msg = triggered.message if triggered else "Denied by control"
            if self._enforce:
                raise ControlViolationError(
                    msg, control_name=name, result=result
                )
            logger.warning(
                "[shadow] Control %s would deny at %s stage: %s",
                name, stage, msg,
            )

        if result.action == "steer" and result.steering_context:
            triggered = next(
                (m for m in result.matches if m.triggered and m.action and m.action.decision == "steer"),
                None,
            )
            name = triggered.control_name if triggered else "unknown"
            if self._enforce:
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
