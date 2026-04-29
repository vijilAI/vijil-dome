"""Data models for the control engine."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Step(BaseModel):
    """Runtime payload being evaluated by the control engine."""

    type: Literal["tool", "llm"]
    name: str
    input: Any = None
    output: Any = None
    context: dict[str, Any] = Field(default_factory=dict)


class ControlScope(BaseModel):
    """When a control applies."""

    step_types: list[Literal["tool", "llm"]] | None = None
    step_names: list[str] | None = None
    step_name_regex: str | None = None
    stages: list[Literal["pre", "post"]] | None = None


class EvaluatorRef(BaseModel):
    """Reference to an evaluator plus its configuration."""

    name: str
    config: dict[str, Any] = Field(default_factory=dict)


class ConditionNode(BaseModel):
    """Recursive boolean condition tree.

    Leaf nodes have ``selector`` + ``evaluator``.
    Composite nodes use ``and``/``or``/``not`` over child conditions.
    The two forms are mutually exclusive.
    """

    model_config = {"populate_by_name": True}

    # Leaf fields
    selector: str | None = None
    evaluator: EvaluatorRef | None = None

    # Composite fields
    and_: list[ConditionNode] | None = Field(None, alias="and")
    or_: list[ConditionNode] | None = Field(None, alias="or")
    not_: ConditionNode | None = Field(None, alias="not")

    def is_leaf(self) -> bool:
        return self.selector is not None or self.evaluator is not None

    def is_composite(self) -> bool:
        return (
            self.and_ is not None
            or self.or_ is not None
            or self.not_ is not None
        )


class SteeringContext(BaseModel):
    """Correction guidance attached to steer actions."""

    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ControlAction(BaseModel):
    """What to do when a control's condition matches."""

    decision: Literal["deny", "steer", "observe"]
    message: str | None = None
    steering_context: SteeringContext | None = None
    on_error: Literal["fail_open", "fail_closed"] = "fail_closed"


class Control(BaseModel):
    """A single policy control definition."""

    model_config = {"populate_by_name": True}

    name: str
    enabled: bool = True
    scope: ControlScope = Field(default_factory=ControlScope)
    condition: ConditionNode
    action: ControlAction
    priority: int = 100


class ControlMatch(BaseModel):
    """Outcome of evaluating one control against a step."""

    control_name: str
    triggered: bool
    action: ControlAction | None = None
    confidence: float = 1.0
    message: str = ""
    exec_time_ms: float = 0.0
    error: str | None = None


class EvaluationResult(BaseModel):
    """Aggregated result of evaluating all applicable controls."""

    permitted: bool = True
    action: Literal["allow", "deny", "steer"] = "allow"
    confidence: float = 1.0
    matches: list[ControlMatch] = Field(default_factory=list)
    steering_context: SteeringContext | None = None
    exec_time_ms: float = 0.0
