"""Data models for the control engine."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field, model_validator


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

    early_exit: bool = Field(
        False,
        alias="early_exit",
        description=(
            "When True, and/or children are evaluated sequentially and "
            "short-circuit on the first decisive result. When False "
            "(default), all children run in parallel."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_and_validate(cls, data: Any) -> Any:
        """Normalize AC selector format and validate mutual exclusivity."""
        if isinstance(data, dict):
            sel = data.get("selector")
            if isinstance(sel, dict) and "path" in sel:
                data = {**data, "selector": sel["path"]}

            has_leaf = data.get("selector") is not None or data.get("evaluator") is not None
            has_composite = (
                data.get("and") is not None
                or data.get("and_") is not None
                or data.get("or") is not None
                or data.get("or_") is not None
                or data.get("not") is not None
                or data.get("not_") is not None
            )
            if has_leaf and has_composite:
                raise ValueError(
                    "ConditionNode cannot have both leaf fields (selector/evaluator) "
                    "and composite fields (and/or/not)"
                )

            if has_leaf:
                has_selector = data.get("selector") is not None
                has_evaluator = data.get("evaluator") is not None
                if has_selector != has_evaluator:
                    missing = "evaluator" if has_selector else "selector"
                    raise ValueError(
                        f"Leaf ConditionNode requires both 'selector' and "
                        f"'evaluator', but '{missing}' is missing"
                    )

            for key in ("and", "and_"):
                if isinstance(data.get(key), list) and len(data[key]) == 0:
                    raise ValueError("'and' must contain at least one condition")
            for key in ("or", "or_"):
                if isinstance(data.get(key), list) and len(data[key]) == 0:
                    raise ValueError("'or' must contain at least one condition")

        return data

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

    model_config = {"populate_by_name": True, "extra": "allow"}

    name: str
    description: str | None = None
    enabled: bool = True
    scope: ControlScope = Field(default_factory=ControlScope)
    condition: ConditionNode
    action: ControlAction
    priority: int = 100
    tags: list[str] = Field(default_factory=list)
    annotations: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Vendor and extension metadata. Keys should use reverse-DNS "
            "form (e.g. 'vijil.ai/trust-delta'). Values are preserved "
            "verbatim through parse and serialize."
        ),
    )


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

    action: Literal["allow", "deny", "steer"] = "allow"
    confidence: float = 1.0
    matches: list[ControlMatch] = Field(default_factory=list)
    steering_context: SteeringContext | None = None
    exec_time_ms: float = 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def permitted(self) -> bool:
        return self.action != "deny"
