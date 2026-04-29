"""Value-in-list matching evaluator."""

from __future__ import annotations

from typing import Any

from vijil_dome.controls.evaluators import register_evaluator
from vijil_dome.controls.evaluators.base import Evaluator, EvaluatorResult


def _normalize(text: str, case_sensitive: bool) -> str:
    return text if case_sensitive else text.lower()


def _check_one(
    text: str, target: str, mode: str, case_sensitive: bool
) -> bool:
    t = _normalize(text, case_sensitive)
    v = _normalize(target, case_sensitive)
    if mode == "exact":
        return t == v
    if mode == "contains":
        return v in t
    if mode == "starts_with":
        return t.startswith(v)
    if mode == "ends_with":
        return t.endswith(v)
    return t == v


@register_evaluator("list")
class ListEvaluator(Evaluator):
    """Evaluate whether a value matches entries in a list.

    Config keys:
        values (list[str]): The list of values to check against.
        match_mode (str): ``"exact"`` (default), ``"contains"``,
            ``"starts_with"``, ``"ends_with"``.
        case_sensitive (bool): Default ``True``.
        logic (str): ``"any"`` (default) or ``"all"``.
        negate (bool): If ``True``, matched when value is NOT in list.
    """

    async def evaluate(
        self, value: Any, config: dict[str, Any]
    ) -> EvaluatorResult:
        values: list[str] = config.get("values", [])
        if not values:
            return EvaluatorResult(matched=False, message="No values provided")

        text = str(value) if value is not None else ""
        mode = config.get("match_mode", "exact")
        case_sensitive = config.get("case_sensitive", True)
        logic = config.get("logic", "any")
        negate = config.get("negate", False)

        results = [_check_one(text, v, mode, case_sensitive) for v in values]

        if logic == "all":
            raw_matched = all(results)
        else:
            raw_matched = any(results)

        matched = (not raw_matched) if negate else raw_matched
        matched_values = [v for v, r in zip(values, results) if r]

        return EvaluatorResult(
            matched=matched,
            confidence=1.0,
            message=f"Matched values: {matched_values}" if matched_values else "No match",
            metadata={"matched_values": matched_values, "mode": mode, "logic": logic},
        )
