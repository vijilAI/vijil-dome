"""CEL expression evaluator."""

from __future__ import annotations

import logging
from typing import Any

from vijil_dome.controls.evaluators import register_evaluator
from vijil_dome.controls.evaluators.base import Evaluator, EvaluatorResult

logger = logging.getLogger(__name__)

try:
    import celpy  # type: ignore[import-untyped]
    from celpy import celtypes  # type: ignore[import-untyped]

    _HAS_CEL = True
except ImportError:
    celpy = None  # type: ignore[assignment]
    celtypes = None  # type: ignore[assignment]
    _HAS_CEL = False


def _to_cel_value(obj: Any) -> Any:
    """Convert a Python value to a CEL-compatible type."""
    if celtypes is None:
        raise RuntimeError("cel-python not installed")
    if obj is None:
        return None
    if isinstance(obj, bool):
        return celtypes.BoolType(obj)
    if isinstance(obj, int):
        return celtypes.IntType(obj)
    if isinstance(obj, float):
        return celtypes.DoubleType(obj)
    if isinstance(obj, str):
        return celtypes.StringType(obj)
    if isinstance(obj, list):
        return celtypes.ListType([_to_cel_value(v) for v in obj])
    if isinstance(obj, dict):
        return celtypes.MapType(
            {_to_cel_value(k): _to_cel_value(v) for k, v in obj.items()}
        )
    return celtypes.StringType(str(obj))


@register_evaluator("cel")
class CelEvaluator(Evaluator):
    """Evaluate a CEL expression against the extracted value.

    Requires the ``cel-python`` package.  Install with::

        pip install cel-python

    The expression receives ``value`` as the primary variable.
    For dict values, all top-level keys are also available as variables.

    Config keys:
        expression (str): A CEL expression that should evaluate to a
            boolean (``True`` = matched).
    """

    async def evaluate(
        self, value: Any, config: dict[str, Any]
    ) -> EvaluatorResult:
        if not _HAS_CEL:
            raise RuntimeError(
                "cel-python is required for the cel evaluator. "
                "Install with: pip install cel-python"
            )

        expression = config.get("expression")
        if not expression:
            return EvaluatorResult(matched=False, message="No expression provided")

        env = celpy.Environment()
        ast = env.compile(expression)
        prog = env.program(ast)

        activation = {"value": _to_cel_value(value)}
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(k, str):
                    activation[k] = _to_cel_value(v)

        result = prog.evaluate(activation)

        if isinstance(result, celtypes.BoolType):
            matched = bool(result)
        else:
            matched = bool(result)

        return EvaluatorResult(
            matched=matched,
            confidence=1.0,
            message=f"CEL: {expression} = {matched}",
            metadata={"expression": expression},
        )
