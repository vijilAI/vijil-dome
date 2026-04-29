"""JSON Schema validation evaluator."""

from __future__ import annotations

import logging
from typing import Any

from vijil_dome.controls.evaluators import register_evaluator
from vijil_dome.controls.evaluators.base import Evaluator, EvaluatorResult

logger = logging.getLogger(__name__)

try:
    import jsonschema as _jsonschema  # type: ignore[import-untyped]

    _HAS_JSONSCHEMA = True
except ImportError:
    _jsonschema = None  # type: ignore[assignment]
    _HAS_JSONSCHEMA = False


@register_evaluator("json_schema")
class JsonSchemaEvaluator(Evaluator):
    """Validate a value against a JSON Schema.

    Requires the ``jsonschema`` package.  Install with::

        pip install jsonschema

    Config keys:
        schema (dict): A complete JSON Schema object.
        negate (bool): If ``True``, matched when validation *fails*
            (default ``False`` — matched when validation passes).
    """

    async def evaluate(
        self, value: Any, config: dict[str, Any]
    ) -> EvaluatorResult:
        if not _HAS_JSONSCHEMA:
            raise RuntimeError(
                "jsonschema is required for the json_schema evaluator. "
                "Install with: pip install jsonschema"
            )

        schema = config.get("schema")
        if schema is None:
            return EvaluatorResult(matched=False, message="No schema provided")

        negate = config.get("negate", False)

        try:
            _jsonschema.validate(instance=value, schema=schema)
            valid = True
            errors: list[str] = []
        except _jsonschema.ValidationError as exc:
            valid = False
            errors = [exc.message]

        raw_matched = valid
        matched = (not raw_matched) if negate else raw_matched

        return EvaluatorResult(
            matched=matched,
            confidence=1.0,
            message="; ".join(errors) if errors else "Schema valid",
            metadata={"valid": valid, "errors": errors},
        )
