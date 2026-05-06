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
        # AC compat: json_schema key and field_constraints use inverted
        # semantics — match on VIOLATION (constraint broken → trigger),
        # not on valid.  Our "schema" key matches on valid by default.
        ac_mode = False
        if schema is None:
            schema = config.get("json_schema")
            if schema is not None:
                ac_mode = True
        if schema is None:
            schema = _build_schema_from_constraints(config)
            if schema is not None:
                ac_mode = True
        if schema is None:
            return EvaluatorResult(matched=False, message="No schema provided")

        negate_default = ac_mode
        negate = config.get("negate", negate_default)

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


def _build_schema_from_constraints(
    config: dict[str, Any],
) -> dict[str, Any] | None:
    """Build a JSON Schema from AC-style ``field_constraints``.

    AC's ``field_constraints`` maps field names to constraint dicts with
    keys like ``type``, ``min``, ``max``, ``enum``, ``min_length``,
    ``max_length``.
    """
    field_constraints = config.get("field_constraints")
    if not field_constraints:
        return None

    properties: dict[str, Any] = {}
    for field, constraints in field_constraints.items():
        prop: dict[str, Any] = {}
        if "type" in constraints:
            prop["type"] = constraints["type"]
        if "min" in constraints:
            prop["minimum"] = constraints["min"]
        if "max" in constraints:
            prop["maximum"] = constraints["max"]
        if "enum" in constraints:
            prop["enum"] = constraints["enum"]
        if "min_length" in constraints:
            prop["minLength"] = constraints["min_length"]
        if "max_length" in constraints:
            prop["maxLength"] = constraints["max_length"]
        properties[field] = prop

    return {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
    }
