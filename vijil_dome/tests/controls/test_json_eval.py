"""Tests for JSON Schema evaluator."""

import pytest

try:
    import jsonschema  # noqa: F401

    _HAS_JSONSCHEMA = True
except ImportError:
    _HAS_JSONSCHEMA = False

pytestmark = pytest.mark.skipif(
    not _HAS_JSONSCHEMA, reason="jsonschema not installed"
)

from vijil_dome.controls.evaluators import resolve_evaluator


class TestJsonSchemaEvaluator:
    @pytest.fixture
    def evaluator(self):
        return resolve_evaluator("json_schema")

    @pytest.mark.asyncio
    async def test_valid_object(self, evaluator):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name"],
        }
        result = await evaluator.evaluate(
            {"name": "Alice", "age": 30}, {"schema": schema}
        )
        assert result.matched is True

    @pytest.mark.asyncio
    async def test_invalid_object(self, evaluator):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        result = await evaluator.evaluate(
            {"age": 30}, {"schema": schema}
        )
        assert result.matched is False

    @pytest.mark.asyncio
    async def test_negate(self, evaluator):
        schema = {"type": "string"}
        result = await evaluator.evaluate(
            42, {"schema": schema, "negate": True}
        )
        assert result.matched is True

    @pytest.mark.asyncio
    async def test_no_schema(self, evaluator):
        result = await evaluator.evaluate("test", {})
        assert result.matched is False

    @pytest.mark.asyncio
    async def test_string_validation(self, evaluator):
        schema = {"type": "string", "minLength": 3}
        result = await evaluator.evaluate("ab", {"schema": schema})
        assert result.matched is False
        result = await evaluator.evaluate("abc", {"schema": schema})
        assert result.matched is True
