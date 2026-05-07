"""Tests for the CEL expression evaluator."""

from __future__ import annotations

import pytest

try:
    import celpy  # noqa: F401

    _HAS_CEL = True
except ImportError:
    _HAS_CEL = False

pytestmark = pytest.mark.skipif(not _HAS_CEL, reason="cel-python not installed")

from vijil_dome.controls.evaluators.expr import CelEvaluator  # noqa: E402


class TestCelEvaluator:
    @pytest.mark.asyncio
    async def test_simple_comparison(self):
        evaluator = CelEvaluator()
        result = await evaluator.evaluate(100, {"expression": "value > 50"})
        assert result.matched is True

    @pytest.mark.asyncio
    async def test_comparison_false(self):
        evaluator = CelEvaluator()
        result = await evaluator.evaluate(10, {"expression": "value > 50"})
        assert result.matched is False

    @pytest.mark.asyncio
    async def test_string_contains(self):
        evaluator = CelEvaluator()
        result = await evaluator.evaluate(
            "hello world", {"expression": 'value.contains("world")'}
        )
        assert result.matched is True

    @pytest.mark.asyncio
    async def test_string_not_contains(self):
        evaluator = CelEvaluator()
        result = await evaluator.evaluate(
            "hello world", {"expression": 'value.contains("xyz")'}
        )
        assert result.matched is False

    @pytest.mark.asyncio
    async def test_dict_value_access(self):
        evaluator = CelEvaluator()
        result = await evaluator.evaluate(
            {"amount": 50000, "currency": "USD"},
            {"expression": "amount > 10000"},
        )
        assert result.matched is True

    @pytest.mark.asyncio
    async def test_dict_value_key_protected(self):
        """A dict key named 'value' should not overwrite the value variable."""
        evaluator = CelEvaluator()
        data = {"value": "injected", "real_field": "test"}
        result = await evaluator.evaluate(
            data, {"expression": 'value.real_field == "test"'}
        )
        assert result.matched is True

    @pytest.mark.asyncio
    async def test_boolean_value(self):
        evaluator = CelEvaluator()
        result = await evaluator.evaluate(True, {"expression": "value == true"})
        assert result.matched is True

    @pytest.mark.asyncio
    async def test_list_size(self):
        evaluator = CelEvaluator()
        result = await evaluator.evaluate(
            [1, 2, 3], {"expression": "value.size() > 2"}
        )
        assert result.matched is True

    @pytest.mark.asyncio
    async def test_no_expression(self):
        evaluator = CelEvaluator()
        result = await evaluator.evaluate("anything", {})
        assert result.matched is False
        assert "No expression" in result.message

    @pytest.mark.asyncio
    async def test_metadata_includes_expression(self):
        evaluator = CelEvaluator()
        result = await evaluator.evaluate(42, {"expression": "value > 0"})
        assert result.metadata["expression"] == "value > 0"
        assert "CEL:" in result.message

    @pytest.mark.asyncio
    async def test_equality(self):
        evaluator = CelEvaluator()
        result = await evaluator.evaluate("admin", {"expression": 'value == "admin"'})
        assert result.matched is True

    @pytest.mark.asyncio
    async def test_float_comparison(self):
        evaluator = CelEvaluator()
        result = await evaluator.evaluate(3.14, {"expression": "value > 3.0"})
        assert result.matched is True
