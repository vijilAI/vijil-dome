"""Tests for selector path resolution."""

from vijil_dome.controls.models import Step
from vijil_dome.controls.selectors import MISSING, resolve


def _make_step(**kwargs):
    defaults = {"type": "tool", "name": "test"}
    defaults.update(kwargs)
    return Step(**defaults)


class TestResolve:
    def test_wildcard(self):
        step = _make_step(input={"x": 1})
        result = resolve(step, "*")
        assert isinstance(result, dict)
        assert result["input"] == {"x": 1}
        assert result["name"] == "test"

    def test_top_level_field(self):
        step = _make_step()
        assert resolve(step, "name") == "test"
        assert resolve(step, "type") == "tool"

    def test_input(self):
        step = _make_step(input={"query": "hello"})
        assert resolve(step, "input") == {"query": "hello"}

    def test_nested_input(self):
        step = _make_step(input={"user": {"name": "alice", "age": 30}})
        assert resolve(step, "input.user.name") == "alice"
        assert resolve(step, "input.user.age") == 30

    def test_context(self):
        step = _make_step(context={"user_role": "admin"})
        assert resolve(step, "context.user_role") == "admin"

    def test_output(self):
        step = _make_step(output="response text")
        assert resolve(step, "output") == "response text"

    def test_missing_path(self):
        step = _make_step(input={"x": 1})
        assert resolve(step, "input.nonexistent") is MISSING

    def test_missing_nested(self):
        step = _make_step(input={"a": {"b": 1}})
        assert resolve(step, "input.a.c") is MISSING

    def test_none_input(self):
        step = _make_step(input=None)
        assert resolve(step, "input.x") is MISSING

    def test_array_index(self):
        step = _make_step(input={"items": ["a", "b", "c"]})
        assert resolve(step, "input.items[0]") == "a"
        assert resolve(step, "input.items[2]") == "c"

    def test_array_index_out_of_bounds(self):
        step = _make_step(input={"items": ["a"]})
        assert resolve(step, "input.items[5]") is MISSING

    def test_array_index_nested(self):
        step = _make_step(
            input={"tools": [{"name": "search"}, {"name": "calc"}]}
        )
        assert resolve(step, "input.tools[0].name") == "search"
        assert resolve(step, "input.tools[1].name") == "calc"

    def test_missing_sentinel_is_falsy(self):
        assert not MISSING
        assert repr(MISSING) == "MISSING"

    def test_missing_is_singleton(self):
        from vijil_dome.controls.selectors import _Missing

        assert _Missing() is _Missing()
