"""Tests for the adapter registry."""

import pytest

from vijil_dome.trust.adapters.base import (
    BaseAdapter,
    _adapter_registry,
    detect_adapter,
    list_adapters,
    register_adapter,
    resolve_adapter,
)


class TestAdapterRegistry:
    def test_builtin_adapters_registered(self):
        names = list_adapters()
        assert "langgraph" in names
        assert "adk" in names
        assert "strands" in names

    def test_resolve_known(self):
        cls = resolve_adapter("langgraph")
        assert issubclass(cls, BaseAdapter)

    def test_resolve_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown adapter"):
            resolve_adapter("nonexistent_adapter_xyz")

    def test_custom_adapter_registration(self):
        @register_adapter("test_custom_adapter_789")
        class MyAdapter(BaseAdapter):
            @classmethod
            def detect(cls, agent):
                return hasattr(agent, "_test_custom_marker")

            @classmethod
            def wrap(cls, agent, **kwargs):
                agent._wrapped = True
                return agent

        assert "test_custom_adapter_789" in list_adapters()

        cls = resolve_adapter("test_custom_adapter_789")
        assert cls is MyAdapter

        # Clean up
        del _adapter_registry["test_custom_adapter_789"]

    def test_register_non_adapter_raises(self):
        with pytest.raises(TypeError, match="must subclass BaseAdapter"):

            @register_adapter("bad")
            class NotAnAdapter:
                pass

    def test_detect_adapter_finds_match(self):
        @register_adapter("test_detect_adapter_456")
        class DetectableAdapter(BaseAdapter):
            @classmethod
            def detect(cls, agent):
                return hasattr(agent, "_detectable")

            @classmethod
            def wrap(cls, agent, **kwargs):
                return agent

        class FakeAgent:
            _detectable = True

        result = detect_adapter(FakeAgent())
        assert result is DetectableAdapter

        # Clean up
        del _adapter_registry["test_detect_adapter_456"]

    def test_detect_adapter_returns_none_for_unknown(self):
        class UnknownAgent:
            pass

        result = detect_adapter(UnknownAgent())
        assert result is None
