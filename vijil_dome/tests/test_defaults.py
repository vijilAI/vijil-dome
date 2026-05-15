"""BC-12: Centralized model/hub constants in defaults.py."""

from __future__ import annotations

import pytest


class TestCentralizedConstants:
    """Model/hub defaults should be centralized and overridable."""

    def test_defaults_module_has_constants(self) -> None:
        from vijil_dome.defaults import (
            DEFAULT_LLM_HUB,
            DEFAULT_LLM_MODEL,
            DEFAULT_SAFEGUARD_MODEL,
        )

        assert DEFAULT_LLM_MODEL == "gpt-4-turbo"
        assert DEFAULT_LLM_HUB == "openai"
        assert "safeguard" in DEFAULT_SAFEGUARD_MODEL

    def test_llm_detectors_use_default_constant(self) -> None:
        """LLM detector __init__ defaults should reference centralized constants."""
        import inspect
        from vijil_dome.defaults import DEFAULT_LLM_MODEL
        from vijil_dome.detectors.methods.llm_models import LlmSecurity

        sig = inspect.signature(LlmSecurity.__init__)
        default = sig.parameters["model_name"].default
        assert default == DEFAULT_LLM_MODEL

    def test_env_var_overrides_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """VIJIL_LLM_MODEL env var should override the default."""
        monkeypatch.setenv("VIJIL_LLM_MODEL", "claude-3-opus")
        import importlib
        import vijil_dome.defaults
        importlib.reload(vijil_dome.defaults)
        assert vijil_dome.defaults.DEFAULT_LLM_MODEL == "claude-3-opus"
        monkeypatch.delenv("VIJIL_LLM_MODEL")
        importlib.reload(vijil_dome.defaults)
