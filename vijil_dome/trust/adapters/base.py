"""Base adapter class and registry for framework adapters.

This follows the same registration pattern as
``@register_method`` (detectors) and ``@register_evaluator`` (controls).

Usage::

    from vijil_dome.trust.adapters.base import BaseAdapter, register_adapter

    @register_adapter("crewai")
    class CrewAIAdapter(BaseAdapter):
        @classmethod
        def detect(cls, agent):
            return type(agent).__module__.startswith("crewai")

        @classmethod
        def wrap(cls, agent, **kwargs):
            ...
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Type

logger = logging.getLogger(__name__)

_adapter_registry: dict[str, Type[BaseAdapter]] = {}


class BaseAdapter(ABC):
    """Abstract base for framework-specific trust adapters."""

    @classmethod
    @abstractmethod
    def detect(cls, agent: Any) -> bool:
        """Return ``True`` if this adapter handles the given agent type."""
        ...

    @classmethod
    @abstractmethod
    def wrap(
        cls,
        agent: Any,
        *,
        client: Any | None = None,
        agent_id: str,
        constraints: Any = None,
        manifest: Any = None,
        mode: str = "warn",
        policy: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Wrap the agent with trust enforcement and return it."""
        ...


def register_adapter(name: str):
    """Decorator to register a framework adapter.

    Usage::

        @register_adapter("langgraph")
        class LangGraphAdapter(BaseAdapter): ...
    """

    def decorator(cls: Type[BaseAdapter]) -> Type[BaseAdapter]:
        if not (isinstance(cls, type) and issubclass(cls, BaseAdapter)):
            raise TypeError(f"{cls.__name__} must subclass BaseAdapter")
        _adapter_registry[name] = cls
        return cls

    return decorator


def resolve_adapter(name: str) -> Type[BaseAdapter]:
    """Look up a registered adapter by name."""
    _ensure_adapters_loaded()
    if name not in _adapter_registry:
        available = sorted(_adapter_registry.keys())
        raise ValueError(
            f"Unknown adapter: {name!r}. Available: {available}"
        )
    return _adapter_registry[name]


def list_adapters() -> list[str]:
    """Return names of all registered adapters."""
    _ensure_adapters_loaded()
    return sorted(_adapter_registry.keys())


def detect_adapter(agent: Any) -> Type[BaseAdapter] | None:
    """Try each registered adapter's ``detect()`` to find a match."""
    _ensure_adapters_loaded()
    for adapter_cls in _adapter_registry.values():
        try:
            if adapter_cls.detect(agent):
                return adapter_cls
        except Exception:
            continue
    return None


_adapters_loaded = False


def _ensure_adapters_loaded() -> None:
    global _adapters_loaded
    if _adapters_loaded:
        return

    try:
        import vijil_dome.trust.adapters.langgraph  # noqa: F401
    except Exception:
        pass
    try:
        import vijil_dome.trust.adapters.adk  # noqa: F401
    except Exception:
        pass
    try:
        import vijil_dome.trust.adapters.strands  # noqa: F401
    except Exception:
        pass

    _adapters_loaded = True
