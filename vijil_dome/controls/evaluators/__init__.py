"""Evaluator registry with auto-discovery of built-ins and dome bridge."""

from __future__ import annotations

import logging
from typing import Type

from vijil_dome.controls.evaluators.base import Evaluator, EvaluatorResult

logger = logging.getLogger(__name__)

_evaluator_registry: dict[str, Type[Evaluator]] = {}
_builtins_loaded = False


def register_evaluator(name: str):
    """Register a custom evaluator class.

    Mirrors ``@register_method`` from ``vijil_dome/detectors/__init__.py``.

    Usage::

        @register_evaluator("finra-compliance")
        class FINRAEvaluator(Evaluator):
            async def evaluate(self, value, config):
                ...
    """

    def decorator(cls: Type[Evaluator]) -> Type[Evaluator]:
        if not (isinstance(cls, type) and issubclass(cls, Evaluator)):
            raise TypeError(f"{cls.__name__} must subclass Evaluator")
        _evaluator_registry[name] = cls
        return cls

    return decorator


def resolve_evaluator(name: str) -> Evaluator:
    """Resolve an evaluator by name.

    Resolution order:

    1. Built-in evaluators (regex, list, json_schema, cel)
    2. Custom registered evaluators (``@register_evaluator``)
    3. Dome bridge (``dome:*`` namespace) — wraps any Dome detector

    Raises :class:`ValueError` if the evaluator is not found.
    """
    _ensure_builtins_loaded()

    if name in _evaluator_registry:
        return _evaluator_registry[name]()

    # Dome bridge: names starting with "dome:" e.g. "dome:prompt-injection-deberta-v3-base"
    if name.startswith("dome:"):
        from vijil_dome.controls.evaluators.dome_bridge import DomeBridgeEvaluator

        return DomeBridgeEvaluator(detector_name=name[5:])

    available = sorted(_evaluator_registry.keys())
    raise ValueError(
        f"Unknown evaluator: {name!r}. "
        f"Available: {available}. "
        f"Dome detectors use the 'dome:' prefix (e.g. 'dome:prompt-injection-deberta-v3-base')."
    )


def list_evaluators() -> list[str]:
    """Return names of all registered evaluators."""
    _ensure_builtins_loaded()
    return sorted(_evaluator_registry.keys())


def _ensure_builtins_loaded() -> None:
    global _builtins_loaded
    if _builtins_loaded:
        return
    # Import modules to trigger their @register_evaluator decorators
    import vijil_dome.controls.evaluators.regex  # noqa: F401
    import vijil_dome.controls.evaluators.list_eval  # noqa: F401

    try:
        import vijil_dome.controls.evaluators.json_eval  # noqa: F401
    except Exception:
        pass  # jsonschema may not be installed

    try:
        import vijil_dome.controls.evaluators.expr  # noqa: F401
    except Exception:
        pass  # cel-python may not be installed

    _builtins_loaded = True


__all__ = [
    "Evaluator",
    "EvaluatorResult",
    "register_evaluator",
    "resolve_evaluator",
    "list_evaluators",
]
