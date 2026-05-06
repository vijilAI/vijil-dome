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


_EVALUATOR_ALIASES: dict[str, str] = {
    "json": "json_schema",
}

_UNSUPPORTED_EVALUATORS: dict[str, str] = {
    "sql": (
        "The 'sql' evaluator from AgentControl is not available in Dome. "
        "Consider using the 'regex' evaluator for pattern-based SQL checks, "
        "or register a custom evaluator with @register_evaluator('sql')."
    ),
    "galileo.luna2": (
        "The 'galileo.luna2' evaluator is a proprietary Galileo integration "
        "from AgentControl and is not available in Dome. "
        "Use 'dome:toxicity-deberta' or 'dome:prompt-harmfulness-fast' for similar coverage."
    ),
    "cisco.ai_defense": (
        "The 'cisco.ai_defense' evaluator is a proprietary Cisco integration "
        "from AgentControl and is not available in Dome. "
        "Use Dome's built-in detectors (dome:prompt-injection-*, dome:toxicity-*) for similar coverage."
    ),
    "budget": (
        "The 'budget' evaluator from AgentControl (token/cost tracking) "
        "is not available in Dome. Register a custom evaluator with "
        "@register_evaluator('budget') to implement budget tracking."
    ),
}


def resolve_evaluator(name: str) -> Evaluator:
    """Resolve an evaluator by name.

    Resolution order:

    1. Aliases (e.g. ``"json"`` → ``"json_schema"``)
    2. Built-in evaluators (regex, list, json_schema, cel)
    3. Custom registered evaluators (``@register_evaluator``)
    4. Dome bridge (``dome:*`` namespace) — wraps any Dome detector
    5. Known unsupported evaluators — clear error with alternatives

    Raises :class:`ValueError` if the evaluator is not found.
    """
    _ensure_builtins_loaded()

    resolved_name = _EVALUATOR_ALIASES.get(name, name)

    if resolved_name in _evaluator_registry:
        return _evaluator_registry[resolved_name]()

    # Dome bridge: names starting with "dome:" e.g. "dome:prompt-injection-deberta-v3-base"
    if resolved_name.startswith("dome:"):
        from vijil_dome.controls.evaluators.dome_bridge import DomeBridgeEvaluator

        return DomeBridgeEvaluator(detector_name=resolved_name[5:])

    if name in _UNSUPPORTED_EVALUATORS:
        raise ValueError(_UNSUPPORTED_EVALUATORS[name])

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
