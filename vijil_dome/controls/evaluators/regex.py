"""Regex pattern matching evaluator."""

from __future__ import annotations

import logging
from typing import Any

from vijil_dome.controls.evaluators import register_evaluator
from vijil_dome.controls.evaluators.base import Evaluator, EvaluatorResult

logger = logging.getLogger(__name__)

try:
    import re2 as _re

    _USING_RE2 = True
except ImportError:
    import re as _re  # type: ignore[no-redef]

    _USING_RE2 = False

_FLAG_SHORT = {
    "i": _re.IGNORECASE,
    "m": _re.MULTILINE,
    "s": _re.DOTALL,
}

_FLAG_LONG = {
    "ignorecase": _re.IGNORECASE,
    "multiline": _re.MULTILINE,
    "dotall": _re.DOTALL,
}


def _resolve_flags(flags: str | list[str] | None) -> int:
    """Accept short string (``"i"``), long-name list (``["IGNORECASE"]``), or None."""
    if not flags:
        return 0
    result = 0
    if isinstance(flags, str):
        for ch in flags:
            result |= _FLAG_SHORT.get(ch, 0)
    else:
        for name in flags:
            result |= _FLAG_LONG.get(name.lower(), 0)
    return result


def _compile(pattern: str, flags: str | list[str] | None = None) -> Any:
    return _re.compile(pattern, _resolve_flags(flags))


@register_evaluator("regex")
class RegexEvaluator(Evaluator):
    """Evaluate a value against one or more regex patterns.

    Config keys:
        pattern (str): Single regex pattern.
        patterns (list[str]): Multiple patterns (any match = matched).
        flags (str): Regex flags string, e.g. ``"i"`` for IGNORECASE.
        negate (bool): If ``True``, matched when pattern does NOT match.
    """

    async def evaluate(
        self, value: Any, config: dict[str, Any]
    ) -> EvaluatorResult:
        text = str(value) if value is not None else ""
        flags = config.get("flags")
        negate = config.get("negate", False)

        patterns: list[str] = []
        if "pattern" in config:
            patterns.append(config["pattern"])
        if "patterns" in config:
            patterns.extend(config["patterns"])

        if not patterns:
            return EvaluatorResult(matched=False, message="No pattern provided")

        for pattern in patterns:
            compiled = _compile(pattern, flags)
            if compiled.search(text):
                matched = not negate
                return EvaluatorResult(
                    matched=matched,
                    confidence=1.0,
                    message=f"Pattern matched: {pattern}",
                    metadata={"pattern": pattern, "using_re2": _USING_RE2},
                )

        matched = negate
        return EvaluatorResult(
            matched=matched,
            confidence=1.0,
            message="No pattern matched",
            metadata={"using_re2": _USING_RE2},
        )
