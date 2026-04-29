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

_FLAG_MAP = {
    "i": _re.IGNORECASE,
    "m": _re.MULTILINE,
    "s": _re.DOTALL,
}


def _compile(pattern: str, flags_str: str = "") -> Any:
    flags = 0
    for ch in flags_str:
        flags |= _FLAG_MAP.get(ch, 0)
    return _re.compile(pattern, flags)


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
        flags_str = config.get("flags", "")
        negate = config.get("negate", False)

        patterns: list[str] = []
        if "pattern" in config:
            patterns.append(config["pattern"])
        if "patterns" in config:
            patterns.extend(config["patterns"])

        if not patterns:
            return EvaluatorResult(matched=False, message="No pattern provided")

        for pattern in patterns:
            compiled = _compile(pattern, flags_str)
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
