"""Regex pattern matching evaluator."""

from __future__ import annotations

import re
import logging
from typing import Any

from vijil_dome.controls.evaluators import register_evaluator
from vijil_dome.controls.evaluators.base import Evaluator, EvaluatorResult

logger = logging.getLogger(__name__)

try:
    import re2 as _re_engine

    _USING_RE2 = True
except ImportError:
    _re_engine = re  # type: ignore[assignment]
    _USING_RE2 = False

# Flag constants always come from stdlib re (re2 packages may not expose them)
_FLAG_SHORT = {
    "i": re.IGNORECASE,
    "m": re.MULTILINE,
    "s": re.DOTALL,
}

_FLAG_LONG = {
    "ignorecase": re.IGNORECASE,
    "multiline": re.MULTILINE,
    "dotall": re.DOTALL,
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
    resolved = _resolve_flags(flags)
    if _USING_RE2 and resolved:
        logger.warning(
            "re2 does not support stdlib re flags — falling back to "
            "stdlib re for pattern '%s'. Linear-time ReDoS protection "
            "is not active for this pattern.",
            pattern,
        )
        return re.compile(pattern, resolved)
    return _re_engine.compile(pattern, resolved)


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
