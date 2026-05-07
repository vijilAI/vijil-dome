"""Control engine — parallel evaluation with cancel-on-deny."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Literal

try:
    import re2 as _re2_engine

    _SCOPE_HAS_RE2 = True
except ImportError:
    _re2_engine = None  # type: ignore[assignment]
    _SCOPE_HAS_RE2 = False

from vijil_dome.controls.models import (
    ConditionNode,
    Control,
    ControlMatch,
    EvaluationResult,
    Step,
)
from vijil_dome.controls.selectors import MISSING, resolve

logger = logging.getLogger(__name__)


class ControlEngine:
    """Evaluates a set of controls against a :class:`Step`.

    Controls are sorted by priority (lower = first). Deny controls use
    cancel-on-deny: as soon as any deny-control triggers, remaining
    evaluations are cancelled.  Steer and observe controls run after
    the deny pass completes.

    The ``on_error`` field on each control's action determines
    failure semantics: ``fail_closed`` treats evaluator errors as a
    match (blocks), ``fail_open`` treats them as a non-match (passes).
    """

    def __init__(self, controls: list[Control] | None = None) -> None:
        self._controls: list[Control] = sorted(
            controls or [],
            key=lambda c: c.priority,
        )
        self._evaluator_cache: dict[str, Any] = {}

    @property
    def controls(self) -> list[Control]:
        return list(self._controls)

    def add_control(self, control: Control) -> None:
        self._controls.append(control)
        self._controls.sort(key=lambda c: c.priority)

    def load_controls(self, definitions: list[dict[str, Any]]) -> None:
        for defn in definitions:
            self.add_control(Control.model_validate(defn))

    def load_controls_from_file(self, path: str | Path) -> None:
        p = Path(path)
        text = p.read_text()

        if p.suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "PyYAML is required to load YAML policy files. "
                    "Install with: pip install 'vijil-dome[controls]'"
                ) from exc
            data = yaml.safe_load(text)
        elif p.suffix == ".json":
            data = json.loads(text)
        elif p.suffix == ".toml":
            self._load_from_toml(p, text)
            return
        else:
            raise ValueError(
                f"Unsupported policy file format: '{p.suffix}'. "
                f"Use .json, .yaml, .yml, or .toml."
            )

        controls_list = data if isinstance(data, list) else data.get("controls", [])
        self.load_controls(controls_list)

    def _load_from_toml(self, path: Path, text: str) -> None:
        """Load controls from TOML.

        Supports two formats:
        1. Native control definitions under ``[controls]``
        2. Legacy Dome guard config under ``[guardrail]``
        """
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ImportError as exc:
                raise ImportError(
                    "Python 3.11+ or tomli is required for TOML policy files."
                ) from exc

        data = tomllib.loads(text)

        if "controls" in data:
            controls_list = data["controls"]
            if isinstance(controls_list, list):
                self.load_controls(controls_list)
                return

        if "guardrail" in data:
            controls = _translate_dome_toml(data)
            self.load_controls(controls)
            return

        raise ValueError(
            f"TOML policy file '{path.name}' must contain either a "
            f"'controls' list or a 'guardrail' section."
        )

    # ------------------------------------------------------------------
    # Public evaluation API
    # ------------------------------------------------------------------

    async def evaluate(
        self, step: Step, stage: Literal["pre", "post"] = "pre"
    ) -> EvaluationResult:
        start = time.monotonic()
        applicable = [
            c
            for c in self._controls
            if c.enabled and self._scope_matches(c, step, stage)
        ]

        if not applicable:
            return EvaluationResult(
                exec_time_ms=_elapsed_ms(start),
            )

        deny_controls = [c for c in applicable if c.action.decision == "deny"]
        other_controls = [c for c in applicable if c.action.decision != "deny"]

        step_data = step.model_dump()
        matches: list[ControlMatch] = []

        # Phase 1: deny controls with cancel-on-deny
        if deny_controls:
            deny_matches = await self._evaluate_with_cancel_on_deny(
                deny_controls, step, step_data
            )
            matches.extend(deny_matches)
            for m in deny_matches:
                if m.triggered and m.action and m.action.decision == "deny":
                    return EvaluationResult(
                        action="deny",
                        confidence=m.confidence,
                        matches=matches,
                        exec_time_ms=_elapsed_ms(start),
                    )

        # Phase 2: steer / observe controls in parallel
        if other_controls:
            other_matches = await asyncio.gather(
                *(self._evaluate_control(c, step, step_data)
                  for c in other_controls)
            )
            matches.extend(other_matches)

        # Determine aggregate result
        steer_match = next(
            (
                m
                for m in matches
                if m.triggered and m.action and m.action.decision == "steer"
            ),
            None,
        )
        if steer_match and steer_match.action:
            return EvaluationResult(
                action="steer",
                confidence=steer_match.confidence,
                matches=matches,
                steering_context=steer_match.action.steering_context,
                exec_time_ms=_elapsed_ms(start),
            )

        return EvaluationResult(
            action="allow",
            matches=matches,
            exec_time_ms=_elapsed_ms(start),
        )

    def evaluate_sync(
        self, step: Step, stage: Literal["pre", "post"] = "pre"
    ) -> EvaluationResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.evaluate(step, stage))
        # Inside an existing event loop — use nest_asyncio
        import nest_asyncio

        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.evaluate(step, stage))

    # ------------------------------------------------------------------
    # Scope matching
    # ------------------------------------------------------------------

    def _scope_matches(
        self, control: Control, step: Step, stage: Literal["pre", "post"]
    ) -> bool:
        scope = control.scope

        if scope.stages is not None and stage not in scope.stages:
            return False
        if scope.step_types is not None and step.type not in scope.step_types:
            return False
        if scope.step_names is not None and step.name not in scope.step_names:
            return False
        if scope.step_name_regex is not None:
            if not _safe_regex_search(scope.step_name_regex, step.name):
                return False

        return True

    # ------------------------------------------------------------------
    # Cancel-on-deny
    # ------------------------------------------------------------------

    async def _evaluate_with_cancel_on_deny(
        self, controls: list[Control], step: Step,
        step_data: dict[str, Any],
    ) -> list[ControlMatch]:
        tasks: dict[asyncio.Task[ControlMatch], Control] = {}
        for c in controls:
            task = asyncio.create_task(self._evaluate_control(c, step, step_data))
            tasks[task] = c

        results: list[ControlMatch] = []
        pending = set(tasks.keys())

        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                if task.cancelled():
                    continue
                exc = task.exception()
                if exc is not None:
                    ctrl = tasks[task]
                    logger.warning(
                        "Control %s evaluation failed: %s", ctrl.name, exc
                    )
                    results.append(
                        ControlMatch(
                            control_name=ctrl.name,
                            triggered=ctrl.action.on_error == "fail_closed",
                            action=ctrl.action
                            if ctrl.action.on_error == "fail_closed"
                            else None,
                            confidence=0.0,
                            error=str(exc),
                        )
                    )
                    if ctrl.action.on_error == "fail_closed":
                        for p in pending:
                            p.cancel()
                        return results
                    continue

                match = task.result()
                results.append(match)
                if match.triggered:
                    for p in pending:
                        p.cancel()
                    return results

        return results

    # ------------------------------------------------------------------
    # Single control evaluation
    # ------------------------------------------------------------------

    async def _evaluate_control(
        self, control: Control, step: Step,
        step_data: dict[str, Any] | None = None,
    ) -> ControlMatch:
        start = time.monotonic()
        if step_data is None:
            step_data = step.model_dump()
        try:
            triggered = await self._evaluate_condition(
                control.condition, step, step_data
            )
            return ControlMatch(
                control_name=control.name,
                triggered=triggered,
                action=control.action if triggered else None,
                confidence=1.0,
                message=control.action.message or "" if triggered else "",
                exec_time_ms=_elapsed_ms(start),
            )
        except Exception as exc:
            fail_closed = control.action.on_error == "fail_closed"
            logger.warning(
                "Control %s condition error (%s): %s",
                control.name,
                "fail_closed" if fail_closed else "fail_open",
                exc,
            )
            return ControlMatch(
                control_name=control.name,
                triggered=fail_closed,
                action=control.action if fail_closed else None,
                confidence=0.0,
                error=str(exc),
                exec_time_ms=_elapsed_ms(start),
            )

    # ------------------------------------------------------------------
    # Condition tree evaluation
    # ------------------------------------------------------------------

    async def _evaluate_condition(
        self, node: ConditionNode, step: Step,
        step_data: dict[str, Any],
    ) -> bool:
        if node.is_leaf():
            return await self._evaluate_leaf(node, step, step_data)

        if node.and_ is not None:
            results = await asyncio.gather(
                *(self._evaluate_condition(child, step, step_data)
                  for child in node.and_)
            )
            return all(results)

        if node.or_ is not None:
            results = await asyncio.gather(
                *(self._evaluate_condition(child, step, step_data)
                  for child in node.or_)
            )
            return any(results)

        if node.not_ is not None:
            return not await self._evaluate_condition(node.not_, step, step_data)

        raise ValueError("ConditionNode is neither leaf nor composite")

    async def _evaluate_leaf(
        self, node: ConditionNode, step: Step,
        step_data: dict[str, Any],
    ) -> bool:
        if node.selector is None or node.evaluator is None:
            raise ValueError(
                "Leaf condition must have both selector and evaluator"
            )

        value = resolve(step, node.selector, _step_data=step_data)
        if value is MISSING:
            return False

        evaluator = self._get_evaluator(node.evaluator.name)
        result = await evaluator.evaluate(value, node.evaluator.config)
        return result.matched

    def _get_evaluator(self, name: str) -> Any:
        if name not in self._evaluator_cache:
            from vijil_dome.controls.evaluators import resolve_evaluator

            self._evaluator_cache[name] = resolve_evaluator(name)
        return self._evaluator_cache[name]


def _safe_regex_search(pattern: str, text: str) -> re.Match[str] | None:
    """Regex search with ReDoS protection.

    Uses re2 (linear-time) when available, falls back to stdlib re with a
    warning.
    """
    if _SCOPE_HAS_RE2:
        return _re2_engine.search(pattern, text)
    logger.warning(
        "re2 not installed — step_name_regex '%s' uses stdlib re, "
        "which is vulnerable to ReDoS on adversarial patterns. "
        "Install google-re2 for linear-time matching.",
        pattern,
    )
    return re.search(pattern, text)


def _elapsed_ms(start: float) -> float:
    return round((time.monotonic() - start) * 1000, 3)


# ------------------------------------------------------------------
# Legacy Dome TOML translation
# ------------------------------------------------------------------

_STAGE_MAP = {
    "input-guards": "pre",
    "output-guards": "post",
}

_CATEGORY_TO_ACTION: dict[str, str] = {
    "security": "deny",
    "moderation": "deny",
    "privacy": "deny",
    "integrity": "steer",
    "generic": "observe",
    "policy": "deny",
}


def _translate_dome_toml(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate a legacy Dome TOML guardrail config into control dicts."""
    guardrail = data.get("guardrail", {})
    controls: list[dict[str, Any]] = []

    for guard_list_key, stage in _STAGE_MAP.items():
        guard_names = guardrail.get(guard_list_key, [])
        for guard_name in guard_names:
            guard_cfg = data.get(guard_name, {})
            category = guard_cfg.get("type", "generic")
            methods = guard_cfg.get("methods", [])

            if not methods:
                continue

            if len(methods) == 1:
                detector_name = methods[0]
                detector_cfg = guard_cfg.get(detector_name, {})
                condition: dict[str, Any] = {
                    "selector": "input" if stage == "pre" else "output",
                    "evaluator": {
                        "name": f"dome:{detector_name}",
                        "config": detector_cfg,
                    },
                }
            else:
                condition = {
                    "or": [
                        {
                            "selector": "input" if stage == "pre" else "output",
                            "evaluator": {
                                "name": f"dome:{m}",
                                "config": guard_cfg.get(m, {}),
                            },
                        }
                        for m in methods
                    ]
                }

            decision = _CATEGORY_TO_ACTION.get(category, "deny")
            controls.append({
                "name": guard_name,
                "scope": {"stages": [stage]},
                "condition": condition,
                "action": {
                    "decision": decision,
                    "message": guardrail.get(
                        f"{stage}-blocked-message",
                        f"Blocked by {guard_name}",
                    ) if decision == "deny" else None,
                },
                "annotations": {
                    "vijil.ai/source": "dome-toml",
                    "vijil.ai/category": category,
                },
            })

    return controls
