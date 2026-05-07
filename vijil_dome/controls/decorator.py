"""@control() decorator for wrapping functions with policy evaluation."""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from pathlib import Path
from typing import Any, Callable, Literal

from vijil_dome.controls.engine import ControlEngine
from vijil_dome.controls.errors import handle_result
from vijil_dome.controls.models import Control, Step

logger = logging.getLogger(__name__)

_INPUT_PARAM_NAMES = (
    "input", "message", "query", "text", "prompt",
    "content", "user_input", "msg", "request",
)

_SKIP_PARAMS = {"self", "cls"}


def control(
    *,
    engine: ControlEngine | None = None,
    policy: list[Control | dict] | str | Path | None = None,
    step_type: Literal["tool", "llm"] = "llm",
    step_name: str | None = None,
    enforce: bool = True,
    input_mapper: Callable[..., Any] | None = None,
    output_mapper: Callable[[Any], Any] | None = None,
    context_mapper: Callable[..., dict[str, Any] | None] | None = None,
) -> Callable:
    """Decorator that wraps a function with control evaluation.

    The policy is the single config surface — it defines what to detect,
    on what data, and what action to take.  Dome detectors are referenced
    in the policy as ``dome:*`` evaluators.

    Either ``engine`` or ``policy`` should be provided.  If ``policy`` is
    given, a new :class:`ControlEngine` is created from it.

    Parameters
    ----------
    engine:
        A pre-configured :class:`ControlEngine`.
    policy:
        Control definitions.  Can be a list of dicts/Control objects, or
        a path to a YAML/JSON file.
    step_type:
        The step type used when building the :class:`Step`.
    step_name:
        The step name.  Defaults to the wrapped function's name.
    enforce:
        If ``True``, deny/steer raise exceptions.  If ``False``,
        violations are logged but execution continues (shadow mode).
    input_mapper:
        Optional callable that receives the same ``(*args, **kwargs)``
        as the decorated function and returns the value to use as
        ``step.input``.  When ``None``, a built-in heuristic is used.
    output_mapper:
        Optional callable that receives the function's return value
        and returns the value to use as ``step.output``.  When ``None``,
        the return value is used directly.
    context_mapper:
        Optional callable that receives the same ``(*args, **kwargs)``
        as the decorated function and returns a dict to use as
        ``step.context``.  When ``None``, context is empty.
    """

    def decorator(fn: Callable) -> Callable:
        _engine = _resolve_engine(engine, policy)
        _step_name = step_name or fn.__name__
        _sig = inspect.signature(fn)

        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                step = _build_step(
                    fn, args, kwargs,
                    step_type=step_type,
                    step_name=_step_name,
                    input_mapper=input_mapper,
                    context_mapper=context_mapper,
                    sig=_sig,
                )

                pre_result = await _engine.evaluate(step, stage="pre")
                handle_result(pre_result, enforce, "pre")

                output = await fn(*args, **kwargs)

                step.output = output_mapper(output) if output_mapper else output
                post_result = await _engine.evaluate(step, stage="post")
                handle_result(post_result, enforce, "post")

                return output

            return async_wrapper
        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                step = _build_step(
                    fn, args, kwargs,
                    step_type=step_type,
                    step_name=_step_name,
                    input_mapper=input_mapper,
                    context_mapper=context_mapper,
                    sig=_sig,
                )

                pre_result = _engine.evaluate_sync(step, stage="pre")
                handle_result(pre_result, enforce, "pre")

                output = fn(*args, **kwargs)

                step.output = output_mapper(output) if output_mapper else output
                post_result = _engine.evaluate_sync(step, stage="post")
                handle_result(post_result, enforce, "post")

                return output

            return sync_wrapper

    return decorator


# ------------------------------------------------------------------
# Step construction
# ------------------------------------------------------------------


def _build_step(
    fn: Callable,
    args: tuple,
    kwargs: dict,
    *,
    step_type: Literal["tool", "llm"],
    step_name: str,
    input_mapper: Callable[..., Any] | None,
    context_mapper: Callable[..., dict[str, Any] | None] | None,
    sig: inspect.Signature | None = None,
) -> Step:
    if input_mapper is not None:
        step_input = input_mapper(*args, **kwargs)
    else:
        step_input = _extract_input(fn, args, kwargs, step_type, sig=sig)

    context: dict[str, Any] = {}
    if context_mapper is not None:
        ctx = context_mapper(*args, **kwargs)
        if ctx:
            context = ctx

    return Step(
        type=step_type,
        name=step_name,
        input=step_input,
        context=context,
    )


def _extract_input(
    fn: Callable, args: tuple, kwargs: dict, step_type: str,
    *, sig: inspect.Signature | None = None,
) -> Any:
    """Heuristic input extraction from function arguments.

    Strategy depends on step_type:

    **tool** — all bound arguments (minus self/cls) as a dict, since
    tool inputs are structured.

    **llm** — extract a single value:
      1. Well-known parameter names (message, query, prompt, …)
      2. First string-typed argument
      3. Single non-self parameter → use its value directly
      4. Multiple parameters → dict of all (minus self/cls)
      5. Last resort → stringify everything
    """
    bound = _safe_bind(fn, args, kwargs, sig=sig)
    if bound is None:
        return _stringify_fallback(args, kwargs)

    params = {
        k: v for k, v in bound.items()
        if k not in _SKIP_PARAMS
    }

    if step_type == "tool":
        return params if params else _stringify_fallback(args, kwargs)

    # LLM heuristic: try well-known names first
    for name in _INPUT_PARAM_NAMES:
        if name in params and params[name] is not None:
            return params[name]

    # First string argument
    for value in params.values():
        if isinstance(value, str):
            return value

    # Single parameter → use its value directly
    if len(params) == 1:
        return next(iter(params.values()))

    # Multiple parameters → dict
    if params:
        return params

    return _stringify_fallback(args, kwargs)


def _safe_bind(
    fn: Callable, args: tuple, kwargs: dict,
    *, sig: inspect.Signature | None = None,
) -> dict[str, Any] | None:
    """Bind args to function signature, return dict or None on failure."""
    try:
        if sig is None:
            sig = inspect.signature(fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except (TypeError, ValueError):
        return None


def _stringify_fallback(args: tuple, kwargs: dict) -> str:
    """Last resort: stringify args for evaluation."""
    parts = [str(a) for a in args if not _looks_like_self(a)]
    parts.extend(f"{k}={v}" for k, v in kwargs.items())
    return " ".join(parts) if parts else ""


def _looks_like_self(value: Any) -> bool:
    """Best-effort check for self/cls in positional args."""
    return hasattr(value, "__dict__") and not isinstance(value, (str, int, float, bool, list, dict, tuple))


# ------------------------------------------------------------------
# Engine resolution
# ------------------------------------------------------------------


def _resolve_engine(
    engine: ControlEngine | None,
    policy: list[Control | dict] | str | Path | None,
) -> ControlEngine:
    if engine is not None:
        return engine

    eng = ControlEngine()
    if policy is None:
        return eng

    if isinstance(policy, (str, Path)):
        eng.load_controls_from_file(str(policy))
    elif isinstance(policy, list):
        defs = [
            p if isinstance(p, dict) else p.model_dump(by_alias=True)
            for p in policy
        ]
        eng.load_controls(defs)

    return eng


