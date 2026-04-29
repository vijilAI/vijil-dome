"""Dot-notation path resolution against Step objects."""

from __future__ import annotations

import re
from typing import Any

from vijil_dome.controls.models import Step


class _Missing:
    """Sentinel for unresolved selector paths."""

    _instance: _Missing | None = None

    def __new__(cls) -> _Missing:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "MISSING"


MISSING = _Missing()

_INDEX_RE = re.compile(r"^([^\[]+)\[(\d+)\]$")


def resolve(step: Step, path: str) -> Any:
    """Resolve a dot-notation path against a Step.

    Supports:
      ``"input"``              → ``step.input``
      ``"input.amount"``       → ``step.input["amount"]``
      ``"context.user_role"``  → ``step.context["user_role"]``
      ``"output"``             → ``step.output``
      ``"*"``                  → ``step.model_dump()``
      ``"name"``               → ``step.name``
      ``"type"``               → ``step.type``
      ``"input.items[0].id"``  → array index access

    Returns :data:`MISSING` if the path does not resolve.
    """
    if path == "*":
        return step.model_dump()

    data = step.model_dump()
    segments = path.split(".")

    current: Any = data
    for segment in segments:
        if current is None:
            return MISSING

        idx_match = _INDEX_RE.match(segment)
        if idx_match:
            key, index = idx_match.group(1), int(idx_match.group(2))
            current = _get(current, key)
            if current is MISSING:
                return MISSING
            if not isinstance(current, (list, tuple)):
                return MISSING
            if index >= len(current):
                return MISSING
            current = current[index]
        else:
            current = _get(current, segment)
            if current is MISSING:
                return MISSING

    return current


def _get(obj: Any, key: str) -> Any:
    """Retrieve a key from a dict or attribute from an object."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        return MISSING
    if hasattr(obj, key):
        return getattr(obj, key)
    return MISSING
