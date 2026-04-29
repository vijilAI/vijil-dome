"""Base class and result type for control evaluators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class EvaluatorResult(BaseModel):
    """Result of a single evaluator check."""

    matched: bool
    confidence: float = 1.0
    message: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Evaluator(ABC):
    """Base class for all control evaluators.

    Subclasses implement :meth:`evaluate` which receives the value
    extracted by the selector and the evaluator config from the
    control definition.

    Mirrors the ``DetectionMethod`` pattern from
    ``vijil_dome/detectors/__init__.py``.
    """

    @abstractmethod
    async def evaluate(
        self, value: Any, config: dict[str, Any]
    ) -> EvaluatorResult:
        """Evaluate the extracted value against evaluator logic.

        Parameters
        ----------
        value:
            The value extracted by the selector from the Step.
        config:
            Evaluator-specific configuration from the control definition.

        Returns
        -------
        EvaluatorResult indicating whether the check matched.
        """
        ...
