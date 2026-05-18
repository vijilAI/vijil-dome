"""Trust-delta consumer for ``vijil.ai/trust-delta`` annotations on Controls.

The runtime trust vector is a measured value, not a default. ``TrustVector``
intentionally has no defaults — it is seeded from a Diamond evaluation
baseline, then adjusted by trust deltas extracted from triggered Controls
that carry a ``vijil.ai/trust-delta`` annotation. Every reported score
point is traceable to either a probe outcome or a runtime event.

Design notes
------------

- ``TrustDelta`` mirrors ``TrustVector``: one float per trust dimension,
  defaulting to ``0.0``. An annotation only needs to specify the
  dimensions it adjusts. This keeps a single source of truth for the
  set of trust dimensions and lets pydantic enforce the per-field type
  contract instead of a separate validator over a string discriminator.
- ``TrustDelta.from_annotation`` is **fail-closed**: a malformed payload
  raises :class:`TrustDeltaParseError`, and the raise propagates up
  through :func:`extract_trust_deltas` and
  :meth:`TrustRuntime.apply_evaluation` to the caller. A bad annotation
  is a misconfiguration, not a runtime data event — silently dropping
  it would let a typo erase trust signal the operator is counting on.
  In ``enforce`` mode the error reaches the request handler; in
  ``warn`` mode the runtime's outer guard absorbs it (same path as
  every other guard failure).
- An empty annotation or one with all dimensions explicitly set to
  ``0.0`` returns ``None`` (no-op), not an error. Authors writing
  ``{"reliability": 0.0}`` to assert "this control fires but
  contributes no delta" should not get a misconfiguration error.
- ``apply_trust_delta`` returns a new ``TrustVector`` rather than
  mutating in place. The caller decides whether to keep the new value.
- This module is pure: no I/O, no logging, no side effects. Audit
  emission happens in :mod:`vijil_dome.trust.runtime` after applying.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from vijil_dome.controls.models import Control, EvaluationResult

TrustDimension = Literal["reliability", "security", "safety"]

TRUST_DELTA_ANNOTATION = "vijil.ai/trust-delta"
"""Canonical Control.annotations key carrying a TrustDelta payload."""


class TrustDeltaParseError(ValueError):
    """Raised when a ``vijil.ai/trust-delta`` annotation is malformed.

    Carries the offending payload and the originating Control's name
    (when known) so the audit log / error message can identify which
    Control's annotation is bad. A subclass of ``ValueError`` so callers
    that catch ``ValueError`` continue to absorb it.
    """

    def __init__(
        self,
        message: str,
        *,
        raw: Any,
        control_name: str | None = None,
    ) -> None:
        super().__init__(message)
        self.raw = raw
        self.control_name = control_name


class TrustVector(BaseModel):
    """Three-dimensional trust score in [0.0, 1.0] per dimension.

    No defaults. Seeded from a measured baseline (typically a Diamond
    evaluation) and adjusted by runtime ``TrustDelta`` applications.
    """

    reliability: float = Field(ge=0.0, le=1.0)
    security: float = Field(ge=0.0, le=1.0)
    safety: float = Field(ge=0.0, le=1.0)


class TrustDelta(BaseModel):
    """Per-dimension trust adjustment from a triggered Control.

    Carried in ``Control.annotations["vijil.ai/trust-delta"]`` as
    ``{"reliability": <float>, "security": <float>, "safety": <float>}``.
    Each dimension defaults to ``0.0``; only nonzero dimensions need
    appear in the annotation. Negative values indicate a penalty (the
    control fired on a failure mode); positive values indicate evidence
    of correct behavior.
    """

    model_config = ConfigDict(extra="forbid")

    reliability: float = 0.0
    security: float = 0.0
    safety: float = 0.0

    @classmethod
    def from_annotation(cls, raw: Any) -> TrustDelta | None:
        """Parse a raw annotation payload, fail-closed on malformed input.

        Returns ``None`` only for genuinely empty / all-zero payloads
        (no-op annotations). Raises :class:`TrustDeltaParseError` for:

        - non-dict payloads (string, list, scalar, None)
        - bool values (subclass of int — would silently coerce in pydantic)
        - unknown keys (``extra="forbid"`` rejects them)
        - non-numeric values (string, None)

        See module docstring for the fail-closed rationale.
        """
        if not isinstance(raw, dict):
            raise TrustDeltaParseError(
                f"trust-delta annotation must be a dict, got {type(raw).__name__}",
                raw=raw,
            )
        # bool subclasses int in Python; pydantic would silently coerce
        # ``True``/``False`` to ``1.0``/``0.0`` and mask a malformed
        # annotation. Reject explicitly before pydantic sees it.
        for key, value in raw.items():
            if isinstance(value, bool):
                raise TrustDeltaParseError(
                    f"trust-delta annotation key {key!r} has bool value; "
                    f"expected float",
                    raw=raw,
                )
        try:
            delta = cls.model_validate(raw)
        except ValidationError as exc:
            raise TrustDeltaParseError(
                f"trust-delta annotation failed validation: {exc.errors()}",
                raw=raw,
            ) from exc
        if delta.reliability == 0.0 and delta.security == 0.0 and delta.safety == 0.0:
            return None
        return delta


def extract_trust_deltas(
    controls: list[Control],
    result: EvaluationResult,
) -> list[tuple[str, TrustDelta]]:
    """Walk triggered ControlMatches and return ``(control_name, delta)`` pairs.

    Resolves each match's ``control_name`` against the supplied controls,
    reads ``annotations["vijil.ai/trust-delta"]`` from the parent Control,
    and parses. Matches with no parent (renamed/removed controls) or no
    annotation contribute no delta and are silently skipped — those are
    legitimate runtime states, not misconfigurations.

    A malformed annotation, however, raises
    :class:`TrustDeltaParseError` (with ``control_name`` populated). The
    raise propagates to :meth:`TrustRuntime.apply_evaluation`'s caller —
    see module docstring for the fail-closed rationale.
    """
    by_name = {c.name: c for c in controls}
    pairs: list[tuple[str, TrustDelta]] = []
    for match in result.matches:
        if not match.triggered:
            continue
        control = by_name.get(match.control_name)
        if control is None:
            continue
        raw = control.annotations.get(TRUST_DELTA_ANNOTATION)
        if raw is None:
            continue
        try:
            parsed = TrustDelta.from_annotation(raw)
        except TrustDeltaParseError as exc:
            # Attach the originating control name so error consumers can
            # identify which Control's annotation is bad without parsing
            # the message string.
            exc.control_name = match.control_name
            raise
        if parsed is not None:
            pairs.append((match.control_name, parsed))
    return pairs


def apply_trust_delta(
    vector: TrustVector,
    delta: TrustDelta,
) -> TrustVector:
    """Return a new TrustVector with the delta summed and clamped per dimension.

    Each dimension's running total is clamped to ``[0.0, 1.0]`` after
    the sum, so a ``-10.0`` adjustment does not carry negative slack
    into a subsequent positive adjustment on the next call.
    """
    return TrustVector(
        reliability=_clamp(vector.reliability + delta.reliability),
        security=_clamp(vector.security + delta.security),
        safety=_clamp(vector.safety + delta.safety),
    )


def _clamp(value: float) -> float:
    """Clamp to [0.0, 1.0]."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value
