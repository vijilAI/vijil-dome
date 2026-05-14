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
- ``TrustDelta.from_annotation`` is defensive: annotations are
  ``dict[str, Any]`` authored by third parties, so a malformed payload
  returns ``None`` rather than raising. Dome is a guardrail layer; a
  Control author's typo in a TOML annotation must not abort the runtime
  that is meant to protect the agent at request time. Strict schema
  validation belongs at authoring time (TOML loader, schema validation),
  not at runtime trigger time.
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
        """Parse defensively from a raw annotation payload.

        Returns ``None`` if the payload is not a dict, carries unknown
        keys, has non-numeric (or boolean) values, or is all-zeros (a
        no-op annotation contributes no audit signal). Never raises —
        a malformed Control annotation must not crash the runtime
        guardrail layer; see module docstring for the design rationale.
        """
        if not isinstance(raw, dict):
            return None
        # bool subclasses int in Python; pydantic would silently coerce
        # ``True``/``False`` to ``1.0``/``0.0`` and mask a malformed
        # annotation. Reject explicitly before pydantic sees it.
        for value in raw.values():
            if isinstance(value, bool):
                return None
        try:
            delta = cls.model_validate(raw)
        except ValidationError:
            return None
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
    and parses defensively. Matches with no parent (renamed/removed
    controls), no annotation, or a malformed annotation contribute no
    delta. Returning the control name alongside the parsed delta keeps
    audit emission aligned with the producing control even when some
    triggered controls have no annotation.
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
        parsed = TrustDelta.from_annotation(raw)
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
