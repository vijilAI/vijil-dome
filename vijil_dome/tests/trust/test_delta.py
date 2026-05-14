"""Tests for TrustVector / TrustDelta and TrustRuntime trust-delta consumer."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from vijil_dome.controls.models import (
    ConditionNode,
    Control,
    ControlAction,
    ControlMatch,
    EvaluationResult,
    EvaluatorRef,
)
from vijil_dome.trust.audit import AuditEvent
from vijil_dome.trust.delta import (
    TRUST_DELTA_ANNOTATION,
    TrustDelta,
    TrustVector,
    apply_trust_delta,
    extract_trust_deltas,
)
from vijil_dome.trust.runtime import TrustRuntime

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_control(
    name: str,
    *,
    trust_delta: dict[str, Any] | None = None,
    other_annotations: dict[str, Any] | None = None,
) -> Control:
    """Build a Control with an optional trust-delta annotation."""
    annotations: dict[str, Any] = {}
    if other_annotations:
        annotations.update(other_annotations)
    if trust_delta is not None:
        annotations[TRUST_DELTA_ANNOTATION] = trust_delta
    return Control(
        name=name,
        condition=ConditionNode(
            selector="input",
            evaluator=EvaluatorRef(name="regex", config={"pattern": ".*"}),
        ),
        action=ControlAction(decision="deny"),
        annotations=annotations,
    )


def _make_result(*matches: ControlMatch) -> EvaluationResult:
    """Build an EvaluationResult with the given matches."""
    return EvaluationResult(
        action="deny" if any(m.triggered for m in matches) else "allow",
        matches=list(matches),
    )


def _make_match(control_name: str, *, triggered: bool = True) -> ControlMatch:
    return ControlMatch(control_name=control_name, triggered=triggered)


def _make_runtime(events: list[AuditEvent]) -> TrustRuntime:
    """Build a minimal TrustRuntime with an in-memory audit sink.

    Construction-time audit events (e.g. ``identity_unattested`` when
    SPIFFE is unavailable in tests) are dropped so trust-delta tests
    can assert on their own emissions without index drift.
    """
    runtime = TrustRuntime(
        agent_id="test-agent",
        mode="warn",
        audit_sink=events.append,
    )
    events.clear()
    return runtime


# ---------------------------------------------------------------------------
# Adversarial — TrustDelta.from_annotation rejects malformed payloads
# ---------------------------------------------------------------------------


def test_from_annotation_rejects_non_dict() -> None:
    """A list, string, scalar, or None returns None — not raises."""
    assert TrustDelta.from_annotation([1, 2, 3]) is None
    assert TrustDelta.from_annotation("0.5") is None
    assert TrustDelta.from_annotation(0.5) is None
    assert TrustDelta.from_annotation(None) is None


def test_from_annotation_rejects_unknown_keys() -> None:
    """Keys outside {reliability, security, safety} return None (extra='forbid')."""
    assert TrustDelta.from_annotation({"speed": 0.1}) is None
    assert TrustDelta.from_annotation({"reliability": 0.1, "throughput": 0.1}) is None


def test_from_annotation_returns_none_on_all_zero() -> None:
    """An all-zero annotation (including empty dict) contributes no audit signal."""
    assert TrustDelta.from_annotation({}) is None
    assert TrustDelta.from_annotation({"reliability": 0.0}) is None
    assert TrustDelta.from_annotation(
        {"reliability": 0.0, "security": 0.0, "safety": 0.0}
    ) is None


def test_from_annotation_rejects_non_numeric_values() -> None:
    """Non-numeric values — including bool, a subclass of int — return None."""
    assert TrustDelta.from_annotation({"security": "high"}) is None
    assert TrustDelta.from_annotation({"security": True}) is None
    assert TrustDelta.from_annotation({"security": False}) is None
    assert TrustDelta.from_annotation({"security": None}) is None


def test_from_annotation_accepts_partial_payload() -> None:
    """A dict with one dimension parses cleanly; others default to 0.0."""
    parsed = TrustDelta.from_annotation({"security": -0.15})
    assert parsed is not None
    assert parsed.reliability == 0.0
    assert parsed.security == -0.15
    assert parsed.safety == 0.0


def test_from_annotation_accepts_full_payload() -> None:
    """All three dimensions explicit."""
    parsed = TrustDelta.from_annotation(
        {"reliability": 0.05, "security": -0.1, "safety": -0.2}
    )
    assert parsed is not None
    assert parsed.reliability == pytest.approx(0.05)
    assert parsed.security == pytest.approx(-0.1)
    assert parsed.safety == pytest.approx(-0.2)


def test_from_annotation_accepts_int_value() -> None:
    """Integer values coerce to float — JSON has no int/float distinction."""
    parsed = TrustDelta.from_annotation({"reliability": -1})
    assert parsed is not None
    assert parsed.reliability == -1.0
    assert isinstance(parsed.reliability, float)


# ---------------------------------------------------------------------------
# Adversarial — TrustVector rejects out-of-range scores
# ---------------------------------------------------------------------------


def test_trust_vector_rejects_negative_scores() -> None:
    with pytest.raises(ValidationError):
        TrustVector(reliability=-0.1, security=0.5, safety=0.5)


def test_trust_vector_rejects_scores_above_one() -> None:
    with pytest.raises(ValidationError):
        TrustVector(reliability=0.5, security=1.1, safety=0.5)


def test_trust_vector_requires_all_dimensions() -> None:
    """No defaults — every dimension must be supplied."""
    with pytest.raises(ValidationError):
        TrustVector(reliability=0.5, security=0.5)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Boundary + clamp behavior on apply_trust_delta
# ---------------------------------------------------------------------------


def test_apply_clamps_below_zero_to_zero() -> None:
    """A -10.0 adjustment produces 0.0, not a negative number."""
    v = TrustVector(reliability=0.3, security=0.3, safety=0.3)
    result = apply_trust_delta(v, TrustDelta(security=-10.0))
    assert result.security == 0.0


def test_apply_clamps_above_one_to_one() -> None:
    """A +10.0 adjustment produces 1.0, not above."""
    v = TrustVector(reliability=0.3, security=0.3, safety=0.3)
    result = apply_trust_delta(v, TrustDelta(reliability=10.0))
    assert result.reliability == 1.0


def test_negative_then_positive_does_not_carry_slack() -> None:
    """Sequential apply: -10 → clamp to 0, then +0.5 → 0.5 (not -9.5)."""
    v = TrustVector(reliability=0.5, security=0.5, safety=0.5)
    after_first = apply_trust_delta(v, TrustDelta(security=-10.0))
    assert after_first.security == 0.0
    after_second = apply_trust_delta(after_first, TrustDelta(security=0.5))
    assert after_second.security == 0.5


# ---------------------------------------------------------------------------
# Perturbation — applying a delta on one dimension doesn't affect others
# ---------------------------------------------------------------------------


def test_perturbing_security_leaves_reliability_and_safety_unchanged() -> None:
    v = TrustVector(reliability=0.7, security=0.7, safety=0.7)
    result = apply_trust_delta(v, TrustDelta(security=-0.3))
    assert result.reliability == 0.7
    assert result.safety == 0.7
    assert result.security == pytest.approx(0.4)


def test_perturbing_reliability_leaves_security_and_safety_unchanged() -> None:
    v = TrustVector(reliability=0.7, security=0.7, safety=0.7)
    result = apply_trust_delta(v, TrustDelta(reliability=-0.2))
    assert result.security == 0.7
    assert result.safety == 0.7
    assert result.reliability == pytest.approx(0.5)


def test_multi_dimension_delta_sums_per_dimension() -> None:
    """A single delta touching multiple dimensions adjusts each independently."""
    v = TrustVector(reliability=0.5, security=0.5, safety=0.5)
    result = apply_trust_delta(
        v, TrustDelta(reliability=0.1, security=-0.2)
    )
    assert result.reliability == pytest.approx(0.6)
    assert result.security == pytest.approx(0.3)
    assert result.safety == 0.5


# ---------------------------------------------------------------------------
# Round-trip — extract then apply matches direct application
# ---------------------------------------------------------------------------


def test_extract_then_apply_matches_direct_application() -> None:
    """Extracting deltas from an EvaluationResult and applying each
    sequentially produces the same vector as direct application.
    """
    controls = [
        _make_control("deny-pii-leak", trust_delta={"security": -0.15}),
        _make_control("deny-jailbreak", trust_delta={"safety": -0.1}),
    ]
    result = _make_result(
        _make_match("deny-pii-leak"),
        _make_match("deny-jailbreak"),
    )
    pairs = extract_trust_deltas(controls, result)

    v = TrustVector(reliability=0.8, security=0.8, safety=0.8)
    extracted = v
    for _, d in pairs:
        extracted = apply_trust_delta(extracted, d)

    directly = v
    for d in [TrustDelta(security=-0.15), TrustDelta(safety=-0.1)]:
        directly = apply_trust_delta(directly, d)

    assert extracted == directly


# ---------------------------------------------------------------------------
# Extraction — skips controls without annotations or with malformed ones
# ---------------------------------------------------------------------------


def test_extract_skips_untriggered_controls() -> None:
    """Untriggered matches contribute no delta."""
    controls = [_make_control("deny-x", trust_delta={"security": -0.1})]
    result = _make_result(_make_match("deny-x", triggered=False))
    assert extract_trust_deltas(controls, result) == []


def test_extract_skips_renamed_or_removed_controls() -> None:
    """A match whose control_name has no parent in ``controls`` is skipped."""
    controls = [_make_control("deny-x", trust_delta={"security": -0.1})]
    result = _make_result(_make_match("ghost-control"))
    assert extract_trust_deltas(controls, result) == []


def test_extract_skips_controls_without_annotation() -> None:
    """A triggered control with no trust-delta annotation contributes nothing."""
    controls = [
        _make_control(
            "deny-x",
            trust_delta=None,
            other_annotations={"vijil.ai/source": "dome-toml"},
        ),
    ]
    result = _make_result(_make_match("deny-x"))
    assert extract_trust_deltas(controls, result) == []


def test_extract_silently_skips_malformed_annotation() -> None:
    """A malformed annotation (unknown key) does not raise."""
    controls = [_make_control("deny-x", trust_delta={"throughput": -0.1})]
    result = _make_result(_make_match("deny-x"))
    assert extract_trust_deltas(controls, result) == []


def test_extract_silently_skips_all_zero_annotation() -> None:
    """An all-zero annotation produces no audit signal (no-op)."""
    controls = [
        _make_control(
            "deny-x",
            trust_delta={"reliability": 0.0, "security": 0.0, "safety": 0.0},
        ),
    ]
    result = _make_result(_make_match("deny-x"))
    assert extract_trust_deltas(controls, result) == []


# ---------------------------------------------------------------------------
# Integration — TrustRuntime apply_evaluation + seed_trust_vector
# ---------------------------------------------------------------------------


def test_apply_evaluation_returns_none_before_seeding() -> None:
    """Until a baseline is seeded, the runtime's vector is None."""
    events: list[AuditEvent] = []
    runtime = _make_runtime(events)

    controls = [_make_control("deny-pii", trust_delta={"security": -0.1})]
    out = runtime.apply_evaluation(
        controls=controls, result=_make_result(_make_match("deny-pii"))
    )
    assert out is None
    assert runtime.trust_vector is None


def test_unseeded_apply_queues_silently_and_seed_replay_audits_once() -> None:
    """Pre-seed apply queues without auditing; seed replay emits one event per delta.

    The "every score point traceable" invariant requires each logical
    delta to produce one — and only one — audit event. Emitting on
    pre-seed apply would double-count when the same delta is re-emitted
    with concrete before/after during seed replay.
    """
    events: list[AuditEvent] = []
    runtime = _make_runtime(events)

    controls = [_make_control("deny-pii", trust_delta={"security": -0.1})]
    runtime.apply_evaluation(
        controls=controls, result=_make_result(_make_match("deny-pii"))
    )

    # Pre-seed: no audit, delta queued.
    assert events == []
    assert runtime._pending_deltas, "delta should be queued for seed-time replay"

    # Seed: replay emits exactly one event with concrete before/after.
    runtime.seed_trust_vector(
        TrustVector(reliability=0.9, security=0.5, safety=0.8)
    )
    assert len(events) == 1
    assert events[0].event_type == "trust_delta"
    assert events[0].attributes["control_name"] == "deny-pii"
    assert events[0].attributes["delta"] == {
        "reliability": 0.0,
        "security": -0.1,
        "safety": 0.0,
    }
    assert events[0].attributes["before"]["security"] == pytest.approx(0.5)
    assert events[0].attributes["after"]["security"] == pytest.approx(0.4)


def test_seed_twice_raises_to_protect_accumulated_vector() -> None:
    """seed_trust_vector raises if already seeded — silent overwrite would erase runtime adjustments."""
    runtime = _make_runtime([])
    runtime.seed_trust_vector(
        TrustVector(reliability=0.9, security=0.5, safety=0.8)
    )
    with pytest.raises(RuntimeError, match="already seeded"):
        runtime.seed_trust_vector(
            TrustVector(reliability=0.1, security=0.1, safety=0.1)
        )
    # Original vector preserved.
    v = runtime.trust_vector
    assert v is not None
    assert v.security == pytest.approx(0.5)


def test_seed_replays_pending_deltas_in_order() -> None:
    """Pending deltas accumulated before seeding are applied at seed time."""
    events: list[AuditEvent] = []
    runtime = _make_runtime(events)

    controls = [
        _make_control("deny-pii", trust_delta={"security": -0.1}),
        _make_control("deny-jb", trust_delta={"safety": -0.05}),
    ]
    runtime.apply_evaluation(
        controls=controls, result=_make_result(_make_match("deny-pii"))
    )
    runtime.apply_evaluation(
        controls=controls, result=_make_result(_make_match("deny-jb"))
    )

    runtime.seed_trust_vector(
        TrustVector(reliability=0.9, security=0.5, safety=0.8)
    )

    v = runtime.trust_vector
    assert v is not None
    assert v.reliability == pytest.approx(0.9)
    assert v.security == pytest.approx(0.4)
    assert v.safety == pytest.approx(0.75)
    # Replay order: deny-pii (security) then deny-jb (safety).
    assert [e.attributes["control_name"] for e in events] == [
        "deny-pii",
        "deny-jb",
    ]


def test_seeded_apply_updates_vector_and_emits_concrete_audit() -> None:
    """After seeding, audit events carry concrete before/after TrustVectors."""
    events: list[AuditEvent] = []
    runtime = _make_runtime(events)

    runtime.seed_trust_vector(
        TrustVector(reliability=0.9, security=0.5, safety=0.8)
    )

    controls = [_make_control("deny-pii", trust_delta={"security": -0.2})]
    out = runtime.apply_evaluation(
        controls=controls, result=_make_result(_make_match("deny-pii"))
    )

    assert out is not None
    assert out.security == pytest.approx(0.3)
    assert len(events) == 1
    assert events[0].event_type == "trust_delta"
    assert events[0].attributes["before"]["security"] == pytest.approx(0.5)
    assert events[0].attributes["after"]["security"] == pytest.approx(0.3)
    # Reliability and safety unchanged.
    assert events[0].attributes["before"]["reliability"] == pytest.approx(0.9)
    assert events[0].attributes["after"]["reliability"] == pytest.approx(0.9)


def test_apply_with_no_triggered_deltas_returns_current_vector() -> None:
    """If no deltas are extracted, the existing vector is returned unchanged."""
    runtime = _make_runtime([])
    runtime.seed_trust_vector(
        TrustVector(reliability=0.7, security=0.7, safety=0.7)
    )

    out = runtime.apply_evaluation(controls=[], result=_make_result())
    assert out is not None
    assert out.reliability == 0.7
    assert out.security == 0.7
    assert out.safety == 0.7
