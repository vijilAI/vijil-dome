"""A3 (DOME-163): bind the policy to the attested principal.

When an agent is attested AND its constraints are SVID-keyed (subject is a ``spiffe://`` URI),
the attested SVID MUST equal the policy subject — otherwise the loaded constraints belong to a
different identity (the priv-esc the threat model flags: load any agent's constraints with an
API token) and the call fails closed. Auto-applies only to SVID-keyed constraints; legacy
UUID-keyed constraints skip the check (non-bricking). The production source of SVID-keyed
constraints is Console (A9, next week); the dome enforcement point lands here now.
"""

from __future__ import annotations

from vijil_dome.trust.constraints import AgentConstraints
from vijil_dome.trust.policy import ToolPolicy

_SPIFFE = "spiffe://vijil.ai/org/team-1/agent/agent-1"
_OTHER_SPIFFE = "spiffe://vijil.ai/org/team-1/agent/agent-99"


def _constraints(*, subject: str) -> AgentConstraints:
    return AgentConstraints.model_validate(
        {
            "agent_id": subject,
            "dome_config": {"input_guards": [], "output_guards": [], "guards": {}},
            "tool_permissions": [
                {"name": "book_flight", "identity": "spiffe://vijil.ai/tools/book_flight/v1",
                 "endpoint": "mcp+tls://book_flight.internal:8443"},
            ],
            "organization": {"required_input_guards": [], "required_output_guards": [], "denied_tools": []},
            "enforcement_mode": "enforce",
            "updated_at": "2026-04-03T12:00:00+00:00",
        }
    )


def test_svid_matches_policy_subject_permitted() -> None:
    result = ToolPolicy(_constraints(subject=_SPIFFE)).check(
        "book_flight", {}, spiffe_id=_SPIFFE, attested=True
    )
    assert result.permitted is True


def test_svid_mismatch_policy_subject_denied() -> None:
    # Attested as agent-1 but the loaded constraints are for agent-99 -> fail closed.
    result = ToolPolicy(_constraints(subject=_OTHER_SPIFFE)).check(
        "book_flight", {}, spiffe_id=_SPIFFE, attested=True
    )
    assert result.permitted is False
    assert "match" in (result.error or "").lower()
    assert result.enforced is True


def test_legacy_uuid_keyed_constraints_skip_binding() -> None:
    # Subject is a UUID (not spiffe://) -> legacy path, binding check does not apply (non-bricking).
    result = ToolPolicy(_constraints(subject="agent-uuid-123")).check(
        "book_flight", {}, spiffe_id=_SPIFFE, attested=True
    )
    assert result.permitted is True


def test_unattested_svid_keyed_permitted_under_default_warn_policy() -> None:
    # DOCUMENTED STAGED GAP: an unattested agent against an SVID-keyed policy is governed by A2's
    # unattested_tool_policy. The default "warn" permits it (non-bricking — many agents run
    # unattested today since X.509 attestation isn't reliably wired). To enforce binding for
    # unattested agents, set unattested_tool_policy="deny" (next test). FOLLOW-UP: flip the
    # default to fail-closed once attestation is the norm (paired with the A2 default-flip).
    result = ToolPolicy(_constraints(subject=_SPIFFE)).check(
        "book_flight", {}, spiffe_id=None, attested=False
    )
    assert result.permitted is True


def test_unattested_svid_keyed_denied_when_policy_deny() -> None:
    # The secure config closes the gap: deny-policy denies an unattested agent regardless of
    # subject shape (A2), so an SVID-keyed policy with deny is fully identity-bound.
    constraints = _constraints(subject=_SPIFFE).model_copy(update={"unattested_tool_policy": "deny"})
    result = ToolPolicy(constraints).check("book_flight", {}, spiffe_id=None, attested=False)
    assert result.permitted is False
    assert "attest" in (result.error or "").lower()
