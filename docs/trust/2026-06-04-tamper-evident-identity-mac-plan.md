# Plan: tamper-evident in-process identity-MAC

- **Status:** Plan (2026-06-04). Companion to `2026-06-04-in-process-control-bypass-threat-model.md`.
- **Goal:** make Vijil's in-process Dome controls enforce **mandatory access control keyed on
  the agent's attested SPIFFE identity**, AND make evasion of that control **loud** — so the
  only way out is to go dark, and going dark alarms.
- **Why now:** this is the differentiated "trustworthy AI" primitive, it is almost entirely
  **offline-buildable**, and it sidesteps every cluster-gated blocker the out-of-process proxy
  MAC carries (the X.509↔JWT seam, admin-socket exposure, single-cluster convergence).

## Resource model

The only two actors are the agent (me, fanning out sub-agents) and the founder. The only
external blockers are **(a) human PR approval on protected repos** and **(b) decisions only the
founder makes**. Repo gating (verified `gh api`): `vijil-dome` is protected **but
`enforce_admins=false`** → founder-admin-mergeable without waiting on CODEOWNERS; `vijil-console`
needs a review; `vijil-proxy-servers`/`vijil-sdk` unprotected. **The bulk of this work is in
`vijil-dome` — the repo the founder can self-unblock.**

## Decided forks (founder, 2026-06-04)

1. **MAC policy key = the attested SPIFFE ID** (`spiffe://vijil.ai/org/<team>/agent/<uuid>`),
   not the Console agent UUID.
2. **Policy source = operator-authored** MAC rulesets for now; the ODRL-policy→MAC compiler is
   a fast-follow (it is lossy and shouldn't gate the demo).

## The build — two halves, one spine

### Half A — Identity-MAC (the decision is keyed on who the agent is)

| ID | Unit | Repo | Size | Depends |
|---|---|---|---|---|
| **A1** | **Keystone:** thread `spiffe_id`+attested into `ToolPolicy.check` / `check_tool_call` / `wrap_tool`; set `ToolCallResult.identity_verified` from real state (it is hardcoded `False` today) | dome | M | — |
| A2 | Fail-closed-on-unattested in enforce mode (today an unattested process still gets policy) | dome | S | A1 |
| A3 | Bind policy to the SVID: assert the policy subject matches `self._identity.spiffe_id`; fail-closed on mismatch (closes the "load any agent's constraints with an API token" priv-esc) | dome | S | A1 |
| A4 | Model-MAC: `allowed_models` / `max_tokens` / `allow_tool_use` + a `before_model` authorization hook, mirroring the proxy `checkMAC` field model | dome | M | A1 |
| A5 | SVID-glob policy resolution (first-match), mirroring the proxy `MatchRoute` | dome | M | A1 |
| A6 | Emit the agent SPIFFE ID in the audit stream (the reported-but-absent `agent_identity`) | dome | S | A1 |
| A7 | Declare the `spiffe` package (the py-spiffe project; imported and installed as `spiffe`) as an `identity` extra + lockfile (X.509 attestation is dead in a locked install today) | dome | S | — |
| A8 | Console: materialize real per-identity MAC rulesets (replace the empty stubs at `constraints/service.py:117-123`); operator-authored domain | console | L | — |
| A9 | Console: `get_by_spiffe_id` resolver + a workload-auth `/constraints` endpoint (present an SVID, not a human JWT) | console | M | A8 |
| A10 | Dome: fetch constraints **by SVID** instead of the developer-supplied `agent_id` string | dome | M | A1, A9 |
| A11 | Identity-keyed e2e: same tool/model, two SVIDs → permit vs deny; mismatch → deny; unattested → deny | dome | M | A1, A3 |

### Half B — Tamper-evidence (you can't silently remove or starve the control)

| ID | Unit | Repo | Size | Depends |
|---|---|---|---|---|
| **B1** | **Fail-CLOSED on detector-unreachable** in enforce mode — bring the `ControlEngine.on_error` semantics to the legacy content-guard path (`secure_agent` runs the legacy path, which hardcodes `hit=False`) | dome | M | — |
| B2 | Loud-fail: `detector_unavailable_total` metric that fires **without** a raised exception, + carry `errored_methods`/`guards_disabled` into the audit event | dome | S | B1 |
| B3 | Enforcement-alive heartbeat: a periodic beacon (SVID-signed now; TEE-quote later) carrying effective mode + hooks-attached + detector-reachable | dome | M | A6 |
| B4 | Console liveness reconciler + **missing-agent alert**: `last_dome_report_at`, staleness sweep, `ATTESTED→INCONCLUSIVE`, dashboard wiring, rogue-agent reconciliation (traffic with no registry row) | console | L | B3 |
| B5 | Reject local `warn` overriding a Console-mandated `enforce`; record the downgrade as a governance event | dome | S | A1 |
| B6 | Single-source-of-truth policy-schema doc (proxy field model ↔ in-process) | dome | S | — |

## Execution waves

- **Wave 1** (parallel, no cross-deps): **A1 keystone** ‖ A7 ‖ B1 ‖ B6 ‖ A8 (Console greenfield).
  A1 is a signature change on the MAC hot path consumed by 3 adapters — front-load the
  consumer-graph grep, land it as one consolidated PR, review hard.
- **Wave 2** (after A1): A2 ‖ A3 ‖ A4 ‖ A5 ‖ A6 ‖ B2 ‖ B5 — fan out.
- **Wave 3**: A9 ‖ B3 → A10 ‖ B4 ‖ A11 (the e2e proof).

Each unit = a TDD'd PR through the `vijil-code` loop + adversarial review + Copilot-wave
convergence. I drive the build; the founder clears merges.

## Friday demo cut (offline, dome-only ⇒ founder-admin-mergeable)

A1 + A2 + A3 + A4 + B1 + B2 + A7 + A11 + an operator-authored demo policy. Two beats:

1. **Deny-by-identity:** the same tool/model call, two different attested identities → permit
   vs deny, in-process, no proxy, no cluster.
2. **Tamper-evidence:** starve the detector (block `/v1/detect`) → the call is **BLOCKED** and a
   **loud** governance signal fires — not the silent fail-open it is today.

Beat 2 is the differentiator: a control you cannot quietly delete or starve.

## Needs the founder

- **Forks:** decided (above). ✓
- **Approvals:** dome admin-merges (you) + console PR reviews.
- **One demo decision:** confirm we demo **both** beats (recommend yes — beat 2 is the
  differentiator).
- **Roadmap (no action this sprint):** TPM-rooted measured boot + TEE-hosted agent (Phala ×
  Vijil) as the hardware root of trust — turns items B1–B5 from tamper-*evidence* into
  tamper-*proof*, and the B3 heartbeat into a remote-attestation quote.
